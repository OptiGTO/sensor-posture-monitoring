from __future__ import annotations

from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from src.class_conditional_mixture_v1 import ClassConditionalMixtureClassifier
from src.local_geometry_pipeline_v1 import (PROTOTYPE, MIXTURE, VOTING, build_local_pipeline,
    fit_local_pipeline, local_requires_record_ids, validate_local_pipeline, verify_fitted_local_identity)
from src.posture_core import feature_names
from src.record_prototype_v1 import RecordPrototypeClassifier


def recipe(family=PROTOTYPE, **changes):
    params = {"n_neighbors": 1} if family == PROTOTYPE else {"n_components": 1}
    value = {"family": family, "model_params": params, "training": {"seed": 53}}
    value.update(changes)
    return value


def fixture():
    labels, groups, records = [], [], []
    for label in range(7):
        for recording in range(3):
            count = 3 + recording + label % 2
            labels.extend([label] * count)
            groups.extend([f"person-{recording}"] * count)
            records.extend([f"person-{recording}/class-{label}/recording"] * count)
    y = np.array(labels)
    X = np.random.default_rng(130013).normal(size=(len(y), 596)).astype(np.float32)
    X[:, :7] += np.eye(7, dtype=np.float32)[y] * 1.2
    return X, y, np.array(groups), np.array(records)


class LocalGeometryPipelineTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.groups, self.records = fixture()

    def fit(self, spec=None, **kwargs):
        spec = recipe() if spec is None else spec
        return fit_local_pipeline(build_local_pipeline(spec), spec, self.X, self.y,
            groups=self.groups, record_ids=self.records if local_requires_record_ids(spec) else None, **kwargs)

    def test_scaler_fits_only_original_train_rows_and_backend_receives_exact_float32_values(self):
        original_X, original_y = self.X.copy(), self.y.copy()
        expected_scaler = StandardScaler().fit(self.X)
        expected_values = expected_scaler.transform(self.X)
        for family, backend in ((PROTOTYPE, RecordPrototypeClassifier), (MIXTURE, ClassConditionalMixtureClassifier)):
            with self.subTest(family=family):
                calls = []
                original_fit = backend.fit
                def observe(estimator, values, labels, **kwargs):
                    calls.append((values.copy(), labels.copy(), deepcopy(kwargs)))
                    return original_fit(estimator, values, labels, **kwargs)
                spec, original_spec = recipe(family), recipe(family)
                with patch.object(backend, "fit", autospec=True, side_effect=observe):
                    model = self.fit(spec, validation_data=(np.full((7, 596), np.nan), np.arange(7)))
                self.assertEqual(len(calls), 1)
                values, labels, options = calls[0]
                np.testing.assert_array_equal(values, expected_values)
                np.testing.assert_array_equal(labels, self.y)
                self.assertEqual(values.dtype, np.float32)
                if family == PROTOTYPE:
                    np.testing.assert_array_equal(options["groups"], self.groups)
                    np.testing.assert_array_equal(options["record_ids"], self.records)
                else:
                    self.assertEqual(options, {})
                np.testing.assert_array_equal(model.scaler_.mean_, expected_scaler.mean_)
                np.testing.assert_array_equal(model.scaler_.scale_, expected_scaler.scale_)
                self.assertEqual(model.scaler_.n_samples_seen_, len(self.y))
                self.assertEqual(model.training_details_["model_input_dtype"], "float32")
                self.assertFalse(model.training_details_["external_validation_used"])
                self.assertTrue(verify_fitted_local_identity(model, spec, 53))
                self.assertEqual(spec, original_spec)
                json.dumps(model.training_details_, allow_nan=False)
        np.testing.assert_array_equal(self.X, original_X)
        np.testing.assert_array_equal(self.y, original_y)

    def test_prototypes_are_training_record_means_after_the_global_scaler(self):
        spec = recipe(PROTOTYPE, model_params={"n_neighbors": 3})
        model = self.fit(spec)
        scaled = model.scaler_.transform(self.X)
        for row, prototype in zip(model.estimator_.record_metadata_, model.estimator_.prototypes_):
            values = scaled[row["row_indices"]].astype(np.float64)
            expected = np.sort(values, axis=0).mean(axis=0)
            np.testing.assert_array_equal(prototype, expected)
            self.assertEqual(set(self.records[row["row_indices"]]), {row["record_id"]})
        self.assertEqual(model.estimator_.n_records_, 21)
        np.testing.assert_array_equal(model.estimator_.class_record_counts_, np.repeat(3, 7))
        self.assertTrue(verify_fitted_local_identity(model, spec, 53))

    def test_mixture_class_means_follow_train_scaling_and_never_require_record_ids(self):
        spec = recipe(MIXTURE)
        model = build_local_pipeline(spec).fit(self.X, self.y)
        scaled = model.scaler_.transform(self.X).astype(np.float64)
        for label, estimator in enumerate(model.estimator_.estimators_):
            np.testing.assert_allclose(estimator.means_[0], scaled[self.y == label].mean(0), atol=1e-14, rtol=1e-13)
        self.assertFalse(local_requires_record_ids(spec))
        self.assertIsNone(model.training_details_["training_subjects"])
        self.assertEqual(model.estimator_.effective_seeds_, list(range(53, 60)))
        self.assertTrue(verify_fitted_local_identity(model, spec, 53))

    def test_refit_reload_batches_and_unseen_queries_do_not_change_train_state(self):
        for spec in (recipe(), recipe(MIXTURE, model_params={"n_components": 2})):
            with self.subTest(family=spec["family"]):
                first = self.fit(spec)
                second = self.fit(spec, validation_data=(self.X * 1e9, (self.y + 1) % 7))
                p = first.predict_proba(self.X)
                np.testing.assert_array_equal(p, second.predict_proba(self.X))
                batches = np.concatenate([first.predict_proba(self.X[:9]), first.predict_proba(self.X[9:])])
                np.testing.assert_allclose(p, batches, atol=1e-12, rtol=1e-12)
                np.testing.assert_array_equal(p.argmax(1), batches.argmax(1))
                stream = io.BytesIO(); joblib.dump(first, stream); stream.seek(0)
                restored = joblib.load(stream)
                np.testing.assert_array_equal(restored.predict_proba(self.X), p)
                np.testing.assert_array_equal(restored.predict(self.X), p.argmax(1))
                self.assertTrue(verify_fitted_local_identity(restored, spec, 53))
                scaler_hash = first.scaler_state_sha256_
                details = deepcopy(first.training_details_)
                first.predict_proba(self.X[:3] + 1e6)
                self.assertEqual(first.scaler_state_sha256_, scaler_hash)
                self.assertEqual(first.training_details_, details)
                self.assertTrue(verify_fitted_local_identity(first, spec, 53))
                self.assertEqual(clone(first).spec, validate_local_pipeline(spec))

    def test_ensembles_align_probabilities_preserve_scope_and_propagate_override_seeds(self):
        prototype, mixture = recipe(), recipe(MIXTURE)
        another = recipe(MIXTURE, model_params={"n_components": 2})
        for members, requires_records in (([prototype, mixture], True), ([mixture, another], False)):
            with self.subTest(requires_records=requires_records):
                spec = {"family": VOTING, "members": members, "training": {"seed": 53}}
                self.assertEqual(local_requires_record_ids(spec), requires_records)
                model = build_local_pipeline(spec, seed=71).fit(self.X, self.y, groups=self.groups,
                    record_ids=self.records if requires_records else None)
                self.assertEqual([member.effective_seed_ for member in model.members_], [71, 71])
                for member in model.members_:
                    np.testing.assert_array_equal(member.scaler_.mean_, self.X.mean(0, dtype=np.float64))
                    if member.spec_["family"] == MIXTURE:
                        self.assertEqual(member.estimator_.effective_seeds_, list(range(71, 78)))
                expected = np.mean([member.predict_proba(self.X) for member in model.members_], axis=0)
                np.testing.assert_array_equal(model.predict_proba(self.X), expected)
                self.assertTrue(verify_fitted_local_identity(model, spec, 71))
                stream = io.BytesIO(); joblib.dump(model, stream); stream.seek(0)
                np.testing.assert_array_equal(joblib.load(stream).predict_proba(self.X), expected)

    def test_strict_recipe_policy_seed_and_member_contracts(self):
        first, second = recipe(), recipe(MIXTURE)
        voting = {"family": VOTING, "members": [first, second], "training": {"seed": 53}}
        invalid = [recipe(window_seconds=True), recipe(feature_engineering={"mode": "static"}),
            recipe(preprocessing={"scaler": "quantile_normal"}), recipe(postprocessing={"kind": "macro_f1_bias"}),
            recipe(training={"weighting": "class_balanced"}), recipe(training={"data_policy": "append_unknown"}),
            recipe(model_params={"n_neighbors": 2}), recipe(model_params={"epsilon": 1e-9}),
            recipe(MIXTURE, model_params={"n_components": 3}), recipe(MIXTURE, model_params={"reg_covar": .2}),
            recipe(MIXTURE, model_params={"covariance_type": "full"}), recipe(MIXTURE, model_params={"random_state": 1}),
            recipe(MIXTURE, model_params={"warm_start": True}), recipe(MIXTURE, model_params={"n_init": 1}),
            {**voting, "members": [first]}, {**voting, "members": [first, first]},
            {**voting, "members": [first, voting]},
            {**voting, "members": [first, {**second, "training": {"seed": 54}}]},
            {**voting, "members": [first, {**second, "training": {"seed": 53,
                "data_policy": "include_ordinary_reserved_training_records_v1"}}]}]
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                validate_local_pipeline(spec)
        with self.assertRaises(ValueError):
            fit_local_pipeline(build_local_pipeline(first), second, self.X, self.y)

    def test_invalid_input_and_missing_or_impure_records_fail_before_backend_fit(self):
        mixed_groups = self.groups.astype(object)
        mixed_groups[0] = "different-person"
        mixed_records = self.records.astype(object)
        mixed_records[np.flatnonzero(self.y == 1)[0]] = self.records[0]
        cases = [(self.X, self.y, None, self.records), (self.X, self.y, self.groups, None),
            (self.X, self.y, mixed_groups, self.records), (self.X, self.y, self.groups, mixed_records),
            (self.X, self.y, self.groups[:-1], self.records), (self.X[:, :-1], self.y, self.groups, self.records),
            (np.full_like(self.X, np.nan), self.y, self.groups, self.records),
            (self.X, np.zeros_like(self.y), self.groups, self.records),
            (self.X, self.y.astype(float), self.groups, self.records)]
        with patch.object(RecordPrototypeClassifier, "fit") as backend_fit:
            for values, labels, groups, records in cases:
                with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                    build_local_pipeline(recipe()).fit(values, labels, groups=groups, record_ids=records)
        backend_fit.assert_not_called()
        frame = pd.DataFrame(self.X, columns=feature_names())
        with self.assertRaises(ValueError):
            build_local_pipeline(recipe(MIXTURE)).fit(frame[frame.columns[::-1]], self.y)
        model = self.fit()
        with self.assertRaises(ValueError):
            model.predict_proba(frame[frame.columns[::-1]])

    def test_identity_rejects_scaler_backend_seed_features_and_nested_audit_tampering(self):
        for spec in (recipe(), recipe(MIXTURE)):
            original = self.fit(spec)
            mutations = [lambda m: setattr(m, "effective_seed_", 999),
                lambda m: m.scaler_.mean_.__setitem__(0, 999.),
                lambda m: m.feature_names_in_.__setitem__(0, "changed"),
                lambda m: m.training_details_["backend_training_details"].update(extra="changed"),
                lambda m: m.training_details_["class_counts"].__setitem__(0, 999)]
            if spec["family"] == PROTOTYPE:
                mutations += [lambda m: m.estimator_.prototypes_.__setitem__((0, 0), 999.),
                    lambda m: m.training_details_.update(training_subjects=["unknown"])]
            else:
                mutations += [lambda m: m.estimator_.estimators_[0].means_.__setitem__((0, 0), 999.),
                    lambda m: m.estimator_.estimators_[0].set_params(random_state=999)]
            for mutation in mutations:
                with self.subTest(family=spec["family"], mutation=mutation):
                    changed = deepcopy(original)
                    mutation(changed)
                    with self.assertRaises(ValueError):
                        verify_fitted_local_identity(changed, spec, 53)
        spec = {"family": VOTING, "members": [recipe(), recipe(MIXTURE)], "training": {"seed": 53}}
        original = self.fit(spec)
        ensemble_mutations = [lambda m: m.training_details_["members"][0]["class_counts"].__setitem__(0, 999),
            lambda m: m.training_details_.update(weights=[.9, .1]),
            lambda m: m.training_details_.update(combination="weighted probabilities")]
        for mutation in ensemble_mutations:
            with self.subTest(ensemble_mutation=mutation):
                changed = deepcopy(original)
                mutation(changed)
                with self.assertRaises(ValueError):
                    verify_fitted_local_identity(changed, spec, 53)


if __name__ == "__main__":
    unittest.main()
