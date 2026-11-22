from __future__ import annotations

from collections import Counter
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.record_pipeline_v1 import (BALANCED, BAGGED, VOTING, build_record_pipeline,
    fit_record_pipeline, validate_record_pipeline, verify_fitted_record_identity)


def recipe(family=BALANCED, **changes):
    params = {"n_estimators": 10, "min_samples_leaf": 1}
    if family == BAGGED:
        params["bag_count"] = 2
    value = {"family": family, "model_params": params, "training": {"seed": 53}}
    value.update(changes)
    return value


def fixture():
    labels, groups, record_ids = [], [], []
    for class_id in range(7):
        for record in range(1 + class_id % 3):
            count = 2 + (class_id + record) % 3
            labels.extend([class_id] * count)
            groups.extend([f"person-{record}"] * count)
            record_ids.extend([f"person-{record}/class-{class_id}/record-{record}"] * count)
    y = np.array(labels)
    X = np.random.default_rng(91).normal(size=(len(y), 596)).astype(np.float32)
    X[:, :7] += np.eye(7, dtype=np.float32)[y] * 3
    X[:, 10] = np.arange(len(y))
    return X, y, np.array(groups), np.array(record_ids)


class RecordPipelineTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.groups, self.records = fixture()

    def fit(self, spec=None, **kwargs):
        spec = recipe() if spec is None else spec
        return fit_record_pipeline(build_record_pipeline(spec), spec, self.X, self.y,
            groups=self.groups, record_ids=self.records, **kwargs)

    def test_balancing_equalizes_records_within_classes_without_equalizing_people(self):
        spec = recipe()
        original = deepcopy(spec)
        before = self.X.copy()
        model = self.fit(spec, validation_data=(np.full((3, 596), np.nan), [-1, 7, 9]))
        counts = Counter(self.records)
        for class_id in range(7):
            ids = set(self.records[self.y == class_id])
            np.testing.assert_allclose(model.sample_weight_[self.y == class_id].sum(), len(self.y) / 7)
            for record in ids:
                expected = len(self.y) / (7 * len(ids) * counts[record])
                np.testing.assert_array_equal(model.sample_weight_[self.records == record],
                    np.repeat(expected, counts[record]))
                self.assertAlmostEqual(model.record_weighting_audit_["record_totals"][record],
                    len(self.y) / (7 * len(ids)))
        self.assertGreater(np.ptp([model.sample_weight_[self.groups == person].sum() for person in set(self.groups)]), 1)
        self.assertIsNone(model.estimator_.class_weight)
        np.testing.assert_array_equal(model.scaler_.mean_, self.X.mean(0, dtype=np.float64))
        np.testing.assert_array_equal(self.X, before)
        self.assertEqual(spec, original)
        self.assertFalse(model.training_details_["external_validation_used"])
        self.assertTrue(verify_fitted_record_identity(model, spec, 53))
        json.dumps(model.training_details_, allow_nan=False)

    def test_float32_uniform_record_control_matches_adopted_scaling_and_tree_probabilities(self):
        rng = np.random.default_rng(334)
        X = rng.normal(size=(28, 596)).astype(np.float32)
        y = np.repeat(np.arange(7), 4)
        groups = np.tile(np.repeat(["a", "b"], 2), 7)
        ids = np.array([f"class-{c}/person-{groups[i]}" for i, c in enumerate(y)])
        spec = validate_record_pipeline(recipe())
        model = build_record_pipeline(spec).fit(X, y, groups=groups, record_ids=ids)
        baseline = Pipeline([("scale", StandardScaler()), ("model", ExtraTreesClassifier(
            **spec["model_params"], class_weight="balanced", random_state=53, n_jobs=1))]).fit(X, y)
        np.testing.assert_array_equal(model.sample_weight_, np.ones(len(y)))
        actual, expected = model.scaler_.transform(X), baseline.named_steps["scale"].transform(X)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(model.predict_proba(X), baseline.predict_proba(X))
        self.assertEqual(model.training_details_["training_feature_dtype"], "float32")

    def test_bagging_draws_whole_records_with_all_rows_and_unchanged_global_scaler(self):
        spec = recipe(BAGGED)
        calls = []
        original_fit = ExtraTreesClassifier.fit
        def observe(estimator, X, y, **kwargs):
            calls.append((X.copy(), y.copy(), estimator.random_state, estimator.n_estimators, kwargs))
            return original_fit(estimator, X, y, **kwargs)
        with patch.object(ExtraTreesClassifier, "fit", autospec=True, side_effect=observe):
            model = self.fit(spec)
        scaled = StandardScaler().fit_transform(self.X)
        np.testing.assert_array_equal(model.scaler_.transform(self.X), scaled)
        self.assertEqual(scaled.dtype, np.float32)
        for member, (values, labels, seed, trees, options) in enumerate(calls):
            draw = model.bag_draws_[member]
            expected_ids = []
            random = np.random.default_rng(53 + member)
            for class_id in range(7):
                pool = sorted(set(self.records[self.y == class_id]))
                expected_ids.extend(random.choice(pool, len(pool), replace=True).tolist())
            self.assertEqual(draw["record_ids"], expected_ids)
            expected_indices = np.concatenate([np.flatnonzero(self.records == record) for record in expected_ids])
            np.testing.assert_array_equal(model.bag_sample_indices_[member], expected_indices)
            np.testing.assert_array_equal(values, scaled[expected_indices])
            np.testing.assert_array_equal(labels, self.y[expected_indices])
            self.assertEqual(draw["draw_counts"], dict(sorted(Counter(expected_ids).items())))
            self.assertEqual(seed, 53 + member)
            self.assertEqual(trees, 5)
            self.assertNotIn("sample_weight", options)
            self.assertEqual(model.estimators_[member].class_weight, "balanced")
            self.assertEqual(set(labels), set(range(7)))
        self.assertEqual(sum(len(item.estimators_) for item in model.estimators_), 10)
        self.assertTrue(verify_fitted_record_identity(model, spec, 53))

    def test_same_seed_refit_reload_and_batch_predictions_are_exact(self):
        for family in (BALANCED, BAGGED):
            with self.subTest(family=family):
                spec = recipe(family)
                first = self.fit(spec)
                second = self.fit(spec, validation_data=(self.X * 1e9, (self.y + 1) % 7))
                p = first.predict_proba(self.X)
                np.testing.assert_array_equal(p, second.predict_proba(self.X))
                np.testing.assert_array_equal(p, np.concatenate([first.predict_proba(self.X[:9]), first.predict_proba(self.X[9:])]))
                buffer = io.BytesIO(); joblib.dump(first, buffer); buffer.seek(0)
                restored = joblib.load(buffer)
                np.testing.assert_array_equal(p, restored.predict_proba(self.X))
                np.testing.assert_array_equal(first.predict(self.X), p.argmax(1))
                self.assertTrue(verify_fitted_record_identity(restored, spec, 53))
                self.assertEqual(clone(first).spec, validate_record_pipeline(spec))

    def test_missing_or_cross_subject_cross_class_record_metadata_fails_before_fitting(self):
        cases = [(None, self.records), (self.groups, None), (self.groups[:-1], self.records),
            (self.groups, self.records[:-1]), (self.groups, np.repeat("", len(self.y)))]
        changed_groups = self.groups.astype(object)
        changed_groups[0] = "different-person"
        cases.append((changed_groups, self.records))
        changed_records = self.records.astype(object)
        changed_records[np.flatnonzero(self.y == 1)[0]] = self.records[0]
        cases.append((self.groups, changed_records))
        # Whitespace and Unicode normalization cannot disguise shared records.
        changed_records = self.records.astype(object)
        changed_records[np.flatnonzero(self.y == 1)[0]] = " " + self.records[0] + " "
        cases.append((self.groups, changed_records))
        with patch.object(ExtraTreesClassifier, "fit") as fit:
            for groups, records in cases:
                with self.subTest(groups_missing=groups is None, records_missing=records is None), self.assertRaises(ValueError):
                    build_record_pipeline(recipe()).fit(self.X, self.y, groups=groups, record_ids=records)
        fit.assert_not_called()

    def test_strict_schema_feature_and_label_contracts(self):
        cases = [recipe(family="extra_trees"), recipe(window_seconds=True),
            recipe(preprocessing={"scaler": "quantile_normal"}), recipe(feature_engineering={"mode": "static"}),
            recipe(postprocessing={"kind": "decision_softmax"}), recipe(training={"weighting": "class_balanced"}),
            recipe(model_params={"class_weight": "balanced"}), recipe(model_params={"n_estimators": True}),
            recipe(model_params={"max_features": float("nan")}), recipe(model_params={"max_depth": 0}),
            recipe(BAGGED, model_params={"n_estimators": 11, "bag_count": 2}),
            recipe(BAGGED, model_params={"bag_count": 1}),
            recipe(BAGGED, training={"seed": 2**32 - 1})]
        for spec in cases:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                validate_record_pipeline(spec)
        for values, labels in ((self.X[:, :-1], self.y), (self.X[:0], self.y[:0]),
                (np.full_like(self.X, np.nan), self.y), (self.X.astype(complex), self.y),
                (self.X, self.y.astype(float)), (self.X, np.zeros_like(self.y))):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                build_record_pipeline(recipe()).fit(values, labels, groups=self.groups, record_ids=self.records)
        with self.assertRaises(ValueError):
            fit_record_pipeline(build_record_pipeline(recipe()), recipe(BAGGED), self.X, self.y,
                groups=self.groups, record_ids=self.records)

    def test_voting_preserves_shared_records_and_override_seeds_and_rejects_incompatible_members(self):
        first, second = recipe(), recipe(BAGGED)
        spec = {"family": VOTING, "members": [first, second], "training": {"seed": 53}}
        model = build_record_pipeline(spec, seed=71).fit(self.X, self.y, groups=self.groups, record_ids=self.records)
        self.assertEqual([member.effective_seed_ for member in model.members_], [71, 71])
        self.assertEqual([item.random_state for item in model.members_[1].estimators_], [71, 72])
        self.assertTrue(all(member.record_metadata_ == model.record_metadata_ for member in model.members_))
        p = np.mean([member.predict_proba(self.X) for member in model.members_], axis=0)
        np.testing.assert_array_equal(model.predict_proba(self.X), p)
        self.assertTrue(verify_fitted_record_identity(model, spec, 71))
        stream = io.BytesIO(); joblib.dump(model, stream); stream.seek(0)
        np.testing.assert_array_equal(joblib.load(stream).predict_proba(self.X), p)
        for members in ([first], [first, first], [first, spec],
                [first, {**second, "training": {"seed": 54}}],
                [first, {**second, "training": {"seed": 53, "data_policy": "include_ordinary_reserved_training_records_v1"}}]):
            with self.subTest(members=len(members)), self.assertRaises(ValueError):
                validate_record_pipeline({**spec, "members": members})

    def test_fitted_identity_rejects_parameter_sampling_weight_and_scaler_tampering(self):
        for family in (BALANCED, BAGGED):
            spec = recipe(family)
            original = self.fit(spec)
            mutations = [lambda m: setattr(m, "effective_seed_", 999),
                lambda m: m.spec_["model_params"].update(max_depth=9),
                lambda m: m.record_metadata_[0].update(subject="other"),
                lambda m: m.scaler_.mean_.__setitem__(0, m.scaler_.mean_[0] + 1)]
            if family == BALANCED:
                mutations.extend([lambda m: m.estimator_.set_params(class_weight="balanced"),
                    lambda m: m.sample_weight_.__setitem__(0, 999.)])
            else:
                mutations.extend([lambda m: m.estimators_[0].set_params(random_state=999),
                    lambda m: m.bag_draws_[0]["record_ids"].reverse(),
                    lambda m: m.bag_sample_indices_[0].__setitem__(0, 999)])
            for mutation in mutations:
                with self.subTest(family=family, mutation=mutation):
                    changed = deepcopy(original)
                    mutation(changed)
                    with self.assertRaises(ValueError):
                        verify_fitted_record_identity(changed, spec, 53)
        model = self.fit()
        model.spec["model_params"]["max_depth"] = 9
        with self.assertRaises(ValueError):
            model.predict_proba(self.X)


if __name__ == "__main__":
    unittest.main()
