from __future__ import annotations

from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

from src.spatial_features_v1 import SpatialFeatureTransformer
from src.spatial_pipeline_v1 import (ORDINAL, QDA, VOTING, build_spatial_pipeline,
    fit_spatial_pipeline, validate_spatial_pipeline, verify_fitted_spatial_identity)


def recipe(family=ORDINAL, **changes):
    params = {"n_estimators": 8, "min_samples_leaf": 2} if family == ORDINAL else {"reg_param": .5}
    value = {"family": family, "model_params": params, "training": {"seed": 53}}
    value.update(changes)
    return value


def fixture():
    # Each class exceeds the 32-dimensional QDA representation. Unequal class
    # counts distinguish uniform priors from estimated empirical frequencies.
    y = np.repeat(np.arange(7), np.arange(35, 42))
    rng = np.random.default_rng(913)
    X = rng.normal(size=(len(y), 596)).astype(np.float32)
    X[:, :7] += np.eye(7, dtype=np.float32)[y] * 1.5
    # Deliberately singular selected mean features exercise positive covariance
    # regularization rather than accidentally testing only a full-rank fixture.
    X[:, 298:314] = X[:, :16] * .2
    groups = np.where(np.arange(len(y)) % 2, "person-a", "person-b")
    return X, y, groups


class SpatialPipelineTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.groups = fixture()

    def fit(self, spec=None, **kwargs):
        spec = recipe() if spec is None else spec
        return fit_spatial_pipeline(build_spatial_pipeline(spec), spec,
            self.X, self.y, groups=self.groups, **kwargs)

    def test_ordinal_mapping_and_train_only_float32_scaler_match_explicit_pipeline(self):
        before = self.X.copy()
        for mode, width in (("all_plus_mean_order", 716), ("all_plus_mean_median_order", 836)):
            with self.subTest(mode=mode):
                spec = recipe(feature_engineering={"mode": mode})
                original_spec = deepcopy(spec)
                model = self.fit(spec, validation_data=(np.full((7, 596), np.nan), np.arange(7)))
                engineered = SpatialFeatureTransformer(mode).fit_transform(self.X)
                control_scaler = StandardScaler()
                transformed = control_scaler.fit_transform(engineered)
                control = ExtraTreesClassifier(**model.spec_["model_params"], class_weight="balanced",
                    random_state=53, n_jobs=1).fit(transformed, self.y)
                self.assertEqual(model.n_transformed_features_, width)
                self.assertEqual(transformed.dtype, np.float32)
                np.testing.assert_array_equal(model.scaler_.mean_, engineered.mean(0, dtype=np.float64))
                np.testing.assert_array_equal(model.scaler_.transform(model.features_.transform(self.X)), transformed)
                np.testing.assert_array_equal(model.predict_proba(self.X), control.predict_proba(transformed))
                self.assertEqual(model.training_details_["model_input_dtype"], "float32")
                self.assertFalse(model.training_details_["external_validation_used"])
                self.assertTrue(verify_fitted_spatial_identity(model, spec, 53))
                self.assertEqual(spec, original_spec)
                json.dumps(model.training_details_, allow_nan=False)
        np.testing.assert_array_equal(self.X, before)

    def test_qda_uses_uniform_priors_and_declared_shrinkage_on_training_covariance(self):
        for strength in (.5, .9):
            with self.subTest(strength=strength):
                spec = recipe(QDA, model_params={"reg_param": strength})
                model = self.fit(spec)
                self.assertEqual(model.n_transformed_features_, 32)
                np.testing.assert_array_equal(model.estimator_.priors_, np.repeat(1 / 7, 7))
                self.assertFalse(np.allclose(model.estimator_.priors_, np.bincount(self.y) / len(self.y)))
                self.assertEqual(model.estimator_.solver, "svd")
                self.assertTrue(model.estimator_.store_covariance)
                self.assertIsNone(model.estimator_.shrinkage)
                transformed = model.scaler_.transform(model.features_.transform(self.X))
                for class_id in range(7):
                    values = transformed[self.y == class_id]
                    centered = values - values.mean(0)
                    empirical = centered.T @ centered / (len(values) - 1)
                    expected = (1 - strength) * empirical + strength * np.eye(32)
                    np.testing.assert_allclose(model.estimator_.covariance_[class_id], expected, atol=2e-6, rtol=2e-6)
                    self.assertGreaterEqual(model.estimator_.scalings_[class_id].min(), strength - 1e-6)
                np.testing.assert_allclose(model.predict_proba(self.X).sum(1), 1., atol=1e-12, rtol=0)
                self.assertEqual(model.training_details_["weighting_semantics"],
                    "uniform class priors in QDA posterior; class covariance/mean fitted to each class's unweighted training rows")
                self.assertTrue(verify_fitted_spatial_identity(model, spec, 53))

    def test_same_seed_refit_reload_and_batch_alignment(self):
        for family in (ORDINAL, QDA):
            with self.subTest(family=family):
                spec = recipe(family)
                first = self.fit(spec)
                second = self.fit(spec, validation_data=(self.X * 1e7, (self.y + 1) % 7))
                p = first.predict_proba(self.X)
                np.testing.assert_array_equal(p, second.predict_proba(self.X))
                batches = np.concatenate([first.predict_proba(self.X[:9]), first.predict_proba(self.X[9:])])
                np.testing.assert_allclose(p, batches, atol=2e-6, rtol=1e-6)
                np.testing.assert_array_equal(p.argmax(1), batches.argmax(1))
                buffer = io.BytesIO(); joblib.dump(first, buffer); buffer.seek(0)
                restored = joblib.load(buffer)
                np.testing.assert_array_equal(p, restored.predict_proba(self.X))
                np.testing.assert_array_equal(first.predict(self.X), p.argmax(1))
                self.assertTrue(verify_fitted_spatial_identity(restored, spec, 53))
                self.assertEqual(clone(first).spec, validate_spatial_pipeline(spec))

    def test_qda_does_not_replace_declared_solver_when_training_is_insufficient(self):
        # The installed SVD QDA requires enough rows to span its selected axes.
        # Failure must remain explicit; no data-dependent alternate model exists.
        indices = np.concatenate([np.flatnonzero(self.y == label)[:4] for label in range(7)])
        with self.assertRaisesRegex(ValueError, "at least 32 training rows"):
            build_spatial_pipeline(recipe(QDA)).fit(self.X[indices], self.y[indices])

    def test_voting_preserves_members_shared_seed_and_equal_aligned_probabilities(self):
        first, second = recipe(), recipe(QDA)
        spec = {"family": VOTING, "members": [first, second], "training": {"seed": 53}}
        model = build_spatial_pipeline(spec, seed=71).fit(self.X, self.y, groups=self.groups)
        self.assertEqual([member.effective_seed_ for member in model.members_], [71, 71])
        self.assertEqual(model.members_[0].estimator_.random_state, 71)
        p = np.mean([member.predict_proba(self.X) for member in model.members_], axis=0)
        np.testing.assert_array_equal(model.predict_proba(self.X), p)
        self.assertTrue(verify_fitted_spatial_identity(model, spec, 71))
        stream = io.BytesIO(); joblib.dump(model, stream); stream.seek(0)
        np.testing.assert_array_equal(joblib.load(stream).predict_proba(self.X), p)
        for members in ([first], [first, first], [first, spec],
                [first, {**second, "training": {"seed": 54}}],
                [first, {**second, "training": {"seed": 53, "data_policy": "include_ordinary_reserved_training_records_v1"}}]):
            with self.subTest(members=len(members)), self.assertRaises(ValueError):
                validate_spatial_pipeline({**spec, "members": members})

    def test_strict_recipes_and_training_inputs_fail_before_backend_fit(self):
        cases = [recipe(family="extra_trees"), recipe(window_seconds=True),
            recipe(preprocessing={"scaler": "none"}), recipe(feature_engineering={"mode": "all"}),
            recipe(training={"weighting": "subject_class_balanced"}), recipe(model_params={"random_state": 1}),
            recipe(model_params={"n_estimators": True}), recipe(model_params={"max_features": float("nan")}),
            recipe(model_params={"max_depth": 0}), recipe(postprocessing={"kind": "decision_softmax"}),
            recipe(QDA, model_params={"priors": None}), recipe(QDA, model_params={"priors": [.1] * 7}),
            recipe(QDA, model_params={"solver": "eigen"}), recipe(QDA, model_params={"store_covariance": False}),
            recipe(QDA, model_params={"reg_param": -1}), recipe(QDA, model_params={"reg_param": True}),
            recipe(QDA, feature_engineering={"mode": "all_plus_mean_order"})]
        for spec in cases:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                validate_spatial_pipeline(spec)
        with patch.object(ExtraTreesClassifier, "fit") as backend_fit:
            for values, labels, groups in ((self.X[:, :-1], self.y, self.groups),
                    (self.X[:0], self.y[:0], self.groups[:0]), (np.full_like(self.X, np.nan), self.y, self.groups),
                    (self.X, self.y.astype(float), self.groups), (self.X, np.zeros_like(self.y), self.groups),
                    (self.X, self.y, self.groups[:-1]), (self.X, self.y, np.repeat("", len(self.y)))):
                with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                    build_spatial_pipeline(recipe()).fit(values, labels, groups=groups)
        backend_fit.assert_not_called()
        with self.assertRaises(ValueError):
            fit_spatial_pipeline(build_spatial_pipeline(recipe()), recipe(QDA), self.X, self.y)

    def test_identity_rejects_feature_scaler_backend_and_covariance_tampering(self):
        for family in (ORDINAL, QDA):
            spec = recipe(family)
            original = self.fit(spec)
            mutations = [lambda m: setattr(m, "effective_seed_", 999),
                lambda m: m.scaler_.mean_.__setitem__(0, m.scaler_.mean_[0] + 1),
                lambda m: m.features_.selected_indices_.__setitem__(0, 1),
                lambda m: m.features_.feature_names_out_.__setitem__(0, "changed"),
                lambda m: setattr(m.features_, "n_features_out_", 1),
                lambda m: m.estimator_.classes_.__setitem__(0, 99)]
            if family == ORDINAL:
                mutations.extend([lambda m: m.estimator_.set_params(random_state=999),
                    lambda m: m.estimator_.set_params(class_weight=None)])
            else:
                mutations.extend([lambda m: m.estimator_.set_params(reg_param=.9),
                    lambda m: m.estimator_.covariance_[0].__setitem__((0, 0), 999.),
                    lambda m: m.estimator_.priors_.__setitem__(0, .2),
                    lambda m: m.estimator_.rotations_[0].__setitem__((0, 0), 999.)])
            for mutation in mutations:
                with self.subTest(family=family, mutation=mutation):
                    changed = deepcopy(original)
                    mutation(changed)
                    with self.assertRaises(ValueError):
                        verify_fitted_spatial_identity(changed, spec, 53)
        changed = self.fit()
        changed.spec["model_params"]["max_depth"] = 9
        with self.assertRaises(ValueError):
            changed.predict_proba(self.X)


if __name__ == "__main__":
    unittest.main()
