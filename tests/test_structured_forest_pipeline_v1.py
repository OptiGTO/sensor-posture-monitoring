"""Synthetic end-to-end scope and numerical contracts for structured forests."""
from copy import deepcopy
import io
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src import structured_forest_pipeline_v1 as subject
from src.iterative_pipeline import build_pipeline as legacy_pipeline
from src.pairwise_coupling_v1 import PAIR_INDICES, couple_pairwise
from src.posture_core import feature_names
from src.subject_residual_v1 import verify_subject_residual


def recipe(family=subject.RESIDUAL, mode="remove_subject3", seed=53, policy=None):
    raw = {"family": family, "model_params": {"n_estimators": 4, "min_samples_leaf": 2},
        "training": {"seed": seed}}
    if family == subject.RESIDUAL:
        raw["feature_engineering"] = {"mode": mode}
    if policy is not None:
        raw["training"]["data_policy"] = policy
    return subject.validate_structured_pipeline(raw)


class StructuredForestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)
        labels, groups, records = [], [], []
        for person in range(4):
            for label in range(7):
                for window in range(3 + label % 3):
                    labels.append(label)
                    groups.append(f"person-{person}")
                    records.append(f"person-{person}/posture-{label}/record-{window // 2}")
        cls.y, cls.groups, cls.records = np.asarray(labels, dtype=np.int64), np.asarray(groups), np.asarray(records)
        rng = np.random.default_rng(1901)
        cls.X = rng.normal(size=(len(labels), 596)).astype(np.float32)
        cls.X[:, :7] += 2 * np.eye(7, dtype=np.float32)[cls.y]
        cls.X[:, 20] = 0.
        cls.pair, cls.residual = recipe(subject.PAIR), recipe()
        cls.voting = subject.validate_structured_pipeline({"family": subject.VOTING,
            "members": [cls.pair, cls.residual], "training": {"seed": 53}})
        cls.models = [subject.build_structured_pipeline(spec).fit(cls.X, cls.y,
            groups=cls.groups, record_ids=cls.records) for spec in (cls.pair, cls.residual, cls.voting)]

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def fit(self, spec, X=None, y=None, groups=None, records=None, seed=None, validation_data=None):
        return subject.fit_structured_pipeline(subject.build_structured_pipeline(spec, seed=seed), spec,
            self.X if X is None else X, self.y if y is None else y,
            groups=self.groups if groups is None else groups, record_ids=self.records if records is None else records,
            validation_data=validation_data)

    def test_identity200_matches_original_baseline_scaler_and_probabilities_exactly(self):
        spec = subject.validate_structured_pipeline({"family": subject.RESIDUAL,
            "feature_engineering": {"mode": "identity"}, "training": {"seed": 53, "data_policy": subject.POLICY}})
        reference = {"family": "extra_trees", "model_params": spec["model_params"],
            "training": {"seed": 53, "data_policy": subject.POLICY}}
        self.assertEqual(spec["model_params"], {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 8, "max_features": .5})
        for dtype in (np.float32, np.float64):
            X = self.X.astype(dtype)
            X[0, 20] = -0.
            model = self.fit(spec, X=X)
            baseline = legacy_pipeline(reference).fit(X, self.y, groups=self.groups)
            query = np.concatenate((X[:15], X[:3] * 100))
            self.assertEqual(model.predict_proba(query).tobytes(), baseline.predict_proba(query).tobytes())
            self.assertEqual(model.scaler_.mean_.tobytes(), baseline.named_steps["scale"].mean_.tobytes())
            self.assertEqual(model.scaler_.transform(X).tobytes(), baseline.named_steps["scale"].transform(X).tobytes())
            self.assertIsNone(model.projector_.components_)
            self.assertTrue(subject.verify_fitted_structured_identity(model, spec, 53))

    def test_pair_heads_use_exact_pair_rows_binary_labels_seeds_and_one_shared_train_scaler(self):
        scaler_seen, forest_seen = [], []
        scaler_fit, forest_fit = StandardScaler.fit, ExtraTreesClassifier.fit
        def capture_scaler(scaler, X, y=None, sample_weight=None):
            scaler_seen.append((X.copy(), sample_weight))
            return scaler_fit(scaler, X, y=y, sample_weight=sample_weight)
        def capture_forest(forest, X, y, **kwargs):
            forest_seen.append((X.copy(), y.copy(), forest.random_state, forest.class_weight, deepcopy(kwargs)))
            return forest_fit(forest, X, y, **kwargs)
        class UnreadableValidation:
            def __array__(self, *args, **kwargs):
                raise AssertionError("External validation read")
            def __iter__(self):
                raise AssertionError("External validation iterated")
        with patch.object(StandardScaler, "fit", capture_scaler), patch.object(ExtraTreesClassifier, "fit", capture_forest):
            model = self.fit(self.pair, validation_data=UnreadableValidation())
        self.assertEqual(len(scaler_seen), 1)
        np.testing.assert_array_equal(scaler_seen[0][0], self.X)
        self.assertIsNone(scaler_seen[0][1])
        scaled = StandardScaler().fit_transform(self.X)
        self.assertEqual(len(forest_seen), 21)
        for index, (pair, row, audit, head) in enumerate(zip(PAIR_INDICES, forest_seen, model.pair_audits_, model.heads_)):
            take = np.flatnonzero(np.isin(self.y, pair))
            binary = (self.y[take] == pair[1]).astype(np.int64)
            np.testing.assert_array_equal(row[0], scaled[take])
            self.assertEqual(row[0].dtype, np.dtype(np.float32))
            np.testing.assert_array_equal(row[1], binary)
            self.assertEqual(row[2:], (53 + index, "balanced", {}))
            self.assertEqual(audit["row_indices"], take.tolist())
            self.assertEqual(audit["input_sha256"], subject._array_digest(scaled[take]))
            self.assertEqual(audit["binary_labels_sha256"], subject._array_digest(binary))
            self.assertEqual(audit["binary_counts"], np.bincount(binary).tolist())
            np.testing.assert_allclose(np.asarray(audit["binary_class_weights"]) * np.bincount(binary), len(take)/2)
            self.assertEqual(head.n_features_in_, 596)
        self.assertEqual(model.training_details_["pair_training_row_exposures"], 6 * len(self.y))
        self.assertEqual(model.training_details_["total_trees"], 21 * 4)
        self.assertFalse(model.training_details_["external_validation_used"])

    def test_binary_orientation_and_direct_head_probabilities_feed_coupling_exactly(self):
        model = self.models[0]
        query = self.X[:18]
        scaled = model.scaler_.transform(query).astype(np.float32)
        forward = np.column_stack([head.predict_proba(scaled)[:, np.flatnonzero(head.classes_ == 0)[0]] for head in model.heads_])
        self.assertEqual(model.predict_proba(query).tobytes(), couple_pairwise(forward).tobytes())
        self.assertTrue(np.isfinite(model.predict_proba(query)).all())
        np.testing.assert_allclose(model.predict_proba(query).sum(1), 1., atol=1e-12, rtol=0)
        np.testing.assert_array_equal(model.predict(query), model.predict_proba(query).argmax(1))
        self.assertFalse(model.training_details_["calibrated_posterior"])
        changed = deepcopy(model)
        changed.heads_[0].classes_ = changed.heads_[0].classes_[::-1]
        with self.assertRaisesRegex(ValueError, "orientation"):
            changed.predict_proba(query)

    def test_residual_receives_only_train_scaled_features_and_native_forest_uses_residual(self):
        forest_seen, scaler_seen = [], []
        original_forest, original_scaler = ExtraTreesClassifier.fit, StandardScaler.fit
        def forest_fit(forest, X, y, **kwargs):
            forest_seen.append((X.copy(), y.copy(), deepcopy(kwargs)))
            return original_forest(forest, X, y, **kwargs)
        def scaler_fit(scaler, X, y=None, sample_weight=None):
            scaler_seen.append(X.copy())
            return original_scaler(scaler, X, y=y, sample_weight=sample_weight)
        with patch.object(ExtraTreesClassifier, "fit", forest_fit), patch.object(StandardScaler, "fit", scaler_fit):
            model = self.fit(self.residual, validation_data=object())
        self.assertEqual(len(scaler_seen), 1)
        self.assertEqual(len(forest_seen), 1)
        np.testing.assert_array_equal(scaler_seen[0], self.X)
        scaled = model.scaler_.transform(self.X)
        expected = model.projector_.transform(scaled)
        self.assertEqual(forest_seen[0][0].tobytes(), expected.tobytes())
        self.assertEqual(expected.dtype, np.dtype(np.float64))
        self.assertEqual(forest_seen[0][2], {})
        self.assertEqual(model.projector_.training_input_sha256_, subject._array_digest(scaled))
        self.assertEqual(model.projector_.training_labels_sha256_, subject._array_digest(self.y))
        self.assertEqual(model.projector_.training_groups_sha256_, subject._json_digest(self.groups.tolist()))
        self.assertEqual(model.projector_.training_records_sha256_, subject._json_digest(self.records.tolist()))
        self.assertEqual(model.training_details_["transformed_input_sha256"], subject._array_digest(expected))
        query = self.X[:7] * 1.5
        self.assertEqual(model.predict_proba(query).tobytes(), model.estimator_.predict_proba(
            model.projector_.transform(model.scaler_.transform(query))).tobytes())
        self.assertTrue(verify_subject_residual(model.projector_, "remove_subject3"))

    def test_effective_seed_wraps_each_pair_and_propagates_to_ensemble_members(self):
        seed = 2**32 - 9
        model = self.fit(self.voting, seed=seed)
        self.assertEqual(model.effective_seed_, seed)
        self.assertEqual([member.effective_seed_ for member in model.members_], [seed, seed])
        for index, head in enumerate(model.members_[0].heads_):
            self.assertEqual(head.random_state, (seed + index) % 2**32)
        self.assertEqual(model.members_[1].estimator_.random_state, seed)
        self.assertEqual(self.voting["training"]["seed"], 53)
        self.assertTrue(subject.verify_fitted_structured_identity(model, self.voting, seed))

    def test_refit_reload_query_batches_and_permutations_are_exact_without_mutation(self):
        query = np.concatenate((self.X[:8], self.X[:3] * 9))
        before = query.tobytes()
        for spec, model in zip((self.pair, self.residual, self.voting), self.models):
            initial = model.predict_proba(query)
            state, details = model.fit_state_sha256_, deepcopy(model.training_details_)
            memory = io.BytesIO()
            joblib.dump(model, memory)
            memory.seek(0)
            restored = joblib.load(memory)
            refitted = self.fit(spec)
            self.assertEqual(initial.tobytes(), restored.predict_proba(query).tobytes())
            self.assertEqual(initial.tobytes(), refitted.predict_proba(query).tobytes())
            with patch.object(StandardScaler, "fit", side_effect=AssertionError("query scaler fit")), \
                    patch.object(ExtraTreesClassifier, "fit", side_effect=AssertionError("query forest fit")):
                separate = np.concatenate([model.predict_proba(query[i:i+1]) for i in range(len(query))])
                self.assertEqual(initial.tobytes(), separate.tobytes())
                self.assertEqual(initial[::-1].tobytes(), model.predict_proba(query[::-1]).tobytes())
                self.assertEqual(initial.tobytes(), model.predict_proba(np.asfortranarray(query)).tobytes())
            self.assertEqual(query.tobytes(), before)
            self.assertEqual(details, model.training_details_)
            self.assertEqual(state, model.fit_state_sha256_)
            self.assertTrue(subject.verify_fitted_structured_identity(restored, spec, 53))

    def test_ensemble_is_equal_probability_mean_and_rejects_member_or_scope_replacement(self):
        model = self.models[2]
        query = self.X[:9]
        expected = np.mean([member.predict_proba(query) for member in model.members_], axis=0)
        self.assertEqual(expected.tobytes(), model.predict_proba(query).tobytes())
        for member in model.members_:
            for key in ("training_input_sha256", "training_labels_sha256", "training_groups_sha256", "training_records_sha256"):
                self.assertEqual(member.training_details_[key], model.training_details_[key])
        changed = deepcopy(model)
        changed.members_[0] = self.fit(self.pair, X=self.X + .3)
        with self.assertRaisesRegex(ValueError, "share TRAIN"):
            subject.verify_fitted_structured_identity(changed, self.voting, 53)
        changed = deepcopy(model)
        changed.training_details_["combination"] = "weighted"
        with self.assertRaises(ValueError):
            subject.verify_fitted_structured_identity(changed, self.voting, 53)

    def test_required_identities_record_purity_and_residual_person_coverage_fail(self):
        for spec in (self.pair, self.residual):
            for groups, records in ((None, self.records), (self.groups, None), (self.groups[:-1], self.records),
                    (self.groups, self.records[:-1]), (np.arange(len(self.y)), self.records),
                    (self.groups, [""] * len(self.y)), (self.groups, self.groups),
                    (self.groups, np.asarray([f"class-{c}" for c in self.y]))):
                with self.subTest(family=spec["family"]), self.assertRaises(ValueError):
                    subject.build_structured_pipeline(spec).fit(self.X, self.y, groups=groups, record_ids=records)
            for y in (self.y.astype(float), self.y[:, None], self.y[:-1], np.zeros_like(self.y)):
                with self.assertRaises(ValueError):
                    self.fit(spec, y=y)
        keep = ~((self.groups == "person-0") & (self.y == 6))
        with self.assertRaisesRegex(ValueError, "Every training person"):
            self.fit(self.residual, X=self.X[keep], y=self.y[keep], groups=self.groups[keep], records=self.records[keep])
        keep = self.groups != "person-3"
        with self.assertRaisesRegex(ValueError, "four training people"):
            self.fit(self.residual, X=self.X[keep], y=self.y[keep], groups=self.groups[keep], records=self.records[keep])

    def test_recipe_validation_has_fixed_family_scoring_policies_and_distinct_nonnested_members(self):
        before = deepcopy(self.voting)
        copied = subject.validate_structured_pipeline(self.voting)
        copied["members"][0]["model_params"]["n_estimators"] = 9
        self.assertEqual(self.voting, before)
        self.assertEqual(subject.validate_structured_pipeline({"family": subject.PAIR})["model_params"]["n_estimators"], 32)
        for bad in (None, {**self.pair, "extra": 1}, {**self.pair, "window_seconds": True},
                {**self.pair, "postprocessing": {"kind": "argmax"}},
                {**self.pair, "preprocessing": {"scaler": "none"}},
                {**self.pair, "training": {"seed": -1}}, {**self.pair, "training": {"seed": True}},
                {**self.pair, "training": {"weighting": "class_balanced"}},
                {**self.pair, "training": {"data_policy": "unknown"}},
                {**self.pair, "model_params": {"bootstrap": True}},
                {**self.pair, "model_params": {"n_estimators": False}},
                {**self.residual, "feature_engineering": {"mode": "remove_subject2"}},
                {**self.voting, "members": [self.pair, self.pair]},
                {**self.voting, "members": [self.pair, self.voting]},
                {**self.voting, "members": [self.pair, recipe(seed=54)]},
                {**self.voting, "members": [self.pair, recipe(policy=subject.POLICY)]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                subject.validate_structured_pipeline(bad)
        with self.assertRaises(ValueError):
            subject.fit_structured_pipeline(subject.build_structured_pipeline(self.pair), self.residual, self.X, self.y,
                groups=self.groups, record_ids=self.records)

    def test_full_fitted_verifier_rejects_native_tree_scaler_pair_scope_and_projector_tampering(self):
        common = (
            lambda m: m.scaler_.mean_.__setitem__(0, 100.),
            lambda m: m.record_metadata_[0]["row_indices"].__setitem__(0, 1),
            lambda m: m.classes_.__setitem__(0, 7),
            lambda m: m.training_details_["class_counts"].__setitem__(0, 1),
            lambda m: m.training_details_.update(external_validation_used=True),
            lambda m: setattr(m, "training_groups_sha256_", "a" * 64),
            lambda m: setattr(m, "fit_state_sha256_", "a" * 64),
        )
        for spec, original in zip((self.pair, self.residual), self.models[:2]):
            extra = (
                lambda m: m.heads_[0].estimators_[0].tree_.value.__setitem__((0, 0, 0), .2),
                lambda m: m.heads_[0].estimators_[0].tree_.threshold.__setitem__(0, 100.),
                lambda m: m.heads_[0].set_params(random_state=123),
                lambda m: m.pair_audits_[0].update(seed=123),
                lambda m: m.pair_audits_[0]["row_indices"].__setitem__(0, 100),
                lambda m: m.coupling_rule_.update(epsilon=.01),
            ) if spec["family"] == subject.PAIR else (
                lambda m: m.estimator_.estimators_[0].tree_.value.__setitem__((0, 0, 0), .2),
                lambda m: m.estimator_.estimators_[0].tree_.threshold.__setitem__(0, 100.),
                lambda m: m.estimator_.set_params(random_state=123),
                lambda m: m.projector_.components_.__setitem__((0, 0), 100.),
                lambda m: setattr(m.projector_, "components_", np.asfortranarray(m.projector_.components_)),
                lambda m: m.training_details_.update(standardized_input_sha256="a" * 64),
            )
            for mutate in common + extra:
                changed = deepcopy(original)
                mutate(changed)
                with self.assertRaises(ValueError):
                    subject.verify_fitted_structured_identity(changed, spec, 53)

    def test_named_numeric_input_and_frozen_constructor_contract(self):
        named = pd.DataFrame(self.X[:4], columns=feature_names())
        for model in self.models[:2]:
            self.assertEqual(model.predict_proba(named).tobytes(), model.predict_proba(self.X[:4]).tobytes())
            for bad in (named.iloc[:, ::-1], self.X[:, :-1], self.X[:0], np.full_like(self.X, np.nan), np.full_like(self.X, np.inf)):
                with self.assertRaises(ValueError):
                    model.predict_proba(bad)
            changed = deepcopy(model)
            changed.spec["model_params"]["n_estimators"] += 1
            with self.assertRaises(ValueError):
                changed.predict_proba(self.X[:1])


if __name__ == "__main__":
    unittest.main()
