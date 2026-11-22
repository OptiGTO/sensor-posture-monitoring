"""Synthetic subject-OOF, record weighting and saved native-forest contracts."""
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

from src import hardness_pipeline_v1 as subject
from src.iterative_pipeline import build_pipeline as legacy_pipeline
from src.posture_core import feature_names


def recipe(family=subject.ET, coefficient=.5, seed=53):
    return subject.validate_hardness_pipeline({"family": family,
        "model_params": {"n_estimators": 4, "min_samples_leaf": 1},
        "training": {"seed": seed, "hardness": {"coefficient": coefficient, "n_splits": 4, "seed_offset": 1000}}})


class HardnessPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)
        cls.y = np.tile(np.repeat(np.arange(7), 3), 4)
        cls.groups = np.repeat([f"person-{i}" for i in range(4)], 21)
        cls.records = np.asarray([f"{person}-class-{label}" for person, label in zip(cls.groups, cls.y)])
        cls.X = np.random.default_rng(1801).normal(size=(len(cls.y), 596)).astype(np.float32)
        cls.X[:, 0] += cls.y
        cls.X[:, 1] = 0
        cls.et, cls.rf = recipe(), recipe(subject.RF)
        cls.voting = subject.validate_hardness_pipeline({"family": subject.VOTING,
            "members": [cls.et, cls.rf], "training": {"seed": 53}})
        cls.models = [subject.build_hardness_pipeline(spec).fit(cls.X, cls.y, groups=cls.groups, record_ids=cls.records)
            for spec in (cls.et, cls.rf, cls.voting)]

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_normalization_rejects_ambiguous_hardness_and_invalid_members(self):
        for changed in (None, {**self.et, "unexpected": 1}, {**self.et, "window_seconds": True},
                {**self.et, "feature_engineering": {"mode": "means"}},
                {**self.et, "model_params": {"bootstrap": True}},
                {**self.et, "training": {"hardness": {"coefficient": .5}}},
                {**self.et, "training": {"hardness": {"coefficient": True, "n_splits": 4, "seed_offset": 1000}}},
                {**self.et, "training": {"hardness": {"coefficient": .25, "n_splits": 4, "seed_offset": 1000}}},
                {**self.et, "training": {"hardness": {"coefficient": .5, "n_splits": 3, "seed_offset": 1000}}},
                {**self.et, "training": {"hardness": {"coefficient": .5, "n_splits": 4, "seed_offset": 1}}},
                {**self.et, "training": {"seed": 2**32}},
                {**self.voting, "members": [self.et, self.et]},
                {**self.voting, "members": [self.et, self.voting]},
                {**self.voting, "members": [self.et, recipe(subject.RF, seed=54)]},
                {**self.voting, "training": {"hardness": deepcopy(subject.HARDNESS_DEFAULTS)}}):
            with self.subTest(value=changed), self.assertRaises(ValueError):
                subject.validate_hardness_pipeline(changed)
        normalized = subject.validate_hardness_pipeline(self.voting)
        normalized["members"][0]["training"]["hardness"]["coefficient"] = 0
        self.assertEqual(self.voting["members"][0]["training"]["hardness"]["coefficient"], .5)

    def test_zero_skips_oof_and_sample_weight_and_matches_native200_baseline_exactly(self):
        spec = subject.validate_hardness_pipeline({"family": subject.ET,
            "training": {"seed": 53, "hardness": {"coefficient": 0., "n_splits": 4, "seed_offset": 1000}}})
        reference = {"family": "extra_trees", "model_params": spec["model_params"], "training": {"seed": 53}}
        for dtype in (np.float32, np.float64):
            X = self.X[:21].astype(dtype)
            seen = []
            original = ExtraTreesClassifier.fit
            def capture(estimator, values, labels, **kwargs):
                seen.append(deepcopy(kwargs))
                return original(estimator, values, labels, **kwargs)
            with patch.object(subject, "_oof", side_effect=AssertionError("zero must skip OOF")), \
                    patch.object(ExtraTreesClassifier, "fit", capture):
                model = subject.build_hardness_pipeline(spec).fit(X, self.y[:21],
                    groups=self.groups[:21], record_ids=self.records[:21])
            self.assertEqual(seen, [{}])
            baseline = legacy_pipeline(reference).fit(X, self.y[:21], groups=self.groups[:21])
            query = np.concatenate([X, X[:2] * 1000])
            self.assertEqual(model.predict_proba(query).tobytes(), baseline.predict_proba(query).tobytes())
            self.assertEqual(model.scaler_.mean_.tobytes(), baseline.named_steps["scale"].mean_.tobytes())
            self.assertIsNone(model.sample_weight_)
            self.assertIsNone(model.oof_prediction_)
            self.assertIsNone(model.oof_probability_)
            self.assertFalse(model.hardness_audit_["oof_performed"])
            self.assertTrue(subject.verify_fitted_hardness_identity(model, spec, 53))

    def test_each_oof_scaler_sees_only_inner_train_and_final_scaler_sees_unweighted_outer_train(self):
        class UnreadableValidation:
            def __array__(self, *args, **kwargs):
                raise AssertionError("External validation accessed")
        original = StandardScaler.fit
        seen = []
        def capture(scaler, X, y=None, sample_weight=None):
            seen.append((X.copy(), sample_weight))
            return original(scaler, X, y=y, sample_weight=sample_weight)
        with patch.object(StandardScaler, "fit", capture):
            model = subject.build_hardness_pipeline(self.et)
            subject.fit_hardness_pipeline(model, self.et, self.X, self.y, groups=self.groups,
                record_ids=self.records, validation_data=UnreadableValidation())
        splits = subject._splits(self.y, self.groups)
        self.assertEqual(len(seen), 5)
        for index, (train, heldout) in enumerate(splits):
            np.testing.assert_array_equal(seen[index][0], self.X[train])
            self.assertIsNone(seen[index][1])
            self.assertFalse(set(self.groups[train]) & set(self.groups[heldout]))
            self.assertEqual(len(np.unique(self.groups[train])), 3)
            row = model.hardness_audit_["folds"][index]
            self.assertEqual(row["training_input_sha256"], subject._array_digest(self.X[train]))
            self.assertEqual(row["heldout_input_sha256"], subject._array_digest(self.X[heldout]))
            np.testing.assert_allclose(row["scaler"]["mean_"], self.X[train].mean(axis=0, dtype=np.float64))
            self.assertEqual(row["model_params"]["n_estimators"], 200)
            self.assertEqual(row["model_params"]["min_samples_leaf"], 8)
            self.assertEqual(row["seed"], 53 + 1000 + index)
        np.testing.assert_array_equal(seen[-1][0], self.X)
        self.assertIsNone(seen[-1][1])
        np.testing.assert_allclose(model.scaler_.mean_, self.X.mean(axis=0, dtype=np.float64))

    def test_record_mean_errors_and_class_window_mean_normalization_have_exact_hand_calculation(self):
        labels = np.repeat(np.arange(7), 6)
        people = np.repeat("person", len(labels))
        records = np.asarray([f"{c}-{'short' if i < 2 else 'long'}" for c in range(7) for i in range(6)])
        predictions = labels.copy()
        for c in range(7):
            predictions[c * 6 + 2:c * 6 + 6] = (c + 1) % 7
        weights, audit = subject._record_weights(subject._metadata(labels, people, records), labels, predictions, .5)
        expected = np.tile([.75, .75, 1.125, 1.125, 1.125, 1.125], 7)
        np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(audit["class_denominators"], np.full(7, 4/3), rtol=0, atol=1e-15)
        np.testing.assert_allclose(np.bincount(labels, weights=weights), np.full(7, 6.))
        self.assertNotEqual(audit["class_denominators"][0], (1 + 1.5) / 2)
        for row in audit["records"]:
            self.assertEqual(row["error_rate"], 0. if row["record_id"].endswith("short") else 1.)

    def test_oof_predictions_cover_each_record_once_and_rf_uses_same_fixed_et_teacher(self):
        et, rf = self.models[:2]
        self.assertEqual(et.oof_probability_.tobytes(), rf.oof_probability_.tobytes())
        self.assertEqual(et.oof_prediction_.tobytes(), rf.oof_prediction_.tobytes())
        self.assertEqual(et.sample_weight_.tobytes(), rf.sample_weight_.tobytes())
        np.testing.assert_array_equal(et.oof_prediction_, et.oof_probability_.argmax(1))
        covered = np.zeros(len(self.y), dtype=int)
        for row in et.hardness_audit_["folds"]:
            covered[row["heldout_indices"]] += 1
            self.assertEqual(set(row["model_params"]) & {"bootstrap"}, {"bootstrap"})
            self.assertFalse(row["model_params"]["bootstrap"])
        np.testing.assert_array_equal(covered, np.ones(len(self.y)))
        for row in et.record_metadata_:
            idx = row["row_indices"]
            self.assertEqual(len(np.unique(et.oof_fold_[idx])), 1)
            expected_error = np.mean(et.oof_prediction_[idx] != self.y[idx])
            audit = next(r for r in et.hardness_audit_["weighting"]["records"] if r["record_id"] == row["record_id"])
            self.assertEqual(audit["error_rate"], expected_error)
        self.assertFalse(rf.estimator_.bootstrap)
        self.assertEqual(rf.estimator_.class_weight, "balanced")
        for model, spec in zip(self.models[:2], (self.et, self.rf)):
            self.assertTrue(np.isfinite(model.sample_weight_).all())
            self.assertTrue(np.all(model.sample_weight_ > 0))
            np.testing.assert_allclose(np.bincount(self.y, weights=model.sample_weight_), np.bincount(self.y), rtol=1e-12)
            self.assertTrue(subject.verify_fitted_hardness_identity(model, spec, 53))

    def test_missing_cross_subject_cross_class_records_and_insufficient_folds_fail_without_fallback(self):
        for groups, records in ((None, self.records), (self.groups, None), (self.groups[:-1], self.records),
                (np.arange(len(self.y)), self.records), (self.groups, [""] * len(self.y)),
                (self.groups, np.asarray([f'class-{c}' for c in self.y])),
                (self.groups, self.groups)):
            with self.assertRaises(ValueError):
                subject.build_hardness_pipeline(self.et).fit(self.X, self.y, groups=groups, record_ids=records)
        keep = self.groups != "person-3"
        with self.assertRaisesRegex(ValueError, "four outerTRAIN"):
            subject.build_hardness_pipeline(self.et).fit(self.X[keep], self.y[keep], groups=self.groups[keep], record_ids=self.records[keep])
        keep = (self.y != 6) | (self.groups == "person-0")
        with self.assertRaisesRegex(ValueError, "all seven classes"):
            subject.build_hardness_pipeline(self.et).fit(self.X[keep], self.y[keep], groups=self.groups[keep], record_ids=self.records[keep])

    def test_seed_override_wraps_inner_uint32_and_ensemble_members_use_same_effective_seed(self):
        seed = 2**32 - 1001
        model = subject.build_hardness_pipeline(self.et, seed=seed).fit(self.X, self.y, groups=self.groups, record_ids=self.records)
        self.assertEqual([row["seed"] for row in model.hardness_audit_["folds"]], [2**32 - 1, 0, 1, 2])
        self.assertEqual(model.estimator_.random_state, seed)
        self.assertEqual(model.spec_, self.et)
        self.assertTrue(subject.verify_fitted_hardness_identity(model, self.et, seed))
        ensemble = self.models[2]
        self.assertEqual([m.effective_seed_ for m in ensemble.members_], [53, 53])
        np.testing.assert_array_equal(ensemble.predict_proba(self.X), np.mean([m.predict_proba(self.X) for m in ensemble.members_], axis=0))
        self.assertTrue(subject.verify_fitted_hardness_identity(ensemble, self.voting, 53))

    def test_reload_refit_and_batch_queries_preserve_probabilities_and_do_not_reweight(self):
        for model, spec in zip(self.models, (self.et, self.rf, self.voting)):
            expected = model.predict_proba(self.X)
            memory = io.BytesIO()
            joblib.dump(model, memory)
            memory.seek(0)
            restored = joblib.load(memory)
            self.assertEqual(expected.tobytes(), restored.predict_proba(self.X).tobytes())
            self.assertTrue(subject.verify_fitted_hardness_identity(restored, spec, 53))
            details = deepcopy(model.training_details_)
            with patch.object(subject, "_oof", side_effect=AssertionError("inference must not refit")):
                self.assertEqual(expected.tobytes(), np.concatenate([model.predict_proba(self.X[:1]), model.predict_proba(self.X[1:])]).tobytes())
                np.testing.assert_array_equal(model.predict_proba(self.X[::-1]), expected[::-1])
            self.assertEqual(details, model.training_details_)
        repeated = subject.build_hardness_pipeline(self.et).fit(self.X, self.y, groups=self.groups, record_ids=self.records)
        self.assertEqual(repeated.predict_proba(self.X).tobytes(), self.models[0].predict_proba(self.X).tobytes())
        self.assertEqual(repeated.oof_probability_.tobytes(), self.models[0].oof_probability_.tobytes())
        self.assertEqual(repeated.fit_state_sha256_, self.models[0].fit_state_sha256_)

    def test_metadata_probabilities_weights_fold_scaler_and_final_forest_tampering_are_rejected(self):
        for mutate in (lambda m: m.oof_prediction_.__setitem__(0, (int(m.oof_prediction_[0]) + 1) % 7),
                lambda m: m.oof_probability_.__setitem__((0, 0), .12345),
                lambda m: m.sample_weight_.__setitem__(0, 0),
                lambda m: m.oof_fold_.__setitem__(0, 3 - int(m.oof_fold_[0])),
                lambda m: m.hardness_audit_["folds"][0]["train_subjects"].append("evaluation-person"),
                lambda m: m.hardness_audit_["folds"][0]["scaler"]["mean_"].__setitem__(0, 1e8),
                lambda m: m.record_metadata_[0].update(subject="evaluation-person"),
                lambda m: m.scaler_.mean_.__setitem__(0, 1e8),
                lambda m: m.estimator_.estimators_[0].tree_.value.__setitem__((0, 0, 0), .5),
                lambda m: setattr(m.estimator_, "bootstrap", True),
                lambda m: m.training_details_.update(external_validation_used=True),
                lambda m: m.training_details_.update(training_feature_dtype="float64"),
                lambda m: m.training_details_.update(preprocessing_fit="evaluation-trained scaler"),
                lambda m: setattr(m, "fit_state_sha256_", "a" * 64)):
            model = deepcopy(self.models[0])
            mutate(model)
            with self.assertRaises(ValueError):
                subject.verify_fitted_hardness_identity(model, self.et, 53)

    def test_replacing_valid_ensemble_member_from_other_training_rows_is_rejected(self):
        model = deepcopy(self.models[2])
        replacement = subject.build_hardness_pipeline(self.rf).fit(self.X + np.float32(1), self.y,
            groups=self.groups, record_ids=self.records)
        model.members_[1] = replacement
        model.training_details_["members"][1] = deepcopy(replacement.training_details_)
        with self.assertRaisesRegex(ValueError, "different TRAIN scopes"):
            subject.verify_fitted_hardness_identity(model, self.voting, 53)

    def test_input_schema_and_constructor_recipe_changes_fail(self):
        for X in (self.X[:, :-1], self.X[:0], self.X[0], np.full_like(self.X, np.nan), self.X.astype(str)):
            with self.assertRaises(ValueError):
                subject.build_hardness_pipeline(self.et).fit(X, self.y, groups=self.groups, record_ids=self.records)
        for y in (self.y.astype(float), self.y[:-1], np.zeros_like(self.y)):
            with self.assertRaises(ValueError):
                subject.build_hardness_pipeline(self.et).fit(self.X, y, groups=self.groups, record_ids=self.records)
        model = deepcopy(self.models[0])
        named = pd.DataFrame(self.X, columns=feature_names())
        np.testing.assert_array_equal(model.predict_proba(named), model.predict_proba(self.X))
        with self.assertRaises(ValueError):
            model.predict_proba(named.iloc[:, ::-1])
        with self.assertRaises(ValueError):
            subject.fit_hardness_pipeline(model, self.rf, self.X, self.y, groups=self.groups, record_ids=self.records)
        model.spec["training"]["hardness"]["coefficient"] = 0
        with self.assertRaises(ValueError):
            model.predict_proba(self.X)


if __name__ == "__main__":
    unittest.main()
