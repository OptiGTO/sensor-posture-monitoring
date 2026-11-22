"""Leakage boundaries and probability contracts for isolated experiment models."""
import io
import json
import unittest

import joblib
import numpy as np

from experiments.posture_search.classical import build_classical


class PostureSearchClassicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.y = np.tile(np.arange(7), 12)
        cls.groups = np.repeat(np.arange(6), 14)
        cls.X = rng.normal(size=(84, 10)) + cls.y[:, None] * .2

    def test_grouped_models_refuse_missing_group_information(self):
        for kind in ("subject_balanced_svm", "group_bagging", "stacking"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                build_classical(kind).fit(self.X, self.y)

    def test_stacking_meta_features_are_from_disjoint_subjects(self):
        model = build_classical("stacking", seed=9).fit(self.X, self.y, self.groups)
        validation_subjects = []
        for fold in model.config_["oof_folds"]:
            self.assertFalse(set(fold["train_subjects"]) & set(fold["validation_subjects"]))
            self.assertEqual(set(fold["train_subjects"]) | set(fold["validation_subjects"]), set(self.groups))
            validation_subjects.extend(fold["validation_subjects"])
        self.assertEqual(sorted(validation_subjects), sorted(set(self.groups)))

    def test_missing_class_alignment_and_serialization(self):
        selected = self.y != 2
        for kind in ("baseline", "hierarchical_correct", "hierarchical_direction"):
            with self.subTest(kind=kind):
                model = build_classical(kind).fit(self.X[selected], self.y[selected], self.groups[selected])
                prediction = model.predict_proba(self.X[:8])
                self.assertEqual(prediction.shape, (8, 7))
                np.testing.assert_allclose(prediction.sum(axis=1), 1)
                np.testing.assert_array_equal(prediction[:, 2], 0)
                json.dumps(model.config_, allow_nan=False)
                buffer = io.BytesIO()
                joblib.dump(model, buffer)
                buffer.seek(0)
                np.testing.assert_allclose(joblib.load(buffer).predict_proba(self.X[:8]), prediction)

    def test_refit_does_not_keep_previous_single_model_decisions(self):
        model = build_classical("baseline").fit(self.X, self.y)
        model.set_params(kind="hierarchical_correct").fit(self.X, self.y)
        self.assertFalse(hasattr(model, "estimator_"))
        np.testing.assert_array_equal(model.predict(self.X), model.predict_proba(self.X).argmax(axis=1))


if __name__ == "__main__":
    unittest.main()
