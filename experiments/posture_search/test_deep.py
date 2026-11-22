"""Synthetic contract tests; these do not measure posture classification quality."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from experiments.posture_search.deep import SUPPORTED_KINDS, build_deep


class DeepContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(718)
        cls.X = rng.uniform(0, 1023, (21, 36, 16)).astype(np.float32)
        cls.y = np.tile(np.arange(7), 3)
        cls.groups = np.repeat(["subject_A", "subject_B", "subject_C"], 7)
        cls.probe = rng.uniform(0, 1023, (7, 36, 16)).astype(np.float32)

    def test_all_architectures_fit_probabilities_embeddings_and_serialization(self) -> None:
        for kind in SUPPORTED_KINDS:
            with self.subTest(kind=kind):
                estimator = build_deep(kind, seed=17, epochs=1).fit(self.X, self.y, self.groups)
                probability = estimator.predict_proba(self.probe)
                self.assertEqual(probability.shape, (7, 7))
                self.assertTrue(np.isfinite(probability).all())
                np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)
                self.assertEqual(estimator.embedding(self.probe).shape, (7, 64))
                self.assertEqual(len(estimator.training_history_), 1)
                self.assertLess(estimator.complexity_["trainable_parameters"], 100_000)
                self.assertEqual(estimator.config_["training_groups"], ["subject_A", "subject_B", "subject_C"])
                self.assertEqual(estimator.predict_proba(self.probe[:0]).shape, (0, 7))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / f"{kind}.joblib"
                    joblib.dump(estimator, path)
                    restored = joblib.load(path)
                    np.testing.assert_array_equal(restored.predict_proba(self.probe), probability)
                    np.testing.assert_array_equal(restored.embedding(self.probe), estimator.embedding(self.probe))

    def test_determinism_and_train_only_normalization(self) -> None:
        a = build_deep("transformer", seed=23, epochs=1).fit(self.X, self.y)
        b = build_deep("transformer", seed=23, epochs=1).fit(self.X, self.y)
        np.testing.assert_array_equal(a.predict_proba(self.probe), b.predict_proba(self.probe))
        representation = np.concatenate((self.X / 1023, np.log1p(self.X) / np.log(1024),
                                         self.X / self.X.sum(axis=-1, keepdims=True)), axis=-1)
        np.testing.assert_allclose(a.normalizer_mean_.numpy(), representation.mean(axis=(0, 1), keepdims=True), rtol=2e-6)
        before = a.normalizer_mean_.numpy().copy()
        a.predict_proba(self.probe + 10_000)
        np.testing.assert_array_equal(a.normalizer_mean_.numpy(), before)

    def test_invalid_data_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "all seven classes"):
            build_deep("gru", epochs=1).fit(self.X[:6], self.y[:6])
        with self.assertRaisesRegex(ValueError, "integer labels"):
            build_deep("gru", epochs=1).fit(self.X, self.y + 0.1)
        with self.assertRaisesRegex(ValueError, "one entry"):
            build_deep("gru", epochs=1).fit(self.X, self.y, self.groups[:2])
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            build_deep("gru", epochs=1).fit(-self.X, self.y)


if __name__ == "__main__":
    unittest.main()
