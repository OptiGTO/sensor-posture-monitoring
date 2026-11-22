"""Fast contract tests; synthetic labels do not claim physiological accuracy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch

from experiments.posture_study.neural import NeuralEstimator, SUPPORTED_KINDS


class NeuralEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(19)
        cls.train = {
            "seq": rng.uniform(30, 850, (28, 36, 16)).astype(np.float32),
            "features": rng.normal(size=(28, 596)).astype(np.float32),
        }
        cls.y = np.tile(np.arange(7), 4)
        cls.validation = {
            "seq": rng.uniform(30, 850, (14, 36, 16)).astype(np.float32),
            "features": rng.normal(size=(14, 596)).astype(np.float32),
        }
        cls.yval = np.tile(np.arange(7), 2)

    def test_every_architecture_trains_and_produces_valid_outputs(self) -> None:
        for kind in SUPPORTED_KINDS:
            with self.subTest(kind=kind):
                estimator = NeuralEstimator(kind, seed=19, epochs=2, batch_size=14)
                estimator.fit(self.train, self.y, validation=(self.validation, self.yval))
                probabilities = estimator.predict_proba(self.validation)
                self.assertEqual(probabilities.shape, (14, 7))
                self.assertTrue(np.isfinite(probabilities).all())
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                self.assertEqual(estimator.embedding(self.validation).shape, (14, 64))
                self.assertGreater(estimator.complexity_["trainable_parameters"], 0)
                self.assertIn(estimator.best_epoch_, (1, 2))

    def test_seed_determinism_with_augmentation(self) -> None:
        estimators = [NeuralEstimator("1dcnn", seed=19, epochs=2, augmentation=True)
                      .fit(self.train, self.y, validation=(self.validation, self.yval))
                      for _ in range(2)]
        np.testing.assert_array_equal(estimators[0].predict_proba(self.validation),
                                      estimators[1].predict_proba(self.validation))

    def test_normalizer_uses_training_partition_only_and_survives_joblib(self) -> None:
        shifted_validation = {key: value + 100 for key, value in self.validation.items()}
        estimator = NeuralEstimator("gru", seed=19, epochs=2).fit(
            self.train, self.y, validation=(shifted_validation, self.yval))
        proportions = self.train["seq"] / self.train["seq"].sum(axis=-1, keepdims=True)
        representation = np.concatenate((self.train["seq"] / 1023, proportions), axis=-1)
        np.testing.assert_allclose(estimator.normalizer_mean_.numpy(),
                                   representation.mean(axis=(0, 1), keepdims=True), rtol=2e-6)
        before = estimator.normalizer_mean_.numpy().copy()
        probabilities = estimator.predict_proba(shifted_validation)
        np.testing.assert_array_equal(estimator.normalizer_mean_.numpy(), before)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "estimator.joblib"
            joblib.dump(estimator, checkpoint)
            reloaded = joblib.load(checkpoint)
            np.testing.assert_array_equal(reloaded.predict_proba(shifted_validation), probabilities)

    def test_missing_training_class_rejected(self) -> None:
        inputs = {key: value[:6] for key, value in self.train.items()}
        with self.assertRaisesRegex(ValueError, "every configured class"):
            NeuralEstimator("mlp", epochs=1).fit(inputs, np.arange(6))


class UnitAwareAugmentationTests(unittest.TestCase):
    """Exercise the recorded 1..3 sensor scale rather than assumed ADC counts."""

    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(2901)
        cls.train = {"seq": rng.uniform(1.0, 3.0, (28, 36, 16)).astype(np.float32)}
        cls.y = np.tile(np.arange(7), 4)
        cls.validation = {"seq": rng.uniform(1.0, 3.0, (14, 36, 16)).astype(np.float32)}
        cls.yval = np.tile(np.arange(7), 2)

    def fitted(self, training=None, validation=None, **options):
        return NeuralEstimator("1dcnn", seed=2901, epochs=1, augmentation=True,
                               batch_size=28, **options).fit(
            self.train if training is None else training, self.y,
            validation=(self.validation if validation is None else validation, self.yval))

    def test_augmentation_is_invariant_to_a_change_of_sensor_units(self) -> None:
        original = self.fitted()
        scaled_train = {"seq": self.train["seq"] * 100}
        scaled_validation = {"seq": self.validation["seq"] * 100}
        scaled = self.fitted(scaled_train, scaled_validation)
        raw = torch.from_numpy(self.validation["seq"])
        torch.manual_seed(9029)
        actual = original._transform(raw, augment=True).numpy()
        torch.manual_seed(9029)
        changed_units = scaled._transform(raw * 100, augment=True).numpy()
        np.testing.assert_allclose(actual, changed_units, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(scaled.augmentation_noise_scale_.numpy(),
                                   original.augmentation_noise_scale_.numpy() * 100,
                                   atol=1e-5, rtol=1e-6)

    def test_noise_scale_is_exact_training_population_std_and_validation_independent(self) -> None:
        original = self.fitted()
        shifted = self.fitted(validation={"seq": self.validation["seq"] * 10000 + 50000})
        expected = torch.from_numpy(self.train["seq"]).std(
            dim=(0, 1), keepdim=True, unbiased=False).clamp(min=1e-8)
        torch.testing.assert_close(original.augmentation_noise_scale_, expected, rtol=0, atol=0)
        torch.testing.assert_close(shifted.augmentation_noise_scale_, expected, rtol=0, atol=0)
        torch.testing.assert_close(original.normalizer_mean_, shifted.normalizer_mean_, rtol=0, atol=0)
        torch.testing.assert_close(original.normalizer_std_, shifted.normalizer_std_, rtol=0, atol=0)
        before = original.augmentation_noise_scale_.clone()
        original.predict_proba({"seq": self.validation["seq"] * 10000})
        torch.testing.assert_close(original.augmentation_noise_scale_, before, rtol=0, atol=0)

    def test_additive_noise_is_small_at_actual_recorded_units(self) -> None:
        model = self.fitted(gain_std=0.0)
        raw = torch.from_numpy(self.train["seq"])
        torch.manual_seed(391)
        transformed = model._transform(raw, augment=True)
        # Invert only the raw-value half of the representation to measure noise
        # in the input's actual units, independently of the proportional inputs.
        perturbed_raw = ((transformed * model.normalizer_std_ + model.normalizer_mean_)[..., :16] * 1023)
        noise = perturbed_raw - raw
        observed_std = noise.std(dim=(0, 1), keepdim=True, unbiased=False)
        expected_std = model.noise_std * model.augmentation_noise_scale_
        torch.testing.assert_close(observed_std, expected_std, rtol=0.12, atol=1e-6)
        self.assertLess(float(noise.abs().max()), 0.02)
        self.assertLess(float(noise.abs().mean()), 0.004)


if __name__ == "__main__":
    unittest.main()
