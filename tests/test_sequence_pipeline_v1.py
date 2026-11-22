from __future__ import annotations

from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone

from src.sequence_pipeline_v1 import (SequenceIterativePipeline, build_sequence_pipeline,
    fit_sequence_pipeline, validate_sequence_pipeline)


ROOT = Path(__file__).resolve().parents[1]


def recipe(family="cnn1d", **changes):
    value = {"family": family, "base_method": "standard", "window_seconds": 3,
        "feature_engineering": {"mode": "raw_sequence"},
        "preprocessing": {"scaler": "model_internal"},
        "model_params": ({"hidden_size": 8, "epochs": 2, "batch_size": 7, "patience": 1}
            if family == "cnn1d" else {"num_kernels": 84, "C": .1}),
        "training": {"seed": 53, "weighting": "class_balanced"},
        "postprocessing": {"kind": "argmax"}}
    value.update(changes)
    return value


class SequenceContractTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(51).normal(size=(21, 36, 16))
        self.y = np.tile(np.arange(7), 3)

    def test_spec_normalization_clone_and_builder_never_mutate_original(self):
        original = recipe()
        before = deepcopy(original)
        normalized = validate_sequence_pipeline(original)
        self.assertEqual(original, before)
        self.assertEqual(normalized["model_params"]["gain_std"], 0.)
        self.assertEqual(normalized["model_params"]["learning_rate"], .001)
        model = build_sequence_pipeline(original, seed=71)
        self.assertEqual(clone(model).get_params(), model.get_params())
        model.spec["model_params"]["epochs"] = 99
        self.assertEqual(original, before)
        json.dumps(normalized, allow_nan=False)

    def test_schema_rejects_incompatible_preprocessing_labels_parameters_and_seed(self):
        cases = [dict(family="mlp"), dict(family=[]), dict(window_seconds=True),
            dict(window_seconds=4), dict(feature_engineering={"mode": "all"}),
            dict(preprocessing={"scaler": "standard"}), dict(postprocessing={"kind": "macro_f1_bias"}),
            dict(training={"seed": -1}), dict(training={"seed": 2**31}),
            dict(training={"weighting": "subject_class_balanced"}),
            dict(training={"data_policy": "unreviewed"}),
            dict(model_params={"architecture": "mlp"}), dict(model_params={"gain_std": .03}),
            dict(model_params={"noise_fraction": .01}), dict(model_params={"epochs": True}),
            dict(model_params={"dropout": 1.}), dict(model_params={"weight_decay": float("nan")}),
            dict(model_params={"learning_rate": 0.}), dict(model_params={"hidden_size": 0})]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_sequence_pipeline(recipe(**change))
        for params in ({"num_kernels": 83}, {"num_kernels": 1000}, {"C": 0.},
                {"C": float("inf")}, {"num_kernels": True}, {"random_state": 2}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                validate_sequence_pipeline(recipe("minirocket", model_params=params))

    def test_invalid_input_and_groups_fail_before_native_fit(self):
        arrays = [np.zeros((21, 596)), self.X[:, :35], self.X[:, :, :15], self.X[:0],
            np.full_like(self.X, np.nan), np.full_like(self.X, 1e300), self.X.astype(complex),
            np.ones_like(self.X, dtype=bool)]
        for values in arrays:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                build_sequence_pipeline(recipe()).fit(values, self.y)
        for labels in (self.y[:-1], np.zeros_like(self.y), self.y.astype(float), self.y.reshape(-1, 1)):
            with self.subTest(labels=labels.shape), self.assertRaises(ValueError):
                build_sequence_pipeline(recipe()).fit(self.X, labels)
        for groups in (np.arange(20), np.full(21, ""), np.full(21, np.nan)):
            with self.subTest(groups=groups.dtype), self.assertRaises(ValueError):
                build_sequence_pipeline(recipe()).fit(self.X, self.y, groups=groups)
        with self.assertRaises(ValueError):
            fit_sequence_pipeline(build_sequence_pipeline(recipe()), recipe("minirocket"), self.X, self.y)

    def test_ensemble_accepts_only_distinct_compatible_sequence_members(self):
        first, second = recipe(), recipe("minirocket")
        combined = {"family": "sequence_soft_voting", "members": [first, second], "training": {"seed": 53}}
        normalized = validate_sequence_pipeline(combined)
        self.assertEqual([item["family"] for item in normalized["members"]], ["cnn1d", "minirocket"])
        for members in ([first], [first, first], [first, recipe("mlp")],
                [first, {**second, "training": {"seed": 54}}],
                [first, {**second, "training": {"seed": 53, "data_policy": "include_ordinary_reserved_training_records_v1"}}],
                [first, normalized]):
            with self.subTest(members=[item["family"] for item in members]), self.assertRaises(ValueError):
                validate_sequence_pipeline({**combined, "members": members})

    def test_aligned_probability_contract_and_fitted_recipe_mutation_guard(self):
        model = build_sequence_pipeline(recipe("minirocket"))
        model.spec_ = deepcopy(model.spec)
        model.effective_seed_ = 53
        model.classes_ = np.arange(7)
        reverse = np.arange(6, -1, -1)
        probability = np.arange(1, 8, dtype=float) / 28.
        estimator = unittest.mock.Mock()
        estimator.classes_ = reverse
        estimator.predict_proba.return_value = np.tile(probability, (21, 1))
        model.estimator_ = estimator
        np.testing.assert_array_equal(model.predict_proba(self.X), np.tile(probability[::-1], (21, 1)))
        np.testing.assert_array_equal(model.predict(self.X), np.zeros(21, dtype=int))
        for bad in (np.full((21, 7), .1), np.full((21, 7), np.nan), np.ones((21, 6))):
            estimator.predict_proba.return_value = bad
            with self.assertRaises(ValueError):
                model.predict_proba(self.X)
        estimator.predict_proba.return_value = np.tile(probability, (21, 1))
        model.spec["model_params"]["C"] = 9.
        with self.assertRaises(ValueError):
            model.predict_proba(self.X)


NATIVE_CHECK = r'''
from copy import deepcopy
import io
import json
import sys
import joblib
import numpy as np
from sklearn.base import clone
from src.sequence_pipeline_v1 import build_sequence_pipeline, fit_sequence_pipeline, validate_sequence_pipeline

family = sys.argv[1]
params = ({'hidden_size': 8, 'epochs': 2, 'batch_size': 7, 'patience': 1}
    if family == 'cnn1d' else {'num_kernels': 84, 'C': .1})
spec = validate_sequence_pipeline({'family': family, 'model_params': params, 'training': {'seed': 53}})
rng = np.random.default_rng(191)
X = rng.normal(size=(28, 36, 16))
y = np.tile(np.arange(7), 4)
X[:, :, :7] += np.eye(7)[y, None, :] * 2
groups = np.repeat(['a', 'b', 'c', 'd'], 7)
original, original_y, original_groups, original_spec = X.copy(), y.copy(), groups.copy(), deepcopy(spec)
first = fit_sequence_pipeline(build_sequence_pipeline(spec), spec, X, y, groups=groups)
second = build_sequence_pipeline(spec).fit(X, y, groups=groups,
    validation_data=(np.full((2, 36, 16), np.nan), np.array([-1, 99])))
p = first.predict_proba(X)
assert p.shape == (28, 7) and np.isfinite(p).all() and (p >= 0).all()
np.testing.assert_allclose(p.sum(1), 1, atol=1e-7, rtol=1e-6)
np.testing.assert_array_equal(first.predict(X), p.argmax(1))
np.testing.assert_array_equal(p, second.predict_proba(X))
np.testing.assert_allclose(p, np.concatenate([first.predict_proba(X[:13]), first.predict_proba(X[13:])]), atol=1e-7, rtol=0)
np.testing.assert_array_equal(X, original)
np.testing.assert_array_equal(y, original_y)
np.testing.assert_array_equal(groups, original_groups)
assert spec == original_spec and clone(first).spec == spec
assert first.input_shape_ == (36, 16) and first.n_features_in_ == 16
assert first.training_details_['training_subject_count'] == 4
assert first.training_details_['class_counts'] == [4] * 7
assert first.training_details_['external_validation_used'] is False
json.dumps(first.training_details_, allow_nan=False)
stream = io.BytesIO(); joblib.dump(first, stream); stream.seek(0)
loaded = joblib.load(stream)
np.testing.assert_array_equal(p, loaded.predict_proba(X))
with np.testing.assert_raises(ValueError):
    loaded.predict_proba(X[:, :35])
loaded.seed = 54
with np.testing.assert_raises(ValueError):
    loaded.predict_proba(X)
if family == 'cnn1d':
    values = X.astype(np.float32)
    np.testing.assert_array_equal(first.estimator_.mean_, values.mean(axis=(0, 1), dtype=np.float64).astype(np.float32))
    np.testing.assert_array_equal(first.estimator_.scale_, values.std(axis=(0, 1), dtype=np.float64).astype(np.float32))
    assert first.estimator_.training_details_['epochs_run'] == 2
    assert all('validation_macro_f1' not in row for row in first.estimator_.training_details_['history'])
    other = deepcopy(spec); other['model_params']['hidden_size'] = 4
    ensemble_spec = {'family': 'sequence_soft_voting', 'members': [spec, other], 'training': {'seed': 53}}
    ensemble = build_sequence_pipeline(ensemble_spec, seed=59).fit(X, y, groups=groups)
    assert all(member.effective_seed_ == 59 for member in ensemble.members_)
    expected = np.mean([member.predict_proba(X) for member in ensemble.members_], axis=0)
    np.testing.assert_array_equal(ensemble.predict_proba(X), expected)
    stream = io.BytesIO(); joblib.dump(ensemble, stream); stream.seek(0)
    np.testing.assert_array_equal(joblib.load(stream).predict_proba(X), expected)
    assert len(ensemble.training_details_['members']) == 2
else:
    assert first.training_details_['normalization']['actual_transform_features'] == 84
    assert first.estimator_.readout_.named_steps['scale'].n_samples_seen_ == len(X)
    for one, two in zip(first.estimator_.parameters_, second.estimator_.parameters_):
        np.testing.assert_array_equal(one, two)
print(json.dumps({'family': family, 'repeated_probability_error': float(np.max(np.abs(p-second.predict_proba(X)))),
    'reload_probability_error': 0., 'rows': len(X), 'passed': True}))
'''


class NativeSequenceTests(unittest.TestCase):
    def test_import_and_build_do_not_load_optional_native_libraries(self):
        script = ("import sys; from src.sequence_pipeline_v1 import build_sequence_pipeline; "
            "build_sequence_pipeline({'family':'cnn1d'}); build_sequence_pipeline({'family':'minirocket'}); "
            "assert 'torch' not in sys.modules and 'numba' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def native_check(self, family, package):
        if importlib.util.find_spec(package) is None:
            self.skipTest(package + " is not installed")
        with tempfile.TemporaryDirectory(prefix="sequence-v1-native-") as cache:
            env = {**os.environ, "NUMBA_CACHE_DIR": cache, "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}
            result = subprocess.run([sys.executable, "-c", NATIVE_CHECK, family], cwd=ROOT,
                text=True, capture_output=True, env=env, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        evidence = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["repeated_probability_error"], 0.)
        self.assertEqual(evidence["reload_probability_error"], 0.)

    def test_cnn_training_only_normalization_repeated_fit_reload_and_ensemble(self):
        self.native_check("cnn1d", "torch")

    def test_minirocket_training_only_transform_repeated_fit_and_reload(self):
        self.native_check("minirocket", "numba")

    def test_mixed_cnn_minirocket_voting_fits_and_reloads_in_isolated_process(self):
        if any(importlib.util.find_spec(package) is None for package in ("torch", "numba")):
            self.skipTest("Torch and Numba are required for the mixed sequence ensemble")
        script = r'''
import io
import joblib
import numpy as np
from src.sequence_pipeline_v1 import build_sequence_pipeline
rng = np.random.default_rng(701)
X = rng.normal(size=(21, 36, 16))
y = np.tile(np.arange(7), 3)
members = [
    {'family': 'cnn1d', 'model_params': {'epochs': 1, 'hidden_size': 4}},
    {'family': 'minirocket', 'model_params': {'num_kernels': 84, 'C': .1}}]
model = build_sequence_pipeline({'family': 'sequence_soft_voting', 'members': members}, seed=53).fit(X, y)
expected = np.mean([member.predict_proba(X) for member in model.members_], axis=0)
np.testing.assert_array_equal(model.predict_proba(X), expected)
buffer = io.BytesIO(); joblib.dump(model, buffer); buffer.seek(0)
np.testing.assert_array_equal(joblib.load(buffer).predict_proba(X), expected)
assert all(member.effective_seed_ == 53 for member in model.members_)
print('Mixed native sequence ensemble fit/reload passed')
'''
        with tempfile.TemporaryDirectory(prefix="sequence-v1-mixed-") as cache:
            result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True,
                capture_output=True, timeout=180, env={**os.environ, "NUMBA_CACHE_DIR": cache,
                    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "NUMBA_NUM_THREADS": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Mixed native sequence ensemble fit/reload passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
