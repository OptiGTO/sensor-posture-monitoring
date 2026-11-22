"""Synthetic-only tests for stored TRAIN bounds and exact identity precision."""
from copy import deepcopy
import hashlib
import io
import json
import unittest

import joblib
import numpy as np
from sklearn.base import clone

from src.tail_clipping_v1 import (TailClipperV1, tail_clipping_rule,
    tail_clipping_state_sha256, verify_tail_clipper)


def fixture(dtype=np.float32):
    rows = np.arange(101, dtype=dtype)[:, None]
    return np.broadcast_to(rows, (101, 596)).copy()


class TailClippingTests(unittest.TestCase):
    def test_known_train_bounds_clip_then_round_to_input_dtype(self):
        for dtype in (np.float32, np.float64):
            for fraction, expected in ((.005, [.5, 99.5]), (.01, [1., 99.])):
                train = fixture(dtype)
                clipper = TailClipperV1(fraction).fit(train)
                self.assertEqual(clipper.lower_bounds_.dtype, np.float64)
                np.testing.assert_array_equal(clipper.lower_bounds_, np.full(596, expected[0]))
                np.testing.assert_array_equal(clipper.upper_bounds_, np.full(596, expected[1]))
                query = np.broadcast_to(np.asarray([-100, .75, 50, 99.25, 200], dtype=dtype)[:, None], (5, 596)).copy()
                reference = np.clip(query.astype(np.float64), expected[0], expected[1]).astype(dtype)
                result = clipper.transform(query)
                self.assertEqual(result.dtype, np.dtype(dtype))
                self.assertEqual(result.tobytes(), reference.tobytes())
                self.assertTrue(verify_tail_clipper(clipper, fraction))
                self.assertEqual(clipper.training_details_["lower_bounds"], clipper.lower_bounds_.tolist())

    def test_bounds_use_only_fit_input_and_ignore_labels_or_query_distribution(self):
        class ForbiddenLabels:
            def __array__(self, *args, **kwargs):
                raise AssertionError("Labels must not be inspected")
        train = fixture()
        clipper = TailClipperV1(.01).fit(train, ForbiddenLabels())
        state = tail_clipping_state_sha256(clipper)
        query = np.full((3, 596), 10000, dtype=np.float32)
        np.testing.assert_array_equal(clipper.transform(query), np.full_like(query, 99))
        alone = clipper.transform(query[:1])
        huge_batch = np.concatenate((query, np.full_like(query, -1e20)))
        self.assertEqual(clipper.transform(huge_batch)[:1].tobytes(), alone.tobytes())
        self.assertEqual(tail_clipping_state_sha256(clipper), state)
        self.assertEqual(clipper.n_samples_seen_, len(train))
        self.assertEqual(TailClipperV1(.01).fit(train, np.arange(len(train))).state_sha256_, state)

    def test_identity_is_same_object_and_bytes_even_outside_train_range(self):
        for dtype in (np.float32, np.float64):
            clipper = TailClipperV1(0.).fit(fixture(dtype))
            query = np.full((3, 596), -10000, dtype=dtype)
            query[:, 1] = np.asarray(-0., dtype=dtype)
            query[:, 2] = 10000
            before = query.tobytes()
            self.assertIs(clipper.transform(query), query)
            self.assertEqual(clipper.transform(query).tobytes(), before)
            self.assertEqual(clipper.transform(query).dtype, dtype)
            self.assertIsNone(clipper.lower_bounds_)
            self.assertIsNone(clipper.upper_bounds_)
            self.assertTrue(verify_tail_clipper(clipper, 0))
        train = fixture()
        self.assertIs(TailClipperV1().fit_transform(train), train)

    def test_constant_columns_ties_and_single_train_row(self):
        train = np.zeros((201, 596), np.float32)
        train[:, 0] = 7
        train[-1, 1] = 100
        clipper = TailClipperV1(.01).fit(train)
        self.assertEqual(clipper.lower_bounds_[1], 0)
        self.assertEqual(clipper.upper_bounds_[1], 0)
        query = np.full((2, 596), -50, np.float32)
        expected = np.zeros_like(query)
        expected[:, 0] = 7
        np.testing.assert_array_equal(clipper.transform(query), expected)
        clipper = TailClipperV1(.005).fit(train[:1])
        np.testing.assert_array_equal(clipper.transform(query), np.repeat(train[:1], 2, axis=0))

    def test_batch_order_reload_no_input_mutation_and_sklearn_clone(self):
        train = fixture()
        before_train = train.tobytes()
        query = np.random.default_rng(42).normal(50, 100, size=(19, 596)).astype(np.float32)
        before_query = query.tobytes()
        for fraction in (0., .005, .01):
            clipper = TailClipperV1(fraction).fit(train)
            reference = clipper.transform(query)
            state = tail_clipping_state_sha256(clipper)
            together = np.concatenate([clipper.transform(query[:1]), clipper.transform(query[1:7]), clipper.transform(query[7:])])
            self.assertEqual(together.tobytes(), reference.tobytes())
            self.assertEqual(clipper.transform(query[::-1]).tobytes(), reference[::-1].tobytes())
            stream = io.BytesIO()
            joblib.dump(clipper, stream)
            stream.seek(0)
            restored = joblib.load(stream)
            self.assertTrue(verify_tail_clipper(restored, fraction))
            self.assertEqual(restored.transform(query).tobytes(), reference.tobytes())
            self.assertEqual(tail_clipping_state_sha256(restored), state)
            self.assertEqual(clone(clipper).get_params(), {"tail_fraction": fraction})
        self.assertEqual(train.tobytes(), before_train)
        self.assertEqual(query.tobytes(), before_query)

    def test_input_digest_matches_existing_c_order_shape_dtype_algorithm(self):
        for dtype in (np.float32, np.float64):
            train = fixture(dtype)
            digest = hashlib.sha256(json.dumps([train.dtype.str, list(train.shape)]).encode("utf-8"))
            digest.update(train.tobytes(order="C"))
            clipper = TailClipperV1(.005).fit(np.asfortranarray(train))
            self.assertEqual(clipper.training_input_sha256_, digest.hexdigest())
            self.assertEqual(clipper.state_sha256_, TailClipperV1(.005).fit(train).state_sha256_)
            self.assertEqual(clipper.training_dtype_, str(train.dtype))

    def test_rule_is_fresh_json_normalizes_fraction_and_hashes_exact_semantics(self):
        for fraction in (0., .005, .01):
            rule = tail_clipping_rule(fraction)
            self.assertIs(type(rule["tail_fraction"]), float)
            stored = rule.pop("rule_sha256")
            self.assertEqual(stored, hashlib.sha256(json.dumps(rule, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest())
            self.assertFalse(rule["uses_labels"])
            self.assertFalse(rule["evaluation_batch_statistics"])
            self.assertFalse(rule["refit_on_transform"])
            rule["input_dtypes"][0] = "changed"
            self.assertEqual(tail_clipping_rule(fraction)["input_dtypes"][0], "float32")

    def test_invalid_fractions_input_schema_and_unfitted_or_changed_config_rejected(self):
        train = fixture()
        for fraction in (True, np.bool_(False), -.005, .001, .1, np.nan, np.inf, "0.01", None):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                TailClipperV1(fraction).fit(train)
            with self.assertRaises(ValueError):
                tail_clipping_rule(fraction)
        bad_inputs = (train.astype(int), train.astype(np.float16), train.astype(complex), train.astype(object),
            train[:, :-1], train[0], train[:0], np.full_like(train, np.nan), np.full_like(train, np.inf))
        clipper = TailClipperV1(.01).fit(train)
        for bad in bad_inputs:
            with self.subTest(dtype=bad.dtype, shape=bad.shape), self.assertRaises(ValueError):
                TailClipperV1(.01).fit(bad)
            with self.assertRaises(ValueError):
                clipper.transform(bad)
        with self.assertRaises(ValueError):
            TailClipperV1(.01).transform(train)
        with self.assertRaises(ValueError):
            verify_tail_clipper(object(), .01)
        clipper.tail_fraction = .005
        with self.assertRaises(ValueError):
            clipper.transform(train)

    def test_complete_fitted_state_tampering_is_detected(self):
        fitted = TailClipperV1(.01).fit(fixture())
        mutations = [
            lambda x: x.lower_bounds_.__setitem__(0, -1.),
            lambda x: x.upper_bounds_.__setitem__(0, 101.),
            lambda x: setattr(x, "n_samples_seen_", 102),
            lambda x: setattr(x, "n_features_in_", 595),
            lambda x: setattr(x, "n_features_out_", 597),
            lambda x: setattr(x, "training_dtype_", "float64"),
            lambda x: setattr(x, "training_input_sha256_", "0" * 64),
            lambda x: setattr(x, "tail_fraction_", .005),
            lambda x: x.rule_.__setitem__("uses_labels", True),
            lambda x: x.training_details_.__setitem__("training_windows", 999),
            lambda x: setattr(x, "state_sha256_", "0" * 64),
            lambda x: setattr(x, "lower_bounds_", x.lower_bounds_.astype(np.float32)),
            lambda x: x.upper_bounds_.__setitem__(0, np.inf),
        ]
        for mutation in mutations:
            changed = deepcopy(fitted)
            mutation(changed)
            with self.assertRaises(ValueError):
                verify_tail_clipper(changed, .01)
        identity = TailClipperV1().fit(fixture())
        identity.lower_bounds_ = np.zeros(596)
        with self.assertRaises(ValueError):
            verify_tail_clipper(identity, 0)


if __name__ == "__main__":
    unittest.main()
