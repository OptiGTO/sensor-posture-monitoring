"""Synthetic checks of fixed distance geometry; no classifier/data fitting."""
from copy import deepcopy
import hashlib
import json
import math
import unittest

import numpy as np
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.metrics.pairwise import laplacian_kernel, rbf_kernel

from src import distance_kernels_v1 as subject


class DistanceKernelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260922)
        self.X = rng.normal(size=(13, 596)).astype(np.float32)
        self.Y = rng.normal(size=(9, 596)).astype(np.float64)

    def test_independent_scalar_formula_and_sklearn_references(self):
        # Independent scalar fsum avoids reproducing the implementation reduction.
        for kind in subject.KINDS:
            actual = subject.distance_kernel(self.X, self.Y, kind)
            scalar = np.empty_like(actual)
            for i, x in enumerate(self.X.astype(np.float64)):
                for j, y in enumerate(self.Y):
                    delta = [float(a) - float(b) for a, b in zip(x, y)]
                    total = math.fsum(abs(v) if kind == "laplacian" else v * v for v in delta)
                    scalar[i, j] = (1 / (1 + total / 596) if kind == "rational_quadratic"
                                    else math.exp(-total / 596))
            np.testing.assert_allclose(actual, scalar, rtol=3e-15, atol=3e-15)
            if kind == "rbf":
                reference = rbf_kernel(self.X.astype(np.float64), self.Y, gamma=1 / 596)
            elif kind == "laplacian":
                reference = laplacian_kernel(self.X.astype(np.float64), self.Y, gamma=1 / 596)
            else:
                reference = RationalQuadratic(length_scale=np.sqrt(298), alpha=1,
                    length_scale_bounds="fixed", alpha_bounds="fixed")(self.X.astype(np.float64), self.Y)
            np.testing.assert_allclose(actual, reference, rtol=3e-15, atol=3e-15)
            self.assertEqual(actual.shape, (13, 9))
            self.assertEqual(actual.dtype, np.float64)
            self.assertTrue(actual.flags.c_contiguous)

    def test_geometry_distinguishes_l1_and_rq_from_single_rbf_scale(self):
        origin = np.zeros((1, 596), np.float64)
        # Equal L2 distance, different L1 distances: concentrated versus diffuse.
        targets = np.zeros((2, 596), np.float64)
        targets[0, 0], targets[1, :4] = 2., 1.
        for kind in ("rbf", "rational_quadratic"):
            probability = subject.distance_kernel(origin, targets, kind)[0]
            self.assertEqual(probability[0], probability[1])
        laplace = subject.distance_kernel(origin, targets, "laplacian")[0]
        self.assertGreater(laplace[0], laplace[1])
        anchors = np.stack([np.ones(596), np.full(596, 2.)])
        rq = subject.distance_kernel(origin, anchors, "rational_quadratic")[0]
        rbf = subject.distance_kernel(origin, anchors, "rbf")[0]
        np.testing.assert_array_equal(rq, [0.5, 0.2])
        np.testing.assert_allclose(rbf, np.exp([-1., -4.]), rtol=0, atol=0)
        # Matching RBF at t=1 gives exp(-log(2)*4)=1/16 at t=4, not RQ's 1/5.
        self.assertNotAlmostEqual(math.exp(-math.log(2) * 4), rq[1])

    def test_symmetric_unit_diagonal_finite_and_numerical_psd_with_duplicates(self):
        values = np.concatenate([self.X, self.X[:2], np.zeros((1, 596), np.float32)])
        for kind in subject.KINDS:
            kernel = subject.distance_kernel(values, values, kind)
            self.assertTrue(np.isfinite(kernel).all())
            self.assertTrue(np.all((kernel >= 0) & (kernel <= 1)))
            self.assertEqual(kernel.tobytes(), kernel.T.copy(order="C").tobytes())
            np.testing.assert_array_equal(np.diag(kernel), np.ones(len(values)))
            self.assertGreaterEqual(np.linalg.eigvalsh(kernel).min(), -1e-12)
            np.testing.assert_array_equal(kernel[:2], kernel[13:15])

    def test_row_and_reference_batch_order_layout_and_dtype_conversion_are_exact(self):
        query_order = np.random.default_rng(501).permutation(len(self.X))
        reference_order = np.random.default_rng(502).permutation(len(self.Y))
        for kind in subject.KINDS:
            expected = subject.distance_kernel(self.X, self.Y, kind)
            variants = [
                np.vstack([subject.distance_kernel(row[None], self.Y, kind) for row in self.X]),
                np.hstack([subject.distance_kernel(self.X, row[None], kind) for row in self.Y]),
                subject.distance_kernel(self.X[query_order], self.Y, kind)[np.argsort(query_order)],
                subject.distance_kernel(self.X, self.Y[reference_order], kind)[:, np.argsort(reference_order)],
                subject.distance_kernel(np.asfortranarray(self.X), np.asfortranarray(self.Y), kind),
                subject.distance_kernel(self.X.astype(np.float64), self.Y, kind),
                subject.distance_kernel(self.X[::-1], self.Y[::-1], kind)[::-1, ::-1],
            ]
            for value in variants:
                self.assertEqual(value.tobytes(order="C"), expected.tobytes())

    def test_readonly_callers_signed_zero_and_results_have_no_shared_state(self):
        X, Y = self.X.copy(), self.Y.copy()
        X[0, :2], Y[0, :2] = [-0., +0.], [+0., -0.]
        x_bytes, y_bytes = X.tobytes(), Y.tobytes()
        X.flags.writeable = Y.flags.writeable = False
        for kind in subject.KINDS:
            first = subject.distance_kernel(X, Y, kind)
            expected = first.copy()
            first[0, 0] = 123.
            self.assertEqual(subject.distance_kernel(X, Y, kind).tobytes(), expected.tobytes())
            self.assertEqual(X.tobytes(), x_bytes)
            self.assertEqual(Y.tobytes(), y_bytes)
            zeros = np.zeros((2, 596), np.float64)
            zeros[0, ::2] = -0.
            np.testing.assert_array_equal(subject.distance_kernel(zeros, zeros, kind), np.ones((2, 2)))

    def test_reject_invalid_inputs_and_kinds(self):
        invalid = [np.empty((0, 596)), np.zeros((2, 595)), np.zeros(596),
                   np.zeros((1, 2, 596)), np.zeros((1, 596), np.int64),
                   np.zeros((1, 596), bool), np.zeros((1, 596), np.float16),
                   np.zeros((1, 596), complex), np.zeros((1, 596), object),
                   np.full((1, 596), np.nan), np.full((1, 596), np.inf)]
        for kind in subject.KINDS:
            for value in invalid:
                with self.subTest(kind=kind, shape=value.shape, dtype=value.dtype):
                    with self.assertRaises(ValueError):
                        subject.distance_kernel(value, self.Y, kind)
                    with self.assertRaises(ValueError):
                        subject.distance_kernel(self.X, value, kind)
        for kind in ("cauchy", "RBF", "", None, False, {}, [], np.nan):
            with self.assertRaises(ValueError):
                subject.distance_kernel(self.X, self.Y, kind)
            with self.assertRaises(ValueError):
                subject.distance_kernel_rule(kind)

    def test_difference_and_distance_overflow_fail_without_clipping(self):
        maximum = np.finfo(np.float64).max
        for kind in subject.KINDS:
            with self.assertRaises(ValueError):
                subject.distance_kernel(np.full((1, 596), maximum),
                                        np.full((1, 596), -maximum), kind)
        # Every individual difference is finite; only the reduction overflows.
        for kind, large in (("laplacian", maximum / 2), ("rbf", 1e154),
                            ("rational_quadratic", 1e154)):
            with self.assertRaises(ValueError):
                subject.distance_kernel(np.full((1, 596), large), np.zeros((1, 596)), kind)
        # Extremely large equal inputs must work: no x*x+y*y-2*x*y shortcut.
        for kind in subject.KINDS:
            equal = np.full((1, 596), maximum)
            np.testing.assert_array_equal(subject.distance_kernel(equal, equal, kind), [[1.]])

    def test_exponential_underflow_is_zero_and_global_numpy_error_state_is_unchanged(self):
        zeros, far = np.zeros((1, 596)), np.full((1, 596), 1e3)
        before = np.geterr()
        with np.errstate(all="raise"):
            for kind in ("rbf", "laplacian"):
                np.testing.assert_array_equal(subject.distance_kernel(zeros, far, kind), [[0.]])
            rq = subject.distance_kernel(zeros, far, "rational_quadratic")
            self.assertEqual(rq[0, 0], 1 / 1000001)
            self.assertEqual(np.geterr(), dict(divide="raise", over="raise", under="raise", invalid="raise"))
        self.assertEqual(np.geterr(), before)

    def test_complete_json_rules_are_fresh_sealed_and_reject_rehashed_changes(self):
        for kind in subject.KINDS:
            rule = subject.distance_kernel_rule(kind)
            self.assertTrue(subject.verify_distance_kernel_rule(rule, kind))
            reloaded = json.loads(json.dumps(rule, allow_nan=False))
            self.assertTrue(subject.verify_distance_kernel_rule(reloaded, kind))
            digest = reloaded.pop("rule_sha256")
            payload = json.dumps(reloaded, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
            modified = deepcopy(rule)
            modified["input_shapes"][0][1] = 595
            self.assertEqual(subject.distance_kernel_rule(kind)["input_shapes"][0][1], 596)
            modified.pop("rule_sha256")
            modified["rule_sha256"] = hashlib.sha256(json.dumps(modified, ensure_ascii=False,
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            with self.assertRaises(ValueError):
                subject.verify_distance_kernel_rule(modified, kind)
            for changed in (None, {**rule, "extra": True}, {**rule, "gamma": np.nan},
                            {**rule, "schema_version": True}):
                with self.assertRaises(ValueError):
                    subject.verify_distance_kernel_rule(changed, kind)
            other = next(value for value in subject.KINDS if value != kind)
            with self.assertRaises(ValueError):
                subject.verify_distance_kernel_rule(rule, other)


if __name__ == "__main__":
    unittest.main()
