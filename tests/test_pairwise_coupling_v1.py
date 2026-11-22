"""Pure synthetic numerical contracts for direct seven-class pair coupling."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import io
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from src import pairwise_coupling_v1 as subject


def consistent(probability):
    return np.asarray([probability[i] / (probability[i] + probability[j]) for i, j in subject.PAIR_INDICES])


def permute(forward, order):
    matrix = np.zeros((7, 7), dtype=np.float64)
    for value, (i, j) in zip(forward, subject.PAIR_INDICES):
        matrix[i, j], matrix[j, i] = value, 1. - value
    return np.asarray([matrix[order[i], order[j]] for i, j in subject.PAIR_INDICES])


class PairwiseCouplingTests(unittest.TestCase):
    def test_uniform_is_exact_equal_and_tie_canonicalization_is_only_exact_half(self):
        result, audit = subject.couple_pairwise(np.full((3, 21), .5), return_diagnostics=True)
        expected = np.full((3, 7), 1 / 7.)
        self.assertEqual(result.tobytes(), expected.tobytes())
        np.testing.assert_array_equal(result.argmax(1), np.zeros(3, dtype=int))
        self.assertEqual(audit["exact_uniform_rows"], 3)
        near = np.full((1, 21), .5)
        near[0, 0] = np.nextafter(.5, 1.)
        _, audit = subject.couple_pairwise(near, return_diagnostics=True)
        self.assertEqual(audit["exact_uniform_rows"], 0)

    def test_known_consistent_simplex_restored_including_high_confidence_without_twosevenths_cap(self):
        rng = np.random.default_rng(1902)
        probabilities = np.vstack([np.array([.7] + [.05] * 6), rng.dirichlet(np.ones(7), 30)])
        forward = np.stack([consistent(p) for p in probabilities])
        coupled, audit = subject.couple_pairwise(forward, return_diagnostics=True)
        np.testing.assert_allclose(coupled, probabilities, rtol=0, atol=1e-12)
        self.assertGreater(coupled[0, 0], .45)
        self.assertLess(audit["maximum_pairwise_squared_residual"], 1e-24)
        self.assertEqual(audit["clipped_pair_count"], 0)

    def test_singular_Q_is_valid_because_augmented_KKT_remains_well_conditioned(self):
        values = consistent(np.array([.7] + [.05] * 6))
        q = subject._q_matrix(values)
        self.assertGreater(np.linalg.cond(q), 1e12)
        result, audit = subject.couple_pairwise(values[None, :], return_diagnostics=True)
        self.assertLess(audit["maximum_condition_KKT"], 100)
        self.assertLess(audit["maximum_backward_residual"], 1e-12)
        self.assertTrue(np.all(result > 0))

    def test_cycles_allow_positive_pair_objective_with_small_optimality_residual(self):
        cyclic = np.asarray([1. if (j - i) % 7 in (1, 2, 3) else 0. for i, j in subject.PAIR_INDICES])
        result, audit = subject.couple_pairwise(cyclic[None, :], return_diagnostics=True)
        np.testing.assert_allclose(result, np.full((1, 7), 1 / 7.), atol=1e-12, rtol=0)
        self.assertGreater(audit["maximum_pairwise_squared_residual"], .4)
        self.assertLess(audit["maximum_stationarity_residual"], 1e-12)
        self.assertEqual(audit["clipped_pair_count"], 21)
        self.assertEqual(audit["exact_uniform_rows"], 0)

    def test_boundary_tournaments_and_extremely_skewed_inputs_remain_finite_strictly_positive(self):
        rng = np.random.default_rng(1903)
        values = np.vstack([np.zeros(21), np.ones(21), rng.integers(0, 2, size=(100, 21)),
            rng.beta(.01, .01, size=(100, 21)), consistent(10. ** np.linspace(0, -120, 7))])
        original = values.copy()
        result, audit = subject.couple_pairwise(values, return_diagnostics=True)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all((result > 0) & (result < 1)))
        np.testing.assert_allclose(result.sum(1), 1., atol=1e-12, rtol=0)
        self.assertGreater(result[1, 0], .99)
        self.assertGreater(result[0, 6], .99)
        self.assertLess(audit["maximum_condition_KKT"], 100.)
        self.assertLess(audit["maximum_backward_residual"], 1e-12)
        self.assertLess(audit["maximum_stationarity_residual"], 1e-12)
        np.testing.assert_array_equal(values, original)

    def test_q_matches_independent_sum_of_pairwise_squared_residuals(self):
        rng = np.random.default_rng(1904)
        for values in rng.uniform(size=(10, 21)):
            q = subject._q_matrix(values)
            b = np.zeros((21, 7))
            for row, (value, (i, j)) in enumerate(zip(values, subject.PAIR_INDICES)):
                b[row, i], b[row, j] = 1 - value, -value
            np.testing.assert_allclose(q, b.T @ b, atol=1e-15, rtol=0)
            p = subject.couple_pairwise(values[None, :])[0]
            gradient = q @ p
            np.testing.assert_allclose(gradient, np.full(7, p @ gradient), atol=1e-12, rtol=0)
            for perturbation in rng.normal(size=(10, 7)):
                perturbation -= perturbation.mean()
                alternative = p + perturbation * .01
                self.assertLessEqual(float(np.sum((b @ p) ** 2)), float(np.sum((b @ alternative) ** 2)) + 1e-14)

    def test_class_permutations_are_equivariant_with_declared_numerical_tolerance(self):
        rng = np.random.default_rng(1905)
        values = np.vstack([rng.uniform(size=(20, 21)), rng.integers(0, 2, size=(10, 21)), consistent(np.array([.7] + [.05] * 6))])
        original = subject.couple_pairwise(values)
        for _ in range(12):
            order = rng.permutation(7)
            mapped = np.stack([permute(row, order) for row in values])
            np.testing.assert_allclose(subject.couple_pairwise(mapped), original[:, order], atol=1e-12, rtol=0)

    def test_row_order_batch_size_memory_layout_and_rule_reload_preserve_bytes(self):
        values = np.random.default_rng(1906).uniform(size=(37, 21))
        expected = subject.couple_pairwise(values)
        self.assertTrue(expected.flags.c_contiguous)
        self.assertEqual(expected.dtype, np.dtype(np.float64))
        self.assertEqual(expected.tobytes(), subject.couple_pairwise(values[::-1])[::-1].tobytes())
        self.assertEqual(expected.tobytes(), np.concatenate([subject.couple_pairwise(values[:3]), subject.couple_pairwise(values[3:])]).tobytes())
        self.assertEqual(expected.tobytes(), np.concatenate([subject.couple_pairwise(row[None, :]) for row in values]).tobytes())
        self.assertEqual(expected.tobytes(), subject.couple_pairwise(np.asfortranarray(values)).tobytes())
        stream = io.BytesIO()
        joblib.dump({"rule": subject.pairwise_coupling_rule(), "input": values}, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        self.assertTrue(subject.verify_pairwise_coupling_rule(restored["rule"]))
        self.assertEqual(expected.tobytes(), subject.couple_pairwise(restored["input"]).tobytes())

    def test_configuration_is_immutable_rules_are_fresh_and_diagnostics_are_return_only(self):
        with self.assertRaises(FrozenInstanceError):
            subject._CONFIG.epsilon = .1
        first = subject.pairwise_coupling_rule()
        first["pair_indices"][0][0] = 1
        self.assertNotEqual(first, subject.pairwise_coupling_rule())
        with self.assertRaises(ValueError):
            subject.verify_pairwise_coupling_rule(first)
        self.assertEqual(subject.pairwise_coupling_rule()["condition_limit"], 1e8)
        before = deepcopy(vars(subject._CONFIG))
        values = np.full((2, 21), .5)
        p, diagnostic = subject.couple_pairwise(values, return_diagnostics=True)
        diagnostic["rows"][0]["condition_KKT"] = -1
        _, fresh = subject.couple_pairwise(values, return_diagnostics=True)
        self.assertGreater(fresh["rows"][0]["condition_KKT"], 0)
        self.assertEqual(before, vars(subject._CONFIG))
        json.dumps(fresh, allow_nan=False)

    def test_bad_shapes_types_ranges_and_nonfinite_probabilities_fail(self):
        for values in (np.ones(21), np.ones((3, 20)), np.ones((0, 21)), np.ones((2, 7, 7)),
                np.full((2, 21), -1e-15), np.full((2, 21), 1. + 1e-15),
                np.full((2, 21), np.nan), np.full((2, 21), np.inf),
                np.ones((2, 21), dtype=bool), np.ones((2, 21), dtype=complex), np.full((2, 21), "0.5")):
            with self.subTest(shape=values.shape, dtype=values.dtype), self.assertRaises(ValueError):
                subject.couple_pairwise(values)
        with self.assertRaises(ValueError):
            subject.couple_pairwise(np.full((1, 21), .5), return_diagnostics=1)

    def test_solver_errors_nonpositive_outputs_and_condition_fail_closed_without_uniform_fallback(self):
        values = np.full((1, 21), .5)
        wrong = np.r_[np.array([.2, .1, .14, .14, .14, .14, .14]), 0.]
        negative = np.r_[np.array([-.1, .2, .2, .2, .2, .2, .1]), 0.]
        zero = np.r_[np.array([0., .2, .2, .2, .2, .1, .1]), 0.]
        for solution in (wrong, negative, zero, np.r_[np.full(7, .1), 0.], np.full(8, np.nan), np.ones(7)):
            with patch.object(subject.np.linalg, "solve", return_value=solution), self.assertRaises(ValueError):
                subject.couple_pairwise(values)
        with patch.object(subject.np.linalg, "solve", side_effect=np.linalg.LinAlgError("synthetic singular")), self.assertRaisesRegex(ValueError, "no fallback"):
            subject.couple_pairwise(values)
        for condition in (1e8 + 1, np.inf, np.nan):
            with patch.object(subject.np.linalg, "cond", return_value=condition), self.assertRaisesRegex(ValueError, "condition"):
                subject.couple_pairwise(values)

    def test_overflowed_scaled_residual_cannot_hide_an_invalid_solver_result(self):
        solution = np.r_[np.full(7, 1 / 7.), 1e308]
        with np.errstate(over="ignore", invalid="ignore"), patch.object(subject.np.linalg, "solve", return_value=solution):
            with self.assertRaisesRegex(ValueError, "not finite"):
                subject.couple_pairwise(np.full((1, 21), .5))


if __name__ == "__main__":
    unittest.main()
