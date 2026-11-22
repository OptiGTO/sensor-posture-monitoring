"""Checks for the label-review extraction helpers; no data/ or model access."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.label_review import extract_all_wrong as ex


class SimilarityHintTests(unittest.TestCase):
    def setUp(self):
        base = np.zeros(16)
        forward, backward = base.copy(), base.copy()
        forward[:4], backward[-4:] = 1.0, 1.0
        self.means = {"a/r": forward + 0.01 * np.arange(16), "a/f1": forward, "a/b1": backward, "b/f": backward}
        self.meta = {"a/r": ("a", 2), "a/f1": ("a", 1), "a/b1": ("a", 2), "b/f": ("b", 1)}

    def test_closer_to_predicted_excludes_self_and_other_people(self):
        hint = ex.similarity_hint("a/r", "a", 2, 1, self.means, self.meta)
        self.assertEqual(hint["hint"], "closer_to_own_predicted")
        self.assertEqual((hint["own_truth_records"], hint["own_predicted_records"]), (1, 1))
        self.assertEqual(hint["nearest_own_class_id"], 1)

    def test_missing_comparison(self):
        hint = ex.similarity_hint("a/r", "a", 2, 5, self.means, self.meta)
        self.assertEqual(hint["hint"], "no_comparison")

    def test_centered_cosine_ignores_level(self):
        pattern = np.arange(16, dtype=float)
        self.assertAlmostEqual(ex.centered_cosine(pattern, pattern + 5.0), 1.0)


class CrossModelTests(unittest.TestCase):
    def test_counts_all_wrong_across_every_evaluating_split(self):
        parts = [{"record_ids": np.array(["r", "r", "s"]), "truth": np.array([0, 0, 1])},
                 {"record_ids": np.array(["r", "r"]), "truth": np.array([0, 0])}]
        loaded = {"always_wrong": [np.array([2, 2, 1]), np.array([2, 2])],
                  "wrong_other": [np.array([3, 3, 1]), np.array([3, 3])],
                  "one_split_right": [np.array([2, 2, 1]), np.array([0, 2])]}
        result, wrong, same, pattern = ex.cross_model("r", parts, 2, loaded)
        self.assertEqual((wrong, same, pattern), (2, 1, "most_models_wrong"))
        self.assertEqual(result["one_split_right"], (0.25, 2))


class OutputGuardTests(unittest.TestCase):
    def test_rejects_output_outside_results_and_existing_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "research/results/existing").mkdir(parents=True)
            with self.assertRaises(ValueError):
                ex.extract(root, root / "elsewhere", make_plots=False)
            with self.assertRaises(ValueError):
                ex.extract(root, root / "research/results/existing", make_plots=False)


if __name__ == "__main__":
    unittest.main()
