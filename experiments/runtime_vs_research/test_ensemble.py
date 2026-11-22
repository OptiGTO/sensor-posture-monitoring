import unittest

from experiments.runtime_vs_research.ensemble import select


def summary(scores):
    return {"development_macro_f1": {a: {"primary_seed_mean": v} for a, v in scores.items()}}


class SelectTest(unittest.TestCase):
    def test_best_per_family_and_strict_top_two(self):
        s = select(summary({"svm_standard": 0.67, "svm_ordinary": 0.69, "et_standard": 0.70, "et_ordinary": 0.72}))
        self.assertEqual(s["primary"], ["et_ordinary", "svm_ordinary"])
        self.assertEqual(s["secondary_strict_top_two"], ["et_ordinary", "et_standard"])

    def test_family_order_follows_score(self):
        s = select(summary({"svm_standard": 0.75, "svm_ordinary": 0.60, "et_standard": 0.70, "et_ordinary": 0.65}))
        self.assertEqual(s["primary"], ["svm_standard", "et_standard"])
        self.assertEqual(s["secondary_strict_top_two"], ["svm_standard", "et_standard"])


if __name__ == "__main__":
    unittest.main()
