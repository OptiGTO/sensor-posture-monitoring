import unittest
import numpy as np
from experiments.consensus_filter.analyze import f1_from_cm, paired_bootstrap, markdown_to_html
from src.model_comparison import scores


class AnalysisTests(unittest.TestCase):
    def test_matrix_macro_f1_matches_project_metric(self):
        y = np.tile(np.arange(7), 5)
        pred = (y + np.arange(len(y)) % 3) % 7
        cm = np.bincount(7*y+pred, minlength=49).reshape(7, 7)
        self.assertAlmostEqual(f1_from_cm(cm), scores(y, pred)["macro_f1"])

    def test_paired_bootstrap_has_zero_delta_for_identical_predictions(self):
        y = np.tile(np.arange(7), 8)
        groups = np.repeat(np.arange(8), 7)
        r = paired_bootstrap(y, y, y, groups, draws=100)
        self.assertEqual([r[k] for k in ("delta", "low", "high")], [0., 0., 0.])

    def test_direction_and_group_order_invariance(self):
        y = np.tile(np.arange(7), 8)
        groups = np.repeat(np.arange(8), 7)
        bad = (y + 1) % 7
        r = paired_bootstrap(y, bad, y, groups, draws=100)
        self.assertEqual(r["delta"], 1.)
        self.assertEqual(r["low"], 1.)
        self.assertEqual(r, paired_bootstrap(y[::-1], bad[::-1], y[::-1], groups[::-1], draws=100))

    def test_report_renderer_keeps_tables_and_escapes_text(self):
        h = markdown_to_html("# Test\n\n| model | F1 |\n| --- | ---: |\n| x | 0.5 |\n\n<script>\n")
        self.assertIn("<table>", h)
        self.assertIn("<td>0.5</td>", h)
        self.assertIn("&lt;script&gt;", h)


if __name__ == "__main__":
    unittest.main()
