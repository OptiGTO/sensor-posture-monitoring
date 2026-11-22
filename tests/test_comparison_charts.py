"""Keep displayed scores, class ordering and missing-data semantics faithful."""

import copy
import unittest

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.comparison_charts import (
    MODEL_METRICS, confusion_chart, matrix_values, model_chart, model_value,
    number, posture_chart, posture_values,
)


def sample_report():
    return {"labels": ["왼쪽 기울임", "정자세"], "models": [{
        "name": "RBF SVM", "window_seconds": 3, "selected": True,
        "model_size_bytes": 2048, "inference_ms_median": 1.25,
        "validation": {"accuracy": .8, "macro_f1": .7, "worst_class_f1": .6,
                       "per_class": {"정자세": {"recall": 0, "precision": .2, "f1": 0, "support": 2},
                                     "왼쪽 기울임": {"recall": .75, "precision": .6, "f1": .67, "support": 4}},
                       "confusion_matrix": [[3, 1], [2, 0]]},
        "final_evaluation": {"accuracy": .25, "per_class": {"왼쪽 기울임": {"recall": .25, "support": 4}},
                             "confusion_matrix": [[1, 3], [0, 0]]},
    }]}


class ComparisonChartTests(unittest.TestCase):
    def test_model_scores_use_requested_partition_and_units(self):
        report = sample_report()
        before = copy.deepcopy(report)
        row = report["models"][0]
        self.assertEqual(model_value(row, "validation", "전체 정확도 (Accuracy)"), 80)
        self.assertEqual(model_value(row, "final_evaluation", "전체 정확도 (Accuracy)"), 25)
        self.assertEqual(model_value(row, "final_evaluation", "모델 크기"), 2)
        self.assertEqual(model_value(row, "validation", "추론 시간 (중앙값)"), 1.25)
        self.assertIsNone(model_value(row, "final_evaluation", "Macro F1"))
        figure = model_chart(report, "final_evaluation", "전체 정확도 (Accuracy)")
        self.assertEqual([bar.get_width() for bar in figure.axes[0].patches], [25])
        self.assertIn("3초", figure.axes[0].get_yticklabels()[0].get_text())
        self.assertEqual(report, before)

    def test_class_order_matches_report_and_recall_is_not_precision(self):
        report = sample_report()
        row = report["models"][0]
        labels, values, support = posture_values(report, row, "validation", "recall")
        self.assertEqual(labels, ["왼쪽 기울임", "정자세"])
        self.assertEqual(values, [75, 0])
        self.assertEqual(support, [4, 2])
        figure = posture_chart(report, row, "validation", "자세별 정답률 (Recall)")
        self.assertEqual([bar.get_width() for bar in figure.axes[0].patches], [75, 0])

    def test_missing_values_do_not_become_zero_bars(self):
        for invalid in (None, True, "bad", float("nan"), float("inf"), -1, 1.1):
            self.assertIsNone(number(invalid, score=True))
        self.assertEqual(number(0, score=True), 0)
        report = sample_report()
        row = report["models"][0]
        self.assertEqual(posture_values(report, row, "final_evaluation", "recall")[1], [25, None])
        row["final_evaluation"]["per_class"]["왼쪽 기울임"]["support"] = 0
        self.assertEqual(posture_values(report, row, "final_evaluation", "recall")[1], [None, None])
        figure = model_chart(report, "final_evaluation", "Macro F1")
        self.assertFalse(figure.axes[0].patches)
        self.assertIn("미기록", [text.get_text() for text in figure.axes[0].texts])

    def test_old_per_class_f1_remains_available_without_inventing_recall(self):
        report = {"labels": ["정자세"]}
        row = {"validation": {"per_class_f1": {"정자세": .5}}}
        self.assertEqual(posture_values(report, row, "validation", "f1")[1], [50])
        self.assertEqual(posture_values(report, row, "validation", "recall")[1], [None])

    def test_confusion_percentages_are_by_actual_class_and_empty_rows_masked(self):
        report = sample_report()
        figure = confusion_chart(report, report["models"][0], "validation")
        np.testing.assert_allclose(figure.axes[0].images[0].get_array(), [[75, 25], [100, 0]])
        self.assertEqual(figure.axes[0].get_xlabel(), "예측 자세")
        self.assertEqual(figure.axes[0].get_ylabel(), "실제 자세")
        empty_row = confusion_chart(report, report["models"][0], "final_evaluation")
        self.assertTrue(empty_row.axes[0].images[0].get_array().mask[1].all())
        for matrix in ([[1]], [[1, -1], [0, 2]], [[.5, 1], [0, 2]], [[float("nan"), 1], [0, 2]]):
            self.assertIsNone(matrix_values(report, {"confusion_matrix": matrix})[1])

    def test_all_views_render_for_empty_partial_and_complete_reports(self):
        report = sample_report()
        figures = [model_chart(report, "validation", metric) for metric in MODEL_METRICS]
        figures += [posture_chart(report, report["models"][0], "validation"),
                    confusion_chart(report, report["models"][0], "validation"),
                    model_chart({}, "validation", "Macro F1"), posture_chart({}, None, "validation"),
                    confusion_chart({}, None, "validation"), confusion_chart({}, {}, "validation")]
        for figure in figures:
            FigureCanvasAgg(figure).draw()


if __name__ == "__main__":
    unittest.main()
