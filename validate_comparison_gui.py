#!/usr/bin/env python3
"""Exercise graph controls, saved results and responsive layout using real Tk."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np

from software import posture_app as module
from src.comparison_charts import MODEL_METRICS, POSTURE_METRICS, model_value, posture_values

ApplicationClass = module.PostureApplication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/comparison_graphs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = sorted((module.PROJECT_ROOT / "artifacts/models").glob("**/model_comparison.json"))
    assert reports, "A saved comparison report is required"
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in reports}
    failures = []
    checks = []
    with tempfile.TemporaryDirectory(prefix="posture-chart-qa-") as temporary:
        module.STATE_ROOT = Path(temporary)
        module.DATA_DIRECTORY = Path(temporary) / "data"
        app = ApplicationClass()
        app.report_callback_exception = lambda *error: failures.append(str(error))
        try:
            app.show_results(); app.update()
            view = app.comparison_view
            assert view.rows, "Bundled/source reports must be found with a separate state root"
            checks.append("saved_report_discovery")
            for split in ("검증", "최종 평가"):
                view.split_choice.set(split)
                view.split_selector.event_generate("<<ComboboxSelected>>"); app.update()
                for metric in MODEL_METRICS:
                    view.metric_choice.set(metric)
                    view.metric_selector.event_generate("<<ComboboxSelected>>"); app.update()
                    expected = [model_value(row, view.split, metric) for row in view.rows]
                    actual = [bar.get_width() for bar in view.canvas.figure.axes[0].patches]
                    np.testing.assert_allclose(actual, [value for value in expected if value is not None])
                view.tabs.select(1); app.update()
                for i, row in enumerate(view.rows):
                    view.model_selector.current(i)
                    view.model_selector.event_generate("<<ComboboxSelected>>"); app.update()
                    for label, key in POSTURE_METRICS.items():
                        view.posture_metric_choice.set(label)
                        view.posture_metric_selector.event_generate("<<ComboboxSelected>>"); app.update()
                        expected = posture_values(view.report, row, view.split, key)[1]
                        np.testing.assert_allclose([bar.get_width() for bar in view.canvas.figure.axes[0].patches],
                                                   [value for value in expected if value is not None])
                view.tabs.select(2); app.update()
                assert view.canvas.figure.axes[0].images
                view.tabs.select(3); app.update()
                assert len(view.table.get_children()) == len(view.rows)
                view.table.selection_set("0"); app.update()
                assert view.model_selector.current() == 0
                view.tabs.select(0); app.update()
            checks.append("both_partitions_all_metrics_all_models_table_selection")
            view.metric_choice.set("전체 정확도 (Accuracy)")
            view.posture_metric_choice.set("모든 지표")
            for width, height in ((980, 760), (850, 700)):
                app.geometry(f"{width}x{height}"); app.update()
                for tab in range(4):
                    view.tabs.select(tab); app.update()
                    app._scroll_canvas.yview_moveto(1.); app.update()
                    assert app._scroll_canvas.yview()[1] == 1., "Chart/footer must be reachable"
                    assert view.winfo_width() <= app.winfo_width(), "No horizontal overflow"
                    if tab < 3:
                        assert view.canvas.get_tk_widget().winfo_width() <= view.winfo_width()
                        path = args.output / f"{('models', 'postures', 'confusion')[tab]}_{width}.png"
                        with mock.patch("software.comparison_view.filedialog.asksaveasfilename", return_value=str(path)):
                            view.save_chart()
                        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
                    app._scroll_canvas.yview_moveto(0.); app.update()
            checks.append("980x760_850x700_layout_and_png_export")
            # A different/empty/broken report must never retain the old chart.
            large = json.loads(reports[0].read_text())
            large["models"] = [{**row, "window_seconds": seconds} for seconds in (3, 5, 20) for row in large["models"]]
            large_path = Path(temporary) / "large.json"
            large_path.write_text(json.dumps(large))
            view.report_choice.set(str(large_path)); view.load_report()
            view.tabs.select(0); app.update()
            assert len(view.canvas.figure.axes[0].patches) == len(large["models"])
            for status, content in (
                ("running", '{"status":"running","models":[]}'), ("invalid", "{"),
                ("invalid_selection", '{"models":[{}],"selected":"bad"}'),
                ("invalid_metrics", '{"models":[{"validation":3}]}'),
                ("invalid_posture", '{"models":[{"validation":{"per_class":{"정자세":4}}}]}'),
                ("invalid_labels", '{"models":[{}],"labels":[{}]}'),
            ):
                path = Path(temporary) / f"{status}.json"; path.write_text(content)
                view.report_choice.set(str(path))
                view.report_selector.event_generate("<<ComboboxSelected>>"); app.update()
                assert not view.rows and not view.canvas.figure.axes[0].patches
                assert str(view.export_button.cget("state")) == "disabled"
                view.tabs.select(3); app.update()
                assert not view.table.get_children()
                view.tabs.select(0); app.update()
            checks.append("12_model_run_and_empty_invalid_report_reset")
            for _ in range(3):
                app.show_management(); app.update()
                app.show_results(); app.update()
            checks.append("repeated_navigation_and_canvas_cleanup")
        finally:
            app.destroy()
    assert not failures, failures
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    result = {"real_tk": True, "checks": checks, "callback_errors": failures, "source_reports_unchanged": True}
    (args.output / "gui_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
