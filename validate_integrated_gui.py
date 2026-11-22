#!/usr/bin/env python3
"""Exercise real Tk screens and nonmodal popup while CSV measurement continues."""
import json
import argparse
import hashlib
from pathlib import Path
import tempfile
import time
from unittest import mock

import numpy as np

from software import posture_app as module
from src.personal_guidance import GuidanceEvent

PRODUCT = module.PRODUCT
ApplicationClass = module.PostureApplication


def check_registered_training_methods(app, layout):
    """Check the actual source/bundled registry without training or publishing fixtures."""
    from src.app_services import TrainingWizardState
    from src.model_catalog import MODEL_NAMES
    from src.training_methods import REGISTRY_PATH, available_training_methods

    registry_bytes = REGISTRY_PATH.read_bytes() if REGISTRY_PATH.is_file() else None
    methods = available_training_methods()
    expected_ids = [method['id'] for method in methods]
    previous_state = app.training_wizard_state
    previous_step = previous_state.current_step
    previous_records = app.selected_records
    previous_retrospective = app.personal_retrospective.get()
    checks = []
    try:
        app.selected_records = []
        app.personal_retrospective.set(False)
        for method in methods:
            assert 3 in method['window_seconds'], f"{method['id']}: 3초 지원 근거가 없습니다."
            for family in method['families']:
                app.training_wizard_state = TrainingWizardState(models=(family,), windows=(3,))
                app.show_training_wizard_setup()
                assert list(app.wizard_training_method_ids.values()) == expected_ids
                assert set(app.wizard_model_vars) == set(MODEL_NAMES)
                app.wizard_training_method.set(method['label'])
                app._refresh_training_method_hint()
                config = app._remember_experiment_setup()
                assert config['training_method'] == method['id']
                assert config['models'] == [family] and config['windows'] == [3]
                assert config['training_method_evidence'] == method.get('evidence')
                layout(f"training_method_{method['id']}_{family}_3s")
                checks.append({'training_method': config['training_method'],
                    'training_method_label': config['training_method_label'],
                    'models': config['models'], 'windows': config['windows'],
                    'configuration_recorded': True})
        assert (REGISTRY_PATH.read_bytes() if REGISTRY_PATH.is_file() else None) == registry_bytes, \
            '검증 중 학습 방식 레지스트리가 변경되었습니다.'
    finally:
        app.training_wizard_state = previous_state
        app.selected_records = previous_records
        app.show_training_wizard_setup()
        previous_state.current_step = previous_step
        app.personal_retrospective.set(previous_retrospective)
    return {'registry_source': 'actual', 'registry_path': str(REGISTRY_PATH),
        'registry_sha256': hashlib.sha256(registry_bytes).hexdigest() if registry_bytes is not None else None,
        'registered_method_ids': expected_ids,
        'registered_methods': [{key: method[key] for key in ('id', 'label', 'families', 'window_seconds')}
                               for method in methods],
        'validated_3s_combinations': checks, 'training_started': False}


def check_live_serial(port, duration, standard_root, output):
    """Exercise the actual Tk/Serial/recorder pipeline in isolated storage.

    This is an explicit hardware check; a loaded sensor is required to verify
    predictions. Original standard recordings are read through a symlink only.
    """
    if duration < 6 or not np.isfinite(duration):
        raise ValueError("실제 센서 검증은 6초 이상 실행하세요.")
    previous_state, previous_data = module.STATE_ROOT, module.DATA_DIRECTORY
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="posture-live-qa-") as temporary:
        module.STATE_ROOT = Path(temporary)
        module.DATA_DIRECTORY = Path(temporary) / "data"
        module.DATA_DIRECTORY.mkdir()
        if standard_root is not None:
            standard_root = standard_root.resolve(strict=True)
            (module.DATA_DIRECTORY / "standard").symlink_to(standard_root, target_is_directory=True)
        app = None
        try:
            app = ApplicationClass()
            if not app.models:
                raise RuntimeError("검증 가능한 승인 모델이 없습니다.")
            model = min(app.models, key=lambda item: item.window.default_seconds)
            app.select_model(model)
            app.mode.set("measurement")
            app.port.set(port)
            app.input_path.set("")
            app.show_mode_select()
            app.start_measurement()
            if app._recorder is None or app._worker is None:
                raise RuntimeError("센서 측정이 시작되지 않았습니다.")
            started = time.monotonic()
            states = set()
            while time.monotonic() - started < duration and not app._session_finished:
                app.update()
                states.add(app.state_text.get())
                time.sleep(.02)
            app._stop.set()
            deadline = time.monotonic() + 10
            while app._worker.is_alive() and time.monotonic() < deadline:
                app.update()
                time.sleep(.02)
            if app._worker.is_alive():
                raise RuntimeError("센서 측정 종료 시간이 초과되었습니다.")
            app.update()
            predictions = [json.loads(line) for line in app._recorder.prediction_path.read_text().splitlines()]
            manifest = json.loads(app._last_manifest.read_text()) if app._last_manifest else None
            result = {
                "real_tk": True, "real_serial": True, "port": port,
                "duration_s": duration, "standard_root": str(standard_root) if standard_root else None,
                "states": sorted(states), "rows": app._recorder.rows,
                "predictions": len(predictions), "error": app._session_error,
                "manifest": manifest,
                "passed": app._recorder.rows > 0 and bool(predictions) and app._session_error is None,
            }
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            if not result["passed"]:
                raise RuntimeError(f"실제 센서 검증 실패: {output} (센서 압력과 오류를 확인하세요.)")
            print(json.dumps({key: value for key, value in result.items() if key != "manifest"}, ensure_ascii=False))
        finally:
            if app is not None:
                app._stop.set()
                if app._worker:
                    app._worker.join(timeout=5)
                app.destroy()
            module.STATE_ROOT, module.DATA_DIRECTORY = previous_state, previous_data


def check_loso_selection_controls(app, layout, work_root):
    """Real Tk: LOSO is the default selector, can be switched, and results open on the LOSO split."""
    import pandas as pd
    import tkinter as tk
    from unittest import mock as _mock
    from software.comparison_view import ComparisonView
    from src.app_services import TrainingWizardState
    from src.experiment_workflow import SELECTION_METRIC_LABELS
    from src.model_comparison import train_comparison
    from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS

    previous_state = app.training_wizard_state
    try:
        app.training_wizard_state = TrainingWizardState(models=("rbf_svm",), windows=(3,))
        app.show_training_wizard_setup()
        layout("training_selection_metric_default")
        assert app.wizard_selection_metric.get() == SELECTION_METRIC_LABELS["loso"]
        default_config = app._remember_experiment_setup()
        assert default_config["selection_metric"] == "loso"
        app.wizard_selection_metric.set(SELECTION_METRIC_LABELS["validation"])
        switched = app._remember_experiment_setup()
        assert switched["selection_metric"] == "validation"
    finally:
        app.training_wizard_state = previous_state
        app.show_training_wizard_setup()

    data = Path(work_root) / "loso_data"
    rng = np.random.default_rng(7)
    for subject in range(5):
        for i, label in enumerate(CLASS_NAMES):
            values = rng.uniform(10, 15, (37, len(SENSOR_COLUMNS))) + i * 15 + subject * 3
            frame = pd.DataFrame(values, columns=SENSOR_COLUMNS)
            frame.insert(0, "label", label); frame.insert(0, "elapsed_s", np.arange(37) / 12)
            (data / f"s{subject}").mkdir(parents=True, exist_ok=True)
            frame.to_csv(data / f"s{subject}" / f"p{i}.csv", index=False)
    grid = {"rbf_svm": [{"C": 1.0, "gamma": "scale"}],
            "extra_trees": [{"n_estimators": 12, "min_samples_leaf": 1, "max_features": "sqrt"}]}
    with _mock.patch("src.model_comparison.MODEL_CANDIDATES", grid):
        report = train_comparison(data, Path(work_root) / "loso_run", (3,), 41, ("rbf_svm", "extra_trees"),
                                  progress=lambda _message: None)
    path = Path(work_root) / "loso_run/artifacts/model_comparison.json"
    window = tk.Toplevel(app)
    try:
        view = ComparisonView(window, [path]); view.pack(fill="both", expand=True)
        window.update()
        assert view.split_choice.get() == "LOSO (기본)" and view.split == "loso"
        assert "LOSO" in view.summary.get()
        assert view.canvas is not None
        view.tabs.select(3); view.refresh(); window.update()
        first = view.table.item(view.table.get_children()[0], "values")
        assert first[2] != "미기록", "LOSO Macro F1 must be shown in the detail table"
        view.split_choice.set("검증"); view.refresh(); window.update()
        return {"default_selector": "loso", "switch_to_validation": True, "report_selection_metric": report["selection_metric"],
                "selected": report["selected"]["family"], "comparison_default_split": "loso",
                "detail_loso_macro_f1": first[2], "training_started_from_gui": False}
    finally:
        window.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=module.PROJECT_ROOT / "results" / "integrated_validation" / "gui_validation.json")
    parser.add_argument("--serial-port", help="실제 아두이노로 격리 검증 (auto 또는 포트 경로)")
    parser.add_argument("--duration", type=float, default=12., help="실제 센서 검증 시간 (초)")
    parser.add_argument("--standard-root", type=Path, help="읽기 전용으로 사용할 빈 좌석 기준 폴더")
    args = parser.parse_args(argv)
    if args.serial_port:
        return check_live_serial(args.serial_port, args.duration, args.standard_root, args.output)
    with tempfile.TemporaryDirectory(prefix="posture-gui-qa-") as temporary:
        module.STATE_ROOT = Path(temporary)
        module.DATA_DIRECTORY = Path(temporary) / "data"
        app = ApplicationClass()
        app.update()
        results = {
            "real_tk": True,
            "product_variant": PRODUCT.variant,
            "window_title": app.title(),
            "default_mode": app.alert_mode.get(),
            "screens": {},
        }

        def layout(name):
            app.update()
            buttons = []
            def walk(widget):
                for child in widget.winfo_children():
                    if widget.winfo_class() == "TNotebook" and str(child) != widget.select():
                        continue
                    if child.winfo_class() in {"Button", "TButton"}:
                        buttons.append({"text": child.cget("text"), "visible": bool(child.winfo_ismapped()),
                                        "bottom": child.winfo_rooty() - app.winfo_rooty() + child.winfo_height()})
                    walk(child)
            walk(app)
            clipped = [button for button in buttons if not button["visible"]]
            below = [button for button in buttons if button["bottom"] > app.winfo_height()]
            if below:
                canvas = app._scroll_canvas
                canvas.yview_moveto(1.); app.update()
                assert canvas.yview()[1] == 1., "Bottom controls must be reachable by scrolling"
                canvas.yview_moveto(0.)
            results["screens"][name] = {"width": app.winfo_width(), "height": app.winfo_height(), "clipped_buttons": clipped, "scroll_required": bool(below)}

        layout("model_selection")
        app.select_model(app.models[0]); layout("measurement_setup")
        modes = ("labeled", "measurement", "calibration")
        expected_radios = ["라벨 데이터 수집", "측정 전용", "정자세 등록"]
        for mode in modes:
            app.mode.set(mode)
            app.show_mode_select()
            layout(f"setup_{mode}")
            widgets = []
            def collect(widget):
                widgets.append(widget)
                for child in widget.winfo_children():
                    collect(child)
            collect(app)
            radios = [widget.cget("text") for widget in widgets if widget.winfo_class() == "TRadiobutton"]
            assert radios == expected_radios, (radios, expected_radios)
            texts = {str(widget.cget("text")) for widget in widgets
                     if widget.winfo_class() in {"TCheckbutton", "TButton", "TLabelframe"}}
            hidden = {"등록된 개인 기준으로 2단계 판별 사용 (실험용 · 3초)", "개인 모델 평가 보기",
                      "개인 기준 2단계 판별 (선택 · 실시간 센서)"}
            assert not texts & hidden, texts & hidden
            if mode == "measurement":
                assert "등록된 개인 기준 사용 (선택 · 실시간 센서)" not in texts
                results["personal_hierarchy_hidden"] = True
            if mode == "calibration":
                assert not any(widget.winfo_ismapped() and widget.winfo_class() == "TButton"
                               and widget.cget("text") == "파일 선택" for widget in widgets)
        app.guidance_user.set("gui-registration-test")
        app.input_path.set("ignored-during-registration.csv")
        def registration_frames(*args, **kwargs):
            controller = app._guidance_session
            controller.command("baseline")
            for step in range(200):
                if step == 46:
                    controller.command("register")
                time.sleep(.005)
                yield step / 10, np.full(16, 1.2 if step < 31 else 2.2)
        with mock.patch.object(module, "serial_frames", side_effect=registration_frames):
            app.start_measurement()
            layout("registration_live")
            registration_monitor = app.signal_monitor
            deadline = time.monotonic() + 5
            while not app._session_finished and time.monotonic() < deadline:
                app.update(); time.sleep(.01)
            assert app._session_finished and app._calibration_result.is_file()
            assert app._recorder is None
            monitor_samples = app._signal_buffer.snapshot().total_samples
            assert monitor_samples > 0 and not registration_monitor.winfo_exists()
            layout("registration_complete")
            results["registration"] = {"profile_saved": True, "measurement_recorder_created": False,
                                       "signal_monitor_samples": monitor_samples}
        app.guidance_user.set("")
        app.input_path.set("")
        app.mode.set("measurement")
        # Reproduce the split between legacy data and product-specific recordings.
        # All fixtures stay in the isolated QA directory, never canonical data/.
        import csv
        from src.app_services import CSV_COLUMNS, DatasetService
        original_service = app.dataset_service
        legacy_data = Path(temporary) / "legacy/data"
        for index in range(12):
            directory = module.DATA_DIRECTORY if index == 0 else legacy_data
            path = directory / f"qa-person-{index}" / "sample.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(CSV_COLUMNS)
                writer.writerows([t, "정자세", *[value + t for value in range(16)]] for t in range(5))
        app.dataset_service = DatasetService(module.STATE_ROOT, data_directory=module.DATA_DIRECTORY,
                                             additional_data_directories=(legacy_data,))
        app.start_training_wizard()
        app._wizard_to_dataset()
        app.update()
        assert len(app.visible_records) == 12
        assert "현재 표시 12개" in app.dataset_list_summary.get()
        for width, height in ((980, 760), (850, 700)):
            app.geometry(f"{width}x{height}"); app.update()
            assert app.dataset_tree.winfo_ismapped()
            app.dataset_tree.see("11"); app.update()
            assert app.dataset_tree.bbox("11"), "Last dataset must be reachable"
        app.select_visible_datasets()
        assert len(app.selected_records) == 12
        app.filter_subject.set("qa-person-11")
        app.refresh_dataset_table()
        assert len(app.visible_records) == 1 and len(app.selected_records) == 12
        app.filter_subject.set("missing-subject")
        app.refresh_dataset_table()
        assert not app.visible_records and "검색 조건" in app.dataset_list_summary.get()
        app.filter_subject.set(""); app.refresh_dataset_table()
        app.dataset_tree.focus("0"); app.dataset_tree.selection_set("0")
        from software import distribution_view
        with mock.patch.object(distribution_view.filedialog, "asksaveasfilename",
                               return_value=str(Path(temporary) / "distribution.png")), \
             mock.patch.object(distribution_view.messagebox, "showinfo"):
            window = app.show_record_distribution()
            assert window is not None and window.canvas is not None
            window.update()
            assert window.canvas.get_tk_widget().winfo_ismapped()
            window.start.set("1"); window.end.set("3"); window.refresh()
            window.save()
            assert (Path(temporary) / "distribution.png").stat().st_size > 0
            results["distribution_map"] = {"opened": True, "axes": len(window.figure.axes),
                "summary": window.summary.get(), "interval_refresh": True, "png_saved": True}
            window.destroy()
        app.clear_dataset_selection()
        app.dataset_tree.focus("0"); app.toggle_dataset()
        assert len(app.selected_records) == 1
        app._wizard_to_review()
        assert app.training_wizard_state.current_step == 3
        app.show_training_wizard_dataset()
        assert len(app.selected_records) == 1
        results["dataset_selection"] = {"records_from_two_roots": 12, "filter_keeps_selection": True,
            "last_row_reachable": True, "empty_filter_explained": True, "review_and_return": True}
        app.dataset_service = original_service
        app.selected_records = []
        app.geometry("980x760")
        app.start_training_wizard(); layout("training_setup")
        results["training_methods"] = check_registered_training_methods(app, layout)
        results["loso_selection"] = check_loso_selection_controls(app, layout, temporary)
        # The same wizard and start button must dispatch all existing recipes.
        from src.experiment_workflow import build_configuration
        from src.model_comparison import scores
        from src.posture_core import CLASS_NAMES
        from software.comparison_view import ComparisonView
        state = app.training_wizard_state
        app.wizard_kind.set("ensemble"); app._change_experiment_kind()
        state.soft_voting_members = ("rbf_svm:3", "extra_trees:3")
        state.voter_selection_initialized = True
        app.show_training_wizard_setup()
        for key, variable in app.wizard_voter_vars.items():
            variable.set(key in state.soft_voting_members)
        app._wizard_to_dataset(); layout("experiment_ensemble_data")
        assert state.current_step == 2 and len(state.soft_voting_members) == 2
        app.show_training_wizard_setup()
        app.wizard_kind.set("personal"); app._change_experiment_kind()
        assert app.wizard_model_vars is None
        app.personal_retrospective.set(False)
        app._wizard_to_dataset(); layout("experiment_personal_data")
        assert build_configuration(state)["windows"] == [3]
        # A separate synthetic registration profile is linked to an actual
        # temporary recording; no training CSV is reassigned as registration.
        from src.personal_guidance import CalibrationContext, CalibrationSession, CalibrationStore
        registration = CalibrationSession("synthetic", CalibrationContext(empty_seat_center=(1.,) * 16),
            guidance_confirmed=True, empty_seat_checked=True)
        for step in range(122):
            registration.push(step / 12, np.full(16, 2.) + .00001 * np.sin(step))
        profile = registration.finish()
        profile_store = CalibrationStore(Path(temporary) / "fixture_calibration_profiles")
        profile_path = profile_store.save(profile)
        assert profile_store.load("synthetic").profile_version == profile.profile_version
        record_path = (Path(temporary) / "personal_fixture" / "synthetic" / "record.csv").resolve()
        record_path.parent.mkdir(parents=True)
        with record_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_COLUMNS)
            writer.writerows([step / 12, "정자세", *[2. + .001 * np.sin(step + channel)
                               for channel in range(16)]] for step in range(37))
        record_path.with_suffix(".manifest.json").write_text(json.dumps({
            "csv": record_path.name, "subject": "synthetic", "mode": "labeled",
            "source_type": "collected", "label_status": "labeled", "training_role": "include",
            "guidance": {"reference_profile": profile.as_dict()}}, ensure_ascii=False), encoding="utf-8")
        record = DatasetService(Path(temporary), data_directory=record_path.parent.parent).records()[0]
        assert DatasetService.is_training_eligible(record)
        app.selected_records = [record]
        app._wizard_to_review(); layout("experiment_personal_review")
        assert state.current_step == 3
        with mock.patch.object(app.dataset_service, "records", return_value=[record]), \
             mock.patch.object(module, "TrainingJob") as job, \
             mock.patch.object(app, "show_training_progress"):
            app.start_training()
            assert job.call_args.kwargs["personalized"] is True
            assert job.call_args.kwargs["allow_retrospective"] is False
            assert job.call_args.kwargs["windows"] == (3,)
            assert job.call_args.kwargs["soft_voting_members"] == ()
            job.return_value.start.assert_called_once()
        app.training_job = None
        metrics = scores(np.arange(7), np.arange(7))
        metrics["confusion_labels"] = [*CLASS_NAMES, "판단 유보 / 종류 미확인"]
        metrics["confusion_matrix"] = np.pad(np.eye(7, dtype=int), ((0,1),(0,1))).tolist()
        personal_report = {"registration_records": [], "results": {
            split: {key: metrics for key in ("single_7class", "generic_hierarchy", "personal_hierarchy")}
            for split in ("validation", "final_evaluation")}}
        report_path = Path(temporary) / "personalized.report.json"
        report_path.write_text(json.dumps(personal_report), encoding="utf-8")
        snapshot = {**build_configuration(state), "selected_records": [str(record.csv_path)]}
        (Path(temporary) / "experiment.json").write_text(json.dumps(snapshot), encoding="utf-8")
        view = ComparisonView(app.clear(), [report_path], clone_callback=app.clone_experiment)
        view.pack(fill="both", expand=True)
        assert len(view.rows) == 3 and view.scope_choice.get() == "개인 표준화"
        for tab in range(4):
            view.tabs.select(tab); view.refresh(); layout(f"experiment_personal_result_{tab}")
        assert view.clone_button.instate(["!disabled"])
        with mock.patch.object(app.dataset_service, "records", return_value=[record]):
            view.clone_button.invoke()
        assert state.configuration_type == "personal" and state.experiment_name.endswith(" · 복제")
        assert state.selected_record_paths == (record.csv_path,)
        assert state.source_experiment == str(Path(temporary).resolve())
        app.wizard_kind.set("ensemble"); app._change_experiment_kind()
        assert {key for key,value in app.wizard_voter_vars.items() if value.get()} == {"rbf_svm:3", "extra_trees:3"}
        layout("experiment_clone_configuration")
        results["experiment_workflow"] = {"configuration_switch_preserves_drafts": True,
            "personal_job_dispatch": True, "unified_result_tabs": 4, "clone_preserves_data": True,
            "personal_confusion_labels": 8,
            "linked_registration_fixture": {"profile_saved": profile_path.is_file(),
                "profile_version": profile.profile_version, "record_sidecar_linked": True,
                "retrospective_reassignment": False}}
        app.selected_records = []

        app.show_management(); layout("management")
        app.management_notebook.select(1); layout("data_management")
        texts = []
        def gather(widget):
            if widget.winfo_class() in {"TButton", "TLabelframe"}:
                texts.append(str(widget.cget("text")))
            for child in widget.winfo_children():
                gather(child)
        gather(app)
        for required in ("필터 초기화", "분포맵 보기", "체크한 기록의 상태 변경"):
            assert required in texts, required
        app.filter_subject.set("x"); app.filter_status.set("폐기"); app.reset_dataset_filters()
        assert app.filter_subject.get() == "" and app.filter_status.get() == "전체"
        app.sort_dataset_table("행"); app.sort_dataset_table("행")
        rows = [record.rows for record in app.visible_records]
        assert rows == sorted(rows, reverse=True) and app.dataset_tree.heading("행")["text"].endswith("▼")
        with mock.patch.object(module.messagebox, "askyesno", return_value=False) as ask, \
             mock.patch.object(module.messagebox, "showerror"):
            app.selected_records = list(app.visible_records[:1])
            app.confirm_selected_status("폐기")
            assert ask.called == bool(app.visible_records)
        app.selected_records = []
        results["ux"] = {"filter_reset": True, "sort_descending": True, "status_confirmation": True,
                         "data_records_visible": len(app.visible_records)}
        app.management_notebook.select(0)
        reports = sorted((module.PROJECT_ROOT / "artifacts/models").glob("**/model_comparison.json"))
        reports = [path for path in reports if json.loads(path.read_text()).get("status") == "completed"]
        if reports:
            app.show_comparison_report(app.clear(), reports); layout("model_comparison")
        app.select_model(app.models[0])
        app.input_path.set(str(module.PROJECT_ROOT / "example" / "sample_input.csv"))
        app.start_measurement(); layout("realtime")
        event = GuidanceEvent(12., "incorrect", "Deviated from reference posture", None, 2., 12., "show", True,
                              "Return to your registered upright posture.", "gui-test", "development", {})
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            app.update(); time.sleep(.01)
        monitor = app.signal_monitor
        monitor.refresh(force=True)
        trace_points = monitor.trace_point_counts()
        assert monitor.canvas.winfo_ismapped() and len(trace_points) == 16 and min(trace_points) >= 2, trace_points
        results["signal_monitor"] = {"panels": len(trace_points), "min_trace_points": min(trace_points),
                                     "buffered_samples": monitor.buffer.snapshot().total_samples,
                                     "summary": monitor.summary.get()}
        before = app._recorder.rows
        app.show_warning(event)
        first_popup = app._warning_popup
        app.show_warning(event)
        duplicate_suppressed = app._warning_popup is first_popup
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            app.update(); time.sleep(.01)
        after = app._recorder.rows
        results["popup"] = {"rows_before": before, "rows_after": after, "measurement_continued": after > before,
                            "duplicate_suppressed": duplicate_suppressed, "modal_grab": str(app.grab_current())}
        app.dismiss_warning(notify=False)
        results["popup"]["dismissed"] = app._warning_popup is None
        app.stop_measurement()
        if app._worker: app._worker.join(timeout=3)
        app.destroy()
        assert results["default_mode"] == "deployment"
        assert after > before and duplicate_suppressed
        destination = args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results, indent=2))
        if any(screen["clipped_buttons"] for screen in results["screens"].values()):
            raise SystemExit("One or more screens has clipped controls")


if __name__ == "__main__":
    main()
