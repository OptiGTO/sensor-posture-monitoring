#!/usr/bin/env python3
"""Native Tk button-level audit using isolated synthetic records; no sensor evidence."""
import argparse
import json
import tempfile
import time
import traceback
from pathlib import Path
from unittest import mock

import numpy as np

from software import posture_app as module
from src.app_services import DatasetService, MeasurementRecorder
from src.posture_core import CLASS_NAMES
from software.posture_realtime import replay_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checks, errors = [], []
    with tempfile.TemporaryDirectory(prefix='posture-ui-workflows-') as temporary:
        root = Path(temporary).resolve()
        module.STATE_ROOT, module.DATA_DIRECTORY = root, root / 'data'
        # Eight disjoint synthetic participants, all seven classes. Seed is fixed.
        rng = np.random.default_rng(42)
        for subject in range(8):
            for index, label in enumerate(CLASS_NAMES):
                recorder = MeasurementRecorder(root, mode='labeled', subject=f'qa-{subject}', label=label)
                for step in range(73):
                    values = 2 + index * .2 + np.sin(np.arange(16) + index) * .1 + rng.normal(0, .005, 16)
                    recorder.write_frame(step / 12, values)
                recorder.close()
        service = DatasetService(root)
        for record in service.records():
            service.set_status(record, '학습 대기열에 추가')
        app = module.PostureApplication()
        app.report_callback_exception = lambda *exc: errors.append(''.join(traceback.format_exception(*exc)))

        def walk(widget):
            for child in widget.winfo_children():
                yield child
                yield from walk(child)

        def click(label):
            app.update()
            buttons = [w for w in walk(app) if w.winfo_class() == 'TButton' and str(w.cget('text')) == label]
            assert len(buttons) == 1, (label, len(buttons))
            assert not buttons[0].instate(['disabled']), label
            buttons[0].invoke(); app.update()

        def layout(name):
            app.update()
            for widget in walk(app):
                if widget.winfo_ismapped() and widget.winfo_class() in {'TButton', 'TEntry', 'TCombobox', 'TCheckbutton', 'TRadiobutton', 'Treeview'}:
                    right = widget.winfo_rootx() - app.winfo_rootx() + widget.winfo_width()
                    assert right <= app.winfo_width(), (name, str(widget), right)
            app._scroll_canvas.yview_moveto(1); app.update()
            assert app._scroll_canvas.yview()[1] == 1, name
            app._scroll_canvas.yview_moveto(0)
            checks.append(name)

        def wait_until(predicate, timeout=120):
            deadline = time.monotonic() + timeout
            while not predicate():
                assert time.monotonic() < deadline, 'UI operation timed out'
                assert not errors, errors
                app.update(); time.sleep(.02)
            app.update()

        try:
            with mock.patch.object(module.messagebox, 'showerror') as popup, mock.patch.object(module.messagebox, 'showinfo'), mock.patch.object(module.messagebox, 'askyesno', return_value=True):
                for dimensions in ('980x760', '850x700'):
                    app.geometry(dimensions); app.show_model_select(); layout('model_selection_' + dimensions)
                    app.select_model(app.models[0]); layout('measurement_setup_' + dimensions)
                    for mode in ('labeled', 'calibration'):
                        app.mode.set(mode); app.show_mode_select(); layout(mode + '_' + dimensions)
                    app.show_management(); layout('management_' + dimensions)
                    app.start_training_wizard(); layout('wizard_setup_' + dimensions)
                    click('다음: 데이터'); layout('wizard_dataset_' + dimensions)
                app.mode.set('measurement'); app.select_model(app.models[0])
                app.input_path.set(str(root / 'missing.csv'))
                app.start_measurement()
                assert popup.call_count == 1 and app._recorder is None
                popup.reset_mock()
                source = module.PROJECT_ROOT / 'example/sample_input.csv'
                app.input_path.set(str(source))
                with mock.patch.object(module, 'replay_csv', side_effect=lambda path, speed, **kwargs: replay_csv(path, 0, **kwargs)):
                    click('센서 연결 및 측정 시작')
                    wait_until(lambda: app._session_finished)
                assert app._recorder.rows > 0 and app._last_manifest.is_file()
                layout('measurement_saved')
                click('다시 측정')
                app.mode.set('labeled'); app.show_mode_select()
                app.subject.set('gui-capture'); app.ground_truth.set('정자세')
                with mock.patch.object(module, 'replay_csv', side_effect=lambda path, speed, **kwargs: replay_csv(path, 0, **kwargs)):
                    click('센서 연결 및 측정 시작'); wait_until(lambda: app._session_finished)
                layout('labeled_saved')
                click('저장만')
                assert json.loads(app._last_manifest.read_text())['training_role'] == 'exclude'
                click('지금 학습 시작')
                assert len(app.selected_records) == 1
                click('취소')
                click('데이터 및 모델 관리')
                app.filter_subject.set('qa-'); click('필터 적용')
                assert len(app.visible_records) == 56
                click('전체 선택 해제'); click('현재 목록 전체 선택')
                assert len(app.selected_records) == 56
                app.filter_subject.set('qa-0'); click('필터 적용')
                assert len(app.visible_records) == 7 and len(app.selected_records) == 56
                app.filter_min_duration.set('nan'); click('필터 적용')
                assert popup.call_count == 1 and len(app.visible_records) == 7
                popup.reset_mock(); app.filter_min_duration.set(''); click('필터 적용')
                click('전체 선택 해제'); click('현재 목록 전체 선택')
                click('평가 전용')
                click('+ 모델 추가')
                click('다음: 데이터')
                app.filter_subject.set('qa-'); click('필터 적용')
                assert len(app.visible_records) == 49 and not app.selected_records, (len(app.visible_records), len(app.selected_records), [(r.subject, r.training_role, r.qc_status) for r in app.dataset_service.records()][:10])
                click('이전: 구성')
                for family, variable in app.wizard_model_vars.items():
                    variable.set(family == 'logistic_regression')
                click('다음: 데이터')
                app.filter_subject.set('qa-'); click('필터 적용')
                click('현재 목록 전체 선택'); click('다음: 검토')
                layout('training_review')
                click('이전: 데이터'); assert len(app.selected_records) == 49
                click('다음: 검토'); click('실험 시작')
                assert app.training_job is not None
                layout('training_progress')
                wait_until(lambda: not app.training_job.is_running)
                wait_until(lambda: any(w.winfo_class() == 'TButton' and str(w.cget('text')) == '실험 결과 보기' for w in walk(app)))
                assert app.training_job.candidate_manifest_paths, app.training_status.get()
                assert all(item['passed'] for item in app.training_job.validation_results)
                checks.append('real_training_worker_and_automatic_validation')
                click('실험 결과 보기'); assert app.comparison_view.rows
                layout('trained_results')
                click('데이터 및 모델 관리')
                # Current management screen: software/model_management_view.ModelManagementView.
                view = app.model_management
                ours = lambda model: Path(model.path).resolve().is_relative_to(root)
                candidate = next((iid for iid, model in view.rows.items() if ours(model) and model.approval_status == 'pending'), None)
                assert candidate, [(m.name, m.approval_status) for m in view.rows.values()]
                model = view.rows[candidate]
                assert model.validation_status == model.deployment_validation_status == 'passed', (
                    model.validation_status, model.deployment_validation_status, model.lifecycle_reason)
                view.tree.selection_set(candidate); view.update_actions(); app.update()
                click('자세히 보기')
                details = [w for w in walk(app) if isinstance(w, module.tk.Toplevel) and w.title() == '모델 검토 정보']
                assert len(details) == 1
                details[0].destroy(); app.update()
                # Exercise review and activation only in temporary application state.
                click('선택 승인 (1)')
                wait_until(lambda: not view.busy)
                approved = next(iid for iid, model in view.rows.items() if ours(model) and model.approval_status == 'approved')
                view.tree.selection_set(approved); view.update_actions(); app.update()
                click('기본 모델로 저장')
                assert Path(app.registry.active_model(app.registry.discover()).path).resolve().is_relative_to(root)
                checks.append('ui_approve_and_activate_isolated_model')
                app.start_training_wizard(app.dataset_service.records())
                for family, variable in app.wizard_model_vars.items(): variable.set(family == 'logistic_regression')
                click('다음: 데이터'); click('다음: 검토'); click('실험 시작'); click('학습 취소')
                wait_until(lambda: not app.training_job.is_running)
                wait_until(lambda: any(w.winfo_class() == 'TButton' and str(w.cget('text')) == '학습 설정으로' for w in walk(app)))
                click('학습 설정으로'); layout('cancelled_training_retry')
                assert not popup.called, popup.call_args_list
                assert not errors, errors
                app._close()
                checks.append('window_close')
        finally:
            try:
                exists = app.winfo_exists()
            except module.tk.TclError:
                exists = False
            if exists:
                app._stop.set()
                if app._worker: app._worker.join(5)
                if app.training_job: app.training_job.cancel()
                app.destroy()
    result = {'native_tk': True, 'synthetic_training_data': True, 'physical_sensor': False, 'checks': checks, 'callback_errors': errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
