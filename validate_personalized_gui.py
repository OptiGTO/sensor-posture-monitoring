#!/usr/bin/env python3
"""Native Tk + real saved hierarchy + synthetic Serial input; no hardware evidence."""
import argparse
import json
from pathlib import Path
import tempfile
import time
from unittest import mock

import numpy as np

from software import posture_app as module
from src.personal_guidance import CalibrationContext, CalibrationSession, CalibrationStore
from src.personalized_model import available_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/personalized_native_gui.json'))
    parser.add_argument('--hold-seconds', type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='personalized-gui-') as temporary:
        root = Path(temporary)
        module.STATE_ROOT = root; module.DATA_DIRECTORY = root / 'data'
        app = module.PostureApplication(); app.title('개인 기준 2단계 검증')
        errors = []
        try:
            context = CalibrationContext(chair_id='chair-1', placement_id='placement-1', empty_seat_center=(1.,) * 16)
            calibration = CalibrationSession('gui-personal', context, guidance_confirmed=True, empty_seat_checked=True)
            for step in range(122): calibration.push(step / 12, np.full(16, 2.) + .00001 * np.sin(step))
            CalibrationStore(root / 'calibration_profiles').save(calibration.finish())
            # The personalized 3-second model must work even if the legacy card is 5 seconds.
            app.select_model(next(m for m in app.models if m.window_seconds == 5))
            # The submission build hides this experimental path; re-enable it here.
            app.SHOW_PERSONAL_HIERARCHY = True
            app.guidance_user.set('gui-personal'); app.personal_enabled.set(True); app.show_mode_select(); app.update()
            assert app._personal_choices
            app._scroll_canvas.yview_moveto(1.); app.update()
            if args.hold_seconds:
                deadline = time.monotonic() + args.hold_seconds
                while time.monotonic() < deadline: app.update(); time.sleep(.02)
            app.show_personal_report()
            app.update()
            for window in list(app.winfo_children()):
                if window.winfo_class() == 'Toplevel': window.destroy()
            def frames(*_args, **_kwargs):
                app._guidance_session.command('baseline')
                for step in range(361):
                    time.sleep(.003)
                    values = np.full(16, 1. if step < 38 else 2.)
                    if step >= 160: values[:8] += .8
                    yield step / 12, values + .00001 * np.sin(step * .3 + np.arange(16))
            with mock.patch.object(module, 'serial_frames', side_effect=frames), \
                 mock.patch.object(module, 'PosturePredictor', side_effect=AssertionError('Legacy predictor must be skipped')), \
                 mock.patch.object(module.messagebox, 'showerror', side_effect=lambda *args: errors.append(args)):
                app.start_measurement()
                assert app.session_snapshot.model_family == 'personalized_hierarchy'
                assert app.session_snapshot.selected_window_seconds == 3
                deadline = time.monotonic() + 8
                while not app._session_finished and time.monotonic() < deadline:
                    app.update(); time.sleep(.01)
                assert app._session_finished
                assert not errors, errors
                predictions = [json.loads(line) for line in app._recorder.prediction_path.read_text().splitlines()]
                assert predictions and app._recorder.rows == 361
                assert all(row['model_snapshot']['model_family'] == 'personalized_hierarchy' for row in predictions)
                assert all(row['window_timestamp_s'] is not None for row in predictions)
                assert app._recorder.manifest_path.is_file()
            # Check the training review actions without starting a second real-data run.
            # Current wizard: the personal path is the '개인 표준화' configuration, started by '실험 시작'.
            app.start_training_wizard(); app.update()
            def widgets(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from widgets(child)
            radios = [w for w in widgets(app) if w.winfo_class() == 'TRadiobutton' and w.cget('text') == '개인 표준화']
            assert len(radios) == 1, [w.cget('text') for w in widgets(app) if w.winfo_class() == 'TRadiobutton']
            radios[0].invoke(); app.update()
            assert app.training_wizard_state.configuration_type == 'personal'
            app.show_training_wizard_review(); app.update()
            assert app.personal_retrospective.get() is False
            controls = [w.cget('text') for w in widgets(app) if w.winfo_class() == 'TButton']
            labels = [str(w.cget('text')) for w in widgets(app) if w.winfo_class() in {'TLabel', 'TLabelframe'}]
            assert '실험 시작' in controls, controls
            assert '학습 설정: 기존 개인 기준 2단계의 고정 후보군' in labels, labels
            assert '등록 방식: 연결된 정자세 등록 기록' in labels, labels
            report = {'native_tk': True, 'real_saved_model': True, 'synthetic_serial': True, 'physical_sensor': False,
                      'legacy_model_window': 5, 'personal_window': 3, 'rows': app._recorder.rows,
                      'prediction_windows': len(predictions), 'training_review': True, 'errors': errors}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps(report, ensure_ascii=False, indent=2))
        finally:
            app._stop.set()
            if app._worker: app._worker.join(timeout=3)
            app.destroy()


if __name__ == '__main__': main()
