#!/usr/bin/env python3
"""Run the production services in an isolated state directory; never activate a live model."""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.app_services import DatasetService, MeasurementRecorder, TrainingJob
from src.model_runtime import ModelRegistry, SessionModelSnapshot
from software import posture_app
from software.posture_realtime import replay_csv

ROOT = Path(__file__).resolve().parent


def wait_job(job, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            kind, payload = job.events.get(timeout=.25)
        except queue.Empty:
            continue
        print(f"{kind}: {payload}", flush=True)
        if kind == 'finished':
            if job._reader:
                job._reader.join(5)
            return payload
    job.cancel()
    raise TimeoutError('Training did not finish within the validation deadline')


# The default model selection became LOSO on 2026-09-28 (27 extra fits per model and window on
# the current cohort), so real-data training takes tens of minutes, not the old 15.
TRAINING_TIMEOUT_SECONDS = 3600


def run(output: Path, training_timeout: float = TRAINING_TIMEOUT_SECONDS) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    state = output / 'state'
    report = {'status': 'running', 'state_root': str(state), 'checks': {}}
    checks = report['checks']
    try:
        started = time.perf_counter()
        service = DatasetService(ROOT)
        records = service.records()
        eligible = [record for record in records if service.is_training_eligible(record)]
        checks['dataset'] = {'records': len(records), 'eligible': len(eligible),
            'subjects': len({record.subject for record in eligible}),
            'scan_seconds': time.perf_counter() - started}
        registry = ModelRegistry(ROOT, state)
        started = time.perf_counter()
        models = registry.discover()
        checks['startup'] = {'model_count': len(models), 'cold_discovery_seconds': time.perf_counter() - started}
        started = time.perf_counter(); registry.discover()
        checks['startup']['warm_discovery_seconds'] = time.perf_counter() - started
        assert models
        checks['replay'] = []
        for model in models:
            snapshot = SessionModelSnapshot.create(model, session_id=f'audit-{model.window_seconds:g}s',
                window_seconds=model.window_seconds, input_source='csv_replay')
            recorder = MeasurementRecorder(state, mode='measurement', model_snapshot=snapshot)
            session = SimpleNamespace(selected_model=model, registry=registry, session_snapshot=snapshot,
                _recorder=recorder, _session_input=(str(ROOT / 'example/sample_input.csv'), 'auto'),
                _stop=threading.Event(), _messages=queue.Queue())
            with patch.object(posture_app, 'replay_csv', side_effect=lambda path, speed, **kwargs: replay_csv(path, 0, **kwargs)):
                posture_app.PostureApplication._run_session(session)
            events = list(session._messages.queue)
            errors = [str(payload) for kind, payload in events if kind == 'error']
            predictions = [json.loads(line) for line in recorder.prediction_path.read_text().splitlines()]
            assert not errors and predictions and events[-1][0] == 'finished', errors
            checks['replay'].append({'window_seconds': model.window_seconds, 'rows': recorder.rows,
                'predictions': len(predictions), 'manifest': str(recorder.manifest_path)})
        started = time.perf_counter()
        job = TrainingJob(ROOT, eligible, 42, windows=(3, 5, 20), state_root=state)
        job.start()
        assert wait_job(job, training_timeout) == '0'
        assert len(job.validation_results) == 12 and all(item['passed'] for item in job.validation_results), job.validation_results
        comparison = json.loads(job.comparison_report_path.read_text())
        assert comparison['selection_locked'] and not comparison['final_evaluation_used_for_selection']
        checks['training'] = {'seconds': time.perf_counter() - started, 'candidate_count': len(job.candidate_manifest_paths),
            'comparison_report': str(job.comparison_report_path), 'selected': comparison['selected'],
            'models': [{'family': row['family'], 'window_seconds': row['window_seconds'],
                'validation_macro_f1': row['validation']['macro_f1'],
                'final_macro_f1': row['final_evaluation']['macro_f1'],
                'inference_ms_median': row['inference_ms_median']} for row in comparison['models']],
            'validation': job.validation_results}
        candidate_paths = {path.resolve() for path in job.candidate_manifest_paths}
        candidates = [model for model in registry.management_models() if model.manifest_path in candidate_paths]
        assert len(candidates) == 12 and all(model.approval_status == 'pending' for model in candidates)
        assert not registry.active_path.exists(), 'Training must not activate models'
        candidate = registry.approve(candidates[0], approved_by='isolated-pipeline-test')
        registry.activate(models[0]); registry.activate(candidate)
        restarted = ModelRegistry(ROOT, state)
        assert restarted.active_model(restarted.discover()).artifact_id == candidate.artifact_id
        assert restarted.rollback().artifact_id == models[0].artifact_id
        restarted.reject(candidates[1], reason='isolated review test')
        restarted.revoke(candidate, reason='isolated revocation test')
        try:
            restarted.activate(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError('A revoked candidate was activated')
        checks['lifecycle'] = {'approval_activation_restart_rollback_rejection_revocation': 'passed', 'isolated_state_only': True}
        cancelled = TrainingJob(ROOT, eligible, 42, windows=(3,), state_root=state)
        cancelled.start(); cancelled.cancel()
        assert wait_job(cancelled, 30) == '1'
        assert not cancelled.is_running and not cancelled.candidate_manifest_paths
        checks['cancellation'] = {'status': 'passed', 'run_dir': str(cancelled.run_dir)}
        from serial.tools.list_ports import comports
        checks['physical_serial'] = {'ports': [port.device for port in comports()],
            'status': 'not_tested', 'reason': 'No physical measurement is performed by this recorded-data validation.'}
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (output / 'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New, empty results directory')
    parser.add_argument('--training-timeout', type=float, default=TRAINING_TIMEOUT_SECONDS,
                        help='Seconds allowed for the real-data training job (default %(default)s)')
    args = parser.parse_args()
    run(args.output.resolve(), args.training_timeout)
