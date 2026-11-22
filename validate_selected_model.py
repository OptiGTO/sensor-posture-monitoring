#!/usr/bin/env python3
"""Review a locked comparison winner, apply it, and exercise the actual GUI worker."""
import argparse
import json
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
from unittest import mock

from software import posture_app as app
from software.posture_realtime import replay_csv
from src.app_services import MeasurementRecorder
from src.model_runtime import ModelRegistry, SessionModelSnapshot

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run_dir / "artifacts/model_comparison.json").read_text())
    if not report.get("selection_locked") or report.get("final_evaluation_used_for_selection") or report["status"] != "completed":
        raise ValueError("A completed, locked validation selection is required.")
    selected = report["selected"]
    registry = ModelRegistry(ROOT, ROOT)
    manifest = args.run_dir / "artifacts/manifests" / f"posture_classifier_{selected['window_seconds']}s_{selected['family']}_candidate.json"
    model = next(item for item in registry.management_models() if item.manifest_path == manifest.resolve())
    model = registry.validate_runtime(model)
    model = registry.validate_deployment(model)
    model = registry.approve(model, approved_by="Codex technical review under user-authorized integrated implementation")
    previous = registry.active_model(registry.discover())
    if not registry.active_path.exists() and previous:
        registry.activate(previous)
    registry.activate(model)
    snapshot = SessionModelSnapshot.create(model, session_id="selected-model-replay", window_seconds=3, input_source="csv_replay")
    output = ROOT / "results/integrated_validation"
    recorder = MeasurementRecorder(output, mode="measurement", model_snapshot=snapshot)
    session = SimpleNamespace(selected_model=model, registry=registry, session_snapshot=snapshot,
        _recorder=recorder, _session_input=(str(ROOT / "example/sample_input.csv"), "auto"),
        _stop=threading.Event(), _messages=queue.Queue())
    with mock.patch.object(app, "replay_csv", side_effect=lambda path, speed, **kw: replay_csv(path, 0, **kw)):
        app.PostureApplication._run_session(session)
    messages = list(session._messages.queue)
    errors = [payload for kind, payload in messages if kind == "error"]
    predictions = [payload[0].as_dict() for kind, payload in messages if kind == "prediction"]
    if errors or not predictions:
        raise RuntimeError(f"Selected model replay failed: {errors}")
    release = {"schema_version": 1, "artifact_id": model.artifact_id, "artifact_hash": model.artifact_hash,
        "manifest_hash": model.manifest_hash, "manifest_file": str(model.manifest_path.relative_to(ROOT / "artifacts")),
        "selection_basis": "locked validation Macro F1, class performance and measured latency review",
        "release_status": "experimental; physical sensor validation and personal thresholds pending"}
    (ROOT / "artifacts/release_selection.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    evidence = {"status": "passed", "selected": release, "active_artifact_id": registry.active_model(registry.discover()).artifact_id,
        "previous_artifact_id": previous.artifact_id if previous else None, "frames_recorded": recorder.rows,
        "prediction_count": len(predictions), "model_snapshot": snapshot.as_dict(),
        "measurement_manifest": str(recorder.manifest_path.relative_to(ROOT)),
        "predictions": predictions, "errors": errors}
    (output / "selected_model_replay.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in evidence.items() if key not in {"predictions", "model_snapshot"}}, indent=2))


if __name__ == "__main__":
    main()
