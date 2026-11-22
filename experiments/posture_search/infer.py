"""Run the study's locked selected model on an unlabeled sensor recording.

Example:
    .venv/bin/python -m experiments.posture_search.infer query.csv \
        --reference registered_correct.csv --output /tmp/posture_predictions.json

This is retrospective experimental inference. It does not activate or replace
the application's runtime model. Only the selected primary checkpoint listed
and hashed in a trusted local study selection lock is accepted. CSV ``label``
columns, if present, are discarded; prediction inputs contain no labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .data import OUT, ROOT, digest, write_json
from .run import Arm, Fusion, Temporal
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, read_sensor_csv
from src.window_pipeline import WindowSpec, segment_sensor_frame


def _requires_reference(model) -> bool:
    if isinstance(model, Arm):
        return model.spec["features"] in {"personal", "personal_spatial"}
    if isinstance(model, Fusion):
        return any(_requires_reference(member) for member in model.members)
    if isinstance(model, Temporal):
        return _requires_reference(model.parent)
    raise ValueError(f"Unsupported locked model type: {type(model).__name__}")


def load_selected(study_dir: Path = OUT):
    """Verify the study lock and load its selected model, never a supplied pickle."""
    study_dir = Path(study_dir).resolve()
    lock_path = study_dir / "selection_lock.json"
    if not lock_path.is_file():
        raise ValueError("Complete validation selection and create selection_lock.json first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selected_id = lock["selected_id"]
    if not isinstance(selected_id, str) or not re.fullmatch(r"[BHCAP]\d{2}", selected_id):
        raise ValueError("Unexpected selected experiment ID in lock")
    if lock["selected_spec"]["id"] != selected_id:
        raise ValueError("Selected experiment and selected specification disagree")
    if lock.get("test_used_for_selection") is not False:
        raise ValueError("Expected a model selected without test outcomes")
    manifest = json.loads((study_dir / "dataset.json").read_text(encoding="utf-8"))
    if manifest["fingerprint"] != lock["dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint disagrees with selection lock")
    if manifest["class_names"] != CLASS_NAMES:
        raise ValueError("Stored class order disagrees with the inference implementation")
    expected_window = {"window_seconds": 3, "sampling_rate": 12,
                       "stride_seconds": 3, "min_tail_fraction": .95}
    if any(manifest.get(key) != value for key, value in expected_window.items()):
        raise ValueError("The selected study does not use the supported 3-second, 12-Hz window protocol")
    if digest(study_dir / "protocol.json") != lock["protocol_sha256"]:
        raise ValueError("Study protocol changed after selection")
    for relative, expected in lock["source_hashes"].items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT.resolve()) or digest(source) != expected:
            raise ValueError(f"Locked source changed or escaped repository: {relative}")
    relative_checkpoint = f"{selected_id}/primary/model.joblib"
    checkpoint = study_dir / relative_checkpoint
    expected = lock["checkpoints"].get(relative_checkpoint)
    if not expected or digest(checkpoint) != expected:
        raise ValueError("Selected checkpoint is missing from the lock or its hash changed")
    model = joblib.load(checkpoint)
    _requires_reference(model)  # Validate supported composition before reading user data.
    return model, manifest, {
        "selected_id": selected_id, "selected_spec": lock["selected_spec"],
        "selection_lock_sha256": digest(lock_path), "checkpoint_sha256": expected,
        "dataset_fingerprint": manifest["fingerprint"],
    }


def make_inputs(csv_path: Path, reference_path: Path | None = None,
                require_reference: bool = False):
    """Reproduce study segmentation/personalization with no true or dummy labels."""
    csv_path = Path(csv_path).resolve()
    frame = read_sensor_csv(csv_path).loc[:, ["elapsed_s", *SENSOR_COLUMNS]]
    if np.any(frame[SENSOR_COLUMNS].to_numpy() < 0):
        raise ValueError("Sensor pressure values must be nonnegative")
    windows, metadata = segment_sensor_frame(frame, WindowSpec(3.0))
    data = {
        "seq": windows,
        "record_ids": np.repeat(str(csv_path), len(windows)),
        "start_s": metadata["start_s"].to_numpy(dtype=float),
    }
    provenance = {"path": str(csv_path), "sha256": digest(csv_path),
                  "labels_used": False, "reference": None}
    if require_reference and reference_path is None:
        raise ValueError("Selected model requires --reference with a separate registered correct-posture recording")
    if reference_path is not None:
        reference_path = Path(reference_path).resolve()
        if reference_path == csv_path or digest(reference_path) == provenance["sha256"]:
            raise ValueError("The registered reference must be a separate recording from the query")
        reference_frame = read_sensor_csv(reference_path).loc[:, ["elapsed_s", *SENSOR_COLUMNS]]
        reference_values = reference_frame[SENSOR_COLUMNS].to_numpy(dtype=float)
        query_values = frame[SENSOR_COLUMNS].to_numpy(dtype=float)
        if reference_values.shape == query_values.shape and np.array_equal(reference_values, query_values):
            raise ValueError("The registered reference duplicates the query sensor sequence")
        elapsed = reference_frame["elapsed_s"].to_numpy(dtype=float)
        relative = elapsed - elapsed[0]
        if relative[-1] < 10.0 or np.any(np.diff(elapsed) > 1.0):
            raise ValueError("Reference requires at least 10 continuous seconds with no sensor gap above 1 second")
        if np.any(reference_values < 0):
            raise ValueError("Reference pressure values must be nonnegative")
        reference_mean = reference_values[relative <= 10.0].mean(axis=0)
        reference_ratio = reference_mean / max(reference_mean.sum(), 1e-6)
        means = windows.mean(axis=1)
        ratio = means / np.maximum(means.sum(axis=1, keepdims=True), 1e-6)
        data["personal"] = np.concatenate([
            np.tile(reference_ratio, (len(windows), 1)), ratio - reference_ratio,
            np.log((means + 1) / (reference_mean + 1)),
            ((means.sum(axis=1) + 1) / (reference_mean.sum() + 1))[:, None],
        ], axis=1).astype(np.float32)
        provenance["reference"] = {
            "path": str(reference_path), "sha256": digest(reference_path),
            "reference_mean": reference_mean.tolist(), "reference_ratio": reference_ratio.tolist(),
            "window": "first_10_seconds_inclusive", "labels_used": False,
            "correct_posture_confirmation": "caller_supplied_reference_not_inferred_from_CSV_labels",
        }
    return data, metadata, provenance


def infer_recording(csv_path: Path, reference_path: Path | None = None,
                    study_dir: Path = OUT) -> dict:
    model, manifest, model_info = load_selected(study_dir)
    required_reference = _requires_reference(model)
    data, metadata, provenance = make_inputs(csv_path, reference_path, required_reference)
    with threadpool_limits(limits=1):
        started = time.perf_counter()
        probabilities = np.asarray(model.predict_proba(data))
        seconds = time.perf_counter() - started
    if (probabilities.shape != (len(metadata), len(CLASS_NAMES)) or
            not np.isfinite(probabilities).all() or np.any(probabilities < 0) or
            not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError("Selected model returned invalid class probabilities")
    predicted = probabilities.argmax(axis=1)
    rows = []
    for index, window in enumerate(metadata.to_dict(orient="records")):
        rows.append({
            **window, "predicted_class_id": int(predicted[index]),
            "predicted_class": CLASS_NAMES[predicted[index]],
            "probabilities": {name: float(probabilities[index, i]) for i, name in enumerate(CLASS_NAMES)},
        })
    return {
        "status": "retrospective_experimental_inference",
        "runtime_model_activated": False, "physically_validated": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_info, "input": provenance, "reference_required": required_reference,
        "window_config": {key: manifest[key] for key in (
            "window_seconds", "sampling_rate", "stride_seconds", "min_tail_fraction")},
        "temporal_state": "reset_at_start_of_recording_then_preserved_causally_across_its_windows",
        "scope": "CSV retrospective full-record inference; no runtime registry, GUI, serial, or model activation changes",
        "probability_note": "normalized model scores; calibration and posture-transition performance are not established",
        "inference_seconds": seconds, "window_count": len(rows), "windows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Query CSV with elapsed_s and R11..R44 columns; labels ignored")
    parser.add_argument("--reference", type=Path, help="Separate correct-posture CSV; first 10 seconds are used")
    parser.add_argument("--study-dir", type=Path, default=OUT, help="Trusted study output containing the selection lock")
    parser.add_argument("--output", type=Path, help="Write JSON here; otherwise print JSON to stdout")
    args = parser.parse_args()
    if args.output is not None and args.output.resolve() in {
            args.csv.resolve(), args.reference.resolve() if args.reference else None}:
        parser.error("--output must not overwrite the query or reference CSV")
    try:
        result = infer_recording(args.csv, args.reference, args.study_dir)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    if args.output is None:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        write_json(args.output, result)
        print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
