#!/usr/bin/env python3
"""Audit personal-reference guidance using separate historical CSV records.

This is a retrospective replay, not a supervised physical sensor validation.
The first correct-posture recording is reserved before inspecting outcomes;
default registration/detection thresholds are never fitted to evaluation data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from src.app_services import quality_check
from src.posture_core import (
    CLASS_NAMES, SENSOR_COLUMNS, normalize_text, read_sensor_csv,
    recording_label, training_csv_paths,
)


@dataclass(frozen=True)
class ReplayRecord:
    path: Path
    relative_path: str
    subject: str
    label: str
    rows: int
    duration_s: float
    sha256: str
    sensor_sha256: str

    def evidence(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key != "path"}


def audit_dataset(data_root: Path) -> tuple[list[ReplayRecord], dict]:
    """Inspect records without relabeling, changing manifests, or dropping errors silently."""
    records: list[ReplayRecord] = []
    excluded: list[dict] = []
    raw_duplicates: dict[str, list[str]] = defaultdict(list)
    sensor_duplicates: dict[str, list[str]] = defaultdict(list)
    for path in training_csv_paths(data_root):
        relative = path.relative_to(data_root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            frame = read_sensor_csv(path)
            _, label = recording_label(frame)
            qc = quality_check(path)
            if qc["status"] != "pass":
                raise ValueError("Quality check: " + "; ".join(qc["issues"]))
            values = np.asarray(frame[SENSOR_COLUMNS], dtype="<f8")
            sensor_digest = hashlib.sha256(values.tobytes()).hexdigest()
            duration = float(frame.elapsed_s.iloc[-1] - frame.elapsed_s.iloc[0])
            records.append(ReplayRecord(path.resolve(), relative,
                                       normalize_text(path.parent.name), label,
                                       len(frame), duration, digest, sensor_digest))
            raw_duplicates[digest].append(relative)
            sensor_duplicates[sensor_digest].append(relative)
        except (ValueError, OSError) as error:
            excluded.append({"relative_path": relative, "sha256": digest,
                             "reason": str(error)})
    coverage: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        coverage[record.subject][record.label] += 1
    report = {
        "eligible_record_count": len(records), "excluded_record_count": len(excluded),
        "subject_count": len(coverage), "class_order": CLASS_NAMES,
        "subject_class_record_counts": dict(coverage),
        "row_count": sum(record.rows for record in records),
        "total_duration_s": sum(record.duration_s for record in records),
        "duration_range_s": ([min(record.duration_s for record in records),
                              max(record.duration_s for record in records)] if records else []),
        "duplicate_raw_files": [paths for paths in raw_duplicates.values() if len(paths) > 1],
        "duplicate_sensor_sequences": [paths for paths in sensor_duplicates.values() if len(paths) > 1],
        "excluded_records": excluded,
        "records": [record.evidence() for record in records],
    }
    return records, report


def assert_separate(registration: ReplayRecord, evaluation: Iterable[ReplayRecord]) -> None:
    """Reject path/content overlap even when a duplicate has a different filename/label."""
    for record in evaluation:
        if record.subject != registration.subject:
            raise ValueError("Registration and evaluation must belong to the same subject")
        if (record.path.resolve() == registration.path.resolve()
                or record.sha256 == registration.sha256
                or record.sensor_sha256 == registration.sensor_sha256):
            raise ValueError("Registration/evaluation record or sensor content overlap")


def registration_plan(records: Iterable[ReplayRecord], *, enabled=False) -> list[tuple[ReplayRecord, list[ReplayRecord]]]:
    """Historical reservation requires explicit opt-in; ordinary records stay unassigned."""
    if not enabled:
        return []
    subjects: dict[str, list[ReplayRecord]] = defaultdict(list)
    for record in records:
        subjects[record.subject].append(record)
    plans = []
    for subject in sorted(subjects):
        ordered = sorted(subjects[subject], key=lambda item: normalize_text(item.relative_path))
        correct = [record for record in ordered if record.label == CLASS_NAMES[0]]
        if not correct:
            continue
        registration = correct[0]
        evaluation = [record for record in ordered if record.path != registration.path]
        assert_separate(registration, evaluation)
        plans.append((registration, evaluation))
    return plans


def summarize_records(results: Iterable[dict]) -> dict:
    results = list(results)
    correct_s = sum(row["duration_s"] for row in results if row["ground_truth"] == CLASS_NAMES[0])
    incorrect_s = sum(row["duration_s"] for row in results if row["ground_truth"] != CLASS_NAMES[0])
    false_positive_s = sum(row["state_duration_s"].get("incorrect", 0.0) for row in results
                           if row["ground_truth"] == CLASS_NAMES[0])
    missed_s = sum(row["state_duration_s"].get("correct", 0.0) for row in results
                   if row["ground_truth"] != CLASS_NAMES[0])
    false_alerts = sum(len(row["popup_timestamps_s"]) for row in results
                      if row["ground_truth"] == CLASS_NAMES[0])
    total_states = Counter()
    for row in results:
        total_states.update(row["state_duration_s"])
    return {
        "record_count": len(results), "correct_exposure_s": correct_s,
        "incorrect_exposure_s": incorrect_s,
        "incorrect_missed_as_correct_rate": missed_s / incorrect_s if incorrect_s else None,
        "correct_classified_incorrect_rate": false_positive_s / correct_s if correct_s else None,
        "false_alert_count": false_alerts,
        "false_alerts_per_correct_hour": false_alerts * 3600 / correct_s if correct_s else None,
        "state_duration_s": dict(total_states),
        "metric_definition": "Time weighted, all eligible exposure included; uncertain/movement/insufficient input are reported separately and are not relabeled correct.",
        "actual_posture_onset_to_popup_s": None,
        "actual_correct_return_to_warning_dismissal_s": None,
        "continuous_deployment_alert_validation": "Unavailable: separate short single-posture records do not establish continuous 60-second posture episodes or actual transition times.",
    }


def register_from_record(record: ReplayRecord):
    from src.personal_guidance import CalibrationContext, CalibrationSession

    # Historical files do not identify the matching unloaded baseline or chair.
    # These confirmations are explicit replay protocol assumptions, never a
    # claim that a person confirmed the posture or checked an empty chair today.
    context = CalibrationContext(chair_id="historical-chair-unknown",
                                 placement_id="historical-placement-unknown",
                                 sensor_configuration="historical-pressure-4x4-v1")
    registration = CalibrationSession(record.subject, context,
                                      guidance_confirmed=True, empty_seat_checked=True)
    frame = read_sensor_csv(record.path)
    times = frame.elapsed_s.to_numpy(dtype=float)
    times -= times[0]
    for timestamp, values in zip(times, frame[SENSOR_COLUMNS].to_numpy(dtype=float)):
        progress = registration.push(timestamp, values)
        if progress >= 1.0:
            break
    return registration.finish()


def replay_record(record: ReplayRecord, profile, mode: str) -> dict:
    from src.personal_guidance import AlertSettings, GuidanceEngine

    frame = read_sensor_csv(record.path)
    times = frame.elapsed_s.to_numpy(dtype=float)
    times -= times[0]
    engine = GuidanceEngine(profile, settings=AlertSettings.for_mode(mode))
    durations: Counter = Counter()
    popups, dismissals, transitions = [], [], []
    for index, (timestamp, values) in enumerate(zip(times, frame[SENSOR_COLUMNS].to_numpy(dtype=float))):
        # The personal gate is independent of model posture typing. Unknown
        # type deliberately exercises the general correction notification.
        event = engine.update(timestamp, values, seated=True)
        if index + 1 < len(times):
            durations[event.state] += float(times[index + 1] - timestamp)
        if event.popup_action == "show":
            popups.append(float(timestamp))
        elif event.popup_action == "dismiss":
            dismissals.append(float(timestamp))
        if not transitions or transitions[-1]["state"] != event.state:
            transitions.append({"timestamp_s": float(timestamp), "state": event.state})
    return {
        **record.evidence(), "ground_truth": record.label,
        "state_duration_s": dict(durations), "popup_timestamps_s": popups,
        "popup_dismissal_timestamps_s": dismissals, "transitions": transitions,
        "timing_origin": "Recording start; actual posture onset and return were not annotated.",
    }


def evaluate_personalization(data_root: Path, *, modes: tuple[str, ...] = ("development", "deployment"),
                             subjects: set[str] | None = None, allow_retrospective=False) -> tuple[dict, dict]:
    from src.personal_guidance import AlertSettings, CalibrationConfig, GuidanceConfig

    if not allow_retrospective:
        raise ValueError("과거 정자세 기록의 등록용 자동 배정은 해제되었습니다. 재현 실험은 --allow-retrospective를 명시하세요.")
    records, audit = audit_dataset(data_root)
    selected = records if subjects is None else [record for record in records if record.subject in subjects]
    plans = registration_plan(selected, enabled=allow_retrospective)
    profiles = []
    outcomes = []
    mode_records: dict[str, list[dict]] = {mode: [] for mode in modes}
    for registration, evaluation in plans:
        outcome = {"subject": registration.subject, "registration_record": registration.evidence(),
                   "evaluation_records": [record.evidence() for record in evaluation],
                   "has_independent_correct_evaluation": any(record.label == CLASS_NAMES[0] for record in evaluation)}
        try:
            profile = register_from_record(registration)
        except ValueError as error:
            outcome.update({"status": "registration_failed", "reason": str(error)})
            outcomes.append(outcome)
            continue
        outcome.update({"status": "registered", "profile_version": profile.profile_version,
                        "stability_metrics": profile.stability_metrics})
        outcomes.append(outcome)
        profiles.append({"artifact_type": "historical_replay_profile_not_live_registration",
                         "actual_sensor_validated": False,
                         "source_record": registration.evidence(),
                         "confirmation_basis": "Historical correct-posture label; replay assumes guidance and empty-seat checks. No supervised confirmation was performed.",
                         "profile": profile.as_dict()})
        for mode in modes:
            mode_records[mode].extend(replay_record(record, profile, mode) for record in evaluation)
    try:
        from serial.tools import list_ports
        ports = [{"device": port.device, "description": port.description, "hardware_id": port.hwid}
                 for port in list_ports.comports()]
    except ImportError:
        ports = []
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_kind": "retrospective_model_independent_personal_reference_gate",
        "data_root": str(data_root.resolve()),
        "dataset_fingerprint": hashlib.sha256(json.dumps(audit["records"], sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "protocol": {
            "registration_selection": "First correct record by normalized relative path; first 10 seconds; one attempt; no retry using other records or evaluation outcomes.",
            "record_separation": "Registration excluded in full. Resolved paths, raw SHA-256, and sensor-array SHA-256 must differ from every same-user evaluation record.",
            "threshold_selection": "Unmodified implementation defaults; no fitting or adjustment using any replay outcome.",
            "baseline_assumption": "Zero offset: matching empty-seat baseline, chair, placement, and physical guidance confirmation are unavailable in historical recordings.",
            "occupancy_assumption": "Seated=True for labeled historical records; live occupancy, disconnect, and true onset/return validation require prospective sensors.",
            "scope": "All eligible historical subjects" if subjects is None else "Explicitly selected subjects: " + ", ".join(sorted(subjects)),
            "model_typing": "Unavailable during this independent personal-gate audit; confirmed incorrect state can issue a general correction.",
        },
        "calibration_config": asdict(CalibrationConfig()), "guidance_config": asdict(GuidanceConfig()),
        "registration_subject_count": len(plans),
        "successful_registration_count": len(profiles),
        "failed_registration_count": len(plans) - len(profiles),
        "registration_outcomes": outcomes,
        "profiles": profiles,
        "modes": {mode: {"settings": AlertSettings.for_mode(mode).as_dict(),
                         "summary": summarize_records(results), "records": results}
                  for mode, results in mode_records.items()},
        "standalone_model_performance": {"status": "reported_separately_in_model_comparison",
                                         "reason": "No classification-model score is inferred from personal-gate outcomes."},
        "actual_sensor_validation": {"status": "not_performed", "ports_observed": ports,
                                     "reason": "No supervised seated-user registration, incorrect-posture, empty-seat, disconnection, and return sequence has been collected."},
        "limitations": [
            "Historical labels cannot establish supervised correct reference posture, matching chair/placement, or empty-seat baseline.",
            "No actual sensors were opened. These results are CSV replay evidence only.",
            "Short separate posture clips cannot validate continuous deployment alert delay, repeat alerts, true onset, or recovery latency.",
            "False alerts/hour is a rate over short independent clips; each clip starts a new timer, so this does not estimate long-session alert burden.",
            "Registration failures are retained as failures; thresholds remain provisional and are not selected from evaluation outcomes.",
        ],
    }
    return report, audit


def report_markdown(report: dict, audit: dict) -> str:
    def percent(value):
        return "Unavailable" if value is None else f"{value:.2%}"

    lines = ["# Personal Reference CSV Replay Evaluation", "",
             "This report evaluates the implemented personal gate against separate historical recordings. It is not actual sensor validation.", "",
             f"Eligible data: {audit['eligible_record_count']} recordings from {audit['subject_count']} subjects; {audit['excluded_record_count']} recordings excluded with reasons in dataset_audit.json.", "",
             f"Registration: {report['successful_registration_count']} succeeded; {report['failed_registration_count']} failed. The first correct recording per subject was reserved in full, and only its first 10 seconds were used. No alternative recording was tried after a failure.", "",
             "The replay uses unmodified provisional thresholds and a zero baseline because historical clips lack a matched empty-seat reference. Saved profiles are marked as retrospective evaluation artifacts and must not be treated as supervised live registrations.", "",
             "| Mode | Incorrect missed as correct | Correct classified incorrect | False alerts | False alerts/hour of correct clips |",
             "|---|---:|---:|---:|---:|"]
    for mode, data in report["modes"].items():
        summary = data["summary"]
        rate = summary["false_alerts_per_correct_hour"]
        lines.append(f"| {mode} | {percent(summary['incorrect_missed_as_correct_rate'])} | {percent(summary['correct_classified_incorrect_rate'])} | {summary['false_alert_count']} | {'Unavailable' if rate is None else f'{rate:.2f}'} |")
    lines.extend(["", "Rates use elapsed time, include all eligible clip exposure, and report uncertain/movement/insufficient states separately in evaluation.json. Timers reset between files; short clips cannot estimate the alert burden of a continuous long session.", "",
                  "Actual posture-onset-to-popup delay and actual correct-return-to-dismissal delay: unavailable. The recordings are approximately 20 seconds long and contain no annotated continuous transitions. A zero deployment popup count is not proof of a safe false-alert rate.", "",
                  "Classification-model metrics are reported separately in the model comparison. This audit does not use model predictions as ground truth or adjust personal thresholds from evaluation records.", "",
                  "## Registration outcomes", "", "| Subject | Outcome | Separate correct evaluation |", "|---|---|---|"])
    for outcome in report["registration_outcomes"]:
        status = outcome["status"] + (": " + outcome["reason"] if "reason" in outcome else "")
        lines.append(f"| {outcome['subject']} | {status} | {'Yes' if outcome['has_independent_correct_evaluation'] else 'No'} |")
    lines.extend(["", "## Actual sensor validation", "", "Not performed. Enumerated ports:", ""])
    lines.extend(f"- {port['device']} ({port['description']}; {port['hardware_id']})" for port in report["actual_sensor_validation"]["ports_observed"])
    lines.extend(["", "## Limitations", ""])
    lines.extend("- " + limitation for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def compare_selected_model(report: dict, manifest_path: Path) -> dict:
    """Compare the locked model and personal gate on identical held-out windows.

    Registration records are excluded even though the original standalone model
    report also evaluated them. This supplemental subset cannot select a model.
    """
    from bisect import bisect_right
    import joblib
    from sklearn.metrics import accuracy_score, f1_score
    from src.inference_core import predict_probabilities
    from src.window_pipeline import WindowSpec, extract_window_features, segment_sensor_frame

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = (manifest_path.parent / manifest["artifact_file"]).resolve()
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest["artifact_sha256"]:
        raise ValueError("Selected model hash does not match its manifest")
    component = next(item for item in manifest["components"] if item["role"] == "evaluation")
    evidence_path = (manifest_path.parent / component["file"]).resolve()
    if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != component["sha256"]:
        raise ValueError("Selected evaluation hash does not match its manifest")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    parts = evidence["partitions"]
    final_subjects = set(parts["final_evaluation"]["subjects"])
    if (final_subjects & (set(parts["train"]["subjects"]) | set(parts["validation"]["subjects"]))
            or evidence["final_evaluation_used_for_selection"]):
        raise ValueError("Selected model does not have independent final subjects")
    final_ids = set(parts["final_evaluation"]["record_ids"])
    final_hashes = {item["sha256"] for item in evidence["source_records"] if item["record_id"] in final_ids}
    bundle = joblib.load(artifact)
    spec = WindowSpec(bundle["window_seconds"], bundle["sampling_rate"], bundle["min_tail_fraction"])
    rows = []
    mode = next(iter(report["modes"]))
    for record in report["modes"][mode]["records"]:
        if record["subject"] not in final_subjects:
            continue
        path = Path(report["data_root"]) / record["relative_path"]
        if record["sha256"] not in final_hashes or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Replay record does not match the held-out evaluation source")
        frame = read_sensor_csv(path)
        windows, window_info = segment_sensor_frame(frame, spec)
        predicted = predict_probabilities(bundle, extract_window_features(windows)).argmax(axis=1)
        transitions = record["transitions"]
        times = [transition["timestamp_s"] for transition in transitions]
        for window, prediction in zip(window_info.to_dict(orient="records"), predicted):
            endpoint = min(window["nominal_end_s"], record["duration_s"])
            state = transitions[max(0, bisect_right(times, endpoint) - 1)]["state"]
            rows.append({"subject": record["subject"], "relative_path": record["relative_path"],
                         "sha256": record["sha256"], "window_end_s": endpoint,
                         "ground_truth": record["ground_truth"], "model_posture": CLASS_NAMES[prediction],
                         "personal_gate_state": state})
    correct = [row for row in rows if row["ground_truth"] == CLASS_NAMES[0]]
    incorrect = [row for row in rows if row["ground_truth"] != CLASS_NAMES[0]]

    def rate(predicate, data):
        return sum(predicate(row) for row in data) / len(data) if data else None

    return {
        "evaluation_kind": "supplemental_locked_model_before_after_personal_gate",
        "manifest": str(manifest_path.resolve()), "model_sha256": manifest["artifact_sha256"],
        "model_selected_again": False, "personal_thresholds_fitted": False,
        "final_subjects": sorted(final_subjects), "window_count": len(rows),
        "record_count": len({row["sha256"] for row in rows}),
        "correct_window_count": len(correct), "incorrect_window_count": len(incorrect),
        "comparison_basis": "Same independent final-subject records and 3-second window endpoints. Reserved personal-registration recordings are excluded in full. Personal state is sampled from the continuous per-record replay; no threshold or model is reselected.",
        "standalone_model": {
            "accuracy": float(accuracy_score([row["ground_truth"] for row in rows], [row["model_posture"] for row in rows])) if rows else None,
            "macro_f1": float(f1_score([row["ground_truth"] for row in rows], [row["model_posture"] for row in rows], labels=CLASS_NAMES, average="macro", zero_division=0)) if rows else None,
            "incorrect_missed_as_correct_rate": rate(lambda row: row["model_posture"] == CLASS_NAMES[0], incorrect),
            "correct_classified_incorrect_rate": rate(lambda row: row["model_posture"] != CLASS_NAMES[0], correct),
        },
        "personal_gate": {
            "incorrect_missed_as_correct_rate": rate(lambda row: row["personal_gate_state"] == "correct", incorrect),
            "correct_classified_incorrect_rate": rate(lambda row: row["personal_gate_state"] == "incorrect", correct),
            "uncertain_rate": rate(lambda row: row["personal_gate_state"] not in {"correct", "incorrect"}, rows),
        },
        "limitations": report["limitations"], "windows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/integrated_personalization")
    parser.add_argument("--subject", action="append", help="Optional explicit subject subset; repeat for each subject.")
    parser.add_argument("--selected-manifest", type=Path, help="Optional locked selected model for a separate held-out before/after comparison.")
    parser.add_argument("--allow-retrospective", action="store_true", help="Explicitly re-enable historical registration for this replay only.")
    args = parser.parse_args()
    subjects = {normalize_text(value) for value in args.subject} if args.subject else None
    report, audit = evaluate_personalization(args.data.resolve(), subjects=subjects, allow_retrospective=args.allow_retrospective)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(report_markdown(report, audit), encoding="utf-8")
    if args.selected_manifest:
        comparison = compare_selected_model(report, args.selected_manifest.resolve())
        (args.output / "selected_model_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    profiles = args.output / "profiles"
    profiles.mkdir(exist_ok=True)
    for profile in report["profiles"]:
        name = hashlib.sha256(profile["profile"]["user_id"].encode()).hexdigest()[:16]
        (profiles / f"{name}.evaluation.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "successful_registration_count": report["successful_registration_count"],
                      "failed_registration_count": report["failed_registration_count"],
                      "modes": {mode: data["summary"] for mode, data in report["modes"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
