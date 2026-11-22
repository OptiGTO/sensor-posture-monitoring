"""Fit all currently eligible people and include the previously reserved records.

Explicit user override: registration recordings are included in supervised fit.
Resubstitution scores are NOT new-subject evaluation or learning-curve points.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

# Set native runtime thread limits before model imports.
from experiments.learning_curve.run import ROOT, SEED, MODELS, now, digest, read, protected_snapshot
import joblib
import numpy as np
from src.app_services import DatasetService
from src.model_comparison import build_model, scores, write_json
from src.posture_core import CLASS_NAMES, normalize_text, read_sensor_csv
from src.window_pipeline import WindowSpec, segment_sensor_frame, extract_window_features, aligned_predict_proba
from validate_personalization import audit_dataset


def assert_protected(out):
    if protected_snapshot() != read(out / "protected_before.json"):
        raise ValueError("Protected original inventory/hash/mtime changed")


def assert_registrations_included(old_protocol, staged_sources):
    indexed = {str(Path(r["source_file"]).resolve()): r for r in staged_sources}
    included = []
    for r in old_protocol["registrations_excluded"]:
        path = str((ROOT / "data" / r["relative_path"]).resolve())
        if path not in indexed:
            raise ValueError(f"Previously reserved registration missing: {r['relative_path']}")
        if indexed[path]["source_sha256"] != r["sha256"]:
            raise ValueError("Previously reserved registration content changed")
        included.append(indexed[path]["staged_file"])
    return included


def verify(out):
    p = read(out / "protocol.json")
    if digest(out / "protocol.json") != read(out / "protocol_lock.json")["sha256"]:
        raise ValueError("Full-cohort protocol changed")
    if digest(out / "data.npz") != p["cache_sha256"]:
        raise ValueError("Frozen features changed")
    for relative, expected in p["source_hashes"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Source changed after freeze: {relative}")
    for row in read(out / "training_data/staged_sources.json"):
        if digest(out / "training_data" / row["staged_file"]) != row["sha256"]:
            raise ValueError("Staged recording changed")
    return p


def prepare(out, previous, report):
    out.mkdir(parents=True, exist_ok=False)
    started = now()
    write_json(out / "protected_before.json", protected_snapshot())
    service = DatasetService(ROOT)
    indexed = service.records()
    selected = [r for r in indexed if service.is_training_eligible(r)]
    excluded = [{"file": str(r.csv_path.relative_to(ROOT)), "source_type": r.source_type,
                 "label_status": r.label_status, "training_role": r.training_role, "qc_status": r.qc_status}
                for r in indexed if not service.is_training_eligible(r)]
    service.stage_for_training(selected, out / "training_data")
    staged = read(out / "training_data/staged_sources.json")
    restored = assert_registrations_included(read(previous / "protocol.json"), staged)
    records, audit = audit_dataset(out / "training_data")
    if audit["excluded_records"] or audit["duplicate_raw_files"] or audit["duplicate_sensor_sequences"]:
        raise ValueError("Staged cohort failed quality/duplicate audit")
    if len(records) != len(selected):
        raise ValueError("Not all eligible records reached training")
    if {r.subject for r in records} != {normalize_text(r.subject) for r in selected}:
        raise ValueError("Not all eligible people reached training")
    windows, y, subjects, ids, starts = [], [], [], [], []
    for r in records:
        seq, meta = segment_sensor_frame(read_sensor_csv(r.path), WindowSpec(3))
        windows.extend(seq)
        y.extend([CLASS_NAMES.index(r.label)] * len(seq))
        subjects.extend([r.subject] * len(seq))
        ids.extend([r.relative_path] * len(seq))
        starts.extend(meta.start_s.tolist())
    data = {"X": extract_window_features(np.asarray(windows)), "y": np.asarray(y),
            "groups": np.asarray(subjects), "record_ids": np.asarray(ids), "start_s": np.asarray(starts)}
    if not np.isfinite(data["X"]).all() or set(y) != set(range(7)):
        raise ValueError("Invalid complete feature matrix")
    if not {normalize_text(r) for r in restored} <= {normalize_text(r) for r in ids}:
        raise ValueError("A reserved recording produced no training windows")
    np.savez_compressed(out / "data.npz", **data)
    write_json(out / "audit.json", {"staged_audit": audit, "excluded_indexed_records": excluded})
    # Snapshot all Python files needed to reconstruct the current dirty source.
    sources = [*ROOT.glob("*.py"), *(ROOT / "src").rglob("*.py"),
               Path(__file__), Path(__file__).with_name("run.py")]
    source_hashes = {}
    for source in sorted(set(sources)):
        relative = source.relative_to(ROOT)
        dest = out / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        source_hashes[str(relative)] = digest(source)
    correction_path = ROOT / "research/models/training_label_corrections.json"
    shutil.copy2(correction_path, out / "label_corrections.json")
    before = out / "source_snapshot/report_before.md"
    shutil.copy2(report, before)
    protocol = {"schema_version": 1, "started_at": started, "locked_at": now(), "seed": SEED,
        "purpose": "Requested full-person fit including previously reserved correct-posture recordings",
        "registration_override": "Explicit user request: include all 32 previously reserved files in supervised training",
        "models": MODELS, "class_names": list(CLASS_NAMES), "window_seconds": 3, "sampling_rate_hz": 12,
        "stride_seconds": 3, "feature_dimension": data["X"].shape[1],
        "cohort": {"subjects": len(set(subjects)), "subject_ids": sorted(set(subjects)),
                   "records": len(records), "windows": len(y), "raw_rows": audit["row_count"],
                   "class_window_counts": np.bincount(y, minlength=7).tolist(),
                   "class_record_counts": {c: sum(r.label == c for r in records) for c in CLASS_NAMES},
                   "restored_registration_records": len(restored), "restored_registration_paths": restored,
                   "corrected_label_records": sum(r["label_correction"] is not None for r in staged)},
        "partition": "All rows used for training; no held-out validation or test in this run",
        "score_kind": "training_resubstitution_only_not_generalization",
        "inference": "predict_proba argmax for all three models; SVM internal calibration uses training rows",
        "candidate_approval": False, "previous_run": str(previous), "report": str(report),
        "report_before_sha256": digest(report), "previous_summary_sha256": digest(previous / "summary.json"),
        "cache_sha256": digest(out / "data.npz"), "source_hashes": source_hashes,
        "correction_manifest_sha256": digest(correction_path), "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "command": [sys.executable, *sys.argv], "sensor_acquisition_time": "Not inferred from mtime; canonical sidecars remain authoritative",
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "packages": {k: version(k) for k in ["numpy", "scikit-learn", "xgboost", "joblib"]}}}
    write_json(out / "protocol.json", protocol)
    write_json(out / "protocol_lock.json", {"sha256": digest(out / "protocol.json"), "locked_at": now()})
    assert_protected(out)
    print("PREPARED", {k:v for k,v in protocol["cohort"].items() if k not in ["subject_ids","restored_registration_paths"]}, flush=True)


def worker(out, family):
    p = verify(out)
    data = np.load(out / "data.npz", allow_pickle=False)
    model = build_model(family, p["models"][family], p["seed"])
    fit_args = {}
    if family == "xgboost":
        from sklearn.utils.class_weight import compute_sample_weight
        fit_args["model__sample_weight"] = compute_sample_weight("balanced", data["y"])
    started, tick = now(), time.perf_counter()
    model.fit(data["X"], data["y"], **fit_args)
    elapsed = time.perf_counter() - tick
    np.testing.assert_allclose(model.named_steps["scale"].mean_, data["X"].mean(axis=0, dtype=np.float64), rtol=1e-10, atol=1e-10)
    pred = aligned_predict_proba(model, data["X"], 7).argmax(1)
    dest = out / "models" / family
    dest.mkdir(parents=True)
    joblib.dump({"model": model, "family": family, "label_names": list(CLASS_NAMES),
                 "params": p["models"][family], "protocol_sha256": digest(out / "protocol.json"),
                 "training_subjects": p["cohort"]["subject_ids"], "registration_records_included": True,
                 "approved": False, "score_kind": p["score_kind"]}, dest / "model.joblib", compress=3)
    loaded = joblib.load(dest / "model.joblib")["model"]
    np.testing.assert_array_equal(aligned_predict_proba(loaded, data["X"], 7).argmax(1), pred)
    np.savez_compressed(dest / "predictions.npz", truth=data["y"], prediction=pred)
    write_json(dest / "metrics.json", {"family": family, "score_kind": p["score_kind"],
        "training": scores(data["y"], pred), "heldout": None, "fit_seconds": elapsed,
        "started_at": started, "finished_at": now(), "saved_model_prediction_parity": True,
        "model_sha256": digest(dest / "model.joblib"), "predictions_sha256": digest(dest / "predictions.npz")})
    print(f"{family}: full fit complete in {elapsed:.1f}s; training F1={scores(data['y'], pred)['macro_f1']:.4f}", flush=True)


def finish(out):
    p = verify(out)
    assert_protected(out)
    metrics = {family: read(out / "models" / family / "metrics.json") for family in MODELS}
    for family, m in metrics.items():
        for filename, key in [("model.joblib", "model_sha256"), ("predictions.npz", "predictions_sha256")]:
            if digest(out / "models" / family / filename) != m[key]:
                raise ValueError("Saved model/predictions changed")
    write_json(out / "summary.json", {"cohort": p["cohort"], "metrics": metrics, "score_kind": p["score_kind"],
        "finished_at": now(), "raw_data_and_existing_models_unchanged": True})
    report = Path(p["report"])
    if digest(report) != p["report_before_sha256"]:
        raise ValueError("Report changed during fitting; preserve changes and merge explicitly")
    names = {"xgboost": "XGBoost", "rbf_svm": "RBF SVM", "random_forest": "Random Forest"}
    c = p["cohort"]
    rel = "../../research/results/" + out.name
    section = ["", "## 7. 추가 실험: 전체 인원·등록용 정자세 포함 학습", "",
        f"사용자 추가 요청에 따라 현재 전체 **{c['subjects']}명, {c['records']}개 기록, {c['windows']:,}개 3초 구간**으로 세 모델을 각각 다시 학습했다. 각 모델이 전체 인원을 모두 학습했으며, 이번 실행에는 학습/평가 인원 분할을 적용하지 않았다.", "",
        f"- 실행 시작: {p['started_at']}; 학습 완료: {max(m['finished_at'] for m in metrics.values())}",
        f"- 기존에 등록용으로 제외한 첫 정자세 기록 **{c['restored_registration_records']}개를 모두 포함**했다.",
        f"- 현재 라벨 교정 manifest를 원본의 복사본에 적용하여, 이전 실험에서 라벨 오류로 제외한 **{c['corrected_label_records']}개 기록도 포함**했다. 원본 CSV는 변경하지 않았다.",
        "- 이전 699개 + 등록용 정자세 32개 + 교정 기록 4개 = 735개. 측정 전용 무라벨 기록 및 명시적으로 학습 제외된 기록은 계속 제외했다.",
        "- 3초/12 Hz/596개 특징과 세 모델의 하이퍼파라미터는 앞선 실험과 동일하다. 전체 집합으로 각 scaler를 새로 fit했으며 모델은 연구용으로 저장했다.", "",
        "**아래 점수는 학습에 사용한 동일 데이터로 재평가한 값이다. 미참여자 성능이나 독립 테스트 성능이 아니다.** 기존 25명 교차검증 F1 약 66%에서 아래 수치로 일반화 성능이 향상됐다는 의미로 해석할 수 없다.", "",
        "| 모델 | 실제 학습 인원 | 실제 기록 수 | 학습 Accuracy | 학습 Macro F1 | 학습 시간 | 저장 모델 재예측 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for family, m in metrics.items():
        section.append(f"| {names[family]} | {c['subjects']} | {c['records']} | {m['training']['accuracy']*100:.2f}% | {m['training']['macro_f1']*100:.2f}% | {m['fit_seconds']:.1f}초 | 전체 구간 일치 |")
    section += ["", "### 등록용 기록 포함 여부에 따른 데이터 구성", "",
        "| 자세 | 기록 수 | 3초 학습 구간 수 |", "| --- | ---: | ---: |"]
    for i, label in enumerate(CLASS_NAMES):
        section.append(f"| {label} | {c['class_record_counts'][label]} | {c['class_window_counts'][i]} |")
    section += ["", "### 기존 학습 곡선과의 관계", "",
        "앞선 4–25명 학습 곡선과 32/40/48/60명 전망은 등록용 기록을 제외한 이전 프로토콜에 대한 결과로 유지한다. 이번 전체 인원 학습점은 평가 방식과 데이터 구성이 다르므로 기존 미참여자 성능 곡선에 연결하거나 외삽 함수를 다시 맞추는 데 사용하지 않았다. 등록용 기록까지 포함한 데이터의 미참여자 성능을 구하려면 동일 사람 단위 교차검증을 별도로 실행하거나 새 피험자 데이터로 평가해야 한다.", "",
        f"[전체 학습 결과 JSON]({rel}/summary.json) · [고정 프로토콜]({rel}/protocol.json) · [학습 원본·교정 대응표]({rel}/training_data/staged_sources.json)", ""]
    for family in MODELS:
        section.append(f"- [{names[family]} 전체 인원 학습 모델]({rel}/models/{family}/model.joblib)")
    section += ["", "재실행:", "", "```sh",
        ".venv/bin/python -m experiments.learning_curve.full_cohort --output research/results/<새로운-실행-폴더> --previous research/results/2026-09-21_144323_learning_curve --report docs/reports/2026-09-21_144323_learning_curve_report.md", "```", ""]
    text = report.read_text()
    first, rest = text.split("\n", 1)
    notice = "\n> **추가 결과:** 현재 전체 32명과 등록용 정자세 기록을 포함한 학습 결과를 7장에 추가했습니다. 1–6장은 이전 교차검증 실험입니다.\n"
    report.write_text(first + "\n" + notice + rest + "\n".join(section))
    write_json(out / "report_update.json", {"updated_at": now(), "report": str(report),
        "before_sha256": p["report_before_sha256"], "after_sha256": digest(report),
        "section": "7. 추가 실험: 전체 인원·등록용 정자세 포함 학습"})
    print("REPORT UPDATED", report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", choices=list(MODELS))
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        worker(out, args.worker)
        return
    if args.previous is None or args.report is None:
        parser.error("--previous and --report are required")
    prepare(out, args.previous.resolve(), args.report.resolve())
    def launch(family):
        with (out / f"{family}.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "experiments.learning_curve.full_cohort", "--output", str(out),
                            "--worker", family], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(launch, MODELS))
    finish(out)


if __name__ == "__main__":
    main()
