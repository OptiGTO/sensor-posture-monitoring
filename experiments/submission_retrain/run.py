#!/usr/bin/env python3
"""Submission retrain: RBF-SVM 3 s through the app's own TrainingJob path.

Uses exactly the training-wizard pipeline (staging with hash-bound exclusions,
subject-disjoint train/validation/final split, LOSO selection, candidate
manifest, automatic runtime + deployment validation, pending approval). The
candidate is NOT approved or activated here; that remains a separate user step.

Record selection rule (fixed before training, not score based): all
training-eligible labeled records, minus subjects whose eligible records do not
cover all seven postures. ``data/`` is read only; its file hashes are compared
before and after the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.app_services import DatasetService, TrainingJob, TrainingWizardState  # noqa: E402
from src.experiment_workflow import build_configuration  # noqa: E402
from src.posture_core import CLASS_NAMES  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def data_snapshot(root: Path) -> dict[str, list]:
    data = root / "data"
    return {path.relative_to(data).as_posix(): [path.stat().st_size, path.stat().st_mtime_ns, sha256(path)]
            for path in sorted(data.rglob("*")) if path.is_file()}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def select_records(service: DatasetService):
    eligible = [record for record in service.records() if service.is_training_eligible(record)]
    postures = defaultdict(set)
    for record in eligible:
        postures[unicodedata.normalize("NFC", record.subject)].add(record.posture)
    incomplete = sorted(subject for subject, seen in postures.items() if seen != set(CLASS_NAMES))
    selected = [record for record in eligible
                if unicodedata.normalize("NFC", record.subject) not in incomplete]
    excluded = [{"source_file": str(record.csv_path.relative_to(ROOT / "data")),
                 "subject": unicodedata.normalize("NFC", record.subject), "label": record.posture,
                 "reason": "subject_does_not_cover_all_seven_postures"}
                for record in eligible if unicodedata.normalize("NFC", record.subject) in incomplete]
    return eligible, selected, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New empty research/results run directory")
    parser.add_argument("--seed", type=int, default=TrainingWizardState.seed)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now().astimezone()
    before = data_snapshot(ROOT)

    service = DatasetService(ROOT, data_directory=ROOT / "data")
    eligible, selected, excluded = select_records(service)
    state = TrainingWizardState()
    state.experiment_name = "제출용 RBF-SVM 3초 재학습 (32명)"
    state.training_method = "standard"
    state.configure((3,), args.seed, ("rbf_svm",))
    state.selection_metric = "loso"
    state.remember_records(selected)
    config = build_configuration(state)
    job = TrainingJob(ROOT, selected, config["seed"], windows=(3,), models=("rbf_svm",), state_root=ROOT,
                      experiment_config=config, training_method="standard", selection_metric="loso")
    log_path = output / "training.log"
    finished = None
    with log_path.open("w", encoding="utf-8") as log:
        job.start()
        while finished is None:
            while not job.events.empty():
                kind, message = job.events.get()
                log.write(f"[{kind}] {message}\n")
                log.flush()
                if kind == "finished":
                    finished = message
            time.sleep(0.5)

    after = data_snapshot(ROOT)
    run_dir = job.run_dir
    manifest = json.loads((run_dir / "training.manifest.json").read_text(encoding="utf-8"))
    comparison = json.loads(job.comparison_report_path.read_text(encoding="utf-8"))
    evaluation = json.loads((run_dir / "artifacts" / "window_3s_rbf_svm_evaluation.json").read_text(encoding="utf-8"))
    model_config = json.loads((run_dir / "artifacts" / "window_3s_rbf_svm_model_config.json").read_text(encoding="utf-8"))
    loso = evaluation.get("loso_metrics", {})
    result = {
        "purpose": "제출용 앱 모델: 32명 데이터로 앱 학습 마법사 경로(RBF-SVM 3초, LOSO 선택) 재학습",
        "kind": "CSV training/evaluation (no physical sensor)",
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "source_revision": git("rev-parse", "HEAD"),
        "dirty_files": git("status", "--porcelain").splitlines(),
        "command": [sys.executable, *sys.argv],
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        **comparison.get("environment", {})},
        "seed": config["seed"],
        "selection_rule": "training-eligible labeled records; exclude subjects without all seven postures",
        "eligible_records": len(eligible),
        "eligible_subjects": len({unicodedata.normalize("NFC", r.subject) for r in eligible}),
        "selected_records": len(selected),
        "selected_subjects": len({unicodedata.normalize("NFC", r.subject) for r in selected}),
        "rule_excluded_records": excluded,
        "training_exclusions_file": "research/models/training_exclusions.json (applied by DatasetService)",
        "split_rule": "seeded subject shuffle; final_evaluation = first n//5, validation = next n//5, train = rest",
        "model_run": str(run_dir.relative_to(ROOT)),
        "training_status": manifest["status"],
        "exit_code": manifest["exit_code"],
        "validation_results": manifest["validation_results"],
        "candidate_manifests": [str(Path(p).relative_to(ROOT)) for p in manifest["candidate_manifests"]],
        "partitions": {name: {"subjects": len(part["subjects"]), "records": len(part["record_ids"]),
                              "windows": part["window_count"]} for name, part in evaluation["partitions"].items()},
        "final_evaluation_subjects": evaluation["partitions"]["final_evaluation"]["subjects"],
        "selected_params": evaluation["selected_params"],
        "metrics": {
            "validation_macro_f1": evaluation["validation_metrics"]["macro_f1"],
            "loso_development_macro_f1": loso.get("macro_f1"),
            "loso_development_subjects": loso.get("subjects"),
            "final_macro_f1": evaluation["metrics"]["macro_f1"],
            "final_accuracy": evaluation["metrics"].get("accuracy"),
            "final_per_class_f1": evaluation["metrics"].get("per_class_f1"),
        },
        "final_model_training_counts": {key: model_config.get(key) for key in (
            "training_subject_count", "training_record_count", "training_window_count")},
        "approval": "pending (not approved or activated by this script)",
        "data_unchanged": before == after,
        "data_files_checked": len(before),
        "not_verified": ["physical Arduino sensor stream", "rebuilt macOS app bundle"],
    }
    (output / "run.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("training_status", "model_run", "metrics", "partitions",
                                                   "final_model_training_counts", "data_unchanged")},
                     ensure_ascii=False, indent=2))
    if finished != "0" or before != after:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
