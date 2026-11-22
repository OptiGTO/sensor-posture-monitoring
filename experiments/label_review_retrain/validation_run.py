"""Single validation split (app 'single validation' procedure) after review revision 2.

Split: the app/submission seed 20260826 (train 20 / validation 6 / final 6 subjects).
Per condition x family x model seed: every app grid candidate (MODEL_CANDIDATES) is
fitted on the condition's train-subject fit rows; validation Macro F1 (ties: worst-class
F1) selects the parameters; the selected train-only model is scored once on final.
No LOSO.

Conditions change only fitted train-subject rows (validation/final rows never change):
  baseline               724 records (the submission model's data)
  drop_sensor5           the 5 records added to training_exclusions.json on 2026-09-30
  drop_delete            the 16 rev2 'delete' records
  drop_invalid_segments  rev2 with reviewer intervals (primary)
  drop_delete_mismatch   16 'delete' + 10 'posture_mismatch'

Selection rows: 'all' validation rows (primary; the app procedure) and, from the same
fitted candidates, 'valid' rows (sensitivity). Scores are reported on validation and
final for the sets all / valid / valid_no_mismatch / review_ineligible.

Usage (new output folder only):
  .venv/bin/python -B -W ignore -m experiments.label_review_retrain.validation_run --output research/results/<ts>_label_review_validation
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.label_review_retrain.analysis import (  # noqa: E402
    condition_fit_masks, decide, evaluation_masks, macro_f1, paired_subject_bootstrap, per_class_f1,
    present_macro_f1, review_masks)
from experiments.runtime_vs_research.run import now, protected_state, read, sha, write  # noqa: E402

REVIEW_DECISIONS = "research/results/2026-09-28_130815_label_review_all_wrong/review_decisions_2026-09-30_rev2.json"
REVIEW_DECISIONS_SHA256 = "978ef2332e6b31a9771a33f302ee8412e18b4348a2762f9f1fa47bd0492c2a34"
EXCLUSIONS = "research/models/training_exclusions.json"
ORIGIN_PROTOCOL = "research/results/2026-09-22_015526_iterative_008/protocol.json"
SUBMISSION_EVALUATION = "research/models/models/20260929_221617_db0a4f57/artifacts/window_3s_rbf_svm_evaluation.json"
EXPECTED_FINGERPRINT = "00ed992fa24c6e8265712d7c761b2e0d14f8b78750f09ecc7deed25c71aa3cc4"
SPLIT_SEED = 20260826
SEEDS = [20260826, 20260827, 20260828]
WINDOW_SECONDS = 3
FAMILIES = ("rbf_svm", "logistic_regression", "random_forest", "extra_trees")
CONDITIONS = ("baseline", "drop_sensor5", "drop_delete", "drop_invalid_segments", "drop_delete_mismatch")
SELECTIONS = ("all", "valid")
SETS = ("all", "valid", "valid_no_mismatch", "review_ineligible")
PRIMARY = {"model": "rbf_svm", "condition": "drop_invalid_segments", "selection": "all",
           "partition": "final_evaluation", "evaluation_set": "valid"}
CRITERIA = {"minimum_mean_delta": 0.01, "maximum_per_class_drop": 0.05,
            "non_inferiority_mean_margin": 0.005, "non_inferiority_bootstrap_margin": 0.015,
            "bootstrap_replicates": 10000, "bootstrap_seed": 20260930}
SOURCES = ["experiments/label_review_retrain/validation_run.py", "experiments/label_review_retrain/analysis.py",
           "experiments/label_review_retrain/encode_review.py", "experiments/runtime_vs_research/run.py",
           "experiments/runtime_vs_research/analysis.py", "experiments/submission_retrain/run.py",
           "src/model_comparison.py", "src/training_methods.py", "src/window_pipeline.py",
           "src/posture_core.py", "src/app_services.py", "src/training_exclusions.py"]


def stage_cohort(target: Path) -> tuple[Path, list[str], dict]:
    """Stage the 724-record cohort: app staging of the 719 eligible records plus the 5
    records excluded on 2026-09-30, copied the same way (no sidecars). The fingerprint
    check in prepare() proves the stage equals the pre-exclusion cohort."""
    from experiments.submission_retrain.run import select_records
    from src.app_services import DatasetService
    from src.posture_core import normalize_text
    service = DatasetService(ROOT, data_directory=ROOT / "data")
    eligible, selected, rule_excluded = select_records(service)
    stage = service.stage_for_training(selected, target)
    manifest = read(ROOT / EXCLUSIONS)
    today = [e for e in manifest["exclusions"] if e["requested_at"].startswith("2026-09-30")]
    if len(today) != 5:
        raise ValueError("Expected exactly five exclusions added on 2026-09-30")
    added = []
    for entry in today:
        nominal = ROOT / "data" / entry["source_file"]
        # Use the on-disk file name (app staging copies csv_path.name from a directory listing;
        # decomposed and composed Hangul sort differently).
        matches = [p for p in nominal.parent.iterdir() if normalize_text(p.name) == normalize_text(nominal.name)]
        if len(matches) != 1:
            raise ValueError(f"Cannot resolve {entry['source_file']}")
        source = matches[0]
        if source.with_suffix(".manifest.json").exists() or sha(source) != entry["source_sha256"]:
            raise ValueError(f"Unexpected sidecar or hash for {entry['source_file']}")
        destination = stage / normalize_text(source.parent.name) / source.name
        if destination.exists():
            raise ValueError(f"Already staged: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        added.append(entry["source_sha256"])
    info = {"eligible_records": len(eligible), "app_selected_records": len(selected),
            "rule_excluded": rule_excluded, "re_added_exclusions": today,
            "exclusions_manifest_sha256": sha(ROOT / EXCLUSIONS)}
    return stage, added, info


def prepare(out: Path, stage: Path, sensor5: list[str], decisions: list[dict]) -> dict:
    from src.model_comparison import _dataset
    from src.posture_core import CLASS_NAMES
    from src.signal_preprocessing import SignalPreprocessing
    data = _dataset(stage, WINDOW_SECONDS, SPLIT_SEED, None, SignalPreprocessing())
    if data["fingerprint"] != EXPECTED_FINGERPRINT:
        raise ValueError("Staged cohort differs from the 724-record submission cohort")
    submission = read(ROOT / SUBMISSION_EVALUATION)
    partitions = {name: data["partitions"][name]["subjects"] for name in ("train", "validation", "final_evaluation")}
    if partitions != {k: sorted(v["subjects"]) for k, v in submission["partitions"].items()}:
        raise ValueError("Split differs from the submission model's split")
    meta, records = data["meta"], data["source_records"]
    by_sha = {r["sha256"]: r for r in records}
    review = {}
    for item in decisions:
        record = by_sha[item["source_sha256"]]
        if (record["subject"], record["label"]) != (item["subject"], item["label"]):
            raise ValueError(f"{item['review_id']}: subject/label mismatch")
        review[record["record_id"]] = item
    sensor_ids = {by_sha[h]["record_id"] for h in sensor5}
    if any(review[r]["decision"] != "delete" for r in sensor_ids):
        raise ValueError("Every excluded sensor record must be a rev2 'delete'")
    record_ids = meta["record_id"].to_numpy(dtype=str)
    groups = meta["subject"].to_numpy(dtype=str)
    masks = review_masks(record_ids, meta["start_s"].to_numpy(dtype=float), review)
    fit = condition_fit_masks(masks)
    fit["drop_sensor5"] = ~np.isin(record_ids, sorted(sensor_ids))
    train = data["masks"]["train"]
    fit = {name: fit[name] & train for name in CONDITIONS}
    shown = set().union(*(set(p["validation"]) for p in read(ROOT / ORIGIN_PROTOCOL)["partitions"]))
    sets = evaluation_masks(masks)
    sets.pop("unreviewed")
    sets["review_ineligible"] = ~np.isin(groups, sorted(shown))
    if np.any(sets["review_ineligible"] & masks["reviewed"]):
        raise ValueError("A review-ineligible subject has reviewed records")
    y = data["y"]
    for name, mask in fit.items():
        if set(y[mask].tolist()) != set(range(len(CLASS_NAMES))):
            raise ValueError(f"{name}: training rows lack a class")
    cache = {"X": data["X"], "y": y, "groups": groups, "records": record_ids,
             **{f"part__{k}": v for k, v in data["masks"].items()},
             **{f"fit__{k}": v for k, v in fit.items()}, **{f"set__{k}": v for k, v in sets.items()}}
    cache_dir = out / "cache"
    cache_dir.mkdir()
    for key, value in cache.items():
        np.save(cache_dir / f"{key}.npy", value, allow_pickle=False)
    decision_of = np.array([review[r]["decision"] if r in review else "unreviewed" for r in record_ids])
    count = lambda part: {d: {"records": len(set(record_ids[part & (decision_of == d)])),
                              "windows": int(np.sum(part & (decision_of == d)))}
                          for d in ("delete", "hold", "posture_mismatch", "unreviewed")}
    return {
        "dataset_fingerprint": data["fingerprint"], "source_records": records, "partitions": partitions,
        "records": len(records), "windows": int(len(y)),
        "review_by_partition": {p: count(data["masks"][p]) for p in partitions},
        "fit_windows": {k: int(v.sum()) for k, v in fit.items()},
        "fit_records": {k: len(set(record_ids[v])) for k, v in fit.items()},
        "evaluation_windows": {p: {s: int(np.sum(data["masks"][p] & m)) for s, m in sets.items()}
                               for p in ("validation", "final_evaluation")},
        "evaluation_subjects": {p: {s: sorted(set(groups[data["masks"][p] & m])) for s, m in sets.items()}
                                for p in ("validation", "final_evaluation")},
        "sensor5_record_ids": sorted(sensor_ids),
        "cache_sha256": {key: sha(cache_dir / f"{key}.npy") for key in cache},
    }


_CACHE: dict | None = None


def _cache(directory: str) -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in Path(directory).glob("*.npy")}
    return _CACHE


def task_job(job: dict) -> dict:
    """One condition x family x seed: app grid on validation, then final for the selected models only."""
    from src.model_comparison import scores
    from src.posture_core import CLASS_NAMES
    from src.training_methods import build_method_model, fit_method_model, method_candidates
    from src.window_pipeline import aligned_predict_proba
    cache = _cache(job["cache"])
    condition, family, seed = job["condition"], job["family"], job["seed"]
    X, y, groups, records = (np.asarray(cache[k]) for k in ("X", "y", "groups", "records"))
    train = np.flatnonzero(np.asarray(cache[f"fit__{condition}"]))
    validation = np.flatnonzero(np.asarray(cache["part__validation"]))
    final = np.flatnonzero(np.asarray(cache["part__final_evaluation"]))
    valid_in_validation = np.asarray(cache["set__valid"])[validation]
    k = len(CLASS_NAMES)
    candidates, best = [], {name: None for name in SELECTIONS}
    tick = time.perf_counter()
    for params in method_candidates(family, "standard"):
        model = build_method_model(family, params, seed, "standard")
        # Identical call to src.model_comparison.train_comparison (train rows, validation_data).
        fit_method_model(model, family, "standard", X[train], y[train], groups=groups[train],
                         record_ids=records[train], validation_data=(X[validation], y[validation]))
        probability = aligned_predict_proba(model, X[validation], k)
        prediction = probability.argmax(axis=1)
        row = {"params": dict(params)}
        for name, rows in (("all", slice(None)), ("valid", valid_in_validation)):
            result = scores(y[validation][rows], prediction[rows])
            row[f"validation_{name}_macro_f1"] = result["macro_f1"]
            row[f"validation_{name}_worst_class_f1"] = result["worst_class_f1"]
            key = (result["macro_f1"], result["worst_class_f1"])
            if best[name] is None or key > best[name][0]:  # first candidate wins exact ties, as in the app
                best[name] = (key, model, dict(params), probability)
        candidates.append(row)
    selected = {}
    for name in SELECTIONS:
        _, model, params, probability = best[name]
        final_probability = aligned_predict_proba(model, X[final], k)
        for values in (probability, final_probability):
            if not np.isfinite(values).all() or not np.allclose(values.sum(axis=1), 1.0):
                raise ValueError("Invalid probability contract")
        selected[name] = {"params": params, "validation_probability": probability.tolist(),
                          "final_probability": final_probability.tolist()}
    return {"condition": condition, "family": family, "seed": seed, "candidates": candidates,
            "selected": selected, "training_windows": int(len(train)), "seconds": time.perf_counter() - tick}


def block(truth, prediction, groups) -> dict:
    return {"macro_f1": macro_f1(truth, prediction), "accuracy": float((truth == prediction).mean()),
            "worst_class_f1": float(per_class_f1(truth, prediction).min()), "windows": int(len(truth)),
            "per_class_f1": per_class_f1(truth, prediction).tolist(),
            "subject_present_macro_f1": {s: present_macro_f1(truth[groups == s], prediction[groups == s])
                                         for s in sorted(set(groups.tolist()))}}


def analyse(cache: dict, results: list[dict], seeds: list[int], families) -> dict:
    y, groups = np.asarray(cache["y"]), np.asarray(cache["groups"])
    rows = {p: np.flatnonzero(np.asarray(cache[f"part__{p}"])) for p in ("validation", "final_evaluation")}
    sets = {s: np.asarray(cache[f"set__{s}"]) for s in SETS}
    predictions, metrics = {}, {}
    for r in results:
        for selection, chosen in r["selected"].items():
            key = (r["family"], r["condition"], r["seed"], selection)
            entry = {"params": chosen["params"]}
            for part, name in (("validation", "validation_probability"), ("final_evaluation", "final_probability")):
                prediction = np.asarray(chosen[name]).argmax(axis=1)
                predictions[key + (part,)] = prediction
                index = rows[part]
                entry[part] = {s: block(y[index][m[index]], prediction[m[index]], groups[index][m[index]])
                               for s, m in sets.items()}
            metrics[key] = entry
    comparisons = {}
    for family in families:
        for condition in CONDITIONS[1:]:
            for selection in SELECTIONS:
                for part in ("validation", "final_evaluation"):
                    for s in SETS:
                        base = [metrics[family, "baseline", seed, selection][part][s] for seed in seeds]
                        arm = [metrics[family, condition, seed, selection][part][s] for seed in seeds]
                        index = rows[part][sets[s][rows[part]]]
                        keep = sets[s][rows[part]]
                        bootstrap = paired_subject_bootstrap(
                            y[index], predictions[family, "baseline", seeds[0], selection, part][keep],
                            predictions[family, condition, seeds[0], selection, part][keep], groups[index],
                            replicates=CRITERIA["bootstrap_replicates"], seed=CRITERIA["bootstrap_seed"])
                        decision = decide([b["macro_f1"] for b in base], [a["macro_f1"] for a in arm],
                                          [b["per_class_f1"] for b in base], [a["per_class_f1"] for a in arm],
                                          bootstrap["lower_2_5"], CRITERIA)
                        people = sorted(base[0]["subject_present_macro_f1"])
                        delta = [np.mean([a["subject_present_macro_f1"][p] - b["subject_present_macro_f1"][p]
                                          for a, b in zip(arm, base)]) for p in people]
                        comparisons[f"{family}:{condition}:{selection}:{part}:{s}"] = {
                            **decision, "bootstrap": bootstrap,
                            "subjects_improved": int(sum(d > 1e-12 for d in delta)),
                            "subjects_worse": int(sum(d < -1e-12 for d in delta))}
    winners = {}
    for condition in CONDITIONS:
        for seed in seeds:
            ranked = max(families, key=lambda f: (metrics[f, condition, seed, "all"]["validation"]["all"]["macro_f1"],
                                                  metrics[f, condition, seed, "all"]["validation"]["all"]["worst_class_f1"]))
            winners[f"{condition}:{seed}"] = {"family": ranked, "params": metrics[ranked, condition, seed, "all"]["params"],
                                              "validation_all": metrics[ranked, condition, seed, "all"]["validation"]["all"]["macro_f1"],
                                              "final_all": metrics[ranked, condition, seed, "all"]["final_evaluation"]["all"]["macro_f1"],
                                              "final_valid": metrics[ranked, condition, seed, "all"]["final_evaluation"]["valid"]["macro_f1"]}
    return {"metrics": {":".join(map(str, k)): v for k, v in metrics.items()}, "comparisons": comparisons,
            "app_winner_by_validation": winners}, predictions


def reproduction(summary: dict) -> dict:
    """Baseline RBF-SVM, seed 20260826, app selection must equal the submission model's evaluation."""
    submission = read(ROOT / SUBMISSION_EVALUATION)
    mine = summary["metrics"][f"rbf_svm:baseline:{SEEDS[0]}:all"]
    checks = {"params": mine["params"] == submission["selected_params"],
              "validation_macro_f1": abs(mine["validation"]["all"]["macro_f1"] - submission["validation_metrics"]["macro_f1"]) < 1e-12,
              "final_macro_f1": abs(mine["final_evaluation"]["all"]["macro_f1"] - submission["metrics"]["macro_f1"]) < 1e-12}
    return {"checks": checks, "identical": all(checks.values()), "submission_params": submission["selected_params"],
            "submission_validation_macro_f1": submission["validation_metrics"]["macro_f1"],
            "submission_final_macro_f1": submission["metrics"]["macro_f1"],
            "this_run": {"params": mine["params"], "validation_macro_f1": mine["validation"]["all"]["macro_f1"],
                         "final_macro_f1": mine["final_evaluation"]["all"]["macro_f1"]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--smoke", action="store_true", help="plumbing check: LR + ET, one seed; output outside research/")
    args = parser.parse_args()
    out = Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output
    if out.exists():
        raise SystemExit(f"Output exists; use a new folder: {out}")
    if args.smoke == out.resolve().is_relative_to((ROOT / "research").resolve()):
        raise SystemExit("Smoke runs must write outside research/; real runs under research/results/")
    out.mkdir(parents=True)
    started = now()
    os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    before = protected_state()
    write(out / "protected_before.json", before)
    for relative in SOURCES:
        target = out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    decisions_path = ROOT / REVIEW_DECISIONS
    if sha(decisions_path) != REVIEW_DECISIONS_SHA256:
        raise SystemExit("Review decision file differs from the encoded rev2")
    review = read(decisions_path)
    stage_parent = Path(tempfile.mkdtemp(prefix="label_review_validation_stage_"))
    stage, sensor5, staging = stage_cohort(stage_parent / "training_data")
    (out / "staging").mkdir()
    for name in ("staged_sources.json", "staging_summary.json"):
        shutil.copy2(stage / name, out / "staging" / name)
    cohort = prepare(out, stage, sensor5, review["records"])
    cohort["staging"] = staging
    write(out / "cohort.json", cohort)

    families = ["logistic_regression", "extra_trees"] if args.smoke else list(FAMILIES)
    seeds = SEEDS[:1] if args.smoke else SEEDS
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    from src.model_comparison import MODEL_CANDIDATES
    protocol = {
        "schema_version": 1, "started_at": started, "smoke": args.smoke,
        "question": "Single validation split (no LOSO): does applying review rev2 to training improve Macro F1?",
        "review": {"decisions": REVIEW_DECISIONS, "sha256": REVIEW_DECISIONS_SHA256, "counts": review["counts"]},
        "split": {"seed": SPLIT_SEED, "rule": "src.model_comparison._dataset seeded subject shuffle (app/submission)",
                  "subjects": cohort["partitions"]},
        "conditions": {
            "baseline": "all 724 cohort records (the submission model's data)",
            "drop_sensor5": "fit rows exclude the 5 records added to training_exclusions.json on 2026-09-30 (app now)",
            "drop_delete": "fit rows exclude the 16 rev2 'delete' recordings",
            "drop_invalid_segments": ("fit rows exclude rev2-invalid windows: 'delete' recordings except windows fully "
                                      "inside a stated normal interval; windows overlapping a stated abnormal interval"),
            "drop_delete_mismatch": "fit rows exclude the 16 'delete' and 10 'posture_mismatch' recordings",
        },
        "procedure": ("Per family: every src.model_comparison.MODEL_CANDIDATES entry fitted on train-subject fit rows "
                      "(validation_data passed as in train_comparison); parameters chosen by validation Macro F1, ties "
                      "worst-class F1, first candidate on exact ties. Selection rows: 'all' (primary) or 'valid' "
                      "(sensitivity), from the same fits. Final subjects are predicted only by the selected models."),
        "grids": {f: MODEL_CANDIDATES[f] for f in families}, "model_seeds": seeds,
        "evaluation_sets": {"all": "every window", "valid": "not invalid under rev2",
                            "valid_no_mismatch": "valid minus posture_mismatch recordings",
                            "review_ineligible": "subjects never in 008 development validation (never review candidates)"},
        "primary": PRIMARY, "criteria": CRITERIA,
        "decision_rule": ("Exploratory single split. Primary passes if RBF-SVM drop_invalid_segments vs baseline "
                          "(selection 'all') on final 'valid': seed-mean Macro F1 delta >= +1.0%p, every seed > 0, "
                          "primary-seed paired subject bootstrap (6 final subjects) 2.5% bound > 0, no class drop > 5%p. "
                          "Confirmation requires the preregistered 32-subject LOSO protocol."),
        "known_biases": [
            "Review candidates were ET all-wrong recordings; reviewer saw predictions (not blind).",
            "'valid' removes rows models tend to get wrong, inflating absolute scores; deltas on the same rows compare.",
            "Six final subjects: wide intervals; only 4 final + 2 validation subjects are review-ineligible.",
            "The split and hyperparameter grid were already used for the submission model and earlier studies.",
        ],
        "cohort": {k: cohort[k] for k in ("dataset_fingerprint", "records", "windows", "fit_windows", "fit_records",
                                           "evaluation_windows", "review_by_partition", "cache_sha256")},
        "source_revision": git("rev-parse", "HEAD").strip(), "dirty_state": git("status", "--short"),
        "source_hashes": {r: sha(ROOT / r) for r in SOURCES}, "command": [sys.executable, *sys.argv],
        "environment": {"python": sys.version, "platform": platform.platform(), "workers": args.workers,
                        "threads_per_worker": 1,
                        "packages": subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                                   text=True).stdout.split()},
        "models_saved": False, "active_model_changed": False, "raw_data_mutated": False,
    }
    write(out / "protocol.json", protocol)
    write(out / "protocol_lock.json", {"sha256": sha(out / "protocol.json"), "locked_at": now()})

    cache_dir = str(out / "cache")
    cache = _cache(cache_dir)
    order = {"rbf_svm": 0, "random_forest": 1, "extra_trees": 2, "logistic_regression": 3}
    jobs = sorted(({"condition": c, "family": f, "seed": s, "cache": cache_dir}
                   for f in families for s in seeds for c in CONDITIONS), key=lambda j: order[j["family"]])
    results, tick = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, future in enumerate(as_completed([pool.submit(task_job, j) for j in jobs]), 1):
            results.append(future.result())
            print(f"{done}/{len(jobs)} tasks done ({time.perf_counter() - tick:.0f}s)", flush=True)
    training_finished = now()
    summary, predictions = analyse(cache, results, seeds, families)
    write(out / "candidates.json", [{k: r[k] for k in ("condition", "family", "seed", "candidates",
                                                       "training_windows", "seconds")} for r in results])
    np.savez_compressed(out / "predictions.npz", truth=np.asarray(cache["y"]), groups=np.asarray(cache["groups"]),
                        record_ids=np.asarray(cache["records"]),
                        **{f"part__{p}": np.asarray(cache[f"part__{p}"]) for p in ("validation", "final_evaluation")},
                        **{f"set__{s}": np.asarray(cache[f"set__{s}"]) for s in SETS},
                        **{f"fit__{c}": np.asarray(cache[f"fit__{c}"]) for c in CONDITIONS},
                        **{"__".join(map(str, k)): v for k, v in predictions.items()})
    summary.update(started_at=started, training_finished_at=training_finished)
    if "rbf_svm" in families:
        summary["reproduction_vs_submission"] = reproduction(summary)
        primary = summary["comparisons"][":".join(PRIMARY[k] for k in ("model", "condition", "selection", "partition",
                                                                        "evaluation_set"))]
        summary["primary_decision"] = {**PRIMARY, "performance_pass": primary["performance_pass"],
                                       "mean_delta": primary["mean_delta"], "bootstrap": primary["bootstrap"]}
    after = protected_state()
    write(out / "protected_after.json", after)
    summary.update(protected_unchanged=before == after,
                   sources_unchanged=all(sha(ROOT / r) == protocol["source_hashes"][r] for r in SOURCES),
                   protocol_unchanged=sha(out / "protocol.json") == read(out / "protocol_lock.json")["sha256"],
                   review_decisions_unchanged=sha(decisions_path) == REVIEW_DECISIONS_SHA256, finished_at=now())
    write(out / "summary.json", summary)
    global _CACHE
    _CACHE = None
    shutil.rmtree(out / "cache")
    shutil.rmtree(stage_parent)
    print({k: summary.get(k) for k in ("primary_decision", "reproduction_vs_submission", "protected_unchanged",
                                        "sources_unchanged", "protocol_unchanged")}, flush=True)


if __name__ == "__main__":
    main()
