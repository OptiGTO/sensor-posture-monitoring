"""Equal-weight (0.5:0.5) probability ensemble of the two best models from the
2026-09-28 runtime-vs-research run. No refitting: saved member probabilities are
averaged, and each member's probabilities are re-derived from its saved model.

Selection (fixed before any ensemble score is computed):
  rank arms by development Macro F1 (primary seed, mean of 3 splits); take the best
  arm of each model family (ET, RBF-SVM). The strict top-two (both ET, differing only
  in data policy) is reported as a secondary ensemble.
The historical six-subject holdout has been observed repeatedly; results are
retrospective, not new-user evidence.

Usage (new output folder only):
  .venv/bin/python -B -m experiments.runtime_vs_research.ensemble \
    --output research/results/<YYYY-MM-DD_HHMMSS>_top_two_equal_ensemble
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.runtime_vs_research.analysis import (  # noqa: E402
    decide, macro_f1, paired_subject_bootstrap, per_class_f1)
from experiments.runtime_vs_research.run import (  # noqa: E402
    CRITERIA, SEEDS, load_frozen, now, protected_state, read, sha, split_rows, write)

SOURCE = ROOT / "research/results/2026-09-28_131451_runtime_vs_research"
ARMS = ["svm_standard", "svm_ordinary", "et_standard", "et_ordinary"]
FAMILY = {"svm_standard": "rbf_svm", "svm_ordinary": "rbf_svm", "et_standard": "extra_trees", "et_ordinary": "extra_trees"}
SPLITS = ["development_1", "development_2", "development_3"]
SOURCES = ["experiments/runtime_vs_research/ensemble.py", "experiments/runtime_vs_research/run.py",
           "experiments/runtime_vs_research/analysis.py"]
WEIGHTS = (0.5, 0.5)


def cell(seed: int, arm: str, split: str | None) -> Path:
    return SOURCE / (f"development/{seed}/{arm}/{split}" if split else f"final/{seed}/{arm}")


def select(summary: dict) -> dict:
    dev = {a: summary["development_macro_f1"][a]["primary_seed_mean"] for a in ARMS}
    ranking = sorted(ARMS, key=lambda a: -dev[a])
    best_per_family = []
    for arm in ranking:
        if FAMILY[arm] not in {FAMILY[b] for b in best_per_family}:
            best_per_family.append(arm)
    return {"criterion": "development Macro F1, primary seed 20260921, mean of 3 splits",
            "ranking": [{"arm": a, "family": FAMILY[a], "development_primary_seed_mean": dev[a]} for a in ranking],
            "primary": best_per_family[:2], "secondary_strict_top_two": ranking[:2]}


def load_members(members: list[str], seed: int, split: str | None) -> list[dict]:
    loaded = []
    for arm in members:
        folder = cell(seed, arm, split)
        metrics = read(folder / "metrics.json")
        if sha(folder / "predictions.npz") != metrics["predictions_sha256"] or sha(folder / "model.joblib") != metrics["model_sha256"]:
            raise ValueError(f"Saved artifact hash mismatch: {folder}")
        loaded.append({"arm": arm, "folder": folder, "metrics": metrics, "p": dict(np.load(folder / "predictions.npz"))})
    a, b = loaded
    for key in ("indices", "truth", "groups", "record_ids"):
        if not np.array_equal(a["p"][key], b["p"][key]):
            raise ValueError(f"Members misaligned on {key}: {a['folder']}")
    if a["metrics"]["evaluation_subjects"] != b["metrics"]["evaluation_subjects"]:
        raise ValueError("Evaluation subjects differ")
    if set(a["metrics"]["training_subjects"]) & set(a["metrics"]["evaluation_subjects"]):
        raise ValueError("Subject leakage")
    return loaded


def reproduce(member: dict, data: dict) -> float:
    """Reload the saved model and recompute its probabilities on the saved rows."""
    model = joblib.load(member["folder"] / "model.joblib")
    probability = model.predict_proba(data["X"][member["p"]["indices"]])
    return float(np.abs(probability - member["p"]["probability"]).max())


def evaluate(members: list[str], data: dict, partitions: dict) -> dict:
    cells, repro = {}, []
    for seed in SEEDS:
        for split in [*SPLITS, None]:
            m = load_members(members, seed, split)
            part = partitions[split or "development_1"]
            _, valid = split_rows(data, part, final=split is None)
            if not np.array_equal(valid, m[0]["p"]["indices"]):
                raise ValueError("Saved rows differ from frozen split")
            for member in m:
                repro.append({"cell": str(member["folder"].relative_to(ROOT)), "max_abs_probability_difference": reproduce(member, data)})
            prob = WEIGHTS[0] * m[0]["p"]["probability"] + WEIGHTS[1] * m[1]["p"]["probability"]
            truth, groups = m[0]["p"]["truth"], m[0]["p"]["groups"]
            pred = prob.argmax(1)
            entry = {"truth": truth, "groups": groups, "probability": prob, "prediction": pred}
            for member in m:
                entry[member["arm"]] = member["p"]["prediction"]
            cells[(seed, split)] = entry
    return {"cells": cells, "reproduction": repro}


def scores(cells: dict, key: str) -> dict:
    dev = np.array([[macro_f1(cells[s, k]["truth"], cells[s, k][key]) for k in SPLITS] for s in SEEDS])
    dev_cls = np.array([[per_class_f1(cells[s, k]["truth"], cells[s, k][key]) for k in SPLITS] for s in SEEDS])
    dev_acc = np.array([[float((cells[s, k]["truth"] == cells[s, k][key]).mean()) for k in SPLITS] for s in SEEDS])
    hold = np.array([macro_f1(cells[s, None]["truth"], cells[s, None][key]) for s in SEEDS])
    hold_cls = np.array([per_class_f1(cells[s, None]["truth"], cells[s, None][key]) for s in SEEDS])
    hold_acc = np.array([float((cells[s, None]["truth"] == cells[s, None][key]).mean()) for s in SEEDS])
    c = cells[SEEDS[0], None]
    per_subject = {str(g): macro_f1(c["truth"][c["groups"] == g], c[key][c["groups"] == g]) for g in sorted(set(c["groups"]))}
    return {"dev": dev, "dev_cls": dev_cls, "hold": hold, "hold_cls": hold_cls, "json": {
        "development_by_seed_split": dev.tolist(), "development_primary_seed_mean": float(dev[0].mean()),
        "development_all_seed_mean": float(dev.mean()), "development_accuracy_all_seed_mean": float(dev_acc.mean()),
        "development_per_class_f1_mean": dev_cls.mean((0, 1)).tolist(),
        "holdout_by_seed": hold.tolist(), "holdout_mean": float(hold.mean()), "holdout_accuracy_mean": float(hold_acc.mean()),
        "holdout_per_class_f1_seed_mean": hold_cls.mean(0).tolist(), "holdout_per_subject_primary_seed": per_subject}}


def compare(cells: dict, base: str, all_scores: dict) -> dict:
    a, b = all_scores[base], all_scores["ensemble"]
    c = cells[SEEDS[0], None]
    boot = paired_subject_bootstrap(c["truth"], c[base], c["prediction"], c["groups"],
                                    replicates=CRITERIA["bootstrap_replicates"], seed=CRITERIA["bootstrap_seed"])
    decision = decide(a["dev"], b["dev"], a["dev_cls"], b["dev_cls"], a["hold"], b["hold"], a["hold_cls"], b["hold_cls"],
                      boot["lower_2_5"], CRITERIA)
    fixed = sum(int(((cells[s, None][base] != cells[s, None]["truth"]) & (cells[s, None]["prediction"] == cells[s, None]["truth"])).sum()) for s in SEEDS[:1])
    broken = sum(int(((cells[s, None][base] == cells[s, None]["truth"]) & (cells[s, None]["prediction"] != cells[s, None]["truth"])).sum()) for s in SEEDS[:1])
    return {"bootstrap_primary_seed": boot, "decision": decision,
            "holdout_primary_seed_fixed_errors": fixed, "holdout_primary_seed_new_errors": broken}


def run_ensemble(members: list[str], data: dict, partitions: dict, out: Path, label: str) -> dict:
    result = evaluate(members, data, partitions)
    cells = result["cells"]
    for (seed, split), c in cells.items():
        dest = out / label / (f"development/{seed}/{split}" if split else f"final/{seed}")
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "predictions.npz", truth=c["truth"], groups=c["groups"], probability=c["probability"], prediction=c["prediction"])
    all_scores = {m: scores(cells, m) for m in members}
    all_scores["ensemble"] = scores(cells, "prediction")
    return {"members": members, "weights": list(WEIGHTS),
            "scores": {k: v["json"] for k, v in all_scores.items()},
            "ensemble_vs_member": {m: compare(cells, m, all_scores) for m in members},
            "reproduction_max_abs_probability_difference": max(r["max_abs_probability_difference"] for r in result["reproduction"]),
            "reproduction": result["reproduction"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    if out.exists():
        raise SystemExit(f"Output exists; use a new folder: {out}")
    out.mkdir(parents=True)
    started = now()
    before = protected_state()
    write(out / "protected_before.json", before)
    source_summary = read(SOURCE / "summary.json")
    selection = select(source_summary)
    adopt_protocol, data, _ = load_frozen()
    partitions = {p["id"]: p for p in adopt_protocol["partitions"]}
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    protocol = {
        "schema_version": 1, "started_at": started,
        "question": "Does a 0.5:0.5 probability average of the two best models beat its members?",
        "source_run": str(SOURCE.relative_to(ROOT)), "source_summary_sha256": sha(SOURCE / "summary.json"),
        "source_protocol_sha256": sha(SOURCE / "protocol.json"), "selection": selection, "weights": list(WEIGHTS),
        "combination": "P = 0.5*P1 + 0.5*P2 over class-aligned predict_proba; argmax; no weight search, no refit",
        "criteria": CRITERIA,
        "decision_rule": "Recommend the ensemble over a member only if every check in analysis.decide passes (ensemble as B, member as A). Fixed before computing ensemble scores.",
        "exposure": "Selection uses development scores only, but holdout scores of the members were already reported on 2026-09-28; holdout results are retrospective.",
        "source_revision": git("rev-parse", "HEAD").strip(), "dirty_state": git("status", "--short"),
        "source_hashes": {r: sha(ROOT / r) for r in SOURCES}, "command": [sys.executable, *sys.argv],
        "environment": {"python": sys.version, "platform": platform.platform()},
        "active_model_changed": False, "raw_data_mutated": False,
    }
    write(out / "protocol.json", protocol)
    write(out / "protocol_lock.json", {"sha256": sha(out / "protocol.json"), "locked_at": now()})

    summary = {"started_at": started, "selection": selection}
    for label, members in (("primary", selection["primary"]), ("secondary_strict_top_two", selection["secondary_strict_top_two"])):
        summary[label] = run_ensemble(members, data, partitions, out, label)
    after = protected_state()
    write(out / "protected_after.json", after)
    summary["protected_unchanged"] = before == after
    summary["sources_unchanged"] = all(sha(ROOT / r) == protocol["source_hashes"][r] for r in SOURCES)
    summary["protocol_unchanged"] = sha(out / "protocol.json") == read(out / "protocol_lock.json")["sha256"]
    summary["finished_at"] = now()
    write(out / "summary.json", summary)
    for label in ("primary", "secondary_strict_top_two"):
        s = summary[label]["scores"]
        print(label, summary[label]["members"])
        for k, v in s.items():
            print(f"  {k:14s} dev={v['development_primary_seed_mean']:.4f} dev3={v['development_all_seed_mean']:.4f} hold={v['holdout_mean']:.4f} {[round(x,4) for x in v['holdout_by_seed']]}")
        for m, c in summary[label]["ensemble_vs_member"].items():
            print(f"  vs {m}: recommend={c['decision']['replace']} checks={c['decision']['checks']} boot={c['bootstrap_primary_seed']['observed']:.4f} [{c['bootstrap_primary_seed']['lower_2_5']:.4f},{c['bootstrap_primary_seed']['upper_97_5']:.4f}]")
        print("  repro max diff", summary[label]["reproduction_max_abs_probability_difference"])
    print("protected_unchanged", summary["protected_unchanged"], "sources", summary["sources_unchanged"], "protocol", summary["protocol_unchanged"])


if __name__ == "__main__":
    main()
