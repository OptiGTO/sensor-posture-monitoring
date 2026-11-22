#!/usr/bin/env python3
"""Supplementary analysis of a finished spike_trim_study run (reads outputs only).

Writes ``supplementary.json`` into the run directory:
  * per-class LOSO F1 deltas vs baseline;
  * Holm-adjusted Wilcoxon p-values for the paired LOSO subject Macro F1 deltas;
  * live (LOSO) event-level agreement with baseline: how often the raw prediction
    changes, split by windows with / without a review-rule spike, and whether a
    change fixed or broke the prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = ["정자세", "앞으로 숙이기", "뒤로 기대기", "왼쪽 기울임", "오른쪽 기울임", "왼쪽 다리 꼬기", "오른쪽 다리 꼬기"]


def holm(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    items = sorted(((key, value) for key, value in pvalues.items() if value is not None), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for rank, (key, value) in enumerate(items):
        running = max(running, min(1.0, (len(items) - rank) * value))
        adjusted[key] = running
    return {key: adjusted.get(key) for key in pvalues}


def live_agreement(base: pd.DataFrame, other: pd.DataFrame) -> dict:
    keys = ["scope", "record_id", "end_s"]
    if not base[keys].equals(other[keys]):
        raise ValueError("live event order differs between conditions")
    changed = base.raw_pred.to_numpy() != other.raw_pred.to_numpy()
    truth = base.y.to_numpy()
    base_ok, other_ok = base.raw_pred.to_numpy() == truth, other.raw_pred.to_numpy() == truth
    spike = base.has_spike.to_numpy()
    smooth_changed = base.smooth_pred.to_numpy() != other.smooth_pred.to_numpy()

    def part(mask):
        return {"events": int(mask.sum()), "raw_prediction_changed": float(changed[mask].mean()),
                "smoothed_prediction_changed": float(smooth_changed[mask].mean()),
                "fixed": int((changed & ~base_ok & other_ok & mask).sum()),
                "broken": int((changed & base_ok & ~other_ok & mask).sum())}
    return {"all": part(np.ones_like(spike)), "spike_windows": part(spike), "clean_windows": part(~spike)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    tasks = {(t["condition"], t["family"]): t for t in
             (json.loads(path.read_text(encoding="utf-8")) for path in sorted((run / "tasks").glob("*.json")))}
    pvalues = {f"{row['family']}|{row['condition']}": row["loso_subject_macro_f1"]["wilcoxon_p"]
               for row in summary["paired_vs_baseline"]}
    adjusted = holm(pvalues)
    comparisons = []
    for row in summary["paired_vs_baseline"]:
        family, condition = row["family"], row["condition"]
        base, other = tasks[("baseline", family)], tasks[(condition, family)]
        live = "loso_live"
        base_live = pd.read_csv(run / "predictions" / f"baseline__{family}__live.csv.gz")
        other_live = pd.read_csv(run / "predictions" / f"{condition}__{family}__live.csv.gz")
        base_live, other_live = (frame[frame.scope == live].reset_index(drop=True) for frame in (base_live, other_live))
        comparisons.append({
            "family": family, "condition": condition,
            "loso_macro_f1_delta": other["loso"]["macro_f1"] - base["loso"]["macro_f1"],
            "subject_wilcoxon_p": row["loso_subject_macro_f1"]["wilcoxon_p"],
            "subject_wilcoxon_p_holm": adjusted[f"{family}|{condition}"],
            "loso_per_class_f1_delta": {name: other["loso"]["per_class_f1"][name] - base["loso"]["per_class_f1"][name]
                                        for name in CLASS_NAMES},
            "final_macro_f1_delta": other["final"]["macro_f1"] - base["final"]["macro_f1"],
            "selected_params": {"baseline": base["selected_params"], "condition": other["selected_params"]},
            "loso_live_vs_baseline": live_agreement(base_live, other_live),
        })
    payload = {"source_run": run.name, "holm_family_size": sum(value is not None for value in pvalues.values()),
               "comparisons": comparisons}
    (run / "supplementary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in comparisons:
        agreement = row["loso_live_vs_baseline"]
        print(f"{row['family']:20s} {row['condition']:18s} dF1 {100 * row['loso_macro_f1_delta']:+.2f} "
              f"p {row['subject_wilcoxon_p']:.3f} holm {row['subject_wilcoxon_p_holm']:.3f} "
              f"upright dF1 {100 * row['loso_per_class_f1_delta']['정자세']:+.2f} | live changed "
              f"{100 * agreement['all']['raw_prediction_changed']:.2f}% (spike "
              f"{100 * agreement['spike_windows']['raw_prediction_changed']:.2f}%, clean "
              f"{100 * agreement['clean_windows']['raw_prediction_changed']:.2f}%) fixed/broken "
              f"{agreement['all']['fixed']}/{agreement['all']['broken']}")


if __name__ == "__main__":
    main()
