"""Print Markdown tables for the despike study report from a finished run folder.

Read only. Usage:
  .venv/bin/python -m experiments.despike_study.report research/results/<run>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CLASSES = ["정자세", "앞으로 숙이기", "뒤로 기대기", "왼쪽 기울임", "오른쪽 기울임", "왼쪽 다리 꼬기", "오른쪽 다리 꼬기"]
ARMS = ("baseline", "despike", "trim1", "despike_trim1")
MODELS = {"rbf_svm": "RBF-SVM (운영 레시피)", "extra_trees": "Extra Trees (연구 기준)"}


def pct(value: float, sign: bool = False) -> str:
    return f"{value * 100:+.2f}" if sign else f"{value * 100:.2f}"


def main() -> None:
    run = Path(sys.argv[1])
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    seeds = json.loads((run / "protocol.json").read_text(encoding="utf-8"))["seeds"]
    metrics, comparisons = summary["metrics"], summary["comparisons"]

    print("## Pooled LOSO\n")
    print("| 모델 | 조건 | Macro F1 (시드별) | 평균 | Accuracy | 기록 단위 F1 | 실시간 평활 F1 | 실시간 변경/분 |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for family, label in MODELS.items():
        for arm in ARMS:
            rows = [metrics[f"{family}:{arm}:{s}"] for s in seeds]
            f1 = [r["macro_f1"] for r in rows]
            print(f"| {label} | {arm} | {' / '.join(pct(v) for v in f1)} | {pct(np.mean(f1))} | "
                  f"{pct(np.mean([r['accuracy'] for r in rows]))} | {pct(np.mean([r['record_macro_f1'] for r in rows]))} | "
                  f"{pct(np.mean([r['live']['smoothed_macro_f1'] for r in rows]))} | "
                  f"{np.mean([r['live']['smoothed_changes_per_minute'] for r in rows]):.3f} |")

    print("\n## Decisions (arm - baseline)\n")
    print("| 모델 | 조건 | 시드별 차이 (%p) | 평균 | bootstrap 95% (주 시드) | 개선/악화 사람 | 최대 자세 하락 | 변경 감소율 (시드별) | 성능 기준 | 안정성 기준 |")
    print("| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | --- |")
    for family, label in MODELS.items():
        for arm in ARMS[1:]:
            c = comparisons[f"{family}:{arm}_minus_baseline"]
            b = c["bootstrap"]
            print(f"| {label} | {arm} | {' / '.join(pct(v, True) for v in c['seed_deltas'])} | {pct(c['mean_delta'], True)} | "
                  f"[{pct(b['lower_2_5'], True)}, {pct(b['upper_97_5'], True)}] | {c['subjects_improved']}/{c['subjects_worse']} | "
                  f"{pct(min(c['mean_class_deltas']), True)} | "
                  f"{' / '.join(f'{v * 100:+.1f}%' for v in c['relative_change_reduction_by_seed'])} | "
                  f"{'통과' if c['performance_pass'] else '미달'} | {'통과' if c['stability_pass'] else '미달'} |")

    print("\n## Failed checks\n")
    for key, c in comparisons.items():
        failed = [k for k, v in c["performance_checks"].items() if not v]
        failed_s = [k for k, v in c["stability_checks"].items() if not v]
        print(f"- {key}: performance {failed or 'none'}; stability {failed_s or 'none'}")

    print("\n## Per-class pooled F1 (seed mean, %)\n")
    print("| 모델 | 조건 | " + " | ".join(CLASSES) + " |")
    print("| --- | --- | " + " | ".join("---:" for _ in CLASSES) + " |")
    for family, label in MODELS.items():
        for arm in ARMS:
            values = np.mean([metrics[f"{family}:{arm}:{s}"]["per_class_f1"] for s in seeds], axis=0)
            print(f"| {label} | {arm} | " + " | ".join(f"{v * 100:.1f}" for v in values) + " |")

    print("\n## Secondary deltas (seed mean, %p)\n")
    print("| 모델 | 조건 | 기록 단위 F1 | 실시간 평활 F1 |")
    print("| --- | --- | ---: | ---: |")
    for family, label in MODELS.items():
        for arm in ARMS[1:]:
            c = comparisons[f"{family}:{arm}_minus_baseline"]
            print(f"| {label} | {arm} | {pct(np.mean(c['record_macro_f1_delta_by_seed']), True)} | "
                  f"{pct(np.mean(c['live_smoothed_macro_f1_delta_by_seed']), True)} |")

    print("\n## Integrity\n")
    for key in ("primary_decision", "production_loso_reproduction", "protected_unchanged", "sources_unchanged",
                "protocol_unchanged", "started_at", "training_finished_at", "finished_at", "fold_training_seconds_median"):
        value = summary.get(key)
        if key == "production_loso_reproduction":
            value = [{k: r[k] for k in ("arm", "family", "seed", "macro_f1", "fold_loop_macro_f1", "identical")} for r in value]
        print(f"- {key}: {json.dumps(value, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
