#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "window_models"
ARTIFACTS = ROOT / "artifacts"
checks: list[dict[str, object]] = []


def add(name: str, passed: bool, detail: str) -> None:
    checks.append({"검사": name, "통과": bool(passed), "상세": detail})
    if not passed:
        raise AssertionError(f"{name}: {detail}")


for seconds, expected_windows, expected_length in ((3, 1176, 36), (5, 784, 60)):
    model_path = ARTIFACTS / f"posture_classifier_window_{seconds}s.joblib"
    bundle = joblib.load(model_path)
    add(f"{seconds}초 모델 로드", True, model_path.name)
    add(
        f"{seconds}초 메타데이터",
        bundle["training_window_count"] == expected_windows
        and bundle["target_length"] == expected_length
        and len(bundle["feature_names"]) == 596,
        f"창={bundle['training_window_count']}, 시점={bundle['target_length']}, 특징={len(bundle['feature_names'])}",
    )
    classes = np.asarray(bundle["svm_model"].named_steps["model"].classes_, dtype=int)
    add(f"{seconds}초 클래스", np.array_equal(classes, np.arange(7)), str(classes.tolist()))

    for evaluation in ("nested", "fixed"):
        window_df = pd.read_csv(RESULTS / f"window_{seconds}s_{evaluation}_oof_window_predictions.csv")
        record_df = pd.read_csv(RESULTS / f"window_{seconds}s_{evaluation}_oof_record_predictions.csv")
        add(
            f"{seconds}초 {evaluation} 행 수",
            len(window_df) == expected_windows and len(record_df) == 196,
            f"창={len(window_df)}, 기록={len(record_df)}",
        )
        probability_cols = [column for column in window_df.columns if column.startswith("확률_")]
        sums = window_df[probability_cols].sum(axis=1).to_numpy()
        add(
            f"{seconds}초 {evaluation} 확률 합",
            np.allclose(sums, 1.0, atol=1e-6),
            f"최소={sums.min():.9f}, 최대={sums.max():.9f}",
        )

    fold_df = pd.read_csv(RESULTS / f"window_{seconds}s_nested_folds.csv")
    subjects: list[str] = []
    for value in fold_df["test_subjects"]:
        subjects.extend(part.strip() for part in str(value).split("|"))
    add(
        f"{seconds}초 외부 fold 피험자",
        len(subjects) == 14 and len(set(subjects)) == 14,
        f"등장={len(subjects)}, 고유={len(set(subjects))}",
    )

completed = subprocess.run(
    [
        sys.executable,
        str(ROOT / "inference_window.py"),
        "--input",
        str(ROOT / "example" / "sample_input.csv"),
        "--window",
        "both",
        "--output",
        str(ROOT / "example" / "example_prediction_window_both.json"),
    ],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=False,
)
add("통합 추론 실행", completed.returncode == 0, f"종료 코드={completed.returncode}")
result = json.loads((ROOT / "example" / "example_prediction_window_both.json").read_text(encoding="utf-8"))
add(
    "통합 추론 창 수",
    result["결과"]["3초"]["창_개수"] == 6 and result["결과"]["5초"]["창_개수"] == 4,
    f"3초={result['결과']['3초']['창_개수']}, 5초={result['결과']['5초']['창_개수']}",
)

payload = {"전체_검사_수": len(checks), "전체_통과": all(item["통과"] for item in checks), "검사": checks}
(RESULTS / "validation_checks.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
pd.DataFrame(checks).to_csv(RESULTS / "validation_checks.csv", index=False, encoding="utf-8-sig")
print(json.dumps(payload, ensure_ascii=False, indent=2))
