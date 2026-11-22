#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "window_models"
FIGURES = ROOT / "figures" / "window_models"
FIGURES.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = [
    "정자세",
    "앞으로 숙이기",
    "뒤로 기대기",
    "왼쪽 기울임",
    "오른쪽 기울임",
    "왼쪽 다리 꼬기",
    "오른쪽 다리 꼬기",
]

font_path = Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
if font_path.exists():
    font_name = FontProperties(fname=str(font_path)).get_name()
    matplotlib.rcParams["font.family"] = font_name
matplotlib.rcParams["axes.unicode_minus"] = False


def classification_report_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        classification_report(
            df["실제_클래스_ID"],
            df["예측_클래스_ID"],
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).T


def subject_performance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, group in df.groupby("subject_id", sort=True):
        rows.append(
            {
                "subject_id": subject,
                "record_count": len(group),
                "accuracy": accuracy_score(group["실제_클래스_ID"], group["예측_클래스_ID"]),
                "macro_f1": f1_score(
                    group["실제_클래스_ID"],
                    group["예측_클래스_ID"],
                    average="macro",
                    zero_division=0,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["accuracy", "macro_f1", "subject_id"])


def misclassification_pairs(df: pd.DataFrame) -> pd.DataFrame:
    errors = df[df["실제_클래스_ID"] != df["예측_클래스_ID"]]
    if errors.empty:
        return pd.DataFrame(columns=["실제_자세", "예측_자세", "오분류_횟수"])
    return (
        errors.groupby(["실제_자세", "예측_자세"])
        .size()
        .reset_index(name="오분류_횟수")
        .sort_values(["오분류_횟수", "실제_자세", "예측_자세"], ascending=[False, True, True])
    )


def normalized_confusion_figure(seconds: int, evaluation: str) -> None:
    df = pd.read_csv(RESULTS / f"window_{seconds}s_{evaluation}_oof_record_predictions.csv")
    matrix = confusion_matrix(
        df["실제_클래스_ID"],
        df["예측_클래스_ID"],
        labels=np.arange(len(CLASS_NAMES)),
    ).astype(float)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    image = ax.imshow(normalized, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("예측 자세")
    ax.set_ylabel("실제 자세")
    label = "중첩 교차검증" if evaluation == "nested" else "고정 배포 구성"
    ax.set_title(f"{seconds}초 창 모델: 기록 단위 정규화 혼동행렬 ({label})")
    for row in range(len(CLASS_NAMES)):
        for col in range(len(CLASS_NAMES)):
            text_color = "white" if normalized[row, col] >= 0.55 else "black"
            ax.text(col, row, f"{normalized[row, col]:.2f}", ha="center", va="center", color=text_color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / f"window_{seconds}s_{evaluation}_record_confusion_normalized.png", dpi=180)
    plt.close(fig)


def main() -> None:
    comparison = pd.read_csv(RESULTS / "window_length_comparison.csv")
    baseline = json.loads((RESULTS / "baseline_full_record_metrics.json").read_text(encoding="utf-8"))

    position_rows = []
    per_class_rows = []
    summary: dict[str, object] = {"baseline": baseline, "windows": {}}

    for seconds in (3, 5):
        for evaluation in ("nested", "fixed"):
            record_df = pd.read_csv(RESULTS / f"window_{seconds}s_{evaluation}_oof_record_predictions.csv")
            window_df = pd.read_csv(RESULTS / f"window_{seconds}s_{evaluation}_oof_window_predictions.csv")

            report = classification_report_frame(record_df)
            report.to_csv(
                RESULTS / f"window_{seconds}s_{evaluation}_record_classification_report_KO.csv",
                encoding="utf-8-sig",
            )
            subject_performance(record_df).to_csv(
                RESULTS / f"window_{seconds}s_{evaluation}_subject_performance.csv",
                index=False,
                encoding="utf-8-sig",
            )
            misclassification_pairs(record_df).to_csv(
                RESULTS / f"window_{seconds}s_{evaluation}_misclassification_pairs.csv",
                index=False,
                encoding="utf-8-sig",
            )
            for class_name in CLASS_NAMES:
                per_class_rows.append(
                    {
                        "window_seconds": seconds,
                        "evaluation": evaluation,
                        "class_name": class_name,
                        "precision": report.loc[class_name, "precision"],
                        "recall": report.loc[class_name, "recall"],
                        "f1": report.loc[class_name, "f1-score"],
                        "support": report.loc[class_name, "support"],
                    }
                )

            if evaluation == "fixed":
                for index, group in window_df.groupby("window_index", sort=True):
                    position_rows.append(
                        {
                            "window_seconds": seconds,
                            "window_index": int(index),
                            "start_s": float(group["start_s"].iloc[0]),
                            "sample_count": len(group),
                            "accuracy": accuracy_score(group["실제_클래스_ID"], group["예측_클래스_ID"]),
                            "macro_f1": f1_score(
                                group["실제_클래스_ID"],
                                group["예측_클래스_ID"],
                                average="macro",
                                zero_division=0,
                            ),
                        }
                    )

            normalized_confusion_figure(seconds, evaluation)

        model_config = json.loads(
            (ROOT / "artifacts" / f"window_{seconds}s_model_config.json").read_text(encoding="utf-8")
        )
        nested_row = comparison[
            (comparison["window_seconds"] == seconds) & (comparison["evaluation"] == "nested")
        ].iloc[0]
        fixed_row = comparison[
            (comparison["window_seconds"] == seconds) & (comparison["evaluation"] == "fixed")
        ].iloc[0]
        summary["windows"][str(seconds)] = {
            "training_window_count": int(model_config["training_window_count"]),
            "target_length": int(model_config["target_length"]),
            "svm_params": model_config["svm_params"],
            "nested_window_macro_f1": float(nested_row["window_macro_f1"]),
            "nested_record_macro_f1": float(nested_row["record_macro_f1"]),
            "fixed_window_macro_f1": float(fixed_row["window_macro_f1"]),
            "fixed_record_macro_f1": float(fixed_row["record_macro_f1"]),
        }

    position = pd.DataFrame(position_rows)
    position.to_csv(RESULTS / "window_position_performance.csv", index=False, encoding="utf-8-sig")
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(RESULTS / "window_per_class_record_metrics.csv", index=False, encoding="utf-8-sig")
    (RESULTS / "window_retraining_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Primary unbiased comparison: nested subject-wise CV.
    nested = comparison[comparison["evaluation"] == "nested"].sort_values("window_seconds")
    x = np.arange(len(nested))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.bar(x - width / 2, nested["window_macro_f1"], width, label="개별 창 예측")
    ax.bar(x + width / 2, nested["record_macro_f1"], width, label="창 확률 평균(기록 단위)")
    baseline_nested = baseline.get("full_record_nested_macro_f1")
    if baseline_nested is not None:
        ax.axhline(float(baseline_nested), linestyle="--", linewidth=1.3, label="기존 20초 전체기록 모델")
    ax.set_xticks(x, [f"{int(value)}초 창" for value in nested["window_seconds"]])
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Macro F1")
    ax.set_title("창 길이별 중첩 피험자 교차검증 성능")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "nested_cv_window_length_comparison.png", dpi=180)
    plt.close(fig)

    fixed = comparison[comparison["evaluation"] == "fixed"].sort_values("window_seconds")
    x = np.arange(len(fixed))
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.bar(x - width / 2, fixed["window_macro_f1"], width, label="개별 창 예측")
    ax.bar(x + width / 2, fixed["record_macro_f1"], width, label="창 확률 평균(기록 단위)")
    baseline_fixed = baseline.get("full_record_fixed_macro_f1")
    if baseline_fixed is not None:
        ax.axhline(float(baseline_fixed), linestyle="--", linewidth=1.3, label="기존 20초 전체기록 모델")
    ax.set_xticks(x, [f"{int(value)}초 창" for value in fixed["window_seconds"]])
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Macro F1")
    ax.set_title("창 길이별 고정 배포 구성 성능")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fixed_config_window_length_comparison.png", dpi=180)
    plt.close(fig)

    fixed_class = per_class[per_class["evaluation"] == "fixed"].copy()
    pivot = fixed_class.pivot(index="class_name", columns="window_seconds", values="f1").reindex(CLASS_NAMES)
    x = np.arange(len(CLASS_NAMES))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x - width / 2, pivot[3], width, label="3초")
    ax.bar(x + width / 2, pivot[5], width, label="5초")
    ax.set_xticks(x, CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("기록 단위 F1")
    ax.set_title("자세별 기록 단위 F1 비교: 고정 배포 구성")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "per_class_record_f1_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for seconds, group in position.groupby("window_seconds", sort=True):
        ax.plot(group["start_s"], group["macro_f1"], marker="o", label=f"{seconds}초 창")
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("창 시작 시점(초)")
    ax.set_ylabel("Macro F1")
    ax.set_title("기록 내 창 위치별 성능: 고정 배포 구성")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "window_position_macro_f1.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
