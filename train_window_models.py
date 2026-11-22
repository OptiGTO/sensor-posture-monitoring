#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, feature_names, make_subject_pairs
from src.training_windows import MAX_WINDOW_SECONDS, MIN_WINDOW_SECONDS
from src.model_catalog import MODEL_NAMES, CHECKPOINT_MODEL_NAMES
from src.window_pipeline import (
    WindowSpec,
    aggregate_probabilities,
    aligned_predict_proba,
    extract_window_features,
    load_window_dataset,
)

SEED = 20260826
SVM_CANDIDATES = [
    {"C": 0.3, "gamma": "scale"},
    {"C": 1.0, "gamma": "scale"},
    {"C": 3.0, "gamma": "scale"},
    {"C": 10.0, "gamma": "scale"},
    {"C": 30.0, "gamma": "scale"},
    {"C": 1.0, "gamma": 0.0005},
    {"C": 3.0, "gamma": 0.001},
    {"C": 10.0, "gamma": 0.003},
]


def training_window_arg(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("윈도우 길이는 정수여야 합니다.") from error
    if not MIN_WINDOW_SECONDS <= seconds <= MAX_WINDOW_SECONDS:
        raise argparse.ArgumentTypeError(
            f"윈도우 길이는 {MIN_WINDOW_SECONDS}초부터 {MAX_WINDOW_SECONDS}초 사이여야 합니다."
        )
    return seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="사용자가 선택한 고정 길이 창으로 자세 분류 모델 재학습")
    parser.add_argument("--data", required=True, help="해제된 train 데이터 디렉터리")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--personalized", action="store_true", help="개인 기준 2단계 모델과 비교군 학습")
    parser.add_argument("--allow-retrospective", action="store_true", help="빈 좌석 0 가정으로 과거 기록 실험 허용")
    parser.add_argument("--deployment", action="store_true", help="독립 평가 근거를 갖춘 배포 후보 학습")
    parser.add_argument("--soft-voting-members", nargs="+", help="동일 가중 실험 구성: extra_trees:2 rbf_svm:3 등")
    parser.add_argument("--training-method", default="standard",
                        help="기본 standard 또는 재실험에서 개선이 확인되어 등록된 학습 방식")
    parser.add_argument("--selection-metric", choices=("loso", "validation"), default="loso",
                        help="모델·창 선택 기준: loso(개발 피험자 한 명씩 제외, 기본) 또는 validation(단일 검증 분할)")
    parser.add_argument("--despike", choices=("none", "despike_v1"), default="none",
                        help="버전 신호 전처리: despike_v1은 학습·추론 모두에 한 샘플 튐 보정을 적용 (--deployment --models 전용)")
    parser.add_argument("--training-trim-start-seconds", type=float, default=0.0,
                        help="학습 기록마다 처음 N초 제외(0~5초, 학습 전용; 평가·추론에는 미적용, --deployment --models 전용)")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_NAMES) + tuple(CHECKPOINT_MODEL_NAMES),
                        help="동일 분할로 비교할 모델 (배포 학습에서 사용)")
    parser.add_argument(
        "--windows", type=training_window_arg, nargs="+", default=(3, 5, 20),
        help=f"학습할 창 길이(초, {MIN_WINDOW_SECONDS}~{MAX_WINDOW_SECONDS}; 여러 값 입력 가능)",
    )
    parser.add_argument(
        "--baseline-config",
        default=None,
        help="선택 사항: 기존 전체기록 모델의 final_model_config.json 경로",
    )
    return parser.parse_args()


def build_svm(params: dict[str, object], probability: bool, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                SVC(
                    C=params["C"],
                    gamma=params["gamma"],
                    kernel="rbf",
                    class_weight="balanced",
                    probability=probability,
                    random_state=seed,
                ),
            ),
        ]
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def select_candidate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_splits: int = 3,
) -> tuple[dict[str, object], float, pd.DataFrame]:
    if len(np.unique(groups)) < 2:
        raise ValueError("내부 검증에 사용할 피험자가 최소 2명 필요합니다.")
    cv = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    splits = list(cv.split(X, y, groups))
    if any(len(np.unique(y[train_idx])) < 2 for train_idx, _ in splits):
        raise ValueError("각 학습 분할에 최소 두 가지 자세가 필요합니다. 피험자별 자세 데이터를 추가하세요.")
    rows: list[dict[str, object]] = []
    best_params: dict[str, object] | None = None
    best_score = -np.inf
    for candidate_index, params in enumerate(SVM_CANDIDATES):
        fold_scores: list[float] = []
        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            model = build_svm(params, probability=False, seed=seed)
            model.fit(X[train_idx], y[train_idx])
            prediction = model.predict(X[val_idx])
            score = float(f1_score(y[val_idx], prediction, average="macro", zero_division=0))
            fold_scores.append(score)
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "C": params["C"],
                    "gamma": params["gamma"],
                    "fold": fold,
                    "macro_f1": score,
                }
            )
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score + 1e-12:
            best_score = mean_score
            best_params = dict(params)
    if best_params is None:
        raise RuntimeError("SVM 후보 선택에 실패했습니다.")
    return best_params, best_score, pd.DataFrame(rows)


def add_fold_summary(prefix: str, fold_frame: pd.DataFrame, summary: dict[str, object]) -> None:
    for metric_name in ("accuracy", "balanced_accuracy", "macro_f1"):
        values = fold_frame[f"{prefix}_{metric_name}"]
        summary[f"{prefix}_fold_{metric_name}_mean"] = float(values.mean())
        summary[f"{prefix}_fold_{metric_name}_std"] = float(values.std(ddof=0))


def evaluate_nested(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    record_ids: np.ndarray,
    subject_codes: dict[str, str],
    seed: int,
) -> dict[str, object]:
    outer_pairs = make_subject_pairs(groups, seed=seed)
    oof_probability = np.zeros((len(y), len(CLASS_NAMES)), dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    inner_rows: list[pd.DataFrame] = []

    for fold, test_subjects in enumerate(outer_pairs, start=1):
        test_mask = np.isin(groups, test_subjects)
        train_mask = ~test_mask
        if not set(groups[train_mask]).isdisjoint(set(groups[test_mask])):
            raise RuntimeError("피험자 누수가 발생했습니다.")
        best_params, best_score, candidate_folds = select_candidate(
            X[train_mask], y[train_mask], groups[train_mask], seed=seed, n_splits=3
        )
        candidate_folds.insert(0, "outer_fold", fold)
        inner_rows.append(candidate_folds)
        model = build_svm(best_params, probability=True, seed=seed)
        model.fit(X[train_mask], y[train_mask])
        probability = aligned_predict_proba(model, X[test_mask], len(CLASS_NAMES))
        oof_probability[np.flatnonzero(test_mask)] = probability
        window_pred = probability.argmax(axis=1)
        window_metric = metrics(y[test_mask], window_pred)
        record_probability, record_y, _, _ = aggregate_probabilities(
            probability,
            record_ids[test_mask],
            y[test_mask],
            groups[test_mask],
        )
        record_pred = record_probability.argmax(axis=1)
        record_metric = metrics(record_y, record_pred)
        row: dict[str, object] = {
            "fold": fold,
            "test_subjects": " | ".join(subject_codes[str(subject)] for subject in test_subjects),
            "selected_C": best_params["C"],
            "selected_gamma": best_params["gamma"],
            "inner_window_macro_f1": best_score,
        }
        row.update({f"window_{key}": value for key, value in window_metric.items()})
        row.update({f"record_{key}": value for key, value in record_metric.items()})
        fold_rows.append(row)
        parameter_rows.append(
            {
                "fold": fold,
                "test_subjects": row["test_subjects"],
                "C": best_params["C"],
                "gamma": best_params["gamma"],
                "inner_window_macro_f1": best_score,
            }
        )
        print(
            f"  외부 fold {fold}/{len(outer_pairs)}: 창 F1={window_metric['macro_f1']:.3f}, "
            f"기록 F1={record_metric['macro_f1']:.3f}, C={best_params['C']}, gamma={best_params['gamma']}",
            flush=True,
        )

    window_pred = oof_probability.argmax(axis=1)
    window_summary = metrics(y, window_pred)
    record_probability, record_y, aggregated_record_ids, record_subjects = aggregate_probabilities(
        oof_probability, record_ids, y, groups
    )
    record_pred = record_probability.argmax(axis=1)
    record_summary = metrics(record_y, record_pred)
    fold_frame = pd.DataFrame(fold_rows)
    summary: dict[str, object] = {
        **{f"window_{key}": value for key, value in window_summary.items()},
        **{f"record_{key}": value for key, value in record_summary.items()},
    }
    add_fold_summary("window", fold_frame, summary)
    add_fold_summary("record", fold_frame, summary)
    return {
        "summary": summary,
        "folds": fold_frame,
        "parameters": pd.DataFrame(parameter_rows),
        "inner_candidate_folds": pd.concat(inner_rows, ignore_index=True),
        "window_probability": oof_probability,
        "window_prediction": window_pred,
        "record_probability": record_probability,
        "record_y": record_y,
        "record_ids": aggregated_record_ids,
        "record_subjects": record_subjects,
        "record_prediction": record_pred,
    }


def evaluate_fixed(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    record_ids: np.ndarray,
    params: dict[str, object],
    subject_codes: dict[str, str],
    seed: int,
) -> dict[str, object]:
    outer_pairs = make_subject_pairs(groups, seed=seed)
    oof_probability = np.zeros((len(y), len(CLASS_NAMES)), dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    for fold, test_subjects in enumerate(outer_pairs, start=1):
        test_mask = np.isin(groups, test_subjects)
        train_mask = ~test_mask
        model = build_svm(params, probability=True, seed=seed)
        model.fit(X[train_mask], y[train_mask])
        probability = aligned_predict_proba(model, X[test_mask], len(CLASS_NAMES))
        oof_probability[np.flatnonzero(test_mask)] = probability
        window_metric = metrics(y[test_mask], probability.argmax(axis=1))
        record_probability, record_y, _, _ = aggregate_probabilities(
            probability, record_ids[test_mask], y[test_mask], groups[test_mask]
        )
        record_metric = metrics(record_y, record_probability.argmax(axis=1))
        row: dict[str, object] = {
            "fold": fold,
            "test_subjects": " | ".join(subject_codes[str(subject)] for subject in test_subjects),
        }
        row.update({f"window_{key}": value for key, value in window_metric.items()})
        row.update({f"record_{key}": value for key, value in record_metric.items()})
        fold_rows.append(row)

    window_pred = oof_probability.argmax(axis=1)
    record_probability, record_y, aggregated_record_ids, record_subjects = aggregate_probabilities(
        oof_probability, record_ids, y, groups
    )
    record_pred = record_probability.argmax(axis=1)
    fold_frame = pd.DataFrame(fold_rows)
    summary: dict[str, object] = {
        **{f"window_{key}": value for key, value in metrics(y, window_pred).items()},
        **{f"record_{key}": value for key, value in metrics(record_y, record_pred).items()},
    }
    add_fold_summary("window", fold_frame, summary)
    add_fold_summary("record", fold_frame, summary)
    return {
        "summary": summary,
        "folds": fold_frame,
        "window_probability": oof_probability,
        "window_prediction": window_pred,
        "record_probability": record_probability,
        "record_y": record_y,
        "record_ids": aggregated_record_ids,
        "record_subjects": record_subjects,
        "record_prediction": record_pred,
    }


def report_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).T


def probability_columns(probability: np.ndarray) -> dict[str, np.ndarray]:
    return {f"확률_{name}": probability[:, index] for index, name in enumerate(CLASS_NAMES)}


def save_oof_tables(
    output_dir: Path,
    window_seconds: int,
    window_meta: pd.DataFrame,
    subject_codes: dict[str, str],
    nested: dict[str, object],
    fixed: dict[str, object],
) -> None:
    for evaluation_name, result in (("nested", nested), ("fixed", fixed)):
        window_table = window_meta[
            ["record_id", "subject", "window_index", "start_s", "nominal_end_s", "label", "class_id"]
        ].copy()
        window_table = window_table.assign(subject_id=window_table["subject"].map(subject_codes))
        window_table.drop(columns=["subject"], inplace=True)
        window_table.rename(columns={"label": "실제_자세", "class_id": "실제_클래스_ID"}, inplace=True)
        prediction = np.asarray(result["window_prediction"], dtype=int)
        window_table["예측_클래스_ID"] = prediction
        window_table["예측_자세"] = [CLASS_NAMES[index] for index in prediction]
        for name, values in probability_columns(
            np.asarray(result["window_probability"])
        ).items():
            window_table[name] = values
        window_table.to_csv(
            output_dir / f"window_{window_seconds}s_{evaluation_name}_oof_window_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        record_table = pd.DataFrame(
            {
                "record_id": result["record_ids"],
                "subject_id": [subject_codes[str(value)] for value in result["record_subjects"]],
                "실제_클래스_ID": result["record_y"],
                "실제_자세": [CLASS_NAMES[index] for index in result["record_y"]],
                "예측_클래스_ID": result["record_prediction"],
                "예측_자세": [CLASS_NAMES[index] for index in result["record_prediction"]],
            }
        )
        for name, values in probability_columns(
            np.asarray(result["record_probability"])
        ).items():
            record_table[name] = values
        record_table.to_csv(
            output_dir / f"window_{window_seconds}s_{evaluation_name}_oof_record_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )


def save_confusion_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        path, encoding="utf-8-sig"
    )


def save_figures(
    figures_dir: Path,
    comparison: pd.DataFrame,
    fixed_results: dict[int, dict[str, object]],
    baseline_macro_f1: float | None = None,
) -> None:
    # Deployment and personal-model workers do not make plots. In particular,
    # do not invoke macOS system_profiler to build a cold font cache at import.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    for font_path in (Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"), Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf")):
        if font_path.exists():
            matplotlib.rcParams["font.family"] = FontProperties(fname=str(font_path)).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    figures_dir.mkdir(parents=True, exist_ok=True)
    chart = comparison[comparison["evaluation"] == "fixed"].copy()
    labels = [f"{int(value)}초" for value in chart["window_seconds"]]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, chart["window_macro_f1"], width, label="창 단위")
    ax.bar(x + width / 2, chart["record_macro_f1"], width, label="기록 단위 집계")
    if baseline_macro_f1 is not None:
        ax.axhline(baseline_macro_f1, linestyle="--", linewidth=1.2, label="기존 모델 내부 평가 (참고)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1")
    ax.set_title("창 길이에 따른 RBF-SVM 성능 비교")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "window_length_macro_f1_comparison.png", dpi=180)
    plt.close(fig)

    for seconds, result in fixed_results.items():
        matrix = confusion_matrix(
            result["record_y"], result["record_prediction"], labels=np.arange(len(CLASS_NAMES))
        ).astype(float)
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
        fig, ax = plt.subplots(figsize=(9, 7))
        image = ax.imshow(normalized, vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=40, ha="right")
        ax.set_yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set_xlabel("예측 자세")
        ax.set_ylabel("실제 자세")
        ax.set_title(f"{seconds}초 창 모델 기록 단위 정규화 혼동행렬")
        for row in range(len(CLASS_NAMES)):
            for col in range(len(CLASS_NAMES)):
                ax.text(col, row, f"{normalized[row, col]:.2f}", ha="center", va="center")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figures_dir / f"window_{seconds}s_record_confusion_normalized.png", dpi=180)
        plt.close(fig)


def write_json(path: Path, payload: object) -> None:
    def converter(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=converter) + "\n", encoding="utf-8")


def train_deployment(data_root: Path, output_root: Path, windows, seed: int, model_families=None, soft_voting_members=(), training_method="standard", selection_metric="loso", signal_preprocessing=None) -> None:
    """Fit only on train subjects; select on validation, evaluate once on held-out subjects."""
    from src.training_methods import validate_training_method
    validate_training_method(training_method, model_families or ('rbf_svm',), windows)
    if training_method != "standard" and model_families is None:
        raise ValueError("개선 학습 방식은 --deployment와 --models를 지정해 실행하세요.")
    from src.training_input_policy import validate_cli_training_input
    validate_cli_training_input(data_root)
    if model_families is not None:
        from src.model_comparison import train_comparison
        train_comparison(data_root, output_root, windows, seed, model_families, soft_voting_members=soft_voting_members,
                         training_method=training_method, selection_metric=selection_metric,
                         signal_preprocessing=signal_preprocessing)
        return
    if signal_preprocessing is not None and not signal_preprocessing.is_identity:
        raise ValueError("신호 전처리 옵션은 --models를 지정한 배포 학습에서만 사용할 수 있습니다.")
    artifacts = output_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for seconds in windows:
        spec = WindowSpec(window_seconds=float(seconds))
        sequences, meta, records = load_window_dataset(data_root, spec)
        groups = meta["subject"].to_numpy(dtype=str)
        subjects = np.array(sorted(set(groups)))
        if len(subjects) < 4:
            raise ValueError("배포 학습에는 학습·검증·최종 평가를 분리할 피험자 최소 4명이 필요합니다.")
        np.random.default_rng(seed).shuffle(subjects)
        holdout_count = max(1, len(subjects) // 5)
        partitions = {
            "train": subjects[2 * holdout_count:],
            "validation": subjects[holdout_count:2 * holdout_count],
            "final_evaluation": subjects[:holdout_count],
        }
        masks = {name: np.isin(groups, values) for name, values in partitions.items()}
        y = meta["class_id"].to_numpy(dtype=int)
        for name, mask in masks.items():
            missing = set(range(len(CLASS_NAMES))) - set(y[mask])
            if missing:
                raise ValueError(f"{name} 분할에 자세 데이터가 부족합니다: {', '.join(CLASS_NAMES[i] for i in sorted(missing))}. 피험자별 모든 자세를 추가하거나 분할 시드를 변경하세요.")
        # Hash the exact input files and their subject/record identities.
        source_records = [
            {"record_id": str(row.record_id), "subject": str(row.subject),
             "file": str(Path(row.path).relative_to(data_root)),
             "sha256": hashlib.sha256(Path(row.path).read_bytes()).hexdigest()}
            for row in records.itertuples()
        ]
        hashes = [row["sha256"] for row in source_records]
        if len(set(hashes)) != len(hashes):
            raise ValueError("중복된 CSV 내용이 있습니다. 데이터 누출 방지를 위해 중복 기록을 제외하세요.")
        fingerprint = hashlib.sha256(json.dumps(source_records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        X = extract_window_features(sequences)
        best_model, best_params, best_score = None, None, -1.0
        print(f"[{seconds}초] 피험자 분리: " + ", ".join(f"{name}={len(values)}" for name, values in partitions.items()), flush=True)
        for params in SVM_CANDIDATES:
            candidate = build_svm(params, probability=True, seed=seed)
            candidate.fit(X[masks["train"]], y[masks["train"]])
            prediction = aligned_predict_proba(candidate, X[masks["validation"]], len(CLASS_NAMES)).argmax(axis=1)
            score = float(f1_score(y[masks["validation"]], prediction, average="macro", zero_division=0))
            if score > best_score:
                best_model, best_params, best_score = candidate, dict(params), score
        final_mask = masks["final_evaluation"]
        prediction = aligned_predict_proba(best_model, X[final_mask], len(CLASS_NAMES)).argmax(axis=1)
        matrix = confusion_matrix(y[final_mask], prediction, labels=np.arange(len(CLASS_NAMES)))
        per_class = f1_score(y[final_mask], prediction, labels=np.arange(len(CLASS_NAMES)), average=None, zero_division=0)
        evidence = {
            "schema_version": 1, "labels": CLASS_NAMES, "dataset_fingerprint": fingerprint,
            "random_seed": seed, "source_records": source_records,
            "partitions": {name: {"subjects": sorted(map(str, values)),
                "record_ids": sorted(set(meta.loc[masks[name], "record_id"].astype(str)))}
                for name, values in partitions.items()},
            "final_evaluation_used_for_selection": False, "class_coverage": CLASS_NAMES,
            "metrics": {"macro_f1": float(per_class.mean()), "per_class_f1": dict(zip(CLASS_NAMES, per_class)), "confusion_matrix": matrix},
        }
        evidence_name = f"window_{seconds}s_evaluation.json"
        write_json(artifacts / evidence_name, evidence)
        bundle = {
            "version": 4, "selected_model": "RBF SVM", "label_names": CLASS_NAMES,
            "label_to_id": {name: index for index, name in enumerate(CLASS_NAMES)},
            "sensor_columns": SENSOR_COLUMNS, "window_seconds": seconds, "stride_seconds": seconds,
            "sampling_rate": spec.sampling_rate, "target_length": spec.target_length,
            "min_tail_fraction": spec.min_tail_fraction, "feature_names": feature_names(),
            "training_subject_count": len(partitions["train"]),
            "training_record_count": len(set(meta.loc[masks["train"], "record_id"])),
            "training_window_count": int(masks["train"].sum()), "random_seed": seed,
            "confidence_semantics": "calibrated_probability", "svm_params": best_params,
            "validation_macro_f1": best_score, "dataset_fingerprint": fingerprint,
            "evaluation_file": evidence_name, "svm_model": best_model,
        }
        joblib.dump(bundle, artifacts / f"posture_classifier_window_{seconds}s.joblib", compress=3)
        write_json(artifacts / f"window_{seconds}s_model_config.json", {key: value for key, value in bundle.items() if key != "svm_model"})
        print(f"[{seconds}초] 독립 최종 평가 Macro F1={per_class.mean():.3f}, 평가 근거 저장 완료", flush=True)


def main() -> None:
    args = parse_args()
    from src.training_methods import validate_training_method
    if args.training_method != "standard" and (not args.deployment or not args.models or args.personalized):
        raise ValueError("개선 학습 방식은 --personalized 없이 --deployment와 --models를 지정해 실행하세요.")
    validate_training_method(args.training_method, args.models or ('rbf_svm',), args.windows)
    if args.soft_voting_members and (not args.deployment or not args.models or args.personalized):
        raise ValueError("Soft voting requires --deployment and --models, without --personalized.")
    from src.signal_preprocessing import SignalPreprocessing
    signal_preprocessing = SignalPreprocessing(None if args.despike == "none" else args.despike,
                                               args.training_trim_start_seconds)
    if not signal_preprocessing.is_identity and (not args.deployment or not args.models or args.personalized):
        raise ValueError("--despike/--training-trim-start-seconds require --deployment and --models, without --personalized.")
    from src.training_input_policy import validate_cli_training_input
    validate_cli_training_input(args.data)
    if args.personalized:
        from src.personalized_training import train
        train(Path(args.data).resolve(), Path(args.output_dir).resolve(), args.seed,
              allow_retrospective=args.allow_retrospective)
        return
    if args.deployment:
        train_deployment(Path(args.data).resolve(), Path(args.output_dir).resolve(), args.windows, args.seed, args.models,
                         args.soft_voting_members or (), training_method=args.training_method,
                         selection_metric=args.selection_metric,
                         signal_preprocessing=None if signal_preprocessing.is_identity else signal_preprocessing)
        return
    if args.models:
        raise ValueError("Multi-model comparison requires --deployment to reserve final evaluation subjects.")
    data_root = Path(args.data).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    artifacts_dir = output_root / "artifacts"
    results_dir = output_root / "results" / "window_models"
    figures_dir = output_root / "figures" / "window_models"
    for directory in (artifacts_dir, results_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    original_config = {}
    if args.baseline_config:
        original_config_path = Path(args.baseline_config).expanduser().resolve()
        if not original_config_path.exists():
            raise FileNotFoundError(f"기존 모델 설정 파일을 찾지 못했습니다: {original_config_path}")
        original_config = json.loads(original_config_path.read_text(encoding="utf-8"))
    all_comparison_rows: list[dict[str, object]] = []
    fixed_results: dict[int, dict[str, object]] = {}
    model_configs: dict[int, dict[str, object]] = {}

    for seconds in args.windows:
        print(f"\n[{seconds}초 창] 데이터 구성", flush=True)
        started = time.perf_counter()
        spec = WindowSpec(window_seconds=float(seconds))
        windows, window_meta, record_meta = load_window_dataset(data_root, spec)
        y = window_meta["class_id"].to_numpy(dtype=int)
        groups = window_meta["subject"].to_numpy(dtype=str)
        make_subject_pairs(groups, seed=args.seed)
        X = extract_window_features(windows)
        record_ids = window_meta["record_id"].to_numpy(dtype=str)
        unique_subjects = sorted(set(groups))
        subject_codes = {subject: f"S{index:02d}" for index, subject in enumerate(unique_subjects, start=1)}
        print(
            f"  {len(record_meta)}개 기록 -> {len(window_meta)}개 창, 특징 shape={X.shape}",
            flush=True,
        )

        anonymized_window_meta = window_meta.drop(columns=["path", "file"]).copy()
        anonymized_window_meta = anonymized_window_meta.assign(subject=anonymized_window_meta["subject"].map(subject_codes))
        anonymized_window_meta.rename(columns={"subject": "subject_id"}, inplace=True)
        anonymized_window_meta.to_csv(
            results_dir / f"window_{seconds}s_index_anonymized.csv", index=False, encoding="utf-8-sig"
        )

        print(f"[{seconds}초 창] 중첩 피험자 교차검증", flush=True)
        nested = evaluate_nested(X, y, groups, record_ids, subject_codes, args.seed)
        nested_summary = {"window_seconds": seconds, "evaluation": "nested", **nested["summary"]}
        all_comparison_rows.append(nested_summary)
        nested["folds"].to_csv(results_dir / f"window_{seconds}s_nested_folds.csv", index=False, encoding="utf-8-sig")
        nested["parameters"].to_csv(results_dir / f"window_{seconds}s_nested_selected_params.csv", index=False, encoding="utf-8-sig")
        nested["inner_candidate_folds"].to_csv(results_dir / f"window_{seconds}s_nested_inner_candidates.csv", index=False, encoding="utf-8-sig")

        print(f"[{seconds}초 창] 전체 피험자 내부 CV로 배포 파라미터 선택", flush=True)
        final_params, final_inner_score, final_candidate_folds = select_candidate(
            X, y, groups, args.seed, n_splits=7
        )
        final_candidate_folds.to_csv(
            results_dir / f"window_{seconds}s_final_param_search.csv", index=False, encoding="utf-8-sig"
        )
        print(f"  최종 파라미터: C={final_params['C']}, gamma={final_params['gamma']}", flush=True)
        fixed = evaluate_fixed(X, y, groups, record_ids, final_params, subject_codes, args.seed)
        fixed_results[seconds] = fixed
        fixed_summary = {"window_seconds": seconds, "evaluation": "fixed", **fixed["summary"]}
        all_comparison_rows.append(fixed_summary)
        fixed["folds"].to_csv(results_dir / f"window_{seconds}s_fixed_folds.csv", index=False, encoding="utf-8-sig")

        report_frame(y, fixed["window_prediction"]).to_csv(
            results_dir / f"window_{seconds}s_fixed_window_classification_report.csv", encoding="utf-8-sig"
        )
        report_frame(fixed["record_y"], fixed["record_prediction"]).to_csv(
            results_dir / f"window_{seconds}s_fixed_record_classification_report.csv", encoding="utf-8-sig"
        )
        save_confusion_csv(
            results_dir / f"window_{seconds}s_fixed_window_confusion_matrix.csv", y, fixed["window_prediction"]
        )
        save_confusion_csv(
            results_dir / f"window_{seconds}s_fixed_record_confusion_matrix.csv", fixed["record_y"], fixed["record_prediction"]
        )
        save_oof_tables(results_dir, seconds, window_meta, subject_codes, nested, fixed)

        model = build_svm(final_params, probability=True, seed=args.seed)
        model.fit(X, y)
        bundle = {
            "version": 4,
            "selected_model": "RBF SVM",
            "label_names": CLASS_NAMES,
            "label_to_id": {name: index for index, name in enumerate(CLASS_NAMES)},
            "sensor_columns": SENSOR_COLUMNS,
            "window_seconds": seconds,
            "stride_seconds": seconds,
            "overlap_ratio": 0.0,
            "sampling_rate": spec.sampling_rate,
            "target_length": spec.target_length,
            "min_tail_fraction": spec.min_tail_fraction,
            "segmentation": "actual_time_non_overlapping_windows",
            "feature_names": feature_names(),
            "feature_configuration": "absolute_plus_frame_relative_window_features",
            "training_subject_count": len(unique_subjects),
            "training_record_count": len(record_meta),
            "training_window_count": len(window_meta),
            "random_seed": args.seed,
            "confidence_semantics": "calibrated_probability",
            "svm_params": {
                **final_params,
                "kernel": "rbf",
                "class_weight": "balanced",
                "probability": True,
            },
            "final_parameter_search_window_macro_f1": final_inner_score,
            "nested_cv_summary": nested["summary"],
            "fixed_configuration_group_cv_summary": fixed["summary"],
            "svm_model": model,
        }
        model_path = artifacts_dir / f"posture_classifier_window_{seconds}s.joblib"
        joblib.dump(bundle, model_path, compress=3)
        config = {key: value for key, value in bundle.items() if key != "svm_model"}
        config["model_file"] = model_path.name
        config["training_seconds"] = float(time.perf_counter() - started)
        write_json(artifacts_dir / f"window_{seconds}s_model_config.json", config)
        model_configs[seconds] = config

    comparison = pd.DataFrame(all_comparison_rows)
    ordered = [
        "window_seconds", "evaluation", "window_accuracy", "window_balanced_accuracy", "window_macro_f1",
        "record_accuracy", "record_balanced_accuracy", "record_macro_f1",
        "window_fold_macro_f1_mean", "window_fold_macro_f1_std",
        "record_fold_macro_f1_mean", "record_fold_macro_f1_std",
    ]
    remaining = [column for column in comparison.columns if column not in ordered]
    comparison = comparison[[column for column in ordered if column in comparison.columns] + remaining]
    comparison.to_csv(results_dir / "window_length_comparison.csv", index=False, encoding="utf-8-sig")

    baseline = {
        "full_record_fixed_macro_f1": original_config.get("fixed_configuration_group_cv_summary", {}).get("macro_f1"),
        "full_record_nested_macro_f1": original_config.get("nested_cv_summary", {}).get("macro_f1"),
        "full_record_target_length": original_config.get("target_length"),
    }
    write_json(results_dir / "baseline_full_record_metrics.json", baseline)
    write_json(results_dir / "window_model_configs_summary.json", model_configs)
    save_figures(figures_dir, comparison, fixed_results, baseline["full_record_fixed_macro_f1"])

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    write_json(results_dir / "environment_versions.json", environment)
    print("\n재학습과 저장이 완료되었습니다.", flush=True)


if __name__ == "__main__":
    main()
