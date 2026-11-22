from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .posture_core import CLASS_NAMES, SENSOR_COLUMNS, feature_names, make_subject_pairs

DEFAULT_SEED = 20260826


@dataclass(frozen=True)
class CandidateResult:
    params: dict
    score: float
    candidate_index: int


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def _optional_xgboost(seed: int) -> BaseEstimator | None:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=4,
        tree_method="hist",
        random_state=seed,
    )


def _optional_lightgbm(seed: int) -> BaseEstimator | None:
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None
    return LGBMClassifier(
        n_estimators=120,
        num_leaves=15,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.8,
        class_weight="balanced",
        verbosity=-1,
        n_jobs=4,
        random_state=seed,
    )


def _optional_catboost(seed: int) -> BaseEstimator | None:
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None
    return CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        verbose=False,
        random_seed=seed,
        thread_count=1,
        allow_writing_files=False,
        l2_leaf_reg=3,
    )


def build_screen_models(
    seed: int = DEFAULT_SEED,
    include_slow_models: bool = False,
) -> dict[str, BaseEstimator]:
    # XGBoost/LightGBM/CatBoost are evaluated in fresh subprocesses because
    # mixing several OpenMP runtimes in one long process can deadlock on some systems.
    return {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "k-NN": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5, weights="distance")),
            ]
        ),
        "RBF SVM (기본)": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=10.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "Shrinkage LDA": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=4,
            random_state=seed,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=4,
            random_state=seed,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=seed,
        ),
    }


def _run_isolated_screen_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    subject_codes: dict[str, str],
    split_seed: int,
    model_seed: int = 42,
    timeout_seconds: int = 900,
) -> tuple[dict[str, object], list[dict[str, object]]] | None:
    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="posture_booster_") as temporary:
        temporary_path = Path(temporary)
        input_path = temporary_path / "data.npz"
        output_path = temporary_path / "result.json"
        np.savez_compressed(input_path, X=X, y=y, groups=groups)
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(project_root)
            if not existing_pythonpath
            else str(project_root) + os.pathsep + existing_pythonpath
        )
        command = [
            sys.executable,
            "-m",
            "src.booster_worker",
            "--model",
            model_name,
            "--input-npz",
            str(input_path),
            "--output-json",
            str(output_path),
            "--split-seed",
            str(split_seed),
            "--model-seed",
            str(model_seed),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"{model_name}: 제한 시간 초과로 비교에서 제외했습니다.", flush=True)
            return None
        if completed.returncode != 0 or not output_path.exists():
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "알 수 없는 오류"
            print(f"{model_name}: 실행 실패로 제외했습니다. ({detail})", flush=True)
            return None
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        fold_rows = []
        for row in payload["rows"]:
            raw_subjects = row.pop("test_subjects_raw")
            row["test_subjects"] = " | ".join(subject_codes[str(s)] for s in raw_subjects)
            fold_rows.append(row)
        fold_df = pd.DataFrame(fold_rows)
        summary = {
            "model": model_name,
            "category": "전통 ML",
            "accuracy_mean": float(fold_df["accuracy"].mean()),
            "accuracy_std": float(fold_df["accuracy"].std(ddof=0)),
            "balanced_accuracy_mean": float(fold_df["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(fold_df["balanced_accuracy"].std(ddof=0)),
            "macro_f1_mean": float(fold_df["macro_f1"].mean()),
            "macro_f1_std": float(fold_df["macro_f1"].std(ddof=0)),
            "training_seconds": float(payload["training_seconds"]),
        }
        return summary, fold_rows


def run_model_screen(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    subject_codes: dict[str, str],
    seed: int = DEFAULT_SEED,
    include_slow_models: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = make_subject_pairs(groups, seed=seed)
    rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    # Run external boosting libraries before starting threaded scikit-learn models.
    # This avoids OpenMP/fork interactions seen on some Linux and macOS setups.
    isolated_models = ["XGBoost", "LightGBM"]
    if include_slow_models:
        isolated_models.extend(["Gradient Boosting", "CatBoost"])
    for model_name in isolated_models:
        result = _run_isolated_screen_model(
            model_name, X, y, groups, subject_codes, split_seed=seed
        )
        if result is None:
            continue
        summary, isolated_folds = result
        rows.append(summary)
        fold_rows.extend(isolated_folds)
        print(
            f"{model_name}: Macro F1 "
            f"{summary['macro_f1_mean']:.3f} ± {summary['macro_f1_std']:.3f}",
            flush=True,
        )

    # Use a separate fixed estimator seed to reproduce the delivered screening table.
    for model_name, prototype in build_screen_models(42, include_slow_models).items():
        start = time.perf_counter()
        current_rows = []
        for fold_idx, test_subjects in enumerate(pairs, start=1):
            test_mask = np.isin(groups, test_subjects)
            train_mask = ~test_mask
            assert set(groups[train_mask]).isdisjoint(set(groups[test_mask]))
            model = clone(prototype)
            model.fit(X[train_mask], y[train_mask])
            pred = np.asarray(model.predict(X[test_mask])).reshape(-1).astype(int)
            metrics = classification_metrics(y[test_mask], pred)
            row = {
                "model": model_name,
                "fold": fold_idx,
                "test_subjects": " | ".join(subject_codes[s] for s in test_subjects),
                **metrics,
            }
            current_rows.append(row)
            fold_rows.append(row)
        fold_df = pd.DataFrame(current_rows)
        summary = {
            "model": model_name,
            "category": "전통 ML",
            "accuracy_mean": float(fold_df["accuracy"].mean()),
            "accuracy_std": float(fold_df["accuracy"].std(ddof=0)),
            "balanced_accuracy_mean": float(fold_df["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(fold_df["balanced_accuracy"].std(ddof=0)),
            "macro_f1_mean": float(fold_df["macro_f1"].mean()),
            "macro_f1_std": float(fold_df["macro_f1"].std(ddof=0)),
            "training_seconds": float(time.perf_counter() - start),
        }
        rows.append(summary)
        print(
            f"{model_name}: Macro F1 "
            f"{summary['macro_f1_mean']:.3f} ± {summary['macro_f1_std']:.3f}",
            flush=True,
        )

    return (
        pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False),
        pd.DataFrame(fold_rows),
    )


def extra_trees_candidates() -> list[dict[str, object]]:
    return [
        {
            "n_estimators": 180,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "max_depth": None,
            "criterion": "gini",
            "bootstrap": False,
        },
        {
            "n_estimators": 220,
            "max_features": 0.30,
            "min_samples_leaf": 1,
            "max_depth": None,
            "criterion": "entropy",
            "bootstrap": False,
        },
        {
            "n_estimators": 220,
            "max_features": 0.50,
            "min_samples_leaf": 1,
            "max_depth": None,
            "criterion": "log_loss",
            "bootstrap": False,
        },
        {
            "n_estimators": 180,
            "max_features": 0.70,
            "min_samples_leaf": 1,
            "max_depth": None,
            "criterion": "gini",
            "bootstrap": False,
        },
        {
            "n_estimators": 200,
            "max_features": 0.40,
            "min_samples_leaf": 2,
            "max_depth": 12,
            "criterion": "entropy",
            "bootstrap": False,
        },
        {
            "n_estimators": 200,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "max_depth": None,
            "criterion": "gini",
            "bootstrap": True,
            "max_samples": 0.80,
        },
    ]


def svm_candidates() -> list[dict[str, object]]:
    return [
        {"C": c_value, "gamma": gamma}
        for c_value, gamma in [
            (0.3, "scale"),
            (1.0, "scale"),
            (3.0, "scale"),
            (10.0, "scale"),
            (30.0, "scale"),
            (1.0, 0.0005),
            (3.0, 0.001),
            (10.0, 0.003),
        ]
    ]


def build_extra_trees(
    params: dict[str, object], seed: int = DEFAULT_SEED
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        **params,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
    )


def build_svm(
    params: dict[str, object],
    probability: bool = False,
    seed: int = DEFAULT_SEED,
) -> Pipeline:
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


def select_candidate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    candidates: Iterable[dict[str, object]],
    builder: Callable[[dict[str, object]], BaseEstimator],
    inner_cv: GroupKFold,
) -> CandidateResult:
    best: CandidateResult | None = None
    for candidate_index, params in enumerate(candidates):
        fold_scores: list[float] = []
        for train_idx, validation_idx in inner_cv.split(X, y, groups):
            model = builder(params)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[validation_idx])
            fold_scores.append(
                float(
                    f1_score(
                        y[validation_idx], pred, average="macro", zero_division=0
                    )
                )
            )
        score = float(np.mean(fold_scores))
        if best is None or score > best.score + 1e-12:
            best = CandidateResult(dict(params), score, candidate_index)
    if best is None:
        raise RuntimeError("후보 모델이 비어 있습니다.")
    return best


def model_probabilities(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X), dtype=np.float64)
        except AttributeError:
            pass
    scores = np.asarray(model.decision_function(X), dtype=np.float64)
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def grouped_oof_probabilities(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model: BaseEstimator,
    cv: GroupKFold,
) -> np.ndarray:
    probabilities = np.zeros((len(y), len(CLASS_NAMES)), dtype=np.float64)
    for train_idx, validation_idx in cv.split(X, y, groups):
        fitted = clone(model)
        fitted.fit(X[train_idx], y[train_idx])
        probabilities[validation_idx] = model_probabilities(fitted, X[validation_idx])
    return probabilities


def choose_ensemble_weight(
    y: np.ndarray, extra_trees_probability: np.ndarray, svm_probability: np.ndarray
) -> tuple[float, float]:
    candidates = []
    for weight in np.linspace(0.0, 1.0, 21):
        combined = weight * extra_trees_probability + (1.0 - weight) * svm_probability
        pred = combined.argmax(axis=1)
        score = float(f1_score(y, pred, average="macro", zero_division=0))
        candidates.append((float(weight), score))
    return max(candidates, key=lambda item: (item[1], item[0]))


def run_nested_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    subject_codes: dict[str, str],
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    outer_pairs = make_subject_pairs(groups, seed=seed)
    oof = {
        "Extra Trees": np.zeros((len(y), len(CLASS_NAMES)), dtype=np.float64),
        "RBF SVM": np.zeros((len(y), len(CLASS_NAMES)), dtype=np.float64),
        "Soft Voting": np.zeros((len(y), len(CLASS_NAMES)), dtype=np.float64),
    }
    fold_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for fold_index, test_subjects in enumerate(outer_pairs, start=1):
        test_mask = np.isin(groups, test_subjects)
        train_mask = ~test_mask
        test_indices = np.where(test_mask)[0]
        X_train, y_train, groups_train = X[train_mask], y[train_mask], groups[train_mask]
        assert set(groups_train).isdisjoint(set(groups[test_mask]))
        inner_cv = GroupKFold(n_splits=min(3, len(np.unique(groups[train_mask]))))

        extra_trees_best = select_candidate(
            X_train,
            y_train,
            groups_train,
            extra_trees_candidates(),
            lambda p: build_extra_trees(p, seed),
            inner_cv,
        )
        svm_best = select_candidate(
            X_train,
            y_train,
            groups_train,
            svm_candidates(),
            lambda p: build_svm(p, probability=False, seed=seed),
            inner_cv,
        )

        extra_trees_inner = grouped_oof_probabilities(
            X_train,
            y_train,
            groups_train,
            build_extra_trees(extra_trees_best.params, seed),
            inner_cv,
        )
        svm_inner = grouped_oof_probabilities(
            X_train,
            y_train,
            groups_train,
            build_svm(svm_best.params, probability=False, seed=seed),
            inner_cv,
        )
        ensemble_weight, ensemble_inner_f1 = choose_ensemble_weight(
            y_train, extra_trees_inner, svm_inner
        )

        extra_trees_model = build_extra_trees(extra_trees_best.params, seed).fit(
            X_train, y_train
        )
        svm_model = build_svm(svm_best.params, probability=False, seed=seed).fit(
            X_train, y_train
        )
        extra_trees_test = extra_trees_model.predict_proba(X[test_mask])
        svm_test = model_probabilities(svm_model, X[test_mask])
        ensemble_test = (
            ensemble_weight * extra_trees_test + (1.0 - ensemble_weight) * svm_test
        )
        probabilities = {
            "Extra Trees": extra_trees_test,
            "RBF SVM": svm_test,
            "Soft Voting": ensemble_test,
        }
        for model_name, probability in probabilities.items():
            oof[model_name][test_indices] = probability
            pred = probability.argmax(axis=1)
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_index,
                    "test_subjects": " | ".join(
                        subject_codes[s] for s in test_subjects
                    ),
                    **classification_metrics(y[test_mask], pred),
                }
            )
        selected_rows.append(
            {
                "fold": fold_index,
                "test_subjects": " | ".join(subject_codes[s] for s in test_subjects),
                "extra_trees_inner_macro_f1": extra_trees_best.score,
                "svm_inner_macro_f1": svm_best.score,
                "ensemble_weight_extra_trees": ensemble_weight,
                "ensemble_inner_macro_f1": ensemble_inner_f1,
                "extra_trees_params": json.dumps(
                    extra_trees_best.params, ensure_ascii=False, sort_keys=True
                ),
                "svm_params": json.dumps(
                    svm_best.params, ensure_ascii=False, sort_keys=True
                ),
            }
        )
        print(
            f"중첩 교차검증 {fold_index}/{len(outer_pairs)}: "
            f"SVM Macro F1={fold_rows[-2]['macro_f1']:.3f}",
            flush=True,
        )

    fold_df = pd.DataFrame(fold_rows)
    summaries: list[dict[str, object]] = []
    reports: dict[str, pd.DataFrame] = {}
    confusion_matrices: dict[str, np.ndarray] = {}
    for model_name, probability in oof.items():
        pred = probability.argmax(axis=1)
        model_folds = fold_df[fold_df["model"] == model_name]
        summaries.append(
            {
                "model": model_name,
                **classification_metrics(y, pred),
                "fold_accuracy_mean": float(model_folds["accuracy"].mean()),
                "fold_accuracy_std": float(model_folds["accuracy"].std(ddof=0)),
                "fold_balanced_accuracy_mean": float(
                    model_folds["balanced_accuracy"].mean()
                ),
                "fold_balanced_accuracy_std": float(
                    model_folds["balanced_accuracy"].std(ddof=0)
                ),
                "fold_macro_f1_mean": float(model_folds["macro_f1"].mean()),
                "fold_macro_f1_std": float(model_folds["macro_f1"].std(ddof=0)),
            }
        )
        reports[model_name] = pd.DataFrame(
            classification_report(
                y,
                pred,
                labels=np.arange(len(CLASS_NAMES)),
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            )
        ).T
        confusion_matrices[model_name] = confusion_matrix(
            y, pred, labels=np.arange(len(CLASS_NAMES))
        )
    summary_df = pd.DataFrame(summaries).sort_values(
        "fold_macro_f1_mean", ascending=False
    )
    return {
        "summary": summary_df,
        "folds": fold_df,
        "selected_hyperparameters": pd.DataFrame(selected_rows),
        "oof_probabilities": oof,
        "reports": reports,
        "confusion_matrices": confusion_matrices,
        "selected_model": str(summary_df.iloc[0]["model"]),
    }


def _evaluate_deployment_configuration(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    selected_model: str,
    extra_trees_params: dict[str, object] | None,
    svm_params: dict[str, object] | None,
    ensemble_weight: float | None,
    seed: int,
) -> dict[str, float]:
    """Evaluate one fixed deployment configuration on subject-wise outer pairs."""
    probability = np.zeros((len(y), len(CLASS_NAMES)), dtype=np.float64)
    fold_rows: list[dict[str, float]] = []
    for test_subjects in make_subject_pairs(groups, seed=seed):
        test_mask = np.isin(groups, test_subjects)
        train_mask = ~test_mask
        assert set(groups[train_mask]).isdisjoint(set(groups[test_mask]))

        if selected_model == "RBF SVM":
            if svm_params is None:
                raise ValueError("RBF SVM 파라미터가 없습니다.")
            model = build_svm(svm_params, probability=True, seed=seed)
            model.fit(X[train_mask], y[train_mask])
            fold_probability = model_probabilities(model, X[test_mask])
        elif selected_model == "Extra Trees":
            if extra_trees_params is None:
                raise ValueError("Extra Trees 파라미터가 없습니다.")
            model = build_extra_trees(extra_trees_params, seed)
            model.fit(X[train_mask], y[train_mask])
            fold_probability = model_probabilities(model, X[test_mask])
        elif selected_model == "Soft Voting":
            if extra_trees_params is None or svm_params is None or ensemble_weight is None:
                raise ValueError("소프트보팅 구성 정보가 부족합니다.")
            extra_trees = build_extra_trees(extra_trees_params, seed).fit(
                X[train_mask], y[train_mask]
            )
            svm = build_svm(svm_params, probability=True, seed=seed).fit(
                X[train_mask], y[train_mask]
            )
            extra_trees_probability = model_probabilities(extra_trees, X[test_mask])
            svm_probability = model_probabilities(svm, X[test_mask])
            fold_probability = (
                float(ensemble_weight) * extra_trees_probability
                + (1.0 - float(ensemble_weight)) * svm_probability
            )
        else:
            raise ValueError(f"지원하지 않는 최종 모델 계열: {selected_model}")

        probability[test_mask] = fold_probability
        prediction = fold_probability.argmax(axis=1)
        fold_rows.append(classification_metrics(y[test_mask], prediction))

    prediction = probability.argmax(axis=1)
    summary = classification_metrics(y, prediction)
    fold_frame = pd.DataFrame(fold_rows)
    summary.update(
        {
            "fold_accuracy_mean": float(fold_frame["accuracy"].mean()),
            "fold_accuracy_std": float(fold_frame["accuracy"].std(ddof=0)),
            "fold_macro_f1_mean": float(fold_frame["macro_f1"].mean()),
            "fold_macro_f1_std": float(fold_frame["macro_f1"].std(ddof=0)),
        }
    )
    return summary


def fit_deployment_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    selected_model: str,
    nested_summary: dict[str, object],
    output_path: str | Path,
    target_length: int = 240,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Select final hyperparameters using all subjects, fit, and serialize the model.

    The serialized estimator is trained on all records. Its reported fixed-configuration
    score is obtained separately from subject-wise out-of-fold predictions and is not a
    training-set score.
    """
    n_subjects = len(np.unique(groups))
    cv = GroupKFold(n_splits=min(7, n_subjects))
    extra_trees_best: CandidateResult | None = None
    svm_best: CandidateResult | None = None
    ensemble_weight: float | None = None

    if selected_model in {"Extra Trees", "Soft Voting"}:
        extra_trees_best = select_candidate(
            X,
            y,
            groups,
            extra_trees_candidates(),
            lambda p: build_extra_trees(p, seed),
            cv,
        )
    if selected_model in {"RBF SVM", "Soft Voting"}:
        svm_best = select_candidate(
            X,
            y,
            groups,
            svm_candidates(),
            lambda p: build_svm(p, probability=False, seed=seed),
            cv,
        )

    if selected_model == "Soft Voting":
        assert extra_trees_best is not None and svm_best is not None
        extra_trees_oof = grouped_oof_probabilities(
            X, y, groups, build_extra_trees(extra_trees_best.params, seed), cv
        )
        svm_oof = grouped_oof_probabilities(
            X, y, groups, build_svm(svm_best.params, probability=False, seed=seed), cv
        )
        ensemble_weight, _ = choose_ensemble_weight(y, extra_trees_oof, svm_oof)

    fixed_summary = _evaluate_deployment_configuration(
        X,
        y,
        groups,
        selected_model,
        extra_trees_best.params if extra_trees_best is not None else None,
        svm_best.params if svm_best is not None else None,
        ensemble_weight,
        seed,
    )

    bundle: dict[str, object] = {
        "version": 3,
        "selected_model": selected_model,
        "label_names": CLASS_NAMES,
        "label_to_id": {name: index for index, name in enumerate(CLASS_NAMES)},
        "sensor_columns": SENSOR_COLUMNS,
        "target_length": target_length,
        "feature_names": feature_names(),
        "feature_configuration": "absolute_plus_frame_relative_full_record_features",
        "training_subject_count": n_subjects,
        "training_record_count": int(len(y)),
        "random_seed": seed,
        "fixed_configuration_group_cv_summary": fixed_summary,
        "nested_cv_summary": nested_summary,
    }

    if selected_model in {"Extra Trees", "Soft Voting"}:
        assert extra_trees_best is not None
        bundle["extra_trees_params"] = extra_trees_best.params
        bundle["extra_trees_model"] = build_extra_trees(
            extra_trees_best.params, seed
        ).fit(X, y)
    if selected_model in {"RBF SVM", "Soft Voting"}:
        assert svm_best is not None
        bundle["svm_params"] = {
            **svm_best.params,
            "kernel": "rbf",
            "class_weight": "balanced",
            "probability": True,
        }
        bundle["svm_model"] = build_svm(
            svm_best.params, probability=True, seed=seed
        ).fit(X, y)
    if selected_model == "Soft Voting":
        bundle["extra_trees_weight"] = float(ensemble_weight)

    joblib.dump(bundle, output_path, compress=3)
    return bundle
