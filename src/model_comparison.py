"""Shared subject-disjoint comparison, with selection locked before final scoring."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version as package_version
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .soft_voting import validate_members, aligned_indices, evaluate as evaluate_soft_voting
from .posture_core import CLASS_NAMES, SENSOR_COLUMNS, feature_names, read_sensor_csv, sensor_identity
from .signal_preprocessing import SignalPreprocessing, bundle_entry
from .training_windows import validate_training_windows
from .window_pipeline import WindowSpec, aligned_predict_proba, extract_window_features, load_window_dataset
from .model_catalog import (DEFAULT_MODEL_FAMILIES, MODEL_NAMES, MODEL_PACKAGES,
    input_contract, missing_model_packages, model_input, model_display_name,
    pipeline_runtime_packages, missing_pipeline_packages)
from .training_methods import (build_method_model, fit_method_model, method_candidates,
    method_details, validate_training_method)
MODEL_CANDIDATES = {
    "rbf_svm": [{"C": c, "gamma": "scale"} for c in (0.3, 1.0, 3.0, 10.0, 30.0)]
        + [{"C": 1.0, "gamma": 0.0005}, {"C": 3.0, "gamma": 0.001}, {"C": 10.0, "gamma": 0.003}],
    "logistic_regression": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "random_forest": [{"n_estimators": 200, "min_samples_leaf": leaf, "max_features": features}
        for leaf in (1, 3) for features in ("sqrt", 0.5)],
    "extra_trees": [{"n_estimators": 200, "min_samples_leaf": leaf, "max_features": features}
        for leaf in (1, 3) for features in ("sqrt", 0.5)],
    "catboost": [{"iterations": 200, "depth": depth, "learning_rate": 0.05, "l2_leaf_reg": 5.0}
        for depth in (4, 6)],
    "xgboost": [{"n_estimators": 200, "max_depth": depth, "learning_rate": 0.05,
        "min_child_weight": 3, "reg_lambda": 5.0, "subsample": 0.8, "colsample_bytree": 0.8}
        for depth in (3, 5)],
    "minirocket": [{"num_kernels": 1008, "C": c} for c in (0.1, 1.0)],
    "cnn1d": [{"hidden_size": width, "epochs": 40, "patience": 8} for width in (16, 32)],
    "mlp": [{"hidden_size": width, "epochs": 60, "patience": 10} for width in (32, 64)],
}


def validate_model_families(families: Iterable[str], *, training_method: str = "standard") -> tuple[str, ...]:
    result = tuple(dict.fromkeys(families))
    from .checkpoint_registry import get_checkpoint, is_checkpoint
    allowed = {get_checkpoint(training_method)["pipeline"]["family"]} if is_checkpoint(training_method) else MODEL_NAMES
    if not result or any(item not in allowed for item in result):
        raise ValueError("Select at least one supported model: " + ", ".join(MODEL_NAMES))
    return result


def build_model(family: str, params: dict, seed: int):
    if family == "rbf_svm":
        estimator = SVC(**params, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
    elif family == "logistic_regression":
        estimator = LogisticRegression(**params, max_iter=5000, class_weight="balanced", random_state=seed)
    elif family == "random_forest":
        estimator = RandomForestClassifier(**params, class_weight="balanced", random_state=seed, n_jobs=1)
    elif family == "extra_trees":
        estimator = ExtraTreesClassifier(**params, class_weight="balanced", random_state=seed, n_jobs=1)
    elif family == "catboost":
        from catboost import CatBoostClassifier
        estimator = CatBoostClassifier(**params, loss_function="MultiClass", auto_class_weights="Balanced",
            random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False)
    elif family == "xgboost":
        from xgboost import XGBClassifier
        estimator = XGBClassifier(**params, objective="multi:softprob", eval_metric="mlogloss",
            tree_method="hist", random_state=seed, n_jobs=1)
    elif family == "minirocket":
        from .extended_models import MiniRocketClassifier
        return MiniRocketClassifier(**params, random_state=seed)
    elif family in {"cnn1d", "mlp"}:
        from .extended_models import SmallNeuralClassifier
        return SmallNeuralClassifier(architecture=family, **params, random_state=seed)
    else:
        raise ValueError(f"Unsupported model family: {family}")
    # Keep the existing feature preprocessing contract. Trees use the same
    # train-fitted scaler so every candidate sees exactly the same inputs.
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def scores(y: np.ndarray, prediction: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=np.arange(len(CLASS_NAMES)), zero_division=0,
    )
    return {
        "macro_f1": float(f1.mean()), "accuracy": float(accuracy_score(y, prediction)),
        "worst_class_f1": float(f1.min()),
        "per_class_f1": dict(zip(CLASS_NAMES, f1.tolist())),
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(CLASS_NAMES)},
        "confusion_matrix": confusion_matrix(y, prediction, labels=np.arange(len(CLASS_NAMES))).tolist(),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _dataset(data_root: Path, seconds: int, seed: int,
        fixed_subject_partitions: dict[str, Iterable[str]] | None = None,
        signal_preprocessing: SignalPreprocessing | None = None) -> dict:
    spec = WindowSpec(window_seconds=float(seconds))
    preprocessing = signal_preprocessing or SignalPreprocessing()
    # Evaluation rows (validation, final, LOSO held-out) see exactly the
    # inference transform; only fitted rows may additionally be trimmed.
    sequences, meta, records = load_window_dataset(
        data_root, spec, preprocessing.apply_inference if preprocessing.despike else None)
    groups = meta["subject"].to_numpy(dtype=str)
    subjects = np.array(sorted(set(groups)))
    if len(subjects) < 4:
        raise ValueError("At least four subjects are required for train/validation/final evaluation.")
    if fixed_subject_partitions is None:
        np.random.default_rng(seed).shuffle(subjects)
        holdout_count = max(1, len(subjects) // 5)
        partition_subjects = {"train": subjects[2 * holdout_count:],
            "validation": subjects[holdout_count:2 * holdout_count], "final_evaluation": subjects[:holdout_count]}
    else:
        if set(fixed_subject_partitions) != {"train", "validation", "final_evaluation"}:
            raise ValueError("Fixed partitions must contain train, validation and final_evaluation.")
        partition_subjects = {name: np.array(list(members), dtype=str)
                              for name, members in fixed_subject_partitions.items()}
        assigned = [str(subject) for members in partition_subjects.values() for subject in members]
        if len(assigned) != len(set(assigned)):
            raise ValueError("A subject occurs more than once in fixed partitions.")
        if set(assigned) != set(subjects):
            raise ValueError("Fixed partitions must cover exactly the subjects in the selected dataset.")
        if any(not len(members) for members in partition_subjects.values()):
            raise ValueError("Every fixed subject partition must be nonempty.")
    masks = {name: np.isin(groups, values) for name, values in partition_subjects.items()}
    y = meta["class_id"].to_numpy(dtype=int)
    for name, mask in masks.items():
        missing = set(range(len(CLASS_NAMES))) - set(y[mask])
        if missing:
            raise ValueError(f"{name} lacks classes: " + ", ".join(CLASS_NAMES[i] for i in sorted(missing)))
    source_records, content_hashes = [], {}
    for row in records.itertuples():
        path = Path(row.path)
        frame = read_sensor_csv(path)
        # Detect identical pressure recordings even with relabeled files or a
        # shifted starting timestamp; reject both duplicate and conflicting labels.
        content_hash = sensor_identity(frame)
        if content_hash in content_hashes:
            raise ValueError(f"Duplicate sensor recording: {path.name} and {content_hashes[content_hash]}")
        content_hashes[content_hash] = str(path)
        source_records.append({"record_id": str(row.record_id), "subject": str(row.subject),
            "file": str(path.relative_to(data_root)), "label": str(row.label),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sensor_content_sha256": content_hash,
            "duration_seconds": float(row.duration_s), "window_count": int(row.window_count)})
    fingerprint = hashlib.sha256(json.dumps(source_records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    partitions = {name: {"subjects": sorted(map(str, values)),
        "record_ids": sorted(set(meta.loc[masks[name], "record_id"].astype(str))),
        "window_count": int(masks[name].sum()),
        "class_window_counts": {label: int(np.sum(y[masks[name]] == i)) for i, label in enumerate(CLASS_NAMES)}}
        for name, values in partition_subjects.items()}
    X = extract_window_features(sequences)
    fit = {"sequences": sequences, "meta": meta, "X": X, "y": y, "masks": masks}
    if preprocessing.training_trim_start_seconds:
        fit_sequences, fit_meta, fit_records = load_window_dataset(data_root, spec, preprocessing.apply_training)
        if (fit_records["record_id"].tolist(), fit_records["path"].tolist()) != (
                records["record_id"].tolist(), records["path"].tolist()):
            raise ValueError("Trimmed training rows do not map to the same source records.")
        fit_groups = fit_meta["subject"].to_numpy(dtype=str)
        fit_y = fit_meta["class_id"].to_numpy(dtype=int)
        fit_masks = {name: np.isin(fit_groups, values) for name, values in partition_subjects.items()}
        missing = set(range(len(CLASS_NAMES))) - set(fit_y[fit_masks["train"]])
        if missing:
            raise ValueError("Trimmed train rows lack classes: " + ", ".join(CLASS_NAMES[i] for i in sorted(missing)))
        for name, mask in fit_masks.items():
            partitions[name]["fit_window_count"] = int(mask.sum())
        fit = {"sequences": fit_sequences, "meta": fit_meta, "X": extract_window_features(fit_sequences),
               "y": fit_y, "masks": fit_masks}
    return {"spec": spec, "sequences": sequences, "meta": meta, "X": X,
        "y": y, "masks": masks, "partitions": partitions, "source_records": source_records,
        "fingerprint": fingerprint, "fit": fit, "signal_preprocessing": preprocessing}


SELECTION_METRICS = ("loso", "validation")
DEFAULT_SELECTION_METRIC = "loso"


def validate_selection_metric(value: str) -> str:
    if value not in SELECTION_METRICS:
        raise ValueError("Selection metric must be one of: " + ", ".join(SELECTION_METRICS))
    return value


def loso_evaluation(family: str, params: dict, seed: int, training_method: str, X, y, subjects, record_ids,
                    development_mask, *, check_cancel=lambda: None, emit=lambda _message: None,
                    fit: tuple | None = None) -> dict:
    """Leave-one-subject-out over development subjects only (train + validation).

    Every development subject is predicted once by a model fitted on the others.
    Final-evaluation subjects are never read. Parameters are fixed by the caller;
    neural models receive no external validation data inside a fold. ``fit`` is an
    optional ``(X, y, subjects, record_ids, development_mask)`` of training rows
    (e.g. start-trimmed recordings); by default the evaluation rows are fitted.
    """
    rows = np.flatnonzero(development_mask)
    people = sorted(set(map(str, subjects[rows])))
    if len(people) < 2:
        raise ValueError("LOSO requires at least two development subjects.")
    if fit is None:
        fit_X, fit_y, fit_subjects, fit_record_ids, fit_rows = X, y, subjects, record_ids, rows
    else:
        fit_X, fit_y, fit_subjects, fit_record_ids, fit_mask = fit
        fit_rows = np.flatnonzero(fit_mask)
        if sorted(set(map(str, fit_subjects[fit_rows]))) != people:
            raise ValueError("LOSO training rows must cover exactly the development subjects.")
    probability = np.full((len(y), len(CLASS_NAMES)), np.nan)
    for index, person in enumerate(people, 1):
        check_cancel()
        held = rows[subjects[rows] == person]
        train = fit_rows[fit_subjects[fit_rows] != person]
        model = build_method_model(family, params, seed, training_method)
        fit_method_model(model, family, training_method, fit_X[train], fit_y[train], groups=fit_subjects[train],
                         record_ids=fit_record_ids[train], validation_data=None)
        probability[held] = aligned_predict_proba(model, X[held], len(CLASS_NAMES))
        emit(f"  LOSO {index}/{len(people)}")
    if np.isnan(probability[rows]).any():
        raise ValueError("LOSO did not predict every development window exactly once.")
    prediction = probability[rows].argmax(axis=1)
    result = scores(y[rows], prediction)
    held_subjects = subjects[rows]
    result["per_subject_macro_f1"] = {person: scores(y[rows][held_subjects == person],
        prediction[held_subjects == person])["macro_f1"] for person in people}
    result.update(subjects=len(people), windows=int(len(rows)), scope="development subjects (train + validation)")
    return result


def train_comparison(data_root: Path, output_root: Path, windows: Iterable[int], seed: int,
        model_families: Iterable[str] = DEFAULT_MODEL_FAMILIES, *,
        progress: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        fixed_subject_partitions: dict[str, Iterable[str]] | None = None,
        soft_voting_members: Iterable[str] = (), training_method: str = "standard",
        selection_metric: str = DEFAULT_SELECTION_METRIC,
        signal_preprocessing: SignalPreprocessing | None = None) -> dict:
    """Fit train subjects, tune validation, then report final results once per winner.

    Parameters within a family are tuned on the validation subjects. The winning
    model/window is chosen by ``selection_metric``: LOSO over all development
    subjects (default) or the single validation split. All choices are locked
    before the final partition is predicted; final scores never update them.
    Cancellation between fits is supported here; TrainingJob can terminate an
    active fit in its isolated worker process.
    """
    selection_metric = validate_selection_metric(selection_metric)
    if signal_preprocessing is not None and not isinstance(signal_preprocessing, SignalPreprocessing):
        raise ValueError("signal_preprocessing must be a SignalPreprocessing instance.")
    signal_entry = bundle_entry(signal_preprocessing)
    families = validate_model_families(model_families, training_method=training_method)
    windows = validate_training_windows(windows)
    method_entry = validate_training_method(training_method, families, windows)
    checkpoint_pipeline = method_entry.get("checkpoint", {}).get("pipeline")
    family_packages = {family: pipeline_runtime_packages(checkpoint_pipeline) if checkpoint_pipeline
                       else MODEL_PACKAGES.get(family, {}) for family in families}
    from .training_input_policy import validate_cli_training_input
    validate_cli_training_input(data_root)
    members = validate_members(soft_voting_members, families, windows)
    if fixed_subject_partitions is not None:
        fixed_subject_partitions = {name: tuple(members) for name, members in fixed_subject_partitions.items()}
    artifacts = Path(output_root) / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    comparison_path = artifacts / "model_comparison.json"
    if comparison_path.exists() or any(artifacts.glob("*_model_config.json")):
        raise ValueError("Training output already exists. Use a new run directory.")
    emit = progress or (lambda message: print(message, flush=True))
    report = {"schema_version": 1, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "model_families": list(families), "windows": list(windows), "labels": list(CLASS_NAMES),
        "training_method": training_method, "training_method_details": method_details(training_method),
        "training_method_evidence": method_entry["evidence"],
        "selection_metric": selection_metric,
        "selection_policy": ("parameters by validation Macro F1; model/window by development-subject LOSO Macro F1; "
                             "ties use worst-class F1 then inference latency") if selection_metric == "loso"
                            else "highest validation Macro F1; ties use worst-class F1 then inference latency",
        "partition_policy": "explicit fixed subjects" if fixed_subject_partitions is not None else "seeded subject shuffle",
        "final_evaluation_used_for_selection": False, "selection_locked": False,
        "preprocessing": {"features": feature_names(), "sensor_columns": list(SENSOR_COLUMNS),
            "resampling": "linear_interpolation_on_elapsed_s", "scaler": "StandardScaler fit on train subjects only",
            "input_by_family": {family: input_contract(family) for family in families},
            "normalization_policy": "feature scaler, MiniRocket transform/readout, and neural normalization fit on train subjects only",
            "personal_calibration_used_as_model_input": False, **signal_entry},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "datasets": {}, "candidates": [], "models": [], "failures": []}
    if "checkpoint" in method_entry:
        pipeline = method_entry["checkpoint"]["pipeline"]
        report["checkpoint"] = method_entry["checkpoint"]
        report["preprocessing"].update(
            scaler=pipeline["preprocessing"]["scaler"] + " fit on train subjects only",
            checkpoint_pipeline=pipeline,
            normalization_policy="exact checkpoint pipeline; every fitted transform uses train subjects only")
    def check_cancel() -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError("Model comparison cancelled")
    def persist() -> None:
        write_json(comparison_path, report)
    datasets, fitted = {}, []
    try:
        for family in families:
            missing = missing_pipeline_packages(checkpoint_pipeline) if checkpoint_pipeline else missing_model_packages(family)
            if missing:
                raise RuntimeError(f"{model_display_name(family)} requires {', '.join(missing)}. Install requirements.txt first.")
            for package in family_packages[family]:
                report["environment"][package] = package_version(package)
        # Prepare all windows before any fit: unusable input cannot silently
        # change the dataset between model families or window lengths.
        for seconds in windows:
            check_cancel()
            emit(f"[{seconds}s] Loading and auditing shared dataset")
            data = datasets[seconds] = _dataset(Path(data_root), seconds, seed, fixed_subject_partitions,
                                                signal_preprocessing)
            report["datasets"][str(seconds)] = {"dataset_fingerprint": data["fingerprint"],
                "partitions": data["partitions"], "source_records": data["source_records"]}
            emit(f"[{seconds}s] {len(data['source_records'])} records, {len(data['y'])} windows; "
                + ", ".join(f"{name}={len(value['subjects'])} subjects" for name, value in data["partitions"].items()))
        if members:
            for split in ("validation", "final_evaluation"):
                aligned_indices(datasets, members, split)
            report["soft_voting"] = {"members": list(members), "weights": [1 / len(members)] * len(members),
                "experimental_only": True, "alignment": "record_id + nominal_end_s intersection",
                "tie_policy": "lowest class ID", "models": [], "counts": {}}
        total = sum(len(method_candidates(family, training_method)) for family in families) * len(windows)
        step = 0
        for seconds, data in datasets.items():
            y, masks = data["y"], data["masks"]
            fit = data["fit"]
            for family in families:
                X = model_input(family, data["sequences"], data["X"])
                X_fit = X if fit["X"] is data["X"] else model_input(family, fit["sequences"], fit["X"])
                best = None
                for params in method_candidates(family, training_method):
                    check_cancel()
                    step += 1
                    emit(f"[{step}/{total}] {model_display_name(family)} {seconds}s parameters={params}")
                    started = time.perf_counter()
                    try:
                        model = build_method_model(family, params, seed, training_method)
                        fit_method_model(model, family, training_method, X_fit[fit["masks"]["train"]],
                            fit["y"][fit["masks"]["train"]],
                            groups=fit["meta"].loc[fit["masks"]["train"], "subject"].to_numpy(),
                            record_ids=fit["meta"].loc[fit["masks"]["train"], "record_id"].to_numpy(),
                            validation_data=(X[masks["validation"]], y[masks["validation"]]))
                        validation = scores(y[masks["validation"]], aligned_predict_proba(model, X[masks["validation"]], len(CLASS_NAMES)).argmax(axis=1))
                        candidate = {"family": family, "window_seconds": seconds, "params": dict(params),
                            "training_method": training_method,
                            "validation": validation, "training_seconds": time.perf_counter() - started,
                            "input_representation": input_contract(family),
                            "training_details": getattr(model, "training_details_", {})}
                        report["candidates"].append(candidate)
                        key = (validation["macro_f1"], validation["worst_class_f1"])
                        if best is None or key > best[0]:
                            best = (key, model, candidate)
                        emit(f"  validation Macro F1={validation['macro_f1']:.4f}")
                    except (ValueError, RuntimeError, FloatingPointError, ImportError, OSError) as error:
                        failure = {"family": family, "window_seconds": seconds, "params": dict(params), "reason": str(error)}
                        report["failures"].append(failure)
                        emit(f"  FAILED: {error}")
                    persist()
                if best is None:
                    continue
                _, model, candidate = best
                # Single-window latency includes feature extraction and inference.
                probe = data["sequences"][np.flatnonzero(masks["validation"])[0]:][:1]
                aligned_predict_proba(model, model_input(family, probe), len(CLASS_NAMES))
                elapsed = []
                for _ in range(20):
                    check_cancel()
                    started = time.perf_counter()
                    aligned_predict_proba(model, model_input(family, probe), len(CLASS_NAMES))
                    elapsed.append((time.perf_counter() - started) * 1000)
                candidate = {**candidate, "name": model_display_name(family),
                    "inference_ms_median": float(np.median(elapsed)), "inference_ms_p95": float(np.percentile(elapsed, 95)),
                    "inference_scope": "single pre-resampled window: feature extraction plus probability prediction"}
                if selection_metric == "loso":
                    emit(f"{model_display_name(family)} {seconds}s LOSO over development subjects")
                    subjects = data["meta"]["subject"].to_numpy(dtype=str)
                    development = masks["train"] | masks["validation"]
                    loso_fit = None if fit["X"] is data["X"] else (X_fit, fit["y"],
                        fit["meta"]["subject"].to_numpy(dtype=str), fit["meta"]["record_id"].to_numpy(dtype=str),
                        fit["masks"]["train"] | fit["masks"]["validation"])
                    candidate["loso"] = loso_evaluation(family, candidate["params"], seed, training_method, X, y,
                        subjects, data["meta"]["record_id"].to_numpy(dtype=str), development,
                        check_cancel=check_cancel, emit=emit, fit=loso_fit)
                    emit(f"  LOSO Macro F1={candidate['loso']['macro_f1']:.4f}")
                    persist()
                fitted.append((seconds, family, model, candidate))
        check_cancel()
        if not fitted:
            raise ValueError("Every model failed; inspect the comparison failure reasons.")
        missing = [(seconds, family) for seconds in windows for family in families
            if not any(item[0] == seconds and item[1] == family for item in fitted)]
        if missing:
            raise ValueError("Comparison incomplete; no valid candidate for "
                + ", ".join(f"{model_display_name(family)} {seconds}s" for seconds, family in missing))
        if members:
            probabilities = {}
            for seconds, family, model, _ in fitted:
                check_cancel()
                member = f"{family}:{seconds}"
                if member in members:
                    data = datasets[seconds]
                    probabilities[member] = aligned_predict_proba(model,
                        model_input(family, data["sequences"], data["X"])[data["masks"]["validation"]], len(CLASS_NAMES))
            rows, evidence = evaluate_soft_voting(datasets, members, probabilities, "validation", scores)
            report["soft_voting"]["models"] = rows
            report["soft_voting"]["counts"]["validation"] = len(evidence["identities"])
            write_json(artifacts / "soft_voting_validation.json", evidence)
        final_probabilities = {}
        primary = "loso" if selection_metric == "loso" else "validation"
        selected = max(fitted, key=lambda item: (item[3][primary]["macro_f1"],
            item[3][primary]["worst_class_f1"], -item[3]["inference_ms_median"]))
        report["selected"] = {"family": selected[1], "window_seconds": selected[0],
            "training_method": training_method, "selection_metric": selection_metric,
            "name": selected[3]["name"], "params": selected[3]["params"],
            "validation_macro_f1": selected[3]["validation"]["macro_f1"],
            **({"loso_macro_f1": selected[3]["loso"]["macro_f1"]} if primary == "loso" else {})}
        report["selection_locked"] = True
        report["selection_locked_at"] = datetime.now(timezone.utc).isoformat()
        if members:
            report["soft_voting"]["selection_locked_at"] = report["selection_locked_at"]
        persist()
        emit(f"Selection LOCKED before final evaluation: {selected[3]['name']} {selected[0]}s")
        for seconds, family, model, row in fitted:
            check_cancel()
            data = datasets[seconds]
            final = data["masks"]["final_evaluation"]
            final_input = model_input(family, data["sequences"], data["X"])[final]
            final_probability = aligned_predict_proba(model, final_input, len(CLASS_NAMES))
            if f"{family}:{seconds}" in members:
                final_probabilities[f"{family}:{seconds}"] = final_probability
            final_prediction = final_probability.argmax(axis=1)
            final_scores = scores(data["y"][final], final_prediction)
            base = f"window_{seconds}s_{family}"
            evidence_name = f"{base}_evaluation.json"
            evidence = {"schema_version": 1, "labels": list(CLASS_NAMES),
                "training_method": training_method, "training_method_details": method_details(training_method),
                "dataset_fingerprint": data["fingerprint"], "random_seed": seed,
                "source_records": data["source_records"], "partitions": data["partitions"],
                "final_evaluation_used_for_selection": False, "class_coverage": list(CLASS_NAMES),
                "selection_locked_at": report["selection_locked_at"], "selected_params": row["params"],
                "metrics": final_scores, "validation_metrics": row["validation"],
                "selection_metric": selection_metric, **signal_entry,
                **({"loso_metrics": row["loso"]} if "loso" in row else {}),
                "per_subject": {subject: scores(data["y"][final][data["meta"].loc[final, "subject"].to_numpy() == subject],
                    final_prediction[data["meta"].loc[final, "subject"].to_numpy() == subject])
                    for subject in data["partitions"]["final_evaluation"]["subjects"]}}
            write_json(artifacts / evidence_name, evidence)
            spec = data["spec"]
            bundle = {"version": 5, "selected_model": model_display_name(family), "model_family": family,
                "training_method": training_method, "training_method_details": method_details(training_method),
                "training_method_evidence": method_entry["evidence"],
                "label_names": list(CLASS_NAMES), "label_to_id": {label: i for i, label in enumerate(CLASS_NAMES)},
                "sensor_columns": list(SENSOR_COLUMNS), "window_seconds": seconds, "stride_seconds": seconds,
                "sampling_rate": spec.sampling_rate, "target_length": spec.target_length,
                "min_tail_fraction": spec.min_tail_fraction,
                "feature_names": feature_names() if input_contract(family) == "feature_vector" else [],
                "input_representation": input_contract(family),
                "feature_configuration": "absolute_plus_frame_relative_window_features" if input_contract(family) == "feature_vector" else "resampled_sensor_sequence",
                "runtime_packages": dict(family_packages[family]),
                "estimator_backend": "pytorch" if family in {"cnn1d", "mlp"} else "sklearn_compatible",
                "training_details": row["training_details"],
                "training_subject_count": len(data["partitions"]["train"]["subjects"]),
                "training_record_count": len(data["partitions"]["train"]["record_ids"]),
                "training_window_count": data["partitions"]["train"].get(
                    "fit_window_count", data["partitions"]["train"]["window_count"]),
                **signal_entry,
                "random_seed": seed, "confidence_semantics": "score", "model_params": row["params"],
                "validation_macro_f1": row["validation"]["macro_f1"],
                "dataset_fingerprint": data["fingerprint"], "evaluation_file": evidence_name,
                "model_file": f"posture_classifier_{base}.joblib", "model": model}
            if family == "rbf_svm":
                bundle.update(svm_model=model, svm_params=row["params"])
            elif family == "extra_trees":
                bundle["extra_trees_model"] = model
            artifact = artifacts / bundle["model_file"]
            joblib.dump(bundle, artifact, compress=3)
            config = {key: value for key, value in bundle.items() if key not in {"model", "svm_model", "extra_trees_model"}}
            config.update(inference_ms_median=row["inference_ms_median"], model_size_bytes=artifact.stat().st_size)
            write_json(artifacts / f"{base}_model_config.json", config)
            report["models"].append({**row, "final_evaluation": final_scores, "artifact_file": artifact.name,
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "config_file": f"{base}_model_config.json", "model_size_bytes": artifact.stat().st_size,
                "selected": family == selected[1] and seconds == selected[0]})
            emit(f"{model_display_name(family)} {seconds}s final Macro F1={final_scores['macro_f1']:.4f}; "
                f"{row['inference_ms_median']:.2f} ms/window; {artifact.stat().st_size} bytes")
            persist()
        if members:
            check_cancel()
            rows, evidence = evaluate_soft_voting(datasets, members, final_probabilities, "final_evaluation", scores)
            for existing, final_row in zip(report["soft_voting"]["models"], rows):
                existing["final_evaluation"] = final_row["final_evaluation"]
            report["soft_voting"]["counts"]["final_evaluation"] = len(evidence["identities"])
            write_json(artifacts / "soft_voting_final_evaluation.json", evidence)
            emit(f"Equal soft voting completed: {len(members)} members, {len(evidence['identities'])} aligned final points")
        report["status"] = "completed_with_failures" if report["failures"] else "completed"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        persist()
        _write_markdown(artifacts / "model_comparison.md", report)
        return report
    except BaseException as error:
        report["status"] = "cancelled" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed"
        report["error"] = str(error)
        persist()
        raise


def _write_markdown(path: Path, report: dict) -> None:
    lines = ["# Model comparison", "",
        ("Model/window selection was locked using development-subject LOSO Macro F1 before final evaluation."
         if report.get("selection_metric") == "loso"
         else "Selection was locked using validation Macro F1 before final evaluation."), "",
        f"Training method: {report.get('training_method', 'standard')}", "",
        "| Model | Window | LOSO Macro F1 | Validation Macro F1 | Final Macro F1 | Final accuracy | Median inference (ms) | Size (bytes) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["models"]:
        name = row["name"] + (" (selected)" if row["selected"] else "")
        loso = f"{row['loso']['macro_f1']:.4f}" if "loso" in row else "-"
        lines.append(f"| {name} | {row['window_seconds']}s | {loso} | {row['validation']['macro_f1']:.4f} | "
            f"{row['final_evaluation']['macro_f1']:.4f} | {row['final_evaluation']['accuracy']:.4f} | "
            f"{row['inference_ms_median']:.2f} | {row['model_size_bytes']} |")
    lines += ["", "Per-class precision, recall, F1, confusion matrices, per-subject errors, exact data hashes, split, "
        "candidate parameters and environment versions are in model_comparison.json and the evaluation JSON files.", "",
        "Final scores describe this single reserved subject split; they must not be used to retune the candidates. "
        "Inference timing excludes serial I/O and GUI rendering. Personal-reference system performance requires separate "
        "registration and evaluation sessions and is not measured by these model-only results.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
