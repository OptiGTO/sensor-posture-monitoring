#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classical_models import (  # noqa: E402
    DEFAULT_SEED,
    fit_deployment_model,
    run_model_screen,
    run_nested_evaluation,
)
from src.data_analysis import analyze_dataset  # noqa: E402
from src.posture_core import (  # noqa: E402
    CLASS_NAMES,
    extract_feature_vector,
    feature_names,
    load_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="7개 자세 센서 분류 모델의 전체 학습·평가 파이프라인"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="train.zip 또는 CSV가 들어 있는 데이터 디렉터리",
    )
    parser.add_argument(
        "--output-dir",
        default="training_output",
        help="결과를 저장할 디렉터리",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-length", type=int, default=240)
    parser.add_argument(
        "--include-deep",
        action="store_true",
        help="MLP, 1D CNN, TCN, Transformer 비교도 실행",
    )
    parser.add_argument(
        "--include-recurrent",
        action="store_true",
        help="딥러닝 비교에 BiGRU도 추가(계산량 증가)",
    )
    parser.add_argument(
        "--include-slow-models",
        action="store_true",
        help="Gradient Boosting과 CatBoost도 추가(계산량 크게 증가)",
    )
    return parser.parse_args()


def _safe_extract(zip_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    # This archive stores Korean UTF-8 names without the ZIP UTF-8 flag.
    # metadata_encoding prevents CP437 mojibake while remaining harmless for ASCII names.
    with zipfile.ZipFile(zip_path, metadata_encoding="utf-8") as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if "__MACOSX" in member_path.parts or member_path.name.startswith("._"):
                continue
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"안전하지 않은 ZIP 경로가 포함되어 있습니다: {member.filename}")
            archive.extract(member, destination)
    return destination


def prepare_data(data_path: Path, output_dir: Path) -> Path:
    if data_path.is_dir():
        return data_path
    if data_path.suffix.lower() != ".zip":
        raise ValueError("--data에는 ZIP 파일 또는 디렉터리를 지정해야 합니다.")
    extract_dir = output_dir / "_extracted_data"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    print("ZIP 파일을 해제하고 있습니다.", flush=True)
    return _safe_extract(data_path, extract_dir)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"JSON으로 변환할 수 없는 형식: {type(value)!r}")


def write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=json_default)


def make_subject_codes(subjects: np.ndarray) -> dict[str, str]:
    return {
        subject: f"S{index:02d}"
        for index, subject in enumerate(sorted(set(map(str, subjects))), start=1)
    }


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    results_dir = output_dir / "results"
    artifacts_dir = output_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    root = prepare_data(data_path, output_dir)
    print("데이터셋 구조와 품질을 분석하고 있습니다.", flush=True)
    analysis = analyze_dataset(root)
    write_json(results_dir / "dataset_analysis_summary.json", analysis["summary"])
    analysis["label_map"].to_csv(
        results_dir / "label_standardization_map.csv",
        index=False,
        encoding="utf-8-sig",
    )
    analysis["channel_summary"].to_csv(
        results_dir / "sensor_channel_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sequences, metadata = load_dataset(root, target_length=args.target_length)
    subject_codes = make_subject_codes(metadata["subject"].to_numpy(dtype=str))
    anonymized_records = metadata.copy()
    anonymized_records.insert(
        0, "record_id", [f"R{index:03d}" for index in range(1, len(metadata) + 1)]
    )
    anonymized_records["subject"] = anonymized_records["subject"].map(subject_codes)
    anonymized_records = anonymized_records.drop(columns=["path", "file", "raw_label"])
    anonymized_records.rename(columns={"subject": "subject_id"}, inplace=True)
    anonymized_records.to_csv(
        results_dir / "record_index_anonymized.csv",
        index=False,
        encoding="utf-8-sig",
    )
    session_anomalies = analysis["session_anomalies"].copy()
    if not session_anomalies.empty:
        session_anomalies["subject"] = session_anomalies["subject"].map(subject_codes)
    session_anomalies.to_csv(
        results_dir / "session_anomalies_anonymized.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("기록별 특징 596개를 추출하고 있습니다.", flush=True)
    X = np.stack([extract_feature_vector(sequence) for sequence in sequences])
    y = metadata["class_id"].to_numpy(dtype=int)
    groups = metadata["subject"].to_numpy(dtype=str)
    group_codes = np.asarray([subject_codes[group] for group in groups], dtype=str)
    np.savez_compressed(
        results_dir / "feature_dataset_anonymized.npz",
        X=X,
        y=y,
        groups=group_codes,
        feature_names=np.asarray(feature_names(), dtype=str),
    )

    print("전통 머신러닝 모델을 피험자 단위로 비교하고 있습니다.", flush=True)
    classical_summary, classical_folds = run_model_screen(
        X,
        y,
        groups,
        subject_codes,
        seed=args.seed,
        include_slow_models=args.include_slow_models,
    )
    classical_summary.to_csv(
        results_dir / "classical_model_screen.csv",
        index=False,
        encoding="utf-8-sig",
    )
    classical_folds.to_csv(
        results_dir / "classical_model_screen_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_comparisons = [classical_summary]
    if args.include_deep:
        print("딥러닝 모델을 피험자 단위로 비교하고 있습니다.", flush=True)
        from src.deep_models import run_deep_screen

        deep_result = run_deep_screen(
            sequences,
            y,
            groups,
            subject_codes,
            seed=args.seed,
            include_recurrent=args.include_recurrent,
        )
        deep_result["summary"].to_csv(
            results_dir / "deep_model_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        deep_result["folds"].to_csv(
            results_dir / "deep_model_folds.csv",
            index=False,
            encoding="utf-8-sig",
        )
        for model_name, history in deep_result["histories"].items():
            safe_name = model_name.replace(" ", "_")
            history.to_csv(
                results_dir / f"deep_history_{safe_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )
        all_comparisons.append(deep_result["summary"])

    comparison = pd.concat(all_comparisons, ignore_index=True).sort_values(
        "macro_f1_mean", ascending=False
    )
    comparison.to_csv(
        results_dir / "all_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Extra Trees와 RBF-SVM을 중첩 그룹 교차검증으로 튜닝하고 있습니다.", flush=True)
    nested = run_nested_evaluation(
        X, y, groups, subject_codes, seed=args.seed
    )
    nested["summary"].to_csv(
        results_dir / "nested_cv_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nested["folds"].to_csv(
        results_dir / "nested_cv_fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nested["selected_hyperparameters"].to_csv(
        results_dir / "nested_cv_selected_hyperparameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for model_name, report in nested["reports"].items():
        safe_name = model_name.replace(" ", "_")
        report.to_csv(
            results_dir / f"classification_report_{safe_name}.csv",
            encoding="utf-8-sig",
        )
        np.savetxt(
            results_dir / f"confusion_matrix_{safe_name}.csv",
            nested["confusion_matrices"][model_name],
            fmt="%d",
            delimiter=",",
        )
    np.savez_compressed(
        results_dir / "nested_cv_oof_predictions_anonymized.npz",
        y=y,
        groups=group_codes,
        **{
            model_name.replace(" ", "_"): probability
            for model_name, probability in nested["oof_probabilities"].items()
        },
    )

    selected_model = str(nested["selected_model"])
    selected_row = nested["summary"].iloc[0].to_dict()
    print(f"최종 모델 계열: {selected_model}", flush=True)
    bundle = fit_deployment_model(
        X,
        y,
        groups,
        selected_model,
        selected_row,
        artifacts_dir / "posture_classifier.joblib",
        target_length=args.target_length,
        seed=args.seed,
    )
    config = {
        key: value
        for key, value in bundle.items()
        if key not in {"extra_trees_model", "svm_model"}
    }
    write_json(artifacts_dir / "final_model_config.json", config)
    write_json(
        results_dir / "run_summary.json",
        {
            "dataset": analysis["summary"],
            "selected_model": selected_model,
            "selected_nested_cv_metrics": selected_row,
            "deep_models_included": bool(args.include_deep),
            "slow_models_included": bool(args.include_slow_models),
            "seed": args.seed,
            "target_length": args.target_length,
            "feature_count": int(X.shape[1]),
            "class_names": CLASS_NAMES,
        },
    )
    print(
        "완료: 성능 평가는 results 폴더, 배포 모델은 artifacts 폴더에 저장했습니다.",
        flush=True,
    )


if __name__ == "__main__":
    main()
