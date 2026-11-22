"""Evidence checks for deployment; model quality never acts as a pass threshold."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ModelDescriptor
from .schema_validation import read_json_bytes


POSTURE_SEMANTICS = {
    "정자세": "upright", "앞으로 숙이기": "lean_forward", "뒤로 기대기": "lean_back",
    "왼쪽 기울임": "lean_left", "오른쪽 기울임": "lean_right",
    "왼쪽 다리 꼬기": "cross_left_leg", "오른쪽 다리 꼬기": "cross_right_leg",
}


def evidence_path(descriptor: ModelDescriptor, base: Path, value: str) -> Path:
    path = (base / value).resolve()
    root = Path(descriptor.metadata["_package_root"]).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"검증 근거가 artifact 패키지 밖에 있거나 존재하지 않습니다: {value}")
    components = {Path(item["path"]).resolve(): item for item in descriptor.metadata.get("_components", ())}
    if path not in components:
        raise ValueError(f"검증 근거의 구성요소 해시가 없습니다: {value}")
    # Check again immediately before reading, including when called without L2.
    from .validation import digest
    if digest(path) != components[path]["sha256"]:
        raise ValueError(f"검증 근거 해시가 다릅니다: {value}")
    return path


def check_evaluation(descriptor: ModelDescriptor, evaluation: dict[str, Any], base: Path) -> None:
    if evaluation.get("integrity_status") != "passed" or not evaluation.get("evidence_files"):
        raise ValueError("평가 데이터 분리 무결성과 평가 manifest 근거가 필요합니다. 독립 평가 근거가 없는 기존 후보는 학습 마법사에서 다시 학습하세요.")
    reports = [read_json_bytes(evidence_path(descriptor, base, str(value)).read_bytes()) for value in evaluation["evidence_files"]]
    for report in reports:
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            raise ValueError("평가 manifest 버전이 올바르지 않습니다")
        if tuple(report.get("labels", ())) != descriptor.labels:
            raise ValueError("평가 manifest의 라벨 순서가 다릅니다")
        from .validation import SHA256_PATTERN
        fingerprint = report.get("dataset_fingerprint", "")
        if not isinstance(fingerprint, str) or not SHA256_PATTERN.fullmatch(fingerprint) or fingerprint != descriptor.metadata.get("_provenance", {}).get("dataset_fingerprint"):
            raise ValueError("평가 데이터 provenance가 모델과 다릅니다")
        partitions = report.get("partitions", {})
        for key in ("subjects", "record_ids"):
            groups = []
            for name in ("train", "validation", "final_evaluation"):
                values = partitions.get(name, {}).get(key)
                if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value.strip() for value in values) or len(set(values)) != len(values):
                    raise ValueError(f"평가 manifest의 {name}.{key}가 불완전합니다")
                groups.append(set(values))
            if any(left & right for left, right in combinations(groups, 2)):
                raise ValueError(f"Train/Validation/Final Evaluation 사이 {key} 누출이 있습니다")
        if report.get("final_evaluation_used_for_selection") is not False:
            raise ValueError("Final Evaluation이 모델 선택에 사용되지 않았다는 근거가 필요합니다")
        if set(report.get("class_coverage", ())) != set(descriptor.labels):
            raise ValueError("평가 클래스 coverage가 불완전합니다")
        metrics = report.get("metrics", {})
        per_class = metrics.get("per_class_f1", {})
        if set(per_class) != set(descriptor.labels):
            raise ValueError("클래스별 F1 결과가 불완전합니다")
        scores = np.asarray([metrics.get("macro_f1"), *per_class.values()], dtype=float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("평가 지표가 유효한 유한값이 아닙니다")
        matrix = np.asarray(metrics.get("confusion_matrix"), dtype=float)
        if matrix.shape != (len(descriptor.labels), len(descriptor.labels)) or not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix != np.floor(matrix)).any() or (matrix.sum(axis=1) <= 0).any():
            raise ValueError("평가 confusion matrix가 유효하지 않습니다")
        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
        expected_f1 = np.divide(2 * matrix.diagonal(), denominator, out=np.zeros(len(descriptor.labels)), where=denominator > 0)
        recorded_f1 = np.asarray([per_class[label] for label in descriptor.labels])
        if not np.allclose(expected_f1, recorded_f1, rtol=0, atol=1e-6) or not np.isclose(expected_f1.mean(), scores[0], rtol=0, atol=1e-6):
            raise ValueError("평가 지표가 confusion matrix와 일치하지 않습니다")


def check_semantics(descriptor: ModelDescriptor, mapping: dict[str, str]) -> None:
    if set(mapping) != set(descriptor.labels):
        raise ValueError("클래스와 자세 semantic 매핑이 완전하지 않습니다")
    for label, semantic in mapping.items():
        # The current avatar/risk consumers only understand these existing labels.
        if label not in POSTURE_SEMANTICS or semantic not in {label, POSTURE_SEMANTICS[label]}:
            raise ValueError(f"GUI·avatar·위험도에서 지원하지 않는 자세 매핑: {label}")


def validate_probabilities(output: Any, descriptor: ModelDescriptor) -> np.ndarray:
    values = np.asarray(output, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(descriptor.labels) or not len(values) or not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("Golden Replay 출력 계약이 올바르지 않습니다")
    return values


def check_golden(adapter: Any, descriptor: ModelDescriptor, frame: Any, golden: dict[str, Any], tolerance: float, seconds: float) -> np.ndarray:
    if tuple(golden.get("label_order", ())) != descriptor.labels:
        raise ValueError("Golden Replay 라벨 순서가 다릅니다")
    if descriptor.backend == "pytorch":
        from .window_assembler import WindowAssembler
        frame = WindowAssembler().assemble(frame, descriptor, seconds)
    probabilities = validate_probabilities(adapter.predict_proba(frame, seconds), descriptor)
    if len(probabilities) != golden.get("window_count"):
        raise ValueError("Golden Replay 윈도우 개수가 기준과 다릅니다")
    aggregate = probabilities.mean(axis=0)
    expected = np.asarray(golden.get("expected_scores"), dtype=float)
    if expected.shape != aggregate.shape or not np.isfinite(expected).all() or (expected < 0).any() or not np.isclose(expected.sum(), 1, rtol=0, atol=1e-6):
        raise ValueError("Golden Replay 전체 클래스 점수 기준이 필요합니다")
    if not np.allclose(aggregate, expected, rtol=0, atol=tolerance):
        raise ValueError("Golden Replay 클래스 점수가 장치 허용 오차를 벗어났습니다")
    if descriptor.labels[int(np.argmax(aggregate))] != golden.get("expected_label"):
        raise ValueError("Golden Replay 예측 라벨이 기준과 다릅니다")
    return probabilities
