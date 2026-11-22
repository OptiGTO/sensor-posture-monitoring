"""Presentation and bulk review helpers; lifecycle checks remain in the registry."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Callable

from .model_runtime import ModelDescriptor, ModelRegistry
from .model_runtime.deployment_checks import check_evaluation, evidence_path


STATUS_LABELS = {
    "pending": "검증·승인 대기", "failed": "검증 실패", "usable": "사용 가능",
    "environment_unavailable": "환경 확인 필요", "revoked": "폐기됨", "rejected": "승인 거절",
}


def identity(model: ModelDescriptor) -> tuple[str, str, str, str]:
    return model.artifact_id, model.version, model.artifact_hash, model.manifest_hash


def eligible(model: ModelDescriptor, action: str) -> bool:
    if model.revocation_status != "active":
        return False
    if action == "revoke":
        return True
    pending = model.approval_status == "pending"
    if action == "reject":
        return pending
    if action == "approve":
        return (pending and model.validation_status == model.deployment_validation_status == "passed"
                and model.runtime_compatible and model.sensor_compatible)
    return False


@dataclass(frozen=True)
class EvaluationSummary:
    accuracy: float | None = None
    macro_f1: float | None = None
    source: str = "미기록"
    detail: str = "독립 최종 평가 근거가 기록되지 않았습니다."


def evaluation_summary(model: ModelDescriptor) -> EvaluationSummary:
    """Only show independently evaluated, hash-bound metrics, never tuning scores."""
    try:
        config = model.metadata.get("_validation", {}).get("evaluation", {})
        files = config.get("evidence_files", [])
        if not files:
            return EvaluationSummary()
        check_evaluation(model, config, model.manifest_path.parent)
        # Multiple reports have no defined common aggregation; do not select one arbitrarily.
        if len(files) != 1:
            return EvaluationSummary(source="여러 평가 근거", detail="평가 근거가 여러 개여서 단일 점수로 합산하지 않습니다.")
        path = evidence_path(model, model.manifest_path.parent, str(files[0]))
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = report["metrics"]
        matrix = metrics["confusion_matrix"]
        accuracy = sum(row[i] for i, row in enumerate(matrix)) / sum(map(sum, matrix))
        subjects = report["partitions"]["final_evaluation"]["subjects"]
        return EvaluationSummary(accuracy, float(metrics["macro_f1"]), "최종 평가",
            f"평가 피험자: {', '.join(subjects)}\n데이터: {report['dataset_fingerprint']}\n근거: {path}")
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        return EvaluationSummary(source="근거 확인 필요", detail=f"평가 근거를 확인할 수 없습니다: {error}")


def created_timestamp(model: ModelDescriptor) -> float | None:
    try:
        value = datetime.fromisoformat(model.created_at.replace("Z", "+00:00"))
        return value.timestamp() if value.tzinfo else None
    except (ValueError, TypeError):
        return None


def ordered_models(models, summaries, key="created", descending=True):
    def value(model):
        if key in {"accuracy", "macro_f1"}:
            return getattr(summaries[identity(model)], key)
        if key == "created":
            return created_timestamp(model)
        if key == "window":
            return model.window_seconds
        if key == "status":
            return STATUS_LABELS[model.management_status]
        return model.name.casefold()
    # Missing metrics/dates stay last in either direction; identity breaks ties stably.
    stable = sorted(models, key=lambda item: (item.name.casefold(), identity(item)))
    known = [model for model in stable if value(model) is not None]
    missing = [model for model in stable if value(model) is None]
    return sorted(known, key=value, reverse=descending) + missing


@dataclass
class BatchResult:
    succeeded: list[ModelDescriptor] = field(default_factory=list)
    skipped: list[tuple[ModelDescriptor, str]] = field(default_factory=list)
    failed: list[tuple[ModelDescriptor, str]] = field(default_factory=list)


def review_models(registry: ModelRegistry, models: list[ModelDescriptor], action: str, *,
                  actor: str, reason: str = "", progress: Callable[[int, int], None] | None = None) -> BatchResult:
    if action not in {"approve", "reject", "revoke"}:
        raise ValueError("지원하지 않는 모델 작업입니다.")
    if action != "approve" and not reason.strip():
        raise ValueError("사유를 입력하세요.")
    result = BatchResult()
    unique = list({identity(model): model for model in models}.values())
    current = {identity(model): model for model in registry.management_models()}
    for index, selected in enumerate(unique):
        model = current.get(identity(selected))
        if model is None:
            result.failed.append((selected, "모델이 변경되었거나 목록에서 사라졌습니다. 다시 확인하세요."))
        elif not eligible(model, action):
            result.skipped.append((model, "현재 상태에서는 처리할 수 없습니다."))
        else:
            try:
                if action == "approve":
                    updated = registry.approve(model, approved_by=actor)
                else:
                    updated = getattr(registry, action)(model, reason=reason)
                result.succeeded.append(updated)
            except Exception as error:
                # A broken candidate must not prevent the remaining independent reviews.
                result.failed.append((model, str(error)))
        if progress:
            progress(index + 1, len(unique))
    return result
