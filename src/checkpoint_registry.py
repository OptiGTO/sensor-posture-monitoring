"""Immutable, evidence-backed recipes reusable from the model-add wizard.

Registration exposes a training recipe and its saved artifact, never approves
or activates a measurement model. Discovery checks hashes without unpickling.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = Path(__file__).with_name("validated_checkpoints.json")
PREFIX = "checkpoint:"
REQUIRED_FILES = frozenset({"model", "report", "protocol", "protocol_lock", "evaluation", "evaluation_seal",
    "reproducibility", "pipeline", "selection", "selection_seal", "final_metrics", "final_metrics_seal"})
PIPELINE_SOURCES = ("src/iterative_pipeline.py", "src/posture_core.py", "src/window_pipeline.py",
                    "src/pressure_contrast_features.py", "src/domain_feature_selection.py",
                    "src/group_probability_bias.py", "src/extended_models.py",
                    "src/sagittal_hierarchy.py")
USER_ADOPTION_STATUS = "user_adopted_research"
USER_ADOPTION_FILES = frozenset({"user_authorization", "adoption_source_protocol", "adoption_source_selection"})


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"체크포인트 근거는 JSON 객체여야 합니다: {path.name}")
    return value


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("체크포인트 수치 근거가 올바르지 않습니다.")
    return float(value)


def _same(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"체크포인트 근거 연결이 일치하지 않습니다: {label}")


def _validate_user_authorization(entry, base, pipeline, protocol, selected):
    """Bind an explicit one-off research adoption to its original failed recipe."""
    from .pipeline_recipes import validate_pipeline
    files = entry["files"]
    if not USER_ADOPTION_FILES.issubset(files):
        raise ValueError("사용자 채택의 원문 지시와 원 실험 근거가 필요합니다.")
    authorization = _object(base / files["user_authorization"]["path"])
    source_protocol = _object(base / files["adoption_source_protocol"]["path"])
    source_selection = _object(base / files["adoption_source_selection"]["path"])
    directive = protocol.get("user_adoption")
    if not isinstance(directive, dict) or authorization.get("schema_version") != 1:
        raise ValueError("사전 기록한 사용자 채택 지시가 없습니다.")
    for key, expected in {"kind": "explicit_user_research_adoption", "scope": "research_checkpoint_only",
            "checkpoint_id": entry["id"], "candidate_id": selected["id"], "active_model_changed": False}.items():
        _same(authorization.get(key), expected, "사용자 채택 " + key)
    if not isinstance(authorization.get("instruction"), str) or not authorization["instruction"].strip():
        raise ValueError("사용자의 채택 지시 원문이 없습니다.")
    stamp = datetime.fromisoformat(authorization.get("authorized_at", ""))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("사용자 채택 지시 시각에 시간대가 필요합니다.")
    for key, expected in {"authorization_sha256": files["user_authorization"]["sha256"],
            "checkpoint_id": entry["id"], "candidate_id": selected["id"]}.items():
        _same(directive.get(key), expected, "프로토콜 채택 " + key)
    for key, role in (("source_protocol_sha256", "adoption_source_protocol"),
                      ("source_selection_sha256", "adoption_source_selection")):
        _same(authorization.get(key), files[role]["sha256"], "원 실험 " + key)
    source_run = authorization.get("source_run")
    if not isinstance(source_run, str) or Path(source_run).is_absolute() or ".." in Path(source_run).parts or not source_run.startswith("research/results/"):
        raise ValueError("사용자 채택의 원 실험 경로가 올바르지 않습니다.")
    for role, filename in (("adoption_source_protocol", "protocol.json"), ("adoption_source_selection", "selection_lock.json")):
        _same(files[role]["path"], str(Path(source_run) / filename), "원 실험 파일 경로")
    _same(source_selection.get("protocol_sha256"), files["adoption_source_protocol"]["sha256"], "원 실험 선택 잠금")
    source = [row for row in source_selection.get("candidates", ()) if row.get("id") == selected["id"]]
    if len(source) != 1:
        raise ValueError("사용자가 선택한 원 실험 후보를 하나로 식별할 수 없습니다.")
    for recipe in (authorization.get("pipeline"), source[0].get("pipeline")):
        _same(validate_pipeline(recipe), pipeline, "사용자가 채택한 정확한 파이프라인")
    _same(validate_pipeline(protocol.get("baseline")), validate_pipeline(source_protocol.get("baseline")), "채택 비교 기준 모델")
    for key in ("dataset_sha256", "cache_sha256"):
        _same(protocol.get(key), source_protocol.get(key), "채택 고정 입력 " + key)
    original = source_protocol.get("criteria", {})
    _same(directive.get("original_criteria"), original, "보존된 원 채택 기준")
    requested = _number(authorization.get("minimum_development_delta"))
    observed = _number(source[0].get("mean_delta"))
    _same(_number(authorization.get("observed_development_delta")), observed, "원 실험의 실제 개선")
    if not 0 < requested <= observed < _number(original.get("minimum_development_delta")):
        raise ValueError("원 개발 개선과 사용자 지정 일회성 기준이 일치하지 않습니다.")
    _same(protocol["criteria"], {**original, "minimum_development_delta": requested}, "일회성 개발 기준만 변경")
    _same(entry["evidence"].get("adoption_basis"), "explicit_user_request", "채택 근거")
    _same(entry["evidence"].get("standard_validation_passed"), False, "기존 기준 미통과 표시")
    _same(entry["evidence"].get("active_model_changed"), False, "활성 모델 보존")


def _validate_supplement(entry: dict, base: Path, pipeline: dict, protocol: dict, metrics: dict) -> None:
    """Bind restored research records to the actual final training subjects."""
    policy = pipeline["training"].get("data_policy")
    if policy is None:
        return
    from .posture_core import CLASS_NAMES, normalize_text
    required = {"supplemental_manifest", "supplemental_cache", "source_supplemental"}
    files = entry["files"]
    if not required.issubset(files):
        raise ValueError("학습 기록 복원 체크포인트의 추가 입력·명세·소스 근거가 없습니다.")
    supplemental = protocol.get("supplemental_training")
    if not isinstance(supplemental, dict):
        raise ValueError("프로토콜에 학습 기록 복원 정책이 없습니다.")
    manifest = _object(base / files["supplemental_manifest"]["path"])
    _same(supplemental.get("policy"), policy, "추가 학습 정책")
    _same(manifest.get("policy"), policy, "추가 학습 명세 정책")
    _same(files["supplemental_manifest"]["sha256"], supplemental.get("manifest_sha256"), "추가 학습 명세 해시")
    _same(files["supplemental_cache"]["sha256"], supplemental.get("cache_sha256"), "추가 학습 캐시 해시")
    _same(manifest.get("cache_sha256"), files["supplemental_cache"]["sha256"], "명세의 추가 캐시 해시")
    for key, expected in (("origin_dataset_sha256", protocol.get("dataset_sha256")),
                          ("origin_cache_sha256", protocol.get("cache_sha256"))):
        if expected is None:
            raise ValueError("원 코호트 해시가 없습니다.")
        _same(manifest.get(key), expected, "추가 학습 원 코호트")
    _same(files["source_supplemental"]["sha256"],
          protocol.get("source_hashes", {}).get("experiments/iterative_improvement/supplemental.py"), "복원 실행 소스")
    from .pipeline_recipes import requires_record_ids, is_gain_recipe
    if requires_record_ids(pipeline):
        role = "source_record_sequence_training" if is_gain_recipe(pipeline) else "source_record_training"
        source = "record_sequence_training.py" if is_gain_recipe(pipeline) else "record_training.py"
        if role not in files:
            raise ValueError("복원된 학습 기록 ID를 연결한 실행 소스 근거가 없습니다.")
        _same(files[role]["sha256"],
              protocol.get("source_hashes", {}).get("experiments/iterative_improvement/" + source),
              "복원 학습 기록 ID 연결 소스")
    training_subjects = metrics.get("training_subjects")
    if not isinstance(training_subjects, list) or not training_subjects or any(not isinstance(value, str) or not value.strip() for value in training_subjects):
        raise ValueError("최종 학습 피험자 근거가 없습니다.")
    subjects = [normalize_text(value) for value in training_subjects]
    if len(set(subjects)) != len(subjects) or set(subjects) & set(metrics.get("evaluation_subjects", ())):
        raise ValueError("최종 학습·평가 피험자가 중복됩니다.")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("추가 학습 기록 명세가 비어 있습니다.")
    all_records, added = {}, {}
    for record in records:
        identity = normalize_text(record["record_id"])
        windows = record.get("windows")
        if not identity or identity in all_records or type(windows) is not int or windows <= 0 or record.get("label") != "정자세":
            raise ValueError("추가 정자세 기록의 식별자·창 수가 올바르지 않습니다.")
        all_records[identity] = windows
        if normalize_text(record["subject"]) in subjects:
            added[identity] = windows
    _same(manifest.get("record_count"), len(all_records), "추가 명세 기록 수")
    _same(manifest.get("window_count"), sum(all_records.values()), "추가 명세 창 수")
    _same(supplemental.get("records"), len(all_records), "프로토콜 추가 기록 수")
    _same(supplemental.get("windows"), sum(all_records.values()), "프로토콜 추가 창 수")
    from .pipeline_recipes import input_representation
    if input_representation(pipeline) == "sensor_sequence":
        _validate_supplement_sequences(files, manifest, protocol)
    if {normalize_text(record["subject"]) for record in records if normalize_text(record["subject"]) in subjects} != set(subjects):
        raise ValueError("추가 기록이 최종 학습 피험자를 모두 포함하지 않습니다.")
    audit = metrics.get("supplemental_training_audit")
    if not isinstance(audit, dict):
        raise ValueError("최종 모델의 학습 기록 복원 감사 결과가 없습니다.")
    for key, expected in {"policy": policy, "training_subjects": sorted(subjects),
            "added_record_paths": sorted(added), "added_windows_by_record": dict(sorted(added.items())),
            "added_records": len(added), "added_windows": sum(added.values()),
            "evaluation_rows_unchanged": True, "personal_reference_used": False}.items():
        _same(audit.get(key), expected, "최종 복원 감사 " + key)
    before, after = audit.get("class_counts_before"), audit.get("class_counts_after")
    if any(not isinstance(values, list) or len(values) != len(CLASS_NAMES)
           or any(type(value) is not int or value < 0 for value in values) for values in (before, after)):
        raise ValueError("복원 전후 클래스별 창 수가 올바르지 않습니다.")
    expected_after = list(before)
    expected_after[CLASS_NAMES.index("정자세")] += sum(added.values())
    _same(after, expected_after, "복원 후 정자세 창 증가")
    _same(audit.get("base_training_windows"), sum(before), "복원 전 학습 창 수")


def _validate_supplement_sequences(files, manifest, protocol):
    """Inspect sealed sequence alignment evidence without loading array caches."""
    from .posture_core import SENSOR_COLUMNS
    if "source_supplemental_sequence" not in files:
        raise ValueError("시계열 복원 실행 소스 근거가 없습니다.")
    for role, source in (("source_supplemental", "experiments/iterative_improvement/supplemental.py"),
                         ("source_supplemental_sequence", "experiments/iterative_improvement/supplemental_sequence.py")):
        expected = protocol.get("source_hashes", {}).get(source)
        _same(files[role]["sha256"], expected, "시계열 복원 소스 해시")
        _same(manifest.get("supplemental_source_hashes", {}).get(source), expected, "시계열 명세 소스 해시")
    evidence = manifest.get("sequence_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("복원 입력의 시계열 행 정렬 근거가 없습니다.")
    for key, expected in {"schema_version": 1, "shape": [manifest["window_count"], 36, 16],
            "dtype": "float32", "axis_order": ["windows", "time", "sensors"],
            "sensor_order": list(SENSOR_COLUMNS), "aligned_keys": ["seq", "X", "y", "groups", "record_ids", "start_s"],
            "record_alignment": "All rows match manifest record order, subject, label and window_start_s",
            "cache_bytes_reused_without_rewrite": True}.items():
        _same(evidence.get(key), expected, "시계열 복원 " + key)
    parity = evidence.get("feature_parity", {})
    for key, expected in {"method": "extract_window_features(seq); equal dtype, np.array_equal and C-order bytes",
            "bitwise_equal": True, "checked_windows": manifest["window_count"],
            "feature_dimension": 596, "maximum_absolute_difference": 0.0}.items():
        _same(parity.get(key), expected, "시계열·특징 행 일치 " + key)
    digest = evidence.get("sequence_array_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("복원 시계열 배열 해시가 올바르지 않습니다.")
    for record in manifest["records"]:
        starts = record.get("window_start_s")
        if not isinstance(starts, list) or len(starts) != record["windows"]:
            raise ValueError("복원 기록의 시계열 창 수와 시작 시각이 다릅니다.")
        times = [_number(value) for value in starts]
        if any(value < 0 for value in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("복원 기록의 시계열 시작 순서가 올바르지 않습니다.")


def _validate_identity_controls(entry: dict, base: Path, protocol: dict) -> None:
    """Require the preregistered software-path controls, separately from gains."""
    controls = protocol.get("identity_controls", [])
    if not isinstance(controls, list) or any(not isinstance(name, str) or not name for name in controls) or len(set(controls)) != len(controls):
        raise ValueError("동일 경로 대조군 ID가 올바르지 않습니다.")
    if not controls:
        return
    planned = [(name, recipe) for branch in protocol.get("branches", {}).values() for name, recipe in branch.items()]
    if any(sum(name == control for name, _ in planned) != 1 for control in controls):
        raise ValueError("동일 경로 대조군이 사전 등록 후보에 없습니다.")
    from .pipeline_recipes import validate_identity_control
    for control in controls:
        recipe = next(recipe for name, recipe in planned if name == control)
        try:
            validate_identity_control(protocol.get("baseline"), recipe)
        except ValueError as error:
            raise ValueError("동일 경로 대조군 구성이 기준 모델과 일치하지 않습니다: " + str(error)) from error
    files = entry["files"]
    if "identity_controls" not in files:
        raise ValueError("사전 등록한 동일 경로 대조군 검증 근거가 없습니다.")
    audit = _object(base / files["identity_controls"]["path"])
    _same(audit.get("protocol_sha256"), files["protocol"]["sha256"], "동일 경로 대조군 프로토콜")
    if type(audit.get("seed")) is not int:
        raise ValueError("동일 경로 대조군 시드가 정수가 아닙니다.")
    _same(audit.get("seed"), protocol["seeds"][0], "동일 경로 대조군 시드")
    partitions = [row.get("id") for row in protocol.get("partitions", ())]
    if (not partitions or any(not isinstance(name, str) or not name for name in partitions)
            or len(set(partitions)) != len(partitions)):
        raise ValueError("동일 경로 대조군의 개발 분할 ID가 올바르지 않습니다.")
    rows = audit.get("rows")
    expected = {(control, partition) for control in controls for partition in partitions}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("동일 경로 대조군의 모든 개발 분할 근거가 필요합니다.")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("동일 경로 대조군 검증 행이 올바르지 않습니다.")
        identity = (row.get("candidate"), row.get("partition"))
        if identity not in expected or identity in seen or row.get("all_prediction_arrays_bitwise_equal") is not True:
            raise ValueError("동일 경로 대조군이 모든 예측 배열의 완전 일치를 확인하지 않았습니다.")
        seen.add(identity)
        for key in ("baseline_predictions_sha256", "control_predictions_sha256"):
            if not isinstance(row.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError("동일 경로 대조군 예측 파일 해시가 올바르지 않습니다.")


def _validate_linkage(entry: dict, base: Path, pipeline: dict) -> None:
    """Bind one selected fitted artifact to the locked decision and its refit."""
    from .pipeline_recipes import validate_pipeline, pipeline_sources
    from .posture_core import CLASS_NAMES
    files = entry["files"]
    user_adopted = entry["status"] == USER_ADOPTION_STATUS
    objects = {role: _object(base / files[role]["path"]) for role in (
        "protocol", "selection", "final_metrics", "evaluation", "reproducibility")}
    for role, seal in (("protocol", "protocol_lock"), ("selection", "selection_seal"),
                       ("final_metrics", "final_metrics_seal"), ("evaluation", "evaluation_seal")):
        _same(_object(base / files[seal]["path"]).get("sha256"), files[role]["sha256"], seal)
    protocol, selection, metrics, evaluation, reproduction = (objects[key] for key in (
        "protocol", "selection", "final_metrics", "evaluation", "reproducibility"))
    selected = selection.get("selected")
    if not isinstance(selected, dict) or selected.get("qualified") is not True or not selected.get("id"):
        raise ValueError("사전에 확정한 개선 후보가 없습니다.")
    if user_adopted:
        _validate_user_authorization(entry, base, pipeline, protocol, selected)
    for value, label in ((selected.get("pipeline"), "선택 파이프라인"), (metrics.get("pipeline"), "평가 파이프라인")):
        _same(validate_pipeline(value), pipeline, label)
    _same(metrics.get("model_sha256"), files["model"]["sha256"], "평가 모델 해시")
    for record, label in ((selection, "선택"), (metrics, "최종 평가"), (reproduction, "재현")):
        _same(record.get("protocol_sha256"), files["protocol"]["sha256"], label + " 프로토콜")
    _same(evaluation.get("selection_sha256"), files["selection"]["sha256"], "평가 선택 잠금")
    _same(evaluation.get("selected_id"), selected["id"], "평가 후보 ID")
    _same(evaluation.get("scope"), entry["evidence"]["scope"], "평가 범위")
    if evaluation.get("holdout_evaluated") is not True or selection.get("historical_holdout_opened_this_iteration") is not False:
        raise ValueError("선택 잠금 후 최종 평가한 근거가 필요합니다.")
    seeds = protocol.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 2 or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("복수 난수 시드의 재현 근거가 필요합니다.")
    _same(metrics.get("seed"), seeds[0], "평가 주 시드")
    _same(pipeline["training"]["seed"], seeds[0], "파이프라인 주 시드")
    # Research refits override every member's seed. App reuse preserves the
    # declared recipe, so members must declare the seed that was validated.
    for member in pipeline.get("members", ()):
        _same(member["training"]["seed"], seeds[0], "앙상블 구성원 검증 시드")
    for key, expected in {"selected_id": selected["id"], "selected_model_sha256": files["model"]["sha256"],
            "selected_predictions_sha256": metrics.get("predictions_sha256"),
            "final_metrics_sha256": files["final_metrics"]["sha256"],
            "selection_sha256": files["selection"]["sha256"]}.items():
        if expected is None:
            raise ValueError("선택 후보의 예측 근거가 없습니다.")
        _same(reproduction.get(key), expected, "재현 " + key)
    # Prevent a previously validated recipe silently changing with current code.
    for relative in pipeline_sources(pipeline):
        _verified_file({"path": relative, "sha256": protocol.get("source_hashes", {}).get(relative)}, base)
    _validate_identity_controls(entry, base, protocol)
    criteria = protocol.get("criteria")
    if not isinstance(criteria, dict):
        raise ValueError("사전 평가 기준이 없습니다.")
    _same(criteria.get("scope"), evaluation["scope"], "프로토콜 검증 범위")
    minimum_dev = _number(criteria.get("minimum_development_delta"))
    minimum_final = _number(criteria.get("minimum_holdout_delta"))
    maximum_drop = _number(criteria.get("maximum_per_class_drop"))
    tolerance = _number(criteria.get("refit_probability_atol"))
    positive_splits = criteria.get("minimum_positive_splits")
    if minimum_dev <= 0 or minimum_final <= 0 or not 0 <= maximum_drop <= 1 or not 0 <= tolerance <= 1e-12 or type(positive_splits) is not int or positive_splits < 2:
        raise ValueError("사전 개선 기준이 충분하지 않습니다.")
    deltas = [_number(value) for value in selected.get("deltas", ())]
    partitions = protocol.get("partitions")
    if not isinstance(partitions, list) or len(deltas) != len(partitions):
        raise ValueError("개발 분할 수가 프로토콜과 다릅니다.")
    if len(deltas) < positive_splits or sum(deltas) / len(deltas) < minimum_dev or sum(value > 0 for value in deltas) < positive_splits:
        raise ValueError("개발 평가의 유의미한 개선 기준에 미달합니다.")
    if not math.isclose(_number(selected.get("mean_delta")), sum(deltas) / len(deltas), abs_tol=1e-12):
        raise ValueError("개발 개선 평균이 개별 결과와 다릅니다.")
    class_deltas = [_number(value) for value in selected.get("mean_class_deltas", ())]
    if len(class_deltas) != len(CLASS_NAMES) or min(class_deltas) < -maximum_drop:
        raise ValueError("개발 평가의 자세별 하락 기준에 미달합니다.")
    branches = protocol.get("branches", {})
    planned = {name: recipe for branch in branches.values() for name, recipe in branch.items()}
    if selected["id"] in planned:
        _same(validate_pipeline(planned[selected["id"]]), pipeline, "사전 등록 후보")
    elif pipeline["family"] in {"equal_soft_voting", "sequence_soft_voting", "record_soft_voting", "spatial_soft_voting", "gain_soft_voting", "local_soft_voting", "interaction_soft_voting", "support_soft_voting", "alternatives_soft_voting", "projection_soft_voting", "hardness_soft_voting", "structured_soft_voting"}:
        members = pipeline["members"]
        qualified = [row for row in selection.get("candidates", ()) if row.get("qualified") is True
                     and row.get("id") in planned]
        allowed = [validate_pipeline(planned[row["id"]]) for row in qualified]
        if any(member not in allowed for member in members) or selected.get("versus_best_member", {}).get("qualified") is not True:
            raise ValueError("검증된 개선 후보만 조합하고 최우수 구성원보다 개선되어야 합니다.")
    else:
        raise ValueError("선택 후보가 사전 등록한 실험에 없습니다.")
    final = evaluation.get("final")
    if not isinstance(final, list) or [row.get("seed") for row in final] != seeds:
        raise ValueError("최종 평가의 시드가 프로토콜과 다릅니다.")
    _same(metrics.get("metrics"), final[0].get("candidate"), "최종 선택 모델 지표")
    for index, row in enumerate(final):
        delta = _number(row["candidate"]["macro_f1"]) - _number(row["baseline"]["macro_f1"])
        if not math.isclose(delta, _number(row["delta"]), abs_tol=1e-12) or not user_adopted and (delta <= 0 or index == 0 and delta < minimum_final):
            raise ValueError("최종 평가의 개선 기준에 미달하거나 계산이 다릅니다.")
    per_class = [_number(final[0]["candidate"]["per_class_f1"][label]) - _number(final[0]["baseline"]["per_class_f1"][label]) for label in CLASS_NAMES]
    if not user_adopted and min(per_class) < -maximum_drop:
        raise ValueError("최종 평가의 자세별 하락 기준에 미달합니다.")
    interval = evaluation.get("paired_subject_interval", {})
    if interval.get("unit") != "subject" or not user_adopted and _number(interval.get("lower")) <= max(0., _number(criteria.get("bootstrap_lower_must_exceed"))):
        raise ValueError("피험자 단위 불확실성 기준에 미달합니다.")
    if _number(interval.get("upper")) < _number(interval.get("lower")) or type(interval.get("replicates")) is not int or interval["replicates"] != criteria.get("bootstrap_replicates"):
        raise ValueError("피험자 부트스트랩 구간 또는 반복 수가 프로토콜과 다릅니다.")
    for key in ("max_probability_error", "reload_max_error"):
        error = _number(reproduction.get(key))
        if error < 0 or error > tolerance:
            raise ValueError("선택 모델의 재학습 또는 재로딩이 재현되지 않았습니다.")
    robustness = reproduction.get("seed_robustness")
    if not isinstance(robustness, list) or [row.get("seed") for row in robustness] != seeds[1:] or not user_adopted and any(_number(row.get("mean_delta")) <= 0 for row in robustness):
        raise ValueError("추가 시드에서 개발 개선이 재현되지 않았습니다.")
    for row in robustness:
        seed_deltas = [_number(value) for value in row.get("deltas", ())]
        if len(seed_deltas) != len(partitions) or not seed_deltas or not math.isclose(sum(seed_deltas) / len(seed_deltas), _number(row["mean_delta"]), abs_tol=1e-12):
            raise ValueError("개발 시드별 평균이 분할 결과와 다릅니다.")
    _same(evaluation.get("development_seed_robustness"), robustness, "개발 시드 재현 결과")
    if user_adopted:
        # Preserve the evaluator's actual statistical decision, including a
        # failed holdout or seed gate; authorization is never proof of a gain.
        passed = (all(_number(row["mean_delta"]) > 0 for row in robustness)
            and _number(final[0]["delta"]) >= minimum_final
            and all(_number(row["delta"]) > 0 for row in final)
            and _number(interval["lower"]) > max(0., _number(criteria["bootstrap_lower_must_exceed"]))
            and min(per_class) >= -maximum_drop)
        for key in ("improved", "meaningful_gain"):
            _same(evaluation.get(key), passed, "실제 통계 기준 " + key)
            _same(entry["evidence"].get(key), evaluation[key], "사용자 채택 지표 " + key)
    _validate_supplement(entry, base, pipeline, protocol, metrics)


def is_checkpoint(method: str) -> bool:
    return isinstance(method, str) and method.startswith(PREFIX)


def _verified_file(reference: dict, root: Path) -> Path:
    if not isinstance(reference, dict):
        raise ValueError("체크포인트 파일 근거가 없습니다.")
    relative, digest = reference.get("path"), reference.get("sha256")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("체크포인트 파일은 프로젝트 내부 상대 경로여야 합니다.")
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("체크포인트 파일이 프로젝트 경계를 벗어났습니다.") from error
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("체크포인트 SHA-256 형식이 올바르지 않습니다.")
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"체크포인트 파일이 없거나 해시가 달라졌습니다: {relative}")
    return path


def validate_checkpoint(entry: dict, *, root: Path | None = None) -> dict:
    """Validate the complete recipe and immutable evidence before publication."""
    if not isinstance(entry, dict):
        raise ValueError("체크포인트 형식이 올바르지 않습니다.")
    if not isinstance(entry.get("id"), str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,95}", entry["id"]):
        raise ValueError("체크포인트 ID 형식이 올바르지 않습니다.")
    if any(not isinstance(entry.get(key), str) or not entry[key].strip() for key in ("name", "version")):
        raise ValueError("체크포인트 이름과 버전이 필요합니다.")
    user_adopted = entry.get("status") == USER_ADOPTION_STATUS
    if entry.get("status") not in {"validated_retrospective", USER_ADOPTION_STATUS}:
        raise ValueError("후향적 재현 검증을 통과한 체크포인트만 등록할 수 있습니다.")
    evidence = entry.get("evidence")
    if not isinstance(evidence, dict) or any(evidence.get(key) is not True
            for key in (("reproduced",) if user_adopted else ("improved", "meaningful_gain", "reproduced"))):
        raise ValueError("유의미한 개선과 재현 검증 근거가 모두 필요합니다.")
    if evidence.get("scope") not in {"retrospective_historical_holdout", "retrospective_subject_disjoint"}:
        raise ValueError("체크포인트의 후향적 검증 범위를 명시하세요.")
    pipeline = entry.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError("완전한 체크포인트 파이프라인이 필요합니다.")
    from .pipeline_recipes import validate_pipeline
    normalized = validate_pipeline(pipeline)
    from .model_catalog import MODEL_NAMES, CHECKPOINT_MODEL_NAMES
    if normalized["family"] not in MODEL_NAMES | CHECKPOINT_MODEL_NAMES:
        raise ValueError("모델 추가에서 아직 지원하지 않는 체크포인트 모델 구성입니다.")
    files = entry.get("files")
    if not isinstance(files, dict) or not REQUIRED_FILES.issubset(files):
        raise ValueError("모델·보고서·프로토콜·평가·재현 검증·파이프라인 파일이 모두 필요합니다.")
    base = Path(root or PROJECT_ROOT)
    for reference in files.values():
        _verified_file(reference, base)
    if Path(files["report"]["path"]).parts[0] != "docs":
        raise ValueError("체크포인트 보고서는 docs/ 안에 저장하세요.")
    recorded_pipeline = json.loads((base / files["pipeline"]["path"]).read_text(encoding="utf-8"))
    if validate_pipeline(recorded_pipeline) != normalized:
        raise ValueError("등록된 파이프라인과 검증한 파이프라인 파일이 다릅니다.")
    evaluation = json.loads((base / files["evaluation"]["path"]).read_text(encoding="utf-8"))
    reproduction = json.loads((base / files["reproducibility"]["path"]).read_text(encoding="utf-8"))
    if not isinstance(evaluation, dict) or any(evaluation.get(key) is not True
            for key in (("reproduced",) if user_adopted else ("improved", "meaningful_gain", "reproduced"))):
        raise ValueError("평가 파일이 유의미한 개선과 재현 검증을 확인하지 않았습니다.")
    if not isinstance(reproduction, dict) or reproduction.get("reproduced") is not True:
        raise ValueError("재현 검증 파일이 재현 성공을 확인하지 않았습니다.")
    try:
        _validate_linkage(entry, base, normalized)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("체크포인트 평가 연결 근거가 불완전합니다.") from error
    # JSON roundtrip prevents unserializable values and nonfinite parameters.
    clean = json.loads(json.dumps(entry, ensure_ascii=False, allow_nan=False))
    clean["pipeline"] = normalized
    return clean


def _read_registry(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("checkpoints"), list):
        raise ValueError("체크포인트 등록부 형식이 올바르지 않습니다.")
    return payload


def available_checkpoints(registry_path: Path | None = None, *, root: Path | None = None) -> list[dict]:
    """Fail closed for missing, altered, incomplete or duplicated evidence."""
    try:
        payload = _read_registry(Path(registry_path or REGISTRY_PATH))
    except (OSError, ValueError, TypeError):
        return []
    counts = {}
    for entry in payload["checkpoints"]:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            counts[entry["id"]] = counts.get(entry["id"], 0) + 1
    result = []
    for entry in payload["checkpoints"]:
        try:
            clean = validate_checkpoint(entry, root=root)
            if counts.get(clean["id"]) != 1:
                continue
            result.append(clean)
        except (OSError, ValueError, TypeError, KeyError, ImportError):
            continue
    return result


def checkpoint_method_entries() -> list[dict]:
    return [{"id": PREFIX + entry["id"],
             "label": f"{'사용자 채택 · 연구 기준선' if entry['status'] == USER_ADOPTION_STATUS else '체크포인트'} · {entry['name']} ({entry['version']})",
             "families": [entry["pipeline"]["family"]],
             "window_seconds": [entry["pipeline"]["window_seconds"]],
             "evidence": deepcopy(entry["evidence"]), "checkpoint": entry}
            for entry in available_checkpoints()]


def get_checkpoint(method: str) -> dict:
    if not is_checkpoint(method):
        raise ValueError("체크포인트 학습 방식이 아닙니다.")
    selected = next((entry for entry in available_checkpoints() if PREFIX + entry["id"] == method), None)
    if selected is None:
        raise ValueError(f"사용 가능한 검증 체크포인트가 없습니다: {method}")
    return selected


def register_checkpoint(entry: dict, *, registry_path: Path | None = None, root: Path | None = None) -> dict:
    """Append a checkpoint; existing IDs are immutable (including old versions)."""
    clean = validate_checkpoint(entry, root=root)
    path = Path(registry_path or REGISTRY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    try:
        lock_descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError("다른 체크포인트 등록이 진행 중입니다. 기존 등록부는 보존되었습니다.") from error
    name = None
    try:
        os.close(lock_descriptor)
        payload = _read_registry(path) if path.exists() else {"schema_version": 1, "checkpoints": []}
        if any(isinstance(previous, dict) and previous.get("id") == clean["id"] for previous in payload["checkpoints"]):
            raise ValueError("기존 체크포인트를 덮어쓸 수 없습니다. 새 버전 ID를 사용하세요.")
        payload["checkpoints"].append(clean)
        # Atomic replacement is confined to the registry, never measurement data.
        descriptor, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if name is not None and Path(name).exists():
            Path(name).unlink()
        lock.unlink()
    return clean


def checkpoint_resource_files(root: Path) -> list[Path]:
    """Exact evidence/artifacts needed to retain validated choices in a bundle."""
    from .pipeline_recipes import pipeline_sources
    entries = available_checkpoints(root / "src/validated_checkpoints.json", root=root)
    return sorted({root / reference["path"] for entry in entries for reference in entry["files"].values()}
                  | {root / relative for entry in entries for relative in pipeline_sources(entry["pipeline"])})


def checkpoint_required_packages(root: Path) -> set[str]:
    from .model_catalog import pipeline_runtime_packages
    return {package for entry in available_checkpoints(root / "src/validated_checkpoints.json", root=root)
            for package in pipeline_runtime_packages(entry["pipeline"])}
