#!/usr/bin/env python3
"""Retrospective, subject-separated evaluation of personal reference rules.

No deployment files or source recordings are modified. A protocol JSON must be
written before running this script. These experiments do not measure physical
posture correctness, actual reseating, or continuous alert latency.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from src.personal_guidance import GuidanceEngine
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, normalize_text, read_sensor_csv, training_csv_paths
from src.window_pipeline import WindowSpec, segment_sensor_frame
from validate_personalization import ReplayRecord, audit_dataset, assert_separate, register_from_record


RULES = ("maximum", "top4_mean", "total_variation", "spatial_consensus")
THRESHOLDS = (.25, .5, .75, 1., 1.5, 2., 3., 4., 6., 8., 12., 16., 24., 32., 48., 64.)
METHOD_NAMES = {
    "live_existing": "기존 실행 규칙 (0.5초 평활·상태 처리 포함)",
    "fixed_maximum": "기존 최대편차식 (공통 3초 창, 임계값 1.5)",
    "tuned_maximum": "최대편차식 + 다른 사람으로 임계값 조정",
    "aggregate": "전체 분포 규칙 + 다른 사람으로 규칙·임계값 선택",
    "aggregate_temporal": "전체 분포 규칙 + 2개 창 연속 확인",
    "always_incorrect": "항상 이탈로 판정 (비교 기준)",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def frozen_records(data_root, manifest_path):
    """Reproduce an exact audited cohort even while unrelated files are added."""
    audit = json.loads(manifest_path.read_text(encoding="utf-8"))["audit"]
    eligible_now = {path.resolve() for path in training_csv_paths(data_root)}
    records = []
    for evidence in audit["records"] + audit["excluded_records"]:
        path = (data_root / evidence["relative_path"]).resolve()
        if not path.is_relative_to(data_root.resolve()) or digest(path) != evidence["sha256"]:
            raise ValueError("Frozen source path or hash mismatch")
    for evidence in audit["records"]:
        path = (data_root / evidence["relative_path"]).resolve()
        if path not in eligible_now:
            raise ValueError("Frozen record is now excluded by its source manifest")
        record = ReplayRecord(path=path, **evidence)
        values = read_sensor_csv(path)[SENSOR_COLUMNS].to_numpy(dtype="<f8")
        if hashlib.sha256(values.tobytes()).hexdigest() != record.sensor_sha256:
            raise ValueError("Frozen sensor content mismatch")
        records.append(record)
    return records, audit


@dataclass
class Case:
    subject: str
    truth: np.ndarray
    class_ids: np.ndarray
    records: np.ndarray
    windows: list[dict]
    scores: dict[str, np.ndarray]
    live: np.ndarray


def ratios(values):
    values = np.asarray(values, dtype=float)
    if values.shape[-1] != 16 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected nonnegative finite 16-channel pressure")
    total = values.sum(axis=-1, keepdims=True)
    if (total <= 1e-6).any():
        raise ValueError("Zero total pressure: cannot normalize")
    return values / total


def rule_scores(mean_ratios, center, tolerance):
    """Scores have fixed definitions; only one scalar threshold is selected."""
    mean_ratios = np.atleast_2d(mean_ratios)
    delta = mean_ratios - center
    normalized = np.abs(delta) / tolerance
    grid = delta.reshape(-1, 4, 4)
    row = grid.sum(axis=2)
    col = grid.sum(axis=1)
    axis = np.linspace(0., 1., 4)
    tv = np.abs(delta).sum(axis=1) / 2.
    imbalance = np.maximum(np.abs(row[:, :2].sum(axis=1) - row[:, 2:].sum(axis=1)),
                           np.abs(col[:, :2].sum(axis=1) - col[:, 2:].sum(axis=1)))
    center_distance = np.sqrt((row @ axis) ** 2 + (col @ axis) ** 2)
    return {
        "maximum": normalized.max(axis=1),
        "top4_mean": np.sort(normalized, axis=1)[:, -4:].mean(axis=1),
        "total_variation": tv / .05,
        # These cues are correlated, not independent votes.
        "spatial_consensus": np.median(np.stack([tv / .05, imbalance / .1, center_distance / .1]), axis=0),
    }


def registration_groups(records, count=1, reserved=1):
    """Path-ordered whole-record exclusion, equal evaluation sets across counts."""
    ordered = sorted(records, key=lambda r: normalize_text(r.relative_path))
    correct = [r for r in ordered if r.label == CLASS_NAMES[0]]
    if count < 1 or reserved < count or len(correct) <= reserved:
        raise ValueError("Insufficient independent correct recordings")
    registrations = correct[:count]
    excluded = correct[:reserved]
    evaluation = [r for r in ordered if r not in excluded]
    for registration in excluded:
        assert_separate(registration, evaluation)
    return registrations, excluded, evaluation


def profile_parameters(registrations):
    # The original stability checks must pass for every selected registration.
    profiles = [register_from_record(r) for r in registrations]
    if len(profiles) == 1:
        return profiles[0], np.asarray(profiles[0].ratio_center), np.asarray(profiles[0].ratio_tolerance)
    registered_ratios = []
    for record, profile in zip(registrations, profiles):
        frame = read_sensor_csv(record.path)
        times = frame.elapsed_s.to_numpy(float)
        times -= times[0]
        stop = np.searchsorted(times, 10., side="left") + 1
        registered_ratios.append(ratios(profile.context.pressure(frame[SENSOR_COLUMNS].to_numpy(float)[:stop])))
    combined = np.concatenate(registered_ratios)
    center = np.median(combined, axis=0)
    center /= center.sum()
    mad = np.median(np.abs(combined - center), axis=0)
    return profiles[0], center, np.maximum(3 * 1.4826 * mad, .008)


def prepare_case(subject, registrations, evaluation, *, include_live=True):
    profile, center, tolerance = profile_parameters(registrations)
    scores = {name: [] for name in RULES}
    truth, class_ids, record_ids, metadata, live = [], [], [], [], []
    for record in evaluation:
        frame = read_sensor_csv(record.path)
        times = frame.elapsed_s.to_numpy(float)
        times -= times[0]
        if np.max(np.diff(times)) > 1.:
            raise ValueError("Input gap exceeds one second")
        windows, rows = segment_sensor_frame(frame, WindowSpec(3.))
        features = ratios(profile.context.pressure(windows)).mean(axis=1)
        for name, value in rule_scores(features, center, tolerance).items():
            scores[name].extend(value.tolist())
        states = np.full(len(times), -1, dtype=int)
        if include_live:
            engine = GuidanceEngine(profile)
            for i, (timestamp, values) in enumerate(zip(times, frame[SENSOR_COLUMNS].to_numpy(float))):
                event = engine.update(timestamp, values, seated=True)
                states[i] = {"correct": 0, "incorrect": 1}.get(event.state, -1)
        endpoints = np.searchsorted(times, rows.last_sample_s.to_numpy(float), side="right") - 1
        live.extend(states[np.maximum(endpoints, 0)].tolist())
        for row in rows.to_dict(orient="records"):
            truth.append(int(record.label != CLASS_NAMES[0]))
            class_ids.append(CLASS_NAMES.index(record.label))
            record_ids.append(record.relative_path)
            metadata.append({"record": record.relative_path, **row})
    return Case(subject, np.asarray(truth), np.asarray(class_ids), np.asarray(record_ids), metadata,
                {k: np.asarray(v) for k, v in scores.items()}, np.asarray(live))


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [-1, 0, 1]).all():
        raise ValueError("Invalid binary labels/predictions")
    n0, n1 = int((truth == 0).sum()), int((truth == 1).sum())
    tn = int(((truth == 0) & (prediction == 0)).sum())
    tp = int(((truth == 1) & (prediction == 1)).sum())
    fp = int(((truth == 0) & (prediction == 1)).sum())
    fn = int(((truth == 1) & (prediction == 0)).sum())
    if not n0 or not n1:
        raise ValueError("Both true classes required for balanced evaluation")
    return {"windows": len(truth), "correct_windows": n0, "incorrect_windows": n1,
            "accuracy": float(np.mean(truth == prediction)),
            "balanced_accuracy": .5 * (tn / n0 + tp / n1),
            "false_positive_rate": fp / n0, "missed_as_correct_rate": fn / n1,
            "sensitivity": tp / n1, "specificity": tn / n0,
            "uncertain_rate": float(np.mean(prediction == -1)),
            "correct_uncertain_rate": float(np.mean(prediction[truth == 0] == -1)),
            "incorrect_uncertain_rate": float(np.mean(prediction[truth == 1] == -1)),
            "confusion_rows_correct_incorrect_cols_correct_incorrect_uncertain":
                [[tn, fp, n0 - tn - fp], [fn, tp, n1 - fn - tp]]}


def choose_rule(cases, excluded_subjects, families):
    development = [case for case in cases if case.subject not in excluded_subjects]
    if len(development) < 2:
        raise ValueError("At least two development subjects required")
    best, best_key = None, None
    for family in families:
        for threshold in THRESHOLDS:
            stats = [metrics(c.truth, (c.scores[family] >= threshold).astype(int)) for c in development]
            ba = float(np.mean([m["balanced_accuracy"] for m in stats]))
            fpr = float(np.mean([m["false_positive_rate"] for m in stats]))
            sensitivity = float(np.mean([m["sensitivity"] for m in stats]))
            key = (ba, -fpr, sensitivity)
            if best_key is None or key > best_key:
                best_key = key
                best = {"family": family, "threshold": threshold,
                        "development_subjects": [c.subject for c in development],
                        "development_mean_balanced_accuracy": ba}
    return best


def temporal_predictions(scores, threshold, record_ids):
    """Two non-overlapping windows; no state leaks between recordings."""
    output = []
    previous_record = None
    state = candidate = None
    count = 0
    for score, record in zip(scores, record_ids):
        if record != previous_record:
            state = candidate = None
            count = 0
            previous_record = record
        proposed = 1 if score >= threshold else 0 if score <= threshold * 2 / 3 else state
        if proposed is None:
            candidate, count = None, 0
            output.append(-1)
            continue
        if proposed == candidate:
            count += 1
        else:
            candidate, count = proposed, 1
        if count >= 2:
            state = candidate
            output.append(state)
        else:
            output.append(-1)
    return np.asarray(output)


def summarize(cases, predictions):
    by_subject = {c.subject: metrics(c.truth, predictions[c.subject]) for c in cases}
    scalar_keys = ("accuracy", "balanced_accuracy", "false_positive_rate", "missed_as_correct_rate",
                   "sensitivity", "specificity", "uncertain_rate", "correct_uncertain_rate", "incorrect_uncertain_rate")
    pooled = metrics(np.concatenate([c.truth for c in cases]), np.concatenate([predictions[c.subject] for c in cases]))
    per_class = {}
    for index, label in enumerate(CLASS_NAMES):
        predicted = np.concatenate([predictions[c.subject][c.class_ids == index] for c in cases])
        if len(predicted):
            target = int(index != 0)
            per_class[label] = {"windows": len(predicted), "correct_binary_detection_rate": float(np.mean(predicted == target)),
                                "uncertain_rate": float(np.mean(predicted == -1))}
    return {"mean_subject": {k: float(np.mean([m[k] for m in by_subject.values()])) for k in scalar_keys},
            "pooled": pooled, "by_subject": by_subject, "per_posture_binary_detection": per_class}


def paired_bootstrap(summary, comparator, seed=20260913):
    names = sorted(summary["by_subject"])
    if names != sorted(comparator["by_subject"]):
        raise ValueError("Paired bootstrap needs identical subjects")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(names), size=(10000, len(names)))
    result = {}
    for metric in ("balanced_accuracy", "false_positive_rate", "sensitivity"):
        deltas = np.array([summary["by_subject"][n][metric] - comparator["by_subject"][n][metric] for n in names])
        result[metric] = {"mean_difference": float(deltas.mean()),
                          "subject_bootstrap_95_interval": np.quantile(deltas[indices].mean(axis=1), [.025, .975]).tolist()}
    return result


def evaluate(cases):
    predictions = {name: {} for name in METHOD_NAMES}
    choices = []
    for case in cases:
        maximum = choose_rule(cases, {case.subject}, ("maximum",))
        aggregate = choose_rule(cases, {case.subject}, RULES[1:])
        choices.append({"evaluation_subject": case.subject, "maximum": maximum, "aggregate": aggregate})
        predictions["live_existing"][case.subject] = case.live
        predictions["fixed_maximum"][case.subject] = (case.scores["maximum"] >= 1.5).astype(int)
        predictions["tuned_maximum"][case.subject] = (case.scores["maximum"] >= maximum["threshold"]).astype(int)
        prediction = (case.scores[aggregate["family"]] >= aggregate["threshold"]).astype(int)
        predictions["aggregate"][case.subject] = prediction
        predictions["aggregate_temporal"][case.subject] = temporal_predictions(case.scores[aggregate["family"]], aggregate["threshold"], case.records)
        predictions["always_incorrect"][case.subject] = np.ones_like(case.truth)
    summaries = {name: summarize(cases, preds) for name, preds in predictions.items()}
    return summaries, choices, predictions


def report_markdown(result):
    lines = ["# 개인 정자세 rule-based 가설 검증", "",
             "새 분류 모델을 학습하지 않고, 개인 등록 기준에서 벗어나는 정도를 규칙으로 판별했습니다.",
             "**과거 CSV를 이용한 이진 판별 실험이며, 실물 센서·재착석·연속 알림 검증이 아닙니다.**", "",
             "## 데이터와 분리", "",
             f"- 유효 기록: {result['audit_summary']['eligible_record_count']}개, {result['audit_summary']['subject_count']}명.",
             f"- 주 비교: {result['primary']['subjects']}명, 등록 기록 {result['primary']['registration_records']}개 전부 제외, 평가 기록 {result['primary']['evaluation_records']}개, 3초 창 {result['primary']['windows']}개.",
             "- 한 사람씩 평가에서 남겨두고, 나머지 사람에게서만 규칙 종류와 임계값을 선택했습니다. 개인의 정자세 등록값은 평가 시 사용 가능한 입력입니다.",
             "- 원본/센서 내용 해시 중복을 검사했습니다. 같은 기록을 잘라 등록과 평가에 나누지 않았습니다.",
             "- 정자세·이탈이 모두 있는 사람만 주 비교에 포함했습니다. 제외 및 실패는 results.json에 남겼습니다.",
             "- 과거에 이미 검토한 데이터이므로 새로운 외부 최종 평가로 부르지 않습니다.", "",
             "## 주 비교", "",
             "모든 수치는 피험자별 지표의 평균입니다. 이탈이 양성입니다. Balanced accuracy는 정자세와 이탈의 인식률을 동일하게 반영합니다. 판단 유보는 분모에 포함하며 정답으로 처리하지 않습니다.", "",
             "| 방식 | Balanced accuracy | Accuracy | 정자세 오탐 | 이탈 감지 | 이탈→정자세 누락 | 판단 유보 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, summary in result["methods"].items():
        m = summary["mean_subject"]
        fields = [m[k] for k in ("balanced_accuracy", "accuracy", "false_positive_rate", "sensitivity", "missed_as_correct_rate", "uncertain_rate")]
        lines.append("| " + METHOD_NAMES[name] + " | " + " | ".join(f"{x:.2%}" for x in fields) + " |")
    lines += ["", "기존 실행 규칙은 현재 GuidanceEngine을 그대로 실행했습니다. 평활·움직임·상태 처리 방식이 달라, 집계 방식의 효과는 공통 3초 창을 쓰는 최대편차식 및 임계값 조정 최대편차식과 비교해야 합니다.",
              "", "## 사전 정의한 성공 조건", "",
              "기존 최대편차식 대비 정자세 오탐 10%p 이상 감소, 이탈 감지율 저하 5%p 이내, Balanced accuracy 증가를 모두 요구했습니다.",
              f"- 조건 충족: **{'예' if result['success_criterion_met'] else '아니오'}**", ""]
    for name, comparison in result["comparisons"].items():
        lines.append(f"### 전체 분포 규칙 vs {METHOD_NAMES[name]}")
        lines.append("")
        for key, label in (("balanced_accuracy", "Balanced accuracy"), ("false_positive_rate", "정자세 오탐"), ("sensitivity", "이탈 감지")):
            item = comparison[key]
            lo, hi = item["subject_bootstrap_95_interval"]
            lines.append(f"- {label} 차이: {item['mean_difference'] * 100:+.2f}%p; 피험자 bootstrap 95% 구간 [{lo * 100:+.2f}, {hi * 100:+.2f}]%p.")
        lines.append("")
    lines += ["구간은 저장된 교차검증 예측을 피험자 단위로 다시 표집한 설명적 추정입니다. 전체 규칙 선택을 매번 재실행한 불확실성이나 외부 환경 변화를 포함하지 않습니다.",
              "", "## 자세별 이진 감지", "", "| 실제 자세 | 평가 창 | 최대편차식 | 임계값 조정 최대편차식 | 전체 분포 규칙 |", "|---|---:|---:|---:|---:|"]
    for label in CLASS_NAMES:
        entries = [result['methods'][name]['per_posture_binary_detection'].get(label) for name in ('fixed_maximum', 'tuned_maximum', 'aggregate')]
        if entries[0]:
            lines.append(f"| {label} | {entries[0]['windows']} | " + " | ".join(f"{e['correct_binary_detection_rate']:.2%}" for e in entries) + " |")
    lines += ["", "이 표는 이탈 종류를 분류한 7-class 정확도가 아닙니다. 정자세는 정자세로, 다른 자세는 이탈로 판정했는지를 나타냅니다.",
              "", "## 1회·2회·3회 등록", "", f"독립 정자세 기록이 4개 이상인 사람은 **{result['multi_registration']['subjects']}명**입니다. 모든 방식에서 앞의 정자세 3개를 제외한 동일 기록으로 비교했습니다. 이들의 평가 라벨은 임계값 선택에 사용하지 않았습니다.", "",
              "| 등록 횟수 | Balanced accuracy | 정자세 오탐 | 이탈 감지 |", "|---|---:|---:|---:|"]
    for count, summary in result["multi_registration"]["methods"].items():
        m = summary["mean_subject"]
        lines.append(f"| {count} | {m['balanced_accuracy']:.2%} | {m['false_positive_rate']:.2%} | {m['sensitivity']:.2%} |")
    lines += ["", "표본이 매우 작고 날짜·의자·재착석 조건이 확인되지 않았으므로, 다회 등록 효과의 일반화 근거로 사용할 수 없습니다.",
              "", "## 한계와 다음 실물 검증", "",
              "- 일치하는 빈 좌석 기준·등록 프로필이 없어 빈 좌석 값 0을 가정했습니다. 실제 자세 라벨의 정확성을 새로 확인하지 못했습니다.",
              "- 연속 확인 실험은 각 짧은 기록마다 상태를 초기화했습니다. 실제 자세 변경·복귀 시점이 없어 알림 지연·시간당 오경보를 입증하지 못합니다.",
              "- 원본 데이터, 앱 동작, 활성 모델, 기존 등록 프로필은 변경하지 않았습니다. 임계값 선택도 검증용으로만 수행했습니다.",
              "- 다음 검증은 실제 장치에서 빈 좌석 측정 → 감독된 정자세 2~3회 등록 → 별도 재착석·정자세 유지·일상 움직임·의도적인 자세 이탈 순으로 수행해야 합니다. 규칙 조정 세션과 최종 평가 세션을 분리합니다.",
              "", "## 재현", "", "```sh", result["reproduce_command"], "```", "",
              "protocol.json에 가설·후보 규칙·임계값·성공 기준을 평가 실행 전에 기록했습니다. results.json에는 분할, 선택 결과, 피험자별·자세별 지표와 제외 사유가, predictions.json에는 창별 판정이 있습니다. source_manifest.json에는 원본 해시가 있습니다."]
    return "\n".join(lines) + "\n"


def run(data_root, output, protocol_path, source_manifest=None):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise ValueError("Existing results; choose a new output directory")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if tuple(protocol["threshold_grid"]) != THRESHOLDS:
        raise ValueError("Protocol threshold grid mismatch")
    if protocol_path.resolve() != (output / "protocol.json").resolve():
        (output / "protocol.json").write_bytes(protocol_bytes)
    records, audit = frozen_records(data_root, source_manifest) if source_manifest else audit_dataset(data_root)
    if audit["duplicate_raw_files"] or audit["duplicate_sensor_sequences"]:
        raise ValueError("Duplicate recordings must be resolved before evaluation")
    grouped = defaultdict(list)
    for record in records:
        grouped[record.subject].append(record)
    names = {subject: f"S{i:02d}" for i, subject in enumerate(sorted(grouped), 1)}
    cases, exclusions, registrations, multiple_candidates = [], [], [], []
    for subject, subject_records in sorted(grouped.items()):
        sid = names[subject]
        correct_count = sum(r.label == CLASS_NAMES[0] for r in subject_records)
        if correct_count < 2 or not any(r.label != CLASS_NAMES[0] for r in subject_records):
            exclusions.append({"subject": sid, "reason": "Insufficient separate correct and incorrect evaluation", "correct_records": correct_count})
            continue
        try:
            registered, reserved, evaluation = registration_groups(subject_records)
            case = prepare_case(sid, registered, evaluation)
        except ValueError as error:
            exclusions.append({"subject": sid, "reason": str(error), "registration_failed": True})
            continue
        cases.append(case)
        registrations.append({"subject": sid, "registered": [r.relative_path for r in registered],
                              "excluded_in_full": [r.relative_path for r in reserved],
                              "evaluation": [r.relative_path for r in evaluation]})
        if correct_count >= 4:
            multiple_candidates.append(subject)
    if len(cases) < 3:
        raise ValueError("Insufficient evaluable subjects")
    print(f"Primary cases ready: {len(cases)} subjects, {sum(len(c.truth) for c in cases)} windows", flush=True)
    # Candidate scores remain unlabeled observations; selection excludes held subjects.
    summaries, choices, predictions = evaluate(cases)
    multi = {"subjects": len(multiple_candidates), "methods": {}, "selection": None, "registrations": [], "failures": []}
    if multiple_candidates:
        selection = choose_rule(cases, {names[s] for s in multiple_candidates}, RULES[1:])
        multi["selection"] = selection
        complete = []
        for subject in multiple_candidates:
            try:
                subject_cases = {}
                subject_registrations = []
                for count in (1, 2, 3):
                    registered, reserved, evaluation = registration_groups(grouped[subject], count, 3)
                    subject_cases[count] = prepare_case(names[subject], registered, evaluation, include_live=False)
                    subject_registrations.append({"subject": names[subject], "count": count,
                        "registered": [r.relative_path for r in registered], "excluded_in_all_comparisons": [r.relative_path for r in reserved]})
                complete.append(subject_cases)
                multi["registrations"].extend(subject_registrations)
            except ValueError as error:
                multi["failures"].append({"subject": names[subject], "reason": str(error)})
        multi["subjects"] = len(complete)
        if complete:
            for count in (1, 2, 3):
                cohort = [item[count] for item in complete]
                pred = {c.subject: (c.scores[selection["family"]] >= selection["threshold"]).astype(int) for c in cohort}
                multi["methods"][str(count)] = summarize(cohort, pred)
    comparisons = {name: paired_bootstrap(summaries["aggregate"], summaries[name]) for name in ("fixed_maximum", "tuned_maximum")}
    improvement = comparisons["fixed_maximum"]
    passed = (improvement["false_positive_rate"]["mean_difference"] <= -.10
              and improvement["sensitivity"]["mean_difference"] >= -.05
              and improvement["balanced_accuracy"]["mean_difference"] > 0)
    source_changed = [r["relative_path"] for r in audit["records"] + audit["excluded_records"]
                      if digest(data_root / r["relative_path"]) != r["sha256"]]
    if source_changed:
        raise ValueError(f"Sources changed during analysis: {source_changed}")
    try:
        from serial.tools import list_ports
        ports = [{"device": p.device, "description": p.description} for p in list_ports.comports()]
    except ImportError:
        ports = []
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
              "script_sha256": digest(__file__), "python_version": sys.version, "numpy_version": np.__version__,
              "audit_summary": {k: audit[k] for k in ("eligible_record_count", "excluded_record_count", "subject_count")},
              "primary": {"subjects": len(cases), "registration_records": len(registrations),
                          "evaluation_records": sum(len(set(c.records)) for c in cases), "windows": sum(len(c.truth) for c in cases)},
              "exclusions": exclusions, "registrations": registrations, "choices": choices,
              "methods": summaries, "comparisons": comparisons, "success_criterion_met": bool(passed), "multi_registration": multi,
              "physical_validation": {"performed": False, "observed_ports": ports}, "source_unchanged": True,
              "source_selection": "Exact audited manifest; later additions ignored" if source_manifest else "File inventory captured at audit start",
              "input_manifest_sha256": digest(source_manifest) if source_manifest else None,
              "reproduce_command": ".venv/bin/python validate_rule_hypothesis.py --data data --protocol results/rule_hypothesis_20260913/protocol.json --source-manifest results/rule_hypothesis_20260913/source_manifest.json --output results/rule_hypothesis_reproduction"}
    write_json(output / "source_manifest.json", {"subjects": {v: k for k, v in names.items()}, "audit": audit})
    write_json(output / "results.json", result)
    window_predictions = []
    for case in cases:
        for i, window in enumerate(case.windows):
            window_predictions.append({"subject": case.subject, **window, "true_binary": int(case.truth[i]),
                "true_posture": CLASS_NAMES[case.class_ids[i]],
                "scores": {k: float(v[i]) for k, v in case.scores.items()},
                "predictions": {name: int(pred[case.subject][i]) for name, pred in predictions.items()}})
    write_json(output / "predictions.json", window_predictions)
    (output / "REPORT.md").write_text(report_markdown(result), encoding="utf-8")
    print(json.dumps({"primary": result["primary"], "success_criterion_met": passed,
        "methods": {k: v["mean_subject"] for k, v in summaries.items()}, "multi_registration": multi["methods"]}, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, help="Freeze the exact files from a previous audited input manifest")
    args = parser.parse_args()
    run(args.data.resolve(), args.output.resolve(), args.protocol.resolve(),
        args.source_manifest.resolve() if args.source_manifest else None)
