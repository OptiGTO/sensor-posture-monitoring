"""Subject-disjoint training and same-record comparison of three posture architectures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .personal_guidance import CalibrationProfile
from .personalized_model import (FEATURE_VERSION, WINDOW_SECONDS, SAMPLING_RATE,
                                 PersonalPostureModel, features, sha256)
from .posture_core import CLASS_NAMES, SENSOR_COLUMNS, read_sensor_csv, extract_feature_vector
from .window_pipeline import WindowSpec, segment_sensor_frame


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_data(data_root: Path, *, allow_retrospective=False):
    # Reuse the audited record filters and registration selection policy. No
    # measurement-only or excluded files are promoted into training data.
    from validate_personalization import audit_dataset, registration_plan, register_from_record, assert_separate
    records, audit = audit_dataset(data_root)
    seen = set()
    for record in records:
        if record.sensor_sha256 in seen:
            raise ValueError("중복 센서 기록이 있습니다. 피험자·등록 기록 누수를 확인하세요.")
        seen.add(record.sensor_sha256)
    windows, profiles, labels, groups, sources, registrations, rows = [], [], [], [], [], [], []
    retrospective = False
    by_subject = {}
    for record in records:
        by_subject.setdefault(record.subject, []).append(record)
    for subject, subject_records in sorted(by_subject.items()):
        linked = {}
        for record in subject_records:
            sidecar = record.path.with_suffix(".manifest.json")
            payload = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
            reference = payload.get("guidance", {}).get("reference_profile")
            if reference:
                profile = CalibrationProfile.from_dict(reference)
                linked[record.path] = profile
        if linked:
            if len(linked) != len(subject_records):
                raise ValueError(f"{subject}: 개인 기준이 연결된 기록과 없는 기록을 같은 학습에 섞을 수 없습니다.")
            evaluation = subject_records
            # A physical participant must use one identity even across new sessions.
            if len({profile.user_id for profile in linked.values()}) != 1:
                raise ValueError(f"{subject}: 서로 다른 사용자 개인 기준이 연결되어 있습니다.")
            registrations.append({"subject": subject, "method": "linked_live_profiles",
                                  "versions": sorted({p.profile_version for p in linked.values()})})
        else:
            if not allow_retrospective:
                raise ValueError(f"{subject}: 등록 기준이 연결되지 않았습니다. 라벨 수집에서 기준을 연결하거나 과거 기록 실험을 선택하세요.")
            plans = registration_plan(subject_records, enabled=allow_retrospective)
            if not plans:
                raise ValueError(f"{subject}: 등록용 정자세 기록이 없습니다.")
            registration, evaluation = plans[0]
            assert_separate(registration, evaluation)
            profile = register_from_record(registration)
            # The audit helper returns a CalibrationProfile (no real registration).
            linked = {record.path: profile for record in evaluation}
            registrations.append({"subject": subject, "method": "first_correct_record_first_10_seconds",
                                  "file": registration.relative_path, "sha256": registration.sha256,
                                  "entire_record_excluded": True})
            retrospective = True
        for record in evaluation:
            frame = read_sensor_csv(record.path)
            sequence, metadata = segment_sensor_frame(frame, WindowSpec(window_seconds=WINDOW_SECONDS, sampling_rate=SAMPLING_RATE))
            profile = linked[record.path]
            for sample, row in zip(sequence, metadata.to_dict("records")):
                windows.append(sample); profiles.append(profile)
                labels.append(CLASS_NAMES.index(record.label)); groups.append(subject)
                rows.append({"subject": subject, "file": record.relative_path, "window_end_s": row["nominal_end_s"]})
            source = {"subject": subject, "file": record.relative_path, "sha256": record.sha256,
                      "sensor_sha256": record.sensor_sha256,
                      "profile_sha256": hashlib.sha256(json.dumps({k: v for k, v in profile.as_dict().items() if k not in {"profile_version", "created_at"}}, sort_keys=True).encode()).hexdigest()}
            sidecar = record.path.with_suffix(".manifest.json")
            if sidecar.is_file(): source["manifest_sha256"] = sha256(sidecar)
            sources.append(source)
    if not windows:
        raise ValueError("등록과 분리된 평가·학습 윈도우가 없습니다.")
    identities = {}
    for group, profile in zip(groups, profiles):
        if profile.user_id in identities and identities[profile.user_id] != group:
            raise ValueError("같은 개인 기준 사용자가 서로 다른 피험자로 분리되어 있습니다.")
        identities[profile.user_id] = group
    personal, detail = zip(*(features(x, p) for x, p in zip(windows, profiles)))
    return {"windows": np.stack(windows), "profiles": profiles, "personal": np.stack(personal),
            "detail": np.stack(detail), "raw": np.stack([extract_feature_vector(x) for x in windows]),
            "y": np.asarray(labels), "groups": np.asarray(groups), "rows": rows,
            "sources": sources, "registrations": registrations, "excluded": audit["excluded_records"],
            "retrospective": retrospective}


def split_subjects(groups, y, seed):
    subjects = np.array(sorted(set(groups)))
    if len(subjects) < 4:
        raise ValueError("학습·검증·최종 평가를 분리하려면 최소 4명의 피험자가 필요합니다.")
    np.random.default_rng(seed).shuffle(subjects)
    count = max(1, len(subjects) // 5)
    partitions = {"train": subjects[2 * count:], "validation": subjects[count:2 * count],
                  "final_evaluation": subjects[:count]}
    masks = {name: np.isin(groups, members) for name, members in partitions.items()}
    for name, mask in masks.items():
        missing = set(range(7)) - set(y[mask])
        if missing:
            raise ValueError(f"{name}: 등록 기록을 제외한 뒤 자세가 부족합니다: {[CLASS_NAMES[k] for k in sorted(missing)]}")
    return partitions, masks


def candidates(binary=False):
    if binary:
        for c in (.01, .1, 1):
            yield f"Logistic Regression C={c}", make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=3000))
    for c, gamma in ((1., "scale"), (3., "scale"), (1., .0005)):
        yield f"RBF-SVM C={c} gamma={gamma}", make_pipeline(StandardScaler(), SVC(C=c, gamma=gamma, probability=True, class_weight="balanced", random_state=20260911))


def metric(y, pred):
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=list(range(7)), zero_division=0)
    correct, incorrect = y == 0, y != 0
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f.mean()),
            "correct_false_incorrect_rate": float(np.mean(pred[correct] > 0)) if correct.any() else None,
            "incorrect_missed_correct_rate": float(np.mean(pred[incorrect] == 0)) if incorrect.any() else None,
            "unknown_rate": float(np.mean(pred < 0)),
            "correct_unknown_rate": float(np.mean(pred[correct] < 0)) if correct.any() else None,
            "incorrect_unknown_rate": float(np.mean(pred[incorrect] < 0)) if incorrect.any() else None,
            "per_class": {label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(n[i])} for i, label in enumerate(CLASS_NAMES)},
            "confusion_labels": [*CLASS_NAMES, "판단 유보 / 종류 미확인"],
            "confusion_matrix": confusion_matrix(y, pred, labels=[*range(7), -1]).tolist()}


def fit_gate(x, y, masks):
    best, trials = None, []
    binary = (y > 0).astype(int)
    for name, model in candidates(binary=True):
        model.fit(x[masks["train"]], binary[masks["train"]])
        scores = model.predict_proba(x[masks["validation"]])[:, 1]
        truth = binary[masks["validation"]]
        for center in (.35, .5, .65):
            low, high = center - .05, center + .05
            pred = np.where(scores <= low, 0, np.where(scores >= high, 1, -1))
            # Abstentions count as failures in selection; they cannot hide misses.
            balanced = float(np.mean([np.mean(pred[truth == k] == k) for k in (0, 1)]))
            trials.append({"model": name, "thresholds": [low, high], "validation_binary_balanced_with_abstentions": balanced})
            if best is None or balanced > best[0]: best = (balanced, model, [low, high], name)
    return best, trials


def fit_detail(x, y, masks):
    best = None
    for name, model in candidates():
        model.fit(x[masks["train"] & (y > 0)], y[masks["train"] & (y > 0)])
        val = masks["validation"] & (y > 0)
        pred = model.predict(x[val])
        _, _, f, _ = precision_recall_fscore_support(y[val], pred, labels=list(range(1, 7)), zero_division=0)
        score = float(f.mean())
        if best is None or score > best[0]: best = (score, model, name)
    return best


def make_bundle(gate, detail, thresholds, detail_threshold=.35):
    return {"feature_version": FEATURE_VERSION, "labels": list(CLASS_NAMES), "sensor_columns": list(SENSOR_COLUMNS),
            "window_seconds": WINDOW_SECONDS, "sampling_rate": SAMPLING_RATE,
            "gate": gate, "detail": detail, "gate_thresholds": thresholds,
            "detail_threshold": detail_threshold, "confidence_semantics": "uncalibrated_score"}


def routed(gate, detail, x, detail_x, thresholds, detail_threshold):
    scores = gate.predict_proba(x)[:, 1]
    passed = scores >= thresholds[1]
    pred = np.full(len(scores), -1)
    pred[scores <= thresholds[0]] = 0
    if passed.any():
        probabilities = detail.predict_proba(detail_x[passed])
        top = probabilities.argmax(axis=1)
        pred[np.flatnonzero(passed)] = np.where(probabilities.max(axis=1) >= detail_threshold, detail.classes_[top], -1)
    states = np.where(scores <= thresholds[0], "correct", np.where(passed, "incorrect", "uncertain"))
    return pred, states, passed


def train(data_root: Path, output: Path, seed=20260826, *, allow_retrospective=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("personalized.*")):
        raise ValueError("새 학습 실행 폴더를 사용하세요. 기존 모델은 덮어쓰지 않습니다.")
    marker = output / ".incomplete"; marker.touch()
    print("개인 등록 기록 분리·원본 해시 검사", flush=True)
    data = prepare_data(data_root, allow_retrospective=allow_retrospective)
    partitions, masks = split_subjects(data["groups"], data["y"], seed)
    y = data["y"]
    print("피험자 분리: " + ", ".join(f"{k}={len(v)}명" for k, v in partitions.items()), flush=True)
    print("개인 기준 1단계 모델·임계값 선택", flush=True)
    personal, personal_trials = fit_gate(data["personal"], y, masks)
    detail = fit_detail(data["detail"], y, masks)
    # The detail reject threshold is locked using validation only.
    val = masks["validation"]
    detail_threshold = max((.25, .35, .45), key=lambda t: metric(y[val], routed(personal[1], detail[1], data["personal"][val], data["detail"][val], personal[2], t)[0])["macro_f1"])
    bundle = make_bundle(personal[1], detail[1], personal[2], detail_threshold)
    print("비교군 학습: 단일 7-class 및 일반 이진 → 6-class", flush=True)
    standard = None
    for name, candidate in candidates():
        candidate.fit(data["raw"][masks["train"]], y[masks["train"]])
        score = metric(y[val], candidate.predict(data["raw"][val]))["macro_f1"]
        if standard is None or score > standard[0]: standard = (score, candidate, name)
    binary, binary_trials = fit_gate(data["raw"], y, masks)
    plain_detail = fit_detail(data["raw"], y, masks)
    plain_threshold = max((.25, .35, .45), key=lambda t: metric(y[val], routed(binary[1], plain_detail[1], data["raw"][val], data["raw"][val], binary[2], t)[0])["macro_f1"])
    # This selection record is persisted BEFORE the final partition is scored.
    selection = {"personal_gate": personal[3], "personal_gate_thresholds": personal[2], "detail": detail[2],
                 "detail_threshold": detail_threshold, "single_7class": standard[2],
                 "generic_binary": binary[3], "generic_gate_thresholds": binary[2],
                 "generic_detail": plain_detail[2], "generic_detail_threshold": plain_threshold,
                 "personal_trials": personal_trials, "generic_trials": binary_trials}
    write_json(output / "selection.json", selection)
    print("선택 고정 완료·최종 평가 (평가 결과로 재선택하지 않음)", flush=True)
    report = {"schema_version": 1, "evidence_kind": "retrospective_zero_empty_baseline" if data["retrospective"] else "linked_registration_records",
              "final_evaluation_used_for_selection": False, "seed": seed, "model_random_seed": 20260911,
              "partitions": {k: v.tolist() for k, v in partitions.items()}, "selection": selection,
              "registration_records": data["registrations"], "source_records": data["sources"], "excluded_records": data["excluded"],
              "dataset_fingerprint": hashlib.sha256(json.dumps(data["sources"], sort_keys=True).encode()).hexdigest(),
              "limitations": ["실제 센서 연속 사용·알림 부담은 검증하지 않았습니다.", "분류 점수는 보정된 확률이 아닙니다.",
                              "평가는 기록별 3초 윈도우이며 실시간 확인 지연을 포함하지 않습니다."]}
    if data["retrospective"]:
        report["limitations"].append("과거 기록에는 일치하는 빈 좌석·의자·설치 정보가 없어 빈 좌석 0 가정을 사용했습니다.")
    report["results"] = {}
    for partition in ("validation", "final_evaluation"):
        mask = masks[partition]
        p_pred, states, passed = routed(personal[1], detail[1], data["personal"][mask], data["detail"][mask], personal[2], detail_threshold)
        g_pred, g_states, g_passed = routed(binary[1], plain_detail[1], data["raw"][mask], data["raw"][mask], binary[2], plain_threshold)
        predictions = {"single_7class": standard[1].predict(data["raw"][mask]), "generic_hierarchy": g_pred, "personal_hierarchy": p_pred}
        report["results"][partition] = {name: metric(y[mask], values) for name, values in predictions.items()}
        for name, route_states, route_passed in (("personal_hierarchy", states, passed), ("generic_hierarchy", g_states, g_passed)):
            truth = y[mask]
            report["results"][partition][name]["gate"] = {
                "correct_false_incorrect_rate": float(np.mean(route_states[truth == 0] == "incorrect")),
                "incorrect_missed_correct_rate": float(np.mean(route_states[truth > 0] == "correct")),
                "unknown_rate": float(np.mean(route_states == "uncertain")),
                "stage2_execution_rate": float(np.mean(route_passed))}
        if partition == "final_evaluation":
            report["final_windows"] = [{**data["rows"][i], "truth": int(y[i]),
                                        **{name: int(value[j]) for name, value in predictions.items()}}
                                       for j, i in enumerate(np.flatnonzero(mask))]
            report["by_subject"] = {str(subject): {name: metric(y[mask][data["groups"][mask] == subject], value[data["groups"][mask] == subject])
                                                   for name, value in predictions.items()}
                                    for subject in partitions[partition]}
    # Round-trip parity uses training windows only; no performance selection here.
    artifact = output / "personalized.joblib"; joblib.dump(bundle, artifact, compress=3)
    restored = PersonalPostureModel(joblib.load(artifact)); initial = PersonalPostureModel(bundle)
    for i in np.flatnonzero(masks["train"])[:20]:
        if restored.predict(data["windows"][i], data["profiles"][i]) != initial.predict(data["windows"][i], data["profiles"][i]):
            raise ValueError("저장 전후 개인 모델 예측이 다릅니다.")
    report["serialization_parity"] = "passed"
    report_path = output / "personalized.report.json"; write_json(report_path, report)
    final = report["results"]["final_evaluation"]
    text = "# 개인 기준 2단계 모델 비교\n\n학습·검증·최종 평가 피험자를 분리하고, 모든 비교군에서 같은 등록 기록을 제외했습니다.\n\n"
    text += "| 방식 | Macro F1 | Accuracy | 정자세 오탐 | 비정상 누락 | 미확정 |\n|---|---:|---:|---:|---:|---:|\n"
    for name, scores in final.items():
        display = {"single_7class": "단일 7-class", "generic_hierarchy": "일반 이진 → 비정상 6-class", "personal_hierarchy": "개인 기준 이진 → 비정상 6-class"}[name]
        text += f"| {display} | {scores['macro_f1']:.4f} | {scores['accuracy']:.2%} | {scores['correct_false_incorrect_rate']:.2%} | {scores['incorrect_missed_correct_rate']:.2%} | {scores['unknown_rate']:.2%} |\n"
    text += "\n미확정은 평가에서 제외하지 않습니다. 표의 오탐·누락은 최종 출력 기준이며, 1단계 상태 기준은 JSON의 gate에 별도 기록합니다.\n\n" + "\n".join("- " + item for item in report["limitations"]) + "\n"
    (output / "personalized.report.md").write_text(text, encoding="utf-8")
    manifest = {"schema_version": 1, "kind": "personalized_hierarchy", "status": "experimental_validated",
                "model": {"file": artifact.name, "sha256": sha256(artifact)},
                "report": {"file": report_path.name, "sha256": sha256(report_path)},
                "feature_version": FEATURE_VERSION, "window_seconds": WINDOW_SECONDS,
                "confidence_semantics": "uncalibrated_score", "physical_validation": "not_performed"}
    path = output / "personalized.manifest.json"; write_json(path, manifest)
    marker.unlink()
    PersonalPostureModel.load(path)
    print(text, flush=True)
    return path
