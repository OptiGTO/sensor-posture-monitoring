"""Independently audit predictions and summarize the fixed sensor-gain experiment."""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
from experiments.sensor_gain.run import (
    verify, read, write_json, digest, now, condition_id, protected_snapshot,
    partition_indices, CLASS_NAMES, SENSOR_COLUMNS,
)

NAMES = {"rbf_svm": "RBF SVM", "xgboost": "XGBoost", "random_forest": "Random Forest", "extra_trees": "규제 Extra Trees"}


def matrix(y, pred):
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def metrics(cm):
    diag = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    per_class = np.divide(2 * diag, denom, out=np.zeros(7), where=denom != 0)
    return {"macro_f1": float(per_class.mean()), "accuracy": float(diag.sum() / cm.sum()),
            "per_class_f1": dict(zip(CLASS_NAMES, per_class.tolist())), "confusion_matrix": cm.tolist()}


def subject_bootstrap(truth, before, after, groups, draws=2000):
    ids = np.unique(groups)
    raw = np.stack([matrix(truth[groups == s], before[groups == s]) for s in ids])
    cal = np.stack([matrix(truth[groups == s], after[groups == s]) for s in ids])
    rng = np.random.default_rng(20260921)
    deltas = []
    for _ in range(draws):
        idx = rng.integers(0, len(ids), len(ids))
        deltas.append(metrics(cal[idx].sum(0))["macro_f1"] - metrics(raw[idx].sum(0))["macro_f1"])
    return np.percentile(deltas, [2.5, 97.5]).tolist()


def record_predictions(pred):
    ids = np.unique(pred["record_ids"])
    result = {"ids": ids, "truth": [], "prediction": [], "groups": []}
    for rid in ids:
        mask = pred["record_ids"] == rid
        if len(set(pred["truth"][mask])) != 1 or len(set(pred["groups"][mask])) != 1:
            raise ValueError("Record identity is inconsistent")
        result["truth"].append(pred["truth"][mask][0])
        result["prediction"].append(pred["probability"][mask].mean(0).argmax())
        result["groups"].append(pred["groups"][mask][0])
    return {k: np.asarray(v) for k, v in result.items()}


def summarize(out):
    p = verify(out); lock = read(out / "selection_lock.json")
    families = list(p["models"])
    expected_fits = len(p["conditions"]) * (len(p["partitions"]) + 1)
    complete = read(out / "training_complete.json")
    if lock["protocol_sha256"] != digest(out / "protocol.json"):
        raise ValueError("Selection refers to a different protocol")
    for name, sha in lock["development_artifacts"].items():
        if digest(out / name) != sha: raise ValueError("Locked development evidence changed")
    if protected_snapshot(Path(p["root"])) != read(out / "protected_before.json"):
        raise ValueError("Original files changed")
    all_metrics, final_predictions, verified = {}, {}, []
    for c in p["conditions"]:
        data = np.load(out / f"data_{c['seconds']}s.npz", allow_pickle=False)
        for phase in ("development", "final"):
            for part in (p["partitions"] if phase == "development" else p["partitions"][:1]):
                dest = out / phase / c["id"] / ("holdout" if phase == "final" else part["id"])
                m = read(dest / "metrics.json")
                if phase == "final" and datetime.fromisoformat(m["started_at"]) <= datetime.fromisoformat(lock["locked_at"]):
                    raise ValueError("Final fit preceded selection lock")
                for name, sha in m["artifacts"].items():
                    if digest(dest / name) != sha: raise ValueError("Fit evidence changed")
                pred = dict(np.load(dest / "predictions.npz", allow_pickle=False))
                _, te = partition_indices(data, part, final=phase == "final")
                np.testing.assert_array_equal(pred["indices"], te)
                for dest_key, source_key in (("truth", "y"), ("groups", "groups"), ("record_ids", "record_ids")):
                    np.testing.assert_array_equal(pred[dest_key], data[source_key][te])
                np.testing.assert_array_equal(pred["prediction"], pred["probability"].argmax(1))
                independently = metrics(matrix(pred["truth"], pred["prediction"]))
                for key in ("accuracy", "macro_f1"):
                    np.testing.assert_allclose(independently[key], m["evaluation"][key], atol=1e-12)
                np.testing.assert_array_equal(independently["confusion_matrix"], m["evaluation"]["confusion_matrix"])
                for label in CLASS_NAMES:
                    np.testing.assert_allclose(independently["per_class_f1"][label], m["evaluation"]["per_class_f1"][label], atol=1e-12)
                verified.append(str(dest.relative_to(out)))
                if phase == "final":
                    final_predictions[c["id"]] = pred
                    records = record_predictions(pred)
                    all_metrics[c["id"]] = {"window": independently,
                        "record": metrics(matrix(records["truth"], records["prediction"])),
                        "record_count": len(records["ids"]), "fit_seconds": m["fit_seconds"],
                        "batch_prediction_ms_per_window": m["batch_prediction_ms_per_window"],
                        "per_subject": m["per_subject"]}
    if len(verified) != expected_fits: raise ValueError(f"Expected exactly {expected_fits} fits")
    ranking = {c["id"]: c for c in lock["ranking"]}
    for cid, row in ranking.items():
        values = [read(out / "development" / cid / part["id"] / "metrics.json")["evaluation"]["macro_f1"] for part in p["partitions"]]
        np.testing.assert_allclose(values, row["development_f1"], atol=1e-12)
        np.testing.assert_allclose(np.mean(values), row["mean_development_f1"], atol=1e-12)
    comparisons = []
    for seconds in (2, 3):
        for family in families:
            raw_id, cal_id = [condition_id(seconds, a, family) for a in ("raw", "calibrated")]
            raw, cal = [final_predictions[cid] for cid in (raw_id, cal_id)]
            for key in ("indices", "truth", "groups", "record_ids"):
                np.testing.assert_array_equal(raw[key], cal[key])
            a, b = [all_metrics[cid] for cid in (raw_id, cal_id)]
            delta = b["window"]["macro_f1"] - a["window"]["macro_f1"]
            dev_delta = np.array(ranking[cal_id]["development_f1"]) - ranking[raw_id]["development_f1"]
            per_class_delta = {label: b["window"]["per_class_f1"][label] - a["window"]["per_class_f1"][label] for label in CLASS_NAMES}
            passed = (dev_delta.mean() >= .01 and (dev_delta > 0).sum() >= 2 and delta >= .01
                      and min(per_class_delta.values()) >= -.05)
            comparisons.append({"seconds": seconds, "family": family, "raw": a, "calibrated": b,
                "development_raw": ranking[raw_id]["mean_development_f1"],
                "development_calibrated": ranking[cal_id]["mean_development_f1"],
                "development_deltas": dev_delta.tolist(), "holdout_delta": delta,
                "holdout_delta_subject_bootstrap_95": subject_bootstrap(raw["truth"], raw["prediction"], cal["prediction"], raw["groups"]),
                "per_class_delta": per_class_delta, "passed_predefined_descriptive_criterion": bool(passed)})
    # Equal recording units make the unequal 2s/3s window counts interpretable.
    for family in families:
        for arm in ("raw", "calibrated"):
            r2, r3 = [record_predictions(final_predictions[condition_id(s, arm, family)]) for s in (2, 3)]
            for key in ("ids", "truth", "groups"): np.testing.assert_array_equal(r2[key], r3[key])
    summary = {"finished_at": now(), "training_finished_at": complete["finished_at"], "comparisons": comparisons,
        "development_selected_winner": lock["winner"], "counts": p["counts"], "verified_fits": verified,
        "independent_confusion_matrix_recalculation": True, "raw_and_active_models_unchanged": True,
        "approved": False, "fresh_external_validation": False, "protocol_sha256": digest(out / "protocol.json")}
    write_json(out / "summary.json", summary)
    write_json(out / "calibration.json", {"schema_version": 1, "approved": False,
        "method": p["calibration"], "reference_midpoint": p["reference"],
        "sensor_columns": p["sensor_columns"], "gains": p["gains"],
        "source_ranges": p["ranges"], "source_midpoints": p["midpoints"],
        "limitations": [p["calibration_evidence"], p["hardware_assumption"]],
        "protocol_sha256": digest(out / "protocol.json")})
    report = Path(p["root"]) / "docs/reports" / p.get("report_filename", "2026-09-23_sensor_gain_2s3s_report.md")
    rel = "../../research/results/" + out.name
    lines = ["# " + p.get("report_title", "손가락 하중 기반 센서 이득 보정: 2초·3초 재실험"), "",
        "사용자가 제공한 16개 센서 범위 중간값으로 고정 배율 보정을 적용하고 " + ", ".join(NAMES[f] for f in families) + "를 같은 조건에서 재학습했다. 보정 전후 모두 새로 학습했으며, 아래 수치는 학습 데이터 점수가 아닌 사람 분리 평가 결과다.", "",
        f"- 실행 시작: {p['started_at']}; 학습 종료: {complete['finished_at']}; 분석 종료: {summary['finished_at']}",
        "- 32명·692개 기록·7개 자세. 2초는 6,919개 창, 3초는 4,502개 창. 샘플링 12 Hz, 창 간격은 각각 2초/3초, 95% 이상 남은 마지막 창 포함.",
        "- 기존 2026-09-21 개선 재실험의 원본·라벨·등록용 제외 목록을 동결 재사용했다. 당시 제외한 등록용 33개 기록을 복원하지 않았다. 현재 전체 적격 자료를 새로 편입한 실험은 아니다.",
        "- 개발 3분할: 각 학습 20명/검증 6명. 최종: 개발 26명 학습/기존 고정 6명 평가. 최종 평가 기록은 두 길이 모두 83개이며, 창 수는 아래 표에 기록한다.",
        f"- 2길이 × 보정 전후 × {len(families)}모델 × (개발 3회 + 최종 1회) = {expected_fits}회 학습. 하이퍼파라미터 탐색 없이 기존 설정을 고정했다. 세부 설정은 protocol.json의 models에 기록했다.",
        "- 기존 평가 사람·결과는 과거 연구에서 노출된 자료다. 새로운 외부 검증이나 물리 센서 시험이 아니다.", "",
        "## 보정 정의와 전제", "",
        "`보정 신호 = 원 신호 × 3.05 / 센서 범위 중간값`. 3.05는 16개 중간값의 중앙값이다. 영점 보정, 비선형 적합, 클리핑은 적용하지 않았다. 보정은 특징 추출 전에 적용하고, StandardScaler는 각 학습 집합에만 맞췄다.", "",
        "| 센서 | 제공 범위 | 범위 중간값 | 보정 배율 |", "| --- | ---: | ---: | ---: |"]
    for label, bounds, mid, gain in zip(SENSOR_COLUMNS, p["ranges"], p["midpoints"], p["gains"]):
        lines.append(f"| {label} | {bounds[0]:g}–{bounds[1]:g} | {mid:.2f} | {gain:.6f} |")
    lines += ["", "손가락의 최대 누르는 힘과 접촉 면적이 센서마다 같다는 보장이 없다. 제공된 수치는 반복별 원자료나 평균·표준편차가 아니라 관측 범위와 그 중간값이다. 따라서 배율의 통계적 오차는 추정하지 않았으며, 본 보정은 탐색적 가설이다. 단위와 센서 비교 실험의 취득 일시도 미상이다. 같은 센서·배선·R11→R44 행 우선 매핑이 과거 기록 전체에 적용된다고 가정했지만 하드웨어 식별 정보로 확인하지 못했다.", "",
        "## 평가 결과", "",
        "주 지표는 7개 자세 F1의 단순 평균인 Macro-F1이다. Δ는 보정 후 − 보정 전이며 %p 단위다. 개발값은 3분할 평균이고, 최종값은 고정 6명의 전체 창을 모아 계산한다.", "",
        "| 창 | 모델 | 개발 보정 전 | 개발 보정 후 | 최종 보정 전 | 최종 보정 후 | 최종 Δ | Δ 95% 구간 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in comparisons:
        lo, hi = c["holdout_delta_subject_bootstrap_95"]
        lines.append(f"| {c['seconds']}초 | {NAMES[c['family']]} | {100*c['development_raw']:.2f}% | {100*c['development_calibrated']:.2f}% | {100*c['raw']['window']['macro_f1']:.2f}% | {100*c['calibrated']['window']['macro_f1']:.2f}% | {100*c['holdout_delta']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ["", "95% 구간은 평가자 6명을 군집 단위로 2,000회 쌍체 재표집한 기술적 bootstrap 구간이다. 창을 독립 표본으로 보지 않으며 모델 재학습의 불확실성이나 다중비교 보정은 포함하지 않는다. 같은 기록의 창 및 개발 분할 간 사람 재사용으로 관측은 서로 의존한다.", "",
        f"개발 점수로 미리 선택한 조건: **{lock['winner']}**. 이 순위는 최종 평가 예측 전에 고정했다. 최종 결과를 보고 조건을 다시 고르거나 재튜닝하지 않았다.", "",
        "## 정확도와 기록 단위 확인", "",
        "2초/3초는 창 개수가 다르므로 같은 83개 기록마다 창 예측 확률을 평균한 기록 단위 Macro-F1도 함께 제시한다. 이는 전체 기록을 본 보조 지표이며 2초/3초 실시간 성능과 동일하지 않다.", "",
        "| 창 | 모델 | 최종 창 수 | Accuracy 전 → 후 | 기록 F1 전 → 후 |", "| --- | --- | ---: | ---: | ---: |"]
    for c in comparisons:
        cnt = p["counts"][str(c["seconds"])]["partitions"]["development_1"]["test"]
        lines.append(f"| {c['seconds']}초 | {NAMES[c['family']]} | {cnt} | {100*c['raw']['window']['accuracy']:.2f}% → {100*c['calibrated']['window']['accuracy']:.2f}% | {100*c['raw']['record']['macro_f1']:.2f}% → {100*c['calibrated']['record']['macro_f1']:.2f}% |")
    lines += ["", "## 자세별 최종 F1", "", "각 셀은 보정 전 → 후(%)다.", "",
        "| 창·모델 | " + " | ".join(CLASS_NAMES) + " |", "| --- | " + " | ".join(["---:"] * 7) + " |"]
    for c in comparisons:
        cells = [f"{100*c['raw']['window']['per_class_f1'][label]:.2f} → {100*c['calibrated']['window']['per_class_f1'][label]:.2f}" for label in CLASS_NAMES]
        lines.append(f"| {c['seconds']}초 {NAMES[c['family']]} | " + " | ".join(cells) + " |")
    accepted = [f"{c['seconds']}초 {NAMES[c['family']]}" for c in comparisons if c["passed_predefined_descriptive_criterion"]]
    lines += ["", "## 판정과 검증 범위", "",
        "사전 기술적 개선 기준: 개발 평균 +1%p 이상, 개발 3분할 중 2개 이상 개선, 기존 최종 집합 +1%p 이상, 어떤 자세도 F1이 5%p보다 크게 하락하지 않을 것.", "",
        "- 기준 충족: " + (", ".join(accepted) if accepted else "없음") + ". 기준 충족 여부는 신규 사용자에 대한 통계적 우월성이나 배포 승인을 뜻하지 않는다.",
        f"- {expected_fits}개 학습의 사람/기록 분리, 학습 전용 scaler, 확률 합, 모델 저장 후 원시 창→보정→특징→예측 일치를 확인했다.",
        "- 모든 저장 예측에서 혼동행렬을 별도 집계하여 Macro-F1, Accuracy, 자세별 F1을 재계산했다.",
        "- 코드·수치 검사 내역은 아래 테스트 기록에 보존한다. 연구 저장 구조도 별도로 검사했다.",
        "- 3초 보정 전 입력·특징은 기존 동결 캐시와 완전히 일치했다. 포함 기록의 원본·스테이징·센서 수치 해시 및 중복 부재를 확인했다.",
        "- data/와 기존 research/models/의 파일 목록·SHA-256·수정시각은 실행 전후 일치했다. 앱의 활성 모델·보정 경로는 변경하지 않았다.",
        "- 일정한 채널에서 skew/kurtosis 계산의 정밀도 경고가 발생할 수 있다. 기존 특징 추출기의 비유한값 처리 계약을 그대로 사용하고 결과 특징이 모두 유한함을 검사했다.",
        "- 물리 센서, 실제 Tk 화면, 앱 재빌드, 신규 사용자·새 세션 성능은 이번 검증 범위에 포함하지 않았다.", "",
        "## 재현 자료", "",
        f"- [요약 JSON]({rel}/summary.json), [고정 프로토콜]({rel}/protocol.json), [개발 순위 고정]({rel}/selection_lock.json)",
        f"- [보정 계수 JSON]({rel}/calibration.json), [테스트 기록]({rel}/validation/checks.json), [전체 단위 테스트 로그]({rel}/validation/project_unit_tests.log)",
        f"- [원본 자료 명세]({rel}/origin_dataset.json), [원본 보존 기준]({rel}/protected_before.json), [학습 완료]({rel}/training_complete.json)",
        f"- 최종 {len(p['conditions'])}개 연구용 모델: `{rel}/final/<조건>/holdout/model.joblib`. 각 번들에 보정 계수·열 순서·길이·미승인 상태를 포함했다. 앱 설치용 패키지가 아니다.",
        f"- 혼동행렬·개인별 결과: `{rel}/final/<조건>/holdout/metrics.json`; 확률·정답·기록 매핑: 같은 폴더의 `predictions.npz`.",
        "", "```sh", f".venv/bin/python -m {p.get('runner_module', 'experiments.sensor_gain.run')} --output research/results/<새실행폴더>",
        ".venv/bin/python -m experiments.sensor_gain.summarize --output research/results/<새실행폴더>", "```", "",
        "고정 source_snapshot에서 학습을 실행해 진행 중인 다른 소스 변경이 학습 중간에 섞이지 않도록 했다. 정확한 소스 revision, dirty 상태, 라이브러리 버전, 명령, 입력·프로토콜·모델 해시는 JSON에 기록했다.", ""]
    if (out / "historical_et_reproduction.json").exists():
        parity = read(out / "historical_et_reproduction.json")
        if not parity["three_development_f1_exact"] or not parity["holdout_predictions_exact"]:
            raise ValueError("Historical ET reproduction failed")
        lines += ["## 규제 ET 식별과 과거 결과 재현", "",
            "일반 정자세 복원 체크포인트가 아닌 복원 전 규제 ET를 사용했다. 트리 200개, 최대 깊이 8, 최소 리프 표본 8, 특징 비율 0.5, class_weight=balanced, StandardScaler → ExtraTreesClassifier 구성이다.", "",
            "기존 ET 계약에 따라 모델 학습 시드는 모든 분할에서 20260921로 고정했다. 사람 분할과 그 생성 시드는 그대로 보존했다. 앞선 SVM/XGBoost/RF 실험의 분할별 모델 시드 정책과는 이 점이 다르며, 최종 학습 시드는 모두 20260921로 같다.", "",
            "3초·보정 전의 개발 3분할 Macro-F1과 최종 평가의 예측 클래스는 과거 규제 ET와 정확히 일치했다. 확률도 절대 오차 1e-12 이내로 일치했다. 따라서 데이터 복원 정책이나 규제 설정이 다른 ET와 혼동하지 않았음을 확인했다.", "",
            f"[과거 ET 재현 검증]({rel}/historical_et_reproduction.json) · [현재 추가 검사 로그]({rel}/validation/et_contract_tests.log)", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "analysis_verification.json", {"finished_at": now(), "verified_fits": expected_fits,
        "summary_sha256": digest(out / "summary.json"), "report": str(report), "report_sha256": digest(report),
        "summarizer_source_sha256": digest(Path(__file__)), "raw_and_active_models_unchanged": True})
    print(report)
    for c in comparisons:
        print(c["seconds"], c["family"], c["raw"]["window"]["macro_f1"], c["calibrated"]["window"]["macro_f1"], c["holdout_delta"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    summarize(parser.parse_args().output.resolve())
