"""Write the reviewed comparison report and publish only qualified UI recipes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .run import ROOT, read, digest, now
from src.model_comparison import write_json
from src.model_catalog import MODEL_NAMES
from src.training_methods import METHOD_LABELS


def generate(out, report):
    summary, dataset = read(out / "summary.json"), read(out / "dataset.json")
    protocol = read(out / "protocol.json")
    lock = read(out / "selection_lock.json")
    # Availability can only follow successful independent artifact verification.
    audit_file = out / "audits/final.json"
    if not audit_file.is_file():
        raise ValueError("Independent final artifact audit is required before registry publication")
    audit = read(audit_file)
    if audit.get("passed") is not True or digest(audit_file) != read(out / "audits/final.sha256.json")["sha256"]:
        raise ValueError("Final evidence audit failed or changed")
    for stage in ("development", "selection", "final"):
        stage_file = out / f"audits/{stage}.json"
        stage_audit = read(stage_file)
        if stage_audit.get("passed") is not True or digest(stage_file) != read(out / f"audits/{stage}.sha256.json")["sha256"]:
            raise ValueError(f"Independent {stage} audit failed or changed")
        for relative, sha in stage_audit["files"].items():
            if digest(out / relative) != sha:
                raise ValueError(f"Evidence changed after independent audit: {relative}")
    if digest(out / "dataset.json") != protocol["dataset_sha256"] or digest(out / "data.npz") != protocol["cache_sha256"]:
        raise ValueError("Frozen cohort or feature cache changed before publication")
    enabled = [row for row in summary["comparisons"] if row["improved"]]
    cohort = dataset["cohort"]
    split = dataset["partitions"][0]
    train_count, validation_count, test_count = (len(split[k]) for k in ("train", "validation", "test"))
    methods = []
    for method in dict.fromkeys(row["method"] for row in enabled):
        rows = [row for row in enabled if row["method"] == method]
        descriptions = [f"{MODEL_NAMES[r['family']]}: Macro F1 {100*r['baseline']['macro_f1']:.2f}% → {100*r['candidate']['macro_f1']:.2f}% (+{100*r['heldout_delta']:.2f}%p)" for r in rows]
        methods.append({"id": method, "label": METHOD_LABELS[method],
            "families": [row["family"] for row in rows], "window_seconds": [3],
            "evidence": {"improved": True, "scope": "retrospective_historical_holdout",
                "summary": "\n".join(descriptions) + f"\n기존 {cohort['subjects']}명 자료의 후향적 비교 · 신규 사용자 성능 보장 아님",
                "report": str(report.relative_to(ROOT)), "run": str(out.relative_to(ROOT)),
                "protocol_sha256": digest(out / "protocol.json"), "selection_sha256": digest(out / "selection_lock.json"),
                "summary_sha256": digest(out / "summary.json"), "independent_audit_sha256": digest(audit_file),
                "criteria": summary["criteria"],
                "by_family": {row["family"]: {key: row[key] for key in
                    ("mean_development_delta", "heldout_delta", "paired_subject_95_interval", "per_class_deltas", "params")}
                    for row in rows},
                "runtime_approval": "separate_existing_workflow", "active_model_changed": False}})
    registry = {"schema_version": 1, "created_at": now(), "methods": methods}
    write_json(out / "training_improvements.json", registry)

    korean_font = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    if korean_font.exists():
        font_manager.fontManager.addfont(str(korean_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(korean_font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    rows = summary["comparisons"]
    names = [MODEL_NAMES[row["family"]] + " · " + METHOD_LABELS[row["method"]] for row in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, 7.8), layout="constrained")
    fig.patch.set_facecolor("#f6f8fb")
    ax.barh(y - .17, [100 * row["mean_development_delta"] for row in rows], height=.32, color="#8298b8", label="개발 검증 평균 차이")
    evaluated = [(i, row) for i, row in enumerate(rows) if row["heldout_evaluated"]]
    ax.barh([i + .17 for i, _ in evaluated], [100 * row["heldout_delta"] for _, row in evaluated], height=.32,
            color=["#258a76" if row["improved"] else "#c9836e" for _, row in evaluated], label="고정 평가 차이")
    for i, row in enumerate(rows):
        if not row["heldout_evaluated"]:
            ax.text(.15, i + .24, "개발 기준 미충족 · 고정 평가 미실행", fontsize=8, color="#6d7684")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.axvline(0, color="#59677b", linewidth=.8)
    ax.set_xlabel("같은 모델의 기본 학습 대비 Macro F1 변화 (%p)")
    ax.set_title("현재 데이터 재실험 · 방식별 성능 변화", loc="left", fontweight="bold", pad=16)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "method_comparison.png", dpi=170)
    plt.close(fig)

    run_link = "../../" + out.relative_to(ROOT).as_posix()
    lines = ["# 현재 데이터 성능 개선 재실험과 모델 추가 모드", "",
        f"실험 시작: {protocol['started_at']} · 완료: {summary['finished_at']}. 센서 측정 시각을 의미하지 않습니다.", "",
        (f"**사전 기준을 통과한 방식·모델 조합 {len(enabled)}개를 모델 추가 화면의 학습 방식에 연결했습니다.** "
        if enabled else "**사전 기준을 통과한 개선 방식이 없어 모델 추가 화면은 기본 학습만 제공합니다.** ") +
        "기존 데이터를 반복 사용한 후향적 결과이며 신규 피험자 성능을 확정하지 않습니다. 활성 모델과 승인 상태는 변경하지 않았습니다.", "",
        "## 데이터와 평가", "",
        f"- 데이터: {cohort['subjects']}명 · {cohort['records']}개 평가용 기록 · {cohort['windows']}개 3초 창. 12 Hz, 36×16 입력, 기존 596개 특징.",
        (f"- 등록용 정자세 {cohort['reserved_registration_records']}개는 기록 전체를 모든 비교군에서 제외했습니다."
         if cohort['reserved_registration_records'] else "- 등록용 자동 배정을 해제하여 기존 정자세 기록도 일반 학습·평가 후보에 포함했습니다."),
        "- 현재 전역 제외 목록 7개를 반영했습니다. 현재 활성 라벨 교정 manifest가 없어 과거 교정본을 임의 복원하지 않았습니다. 과거 735개 전체 학습 보고서와 입력 조건이 다릅니다.",
        f"- 개발 분할 {len(dataset['partitions'])}개는 각각 학습 {train_count}명·검증 {validation_count}명입니다. 공통 고정 평가 {test_count}명은 후보 선택에 사용하지 않았습니다. 모든 분할에서 7개 클래스가 존재합니다.",
        "- 각 방식의 파라미터를 세 검증 Macro F1 평균으로 선택했습니다. 기본 학습도 현재 앱의 후보 그리드로 동일하게 다시 튜닝했습니다.",
        f"- 후보를 잠근 뒤 개발 {train_count + validation_count}명으로 다시 학습했습니다. CNN 최대 epoch는 개발 분할의 최적 epoch 중앙값으로 고정했고, 기존 최소 학습 손실 체크포인트 정책을 유지했습니다. 고정 평가로 조기 종료하지 않았습니다.",
        "- 사전 성공 기준: 평균 검증 +1%p 이상, 검증 3분할 중 2개 이상 상승, 고정 평가 +1%p 이상, 개별 자세 F1 하락 최대 5%p 이하.",
        "- 개발 기준을 통과하지 못한 방식의 고정 평가는 열지 않았습니다. 별도 근거가 없는 모델·창 길이는 개선 방식 선택에서 차단합니다.", "",
        "## 비교한 학습 방식", "",
        "| 방식 | 변경 내용 |",
        "|---|---|",
        "| 규제 재튜닝 | SVM의 C·gamma, 트리의 깊이·잎 크기, XGBoost의 규제값을 사전 지정 후보에서 다시 선택 |",
        "| 상대압력 특징 | 기존 596개 특징 중 rel_ 특징 298개를 모델 내부에서 선택 |",
        "| 핵심 특징 96개 | 학습 분할의 라벨만 이용한 ANOVA 특징 선택을 모델 내부에 저장 |",
        "| 사용자 균형 학습 | 사람별 총 가중치와 각 사람 안의 자세별 가중치를 균등화 |",
        "| CNN 센서 증강 | 학습 중에만 채널별 크기 변화와 작은 잡음을 적용; 추론에서는 원 신호 사용 |", "",
        "## 방식별 결과", "",
        "| 모델 | 방식 | 평균 검증 F1 | 기본 대비 | 기본 고정 F1 | 방식 고정 F1 | 고정 차이 | UI 제공 |",
        "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        fixed = ([f"{100*row['baseline']['macro_f1']:.2f}%", f"{100*row['candidate']['macro_f1']:.2f}%", f"{100*row['heldout_delta']:+.2f}%p"]
            if row["heldout_evaluated"] else ["—", "미평가", "—"])
        lines.append(f"| {MODEL_NAMES[row['family']]} | {METHOD_LABELS[row['method']]} | {100*row['mean_macro_f1']:.2f}% | {100*row['mean_development_delta']:+.2f}%p | "
            + " | ".join(fixed) + f" | {'제공' if row['improved'] else '미제공'} |")
    lines += ["", f"![방식별 성능 변화]({run_link}/method_comparison.png)", "",
        "## 다시 학습한 기본 모델", "",
        "아래는 각 모델의 기본 후보를 개발 검증에서 선택하고 동일한 26명으로 다시 학습한 기준값입니다. 저장된 활성 모델의 점수와 구분합니다.", "",
        "| 모델 | 평균 개발 검증 Macro F1 | 고정 평가 Macro F1 | 고정 평가 Accuracy |", "|---|---:|---:|---:|"]
    for selected in lock["selected"].values():
        if selected["method"] != "standard":
            continue
        metrics = read(out / "final" / selected["family"] / "standard/metrics.json")["metrics"]
        lines.append(f"| {MODEL_NAMES[selected['family']]} | {100*selected['mean_macro_f1']:.2f}% | {100*metrics['macro_f1']:.2f}% | {100*metrics['accuracy']:.2f}% |")
    lines += ["",
        "## 고정 평가 불확실성과 자세별 변화", "",
        "피험자 전체를 묶은 paired bootstrap 4,000회입니다. 이미 조사한 6명과 학습된 모델에 조건부인 기술적 구간이며 다중 비교나 새로운 학습 모집단의 불확실성을 보정하지 않았습니다. 구간에 0이 포함되면 안정적인 모집단 우월성으로 해석하지 않습니다.", "",
        "| 모델·방식 | F1 차이 95% 구간 (%p) | 가장 큰 자세별 하락 (%p) | 추론 중앙값 (ms) |",
        "|---|---:|---:|---:|"]
    for row in rows:
        if row["heldout_evaluated"]:
            lo, hi = row["paired_subject_95_interval"]
            lines.append(f"| {MODEL_NAMES[row['family']]} · {METHOD_LABELS[row['method']]} | [{100*lo:+.2f}, {100*hi:+.2f}] | {100*min(row['per_class_deltas'].values()):+.2f} | {row['candidate_inference_ms']:.3f} |")
    for row in enabled:
        lines += ["", f"### {MODEL_NAMES[row['family']]} · {METHOD_LABELS[row['method']]}", "",
            "고정 평가에 사용한 선택 파라미터: `" + json.dumps(row["params"], ensure_ascii=False, sort_keys=True) + "`.", "",
            "| 자세 | 기본 F1 | 개선 방식 F1 | 차이 |", "|---|---:|---:|---:|"]
        for label, delta in row["per_class_deltas"].items():
            lines.append(f"| {label} | {100*row['baseline']['per_class_f1'][label]:.2f}% | {100*row['candidate']['per_class_f1'][label]:.2f}% | {100*delta:+.2f}%p |")
    lines += ["", "## 모델 추가 화면", "",
        ("`모델 추가 → 구성 → 학습 방식`에서 제공된 방식을 고르고, 해당 방식에 표시된 모델과 3초 창을 선택합니다. 데이터 선택·검토·실행은 기존 흐름을 사용합니다. 방식과 근거가 실험 구성, 결과 및 모델 설정에 저장되며 설정 복제에도 유지됩니다."
        if enabled else "`모델 추가 → 구성 → 학습 방식` 선택 기능을 마련했으며 현재는 기본 학습만 표시합니다. 개선 기준을 통과하지 못한 실험 방식은 선택할 수 없습니다."), "",
        "학습 방식은 같은 알고리즘을 새 데이터에 적용하는 선택입니다. 기존 연구 모델을 자동 활성화하거나 승인하지 않습니다. 선택한 새 데이터의 검증 결과로 해당 방식 안의 파라미터를 선택하고, 기존 모델 검증·승인 절차를 거칩니다.", "",
        "## 재현·검증 근거", "",
        f"- [고정 입력·분할]({run_link}/dataset.json), [프로토콜]({run_link}/protocol.json), [선택 잠금]({run_link}/selection_lock.json)",
        f"- [전체 결과]({run_link}/summary.json), [독립 산출물 감사]({run_link}/audits/final.json), [UI 방식 근거]({run_link}/training_improvements.json)",
        f"- 공유 작업 공간의 동시 코드 변경을 감지한 후, 최초 source_snapshot의 알고리즘과 기준을 보존한 별도 실행 소스에서 이어 실행했습니다. [재개 소스·해시]({run_link}/execution_source_manifest.json). 원래 프로토콜과 입력은 변경하지 않았습니다.",
        f"- 남은 Random Forest 개발 학습은 완료된 산출물을 보존하고 동일한 동결 worker를 분할별로 병렬 실행했습니다. [실행 일정 변경 기록]({run_link}/partition_resume.json). 후보·시드·분할·선택 기준은 동일합니다.",
        "- 각 후보의 검증 확률·예측·피험자별 지표는 development/, 고정 평가 모델·예측·혼동행렬·지표는 final/에 함께 보관했습니다.",
        f"- 원본 보호와 소프트웨어/실제 Tk 검증 결과는 [validation.json]({run_link}/validation.json)에 별도 기록합니다. 새 macOS 앱 번들은 빌드하지 않았으며 실제 센서·신규 사용자·자세 전환 지연은 검증하지 않았습니다.",
        f"- 소스 revision: `{protocol['source_revision']}`. 시작 당시 dirty 상태·실행 명령·패키지 버전·원본 및 코드 SHA-256은 프로토콜과 source_snapshot에 보존했습니다.", ""]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(out / "training_improvements.json", ROOT / "src/training_improvements.json")
    print(f"REPORT {report}; published {len(methods)} methods for {len(enabled)} family/method pairs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    generate(args.output.resolve(), args.report.resolve())
