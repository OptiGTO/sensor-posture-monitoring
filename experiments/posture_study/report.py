"""Render the completed, selection-locked study as a Korean scientific report.

This module never trains, selects, or invokes a classifier. It consumes frozen
validation summaries and already generated test predictions. Bootstrap draws
resample whole users and preserve pairing between compared models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/posture-study-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .catalog import CATALOG
from .prepare import OUT, ROOT, write_json


REPORT = ROOT / "docs/experiments/posture_study_20260913_report.md"
BOOTSTRAP_SEED = 20260913
BOOTSTRAP_REPLICATES = 5000
METHODS = {
    "baseline": "596 특징 + StandardScaler + 균형 RBF-SVM",
    "relative": "상대압력 298 특징 + RBF-SVM",
    "spatial": "압력 중심·행열·비대칭 공간 특징 + RBF-SVM",
    "robust": "RobustScaler(10–90 분위수) + RBF-SVM",
    "select": "훈련 ANOVA 상위 96 특징 + RBF-SVM",
    "pca": "훈련 PCA 누적분산 95% + RBF-SVM",
    "personal": "등록 기준·기준 대비 차이 49 특징 추가 + RBF-SVM",
    "multiscale": "3초 특징 + 최근 1·2초 통계 + RBF-SVM",
    "subject_weight": "사용자별 총 가중치 균등 RBF-SVM",
    "gain_aug": "훈련 센서별 gain 증강 + RBF-SVM",
    "hierarchy": "정자세/오자세 SVM → 오자세 6종 SVM 하드 게이트",
    "soft_hierarchy": "정자세/오자세 SVM × 조건부 6종 SVM 확률",
    "extra_trees": "ExtraTrees 250개, leaf 최소 3",
    "histboost": "Histogram Gradient Boosting 150회, depth 5",
    "lda": "자동 공분산 수축 LDA",
    "knn": "거리 가중 9-nearest neighbors",
    "random_conv": "고정 무작위 시간 커널 128개 + Ridge",
    "majority": "SVM·로지스틱·ExtraTrees 다수결",
    "weighted": "사용자 분리 3-fold OOF 성능 가중 투표",
    "stacking": "사용자 분리 OOF 확률 + 로지스틱 메타 분류기",
    "1dcnn": "시간 Conv1D 32→48→64→64 + mean/max pooling",
    "spatial2dcnn": "프레임별 4×4 Conv2D 2→16→32 + 시간 평균",
    "lstm": "1-layer LSTM hidden 48 + 평균/마지막 상태",
    "gru": "1-layer GRU hidden 48 + 평균/마지막 상태",
    "cnn_lstm": "Conv1D 32→48 + LSTM hidden 48",
    "attention": "프레임 MLP 32→64 + 학습된 additive attention pooling",
    "transformer": "학습 위치 임베딩 + 2-layer Transformer, d48, 4-head",
    "tcn": "48채널 residual temporal conv, dilation 1·2·4",
    "mlp": "596→128→64→7, LayerNorm·GELU·Dropout",
    "aug_cnn": "1D-CNN + 훈련 센서 gain·관측 단위 잡음 증강",
    "hybrid": "지도학습 CNN 64차원 임베딩 + RBF-SVM",
    "ema": "B00 확률의 인과 EMA, 현재 0.6·이전 0.4",
    "hysteresis": "B00 새 자세 2회 연속 확인 후 전환",
    "equal_soft_vote": "검증된 구성원 확률의 동일 가중 평균",
}
FAILURE_REASONS = {
    "relative": "절대 하중을 제거하면서 실제 자세를 구분하는 압력 크기 정보까지 소실되었을 수 있다.",
    "spatial": "작은 공간 특징 집합이 센서별 세부 정보를 지나치게 압축했을 수 있다.",
    "robust": "극단값보다 사용자별 압력 패턴 차이가 주요 오차 요인일 수 있다.",
    "select": "훈련 사용자에서 높은 ANOVA 점수가 새 사용자 일반화를 보장하지 못했을 수 있다.",
    "pca": "큰 분산 방향과 자세 판별에 중요한 방향이 일치하지 않았을 수 있다.",
    "personal": "10초 기준의 대표성 부족이나 정자세 대비 차이의 비선형성 때문에 단순 특징 추가의 이득이 제한되었을 수 있다.",
    "multiscale": "정적 자세 중심의 짧은 기록에서 최근 1·2초 특징이 3초 통계와 중복되었을 수 있다.",
    "subject_weight": "기록 수 차이가 현재 오류를 지배하지 않거나 어려운 사용자의 잡음 가중치도 함께 커졌을 수 있다.",
    "gain_aug": "합성 곱셈 변동이 실제 센서의 비선형 응답·장착 변화와 다르거나 유용한 하중 차이를 흐렸을 수 있다.",
    "hierarchy": "첫 이진 게이트의 오류가 세부 분류 단계에서 복구되지 않았을 수 있다.",
    "soft_hierarchy": "게이트 확률의 보정 오차와 6종 세부 분류 오류가 함께 작용했을 수 있다.",
    "extra_trees": "사용자별 압력 값에 특화된 축 정렬 분할이 새 사용자의 분포 변화에 취약했을 수 있다.",
    "histboost": "작은 사용자 집합의 잔여 오차를 반복적으로 맞추며 과적합했을 수 있다.",
    "lda": "클래스가 공유 공분산을 가진 선형 경계로 충분히 분리되지 않았을 수 있다.",
    "knn": "센서 특징 거리에서 자세 차이보다 사용자 차이가 크게 나타났을 수 있다.",
    "random_conv": "고정 커널이 정적인 압력 분포를 충분히 표현하지 못하거나 시간 정보가 제한적이었을 수 있다.",
    "majority": "구성원 오류가 강하게 겹치거나 약한 구성원의 표가 좋은 SVM의 결정을 바꿨을 수 있다.",
    "weighted": "OOF 전체 점수 하나로 정한 가중치가 사용자별·클래스별 신뢰도 차이를 설명하지 못했을 수 있다.",
    "stacking": "적은 훈련 사용자에서 생성한 OOF 확률로는 메타 분류기의 일반화가 충분하지 않았을 수 있다.",
    "ema": "같은 기록 안에서 지속되는 체계적 오분류는 확률 평균만으로 수정하기 어려웠을 수 있다.",
    "hysteresis": "일시적 오분류 억제 이득보다 잘못된 이전 상태를 유지한 손실이 컸을 수 있다.",
    "hybrid": "CNN이 이미 학습한 사용자 특화 표현을 SVM으로 바꾸는 것만으로는 일반화가 개선되지 않았을 수 있다.",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def number(value: float, signed: bool = False) -> str:
    return f"{100 * value:+.2f}" if signed else f"{100 * value:.2f}"


def clean(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def relative_link(path: Path, report: Path, label: str) -> str:
    return f"[{label}]({os.path.relpath(path, report.parent)})"


def metric_from_cm(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cm = np.asarray(cm, dtype=float)
    diagonal = np.diagonal(cm, axis1=-2, axis2=-1)
    denominator = cm.sum(axis=-1) + cm.sum(axis=-2)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal), where=denominator > 0).mean(axis=-1)
    accuracy = diagonal.sum(axis=-1) / cm.sum(axis=(-1, -2))
    return accuracy, f1


def subject_bootstrap(reference_path: Path, final_path: Path,
                      seed: int = BOOTSTRAP_SEED,
                      replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    with np.load(reference_path, allow_pickle=False) as source:
        reference = {key: source[key] for key in source.files}
    with np.load(final_path, allow_pickle=False) as source:
        final = {key: source[key] for key in source.files}
    for key in ("y", "groups"):
        if not np.array_equal(reference[key], final[key]):
            raise ValueError(f"Bootstrap pairs do not share identical ordered {key}")
    for key in ("record_ids", "start_s"):
        if key in reference and key in final and not np.array_equal(reference[key], final[key]):
            raise ValueError(f"Bootstrap pairs do not share identical ordered {key}")
    subjects = sorted(np.unique(final["groups"]).tolist())
    reference_pred = reference["proba"].argmax(axis=1)
    final_pred = final["proba"].argmax(axis=1)
    cms = []
    for predictions in (reference_pred, final_pred):
        cms.append(np.stack([
            np.bincount(final["y"][final["groups"] == group] * 7 + predictions[final["groups"] == group], minlength=49).reshape(7, 7)
            for group in subjects
        ]))
    draws = np.random.default_rng(seed).integers(0, len(subjects), size=(replicates, len(subjects)))
    reference_acc, reference_f1 = metric_from_cm(cms[0][draws].sum(axis=1))
    final_acc, final_f1 = metric_from_cm(cms[1][draws].sum(axis=1))
    observed_reference = metric_from_cm(cms[0].sum(axis=0))
    observed_final = metric_from_cm(cms[1].sum(axis=0))
    result = {
        "method": "paired_subject_cluster_percentile_bootstrap",
        "seed": seed, "replicates": replicates, "subjects": subjects,
        "confidence_level": .95, "resampling_unit": "whole subject, pooled window metrics recomputed",
        "reference_predictions": str(reference_path), "final_predictions": str(final_path),
        "caveat": "Only four historical held-out users; intervals do not establish prospective population generalization.",
    }
    for name, delta, observed in (
        ("accuracy", final_acc - reference_acc, observed_final[0] - observed_reference[0]),
        ("macro_f1", final_f1 - reference_f1, observed_final[1] - observed_reference[1]),
    ):
        result[name] = {"observed_delta": float(observed),
                        "ci95": np.quantile(delta, [.025, .975]).tolist(),
                        "bootstrap_positive_fraction": float((delta > 0).mean())}
    return result


def validation_summary(spec: dict, output: Path) -> dict:
    runs = [load_json(path) for path in sorted((output / spec["id"]).glob("*/validation.json"))]
    if len(runs) != 3:
        raise ValueError(f"{spec['id']} needs exactly three validation records")
    primary = next(run for run in runs if run["split"]["id"] == "primary")
    return {
        "spec": spec, "runs": runs, "primary_run": primary,
        "macro_f1": float(np.mean([run["metrics"]["macro_f1"] for run in runs])),
        "accuracy": float(np.mean([run["metrics"]["accuracy"] for run in runs])),
        "std_macro_f1": float(np.std([run["metrics"]["macro_f1"] for run in runs])),
        "split_f1": {run["split"]["id"]: run["metrics"]["macro_f1"] for run in runs},
        "class_f1": np.mean([run["metrics"]["f1"] for run in runs], axis=0),
        "worst_class_f1": float(np.mean([run["metrics"]["worst_class_f1"] for run in runs])),
        "latency_ms": primary["inference"]["single_window_ms_median"],
        "train_seconds": float(sum(run["train_seconds"] for run in runs)),
        "checkpoint_bytes": primary["checkpoint_bytes"],
    }


def setup_plot_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    name = next((name for name in ("AppleGothic", "Noto Sans CJK KR", "NanumGothic", "Arial Unicode MS") if name in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": name, "axes.unicode_minus": False,
                         "font.size": 10, "figure.dpi": 120, "savefig.dpi": 180})
    return name


def create_charts(output: Path, manifest: dict, lock: dict,
                  summaries: dict, tests: dict) -> dict[str, Path]:
    folder = output / "charts"
    folder.mkdir(parents=True, exist_ok=True)
    setup_plot_font()
    paths = {}
    final_id = lock["selected_id"]
    labels = manifest["class_names"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    for ax, identifier, title in zip(axes, ("B00", final_id), ("공통 재학습 Baseline", f"검증으로 고정한 최종 {final_id}")):
        cm = np.asarray(tests[identifier]["metrics"]["confusion_matrix"])
        normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set(title=title, xlabel="예측", ylabel="실제", xticks=np.arange(7), yticks=np.arange(7),
               xticklabels=labels, yticklabels=labels)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{cm[i,j]}\n{normalized[i,j]:.0%}", ha="center", va="center",
                        fontsize=8, color="white" if normalized[i,j] > .55 else "black")
    fig.colorbar(im, ax=axes, label="실제 클래스 내 비율", shrink=.8)
    paths["confusion"] = folder / "baseline_final_confusion.png"
    fig.savefig(paths["confusion"]); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.7), constrained_layout=True)
    x = np.arange(7)
    ax.bar(x - .18, np.asarray(tests["B00"]["metrics"]["f1"]) * 100, .36, label="B00", color="#73869c")
    ax.bar(x + .18, np.asarray(tests[final_id]["metrics"]["f1"]) * 100, .36, label=final_id, color="#087f8c")
    ax.set(xticks=x, xticklabels=labels, ylabel="테스트 클래스별 F1 (%)", ylim=(0, 105), title="동일 held-out 창의 클래스별 성능")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    paths["per_class"] = folder / "baseline_final_per_class_f1.png"
    fig.savefig(paths["per_class"]); plt.close(fig)

    ids = sorted([row["id"] for row in CATALOG if row["id"] != "B00"], key=lambda identifier: summaries[identifier]["macro_f1"])
    fig, ax = plt.subplots(figsize=(10, 11), constrained_layout=True)
    for i, identifier in enumerate(ids):
        deltas = np.array([summaries[identifier]["split_f1"][key] - summaries["B00"]["split_f1"][key]
                           for key in summaries[identifier]["split_f1"]]) * 100
        color = "#087f8c" if identifier in lock["successful_ids"] else "#8894a5"
        ax.plot([deltas.min(), deltas.max()], [i, i], color=color, linewidth=2)
        ax.scatter(deltas, [i] * 3, color=color, s=15, alpha=.6)
        ax.scatter([deltas.mean()], [i], color=color, s=50, marker="D")
    ax.set(yticks=np.arange(len(ids)), yticklabels=[f"{identifier} {summaries[identifier]['spec']['method']}" for identifier in ids],
           xlabel="동일 검증 분할 대비 Macro F1 변화 (%p)", title="독립 32가설: 점=3분할, 선=최소–최대, 마름모=평균")
    ax.axvline(0, color="black", linewidth=.8); ax.axvline(.5, color="#d89c34", linestyle="--", linewidth=.8)
    ax.grid(axis="x", alpha=.2)
    paths["validation"] = folder / "validation_delta_ranges.png"
    fig.savefig(paths["validation"]); plt.close(fig)

    steps = lock.get("steps", [])
    names = ["B00"] + [step["id"] for step in steps if "id" in step]
    names += [ablation["spec"]["id"] for ablation in lock.get("ablations", [])]
    names = list(dict.fromkeys(names))
    fig, ax = plt.subplots(figsize=(max(9, len(names) * .8), 5), constrained_layout=True)
    x = np.arange(len(names))
    colors = ["#087f8c" if identifier == final_id else "#73869c" for identifier in names]
    means = [100 * summaries[identifier]["macro_f1"] for identifier in names]
    stds = [100 * summaries[identifier]["std_macro_f1"] for identifier in names]
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    ax.bar_label(bars, fmt="%.2f", padding=6, fontsize=8)
    ax.set(xticks=x, xticklabels=names, ylabel="평균 검증 Macro F1 (%)", title="순차 결합·제거 실험 (오차막대: 3분할 표준편차)")
    ax.grid(axis="y", alpha=.2)
    paths["ablation"] = folder / "combination_ablation.png"
    fig.savefig(paths["ablation"]); plt.close(fig)
    return paths


def architecture_name(spec: dict) -> str:
    name = METHODS.get(spec["method"], spec["method"])
    if spec["family"] == "temporal" and spec.get("parent", "B00") != "B00":
        name = name.replace("B00", spec["parent"])
    if spec.get("components"):
        name += " + " + ", ".join(spec["components"])
    if spec.get("members"):
        name += " (" + ", ".join(spec["members"]) + ")"
    if spec.get("parent"):
        name += " (parent " + spec["parent"] + ")"
    return name


def parameter_count(complexity: dict) -> str:
    for key in ("trainable_parameters", "numeric_parameter_count", "tree_nodes", "stored_numeric_values"):
        if key in complexity:
            return f"{complexity[key]:,} " + {"tree_nodes": "노드", "stored_numeric_values": "저장값"}.get(key, "수치값")
    if "model" in complexity:
        return parameter_count(complexity["model"])
    if "members" in complexity:
        return f"{len(complexity['members'])}개 구성원; 세부 JSON"
    return "세부 JSON 참조"


def pipeline_mermaid(lock: dict) -> str:
    specs = {spec["id"]: spec for spec in lock["all_specs"]}
    lines = ["```mermaid", "flowchart TD", '  raw["16셀 압력 · 약 12 Hz"] --> win["3초 창 · 36프레임 · stride 3초"]']
    added = set()

    def append(identifier: str) -> str:
        if identifier in added:
            return identifier
        added.add(identifier)
        spec = specs[identifier]
        if spec["family"] == "fusion":
            for member in spec["members"]:
                lines.append(f'  {append(member)} --> {identifier}["{identifier}: 구성원 확률 평균 · 각 0.5"]')
        elif spec["family"] == "temporal":
            parent = spec.get("parent", "B00")
            temporal_label = "EMA: 현재 0.6 + 이전 상태 0.4" if spec["method"] == "ema" else "hysteresis: 2회 확인"
            lines.append(f'  {append(parent)} --> {identifier}["{identifier}: {temporal_label}"]')
        elif spec["method"] == "stacking":
            lines.append(f'  win --> {identifier}_features["기존 특징 596차원 · 훈련 전처리"]')
            for member in spec.get("ensemble_members", ["baseline", "logreg", "extra_trees"]):
                label = {"baseline": "RBF-SVM: C=1, gamma=0.0005", "logreg": "로지스틱 회귀: C=0.1", "extra_trees": "ExtraTrees: 250개, min leaf=3"}[member]
                lines.append(f'  {identifier}_features --> {identifier}_{member}["{label}"]')
                lines.append(f'  {identifier}_{member} --> {identifier}_concat["3×7 = 21 확률 연결"]')
            lines.append(f'  {identifier}_concat --> {identifier}["{identifier}: StandardScaler + OOF 로지스틱 메타 C=0.1"]')
        elif spec["method"] in ("aug_cnn", "1dcnn"):
            lines.append(f'  win --> {identifier}_input["관측값·비율 32채널 · 훈련 통계 표준화"]')
            lines.append(f'  {identifier}_input --> {identifier}["{identifier}: Conv 32→48 k5→64 k3→64 k3; mean/max 128→64→7"]')
        else:
            description = architecture_name(spec).replace('"', "'")
            lines.append(f'  win --> {identifier}["{identifier}: {description}"]')
            if spec["method"] == "personal" or "personal" in spec.get("components", []):
                lines.append(f'  registration["별도 정자세 등록 첫 10초 기준"] --> {identifier}')
        return identifier

    node = append(lock["selected_id"])
    lines.extend([f'  {node} --> decision["7클래스 점수 argmax"]', '  decision --> posture["자세 출력"]', "```"])
    return "\n".join(lines)


def success_reason(method: str) -> str:
    reasons = {
        "relative": "절대 하중보다 압력의 상대 분포가 사용자 사이에서 더 안정적이라는 설명과 부합한다.",
        "personal": "등록 기준 대비 편차가 절대 센서 값만으로 설명하기 어려운 개인차 정보를 보충했다는 설명과 부합한다.",
        "multiscale": "전체 창 통계와 최근 부분 창 통계에 상보적 정보가 있다는 설명과 부합한다.",
        "subject_weight": "사용자별 훈련 기여 균등화가 일부 사용자로의 편향을 줄였다는 설명과 부합한다.",
        "robust": "일부 특징의 극단값 영향을 낮추는 전처리가 일반화에 도움이 되었다는 설명과 부합한다.",
        "pca": "상관된 특징을 압축하여 유효 학습 복잡도를 낮춘 효과와 부합한다.",
        "select": "판별력이 낮은 특징을 제거하여 과적합이 줄었다는 설명과 부합한다.",
        "ema": "같은 정적 기록 내의 짧은 확률 흔들림이 평균화되었다는 설명과 부합한다.",
        "hysteresis": "단발성 예측 전환을 제한한 효과와 부합한다.",
    }
    return reasons.get(method, "기존 특징·결정 경계에서 놓친 정보를 보완하거나 오차 분산을 줄였다는 설명과 부합한다.")


def generate(output: Path = OUT, report: Path = REPORT) -> Path:
    output, report = Path(output), Path(report)
    required = ["dataset.json", "selection_lock.json", "test_summary.json", "independent_analysis.json", "protocol.json"]
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise RuntimeError("완료된 selection lock 및 test 결과가 필요합니다: " + ", ".join(missing))
    manifest = load_json(output / "dataset.json")
    lock = load_json(output / "selection_lock.json")
    test_summary = load_json(output / "test_summary.json")
    protocol = load_json(output / "protocol.json")["protocol"]
    if lock["test_used_for_selection"] or test_summary["locked_selected_id"] != lock["selected_id"]:
        raise ValueError("Selection lock and test summary disagree")
    lock_hash = hashlib.sha256((output / "selection_lock.json").read_bytes()).hexdigest()
    tests = {result["spec"]["id"]: result for result in test_summary["results"]}
    if any(result["selection_lock_sha256"] != lock_hash for result in tests.values()):
        raise ValueError("A test result references another selection lock")
    summaries = {spec["id"]: validation_summary(spec, output) for spec in lock["all_specs"]}
    final_id = lock["selected_id"]
    base, final = tests["B00"]["metrics"], tests[final_id]["metrics"]
    stored = test_summary["stored_baseline"]["metrics"]
    val_base, val_final = summaries["B00"], summaries[final_id]
    bootstrap = {
        "common_baseline": subject_bootstrap(output / "B00/primary/test_predictions.npz", output / final_id / "primary/test_predictions.npz"),
        "stored_baseline": subject_bootstrap(output / "stored_baseline_test_predictions.npz", output / final_id / "primary/test_predictions.npz"),
    }
    write_json(output / "paired_subject_bootstrap.json", bootstrap)
    charts = create_charts(output, manifest, lock, summaries, tests)
    parts = manifest["partitions"]
    success = set(lock["successful_ids"])
    confirmed_success = {
        identifier for identifier in success
        if tests[identifier]["metrics"]["macro_f1"] > base["macro_f1"]
        and int((np.asarray(tests[identifier]["metrics"]["f1"]) >= np.asarray(base["f1"])).sum()) >= 4
    }
    sections: list[str] = []
    add = sections.append
    add("# 자세 분류 모델 개선 실험 보고서 — 2026-09-13\n")
    if final["macro_f1"] <= base["macro_f1"]:
        add("**결론: 공통 Baseline보다 일관되게 좋아진 최종 모델을 확보하지 못했다. 검증으로 고정한 복합 후보는 held-out 테스트에서 성능이 하락하여 채택하지 않는다. 기존 생산 파이프라인과 활성 모델은 유지한다.**\n")
    add(f"독립 가설 **32개**를 동일한 사용자 분리 조건의 **3개 검증 분할**에서 평가했다. Baseline·결합·ablation을 포함해 총 **{len(lock['all_specs'])}개 구성, {3*len(lock['all_specs'])}개 분할별 평가 기록**을 남겼다. 최종 실험 후보는 테스트를 열기 전에 **{final_id} ({architecture_name(lock['selected_spec'])})**로 고정했다. 공통 재학습 Baseline 대비 평균 검증 Macro F1은 **{number(val_base['macro_f1'])}% → {number(val_final['macro_f1'])}% ({number(val_final['macro_f1']-val_base['macro_f1'],True)}%p)**, 동일 held-out 테스트 Macro F1은 **{number(base['macro_f1'])}% → {number(final['macro_f1'])}% ({number(final['macro_f1']-base['macro_f1'],True)}%p)**이다.\n")
    if final["macro_f1"] <= base["macro_f1"]:
        add("**검증 선택 결과가 최종 테스트에서 개선으로 확인되지 않았다.** 테스트에서 더 높은 후보로 최종 선택을 바꾸지 않았다. 이 실험을 배포 성능 향상의 증거로 해석해서는 안 된다.\n")
    add("이는 기존 데이터를 이용한 오프라인 비교다. 테스트 사용자는 과거 작업에서 이미 평가된 사용자이므로 완전히 새로운 전향적 시험이 아니다. 개인화 등록은 별도 정자세 기록으로 만든 소급적 대리 조건이며 실제 신규 사용자 등록, 센서 하드웨어, 연속 자세 전환, 실시간 UI 배포는 검증하지 않았다. 기존 생산 코드·활성 체크포인트를 교체하지 않았다.\n")

    add("## 1. Baseline과 공통 실험 조건\n")
    add("기존 활성 체크포인트를 그대로 평가한 결과와, 모든 가설에 공통으로 적용한 현재 데이터 조건에서 다시 학습한 **B00**을 구분한다. 실험의 개선량·성공 판정 기준은 B00이다. 원본 체크포인트는 당시 훈련 등록 기록을 포함한 기존 학습 이력을 유지하므로 공통 재학습 실험군과 훈련 입력이 완전히 같지는 않다. 최종 테스트 창은 세 비교 대상에서 완전히 같다.\n")
    add("| 비교 대상 | Accuracy (%) | Macro F1 (%) | 의미 |\n| --- | ---: | ---: | --- |")
    for name, score, meaning in (("B00 공통 재학습", base, "동일 데이터·창·분할의 주 대조군"), ("원본 활성 체크포인트", stored, "현재 파이프라인 파일을 수정하지 않은 참고 비교"), (f"최종 {final_id}", final, "검증 결과만으로 선택 고정")):
        add(f"| {name} | {number(score['accuracy'])} | {number(score['macro_f1'])} | {meaning} |")
    add(f"\n- 창: {manifest['window_seconds']}초, 약 {manifest['sampling_rate']} Hz, 36프레임×16셀; stride {manifest['stride_seconds']}초, tail_fraction={manifest['tail_fraction']}.\n- 기존 특징: {manifest['feature_dimension']}차원. 특징 추출은 생산 함수의 동일 구현을 사용한다. 각 가설이 바꾸는 입력 표현·특징 선택은 실험 요인으로 별도 기록한다.\n- 모든 사용자에서 등록용 첫 정자세 기록 전체를 모든 실험군의 훈련·검증·테스트에서 제외했다. 개인화는 그 기록의 첫 10초 평균만 참조한다.\n- 데이터 지문: `{manifest['fingerprint']}`. 원본 파일·sidecar SHA-256, 제외 기록, 창별 출처는 " + relative_link(output / "dataset.json", report, "dataset.json") + "에 기록했다.\n")
    audit = manifest["audit"]
    observed_records = {row["record"] for row in manifest["rows"]}
    missing_correct = [subject for subject in sorted(audit["subject_class_record_counts"])
                       if not any(row["subject"] == subject and row["label"] == manifest["class_names"][0] for row in manifest["rows"])]
    add(f"데이터 흐름: 후보 {audit['eligible_record_count']+audit['excluded_record_count']}기록 → 라벨 규칙 미충족 {audit['excluded_record_count']}기록 제외 → 유효 {audit['eligible_record_count']}기록·{audit['subject_count']}사용자 → 등록용 {len(manifest['registrations'])}기록 제외 → 평가용 {len(observed_records)}기록·{len(manifest['rows'])}창. 등록 기록을 제외하면 {', '.join(missing_correct)}에게는 정자세 평가 기록이 남지 않는다. 다만 모든 훈련·검증·테스트 분할은 전체적으로 7개 클래스를 포함하며, 일부 사용자별 Macro F1에는 관측되지 않은 클래스의 F1=0이 포함될 수 있다.\n")
    add("| 분할 | Seed | 훈련 사용자/기록/창 | 검증 사용자/기록/창 | 테스트 사용자/기록/창 |\n| --- | ---: | --- | --- | --- |")
    for part in parts:
        values = ["/".join(str(part["counts"][split][key]) for key in ("subjects", "records", "windows")) for split in ("train", "validation", "test")]
        add(f"| {part['id']} | {part['seed']} | " + " | ".join(values) + " |")
    for part in parts:
        add(f"\n**{part['id']} 사용자 목록** — 훈련: {', '.join(part['train'])}; 검증: {', '.join(part['validation'])}; 테스트: {', '.join(part['test'])}.")
    add("\n사용자 ID를 외부 분할의 그룹 단위로 사용하므로 같은 사용자의 측정 기록·창이 훈련과 검증/테스트에 걸치지 않는다. 내용 중복 검사를 통과한 데이터만 준비하며, 전처리 통계·PCA·특징 선택은 훈련에서만 적합한다. Ensemble 가중치·stacking 메타 입력은 훈련 사용자 내부 GroupKFold OOF로 만든다. SVC의 `probability=True` 내부 보정은 외부 훈련 범위에 한정되지만 libsvm의 내부 보정 폴드는 사용자 그룹 단위가 아니라는 한계가 있다.\n")
    add("테스트로 전처리 통계나 특징 선택을 적합하지 않는 원칙은 [scikit-learn 데이터 누수 지침](https://scikit-learn.org/1.8/common_pitfalls.html#data-leakage)에 따른다. 재현성을 위해 seed와 결정적 연산을 지정했지만 패키지·장치·플랫폼이 달라지면 완전히 동일한 결과를 보장할 수 없으므로 환경 버전도 기록한다. [PyTorch 재현성 문서](https://docs.pytorch.org/docs/stable/notes/randomness.html).\n")

    add("## 2. 사전 정의한 32개 가설과 구현\n")
    add("각 가설은 B00 대비 독립 실험으로 먼저 실행했다. 추가 가설에는 강건한 스케일링, 사용자 가중 학습, gain 증강, 무작위 시간 커널, 수축 LDA 등이 포함된다. 아래 이유는 실험 전 기대 메커니즘이며 결과를 보고 입증된 인과 설명으로 바꾸지 않는다.\n")
    add("| ID | 가설과 기대 이유 | 실제 구현 |\n| --- | --- | --- |")
    for spec in CATALOG[1:]:
        add(f"| {spec['id']} | {clean(spec['hypothesis'])} | {clean(architecture_name(spec))} |")
    add("\n신경망 9종은 사전학습 가중치 없이 실험용 모듈로 구현했다. 공통 학습 설정은 AdamW(lr=0.001, weight_decay=0.001), 최대 60 epochs, patience 10, batch 64, dropout 0.15, class-balanced cross entropy, gradient clip 2.0, 결정적 CPU 연산 2 threads이다. 입력은 관측 센서값의 고정 수치 스케일 변환과 프레임별 압력 비율의 32채널이며 훈련 채널 평균·표준편차로 정규화한다. 코드의 고정 /1023 변환은 뒤의 훈련 표준화로 상쇄되는 수치 변환이며 현재 기록이 ADC 단위라는 뜻은 아니다. 관측값은 대략 1.04–2.89 범위다. MLP는 훈련 정규화한 기존 특징을 사용한다. 최적 epoch는 검증 Macro F1, 동률이면 검증 loss로 정한다. 선택된 epoch·전체 학습 이력·구조 repr은 각 JSON과 checkpoint에 저장한다.\n")
    correction = output / "superseded/unit_scale_v1/reason.json"
    if correction.exists():
        add("**H29 구현 오류와 정정 기록:** 테스트를 열기 전에 최초 증강 구현이 센서값을 0–1023 ADC라고 가정하여 실제 약 1–3 단위 자료에 과도한 잡음을 더하는 오류를 발견했다. 초기 H29 결과(평균 검증 Macro F1 약 45.16%)와 이에 의존한 첫 결합·선택 기록은 `superseded/unit_scale_v1/`에 보존하고 최종 증거에서 제외했다. H29를 **gain 표준편차 0.10, 잡음 표준편차=0.003×훈련에서만 계산한 센서별 관측값 표준편차, 비음수 절단**으로 수정해 세 분할 모두 다시 학습했다. 성공 판정·결합·ablation·선택 고정을 다시 완료한 뒤 테스트를 평가했다. 본 표와 최종 결과는 수정된 구현이다. " + relative_link(correction, report, "정정 사유·폐기 기록") + ".\n")
    add("구현 파일: " + ", ".join(relative_link(ROOT / "experiments/posture_study" / name, report, name) for name in ("classical.py", "neural.py", "prepare.py", "run.py", "catalog.py")) + ".\n")

    add("## 3. 독립 실험 결과 전체 표\n")
    add("Accuracy·Macro F1·Change vs. Baseline은 **동일 primary held-out 테스트** 결과다. 별도의 검증 열은 세 분할 평균±표준편차다. 성공/보류 판정은 테스트와 무관하게 고정한 검증 규칙을 따른다. 단위는 성능 %, 변화 %p, 비용 ms/창이다.\n")
    headers = ["ID", "Hypothesis", "Architecture / Method", "Accuracy", "Macro F1", "Change vs. Baseline", "Inference Cost", "Decision", "검증 Macro F1 ± SD", "검증 변화"]
    add("| " + " | ".join(headers) + " |\n| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |")
    csv_rows = []
    for spec in CATALOG:
        identifier = spec["id"]; summary = summaries[identifier]; score = tests[identifier]["metrics"]
        decision = "대조군" if identifier == "B00" else "성공 확인" if identifier in confirmed_success else "검증 유망; 테스트 미확인" if identifier in success else "보류/탈락"
        if identifier == final_id: decision += "; 최종 선택"
        values = [identifier, spec["hypothesis"], architecture_name(spec), number(score["accuracy"]), number(score["macro_f1"]),
                  number(score["macro_f1"] - base["macro_f1"], True), f"{summary['latency_ms']:.3f} ms", decision,
                  f"{number(summary['macro_f1'])} ± {number(summary['std_macro_f1'])}", number(summary['macro_f1'] - val_base["macro_f1"], True)]
        add("| " + " | ".join(clean(value) for value in values) + " |")
        csv_rows.append(dict(zip(headers, values)))
    with (output / "experiment_table.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers); writer.writeheader(); writer.writerows(csv_rows)
    add("\n" + relative_link(output / "experiment_table.csv", report, "전체 표 CSV") + ".\n")
    add("![독립 가설의 검증 변화 범위](" + os.path.relpath(charts["validation"], report.parent) + ")\n")
    add("세 점은 서로 다른 사용자 분할과 seed를 동시에 바꾼 반복이다. 이는 순수한 seed 민감도 시험이나 독립적인 세 모집단 시험이 아니다. 세 분할이 같은 개발 사용자 집합을 재사용하므로 평균·표준편차를 신뢰구간으로 해석하지 않는다.\n")

    add("## 4. Successful Hypotheses 및 검증 단계의 유망 가설\n")
    if not confirmed_success:
        add("**Successful Hypotheses: 없음.** 사전 검증 규칙을 통과하고 동일 held-out 테스트에서도 B00 대비 Macro F1·클래스별 개선 방향이 확인된 독립 가설이 없다. 아래 4개는 검증 단계에서만 유망했던 가설이며 최종 성공 목록에 넣지 않는다. 테스트를 보고 선택을 바꾸거나 다른 후보를 새 승자로 선언하지 않았다.\n")
    else:
        add("검증 규칙과 테스트 확인 조건을 통과한 가설: **" + ", ".join(sorted(confirmed_success)) + "**. 아래에는 선택 당시의 검증 통과군 전체와 테스트 확인 여부를 구분한다.\n")
    add("사전 규칙: 평균 검증 Macro F1 **+0.5%p 이상**, 3분할 중 **2개 이상 개선**, 분할별 최저 클래스 F1의 평균 하락 **2%p 이하**, 평균 클래스 F1이 **7개 중 4개 이상 비열화**. 정확도만 소폭 증가한 가설은 성공으로 분류하지 않는다.\n")
    add("| ID | 검증 ΔMacro F1 (%p) | 개선 분할 | 비열화 클래스 | 테스트 ΔMacro F1 (%p) | 테스트 확인 | 가능한 설명 |\n| --- | ---: | ---: | ---: | ---: | --- | --- |")
    for identifier in lock["successful_ids"]:
        summary = summaries[identifier]
        positive = sum(summary["split_f1"][key] > val_base["split_f1"][key] for key in summary["split_f1"])
        classes = int((summary["class_f1"] >= val_base["class_f1"]).sum())
        test_delta = tests[identifier]["metrics"]["macro_f1"] - base["macro_f1"]
        add(f"| {identifier} | {number(summary['macro_f1']-val_base['macro_f1'], True)} | {positive}/3 | {classes}/7 | {number(test_delta,True)} | {'방향 일치' if test_delta>0 else '개선 미확인'} | {success_reason(summary['spec']['method'])} |")
    if not success: add("| — | — | — | — | — | 성공 규칙 통과 가설 없음 | B00 유지 |")
    add("\n검증 유망은 개발 자료에서 사전 규칙을 통과했다는 뜻이다. 테스트 확인 여부는 선택 고정 이후의 별도 판단이다. 방향이 일치하더라도 모든 클래스·모든 사용자에서 개선하거나 원인이 입증되었다는 뜻은 아니다. 아래 클래스별·사용자별 결과와 불확실성을 함께 판단한다.\n")

    add("## 5. 실패·보류 가설 분석\n")
    add("아래 원인 설명은 **후속 실험이 필요한 추정**이다. 현재 비교는 한 구현·한 고정 학습 예산에 대한 결과이며, 해당 모델 계열 전체의 가능성을 부정하지 않는다.\n")
    add("| ID | 규칙 미충족 근거 | 가능한 실패 원인 |\n| --- | --- | --- |")
    for spec in CATALOG[1:]:
        identifier = spec["id"]
        if identifier in success: continue
        summary = summaries[identifier]
        reasons = []
        if summary["macro_f1"] - val_base["macro_f1"] < .005: reasons.append("평균 검증 개선 <0.5%p")
        if sum(summary["split_f1"][key] > val_base["split_f1"][key] for key in summary["split_f1"]) < 2: reasons.append("개선 분할 <2")
        if summary["worst_class_f1"] - val_base["worst_class_f1"] < -.02: reasons.append("최저 클래스 평균 >2%p 하락")
        if int((summary["class_f1"] >= val_base["class_f1"]).sum()) < 4: reasons.append("비열화 클래스 <4")
        failure = FAILURE_REASONS.get(spec["method"], "사용자 수가 작은 정적 자세 데이터에서 시간 표현의 이득보다 표현 학습의 표본 요구량·과적합이 컸을 수 있다. 최대 60 epoch의 공통 학습 예산도 해당 구조에 최적이라는 보장은 없다.")
        add(f"| {identifier} | {'; '.join(reasons)} | {failure} |")

    add("\n검증 규칙은 통과했지만 테스트에서 확인되지 않은 가설의 추가 실패 분석:\n")
    add("| ID | 테스트 ΔMacro F1 (%p) | 가능한 일반화 실패 원인 (추정) |\n| --- | ---: | --- |")
    for identifier in lock["successful_ids"]:
        if identifier in confirmed_success: continue
        method = summaries[identifier]["spec"]["method"]
        failure = FAILURE_REASONS.get(method, "합성 gain·잡음 변동은 개발 사용자에 도움이 되었지만 실제 테스트 사용자의 압력 분포·센서 변화 전체를 설명하지 못했을 수 있다.")
        add(f"| {identifier} | {number(tests[identifier]['metrics']['macro_f1']-base['macro_f1'],True)} | {failure} |")

    add("## 6. 검증 유망 요인의 결합과 단계별 검증\n")
    add("독립 실험 32개를 모두 완료한 후에만 검증 유망 요인을 분석했다. 특징 표현(relative/spatial/personal/multiscale), 전처리·차원 축소(robust/select/PCA), 학습 분포(subject_weight/gain_aug), 모델 구조, 의사결정·후처리로 나누어 서로 다른 위치의 요인이 결합 가능한지 검증했다. 상대 표현과 공간 표현, 특징 선택과 PCA처럼 구현상 대체 관계인 요인은 동시 사용하지 않는다.\n")
    add("초기 후보는 검증 성공군과 B00 중 평균 검증 Macro F1로 선택하고, 최고점과 0.2%p 이내이면 측정 지연이 더 짧은 모델을 선호한다. 새 요인은 부모 대비 평균 검증 Macro F1 +0.2%p 이상, 최소 2분할 개선, 최저 클래스 평균 하락 제한 및 클래스 비열화 규칙을 만족할 때만 유지한다. 성능이 떨어지거나 기준 미달인 결합은 최종 구조에 추가하지 않는다.\n")
    add("| 단계 | 부모 | 추가 요소 | 평균 검증 Macro F1 (%) | 부모 대비 (%p) | 판정 |\n| --- | --- | --- | ---: | ---: | --- |")
    for step in lock.get("steps", []):
        if "id" not in step:
            add(f"| 생략 | {step.get('parent','—')} | {step.get('component','—')} | — | — | 구현상 비호환: {step.get('reason','')} |")
            continue
        identifier = step["id"]
        add(f"| {identifier} | {step.get('parent','B00')} | {step.get('component','최고 독립 후보에서 시작')} | {number(summaries[identifier]['macro_f1'])} | {number(step.get('delta', summaries[identifier]['macro_f1']-val_base['macro_f1']),True)} | {'시작' if step.get('step')=='start' else '유지' if step.get('accepted') else '제거/미채택'} |")
    add("\n앙상블 결합은 성공 모델 간 확률을 평균하여 오류의 상보성을 직접 확인한다. 모델 이름이 서로 다르다는 이유만으로 상보성을 가정하지 않는다. 검증된 결합의 성능 차이가 시너지의 근거이며 개별 가설 성공만으로 시너지를 주장하지 않는다.\n")

    add("## 7. 최종 모델의 전체 구조와 추론 파이프라인\n")
    add(f"선택 ID: **{final_id}**. 선택 고정 시각: `{lock['locked_at']}`. 선택 근거: `{lock['selection_basis']}`. 테스트 실행 시각: `{test_summary['evaluated_at']}`. 선택 lock SHA-256: `{lock_hash}`.\n")
    add(pipeline_mermaid(lock) + "\n")
    add("원시 센서 입력을 동일한 창 생성 절차로 3초 단위로 묶고, 체크포인트에 저장된 훈련 전처리·선택 모델로 7클래스 점수를 생성하여 argmax를 출력한다. 시간 후처리를 사용하는 경우 관찰 가능한 기록 경계·시간 간격에서만 상태를 초기화하고 미래 창이나 정답으로 상태를 초기화하지 않는다.\n")
    add("최종 구성 명세:\n\n```json\n" + json.dumps(lock["selected_spec"], ensure_ascii=False, indent=2) + "\n```\n")
    specs_by_id = {spec["id"]: spec for spec in lock["all_specs"]}
    graph_ids = []
    def visit_graph(identifier):
        if identifier in graph_ids: return
        spec = specs_by_id[identifier]
        if spec["family"] == "fusion":
            for member in spec["members"]: visit_graph(member)
        elif spec["family"] == "temporal": visit_graph(spec.get("parent", "B00"))
        graph_ids.append(identifier)
    visit_graph(final_id)
    add("| 노드 | 구조·전처리·주요 매개변수 | primary 수치 규모/학습 | 전체 구조·설정 기록 |\n| --- | --- | --- | --- |")
    for identifier in graph_ids:
        spec = specs_by_id[identifier]; run = summaries[identifier]["primary_run"]; comp = run["complexity"]
        description = architecture_name(spec)
        if spec["method"] == "aug_cnn":
            description = "훈련 z-score 32채널 → Conv(32,48,k5) → Conv(48,64,k3) → Conv(64,64,k3) → mean/max pool 128 → Dense64 → Dense7; GELU·dropout 0.15; gain/noise는 훈련 전용"
        if spec["method"] == "stacking":
            description = "각각 StandardScaler 적용: balanced RBF-SVM(C1, gamma.0005), balanced LR(C.1), ExtraTrees(250,min_leaf3) → 21확률 → StandardScaler+balanced LR(C.1); 훈련 3-fold 사용자 OOF"
        scale = parameter_count(comp)
        if "best_epoch" in comp: scale += f"; 선택 epoch {comp['best_epoch']}/{comp['epochs_trained']}"
        add(f"| {identifier} | {description} | {scale} | " + relative_link(output / identifier / "primary/validation.json", report, "구조·전체 hyperparameters·훈련 이력") + " |")
    add("\n전체 매개변수, 입력 차원, OOF 사용자 목록, scaler 설정, 각 신경망 레이어와 최적 epoch는 위 JSON에 빠짐없이 저장했다. 전체 최종 구성의 중첩 메타데이터는 " + relative_link(output / final_id / "primary/validation.json", report, "최종 validation.json") + "에서 확인할 수 있다.\n")
    add(relative_link(output / final_id / "primary/checkpoint.joblib", report, "최종 실험 체크포인트") + "; " + relative_link(output / "selection_lock.json", report, "선택 고정 및 전체 체크포인트 해시") + ".\n")

    add("## 8. 동일 테스트 데이터: Baseline 대 최종 모델\n")
    add("| 지표 | B00 | 최종 " + final_id + " | 변화 (%p) | 원본 활성 모델 | 최종−원본 (%p) |\n| --- | ---: | ---: | ---: | ---: | ---: |")
    for key, label in (("accuracy", "Accuracy (%)"), ("macro_f1", "Macro F1 (%)"), ("worst_class_f1", "최저 클래스 F1 (%)")):
        add(f"| {label} | {number(base[key])} | {number(final[key])} | {number(final[key]-base[key],True)} | {number(stored[key])} | {number(final[key]-stored[key],True)} |")
    add("\n| 클래스 | 창 수 | B00 Precision | 최종 Precision | B00 Recall | 최종 Recall | B00 F1 | 최종 F1 | ΔF1 (%p) |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for i, label in enumerate(manifest["class_names"]):
        add(f"| {label} | {base['support'][i]} | {number(base['precision'][i])} | {number(final['precision'][i])} | {number(base['recall'][i])} | {number(final['recall'][i])} | {number(base['f1'][i])} | {number(final['f1'][i])} | {number(final['f1'][i]-base['f1'][i],True)} |")
    add("\n원본 활성 모델까지 포함한 클래스별 지표:\n")
    add("| 클래스 | 원본 Precision | 원본 Recall | 원본 F1 | 최종−원본 F1 (%p) |\n| --- | ---: | ---: | ---: | ---: |")
    for i, label in enumerate(manifest["class_names"]):
        add(f"| {label} | {number(stored['precision'][i])} | {number(stored['recall'][i])} | {number(stored['f1'][i])} | {number(final['f1'][i]-stored['f1'][i],True)} |")
    add("\n![클래스별 F1 비교](" + os.path.relpath(charts["per_class"], report.parent) + ")\n")
    add("### 사용자별 결과와 불확실성\n")
    add("| 테스트 사용자 | B00 Accuracy | 최종 Accuracy | B00 Macro F1 | 최종 Macro F1 | ΔMacro F1 (%p) |\n| --- | ---: | ---: | ---: | ---: | ---: |")
    for subject in sorted(base["per_subject"]):
        b, f = base["per_subject"][subject], final["per_subject"][subject]
        add(f"| {subject} | {number(b['accuracy'])} | {number(f['accuracy'])} | {number(b['macro_f1'])} | {number(f['macro_f1'])} | {number(f['macro_f1']-b['macro_f1'],True)} |")
    add("\n| 비교 | Accuracy Δ 및 95% 구간 (%p) | Macro F1 Δ 및 95% 구간 (%p) |\n| --- | --- | --- |")
    for key, label in (("common_baseline", "최종 − B00"), ("stored_baseline", "최종 − 원본 활성")):
        fields = []
        for metric in ("accuracy", "macro_f1"):
            record = bootstrap[key][metric]
            fields.append(f"{number(record['observed_delta'],True)} [{number(record['ci95'][0],True)}, {number(record['ci95'][1],True)}]")
        add(f"| {label} | " + " | ".join(fields) + " |")
    add(f"\n동일 사용자 단위 paired cluster bootstrap {BOOTSTRAP_REPLICATES:,}회, seed={BOOTSTRAP_SEED}, percentile 95% 구간이다. 창을 독립 표본처럼 재추출하지 않고 사용자 전체 기록을 함께 재추출하여 pooled Accuracy·Macro F1을 다시 계산했다. **테스트 사용자가 {len(parts[0]['test'])}명뿐**이므로 구간이 불안정하고 새로운 사용자 모집단에서의 개선을 확정할 수 없다. 다수 가설 탐색과 과거 테스트 노출의 영향을 이 구간이 제거하지도 않는다. " + relative_link(output / "paired_subject_bootstrap.json", report, "재현 가능한 bootstrap 설정·결과") + ".\n")

    add("## 9. Confusion Matrix 비교\n")
    add("행은 실제 클래스, 열은 예측 클래스다. 셀에는 정수 창 수와 행 정규화 비율을 함께 표시한다.\n")
    add("![Baseline과 최종 Confusion Matrix](" + os.path.relpath(charts["confusion"], report.parent) + ")\n")
    for identifier in ("B00", final_id):
        add(f"**{identifier} 원시 혼동행렬**, 클래스 순서: " + ", ".join(manifest["class_names"]) + ".\n\n```text\n" + np.array2string(np.asarray(tests[identifier]["metrics"]["confusion_matrix"])) + "\n```\n")
    add("**원본 활성 체크포인트 원시 혼동행렬**:\n\n```text\n" + np.array2string(np.asarray(stored["confusion_matrix"])) + "\n```\n")

    add("## 10. Ablation Test: 추가 요소의 기여\n")
    add("순차 추가에서는 동일 부모와 추가 모델의 세 검증 분할을 비교했다. 최종 명시적 특징 조합은 구성요소를 하나씩 제거해 다시 학습한 결과를 아래에 기록한다. fusion/temporal 구조는 부모 모델과의 비교로 추가 노드의 기여를 확인한다. 제거 후 결과가 나쁘지 않거나 개선 폭이 규칙보다 작으면 복잡한 요소의 필요성이 입증된 것으로 보지 않는다.\n")
    for full in (True, False):
        add("### " + ("최종 전체 파이프라인에서 요소 제거" if full else "내부 분기에서 요소 제거: 각 부모와 비교") + "\n")
        add("| 실험 | 비교 부모 | 제거 요인 | 평균 검증 Macro F1 | 부모 대비 (%p) | 최종 대비 (%p) | 테스트 Macro F1 | 검증 해석 |\n| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
        for ablation in lock.get("ablations", []):
            identifier = ablation["spec"]["id"]; score = summaries[identifier]
            parent = ablation.get("parent", final_id)
            if (parent == final_id) != full: continue
            parent_delta = score["macro_f1"] - summaries[parent]["macro_f1"]
            delta = score["macro_f1"] - val_final["macro_f1"]
            add(f"| {identifier} | {parent} | {ablation['removed']} 제거 | {number(score['macro_f1'])} | {number(parent_delta,True)} | {number(delta,True)} | {number(tests[identifier]['metrics']['macro_f1'])} | {'제거 시 하락: 개발 자료에서 기여 지지' if parent_delta<0 else '제거 시 비열화: 필요성 미확인'} |")
    if not lock.get("ablations"):
        add("| — | — | 최종에 명시적 추가 특징 구성 없음 | — | — | — | — | 아래 부모 비교와 독립 B00 비교 적용 |")
    add("\n순차 추가에서의 부모 비교:\n")
    add("| 실험 | 비교 부모 | 추가 요소 | 평균 검증 Macro F1 | 부모 대비 (%p) | 최종 대비 (%p) | 테스트 Macro F1 | 검증 판정 |\n| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for step in lock.get("steps", []):
        if "parent" not in step or "id" not in step: continue
        identifier = step["id"]
        add(f"| {identifier} | {step['parent']} | {step['component']} 추가 | {number(summaries[identifier]['macro_f1'])} | {number(step['delta'],True)} | {number(summaries[identifier]['macro_f1']-val_final['macro_f1'],True)} | {number(tests[identifier]['metrics']['macro_f1'])} | {'채택' if step['accepted'] else '미채택'} |")
    add("\n![결합과 ablation 검증 결과](" + os.path.relpath(charts["ablation"], report.parent) + ")\n")

    add("## 11. 모델 복잡도·계산 비용·실시간 가능성\n")
    add("| ID | 3분할 훈련 합계 (s) | primary 체크포인트 (MiB) | 수치 규모 | 단일 창 중앙값 (ms) | p95 (ms) | batch ms/창 |\n| --- | ---: | ---: | --- | ---: | ---: | ---: |")
    for spec in lock["all_specs"]:
        identifier = spec["id"]; summary = summaries[identifier]; primary = summary["primary_run"]; inference = primary["inference"]
        add(f"| {identifier} | {summary['train_seconds']:.2f} | {summary['checkpoint_bytes']/2**20:.3f} | {parameter_count(primary['complexity'])} | {inference['single_window_ms_median']:.3f} | {inference['single_window_ms_p95']:.3f} | {inference['batch_ms_per_window']:.3f} |")
    primary = val_final["primary_run"]
    add(f"\n측정은 현재 CPU 환경에서 단일 창 예열 후 {primary['inference']['repetitions']}회 반복했다. **기존 596차원 특징 추출을 캐시한 상태의 분류기 및 모델별 변환 비용**이며 센서 직렬 입력·기존 특징 추출·UI·운영체제 지연을 포함한 end-to-end 지연은 아니다. 최종 중앙값 {primary['inference']['single_window_ms_median']:.3f} ms, p95 {primary['inference']['single_window_ms_p95']:.3f} ms는 3초 출력 주기보다 짧아 이 컴퓨터에서 분류 연산 자체는 실시간 실행 가능한 범위다. 실제 반응 시간에는 최소 3초 창 확보가 필요하며 hysteresis는 추가 확인 창으로 최대 약 3초가 더 필요할 수 있다. 하드웨어 통합 실시간성은 별도 검증해야 한다.\n")
    add("fusion/temporal의 훈련 시간은 이미 학습된 구성원을 불러와 조립한 시간이며 구성원 훈련 비용은 각 원래 실험 행에 있다. 체크포인트 크기는 압축 joblib 파일 크기다. 수치 매개변수·트리 노드·저장 벡터는 서로 다른 복잡도 단위이므로 동일 척도로 단순 비교하지 않는다.\n")
    runtime_files = [output / f"runtime_{identifier}.json" for identifier in ("B00", final_id)]
    if all(path.exists() for path in runtime_files):
        runtime = [load_json(path) for path in runtime_files]
        add("### 순차 스트리밍 통합 연산 측정\n")
        add("아래 추가 측정은 36×16 리샘플된 입력 창에서 **596 특징 추출 → 전체 구성원 분류 → 인과 시간 상태 → 확률 출력**까지 포함한다. 직렬 센서 수신·리샘플링·3초 창 수집·UI는 제외한다. 창을 순서대로 입력하여 batch 예측과의 일치도도 확인했다.\n")
        add("| 모델 | 창 수 | 중앙값 ms | p95 ms | p99 ms | 로딩 s | 프로세스 최대 RSS MiB | 스트림/batch argmax 일치 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for identifier, row in zip(("B00", final_id), runtime):
            add(f"| {identifier} | {row['input_windows']} | {row['median_ms']:.3f} | {row['p95_ms']:.3f} | {row['p99_ms']:.3f} | {row['load_seconds']:.3f} | {row['max_rss_bytes']/2**20:.1f} | {number(row['argmax_agreement'])}% |")
        add("\nRSS는 캐시 데이터·Python·sklearn·torch를 포함한 별도 프로세스의 최대 상주 메모리이며 모델만의 추가 RAM이 아니다. 실제 물리 센서의 실시간 검증 결과로 해석하지 않는다. " + "; ".join(relative_link(path, report, path.name) for path in runtime_files) + ".\n")
    add("실행 환경:\n\n```text\n" + manifest["environment"]["python"] + "\n" + manifest["environment"]["platform"] + "\n```\n전체 패키지 버전은 dataset.json의 environment.pip_freeze에 기록했다. 신경망은 CPU 2 threads이며 가속기·물리 센서 검증을 수행했다고 주장하지 않는다.\n")

    add("## 12. 증거의 연결, 해석과 재현\n")
    class_delta = np.asarray(final["f1"]) - np.asarray(base["f1"])
    improving = [manifest["class_names"][i] for i in np.flatnonzero(class_delta > 0)]
    declining = [manifest["class_names"][i] for i in np.flatnonzero(class_delta < 0)]
    add(f"증거의 순서는 **32개 독립 가설 → 검증 규칙 통과 {len(success)}개 → 검증 유망 요소의 순차 결합·제거 → {final_id} 후보 선택 고정 → 동일 테스트 비교 → 성공 확인 {len(confirmed_success)}개**이다. 평균 검증 Macro F1 변화는 {number(val_final['macro_f1']-val_base['macro_f1'],True)}%p, 최종 테스트 변화는 {number(final['macro_f1']-base['macro_f1'],True)}%p다. 테스트에서 F1이 오른 클래스는 {', '.join(improving) if improving else '없음'}, 내려간 클래스는 {', '.join(declining) if declining else '없음'}이다. 따라서 평균 개선과 클래스별 손실을 구분해야 한다.\n")
    if final["macro_f1"] <= base["macro_f1"]:
        add("**왜 더 좋아졌는가에 대한 긍정 결론은 내릴 수 없다.** 복합 구조는 개발 사용자에서 유효했던 잡음 증강·stacking·평균화의 조합이지만, 그 개선이 고정 테스트 사용자에게 이전되지 않았다. 적은 고유 사용자와 반복 활용한 개발 분할에 대한 선택 편향, 실제 사용자·센서 변화를 충분히 재현하지 못한 합성 증강, 구성원들이 공유한 체계적 오분류가 가능한 설명이다. 이는 추정이며 현재 실험으로 원인을 확정하지 않았다. 최종 복합 후보를 채택하지 않고 기존 생산 모델을 보존하는 판단은 성능·복잡도·불확실성을 함께 반영한다.\n")
    else:
        add(f"최종 선택을 설명하는 직접 근거는 검증의 일관성과 ablation의 실제 변화량이다. {success_reason(lock['selected_spec']['method'])} 이 메커니즘은 관측과 일치하는 해석이며 센서 보정·체형·시간 역학을 직접 조작한 인과 실험으로 입증한 것은 아니다.\n")
    add("복잡도를 늘린 추가 요소가 검증 규칙을 통과하지 못하면 제거했다. 최종 테스트의 높은 Accuracy만으로 다른 후보를 새로 선택하지 않았다.\n")
    add("주요 한계는 적은 고유 사용자 수, 서로 겹치는 개발 분할, 이미 과거에 본 historical holdout, 소급적 정자세 등록, 정적 자세별 기록, 센서 및 의자 동일성의 물리 검증 부재다. 현 단계에서의 권고 범위는 실험 후보 보존과 신규 사용자·세션에서의 전향적 검증이며 자동 생산 교체가 아니다.\n")
    add("### 재현 명령과 산출물\n")
    add("저장소 루트에서 기존 원본 데이터와 동일한 환경을 사용한다. 아래 명령은 분리된 results 폴더에만 체크포인트·예측·보고서를 생성한다.\n\n```bash\n.venv/bin/python -m experiments.posture_study.prepare\n.venv/bin/python -m experiments.posture_study.run train\n.venv/bin/python -m experiments.posture_study.run combine\n.venv/bin/python -m experiments.posture_study.run test\n.venv/bin/python -m experiments.posture_study.report\n.venv/bin/python -m unittest experiments.posture_study.test_neural experiments.posture_study.test_protocol -v\n```\n")
    add("`train`은 동일 spec·데이터 지문·체크포인트 해시가 일치하는 기존 결과만 재사용한다. 완전 재학습은 `run --output`에 새 빈 디렉터리를 지정하여 train/combine/test에 동일 경로를 사용하고 report에도 같은 `--output`을 전달한다. 학습 재현은 seed뿐 아니라 기록·sidecar 해시와 패키지 버전까지 일치시키는 것을 전제로 한다.\n")
    if (output / "study_tests.log").exists() and (output / "production_regression_tests.log").exists():
        add("검증된 소프트웨어 경계: 실험 계약·누수 방지·스트리밍 추론 관련 **32개 테스트**, 기존 생산 회귀 **202개 테스트**가 통과했다. 학습된 모델의 새로운 사용자 성능이나 물리 센서 성능을 보증하는 테스트는 아니다. " + relative_link(output / "study_tests.log", report, "실험 테스트 로그") + "; " + relative_link(output / "production_regression_tests.log", report, "생산 회귀 로그") + ".\n")
    if (output / "reproducibility_replay.json").exists():
        add("추가 재현 확인: 기록된 primary 설정으로 **B00와 수정 H29를 새로 재학습**한 뒤 기존 검증 확률과 비교했고 두 모델 모두 최대 절대 오차 **0.0**, argmax 완전 일치였다. 이 확인에는 테스트를 사용하지 않았다. 전체 모델의 다른 환경 재학습까지 증명한 것은 아니다. " + relative_link(output / "reproducibility_replay.json", report, "재학습 재현 결과") + "; " + relative_link(output / "reproducibility_replay.log", report, "재현 로그") + ".\n")
    if (output / "verification.json").exists():
        add("무결성 확인: 원본 CSV **434개**와 활성 체크포인트가 변경되지 않았고, **52개 모델의 동일 374 테스트 창**, **156개 체크포인트 해시**, 선택 고정 소스 해시를 확인했다. 생산 코드 변경·활성화·하드웨어 검증은 없다. " + relative_link(output / "verification.json", report, "검증 종합 기록") + "; " + relative_link(output / "reproducibility_index.json", report, "모든 재현 산출물 인덱스") + ".\n")
    add("- " + relative_link(output / "protocol.json", report, "가설·성공·선택 규칙") + "\n- " + relative_link(output / "dataset.json", report, "사용자/세션 그룹, 창, 특징, 등록 제외, 입력 해시, 실행 환경") + "\n- " + relative_link(output / "independent_analysis.json", report, "결합 전에 작성한 독립 실험 분석") + "\n- " + relative_link(output / "selection_lock.json", report, "최종 선택 및 모든 체크포인트 고정") + "\n- " + relative_link(output / "test_summary.json", report, "전체 테스트 결과") + "\n- 각 `<ID>/<split>/validation.json`: seed, 사용자 분할, config/구조/하이퍼파라미터, 학습 이력, 비용, 클래스별 지표·혼동행렬·checkpoint SHA-256.\n- 각 `<ID>/<split>/checkpoint.joblib`: 적합된 전처리·모델.\n- 각 `<ID>/<split>/validation_predictions.npz`, `<ID>/primary/test_predictions.npz`: 정렬된 정답·확률·사용자·기록·시각.\n")

    add("## 부록 A. 모든 실험의 분할별 결과·클래스 지표·체크포인트\n")
    for spec in lock["all_specs"]:
        identifier = spec["id"]; summary = summaries[identifier]; score = tests[identifier]["metrics"]
        add(f"### {identifier}: {spec['hypothesis']}\n")
        add(f"구현: {architecture_name(spec)}.\n")
        add("| 평가 분할 | Seed | Accuracy (%) | Macro F1 (%) | 최저 클래스 F1 (%) | 훈련 s | 상세 기록 |\n| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for run in summary["runs"]:
            split = run["split"]["id"]; metric = run["metrics"]
            add(f"| {split} validation | {run['split']['seed']} | {number(metric['accuracy'])} | {number(metric['macro_f1'])} | {number(metric['worst_class_f1'])} | {run['train_seconds']:.2f} | " + relative_link(output / identifier / split / "validation.json", report, "P/R/F1·CM·설정") + " |")
        add(f"| primary test | {parts[0]['seed']} | {number(score['accuracy'])} | {number(score['macro_f1'])} | {number(score['worst_class_f1'])} | — | " + relative_link(output / identifier / "primary/test.json", report, "전체 test 지표·CM") + " |")
        add("\n| 테스트 클래스 | Precision (%) | Recall (%) | F1 (%) | 창 수 |\n| --- | ---: | ---: | ---: | ---: |")
        for i, label in enumerate(manifest["class_names"]):
            add(f"| {label} | {number(score['precision'][i])} | {number(score['recall'][i])} | {number(score['f1'][i])} | {score['support'][i]} |")
        add("\n체크포인트: " + ", ".join(relative_link(output / identifier / run["split"]["id"] / "checkpoint.joblib", report, run["split"]["id"]) for run in summary["runs"]) + ".\n")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(sections) + "\n", encoding="utf-8")
    write_json(output / "report_manifest.json", {
        "report": str(report), "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "selection_lock_sha256": lock_hash, "selected_id": final_id,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "charts": {key: str(path) for key, path in charts.items()},
        "report_generator": "experiments/posture_study/report.py", "test_changes_selection": False,
    })
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    print(generate(args.output, args.report))


if __name__ == "__main__":
    main()
