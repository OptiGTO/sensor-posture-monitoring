"""Produce the Korean, auditable report after the model-selection lock and test.

This module never fits a model and never changes the selected model. Training
metrics are descriptive predictions from frozen checkpoints, collected only
after test completion. All reported evaluation metrics use probability argmax.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/private/tmp")) / "posture-search-matplotlib"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from .catalog import CATALOG
from .data import OUT, ROOT, digest, take, write_json
from .run import metrics, summarize


BOOTSTRAP_SEED = 20260913
BOOTSTRAP_DRAWS = 5000
METHODS = {
    "baseline": "StandardScaler + RBF-SVM(C=1, γ=0.0005, balanced, probability=True)",
    "robust_svm": "RobustScaler(중앙값/IQR 25–75%) + 동일 RBF-SVM",
    "feature_select": "훈련 ANOVA SelectKBest(최대 96개) → StandardScaler → 동일 SVM",
    "pca_svm": "StandardScaler → 훈련 PCA(분산 95%, full SVD) → 동일 SVM",
    "subject_balanced_svm": "피험자별 n/(사용자 수 × 사용자 창 수) 가중치를 scaler와 SVM에 적용",
    "hierarchical_correct": "SVM 정자세/오자세 게이트 × 조건부 6종 SVM 확률",
    "hierarchical_direction": "SVM 게이트 [0,1,2]/[3,4]/[5,6] × 각 분기 SVM 확률",
    "extra_trees": "ExtraTrees 300개, leaf≥2, sqrt 특징, balanced",
    "random_forest": "RandomForest 300개, leaf≥2, sqrt 특징, balanced_subsample",
    "histogram_boosting": "HistGradientBoosting 120회, 15 leaves, lr=.06, L2=1; 조기 종료 없음",
    "shrinkage_lda": "StandardScaler + LDA(lsqr, 자동 공분산 수축)",
    "knn": "StandardScaler + 거리 가중 15-NN",
    "logistic": "StandardScaler + balanced LogisticRegression(C=1, max_iter=2500)",
    "group_bagging": "사용자 단위 복원추출 7회 + 각 SVM 확률 평균",
    "hard_vote": "SVM/ExtraTrees/Logistic 3개 다수결; 확률 평균으로 동률 해소",
    "soft_vote": "SVM/ExtraTrees/Logistic 확률을 1/3씩 평균",
    "weighted_vote": "SVM/ExtraTrees/Logistic 확률을 사전 고정 .5/.3/.2로 평균",
    "stacking": "3-fold GroupKFold 사용자 OOF의 21개 확률 → balanced Logistic 메타 모델",
    "cnn1d": "시간 Conv1D 48→32(k5)→48(k3)→48(k3), mean/max pooling→64→7",
    "cnn2d": "프레임별 3×4×4 → Conv2D 3→16→24(k3)→64, 시간 평균→7",
    "lstm": "1층 LSTM(input48, hidden48), 시간 평균+마지막 상태→64→7",
    "gru": "1층 GRU(input48, hidden48), 시간 평균+마지막 상태→64→7",
    "cnn_lstm": "Conv1D 48→32(k5) → LSTM(hidden48) → 평균+마지막 상태→64→7",
    "attention": "48→64 projection, 4-head Self-Attention + 잔차/LayerNorm, mean/max→64→7",
    "transformer": "학습 위치 벡터, d_model48, 4 heads, FFN96, 2 encoder layers→평균→64→7",
    "tcn": "팽창률 1/2/4의 병렬 잔차 Conv1D(24 channels) → mean/max→64→7",
    "hybrid": "CNN1D 지도학습 64차원 임베딩 → StandardScaler → RBF-SVM(C=3, γ=scale)",
    "ema": "현재 확률 .6 + 과거 상태 .4; 첫 창/파일 경계/시간 불연속에서 초기화",
    "vote": "현재 포함 과거 최대 3창 예측의 인과적 다수결",
    "confidence": "현재 최대 확률 ≥.6일 때 상태 갱신, 그 외 이전 상태 유지",
}
FEATURES = {
    "baseline": "596개 절대+프레임 상대 창 통계",
    "relative": "상대 통계 298개만 사용",
    "spatial": "총압 비율 평균/표준편차, 행·열 합, 압력 중심·2차 모멘트·엔트로피 46개",
    "personal": "기존 596개 + 정자세 기준 비율16·비율 차16·로그 센서비16·총압비1 = 645개",
    "personal_spatial": "공간 특징46 + 개인 기준49 = 95개",
    "multiscale": "전체 3초 596개 + 최근 1초 공간46 + 최근 2초 공간46 = 688개",
    "log": "비음수 입력 log1p 후 596개 통계 재계산",
    "proportion": "각 프레임 총압 비율에 기초한 298개 통계",
    "denoise": "3프레임 시간 중앙값 필터 후 596개 통계 재계산",
    "raw": "36×16 원시 창 → 원값/1023, log1p/log1024, 프레임 비율의 48채널",
    "mixed": "구성 모델별 특징 표현",
    "inherited": "부모 모델의 특징 표현",
}
FAILURE_REASONS = {
    "relative": "절대 하중 정보를 제거하면서 구분에 필요한 접촉 강도도 함께 사라졌을 가능성",
    "spatial": "압축된 공간 요약이 개별 센서·시간 통계의 유용한 정보를 버렸을 가능성",
    "personal": "등록 대체 기록과 평가 착석의 위치·날짜가 달라 기준이 일관되지 않거나, 추가 특징이 적은 사용자 수에 과적합했을 가능성",
    "multiscale": "정적 자세 기록에서 추가 시간 규모의 정보가 중복되고 특징 수 증가가 잡음을 늘렸을 가능성",
    "log": "압축이 이상치를 줄이는 동시에 자세 간 압력 크기 차이도 약화했을 가능성",
    "proportion": "총압 불변성이 실제 클래스 구분에 필요한 하중 정보를 제거했을 가능성",
    "denoise": "이미 3초 통계가 잡음을 완화하여 추가 필터 효과가 작거나 유용한 변화를 제거했을 가능성",
    "robust_svm": "IQR 기반 척도가 고정 γ와 조합될 때 유리한 거리 구조를 만들지 못했을 가능성",
    "feature_select": "단변량 ANOVA가 센서 간 상호작용 및 새 사용자에게 유용한 특징을 누락했을 가능성",
    "pca_svm": "분산 보존이 클래스 구분력 보존과 같지 않아 낮은 분산의 유용한 방향을 제거했을 가능성",
    "subject_balanced_svm": "관측 수 보정만으로 사용자별 압력 분포 차이를 해결하기 어렵거나 소량 사용자의 잡음 비중이 증가했을 가능성",
    "hierarchical_correct": "상위 정자세/오자세 게이트의 오차가 하위 분류 확률에도 전파됐을 가능성",
    "hierarchical_direction": "사전 정의한 자세 계층이 센서에서 실제 분리되는 경계와 맞지 않았을 가능성",
    "extra_trees": "훈련 사용자에서 배운 축 정렬 임계값이 새 사용자의 압력 분포에 맞지 않았을 가능성",
    "random_forest": "사용자 변화에 따른 압력 이동과 트리 분할의 비연속성이 일반화를 제한했을 가능성",
    "histogram_boosting": "단계별 잔차 적합이 작은 사용자 집합의 특이 패턴까지 학습했을 가능성",
    "shrinkage_lda": "선형 경계와 공분산 가정이 자세·체형 상호작용을 충분히 설명하지 못했을 가능성",
    "knn": "가까운 훈련 창이 같은 자세보다 비슷한 체형·하중을 나타내 이웃 전이가 어려웠을 가능성",
    "group_bagging": "사용자 부트스트랩이 각 학습기의 서로 다른 사용자 다양성을 줄여 편향을 늘렸을 가능성",
    "hard_vote": "약한 구성 모델의 오답이나 상관된 오류가 강한 모델의 정답을 뒤집었을 가능성",
    "soft_vote": "구성 모델의 확률 척도 차이와 상관 오류로 평균이 유리하지 않았을 가능성",
    "weighted_vote": "사전 고정 가중치가 실제 클래스별 신뢰도를 반영하지 못했을 가능성",
    "stacking": "적은 사용자 OOF 사례로 메타 모델의 사용자 일반화·확률 척도 보정이 어려웠을 가능성",
    "ema": "기존 오답의 지속이나 확률 지연이 단발성 잡음 감소 효과를 상쇄했을 가능성",
    "vote": "연속 창의 오류가 상관되어 다수결이 같은 오답을 유지했을 가능성",
    "confidence": "SVC 확률 .6이 사용자별 정답 신뢰도를 보장하지 않아 오답 상태가 유지됐을 가능성",
}


def read(path):
    return json.loads(Path(path).read_text())


def pct(value):
    return f"{100 * value:.2f}%"


def pp(value):
    return f"{100 * value:+.2f} pp"


def table(headers, rows):
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(clean, row)) + " |" for row in rows])


def method(spec):
    if spec["family"] == "fusion":
        return "확률 결합 " + ", ".join(spec["members"]) + "; weights=" + str(spec.get("weights", "동등"))
    parent = f"; 부모 {spec.get('parent', 'B00')}" if spec["family"] == "temporal" else ""
    members = "; 절제 후 구성=" + ",".join(spec["ensemble_members"]) if spec.get("ensemble_members") else ""
    return METHODS.get(spec["kind"], spec["kind"]) + parent + members + "; " + FEATURES.get(spec["features"], spec["features"])


def paired_subject_bootstrap(y, baseline, final, groups):
    """Resample whole subjects jointly, retaining each subject's correlated rows."""
    subjects = np.unique(groups)
    baseline_cm = np.asarray([metrics(y[groups == s], baseline[groups == s])["confusion_matrix"] for s in subjects])
    final_cm = np.asarray([metrics(y[groups == s], final[groups == s])["confusion_matrix"] for s in subjects])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(subjects), size=(BOOTSTRAP_DRAWS, len(subjects)))

    def scores(cms):
        sampled = cms[draws].sum(axis=1)
        tp = np.diagonal(sampled, axis1=1, axis2=2)
        denominator = sampled.sum(axis=1) + sampled.sum(axis=2)
        f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0).mean(axis=1)
        accuracy = tp.sum(axis=1) / sampled.sum(axis=(1, 2))
        return {"accuracy": accuracy, "macro_f1": f1}

    first, second = scores(baseline_cm), scores(final_cm)
    results = {}
    for name in first:
        delta = second[name] - first[name]
        results[name] = {"mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
                         "percentile_95_interval": np.percentile(delta, [2.5, 97.5]).tolist(),
                         "fraction_positive": float(np.mean(delta > 0)),
                         "point_delta": metrics(y, final)[name] - metrics(y, baseline)[name]}
    return {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "unit": "subject", "paired": True,
            "subjects": subjects.tolist(), "subject_count": len(subjects), "metrics": results,
            "interpretation": "Percentile intervals from resampling observed subjects; few subjects, correlated windows, no new-cohort guarantee"}


def model_counts(model):
    """Useful structural sizes, without conflating neural parameters and tree nodes."""
    result = {"svm_support_vectors": 0, "tree_count": 0, "tree_nodes": 0, "neural_trainable_parameters": 0,
              "learned_array_elements": 0, "knn_training_rows": 0, "leaf_estimators": 0}
    seen = set()

    def walk(obj):
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, (list, tuple)):
            for part in obj:
                walk(part)
            return
        if hasattr(obj, "model_") and hasattr(obj.model_, "parameters"):
            result["neural_trainable_parameters"] += sum(p.numel() for p in obj.model_.parameters() if p.requires_grad)
        if hasattr(obj, "support_vectors_"):
            result["svm_support_vectors"] += len(obj.support_vectors_)
        if hasattr(obj, "tree_"):
            result["tree_count"] += 1
            result["tree_nodes"] += obj.tree_.node_count
        if hasattr(obj, "_fit_X"):
            result["knn_training_rows"] += len(obj._fit_X)
        for name in ("support_vectors_", "dual_coef_", "coef_", "intercept_", "components_", "mean_", "scale_",
                     "center_", "means_", "priors_", "covariance_", "scalings_", "var_"):
            value = getattr(obj, name, None)
            if isinstance(value, np.ndarray):
                result["learned_array_elements"] += value.size
        if hasattr(obj, "steps"):
            for _, step in obj.steps:
                walk(step)
        for name in ("model", "parent", "members", "estimator_", "models_", "gate_", "branch_models_", "meta_", "svm_", "embedding_scaler_"):
            walk(getattr(obj, name, None))
        estimators = getattr(obj, "estimators_", None)
        if estimators is not None:
            walk(list(np.asarray(estimators, dtype=object).ravel()))
        if hasattr(obj, "predict_proba") and not any(hasattr(obj, name) for name in ("model", "members", "parent", "estimator_", "models_", "steps")):
            result["leaf_estimators"] += 1
    walk(model)
    return result


def collect(output, manifest, lock, test_summary):
    fingerprint = manifest["fingerprint"]
    lock_sha = digest(output / "selection_lock.json")
    cache_path = output / "report_metrics.json"
    if cache_path.exists():
        previous = read(cache_path)
        if previous.get("selection_lock_sha256") == lock_sha and previous.get("report_source_sha256") == digest(__file__):
            return previous
    data = dict(np.load(output / "data.npz", allow_pickle=False))
    assert digest(output / "data.npz") == manifest["cache_sha256"]
    results = {}
    test_by_id = {r["spec"]["id"]: r for r in test_summary["results"]}
    for spec in lock["all_specs"]:
        identifier = spec["id"]
        record = {"spec": spec, "summary": summarize(spec, output), "splits": {}, "test": test_by_id[identifier]["metrics"]}
        assert record["summary"] is not None, identifier
        for partition in manifest["partitions"]:
            split = partition["id"]
            folder = output / identifier / split
            validation = read(folder / "validation.json")
            assert validation["dataset_fingerprint"] == fingerprint
            assert digest(folder / "model.joblib") == lock["checkpoints"][str((folder / "model.joblib").relative_to(output))]
            model = joblib.load(folder / "model.joblib")
            train = take(data, np.isin(data["groups"], partition["train"]))
            p = model.predict_proba(train)
            record["splits"][split] = {"train": metrics(train["y"], p.argmax(1), train["groups"]), "validation": validation["metrics"],
                                       "train_seconds": validation["train_seconds"], "inference": validation["inference"],
                                       "checkpoint_bytes": validation["checkpoint_bytes"], "model_counts": model_counts(model),
                                       "checkpoint_sha256": validation["checkpoint_sha256"], "complexity": validation["complexity"]}
        test_row = read(output / identifier / "primary/test.json")
        assert test_row["lock_sha256"] == lock_sha
        assert test_row["model_sha256"] == record["splits"]["primary"]["checkpoint_sha256"]
        results[identifier] = record
        print(f"REPORT: frozen training/validation/test metrics {identifier}", flush=True)
    report = {"version": "1.0.0", "dataset_fingerprint": fingerprint, "selection_lock_sha256": lock_sha,
              "report_source_sha256": digest(__file__), "selected_id": lock["selected_id"], "experiments": results,
              "training_metrics_note": "In-sample descriptive metrics computed after test; no fitting or selection"}
    write_json(cache_path, report)
    return report


def plots(output, report, lock, classes):
    charts = output / "charts"
    charts.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False, "figure.dpi": 130, "savefig.dpi": 180})
    experiments = report["experiments"]
    selected = lock["selected_id"]
    base = experiments["B00"]
    final = experiments[selected]
    names = list(experiments)
    success = set(lock["successful_ids"])
    fig, ax = plt.subplots(figsize=(10, max(8, .26 * len(names))))
    values = [experiments[i]["summary"]["macro_f1"] * 100 for i in names]
    errors = [experiments[i]["summary"]["f1_std"] * 100 for i in names]
    colors = ["#203864" if i == selected else "#2e8b57" if i in success else "#acb6c2" for i in names]
    ax.barh(names, values, xerr=errors, color=colors, error_kw={"elinewidth": .7, "capsize": 2})
    ax.axvline(base["summary"]["macro_f1"] * 100, color="#bc3c29", linestyle="--", label="Baseline 평균")
    ax.invert_yaxis(); ax.set(xlabel="검증 Macro F1 (%) — 오차막대: 3개 분할 표준편차", title="사전 정책에 따른 검증 실험 비교")
    ax.legend(); fig.tight_layout(); fig.savefig(charts / "validation_macro_f1.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    first = np.asarray(base["test"]["confusion_matrix"])
    second = np.asarray(final["test"]["confusion_matrix"])
    maximum = max(first.max(), second.max())
    short = ["정자세", "앞 숙임", "뒤 기대기", "왼 기울임", "오른 기울임", "왼 다리", "오른 다리"]
    for ax, cm, title, difference in zip(axes, [first, second, second - first], ["Baseline B00", f"선택 모델 {selected}", "선택 모델 - Baseline (개수)"], [False, False, True]):
        if difference:
            limit = max(1, np.abs(cm).max()); plot = ax.imshow(cm, cmap="RdBu_r", vmin=-limit, vmax=limit)
        else:
            plot = ax.imshow(cm, cmap="Blues", vmin=0, vmax=maximum)
        for r in range(7):
            for c in range(7):
                ax.text(c, r, f"{cm[r, c]:+d}" if difference else str(cm[r, c]), ha="center", va="center", fontsize=8,
                        color="white" if abs(cm[r, c]) > (np.abs(cm).max() * .65) else "black")
        ax.set(xticks=np.arange(7), yticks=np.arange(7), xticklabels=short, yticklabels=short,
               xlabel="예측", ylabel="실제", title=title)
        ax.tick_params(axis="x", labelrotation=65); fig.colorbar(plot, ax=ax, fraction=.046, pad=.04)
    fig.suptitle("동일 테스트 창의 혼동행렬 — 차이는 개선 방향을 자동 의미하지 않음", fontsize=12)
    fig.tight_layout(); fig.savefig(charts / "confusion_comparison.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric_name, title in zip(axes, ["precision", "recall", "f1"], ["Precision", "Recall", "F1"]):
        x = np.arange(7); ax.bar(x - .18, np.asarray(base["test"][metric_name]) * 100, .36, label="Baseline", color="#acb6c2")
        ax.bar(x + .18, np.asarray(final["test"][metric_name]) * 100, .36, label=selected, color="#203864")
        ax.set(xticks=x, xticklabels=short, ylim=(0, 105), ylabel="%", title=title)
        ax.tick_params(axis="x", labelrotation=65)
    axes[0].legend(); fig.suptitle("동일 테스트 데이터의 클래스별 성능"); fig.tight_layout()
    fig.savefig(charts / "per_class_comparison.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = ["accuracy", "macro_f1", "worst_class_f1"]
    x = np.arange(3)
    baseline_bars = ax.bar(x - .18, [100 * base["test"][k] for k in keys], .36, label="Baseline B00", color="#acb6c2")
    for bar in baseline_bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.2f}%", ha="center", fontsize=9)
    bars = ax.bar(x + .18, [100 * final["test"][k] for k in keys], .36, label=f"선택 모델 {selected}", color="#203864")
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.2f}%\n({pp(final['test'][keys[i]] - base['test'][keys[i]])})", ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=["Accuracy", "Macro F1", "최저 클래스 F1"], ylim=(0, 108), ylabel="%", title="테스트를 열기 전에 선택한 모델의 최종 비교")
    ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig(charts / "final_test_comparison.png"); plt.close(fig)


def export_tsv(output, report):
    experiments = report["experiments"]; base = experiments["B00"]
    rows = []
    for identifier, result in experiments.items():
        summary = result["summary"]
        row = {"experiment_id": identifier, "hypothesis": result["spec"]["hypothesis"], "architecture_method": method(result["spec"]),
               "features": result["spec"]["features"], "validation_accuracy_mean": summary["accuracy"],
               "validation_macro_f1_mean": summary["macro_f1"], "validation_macro_f1_std": summary["f1_std"],
               "validation_macro_f1_delta": summary["macro_f1"] - base["summary"]["macro_f1"],
               "test_accuracy": result["test"]["accuracy"], "test_macro_f1": result["test"]["macro_f1"],
               "test_macro_f1_delta": result["test"]["macro_f1"] - base["test"]["macro_f1"],
               "train_seconds_total": summary["train_seconds"], "primary_raw_window_ms": summary["latency"],
               "primary_checkpoint_bytes": summary["bytes"], "metrics_reference": f"report_metrics.json#experiments/{identifier}"}
        for split, data in result["splits"].items():
            row[f"{split}_train_macro_f1"] = data["train"]["macro_f1"]
            row[f"{split}_validation_accuracy"] = data["validation"]["accuracy"]
            row[f"{split}_validation_macro_f1"] = data["validation"]["macro_f1"]
            row[f"{split}_validation_macro_f1_delta"] = data["validation"]["macro_f1"] - base["splits"][split]["validation"]["macro_f1"]
        rows.append(row)
    with (output / "all_experiments.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def mermaid(lock):
    specs = {s["id"]: s for s in lock["all_specs"]}
    lines = ["flowchart TD", '  INPUT["센서 16개 · 12 Hz · 3초 36프레임"]', '  OUTPUT["7개 자세 점수 argmax"]']
    visited = set()

    def branch(identifier):
        if identifier in visited:
            return identifier
        visited.add(identifier); spec = specs[identifier]
        if spec["family"] == "fusion":
            lines.append(f'  {identifier}["{identifier}: 구성 모델 확률 평균"]')
            for member in spec["members"]:
                lines.append(f"  {branch(member)} --> {identifier}")
        elif spec["family"] == "temporal":
            lines.append(f'  {identifier}["{identifier}: 인과적 {spec["kind"]} 후처리"]')
            lines.append(f"  {branch(spec.get('parent', 'B00'))} --> {identifier}")
        else:
            label = f'{identifier}: {spec["features"]} 특징 + {spec["kind"]}'
            lines.append(f'  {identifier}["{label}"]'); lines.append(f"  INPUT --> {identifier}")
            if "personal" in spec["features"]:
                lines.append(f'  REG["별도 정자세 등록 기록 첫 10초 · 학습 및 평가에서 전체 파일 제외"] --> {identifier}')
        return identifier
    final = branch(lock["selected_id"]); lines.append(f"  {final} --> OUTPUT")
    return "```mermaid\n" + "\n".join(lines) + "\n```"


def failure_reason(spec):
    if spec["family"] == "deep":
        return "학습 사용자 수에 비해 표현 학습의 자유도가 크고, 정적 3초 기록의 순서 정보가 제한적이어서 고정 45 epoch 학습이 새 사용자에게 전이되지 못했을 가능성"
    if spec["family"] == "fusion":
        return "두 구성 모델의 오류가 충분히 독립적이지 않거나 점수 척도가 달라, 단순 평균의 보완 효과가 작았을 가능성"
    return FAILURE_REASONS.get(spec["features"], FAILURE_REASONS.get(spec["kind"], "새 구성의 편향·분산과 사용자 간 분포 차이가 맞지 않았을 가능성"))


def write_report(output, manifest, lock, tests, report, bootstrap):
    experiments = report["experiments"]; base = experiments["B00"]; final = experiments[lock["selected_id"]]
    selected = lock["selected_id"]; robust = set(lock["successful_ids"])
    individual_ids = [s["id"] for s in CATALOG if s["id"] != "B00"]
    numerical = [i for i in individual_ids if experiments[i]["summary"]["macro_f1"] > base["summary"]["macro_f1"]]
    lines = []
    add = lines.append
    delta = final["test"]["macro_f1"] - base["test"]["macro_f1"]
    add("# 자세 분류 모델 성능 개선 실험 보고서\n")
    if delta <= 0:
        add("**이번 실험에서는 안정적인 최종 성능 개선을 입증하지 못했다.** 검증에서 선택한 후보의 테스트 Macro F1이 Baseline보다 낮았으므로, 기존 활성 구현을 유지하고 후보는 실험 결과로만 보존한다.")
    add(f"최종 모델은 **{selected}**이며 테스트 공개 전에 검증 결과만으로 고정했다. 동일한 테스트 {tests['test_windows']}개 창에서 Baseline Macro F1 **{pct(base['test']['macro_f1'])} → {pct(final['test']['macro_f1'])} ({pp(delta)})**, Accuracy **{pct(base['test']['accuracy'])} → {pct(final['test']['accuracy'])} ({pp(final['test']['accuracy'] - base['test']['accuracy'])})**였다. 검증 평균 Macro F1 차이는 **{pp(final['summary']['macro_f1'] - base['summary']['macro_f1'])}**다.")
    if delta <= 0:
        add("검증에서 선택한 모델의 테스트 Macro F1 개선은 확인되지 않았다. 이 보고서는 테스트 순위를 이용해 모델을 다시 고르지 않으며, 성능 향상이 입증됐다고 결론내리지 않는다.")
    add(f"독립 가설 **{len(individual_ids)}개**, Baseline 1개, 추가 조합·절제 실험 {len(experiments) - len(CATALOG)}개를 각각 동일한 3개 사용자 분할에서 평가했다. 데이터와 기존 활성 모델은 보존했으며 결과는 실험 후보이고 운영 모델 교체나 센서 실물 검증을 뜻하지 않는다.\n")

    add("## 1. Baseline와 공통 실험 조건\n")
    add(f"대조군 B00은 현재 활성 3초 RBF-SVM의 StandardScaler, C=1, γ=.0005, balanced class weight, 확률 argmax 규칙을 같은 훈련 데이터로 재학습했다. 모델의 native `predict()`와 확률 argmax는 다를 수 있으므로 **모든 실험의 평가는 확률/점수 argmax로 통일**했다. 원래 저장된 모델과 훈련 표본이 달라지는 영향을 통제하기 위해 B00을 주 비교 대상으로 사용한다.")
    add("등록 기준용 정자세 기록은 사용자별 경로순 첫 기록의 첫 10초 평균으로 정의했고, 해당 **전체 기록을 모든 방법의 학습·검증·테스트에서 제외**했다. 개인화 모델만 이 독립 기록에서 얻은 기준값을 입력으로 사용한다. 이는 사용자 기준 등록을 사후 재현한 대체 실험이며, 실제 선행 등록 여부와 의자·세션 일치 여부는 확인되지 않았다.")
    add("창 길이/stride는 3초/3초, 12 Hz 재표본화로 36×16 입력, 마지막 불완전 창 허용 기준은 95%다. 여러 시간 규모 모델도 같은 3초 테스트 창의 최근 1초·2초만 사용한다. 아래 3개 주 시행은 훈련 시드와 사용자 분할을 함께 바꾸므로 이 세 값만으로 순수한 시드 효과와 분할 효과를 분리하지 못한다. 별도의 primary 고정 분할 시드 점검을 아래에 제시한다. 테스트 사용자는 모든 시행에서 고정되며 **테스트는 primary 체크포인트만 평가**했다.")
    rows = []
    for p in manifest["partitions"]:
        rows.append([p["id"], p["seed"], ", ".join(p["train"]), ", ".join(p["validation"]), ", ".join(p["test"]),
                     "/".join(str(p["counts"][key]["windows"]) for key in ("train", "validation", "test"))])
    add(table(["분할", "시드", "훈련 사용자", "검증 사용자", "테스트 사용자", "창 수 train/val/test"], rows))
    add("\n같은 NFC 정규화 사용자에 속한 모든 기록을 한 분할에 묶었다. 따라서 같은 사용자의 측정 기록/세션이 훈련과 테스트에 교차하지 않는다. 파일 숫자 접미사는 반복 측정이며 검증된 세션 ID로 취급하지 않았다. 원본·센서열 중복 검사를 수행했고, 변환·스케일링·선택·PCA는 훈련에서만 적합했다. 스태킹은 사용자 분리 OOF를 사용한다. SVC 내부 확률 보정은 훈련 데이터 내부 libsvm 5-fold이며 사용자 분리 보정은 아니라는 한계가 있다.")
    add("훈련에만 전처리를 적합하는 원칙과 그룹별 분리 API는 공식 문서에 따라 확인했다. [scikit-learn 데이터 누출 방지](https://scikit-learn.org/1.8/common_pitfalls.html), [GroupKFold](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.GroupKFold.html).")
    add("신경망은 모두 처음부터 CPU 1 thread, 45 epochs, batch64, AdamW(lr=.001, weight_decay=.01), dropout=.15, 클래스 역빈도 가중 CrossEntropy, gradient clipping=2로 학습했다. 정규화 평균/표준편차는 훈련 데이터에서만 계산했으며 조기 종료·테스트 기반 튜닝은 하지 않았다. 신경망 공통 입력은 raw·log·비율 48채널이므로 기존 SVM 대비 신경망 결과는 입력 표현과 학습기 변화가 합쳐진 비교다. 2D-CNN의 4×4 인접성은 배선의 물리적 검증이 없는 모델링 가정이다.")
    add("합성곱·Transformer는 PyTorch의 기본 레이어를 조합해 구현했다. [Conv1d API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html), [TransformerEncoderLayer API](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html).")
    add(f"\n데이터 fingerprint: `{manifest['fingerprint']}`. [분할·원본 SHA256·특징·등록 근거·환경](dataset.json), [사전 가설 및 선택 정책](protocol.json), [테스트 전 선택 잠금](selection_lock.json).")
    stored = tests.get("stored_baseline")
    if stored:
        add(f"원본 저장 모델의 같은 테스트 창 보조 점수는 Accuracy {pct(stored['metrics']['accuracy'])}, Macro F1 {pct(stored['metrics']['macro_f1'])}다. 이 모델은 B00과 훈련 코호트·등록 기록 포함 조건이 다르므로 최종 모델의 구성 효과를 분리하는 대조군으로 사용할 수 없다. [원본 모델 보조 비교](test_summary.json)")

    add("\n## 2. 가설 정의, 이유와 실제 구현\n")
    add("각 행은 별도 체크포인트로 학습·평가한 독립 실험이다. 가설 이유는 실험 전 기대이며 결과가 그 원인을 직접 증명하지는 않는다. 상세 JSON에는 모델 버전, 시드, 전처리, 특징 수, 하이퍼파라미터, 신경망 레이어 및 학습 이력이 포함된다.\n")
    add(table(["ID", "개선 가설과 기대 이유", "구현", "설정·검증 기록"],
              [[s["id"], s["hypothesis"], method(s), f"[config]({s['id']}/primary/validation.json)"] for s in CATALOG]))

    add("\n## 3. 개별 실험 결과\n")
    add("아래 Accuracy/Macro F1은 **3개 검증 분할의 비가중 평균**이며 Change는 Baseline 대비 Macro F1 퍼센트포인트다. 비용은 primary 모델에서 원시 36×16 창을 입력받아 특징을 계산하고 추론하는 CPU 중앙값이다. 파일 크기는 압축 joblib 크기이며 훈련 시간은 세 시행 합계다.\n")

    def decision(identifier):
        if identifier == "B00": return "대조군" if selected != "B00" else "최종 유지"
        if identifier == selected: return "최종 선택"
        if identifier in robust: return "검증 안정성 기준 통과"
        if identifier in numerical: return "수치 개선; 채택 기준 미달"
        return "개선 미확인"

    add(table(["ID", "Hypothesis", "Architecture / Method", "Accuracy", "Macro F1", "Change vs. Baseline", "Inference Cost", "Decision"],
              [[s["id"], s["hypothesis"], s["kind"] + "/" + s["features"], pct(experiments[s["id"]]["summary"]["accuracy"]),
                pct(experiments[s["id"]]["summary"]["macro_f1"]), pp(experiments[s["id"]]["summary"]["macro_f1"] - base["summary"]["macro_f1"]),
                f"{experiments[s['id']]['summary']['latency']:.2f} ms; {experiments[s['id']]['summary']['bytes']/1024**2:.2f} MiB", decision(s["id"])] for s in CATALOG]))
    add("\n![검증 Macro F1과 분할 표준편차](charts/validation_macro_f1.png)")
    add("\n모든 실험의 클래스별 Precision/Recall/F1, support, 혼동행렬, 사용자별 성능 및 훈련 결과는 [report_metrics.json](report_metrics.json)에 있다. 원본 검증/테스트 JSON과 확률 배열은 각 실험 디렉터리에 보존한다. [전체 표 TSV](all_experiments.tsv).\n")
    split_rows = []
    for identifier, row in experiments.items():
        entries = []
        for split, result in row["splits"].items():
            entries.append(f"{pct(result['validation']['accuracy'])}/{pct(result['validation']['macro_f1'])} ({pp(result['validation']['macro_f1']-base['splits'][split]['validation']['macro_f1'])})")
        train_mean = np.mean([p["train"]["macro_f1"] for p in row["splits"].values()])
        split_rows.append([identifier, pct(train_mean), *entries, f"{pct(row['test']['accuracy'])}/{pct(row['test']['macro_f1'])} ({pp(row['test']['macro_f1']-base['test']['macro_f1'])})",
                           f"{row['summary']['train_seconds']:.1f}s", f"[검증]({identifier}/primary/validation.json) · [test]({identifier}/primary/test.json)"])
    add(table(["ID", "훈련 F1 평균", "primary Acc/F1 (ΔF1)", "resplit1 Acc/F1 (ΔF1)", "resplit2 Acc/F1 (ΔF1)", "고정 test Acc/F1 (ΔF1)", "학습 3회 합계", "원자료"], split_rows))
    add("훈련 지표는 학습 완료 체크포인트의 훈련셋 내 예측을 테스트 종료 후 계산한 설명용 수치다. 학습 성능이 높다는 사실은 새로운 사용자 일반화를 보장하지 않는다. resplit의 훈련·검증 사용자들은 서로 겹치므로 세 값을 독립 표본으로 해석하지 않는다.")

    add("\n## 4. Successful Hypotheses\n")
    confirmed = [i for i in robust if experiments[i]["test"]["macro_f1"] > base["test"]["macro_f1"]]
    add(f"**사전 검증 안정성 기준과 테스트 Macro F1 개선을 모두 충족한 개별 가설은 {len(confirmed)}개**다" + (": " + ", ".join(sorted(confirmed)) + "." if confirmed else ". 따라서 아래 개발 단계의 상승을 최종적으로 확인된 성능 개선으로 해석할 수 없다."))
    add("**개발 검증에서만 확인한 개선**: 조합에 사용할 안정성 통과 기준은 평균 +0.50 pp 이상, 3개 분할 중 2개 이상 개선, 평균 최저 클래스 F1 감소 2 pp 이내, 7개 클래스 중 4개 이상 비악화다. 이 기준을 통과한 모델은 " + (", ".join(sorted(robust)) if robust else "없다") + ". 아래에는 검증 평균 Macro F1이 Baseline보다 높은 가설만 기록하고, 작은 수치 상승·조합 후보·테스트 확인을 구분했다.\n")
    if numerical:
        add(table(["ID", "검증 ΔMacro F1", "개선 분할", "평균 최저 클래스 ΔF1", "클래스 비악화", "조합 후보 여부", "테스트 ΔF1 (사후 확인)"],
                  [[i, pp(experiments[i]["summary"]["macro_f1"] - base["summary"]["macro_f1"]),
                    str(sum(experiments[i]["summary"]["split_f1"][k] > base["summary"]["split_f1"][k] for k in base["summary"]["split_f1"])) + "/3",
                    pp(experiments[i]["summary"]["worst_class_f1"] - base["summary"]["worst_class_f1"]),
                    str(int(np.sum(np.asarray(experiments[i]["summary"]["class_f1"]) >= np.asarray(base["summary"]["class_f1"])))) + "/7",
                    "기준 통과" if i in robust else "기준 미달", pp(experiments[i]["test"]["macro_f1"]-base["test"]["macro_f1"])] for i in numerical]))
    else:
        add("Baseline의 검증 평균 Macro F1을 넘은 개별 가설이 없다.")
    for i in numerical:
        if i in robust:
            add(f"- **{i} (개발 검증에서만 기준 통과)**: {experiments[i]['spec']['hypothesis']}. 검증 평균 변화는 {pp(experiments[i]['summary']['macro_f1']-base['summary']['macro_f1'])}, 테스트 변화는 {pp(experiments[i]['test']['macro_f1']-base['test']['macro_f1'])}다. 이유에 대한 설명은 기전 추정이며 별도의 생체·센서 검증은 아니다.")
    test_only = [i for i in individual_ids if i not in robust and experiments[i]["test"]["macro_f1"] > base["test"]["macro_f1"]]
    if test_only:
        add("테스트에서만 수치가 높고 사전 검증 안정성 기준을 통과하지 못한 모델은 " + ", ".join(f"{i} ({pp(experiments[i]['test']['macro_f1']-base['test']['macro_f1'])})" for i in test_only) + "이다. 이는 사후 관찰이며 이들을 새 승자로 선택하지 않았다.")

    add("\n## 5. 실패하거나 채택하지 않은 가설 분석\n")
    add("다음 설명은 관측 결과에 부합하는 **가능한 원인에 대한 추정**이다. 한 설정에서 실패했다고 해당 모델 계열 전체가 효과 없다고 결론내릴 수 없다. 클래스 저하·분할 불안정 때문에 평균 상승 모델도 배제할 수 있다.\n")
    add(table(["ID", "검증 ΔMacro F1", "판정", "가능한 원인 (추정)"],
              [[i, pp(experiments[i]["summary"]["macro_f1"] - base["summary"]["macro_f1"]), decision(i), failure_reason(experiments[i]["spec"])]
               for i in individual_ids if i not in robust]))

    add("\n## 6. 성공 요소의 단계적 결합과 선택 과정\n")
    pilot_present = any(s["id"].startswith("P") for s in lock["all_specs"])
    chronology = ("개별 가설 최소 10개가 완료된 시점에 성공 요소를 분석하고 P01 예비 조합을 시작했다. 이후 35개 가설이 모두 완료된 뒤 추가 결합·절제를 수행했다. [중간 분석](interim_analysis.json), [예비 조합과 판정](pilot_specs.json)." if pilot_present else
                  "개별 가설 최소 10개 완료 조건을 충족한 뒤 성공 요소를 분석하고 결합·절제를 수행했다.")
    add(chronology + f" 테스트를 열기 전 `{lock['locked_at']}`에 선택 ID·모든 체크포인트·실험 소스·프로토콜 SHA256을 잠갔다. 조합 후보는 개별 검증 안정성 기준을 통과한 구성요소로 제한했다. 각 추가 단계는 부모 대비 평균 F1 +0.20 pp 이상, 2/3 분할 개선 및 클래스 보호 기준을 만족해야 유지했다. 최고 평균 F1과 0.20 pp 이내의 후보에서는 측정 지연이 짧은 모델을 우선했다.")
    step_rows = []
    for step in lock["steps"]:
        if "parent" not in step:
            step_rows.append([step["id"], "출발점", "검증 성공 개별 모델 중 선택", pct(step["macro_f1"]), "—", "초기 모델"])
        else:
            step_rows.append([step["id"], step["parent"], step.get("component", step.get("step", "절제 후 변경")), pct(experiments[step["id"]]["summary"]["macro_f1"]), pp(step["delta"]), "유지" if step["accepted"] else "제거"])
    add("\n" + table(["실험", "부모", "추가 요소", "검증 평균 F1", "부모 대비 ΔF1", "결정"], step_rows))
    add("조합은 두 모델을 사용한다는 사실만으로 효과 있다고 간주하지 않았다. 다른 모델 계열의 오차가 보완적일 것이라는 기대도 매 단계의 실제 차이로 확인했다. 여기의 결합 효과는 선택된 데이터·설정에서의 조건부 기여이며 모든 사용자에 대한 보장은 아니다.")
    if (output / "seed_robustness.json").exists():
        seeds = read(output / "seed_robustness.json")
        assert seeds["selection_uses_these_results"] is False
        seed_rows = []
        for seed in sorted({r["seed"] for r in seeds["results"]}):
            matching = {r["id"]: r for r in seeds["results"] if r["seed"] == seed}
            b, f = matching["B00"]["metrics"], matching[selected]["metrics"]
            seed_rows.append([seed, pct(b["accuracy"]), pct(f["accuracy"]), pct(b["macro_f1"]), pct(f["macro_f1"]), pp(f["macro_f1"]-b["macro_f1"])])
        add("\n**고정 사용자 분할에서 추가 시드 확인**: primary 사용자 집합을 그대로 두고 시드만 101/202로 바꿔 Baseline과 선택 후보를 다시 학습했다. 이 진단은 검증 데이터만 사용하고 모델 선택에는 반영하지 않았다. 이는 분할 변경과 분리한 초기화/확률보정 무작위성 확인이며 테스트를 추가로 열지 않는다.\n")
        add(table(["시드", "Baseline Acc", "Final Acc", "Baseline F1", "Final F1", "ΔF1"], seed_rows))
        add("[추가 시드의 클래스별·사용자별 원수치](seed_robustness.json)")

    add("\n## 7. 최종 아키텍처와 추론 파이프라인\n")
    add(f"선택 모델 **{selected}**: {method(lock['selected_spec'])}. [전체 모델 설정]({selected}/primary/validation.json), [고정 체크포인트]({selected}/primary/model.joblib).\n")
    add(mermaid(lock))
    final_nodes = []; visited = set(); by_id = {s["id"]: s for s in lock["all_specs"]}
    def describe_node(identifier):
        if identifier in visited: return
        visited.add(identifier); spec = by_id[identifier]
        if spec["family"] == "fusion":
            for child in spec["members"]: describe_node(child)
        elif spec["family"] == "temporal": describe_node(spec.get("parent", "B00"))
        final_nodes.append([identifier, method(spec), f"[설정]({identifier}/primary/validation.json)"])
    describe_node(selected)
    add("\n" + table(["최종 구성", "정확한 방법 및 입력", "학습 설정"], final_nodes))
    add("\n입력 수집과 전처리 뒤 각 학습된 변환을 적용하고, 학습기 점수/확률을 얻어 결합·후처리 규칙을 통과시킨 후 가장 큰 값의 자세를 출력한다. 테스트와 훈련에서 사용한 특징 순서 및 센서 순서는 dataset.json과 체크포인트에 고정되어 있다. 출력을 실제 보정된 정답 확률로 해석하는 추가 검증은 하지 않았다.")

    add("\n## 8. 동일 테스트 데이터의 Baseline 대 최종 모델\n")
    add(table(["지표", "Baseline B00", f"최종 {selected}", "차이"],
              [[title, pct(base["test"][key]), pct(final["test"][key]), pp(final["test"][key]-base["test"][key])]
               for key, title in [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("worst_class_f1", "최저 클래스 F1")]]))
    add("\n![테스트 최종 비교](charts/final_test_comparison.png)\n")
    add(table(["클래스", "표본", "P Baseline", "P Final", "R Baseline", "R Final", "F1 Baseline", "F1 Final", "ΔF1"],
              [[name, base["test"]["support"][i], pct(base["test"]["precision"][i]), pct(final["test"]["precision"][i]),
                pct(base["test"]["recall"][i]), pct(final["test"]["recall"][i]), pct(base["test"]["f1"][i]), pct(final["test"]["f1"][i]),
                pp(final["test"]["f1"][i]-base["test"]["f1"][i])] for i, name in enumerate(manifest["class_names"])]))
    add("\n![클래스별 Precision Recall F1](charts/per_class_comparison.png)")
    subject_rows = []
    for subject, score in base["test"]["per_subject"].items():
        f = final["test"]["per_subject"][subject]
        subject_rows.append([subject, int(sum(score["support"])), pct(score["macro_f1"]), pct(f["macro_f1"]), pp(f["macro_f1"]-score["macro_f1"])])
    add("\n" + table(["테스트 사용자", "창 수", "Baseline F1", "Final F1", "ΔF1"], subject_rows))
    add("\n사용자 단위 paired bootstrap으로 같은 사용자의 모든 창을 함께 복원추출했다. 개별 창을 독립 재추출하지 않아 시간 상관을 일부 보존하지만, **테스트 사용자가 4명뿐**이므로 신뢰구간의 검정력과 새 코호트에 대한 대표성은 낮다. 아래 positive 비율은 관측된 사용자 재가중 결과이며 보편적 개선 확률이나 p-value가 아니다.\n")
    add(table(["지표", "관측 Δ", "사용자 bootstrap 95% 구간", "Δ>0 재표본 비율"],
              [[k, pp(v["point_delta"]), f"[{pp(v['percentile_95_interval'][0])}, {pp(v['percentile_95_interval'][1])}]", pct(v["fraction_positive"])]
               for k, v in bootstrap["metrics"].items()]))
    add(f"시드 {BOOTSTRAP_SEED}, {BOOTSTRAP_DRAWS}회. [불확실성 계산 설정과 수치](bootstrap_subject.json). 이 테스트 코호트는 과거 실험에서 이미 관측된 후향적 데이터이며 새로 확보한 완전 미관측 코호트가 아니다. 이번 선택 과정에서 테스트 점수 사용은 차단했지만 과거 관측 사실까지 제거할 수는 없다.")

    add("\n## 9. 혼동행렬 비교\n")
    add("행은 실제 클래스, 열은 예측 클래스다. 세 그림은 Baseline, 최종 모델, 최종−Baseline 개수 차이다. 대각선 양수와 비대각선 음수가 개선 방향이며, 색 하나가 모든 셀에서 같은 의미의 개선을 나타내지는 않는다.\n")
    add("![혼동행렬과 차이](charts/confusion_comparison.png)\n")
    for identifier, result in [("B00", base), (selected, final)]:
        add(f"**{identifier} 혼동행렬**\n")
        add(table(["실제\\예측", *[str(i) for i in range(7)]], [[str(i), *row] for i, row in enumerate(result["test"]["confusion_matrix"])]))
    add("클래스 ID: " + "; ".join(f"{i}={name}" for i, name in enumerate(manifest["class_names"])))

    add("\n## 10. Ablation Test\n")
    add("선택 모델의 구성요소를 제거한 모델을 같은 분할에서 다시 평가했다. 기여는 제거 전 모델−제거 모델의 검증 평균 Macro F1이며 양수는 유지 근거다. 테스트 값은 선택 후 참고 비교다. 단일 요소 모델의 제거 실험은 Baseline 또는 이미 실행된 대응 실험을 재사용한다.\n")
    ablation_rows = []
    for ablation in lock["ablations"]:
        row = experiments[ablation["id"]]
        ablation_rows.append([ablation["parent"], ablation["id"], ablation["removed"], pct(ablation["f1"]), pp(ablation["contribution"]),
                              pct(row["test"]["macro_f1"]), pp(experiments[ablation["parent"]]["test"]["macro_f1"]-row["test"]["macro_f1"])])
    add(table(["제거 전", "제거 모델", "제거 요소", "제거 모델 검증 F1", "검증 기여", "제거 모델 test F1", "test 기여"], ablation_rows))
    add("복합 모델은 전체 파이프라인을 유지한 채 내부 분기·메타 분류기·전처리·후처리 등 정의된 구성요소를 하나씩 제거했다. 신경망의 모든 층이나 센서 각각까지 미세 분해한 것은 아니다. `full_pipeline` 플래그와 제거 위치는 선택 잠금에 보존된다.")

    add("\n## 11. 모델 복잡도와 실시간 사용성\n")
    cost_rows = []
    for identifier, result in [("B00", base), (selected, final)]:
        p = result["splits"]["primary"]; c = p["model_counts"]; timing = p["inference"]
        cost_rows.append([identifier, f"{p['train_seconds']:.2f}", f"{timing['cached_single_ms']:.3f}",
                          f"{timing['raw_window_ms']:.3f}", f"{timing['raw_window_p95_ms']:.3f}",
                          f"{timing['batch_ms_per_window']:.3f}", f"{p['checkpoint_bytes']/1024**2:.3f}",
                          c["svm_support_vectors"], f"{c['tree_count']}/{c['tree_nodes']}", c["neural_trainable_parameters"]])
    add(table(["모델", "primary 학습(s)", "캐시 특징 추론(ms)", "특징 포함 중앙값(ms)", "특징 포함 p95(ms)", "batch/window(ms)", "joblib(MiB)", "SVM SV 수", "tree 수/nodes", "신경망 parameter 수"], cost_rows))
    latency = final["splits"]["primary"]["inference"]["raw_window_ms"]
    add(f"최종 모델의 측정 중앙값 {latency:.3f} ms는 3초 창 간격의 {latency/3000*100:.3f}%다. {'계산량 측면에서 3초 간격 추론은 실용적이다' if latency < 300 else '계산 지연을 줄이기 위한 추가 점검이 필요하다'}. 다만 창 자체를 채우는 3초 지연이 따로 있고, Serial/CSV 수집·GUI·실물 센서·장시간 구동은 이 벤치마크에 포함되지 않는다. 후처리는 추가 흔들림 감소와 반응 지연을 바꾸며 최대 3창 투표는 최대 6초의 과거 예측을 사용한다. 동시 호스트 작업에 따라 CPU 시간은 달라질 수 있다.")
    add("확률 결합 및 temporal 체크포인트의 `train_seconds`는 이미 훈련한 구성 모델을 읽고 조립한 시간이다. 구성 모델을 처음부터 만드는 전체 학습 비용과 같지 않으므로 구성 모델별 시간을 함께 봐야 한다. SVM의 support vector 수, 트리 node 수, 신경망 parameter 수는 서로 다른 복잡도 척도다. [구조 크기와 모든 측정값](report_metrics.json).")

    add("\n## 12. 최종 해석, 한계와 재현 방법\n")
    accepted = [s for s in lock["steps"] if s.get("accepted")]
    add(f"검증에서 유지된 단계는 {', '.join(s['id'] + '(' + pp(s['delta']) + ')' for s in accepted) if accepted else '없으며, 가장 적절한 개별 모델을 유지했다'}. 최종 {selected}의 Baseline 대비 검증 평균 Macro F1 차이는 {pp(final['summary']['macro_f1']-base['summary']['macro_f1'])}, 동일 테스트 차이는 {pp(delta)}다. 클래스별 차이와 절제 실험이 어떤 구성에 성능 변화가 동반되는지 보여주므로 Accuracy 순위만으로 결론내리지 않았다.")
    if any(by_id[i]["kind"] == "subject_balanced_svm" for i in visited) and any(by_id[i]["kind"] == "ema" for i in visited):
        add("최종 모델에 남은 두 요소는 사용자별 표본 수 균등화와 인과적 확률 EMA다. 전자는 사용자별 기록 수가 훈련 거리 척도·SVM 경계에 미치는 영향을 조정하고, 후자는 연속 예측의 단발성 변화를 완화한다. 서로 학습 단계와 추론 단계에 작용하기 때문에 보완 가능성이 있으며, 실제 기여 크기는 위의 각 요소 제거 실험으로 확인했다. 그 기전 설명 자체가 센서 편차의 물리적 원인까지 증명하는 것은 아니다.")
    if delta > 0:
        add("이 결과가 지지하는 결론은 최종 구성요소를 포함한 파이프라인이 이 동일 테스트 데이터에서 더 균형 있는 분류 점수를 얻었다는 것이다. 기대한 모델링 원리가 원인이라는 설명은 클래스 변화·절제 결과와 함께 보는 추정이며, 체형·센서 편차 자체의 인과성을 증명하지 않는다.")
    else:
        add("검증 개선이 테스트로 이어지지 않은 만큼 최종 후보가 기존 모델보다 일반적으로 우수하다고 볼 근거는 부족하다. 성능 저하를 숨기지 않고 검증 선택의 한계로 남긴다.")
    lower = bootstrap["metrics"]["macro_f1"]["percentile_95_interval"][0]
    add("사용자 bootstrap Macro F1 구간이 0을 포함하거나 그 아래로 내려가므로, 모든 새 사용자에 대한 안정적 개선 주장은 뒷받침되지 않는다." if lower <= 0 else
        "사용자 bootstrap Macro F1 구간은 관측된 사용자 재가중에서 양수지만, 4명·후향적 코호트라는 제한 때문에 새로운 사용자에 대한 보장을 뜻하지 않는다.")
    add("정적 자세 클립의 temporal smoothing 결과는 실제 자세 전환 구간의 지연/오작동 성능을 입증하지 않는다. 개인화 기준은 사후 기록 대체이며 별도 등록 UI나 현장 평가를 추가한 것이 아니다. 최종 모델을 운영 활성 모델로 자동 교체하지 않았다. 기존 데이터·모델 파일은 유지하고 실험 코드는 `experiments/posture_search`, 생성물은 이 결과 디렉터리에 분리했다.")
    add("\n재현은 저장소 루트의 기존 `.venv`에서 다음 순서로 실행한다. 새 출력 폴더를 사용하면 결과를 재계산하고, 같은 폴더는 일치하는 데이터·설정·체크포인트를 재사용한다. 테스트는 선택 잠금 생성 후에만 실행할 수 있다.\n")
    add("```bash\n.venv/bin/python -m experiments.posture_search.run train --output results/posture_search_reproduce\n.venv/bin/python -m experiments.posture_search.run combine --output results/posture_search_reproduce\n.venv/bin/python -m experiments.posture_search.run test --output results/posture_search_reproduce\n.venv/bin/python -m experiments.posture_search.report --output results/posture_search_reproduce\n```")
    add("동일 분할·시드·라이브러리 버전과 소스가 모델 재현의 기준이며 타 하드웨어/라이브러리 버전에서는 부동소수점·시간 측정이 달라질 수 있다. 데이터 원본 SHA256, 캐시 SHA256, 각 체크포인트 SHA256, 소스 SHA256, 환경 pip freeze와 seed를 모두 저장했다. `report_metrics.json`에는 각 실험/분할의 학습·검증·테스트 Accuracy, Macro F1, 클래스별 P/R/F1·support·혼동행렬과 비용·설정이 들어 있다.")
    add("\n[전체 실험 TSV](all_experiments.tsv) · [전체 수치 JSON](report_metrics.json) · [테스트 요약](test_summary.json) · [개별 성공 분석](individual_analysis.json) · [선택/결합/절제 근거](selection_lock.json) · [데이터/환경/분할](dataset.json) · [bootstrap](bootstrap_subject.json)")
    (output / "report_ko.md").write_text("\n\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(); output = args.output.resolve()
    if not (output / "selection_lock.json").exists() or not (output / "test_summary.json").exists():
        raise SystemExit("Run train, combine (selection lock), and test before report; no partial/test-driven selection report")
    manifest = read(output / "dataset.json"); lock = read(output / "selection_lock.json")
    assert lock["test_used_for_selection"] is False
    assert lock["dataset_fingerprint"] == manifest["fingerprint"]
    assert lock["protocol_sha256"] == digest(output / "protocol.json")
    tests = read(output / "test_summary.json")
    assert tests["selected_id"] == lock["selected_id"]
    with threadpool_limits(limits=1):
        report = collect(output, manifest, lock, tests)
    first = dict(np.load(output / "B00/primary/test_predictions.npz", allow_pickle=False))
    second = dict(np.load(output / lock["selected_id"] / "primary/test_predictions.npz", allow_pickle=False))
    for key in ("y", "groups", "record_ids", "start_s"):
        np.testing.assert_array_equal(first[key], second[key], err_msg=f"Baseline/final test mismatch: {key}")
    bootstrap = paired_subject_bootstrap(first["y"], first["proba"].argmax(1), second["proba"].argmax(1), first["groups"])
    write_json(output / "bootstrap_subject.json", bootstrap)
    export_tsv(output, report); plots(output, report, lock, manifest["class_names"])
    write_report(output, manifest, lock, tests, report, bootstrap)
    print(output / "report_ko.md", flush=True)


if __name__ == "__main__":
    main()
