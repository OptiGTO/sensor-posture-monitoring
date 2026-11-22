"""Read-only plots of saved comparison metrics; never fit or select a model."""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter

from .ui_korean import ko

# LOSO over development subjects is the default comparison metric when saved.
SPLITS = {"LOSO (기본)": "loso", "검증": "validation", "최종 평가": "final_evaluation"}


def default_split_label(report):
    rows = report.get("models") or []
    return "LOSO (기본)" if any(isinstance(row.get("loso"), dict) and row["loso"] for row in rows) else "검증"
# label, source key, multiplier, unit, bounded score
MODEL_METRICS = {
    "전체 정확도 (Accuracy)": ("accuracy", 100, "%", True),
    "Macro F1": ("macro_f1", 100, "%", True),
    "최저 자세 F1": ("worst_class_f1", 100, "%", True),
    "추론 시간 (중앙값)": ("inference_ms_median", 1, "ms", False),
    "추론 시간 (95백분위)": ("inference_ms_p95", 1, "ms", False),
    "학습 시간 (선택된 설정)": ("training_seconds", 1, "초", False),
    "모델 크기": ("model_size_bytes", 1 / 1024, "KiB", False),
}
POSTURE_METRICS = {
    "자세별 정답률 (Recall)": "recall",
    "정밀도 (Precision)": "precision",
    "F1": "f1",
}
COLORS = ("#2563eb", "#0d9488", "#d97706")


def number(value, *, score=False):
    """Missing or invalid values must not look like measured zero scores."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 and (not score or result <= 1) else None


def model_label(row):
    if row.get("display_label"):
        return str(row["display_label"])
    return f"{ko(row.get('name') or row.get('family') or '이름 없음')} · {row.get('window_seconds', '?')}초"


def model_value(row, split, metric):
    key, scale, _unit, score = MODEL_METRICS[metric]
    source = (row.get(split) or {}) if score else row
    value = number(source.get(key), score=score)
    return None if value is None else value * scale


def posture_labels(report, metrics):
    # The saved label order also defines the confusion-matrix row/column order.
    return list(report.get("labels") or (metrics.get("per_class") or {}).keys()
                or (metrics.get("per_class_f1") or {}).keys())


def matrix_values(report, metrics):
    labels = list(metrics.get("confusion_labels") or posture_labels(report, metrics))
    try:
        matrix = np.asarray(metrics.get("confusion_matrix"), dtype=float)
    except (TypeError, ValueError):
        return labels, None
    if (not labels or matrix.shape != (len(labels), len(labels))
            or not np.isfinite(matrix).all() or (matrix < 0).any()
            or not np.equal(matrix, np.floor(matrix)).all()):
        return labels, None
    return labels, matrix


def posture_values(report, row, split, key):
    metrics = row.get(split) or {}
    labels = posture_labels(report, metrics)
    values, supports = [], []
    for label in labels:
        saved = (metrics.get("per_class") or {}).get(label) or {}
        raw = saved.get(key)
        if key == "f1" and raw is None:
            raw = (metrics.get("per_class_f1") or {}).get(label)
        support = number(saved.get("support"))
        value = number(raw, score=True)
        # A class with no evaluated examples has no measured success rate.
        values.append(None if value is None or support == 0 else value * 100)
        supports.append(support)
    return labels, values, supports


@lru_cache(maxsize=1)
def chart_font():
    for family in ("Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"):
        try:
            return font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
        except ValueError:
            continue
    return font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))


def _finish(figure):
    for text in figure.findobj(Text):
        text.set_fontproperties(font_manager.FontProperties(fname=chart_font(), size=text.get_fontsize()))
    return figure


def empty_chart(message, title=""):
    figure = Figure(figsize=(8, 3.8), dpi=100, layout="constrained", facecolor="white")
    axes = figure.add_subplot()
    axes.set_title(title, loc="left", pad=16)
    axes.text(.5, .5, message, ha="center", va="center", transform=axes.transAxes, color="#64748b")
    axes.set_axis_off()
    return _finish(figure)


def _bar_axes(figure, labels, title, unit, score):
    axes = figure.add_subplot()
    axes.set_title(title, loc="left", pad=18, fontsize=12)
    axes.set_yticks(range(len(labels)), labels)
    axes.invert_yaxis()
    axes.set_xlabel(unit)
    axes.set_axisbelow(True)
    axes.grid(axis="x", color="#e2e8f0")
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(axis="both", length=0, labelsize=10)
    if score:
        axes.set_xlim(0, 114)
        axes.set_xticks(range(0, 101, 20))
        axes.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    return axes


def _bars(axes, positions, values, color, *, height=.55, label=None, unit="%"):
    valid = [(position, value) for position, value in zip(positions, values) if value is not None]
    axes.barh([p for p, _ in valid], [v for _, v in valid], height=height, color=color, label=label)
    for position, value in zip(positions, values):
        text = "미기록" if value is None else f"{value:,.1f}{unit}"
        axes.annotate(text, (value or 0, position), xytext=(5, 0), textcoords="offset points",
                      va="center", fontsize=9, color="#475569" if value is None else "#1e293b")


def model_chart(report, split, metric):
    rows = report.get("models") or []
    if not rows:
        return empty_chart("표시할 모델 결과가 없습니다. 학습 상태를 확인하세요.")
    _key, _scale, unit, score = MODEL_METRICS[metric]
    figure = Figure(figsize=(8, max(3.8, len(rows) * .43 + 1.3)), dpi=100, layout="constrained")
    title = f"{ko(split) + ' · ' if score else ''}{metric} · {'높을수록 좋음' if score else '낮을수록 작거나 빠름'}"
    axes = _bar_axes(figure, [model_label(row) for row in rows], title, unit, score)
    values = [model_value(row, split, metric) for row in rows]
    if not score:
        axes.set_xlim(0, max([value for value in values if value is not None] or [1]) * 1.3 or 1)
    _bars(axes, range(len(rows)), values, COLORS[0], unit=unit)
    return _finish(figure)


def posture_chart(report, row, split, metric="모든 지표"):
    if row is None:
        return empty_chart("자세별 성능을 볼 모델을 선택하세요.")
    keys = list(POSTURE_METRICS.items()) if metric == "모든 지표" else [(metric, POSTURE_METRICS[metric])]
    labels, _values, supports = posture_values(report, row, split, keys[0][1])
    if not labels:
        return empty_chart("저장된 자세별 평가 지표가 없습니다.")
    figure = Figure(figsize=(8, max(4.5, len(labels) * (0.78 if len(keys) > 1 else .48) + 1.6)),
                    dpi=100, layout="constrained")
    ticks = [f"{label} (n={int(support) if support is not None else '?'})" for label, support in zip(labels, supports)]
    axes = _bar_axes(figure, ticks, f"{model_label(row)} · {ko(split)}", "%", True)
    height = .72 / len(keys)
    for index, (label, key) in enumerate(keys):
        _labels, values, _supports = posture_values(report, row, split, key)
        positions = np.arange(len(labels)) + (index - (len(keys) - 1) / 2) * height
        _bars(axes, positions, values, COLORS[index], height=height * .86, label=label)
    axes.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncols=len(keys), frameon=False, fontsize=9)
    return _finish(figure)


def confusion_chart(report, row, split):
    if row is None:
        return empty_chart("혼동 행렬을 볼 모델을 선택하세요.")
    labels, matrix = matrix_values(report, row.get(split) or {})
    if matrix is None:
        return empty_chart("저장된 혼동 행렬이 없거나 자세 순서와 크기가 맞지 않습니다.")
    totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(matrix * 100, totals, out=np.zeros_like(matrix), where=totals > 0)
    figure = Figure(figsize=(8, max(5.4, len(labels) * .62 + 1.7)), dpi=100, layout="constrained")
    axes = figure.add_subplot()
    heatmap = axes.imshow(np.ma.masked_where(np.broadcast_to(totals == 0, matrix.shape), percentages),
                          cmap="Blues", vmin=0, vmax=100)
    axes.set_xticks(range(len(labels)), [label.replace(" ", "\n", 1) for label in labels], fontsize=9)
    axes.set_yticks(range(len(labels)), labels, fontsize=10)
    axes.set_xlabel("예측 자세")
    axes.set_ylabel("실제 자세")
    axes.set_title(f"{model_label(row)} · {ko(split)}\n셀: 분석 창 수 / 실제 자세 내 비율", loc="left", pad=15, fontsize=12)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = percentages[i, j]
            text = f"{matrix[i, j]:.0f}\n{value:.1f}%" if totals[i, 0] else "평가 없음"
            axes.text(j, i, text, ha="center", va="center", fontsize=8,
                      color="white" if value > 55 else "#1e293b")
    figure.colorbar(heatmap, ax=axes, shrink=.8, format=PercentFormatter(100, decimals=0))
    return _finish(figure)
