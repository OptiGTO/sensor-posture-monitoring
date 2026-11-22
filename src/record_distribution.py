"""Read-only per-recording sensor distribution maps for the desktop app.

Only CSV files are read. Nothing is written except a PNG the user explicitly
saves. Same-person comparisons are review aids and never assign a label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib import colormaps, font_manager
from matplotlib.figure import Figure
from matplotlib.text import Text

from .comparison_charts import chart_font
from .posture_core import CLASS_NAMES, SENSOR_COLUMNS, normalize_text, read_sensor_csv, standardize_label

GRID = 4
_MEAN_CACHE: dict[Path, tuple[tuple[int, int], np.ndarray]] = {}


def posture_name(value: object) -> str:
    """Standard posture name, or '' for measurement-only / unknown labels."""
    text = normalize_text(value or "")
    if not text:
        return ""
    try:
        return standardize_label(text)
    except ValueError:
        return ""


def centered_cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    """Similarity of 16-channel patterns after removing each pattern's overall level."""
    a, b = np.asarray(a, float) - np.mean(a), np.asarray(b, float) - np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 0 else None


def read_values(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = read_sensor_csv(path)
    elapsed = frame["elapsed_s"].to_numpy(float)
    return elapsed - elapsed[0], frame[SENSOR_COLUMNS].to_numpy(float)


def recording_mean(path: Path) -> np.ndarray:
    """Whole-recording channel mean, cached by file size and mtime."""
    path = Path(path)
    stat = path.stat()
    signature = (stat.st_size, stat.st_mtime_ns)
    cached = _MEAN_CACHE.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    mean = read_values(path)[1].mean(0)
    _MEAN_CACHE[path] = (signature, mean)
    return mean


@dataclass
class Reference:
    mean: np.ndarray
    count: int


@dataclass
class RecordDistribution:
    path: Path
    subject: str
    posture: str
    elapsed: np.ndarray
    values: np.ndarray
    references: dict[str, Reference] = field(default_factory=dict)
    skipped_references: int = 0

    @property
    def duration(self) -> float:
        return float(self.elapsed[-1]) if len(self.elapsed) else 0.0

    def interval_mean(self, start: float | None = None, end: float | None = None) -> np.ndarray:
        start, end = self.clamp(start, end)
        take = (self.elapsed >= start) & (self.elapsed <= end)
        if not take.any():
            raise ValueError("선택한 구간에 측정값이 없습니다.")
        return self.values[take].mean(0)

    def clamp(self, start: float | None, end: float | None) -> tuple[float, float]:
        start = 0.0 if start is None else float(start)
        end = self.duration if end is None else float(end)
        if not (np.isfinite(start) and np.isfinite(end)) or start < 0 or end <= start:
            raise ValueError("구간은 0 이상이고 끝이 시작보다 커야 합니다.")
        return start, min(end, self.duration)

    def similarities(self, start=None, end=None) -> dict[str, float | None]:
        mean = self.interval_mean(start, end)
        return {name: centered_cosine(mean, reference.mean) for name, reference in self.references.items()}


def load_distribution(record, records) -> RecordDistribution:
    """Load one recording plus same-person per-posture references (the recording itself excluded)."""
    path = Path(record.csv_path)
    elapsed, values = read_values(path)
    subject = normalize_text(record.subject)
    grouped: dict[str, list[np.ndarray]] = {}
    skipped = 0
    if subject:
        for other in records:
            name = posture_name(other.posture)
            if not name or normalize_text(other.subject) != subject or Path(other.csv_path).resolve() == path.resolve():
                continue
            try:
                grouped.setdefault(name, []).append(recording_mean(Path(other.csv_path)))
            except (OSError, ValueError):
                skipped += 1
    references = {name: Reference(np.mean(items, 0), len(items))
                  for name in CLASS_NAMES if (items := grouped.get(name))}
    return RecordDistribution(path, subject, posture_name(record.posture), elapsed, values, references, skipped)


def default_comparison(distribution: RecordDistribution, start=None, end=None) -> str | None:
    """Most similar same-person posture other than the recording's own label."""
    similarities = {name: value for name, value in distribution.similarities(start, end).items()
                    if name != distribution.posture and value is not None}
    return max(similarities, key=similarities.get) if similarities else None


def _format(value):
    return "-" if value is None else f"{value:.3f}"


def distribution_figure(distribution: RecordDistribution, comparison: str | None = None,
                        start: float | None = None, end: float | None = None) -> Figure:
    start, end = distribution.clamp(start, end)
    mean = distribution.interval_mean(start, end)
    similarities = distribution.similarities(start, end)
    figure = Figure(figsize=(11, 7.6), dpi=100, facecolor="white")
    grid = figure.add_gridspec(2, 4, height_ratios=[1.15, 1], width_ratios=[1, 1, 1, .06],
                               hspace=.42, wspace=.25, left=.06, right=.95, top=.93, bottom=.05)
    series = figure.add_subplot(grid[0, :3])
    palette = colormaps["tab20"]
    for channel, name in enumerate(SENSOR_COLUMNS):
        series.plot(distribution.elapsed, distribution.values[:, channel], lw=.8, color=palette(channel), label=name)
    if start > 0 or end < distribution.duration:
        series.axvspan(start, end, color="#fde68a", alpha=.35, label="선택 구간")
    series.set_xlabel("경과 시간 (초)")
    series.set_ylabel("센서값")
    label = distribution.posture or "라벨 없음"
    series.set_title(f"{distribution.path.name} · {distribution.subject or '이름 없음'} · 라벨: {label}", loc="left", fontsize=11)
    series.legend(ncol=9, fontsize=7, loc="upper right")

    panels = [(mean, f"이 기록 평균 ({start:.1f}~{end:.1f}초)")]
    for name, role in ((distribution.posture, "같은 사람 라벨 자세"), (comparison, "같은 사람 비교 자세")):
        reference = distribution.references.get(name) if name else None
        if reference is None:
            panels.append((None, f"{role}: {name or '-'}\n비교 기록 없음"))
        else:
            panels.append((reference.mean, f"{role}: {name} ({reference.count}개)\n유사도 {_format(similarities.get(name))}"))
    present = [pattern for pattern, _ in panels if pattern is not None]
    low, high = min(float(p.min()) for p in present), max(float(p.max()) for p in present)
    if high <= low:
        high = low + 1e-6
    image = None
    for column, (pattern, title) in enumerate(panels):
        axes = figure.add_subplot(grid[1, column])
        axes.set_title(title, fontsize=9)
        axes.set_xticks([]); axes.set_yticks([])
        if pattern is None:
            axes.text(.5, .5, "비교 기록 없음", ha="center", va="center", color="#64748b", transform=axes.transAxes)
            continue
        image = axes.imshow(np.asarray(pattern).reshape(GRID, GRID), vmin=low, vmax=high, cmap="viridis")
        for row in range(GRID):
            for col in range(GRID):
                value = pattern[row * GRID + col]
                shade = "black" if (value - low) / (high - low) > .6 else "white"
                axes.text(col, row, f"R{row + 1}{col + 1}\n{value:.2f}", ha="center", va="center", fontsize=7, color=shade)
    if image is not None:
        figure.colorbar(image, cax=figure.add_subplot(grid[1, 3]))
    for text in figure.findobj(Text):
        text.set_fontproperties(font_manager.FontProperties(fname=chart_font(), size=text.get_fontsize()))
    return figure


def similarity_summary(distribution: RecordDistribution, start=None, end=None) -> str:
    similarities = distribution.similarities(start, end)
    if not similarities:
        return "같은 사람의 다른 라벨 기록이 없어 자세별 비교를 할 수 없습니다."
    parts = [f"{name} {_format(value)}" + (" (라벨)" if name == distribution.posture else "")
             for name, value in sorted(similarities.items(), key=lambda item: -(item[1] if item[1] is not None else -2))]
    text = "같은 사람 자세별 패턴 유사도(수준 제거 코사인, 높을수록 비슷): " + ", ".join(parts)
    if distribution.skipped_references:
        text += f" · 읽지 못한 비교 기록 {distribution.skipped_references}개 제외"
    return text + ". 참고용이며 라벨을 판정하지 않습니다."
