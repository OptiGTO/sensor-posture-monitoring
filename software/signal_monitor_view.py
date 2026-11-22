"""Heart-monitor style live view of the sixteen pressure-sensor signals.

Each panel sits at its R<row><column> position in the 4×4 array and sweeps the
last seconds of that sensor from left to right, like a patient monitor. All
panels share one vertical scale so trace heights can be compared directly.
The view only reads a :class:`~src.signal_monitor.SignalBuffer`; it never
touches the recorder or the model.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from src.posture_core import SENSOR_COLUMNS
from src.signal_monitor import CommonScale, SignalBuffer, format_signal, sweep_segments

# Palette shared with the rest of the application (posture_app_base.py).
FONT = "Apple SD Gothic Neo"
PANEL_BG = "#eef2f8"
BORDER = "#d9e2ef"
CELL_BG = "#ffffff"
GRID_MINOR = "#edf1f7"
GRID_MAJOR = "#dfe6f0"
TEXT_MUTED = "#475569"
TEXT_SUBTLE = "#64748b"
TRACE = "#1565c0"
ACCENT = "#536dfe"
STALE = "#90a4ae"
ALERT = "#b71c1c"

GAP = 6
HEADER_HEIGHT = 18
LEVEL_WIDTH = 4


class SignalMonitorView(ttk.LabelFrame):
    """Sweeping 4×4 sensor traces with current value and level bar per sensor."""

    REFRESH_MS = 50

    def __init__(self, parent, buffer: SignalBuffer, *, grid: int = 4, panel_height: int = 64,
                 stale_after_s: float = 1.5) -> None:
        super().__init__(parent, text="실시간 센서 신호", padding=(10, 6))
        self.buffer = buffer
        self.grid_size = grid
        self.panel_height = panel_height
        self.stale_after_s = stale_after_s
        self.scale = CommonScale()
        self.names = SENSOR_COLUMNS[: grid * grid]
        self.summary = tk.StringVar(self, value="센서 신호를 기다리는 중")
        header = ttk.Frame(self)
        header.pack(fill="x")
        self.summary_label = ttk.Label(header, textvariable=self.summary, foreground=TEXT_MUTED)
        self.summary_label.pack(side="left")
        ttk.Label(header, text=f"R행열 번호 배치 · 최근 {buffer.span_s:g}초 · 16개 센서 공통 척도",
                  foreground=TEXT_SUBTLE).pack(side="right")
        height = grid * panel_height + (grid + 1) * GAP
        self.canvas = tk.Canvas(self, height=height, background=PANEL_BG,
                                highlightthickness=1, highlightbackground=BORDER)
        self.canvas.pack(fill="x", pady=(6, 0))
        self._panels: list[dict] = []
        self._width: int | None = None
        self._drawn_state: tuple | None = None
        self._summary_color = TEXT_MUTED
        self._after_id: str | None = None
        self.canvas.bind("<Configure>", self._on_configure)
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._after_id = self.after(self.REFRESH_MS, self._tick)

    # ------------------------------------------------------------------ layout
    def _on_configure(self, event) -> None:
        width = max(int(event.width) - 2, self.grid_size * 60)
        if width != self._width:
            self._width = width
            self._build(width)
            self.refresh(force=True)

    def _build(self, width: int) -> None:
        canvas = self.canvas
        canvas.delete("all")
        self._panels = []
        span = self.buffer.span_s
        seconds = int(round(span))
        panel_width = (width - (self.grid_size + 1) * GAP) / self.grid_size
        for index, name in enumerate(self.names):
            row, column = divmod(index, self.grid_size)
            x0 = GAP + column * (panel_width + GAP)
            y0 = GAP + row * (self.panel_height + GAP)
            x1, y1 = x0 + panel_width, y0 + self.panel_height
            frame = canvas.create_rectangle(x0, y0, x1, y1, fill=CELL_BG, outline=BORDER, width=1)
            plot = (x0 + 6, y0 + HEADER_HEIGHT, x1 - 8 - LEVEL_WIDTH, y1 - 5)
            px0, py0, px1, py1 = plot
            # ECG-paper grid: one vertical line per second, heavier every 5 s.
            for second in range(1, seconds):
                x = px0 + (px1 - px0) * second / span
                canvas.create_line(x, py0, x, py1, fill=GRID_MAJOR if second % 5 == 0 else GRID_MINOR)
            for fraction in (0.25, 0.5, 0.75):
                y = py0 + (py1 - py0) * fraction
                canvas.create_line(px0, y, px1, y, fill=GRID_MINOR)
            canvas.create_text(x0 + 6, y0 + 3, text=name, anchor="nw", font=(FONT, 10, "bold"), fill=TEXT_MUTED)
            value = canvas.create_text(x1 - 6, y0 + 3, text="-", anchor="ne", font=(FONT, 10), fill=TEXT_MUTED)
            level_box = (x1 - 4 - LEVEL_WIDTH, py0, x1 - 4, py1)
            canvas.create_rectangle(*level_box, fill=PANEL_BG, outline="")
            level = canvas.create_rectangle(level_box[0], py1, level_box[2], py1, fill=ACCENT, outline="")
            head = canvas.create_oval(0, 0, 0, 0, fill=ACCENT, outline="", state="hidden")
            self._panels.append({"name": name, "frame": frame, "plot": plot, "level_box": level_box,
                                 "value": value, "level": level, "head": head, "lines": []})
        self._drawn_state = None

    # ----------------------------------------------------------------- drawing
    def _tick(self) -> None:
        self._after_id = None
        try:
            self.refresh()
            self._after_id = self.after(self.REFRESH_MS, self._tick)
        except tk.TclError:
            # The screen was replaced while a refresh was pending.
            self._after_id = None

    def _on_destroy(self, event) -> None:
        if event.widget is self and self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _set_summary(self, text: str, color: str) -> None:
        if self.summary.get() != text:
            self.summary.set(text)
        if color != self._summary_color:
            self.summary_label.configure(foreground=color)
            self._summary_color = color

    def refresh(self, force: bool = False) -> None:
        snapshot = self.buffer.snapshot()
        stale = snapshot.age_s is None or snapshot.age_s > self.stale_after_s
        if not len(snapshot.times):
            self._set_summary("센서 신호를 기다리는 중", TEXT_MUTED)
            return
        if stale:
            self._set_summary(f"신호 끊김 · 마지막 입력 {snapshot.age_s:.1f}초 전", ALERT)
        state = (snapshot.total_samples, stale, self._width)
        if not self._panels or (state == self._drawn_state and not force):
            return
        self._drawn_state = state
        low, high = self.scale.update(snapshot.values)
        latest = snapshot.latest
        strongest = int(np.argmax(latest[: len(self._panels)]))
        if not stale:
            rate = snapshot.sample_rate_hz()
            rate_text = f"입력 {rate:.1f} Hz · " if rate else ""
            self._set_summary(
                f"{rate_text}공통 척도 {format_signal(low, low, high)}–{format_signal(high, low, high)} · "
                f"가장 높은 값 {self.names[strongest]} {format_signal(latest[strongest], low, high)}", TEXT_MUTED)
        positions, segments = sweep_segments(snapshot.times, self.buffer.span_s,
                                             origin_s=snapshot.origin_s or 0.0)
        drawable = [(start, stop) for start, stop in segments if stop - start >= 2]
        trace_color = STALE if stale else TRACE
        canvas = self.canvas
        for channel, panel in enumerate(self._panels):
            px0, py0, px1, py1 = panel["plot"]
            fractions = np.clip((snapshot.values[:, channel] - low) / (high - low), 0.0, 1.0)
            xs = px0 + positions * (px1 - px0)
            ys = py1 - fractions * (py1 - py0)
            lines = panel["lines"]
            while len(lines) < len(drawable):
                lines.append(canvas.create_line(0, 0, 0, 0, fill=trace_color, width=2,
                                                joinstyle="round", capstyle="round"))
            for item, (start, stop) in zip(lines, drawable):
                canvas.coords(item, *np.column_stack((xs[start:stop], ys[start:stop])).ravel().tolist())
                canvas.itemconfigure(item, state="normal", fill=trace_color)
            for item in lines[len(drawable):]:
                canvas.itemconfigure(item, state="hidden")
            head_x, head_y = xs[-1], ys[-1]
            canvas.coords(panel["head"], head_x - 3, head_y - 3, head_x + 3, head_y + 3)
            canvas.itemconfigure(panel["head"], state="hidden" if stale else "normal")
            canvas.tag_raise(panel["head"])
            lx0, ly0, lx1, ly1 = panel["level_box"]
            canvas.coords(panel["level"], lx0, ly1 - fractions[-1] * (ly1 - ly0), lx1, ly1)
            canvas.itemconfigure(panel["level"], fill=STALE if stale else ACCENT)
            highlighted = channel == strongest and not stale
            canvas.itemconfigure(panel["value"], text=format_signal(latest[channel], low, high),
                                 fill=TRACE if highlighted else TEXT_MUTED,
                                 font=(FONT, 10, "bold") if highlighted else (FONT, 10))
            canvas.itemconfigure(panel["frame"], outline=ACCENT if highlighted else BORDER,
                                 width=2 if highlighted else 1)

    def trace_point_counts(self) -> list[int]:
        """Visible trace points per sensor (for automated GUI checks)."""
        counts = []
        for panel in self._panels:
            total = 0
            for item in panel["lines"]:
                if self.canvas.itemcget(item, "state") != "hidden":
                    total += len(self.canvas.coords(item)) // 2
            counts.append(total)
        return counts
