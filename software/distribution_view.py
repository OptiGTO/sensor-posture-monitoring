"""Read-only window showing one recording's sensor distribution map."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.record_distribution import (
    default_comparison, distribution_figure, load_distribution, similarity_summary,
)
from src.ui_korean import ko

NO_COMPARISON = "비교 안 함"


class DistributionWindow(tk.Toplevel):
    def __init__(self, parent, record, records):
        super().__init__(parent)
        self.title("녹화 분포맵")
        self.geometry("1120x860")
        self.minsize(820, 640)
        self.distribution = load_distribution(record, records)
        self.canvas = None
        self.figure = None
        names = [name for name in self.distribution.references if name != self.distribution.posture]
        initial = default_comparison(self.distribution)
        self.comparison = tk.StringVar(self, value=initial or NO_COMPARISON)
        self.start = tk.StringVar(self, value="0")
        self.end = tk.StringVar(self, value=f"{self.distribution.duration:.1f}")
        self.summary = tk.StringVar(self)

        controls = ttk.Frame(self, padding=(10, 8)); controls.pack(fill="x")
        ttk.Label(controls, text="비교 자세").pack(side="left")
        self.comparison_selector = ttk.Combobox(controls, textvariable=self.comparison, state="readonly", width=16,
                                                values=[NO_COMPARISON, *names])
        self.comparison_selector.pack(side="left", padx=(4, 14))
        self.comparison_selector.bind("<<ComboboxSelected>>", self.refresh)
        ttk.Label(controls, text="구간(초)").pack(side="left")
        for variable in (self.start, self.end):
            entry = ttk.Entry(controls, textvariable=variable, width=7)
            entry.pack(side="left", padx=2)
            entry.bind("<Return>", self.refresh)
        ttk.Button(controls, text="구간 적용", command=self.refresh).pack(side="left", padx=4)
        ttk.Button(controls, text="전체 구간", command=self.reset_interval).pack(side="left")
        ttk.Button(controls, text="닫기", command=self.destroy).pack(side="right")
        self.save_button = ttk.Button(controls, text="PNG 저장", command=self.save)
        self.save_button.pack(side="right", padx=6)
        label = ttk.Label(self, textvariable=self.summary, wraplength=1080, padding=(10, 0))
        label.pack(fill="x")
        self.bind("<Configure>", lambda event: label.configure(wraplength=max(400, event.width - 30)) if event.widget is self else None)
        self.host = ttk.Frame(self); self.host.pack(fill="both", expand=True, padx=8, pady=8)
        self.refresh()

    def interval(self):
        return float(self.start.get()), float(self.end.get())

    def reset_interval(self):
        self.start.set("0"); self.end.set(f"{self.distribution.duration:.1f}")
        self.refresh()

    def refresh(self, _event=None):
        try:
            start, end = self.distribution.clamp(*self.interval())
            comparison = None if self.comparison.get() == NO_COMPARISON else self.comparison.get()
            figure = distribution_figure(self.distribution, comparison, start, end)
        except ValueError as error:
            messagebox.showerror("구간 확인", ko(error) or "구간을 숫자로 입력하세요.", parent=self)
            return
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.figure.clear()
        self.figure = figure
        self.canvas = FigureCanvasTkAgg(figure, master=self.host)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()
        self.summary.set(similarity_summary(self.distribution, start, end))

    def save(self):
        path = filedialog.asksaveasfilename(parent=self, title="분포맵 저장", defaultextension=".png",
                                            initialfile=f"{self.distribution.path.stem}_distribution.png",
                                            filetypes=[("PNG", "*.png")])
        if path:
            self.figure.savefig(path, dpi=130)
            messagebox.showinfo("저장 완료", f"분포맵을 저장했습니다.\n{path}", parent=self)
