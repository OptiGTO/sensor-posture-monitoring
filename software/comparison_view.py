"""Interactive training-result charts embedded in the existing Tk application."""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.comparison_charts import (
    MODEL_METRICS, POSTURE_METRICS, SPLITS, confusion_chart, default_split_label, model_chart,
    model_label, model_value, posture_chart,
)
from src.ui_korean import ko, parameters_ko
from src.experiment_workflow import adapt_report


class ComparisonView(ttk.Frame):
    def __init__(self, parent, reports, clone_callback=None):
        super().__init__(parent)
        self.clone_callback = clone_callback
        self.source_report = {}
        self.report = {}
        self.rows = []
        self.canvas = None
        self.report_choice = tk.StringVar(self, value=str(reports[0]) if reports else "")
        self.split_choice = tk.StringVar(self, value="검증")
        self.model_choice = tk.StringVar(self)
        self.metric_choice = tk.StringVar(self, value=next(iter(MODEL_METRICS)))
        self.posture_metric_choice = tk.StringVar(self, value="모든 지표")
        self.summary = tk.StringVar(self)
        self.note = tk.StringVar(self)

        ttk.Label(self, text="학습 실행 결과").pack(anchor="w")
        self.report_selector = ttk.Combobox(self, textvariable=self.report_choice, values=list(map(str, reports)), state="readonly")
        self.report_selector.pack(fill="x", pady=(3, 5))
        self.report_selector.bind("<<ComboboxSelected>>", self.load_report)
        summary_label = ttk.Label(self, textvariable=self.summary, wraplength=780)
        summary_label.pack(fill="x", pady=(0, 10))
        self.bind("<Configure>", lambda event: summary_label.configure(wraplength=max(300, event.width - 10)))

        scope_bar = ttk.Frame(self); scope_bar.pack(fill="x", pady=(0, 8))
        ttk.Label(scope_bar, text="비교 대상").pack(side="left", padx=(0, 6))
        self.scope_choice = tk.StringVar(self, value="개별 모델")
        self.scope_selector = ttk.Combobox(scope_bar, textvariable=self.scope_choice,
            values=["개별 모델"], state="readonly", width=30)
        self.scope_selector.pack(side="left")
        self.scope_selector.bind("<<ComboboxSelected>>", self.change_scope)
        controls = ttk.Frame(self)
        controls.pack(fill="x", pady=(0, 10))
        ttk.Label(controls, text="평가 데이터").pack(side="left", padx=(0, 6))
        self.split_selector = ttk.Combobox(controls, textvariable=self.split_choice, values=list(SPLITS), state="readonly", width=12)
        self.split_selector.pack(side="left")
        self.split_selector.bind("<<ComboboxSelected>>", self.refresh)
        ttk.Label(controls, text="자세 분석 모델").pack(side="left", padx=(18, 6))
        self.model_selector = ttk.Combobox(controls, textvariable=self.model_choice, state="readonly", width=32)
        self.model_selector.pack(side="left", fill="x", expand=True)
        self.model_selector.bind("<<ComboboxSelected>>", self.refresh)

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)
        self.panels = []
        for title in ("모델 간 비교", "자세별 비교", "혼동 행렬", "상세 수치"):
            panel = ttk.Frame(self.tabs, padding=10)
            self.tabs.add(panel, text=title)
            self.panels.append(panel)
        self.tabs.bind("<<NotebookTabChanged>>", self.refresh)

        metric_bar = ttk.Frame(self.panels[0]); metric_bar.pack(fill="x")
        ttk.Label(metric_bar, text="비교 지표").pack(side="left", padx=(0, 8))
        self.metric_selector = ttk.Combobox(metric_bar, textvariable=self.metric_choice, values=list(MODEL_METRICS), state="readonly", width=27)
        self.metric_selector.pack(side="left")
        self.metric_selector.bind("<<ComboboxSelected>>", self.refresh)
        posture_bar = ttk.Frame(self.panels[1]); posture_bar.pack(fill="x")
        ttk.Label(posture_bar, text="자세별 지표").pack(side="left", padx=(0, 8))
        self.posture_metric_selector = ttk.Combobox(posture_bar, textvariable=self.posture_metric_choice,
                                                  values=["모든 지표", *POSTURE_METRICS], state="readonly", width=27)
        self.posture_metric_selector.pack(side="left")
        self.posture_metric_selector.bind("<<ComboboxSelected>>", self.refresh)
        for panel, text in (
            (self.panels[0], "같은 학습 실행의 모델·분석 창을 비교합니다. 정확도와 F1의 축은 0~100%입니다."),
            (self.panels[1], "정답률(Recall): 실제 해당 자세를 맞힌 비율 · Precision: 해당 자세로 예측한 것 중 맞힌 비율\nF1: Precision과 Recall의 조화 평균 · n: 평가한 분석 창 수 · 미기록: 저장된 값 또는 평가 표본 없음"),
            (self.panels[2], "행은 실제 자세, 열은 예측 자세입니다. 대각선이 정답이며, 각 행의 비율 합계는 100%입니다."),
        ):
            label = ttk.Label(panel, text=text, wraplength=760)
            label.pack(fill="x", pady=(8, 6))
            panel.bind("<Configure>", lambda event, item=label: item.configure(wraplength=max(300, event.width - 24)))
        self.chart_hosts = []
        for panel in self.panels[:3]:
            host = ttk.Frame(panel)
            host.pack(fill="both", expand=True)
            self.chart_hosts.append(host)

        table_frame = ttk.Frame(self.panels[3]); table_frame.pack(fill="x")
        columns = ("모델", "정확도", "Macro F1", "최저 F1", "추론 ms", "크기 KiB")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=6, selectmode="browse")
        for key in columns:
            self.table.heading(key, text=key)
            self.table.column(key, width=210 if key == "모델" else 90, minwidth=75, anchor="w")
        self.table.pack(side="left", fill="x", expand=True)
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        scroll.pack(side="right", fill="y"); self.table.configure(yscrollcommand=scroll.set)
        self.table.bind("<<TreeviewSelect>>", self.select_table_model)
        detail_frame = ttk.Frame(self.panels[3]); detail_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.detail = tk.Text(detail_frame, height=19, wrap="word", state="disabled", font=("Apple SD Gothic Neo", 11))
        self.detail.pack(side="left", fill="both", expand=True)
        detail_scroll = ttk.Scrollbar(detail_frame, orient="vertical", command=self.detail.yview)
        detail_scroll.pack(side="right", fill="y"); self.detail.configure(yscrollcommand=detail_scroll.set)

        actions = ttk.Frame(self); actions.pack(fill="x", pady=8)
        self.clone_button = ttk.Button(actions, text="설정 복제해 새 실험",
            command=lambda: self.clone_callback(Path(self.report_choice.get())) if self.clone_callback else None)
        if clone_callback is not None:
            self.clone_button.pack(side="left", padx=(0, 8))
        self.export_button = ttk.Button(actions, text="현재 그래프 PNG 저장", command=self.save_chart)
        self.export_button.pack(side="right")
        ttk.Label(actions, textvariable=self.note).pack(side="left")
        self.load_report()

    @property
    def split(self):
        return SPLITS[self.split_choice.get()]

    @property
    def selected_row(self):
        index = self.model_selector.current()
        return self.rows[index] if 0 <= index < len(self.rows) else None

    def load_report(self, _event=None):
        # Clear the previous run even when the next file is incomplete/unreadable.
        self.source_report, self.report, self.rows = {}, {}, []
        self.scope_choice.set("개별 모델")
        self.scope_selector.configure(values=["개별 모델"])
        self.note.set("")
        self.clone_button.configure(state="disabled")
        try:
            report = adapt_report(json.loads(Path(self.report_choice.get()).read_text(encoding="utf-8")))
            if not isinstance(report, dict) or not isinstance(report.get("models", []), list):
                raise ValueError("모델 비교 보고서 형식이 올바르지 않습니다.")
            if any(not isinstance(row, dict) for row in report.get("models", [])):
                raise ValueError("모델 결과 형식이 올바르지 않습니다.")
            if report.get("selected") is not None and not isinstance(report["selected"], dict):
                raise ValueError("선택 모델 형식이 올바르지 않습니다.")
            labels = report.get("labels")
            if labels is not None and (not isinstance(labels, list) or any(not isinstance(label, str) for label in labels)):
                raise ValueError("자세 목록 형식이 올바르지 않습니다.")
            for row in report.get("models", []):
                for split in SPLITS.values():
                    metrics = row.get(split)
                    if metrics is None:
                        continue
                    if not isinstance(metrics, dict):
                        raise ValueError("평가 지표 형식이 올바르지 않습니다.")
                    for key in ("per_class", "per_class_f1"):
                        values = metrics.get(key)
                        if values is not None and not isinstance(values, dict):
                            raise ValueError("자세별 지표 형식이 올바르지 않습니다.")
                    if any(not isinstance(value, dict) for value in (metrics.get("per_class") or {}).values()):
                        raise ValueError("자세별 평가 형식이 올바르지 않습니다.")
            path = Path(self.report_choice.get())
            run = path.parent.parent if path.name == "model_comparison.json" else path.parent
            if (run / ".incomplete").exists() or (run / "artifacts" / ".incomplete").exists():
                report = {**report, "status": "failed"}
            if self.clone_callback and (run / "experiment.json").is_file():
                self.clone_button.configure(state="normal")
            if report.get("configuration_type") == "personal":
                self.scope_choice.set("개인 표준화")
                self.scope_selector.configure(values=["개인 표준화"])
            self.source_report = report
            self.report = report
            if report.get("soft_voting", {}).get("models"):
                self.scope_selector.configure(values=["개별 모델", "동일 가중 Soft voting"])
            self.rows = report.get("models") or []
            self.split_choice.set(default_split_label(report))
            self.summary.set(self._selection_summary(report))
            if report.get("summary_text"):
                self.summary.set(report["summary_text"] + " · " + ko(report.get("status", "none")))
            if not self.rows:
                self.note.set("완료된 모델 결과가 없습니다.")
        except (OSError, ValueError) as error:
            self.summary.set(f"보고서를 읽을 수 없습니다: {ko(error)}")
        self.model_selector.configure(values=[model_label(row) for row in self.rows])
        self.model_choice.set("")
        if self.rows:
            selected_index = next((i for i, row in enumerate(self.rows) if row.get("selected")), 0)
            self.model_selector.current(selected_index)
        self.refresh()

    def change_scope(self, _event=None):
        soft = self.source_report.get("soft_voting") or {}
        if self.scope_choice.get() == "동일 가중 Soft voting" and soft.get("models"):
            self.report = {**self.source_report, "models": soft["models"]}
            self.split_choice.set("검증")  # soft voting is evaluated on aligned split points, not LOSO
            counts = soft.get("counts", {})
            self.summary.set(f"동일 가중 · {len(soft['members'])}개 모델 · 실험용 (측정 모델 등록 없음)\n"
                f"공통 시점: 검증 {counts.get('validation', 0)}개 / 최종 {counts.get('final_evaluation', 0)}개 · 최종 평가 전 구성 확정")
        else:
            self.report = self.source_report
            self.split_choice.set(default_split_label(self.report))
            self.summary.set(self._selection_summary(self.report))
        if self.report.get("summary_text"):
            self.summary.set(self.report["summary_text"] + " · " + ko(self.report.get("status", "none")))
        self.rows = self.report.get("models") or []
        self.model_selector.configure(values=[model_label(row) for row in self.rows])
        self.model_choice.set("")
        if self.rows:
            self.model_selector.current(len(self.rows)-1 if self.scope_choice.get() != "개별 모델" else 0)
        self.refresh()

    @staticmethod
    def _selection_summary(report):
        selection = report.get("selected") or {}
        selected = model_label(selection) if selection else "없음"
        basis = ("LOSO(개발 피험자 한 명씩 제외)로 선택" if report.get("selection_metric") == "loso"
                 else "검증 결과로 선택")
        note = ("매개변수는 검증 분할로 고르고, 모델·분석 창은 LOSO Macro F1로 확정했습니다. "
                if report.get("selection_metric") == "loso" else
                "이 실행에는 LOSO 결과가 없습니다(단일 검증 분할 선택). ")
        return (f"{basis}: {selected} · 학습 상태: {ko(report.get('status', 'none'))}\n"
                f"{note}최종 평가는 모델 선택에 사용하지 않습니다. 지표는 분석 창 단위입니다.")

    def select_table_model(self, _event=None):
        selection = self.table.selection()
        if selection and int(selection[0]) != self.model_selector.current():
            self.model_selector.current(int(selection[0]))
            self.refresh()

    def refresh(self, _event=None):
        tab = self.tabs.index(self.tabs.select())
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas.figure.clear()
            self.canvas = None
        self.export_button.configure(state="normal" if self.rows and tab < 3 else "disabled")
        if tab == 3:
            self.refresh_detail()
            return
        if tab == 0:
            figure = model_chart(self.report, self.split, self.metric_choice.get())
        elif tab == 1:
            figure = posture_chart(self.report, self.selected_row, self.split, self.posture_metric_choice.get())
        else:
            figure = confusion_chart(self.report, self.selected_row, self.split)
        self.canvas = FigureCanvasTkAgg(figure, master=self.chart_hosts[tab])
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

    def refresh_detail(self):
        self.table.delete(*self.table.get_children())
        for index, row in enumerate(self.rows):
            values = []
            for metric in ("전체 정확도 (Accuracy)", "Macro F1", "최저 자세 F1", "추론 시간 (중앙값)", "모델 크기"):
                value = model_value(row, self.split, metric)
                values.append("미기록" if value is None else f"{value:,.2f}" + ("%" if MODEL_METRICS[metric][3] else ""))
            self.table.insert("", "end", iid=str(index), values=(model_label(row), *values))
        row = self.selected_row
        text = "저장된 모델 결과가 없습니다."
        if row is not None:
            item = str(self.model_selector.current())
            self.table.selection_set(item); self.table.see(item)
            text = f"{model_label(row)} · 매개변수: {parameters_ko(row.get('params'))}\n\n"
            for split in SPLITS.values():
                text += f"{ko(split)}\n{json.dumps(row.get(split) or {}, ensure_ascii=False, indent=2)}\n\n"
        text += f"\n보고서: {self.report_choice.get()}"
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end"); self.detail.insert("1.0", text)
        self.detail.configure(state="disabled")

    def save_chart(self):
        if self.canvas is None or not self.rows:
            return
        tab = self.tabs.index(self.tabs.select())
        filename = f"{Path(self.report_choice.get()).parent.parent.name}_{self.scope_choice.get()}_{self.split}_{('models', 'postures', 'confusion')[tab]}.png"
        path = filedialog.asksaveasfilename(parent=self, title="현재 그래프 저장", initialfile=filename,
                                          defaultextension=".png", filetypes=[("PNG 이미지", "*.png")])
        if path:
            try:
                self.canvas.figure.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
                self.note.set(f"저장 완료: {Path(path).name}")
            except (OSError, ValueError) as error:
                messagebox.showerror("그래프 저장 실패", str(error), parent=self)
