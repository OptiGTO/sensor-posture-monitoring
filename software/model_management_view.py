"""Searchable model review table with explicit multi-selection and bulk actions."""
from __future__ import annotations

import getpass
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

from src.model_management import (
    STATUS_LABELS, eligible, evaluation_summary, identity, ordered_models, review_models,
)
from src.ui_korean import ko


class ModelManagementView(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.registry = app.registry
        self.models = []
        self.summaries = {}
        self.rows = {}
        self.active = None
        self.sort_key = "created"
        self.descending = True
        self.query = tk.StringVar()
        self.status_filter = tk.StringVar(value="전체 상태")
        self.window_filter = tk.StringVar(value="전체 창")
        self.summary = tk.StringVar()
        self.selection_summary = tk.StringVar(value="모델을 선택하세요.")
        self.result_text = tk.StringVar()
        self.active_text = tk.StringVar()
        self.busy = False
        ttk.Label(self, textvariable=self.active_text, wraplength=760, font=("Apple SD Gothic Neo", 13, "bold")).pack(anchor="w", pady=(8, 10))
        filters = ttk.Frame(self); filters.pack(fill="x")
        ttk.Label(filters, text="검색").pack(side="left")
        search = ttk.Entry(filters, textvariable=self.query, width=14)
        search.pack(side="left", padx=(6, 8), fill="x", expand=True)
        statuses = ["전체 상태", "승인 가능", *[label for key, label in STATUS_LABELS.items() if key != "revoked"]]
        ttk.Combobox(filters, textvariable=self.status_filter, values=statuses, state="readonly", width=14).pack(side="left", padx=4)
        self.window_combo = ttk.Combobox(filters, textvariable=self.window_filter, state="readonly", width=10)
        self.window_combo.pack(side="left", padx=4)
        ttk.Button(filters, text="초기화", command=self.reset_filters).pack(side="left", padx=4)
        ttk.Button(filters, text="새로고침", command=self.reload).pack(side="left")
        ttk.Label(self, textvariable=self.summary).pack(anchor="w", pady=(8, 4))
        table = ttk.Frame(self); table.pack(fill="both", expand=True)
        columns = ("check", "name", "window", "status", "accuracy", "macro_f1", "source", "created")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", selectmode="extended", height=11)
        definitions = (("check", "선택", 44), ("name", "모델 · 버전", 260), ("window", "창", 60),
                       ("status", "상태", 125), ("accuracy", "Accuracy", 90), ("macro_f1", "Macro F1", 90),
                       ("source", "평가 기준", 110), ("created", "생성 시각", 165))
        for key, label, width in definitions:
            self.tree.column(key, width=width, minwidth=width, stretch=key == "name", anchor="w" if key == "name" else "center")
            self.tree.heading(key, text=label, command=(lambda key=key: self.sort(key)) if key not in {"check", "source"} else lambda: None)
        self.heading_labels = dict((key, label) for key, label, _ in definitions)
        vertical = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        horizontal = ttk.Scrollbar(table, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        table.columnconfigure(0, weight=1); table.rowconfigure(0, weight=1)
        self.tree.tag_configure("active", foreground="#176044")
        self.tree.tag_configure("failed", foreground="#a33b2f")
        self.tree.bind("<<TreeviewSelect>>", self.update_actions)
        self.tree.bind("<Button-1>", self.click_checkbox)
        self.tree.bind("<space>", self.toggle_focused)
        self.tree.bind("<Double-1>", self.double_click)
        self.tree.bind("<Return>", lambda _event: self.show_detail())
        self.tree.bind("<Command-a>", lambda _event: self.select_visible())
        self.tree.bind("<Control-a>", lambda _event: self.select_visible())
        selection = ttk.Frame(self); selection.pack(fill="x", pady=6)
        for text, command in (("목록 전체 선택", self.select_visible), ("승인 가능만 선택", self.select_ready), ("선택 해제", self.clear_selection)):
            ttk.Button(selection, text=text, command=command).pack(side="left", padx=(0, 5))
        ttk.Label(self, textvariable=self.selection_summary, wraplength=780).pack(anchor="w", pady=(0, 5))
        actions = ttk.Frame(self); actions.pack(fill="x")
        self.buttons = {}
        for key, label in (("approve", "선택 승인"), ("reject", "선택 거절"), ("revoke", "선택 폐기")):
            button = ttk.Button(actions, text=label, command=lambda key=key: self.run_action(key))
            button.pack(side="left", padx=(0, 5)); self.buttons[key] = button
        self.activate_button = ttk.Button(actions, text="기본 모델로 저장", command=self.activate)
        self.activate_button.pack(side="left", padx=(0, 5))
        self.detail_button = ttk.Button(actions, text="자세히 보기", command=self.show_detail)
        self.detail_button.pack(side="left")
        ttk.Button(actions, text="이전 모델 복원", command=app.rollback_model).pack(side="right")
        ttk.Label(self, textvariable=self.result_text, foreground="#176044", wraplength=780).pack(anchor="w", pady=5)
        ttk.Label(self, text="열 제목으로 정렬 · 체크박스 / Shift / ⌘·Ctrl로 다중 선택", foreground="#596575").pack(anchor="w")
        ttk.Label(self, text="성능은 독립 최종 평가 기준입니다. 데이터·평가 피험자가 다르면 직접 비교에 주의하세요.", wraplength=780, foreground="#596575").pack(anchor="w", pady=(3, 8))
        for variable in (self.query, self.status_filter, self.window_filter):
            variable.trace_add("write", lambda *_args: self.render())
        self.reload()

    def reload(self):
        if self.busy:
            return
        selected = {identity(model) for model in self.selected()}
        self.models = [model for model in self.registry.management_models() if model.revocation_status != "revoked"]
        self.invalid = [result for result in self.registry.last_results if result.descriptor is None]
        self.summaries = {identity(model): evaluation_summary(model) for model in self.models}
        self.active = self.registry.active_model([model for model in self.models if model.production_selectable])
        self.active_text.set(f"현재 활성 모델: {ko(self.active.name) + ' · v' + self.active.version if self.active else '없음'}")
        self.window_combo.configure(values=["전체 창", *sorted({model.window_label for model in self.models})])
        self.render(selected)

    def reset_filters(self):
        self.query.set(""); self.status_filter.set("전체 상태"); self.window_filter.set("전체 창")

    def render(self, selected=None):
        if self.busy:
            return
        if selected is None:
            selected = {identity(model) for model in self.selected()}
        query = self.query.get().strip().casefold()
        state = self.status_filter.get()
        models = [model for model in self.models
                  if (not query or query in f"{ko(model.name)} {model.name} {model.version} {model.artifact_id} {model.model_family}".casefold())
                  and (self.window_filter.get() == "전체 창" or model.window_label == self.window_filter.get())
                  and (state == "전체 상태" or (state == "승인 가능" and eligible(model, "approve"))
                       or state == STATUS_LABELS[model.management_status])]
        models = ordered_models(models, self.summaries, self.sort_key, self.descending)
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.rows = {}
        for index, model in enumerate(models):
            iid = f"model-{index}"; self.rows[iid] = model
            score = self.summaries[identity(model)]
            active = self.active is not None and identity(self.active) == identity(model)
            status = "현재 활성" if active else ("승인 가능" if eligible(model, "approve") else STATUS_LABELS[model.management_status])
            self.tree.insert("", "end", iid=iid, values=("☐", f"{ko(model.name)} · v{model.version}", model.window_label,
                status, self.percent(score.accuracy), self.percent(score.macro_f1), score.source,
                model.created_at if model.created_at != "legacy" else "미기록"),
                tags=("active",) if active else (("failed",) if model.management_status == "failed" else ()))
        self.tree.selection_set([iid for iid, model in self.rows.items() if identity(model) in selected])
        # Unreadable manifests stay visible and inspectable but cannot be reviewed.
        if state in {"전체 상태", STATUS_LABELS["failed"]} and self.window_filter.get() == "전체 창":
            for index, result in enumerate(getattr(self, "invalid", [])):
                if not query or query in str(result.reason).casefold():
                    self.tree.insert("", "end", iid=f"invalid-{index}", values=("—", "읽을 수 없는 매니페스트", "—", "검증 실패", "—", "—", "—", "—"), tags=("failed",))
        count = len(self.tree.get_children())
        self.summary.set(f"표시 {count}개 / 전체 {len(self.models) + len(getattr(self, 'invalid', []))}개 · 승인 가능 {sum(eligible(model, 'approve') for model in models)}개"
                         + (" · 검색 조건에 맞는 모델이 없습니다." if not count else ""))
        for key, label in self.heading_labels.items():
            self.tree.heading(key, text=label + ((" ▼" if self.descending else " ▲") if key == self.sort_key else ""))
        self.update_actions()

    @staticmethod
    def percent(value):
        return "미기록" if value is None else f"{value:.2%}"

    def sort(self, key):
        self.descending = not self.descending if key == self.sort_key else key in {"accuracy", "macro_f1", "created"}
        self.sort_key = key
        self.render()

    def selected(self):
        return [self.rows[iid] for iid in self.tree.selection() if iid in self.rows]

    def select_visible(self):
        self.tree.selection_set(list(self.rows)); self.update_actions(); return "break"

    def select_ready(self):
        self.tree.selection_set([iid for iid, model in self.rows.items() if eligible(model, "approve")]); self.update_actions()

    def clear_selection(self):
        self.tree.selection_remove(self.tree.selection()); self.update_actions()

    def click_checkbox(self, event):
        if self.tree.identify_region(event.x, event.y) == "cell" and self.tree.identify_column(event.x) == "#1":
            iid = self.tree.identify_row(event.y)
            if iid in self.rows:
                self.tree.focus_set(); self.tree.focus(iid)
                self.tree.selection_toggle(iid); self.update_actions()
            return "break"

    def toggle_focused(self, _event=None):
        if self.tree.focus() in self.rows:
            self.tree.selection_toggle(self.tree.focus()); self.update_actions()
        return "break"

    def double_click(self, event):
        if self.tree.identify_region(event.x, event.y) == "cell" and self.tree.identify_column(event.x) != "#1":
            self.show_detail()
        return "break"

    def update_actions(self, _event=None):
        models = self.selected()
        selected = set(self.tree.selection())
        for iid in self.rows:
            self.tree.set(iid, "check", "☑" if iid in selected else "☐")
        counts = {action: sum(eligible(model, action) for model in models) for action in self.buttons}
        for action, button in self.buttons.items():
            label = {"approve": "승인", "reject": "거절", "revoke": "폐기"}[action]
            button.configure(text=f"선택 {label} ({counts[action]})", state="normal" if counts[action] and not self.busy else "disabled")
        self.activate_button.configure(state="normal" if len(models) == len(selected) == 1 and models[0].production_selectable and not self.busy else "disabled")
        self.detail_button.configure(state="normal" if len(selected) == 1 and not self.busy else "disabled")
        self.selection_summary.set(f"선택 {len(models)}개 · 승인 가능 {counts['approve']}개 · 거절 가능 {counts['reject']}개 · 작업 가능한 모델만 처리합니다."
                                   if models else "모델을 선택하세요. 필터에서 숨겨진 모델은 선택이 해제됩니다.")

    def show_detail(self):
        selected = self.tree.selection()
        if len(selected) != 1:
            return
        model = self.rows.get(selected[0])
        if model is None:
            result = self.invalid[int(selected[0].split("-")[1])]
            messagebox.showerror("매니페스트 오류", ko(result.reason or "매니페스트를 읽을 수 없습니다."), parent=self)
            return
        score = self.summaries[identity(model)]
        dialog = tk.Toplevel(self); dialog.title("모델 검토 정보"); dialog.geometry("700x560")
        content = tk.Text(dialog, wrap="word", padx=18, pady=18)
        scroll = ttk.Scrollbar(dialog, command=content.yview); scroll.pack(side="right", fill="y")
        content.configure(yscrollcommand=scroll.set); content.pack(fill="both", expand=True)
        lines = [f"{ko(model.name)} · v{model.version}", f"ID: {model.artifact_id}",
                 f"상태: {STATUS_LABELS[model.management_status]}",
                 f"Runtime 검증: {ko(model.validation_status)} · 배포 검증: {ko(model.deployment_validation_status)}",
                 f"승인: {ko(model.approval_status)}", f"사유: {ko(model.lifecycle_reason) if model.lifecycle_reason else '없음'}",
                 f"Accuracy: {self.percent(score.accuracy)} · Macro F1: {self.percent(score.macro_f1)} ({score.source})",
                 score.detail, f"생성 시각: {model.created_at}", f"창: {model.window_label}",
                 f"매니페스트: {model.manifest_path}", f"모델 SHA-256: {model.artifact_hash}"]
        content.insert("1.0", "\n\n".join(lines)); content.configure(state="disabled")
        ttk.Button(dialog, text="닫기", command=dialog.destroy).pack(pady=8)

    def activate(self):
        models = self.selected()
        if len(models) == 1 and models[0].production_selectable:
            try:
                self.registry.activate(models[0])
            except (OSError, ValueError) as error:
                messagebox.showerror("기본 모델 저장 실패", ko(error), parent=self); return
            self.reload(); self.result_text.set("기본 모델을 저장했습니다.")

    def run_action(self, action):
        if self.busy:
            return
        models = self.selected()
        targets = [model for model in models if eligible(model, action)]
        if not targets:
            return
        reason = "사용자가 모델 관리에서 거절함"
        if action == "revoke":
            names = "\n".join(f"• {ko(model.name)} · v{model.version}" for model in targets[:6])
            if len(targets) > 6:
                names += f"\n외 {len(targets) - 6}개"
            active_warning = "\n현재 활성 모델이 포함되어 있습니다. 다음 측정의 기본 모델이 변경될 수 있습니다." if self.active and any(identity(model) == identity(self.active) for model in targets) else ""
            reason = simpledialog.askstring("선택 모델 폐기", f"{len(targets)}개 모델을 목록에서 숨깁니다. 파일과 이력은 보존됩니다.\n{names}{active_warning}\n\n공통 폐기 사유", parent=self)
            if not reason or not reason.strip():
                return
        self.busy = True; self.update_actions()
        progress = tk.Toplevel(self); progress.title("모델 처리 중"); progress.transient(self.winfo_toplevel())
        progress.protocol("WM_DELETE_WINDOW", lambda: None)
        label = ttk.Label(progress, text=f"{len(models)}개 모델의 현재 상태 확인 중…", padding=20); label.pack()
        bar = ttk.Progressbar(progress, maximum=len(models), length=340); bar.pack(padx=20, pady=(0, 20))
        progress.grab_set()
        events = queue.Queue()
        actor = getpass.getuser()
        def work():
            try:
                result = review_models(self.registry, models, action, actor=actor, reason=reason,
                                       progress=lambda done, total: events.put(("progress", (done, total))))
                events.put(("done", result))
            except Exception as error:
                events.put(("error", str(error)))
        threading.Thread(target=work, daemon=True).start()
        def poll():
            try:
                while True:
                    kind, value = events.get_nowait()
                    if kind == "progress":
                        done, total = value; bar.configure(value=done); label.configure(text=f"모델 처리 중 · {done}/{total}")
                        continue
                    progress.grab_release(); progress.destroy(); self.busy = False
                    self.reload()
                    if kind == "error":
                        messagebox.showerror("모델 처리 실패", ko(value), parent=self)
                    else:
                        action_label = {"approve": "승인", "reject": "거절", "revoke": "폐기"}[action]
                        text = f"{action_label} 완료 {len(value.succeeded)}개 · 건너뜀 {len(value.skipped)}개 · 실패 {len(value.failed)}개"
                        self.result_text.set(text)
                        # Keep only unresolved visible models selected, so retry does not repeat successes.
                        unresolved = {identity(model) for model, _ in value.failed}
                        self.tree.selection_set([iid for iid, model in self.rows.items() if identity(model) in unresolved])
                        self.update_actions()
                        if value.failed or value.skipped:
                            self.show_batch_result(text, value)
                    return
            except queue.Empty:
                self.after(80, poll)
        self.after(80, poll)

    def show_batch_result(self, title, result):
        dialog = tk.Toplevel(self); dialog.title("일괄 처리 결과"); dialog.geometry("660x400")
        ttk.Label(dialog, text=title, padding=12).pack(anchor="w")
        content = tk.Text(dialog, wrap="word", padx=12, pady=10)
        scroll = ttk.Scrollbar(dialog, command=content.yview); scroll.pack(side="right", fill="y")
        content.configure(yscrollcommand=scroll.set); content.pack(fill="both", expand=True)
        for status, items in (("실패", result.failed), ("건너뜀", result.skipped)):
            for model, reason in items:
                content.insert("end", f"[{status}] {ko(model.name)} · v{model.version}\n{ko(reason)}\n\n")
        content.configure(state="disabled")
        ttk.Button(dialog, text="닫기", command=dialog.destroy).pack(pady=8)
