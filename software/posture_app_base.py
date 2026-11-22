#!/usr/bin/env python3
"""국문 GUI로 실행하는 압력센서 자세 측정·관리 프로그램."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
import tempfile
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
import tkinter as tk

import numpy as np

PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT))
from src.app_paths import dataset_directories, writable_data_directory  # noqa: E402
from src.app_products import select_product  # noqa: E402

PRODUCT = select_product()

_preferred_data_root = PRODUCT.state_root()
if os.environ.get("POSTURE_APP_STATE_ROOT"):
    _preferred_data_root = Path(os.environ["POSTURE_APP_STATE_ROOT"])
try:
    _preferred_data_root.mkdir(parents=True, exist_ok=True)
    STATE_ROOT = _preferred_data_root
except OSError:
    STATE_ROOT = Path(tempfile.gettempdir()) / PRODUCT.state_directory_name
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
try:
    DATA_DIRECTORY = writable_data_directory(STATE_ROOT / "data", STATE_ROOT)
except OSError:
    STATE_ROOT = Path(tempfile.gettempdir()) / PRODUCT.state_directory_name
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIRECTORY = writable_data_directory(STATE_ROOT / "data", STATE_ROOT)
_numba_cache = STATE_ROOT / ".cache" / "numba"
_numba_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_numba_cache))
AVATAR_ASSET_DIRECTORY = PROJECT_ROOT / "software" / "assets" / "postures"

from src.app_services import (  # noqa: E402
    DatasetService, MeasurementRecorder, ModelDescriptor, ModelRegistry,
    OccupancyDetector, RiskAssessment, TrainingJob, TrainingWizardState,
    occupancy_profile_for_input,
)
from src.avatar_assets import avatar_filename, missing_avatar_assets  # noqa: E402
from src.inference_core import PosturePredictor  # noqa: E402
from src.model_runtime import SessionModelSnapshot  # noqa: E402
from src.realtime import LiveConfig, RealtimePosturePredictor  # noqa: E402
from src.personal_guidance import AlertSettings, CalibrationContext, CalibrationStore  # noqa: E402
from src.ui_korean import ko, parameters_ko
from src.personalized_model import PersonalPostureModel, available_models
from src.guidance_session import GuidanceSession  # noqa: E402
from src.session_events import SessionEventQueue  # noqa: E402
from src.signal_monitor import SignalBuffer  # noqa: E402
from src.training_windows import merge_window_selection  # noqa: E402
from software.model_management_view import ModelManagementView
from software.distribution_view import DistributionWindow
from software.signal_monitor_view import SignalMonitorView
from software.posture_realtime import replay_csv, serial_frames  # noqa: E402


from software.experiment_wizard import ExperimentWizardMixin


class PostureApplication(ExperimentWizardMixin, tk.Tk):
    """The GUI owns rendering only; one worker owns the sensor iterator."""

    # Submission build hides the experimental personal two-stage measurement
    # (retrospective Macro F1 0.4349 vs 0.5756 for the general 7-class model).
    # Registration, labeled-record linking and personal training stay available.
    SHOW_PERSONAL_HIERARCHY = False

    def __init__(self) -> None:
        super().__init__()
        self.title(PRODUCT.display_name)
        self.geometry("980x760")
        self.minsize(850, 700)
        self.configure(background="#f4f6f9")
        self.registry = ModelRegistry(PROJECT_ROOT, STATE_ROOT, use_bundled_release=bool(getattr(sys, "frozen", False)))
        self.models = self.registry.discover()
        self.selected_model = self.registry.active_model(self.models)
        self.selected_window = self.selected_model.window.default_seconds if self.selected_model else 0.0
        self.mode = tk.StringVar(value="measurement")
        self.subject = tk.StringVar()
        self.ground_truth = tk.StringVar()
        self.input_path = tk.StringVar()
        self.port = tk.StringVar(value="자동 탐색")
        self.personal_model_choice = tk.StringVar()
        self.personal_enabled = tk.BooleanVar(value=False)
        self.link_reference = tk.BooleanVar(value=False)
        self.personal_retrospective = tk.BooleanVar(value=False)
        self._session_personal_model = None
        self.guidance_user = tk.StringVar()
        self.chair_id = tk.StringVar(value="chair-1")
        self.placement_id = tk.StringVar(value="placement-1")
        self.sensor_configuration = tk.StringVar(value="pressure-4x4-v1")
        self.alert_mode = tk.StringVar(value="deployment")
        self.return_confirmation = tk.StringVar(value="2")
        self.uncertainty_tolerance = tk.StringVar(value="3")
        self._guidance_session = None
        self._capture_requested = threading.Event()
        self._capture_button = None
        self._warning_popup = None
        self.status = tk.StringVar(value="모델을 선택하세요.")
        self._messages = SessionEventQueue()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._recorder: MeasurementRecorder | None = None
        self._last_manifest: Path | None = None
        self._session_error: str | None = None
        self._closing = False
        self._session_finished = False
        self._avatar_images: dict[str, tk.PhotoImage] = {}
        self._current_posture: str | None = None
        # Display-only copy of the incoming frames for the live signal monitor.
        self._signal_buffer = SignalBuffer()
        directories = dataset_directories(
            DATA_DIRECTORY, PROJECT_ROOT,
            executable=Path(sys.executable) if getattr(sys, "frozen", False) else None,
            isolated=bool(os.environ.get("POSTURE_APP_STATE_ROOT")))
        self.dataset_service = DatasetService(
            STATE_ROOT, data_directory=DATA_DIRECTORY, additional_data_directories=directories[1:])
        self.selected_records = []
        self.training_wizard_state = TrainingWizardState()
        self.training_job: TrainingJob | None = None
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.tk.call("package", "require", "msgcat")
        self.tk.call("::msgcat::mclocale", "ko")
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="종료", command=self._close)
        menu.add_cascade(label="파일", menu=file_menu)
        edit_menu = tk.Menu(menu, tearoff=False)
        for label, event in (("잘라내기", "<<Cut>>"), ("복사", "<<Copy>>"), ("붙여넣기", "<<Paste>>"), ("모두 선택", "<<SelectAll>>")):
            edit_menu.add_command(label=label, command=lambda item=event: (self.focus_get() or self).event_generate(item))
        menu.add_cascade(label="편집", menu=edit_menu)
        self.configure(menu=menu)
        self.bind("<MouseWheel>", self._scroll_page)
        self.show_model_select()

    def clear(self) -> ttk.Frame:
        self.unbind("<Escape>")
        for child in self.winfo_children():
            if isinstance(child, tk.Menu):
                continue
            child.destroy()
        outer = ttk.Frame(self); outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0, background="#f4f6f9")
        scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y"); canvas.pack(side="left", fill="both", expand=True)
        frame = ttk.Frame(canvas, padding=20)
        item = canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(item, width=event.width))
        self._scroll_canvas = canvas
        return frame

    def _scroll_page(self, event):
        if event.widget.winfo_class() in {"Text", "Treeview", "TCombobox"}:
            return
        canvas = getattr(self, "_scroll_canvas", None)
        if canvas and canvas.winfo_exists() and canvas.yview() != (0., 1.):
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def heading(self, parent: ttk.Frame, title: str, subtitle: str = "") -> None:
        ttk.Label(parent, text=title, font=("Apple SD Gothic Neo", 24, "bold")).pack(anchor="w")
        if subtitle:
            label = ttk.Label(parent, text=subtitle, font=("Apple SD Gothic Neo", 12), wraplength=780)
            label.pack(fill="x", pady=(5, 22))
            parent.bind("<Configure>", lambda event: label.configure(wraplength=max(200, event.width - 40)), add="+")

    def show_model_select(self) -> None:
        self.models = self.registry.discover()
        frame = self.clear()
        self.heading(frame, PRODUCT.display_name, "사용할 자세 분류 모델을 선택하세요.")
        if not self.models:
            ttk.Label(frame, text="현재 사용할 수 있는 모델이 없습니다. 데이터 및 모델 관리에서 상태를 확인하세요.").pack(anchor="w")
            for result in self.registry.last_results:
                model = result.descriptor
                detail = ko(result.reason or (model.management_status if model else "모델 정보를 읽을 수 없습니다."))
                name = ko(model.name) if model else "모델 파일"
                ttk.Label(frame, text=f"{name}: {detail}", wraplength=850).pack(anchor="w", pady=3)
        cards = ttk.Frame(frame)
        cards.pack(fill="both", expand=True)
        default = self.registry.active_model(self.models)
        ordered = sorted(self.models, key=lambda item: item is not default)
        for model in ordered:
            is_default = model is default
            card = ttk.LabelFrame(cards, text="★ 기본 모델 (권장)" if is_default else "자세 분류 모델", padding=18)
            card.pack(fill="x", pady=7)
            card.columnconfigure(0, weight=1)
            ttk.Label(card, text=f"모델: {ko(model.name)}", font=("Apple SD Gothic Neo", 13, "bold")).grid(row=0, column=0, sticky="w")
            ttk.Label(card, text=f"버전: {model.version}").grid(row=1, column=0, sticky="w", pady=(8, 0))
            ttk.Label(card, text=f"분석 창: {model.window_label}").grid(row=2, column=0, sticky="w")
            ttk.Label(card, text=f"입력 센서: {model.input_label}").grid(row=3, column=0, sticky="w")
            ttk.Label(card, text="상태: 사용 가능" + (" · 처음 실행 시 자동 선택되는 모델" if is_default else "")).grid(row=4, column=0, sticky="w")
            ttk.Button(card, text="선택", command=lambda item=model: self.select_model(item)).grid(row=0, column=1, rowspan=2, padx=20)
            ttk.Button(card, text="상세 정보", command=lambda item=model: self.show_details(item)).grid(row=2, column=1, rowspan=2, padx=20)
        actions = ttk.Frame(frame); actions.pack(fill="x", pady=(16, 0))
        ttk.Button(actions, text="데이터 및 모델 관리", command=self.show_management).pack(side="right")
        ttk.Button(actions, text="+ 모델 추가", command=self.start_training_wizard).pack(side="right", padx=(0, 8))

    def show_details(self, model: ModelDescriptor) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("모델 상세 정보")
        dialog.geometry("620x560")
        dialog.transient(self)
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        closing = ttk.Frame(dialog, padding=(10, 6)); closing.pack(side="bottom", fill="x")
        ttk.Button(closing, text="닫기", command=dialog.destroy).pack(side="right")
        content = tk.Text(dialog, wrap="word", padx=18, pady=18, font=("Apple SD Gothic Neo", 12))
        content.pack(fill="both", expand=True)
        metadata = model.metadata
        summary = metadata.get("fixed_configuration_group_cv_summary", {})
        lines = [
            f"모델 이름: {ko(model.name)}", f"버전: {model.version}", f"파일 생성 시각: {model.created_at}",
            f"분석 창: {model.window_label}", f"샘플링 주파수: {model.sampling_rate:g} Hz", "필수 센서: 압력 센서 16채널",
            f"지원 자세: {', '.join(model.labels)}", f"학습 피험자 수: {metadata.get('training_subject_count', '기록 없음')}",
            f"학습 기록 수: {metadata.get('training_record_count', '기록 없음')}",
            f"검증 방법: 피험자 단위 분할", f"Macro F1: {summary.get('macro_f1', metadata.get('final_parameter_search_window_macro_f1', '기록 없음'))}",
            f"모델 파일 해시: {__import__('hashlib').sha256(model.path.read_bytes()).hexdigest()}",
        ]
        content.insert("1.0", "\n\n".join(lines))
        content.configure(state="disabled")

    def select_model(self, model: ModelDescriptor) -> None:
        if model.requires_accelerometer:
            messagebox.showerror("센서 호환성", "선택한 모델은 가속도계 연결이 필요합니다. 압력 센서 전용 모델로 자동 전환하지 않습니다.")
            return
        self.selected_model = model
        self.selected_window = model.window.default_seconds
        self.show_mode_select()

    def show_mode_select(self) -> None:
        if self.selected_model is None:
            self.show_model_select()
            return
        frame = self.clear()
        self.window_choice = tk.StringVar(value=f"{self.selected_window:g}")
        personal_selected = (self.SHOW_PERSONAL_HIERARCHY and self.mode.get() == "measurement"
                             and self.personal_enabled.get())
        self.heading(frame, "측정 방식 선택", "개인 기준 2단계 모델 · 3초 · 실험용" if personal_selected else f"선택 모델: {ko(self.selected_model.name)} · {self.selected_model.window_label}")
        window_frame = ttk.LabelFrame(frame, text="윈도우 길이", padding=12)
        window_frame.pack(fill="x", pady=(0, 12))
        if self.selected_model.window.policy == "fixed":
            ttk.Label(window_frame, text=f"{self.selected_model.window_label} (이 모델은 윈도우 길이가 고정되어 있습니다.)").pack(anchor="w")
        else:
            options = self.selected_model.window.values if self.selected_model.window.policy == "discrete" else tuple(np.arange(self.selected_model.window.minimum_seconds, self.selected_model.window.maximum_seconds + 1e-9, self.selected_model.window.step_seconds))
            self.window_choice = tk.StringVar(value=f"{self.selected_window:g}")
            ttk.Label(window_frame, text="윈도우 길이(초)").pack(side="left")
            ttk.Combobox(window_frame, textvariable=self.window_choice, values=[f"{value:g}" for value in options], state="readonly", width=10).pack(side="left", padx=10)
        mode_frame = ttk.LabelFrame(frame, text="측정 방식", padding=(12, 6))
        mode_frame.pack(fill="x")
        for text, value, description in (
                ("라벨 데이터 수집", "labeled", "실제 자세를 정해 두고 기록합니다. 학습 데이터로 사용할 수 있습니다."),
                ("측정 전용", "measurement", "모델로 자세를 분석하고 기록만 남깁니다. 학습에는 사용하지 않습니다."),
                ("정자세 등록", "calibration", "빈 좌석 3초 확인 후 정자세 10초를 개인 기준으로 저장합니다.")):
            row = ttk.Frame(mode_frame); row.pack(fill="x", pady=3)
            ttk.Radiobutton(row, text=text, variable=self.mode, value=value, command=self._select_measurement_mode, width=16).pack(side="left")
            ttk.Label(row, text=description, foreground="#475569").pack(side="left", padx=(6, 0))
        if self.mode.get() == "calibration" or personal_selected:
            window_frame.pack_forget()
        self.fields = ttk.LabelFrame(frame, text="측정 설정", padding=18)
        self.fields.pack(fill="x", pady=18)
        self._mode_changed()
        source = ttk.LabelFrame(frame, text="센서 입력", padding=18)
        source.pack(fill="x")
        ttk.Label(source, text="포트").grid(row=0, column=0, sticky="w")
        self.port_selector = ttk.Combobox(source, textvariable=self.port, values=self.serial_port_choices(), width=32)
        self.port_selector.grid(row=0, column=1, sticky="w", padx=12)
        ttk.Button(source, text="포트 새로고침", command=self.refresh_ports).grid(row=0, column=2, sticky="w")
        ttk.Label(source, text="‘자동 탐색’은 연결된 아두이노를 찾습니다. 목록에 없으면 직접 입력하세요.",
                  foreground="#475569", wraplength=560).grid(row=1, column=1, columnspan=2, sticky="w", padx=12, pady=(4, 0))
        if self.mode.get() != "calibration":
            ttk.Label(source, text="또는 저장된 측정 파일 재생").grid(row=2, column=0, sticky="w", pady=(12, 0))
            ttk.Entry(source, textvariable=self.input_path, width=40).grid(row=2, column=1, sticky="ew", padx=12, pady=(12, 0))
            file_buttons = ttk.Frame(source); file_buttons.grid(row=2, column=2, sticky="w", pady=(12, 0))
            ttk.Button(file_buttons, text="파일 선택", command=self.choose_input).pack(side="left")
            ttk.Button(file_buttons, text="지우기", command=lambda: self.input_path.set("")).pack(side="left", padx=(4, 0))
            source.columnconfigure(1, weight=1)
            self.source_summary = tk.StringVar()
            ttk.Label(source, textvariable=self.source_summary, font=("Apple SD Gothic Neo", 12, "bold")).grid(
                row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))
            self._update_source_summary()
            trace = self.input_path.trace_add("write", lambda *_: self._update_source_summary())
            source.bind("<Destroy>", lambda _event, name=trace: self._remove_input_trace(name), add="+")
        personal = ttk.LabelFrame(frame, text="정자세 등록 정보" if self.mode.get() == "calibration" else "등록된 개인 기준 사용 (선택 · 실시간 센서)", padding=10)
        personal.pack(fill="x", pady=8)
        known_users = [profile.user_id for profile in CalibrationStore(STATE_ROOT / "calibration_profiles").list_profiles()]
        for row, (label, variable) in enumerate((("사용자 식별자", self.guidance_user), ("의자 식별자", self.chair_id), ("설치 위치 식별자", self.placement_id), ("센서 구성", self.sensor_configuration))):
            ttk.Label(personal, text=label).grid(row=row, column=0, sticky="w")
            entry = ttk.Combobox(personal, textvariable=variable, values=known_users, width=28) if row == 0 else ttk.Entry(personal, textvariable=variable, width=30)
            entry.grid(row=row, column=1, padx=8, sticky="w")
        ttk.Label(personal, text="의자나 센서 위치가 바뀌면 해당 식별자도 변경하세요.").grid(row=0, column=2, rowspan=2, sticky="w")
        self.alert_mode_label = tk.StringVar(value=ko(self.alert_mode.get()))
        alert_box = ttk.Frame(personal); alert_box.grid(row=2, column=2, sticky="w")
        ttk.Label(alert_box, text="알림 간격").pack(side="left", padx=(0, 6))
        mode_selector = ttk.Combobox(alert_box, textvariable=self.alert_mode_label, values=("배포 모드", "개발 모드"), state="readonly", width=15)
        mode_selector.pack(side="left")
        mode_selector.bind("<<ComboboxSelected>>", lambda _event: self.alert_mode.set({"배포 모드": "deployment", "개발 모드": "development"}[self.alert_mode_label.get()]))
        ttk.Label(personal, text="배포: 60초 / 120초 · 개발: 12초 / 24초").grid(row=3, column=2, sticky="w")
        timing = ttk.Frame(personal); timing.grid(row=4, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Label(timing, text="정자세 복귀 확인 (초)").pack(side="left")
        ttk.Entry(timing, textvariable=self.return_confirmation, width=5).pack(side="left", padx=8)
        ttk.Label(timing, text="판단 유보 허용 (초)").pack(side="left")
        ttk.Entry(timing, textvariable=self.uncertainty_tolerance, width=5).pack(side="left", padx=8)
        ttk.Label(timing, text="임시 설정 · 실제 센서 검증 필요").pack(side="left")
        if self.mode.get() == "labeled":
            personal.configure(text="개인 기준 연결 (선택 · 학습 데이터용)")
            ttk.Checkbutton(personal, text="등록된 개인 기준을 이 라벨 기록에 연결", variable=self.link_reference).grid(row=5, column=0, columnspan=3, sticky="w")
            for widget in personal.grid_slaves():
                info = widget.grid_info()
                if int(info["row"]) == 4 or (int(info["column"]) == 2 and int(info["row"]) >= 2):
                    widget.grid_remove()
        elif self.mode.get() == "calibration":
            for widget in personal.grid_slaves():
                info = widget.grid_info()
                if int(info["row"]) == 4 or (int(info["column"]) == 2 and int(info["row"]) >= 2):
                    widget.grid_remove()
        elif not self.SHOW_PERSONAL_HIERARCHY:
            # Every field of this frame only feeds the personal two-stage path.
            personal.destroy()
        else:
            personal.configure(text="개인 기준 2단계 판별 (선택 · 실시간 센서)")
            ttk.Checkbutton(personal, text="등록된 개인 기준으로 2단계 판별 사용 (실험용 · 3초)", variable=self.personal_enabled, command=self._select_measurement_mode).grid(row=5, column=0, columnspan=3, sticky="w")
            self._personal_choices = {str(path.relative_to(root)): path for root in dict.fromkeys((STATE_ROOT, PROJECT_ROOT))
                                      for path in available_models(root)}
            if self.personal_model_choice.get() not in self._personal_choices:
                self.personal_model_choice.set(next(iter(self._personal_choices), ""))
            ttk.Combobox(personal, textvariable=self.personal_model_choice, values=list(self._personal_choices), state="readonly", width=65).grid(row=6, column=0, columnspan=3, sticky="w", pady=5)
            ttk.Label(personal, text="1단계: 정자세/비정상 · 2단계: 비정상 6종. 해제하면 선택한 일반 모델을 사용합니다.", wraplength=800).grid(row=7, column=0, columnspan=3, sticky="w")
            ttk.Button(personal, text="개인 모델 평가 보기", command=self.show_personal_report).grid(row=8, column=0, columnspan=3, sticky="w", pady=5)
        bottom = ttk.Frame(frame); bottom.pack(fill="x", pady=24)
        ttk.Button(bottom, text="이전", command=self.show_model_select).pack(side="left")
        ttk.Label(bottom, text="Esc: 이전 화면", foreground="#64748b").pack(side="left", padx=10)
        self.bind("<Escape>", lambda _event: self.show_model_select())
        ttk.Button(bottom, text="센서 연결 및 등록 시작" if self.mode.get() == "calibration" else "센서 연결 및 측정 시작", command=self.start_measurement).pack(side="right")

    @staticmethod
    def serial_port_choices() -> list[str]:
        try:
            import serial.tools.list_ports
            ports = [port.device for port in serial.tools.list_ports.comports()]
        except Exception:  # listing is a convenience; manual entry still works
            ports = []
        return ["자동 탐색", *ports]

    def refresh_ports(self) -> None:
        choices = self.serial_port_choices()
        self.port_selector.configure(values=choices)
        self.status.set(f"연결된 포트 {len(choices) - 1}개를 찾았습니다.")
        messagebox.showinfo("포트 새로고침", f"사용 가능한 포트 {len(choices) - 1}개를 찾았습니다." if len(choices) > 1
                            else "연결된 포트를 찾지 못했습니다. 센서 연결을 확인하거나 ‘자동 탐색’을 사용하세요.")

    def _update_source_summary(self) -> None:
        summary = getattr(self, "source_summary", None)
        if summary is None:
            return
        path = self.input_path.get().strip()
        if not path:
            summary.set("입력: 실시간 센서")
        elif Path(path).is_file():
            summary.set(f"입력: 저장 파일 재생 · {Path(path).name}")
        else:
            summary.set("입력: 저장 파일 재생 · 파일을 찾을 수 없습니다")

    def _remove_input_trace(self, name) -> None:
        try:
            self.input_path.trace_remove("write", name)
        except (tk.TclError, ValueError):
            pass
        self.source_summary = None

    def _select_measurement_mode(self) -> None:
        self.selected_window = float(self.window_choice.get())
        self.show_mode_select()

    def _mode_changed(self) -> None:
        for child in self.fields.winfo_children(): child.destroy()
        if self.mode.get() == "labeled":
            ttk.Label(self.fields, text="측정 대상 실명").grid(row=0, column=0, sticky="w")
            ttk.Entry(self.fields, textvariable=self.subject, width=24).grid(row=0, column=1, padx=12)
            ttk.Label(self.fields, text="실제 자세").grid(row=1, column=0, sticky="w", pady=(12, 0))
            choices = self.selected_model.labels if self.selected_model else ()
            selector = ttk.Combobox(self.fields, textvariable=self.ground_truth, values=choices, state="readonly", width=22)
            selector.grid(row=1, column=1, padx=12, pady=(12, 0))
            if choices and not self.ground_truth.get(): self.ground_truth.set(choices[0])
        elif self.mode.get() == "calibration":
            ttk.Label(self.fields, text="빈 좌석을 3초 확인한 뒤 정자세를 10초 등록합니다. 개인 기준 파일로 저장됩니다.").pack(anchor="w")
        else:
            ttk.Label(self.fields, text="측정 전용에서는 실명·실제 자세·학습 대상 선택을 사용하지 않습니다.").pack(anchor="w")

    def choose_input(self) -> None:
        selected = filedialog.askopenfilename(title="재생할 센서 측정 파일 선택", filetypes=[("측정 파일", "*.csv")])
        if selected: self.input_path.set(selected)

    def start_measurement(self) -> None:
        if self.mode.get() == "calibration":
            self.start_calibration()
            return
        if self.selected_model is None: return
        if self._worker and self._worker.is_alive(): return
        if self.mode.get() == "labeled" and (not self.subject.get().strip() or not self.ground_truth.get()):
            messagebox.showerror("입력 확인", "라벨 수집에는 측정 대상 실명과 실제 자세를 선택해야 합니다.")
            return
        try:
            selected_window = float(self.window_choice.get()) if hasattr(self, "window_choice") else self.selected_window
            input_path = self.input_path.get().strip()
            if input_path and not Path(input_path).is_file():
                raise ValueError("재생할 측정 파일을 찾을 수 없습니다. 파일을 다시 선택하세요.")
            source = "csv_replay" if input_path else "serial"
            self._session_guidance_config = None
            self._session_personal_model = None
            reference = None
            use_personal = (self.SHOW_PERSONAL_HIERARCHY and self.mode.get() == "measurement"
                            and self.personal_enabled.get())
            if use_personal and source != "serial":
                raise ValueError("개인 기준 2단계 판별은 실시간 센서에서 사용하세요. 저장 파일은 학습 평가에서 등록 기록과 함께 비교합니다.")
            if use_personal or (self.mode.get() == "labeled" and self.link_reference.get()):
                context = CalibrationContext(sensor_configuration=self.sensor_configuration.get().strip(), chair_id=self.chair_id.get().strip(), placement_id=self.placement_id.get().strip())
                settings = AlertSettings.for_mode(self.alert_mode.get(), correct_return_confirmation_s=float(self.return_confirmation.get()), uncertainty_tolerance_s=float(self.uncertainty_tolerance.get()))
                reference = CalibrationStore(STATE_ROOT / "calibration_profiles").load(self.guidance_user.get().strip(), context)
                if reference is None:
                    raise ValueError("등록된 개인 기준이 없습니다. 먼저 ‘정자세 등록’ 모드에서 등록하세요.")
                if source != "serial":
                    raise ValueError("개인 기준 연결은 같은 센서·의자·설치 조건의 실시간 라벨 수집에서 사용하세요.")
                if use_personal:
                    model_path = self._personal_choices.get(self.personal_model_choice.get())
                    if model_path is None:
                        raise ValueError("개인 기준 모델을 먼저 학습한 뒤 선택하세요.")
                    self._session_personal_model = PersonalPostureModel.load(model_path)
                    selected_window = 3.
                    self._session_guidance_config = (self.guidance_user.get().strip(), context, settings)
            if self._session_personal_model:
                self.session_snapshot = self._session_personal_model.snapshot(datetime_stamp(), source)
            else:
                self.session_snapshot = SessionModelSnapshot.create(self.selected_model, session_id=datetime_stamp(), window_seconds=selected_window, input_source=source)
                self.selected_window = selected_window
            self._recorder = MeasurementRecorder(STATE_ROOT, mode=self.mode.get(), subject=self.subject.get().strip(), label=self.ground_truth.get(), model_snapshot=self.session_snapshot, data_directory=DATA_DIRECTORY)
            if reference is not None:
                self._recorder.guidance_metadata = {"user_id": self.guidance_user.get().strip(), "context": asdict(context),
                    "settings": asdict(settings), "reference_profile": reference.as_dict(),
                    "personal_model": self._session_personal_model.identity if self._session_personal_model else None}
        except (OSError, ValueError) as error:
            messagebox.showerror("측정 시작 오류", ko(error)); return
        self._last_manifest = None
        self._session_error = None
        self._session_finished = False
        self._guidance_session = None
        self._capture_requested.clear()
        self._capture_requested.set()
        self._capture_button = None
        self._allow_registration = False
        self._messages = SessionEventQueue()
        port = self.port.get().strip()
        self._session_input = (input_path, "auto" if port in {"", "자동 탐색"} else port)
        self._stop.clear()
        self._signal_buffer = SignalBuffer()
        self.show_realtime()
        self._worker = threading.Thread(target=self._run_session, name="센서-측정-파이프라인", daemon=True)
        self._worker.start()
        self.after(80, self.poll_messages)

    def start_calibration(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        try:
            user = self.guidance_user.get().strip()
            if not user:
                raise ValueError("정자세를 등록할 사용자 식별자를 입력하세요.")
            context = CalibrationContext(sensor_configuration=self.sensor_configuration.get().strip(),
                                         chair_id=self.chair_id.get().strip(), placement_id=self.placement_id.get().strip())
        except ValueError as error:
            messagebox.showerror("등록 설정 확인", ko(error))
            return
        self._messages = SessionEventQueue()
        self._stop.clear()
        self._recorder = None
        self._capture_button = None
        self._session_error = None
        self._session_finished = False
        self._guidance_session = None
        self._calibration_context = (user, context)
        self._calibration_port = self.port.get().strip()
        self._calibration_result = None
        frame = self.clear()
        self.heading(frame, "정자세 등록", f"사용자: {user} · 빈 좌석 확인 → 착석 → 정자세 등록")
        self.state_text = tk.StringVar(value="센서 연결 중")
        self.personal_text = tk.StringVar(value="좌석 위의 물건을 치우고 빈 좌석 확인을 진행하세요.")
        ttk.Label(frame, textvariable=self.state_text).pack(anchor="w", pady=10)
        ttk.Label(frame, textvariable=self.personal_text, wraplength=820).pack(anchor="w", pady=10)
        ttk.Button(frame, text="1. 빈 좌석 확인", command=self.check_empty_seat).pack(anchor="w", pady=8)
        ttk.Button(frame, text="2. 정자세 등록 / 다시 등록", command=self.register_correct_posture).pack(anchor="w", pady=8)
        self._signal_buffer = SignalBuffer()
        self.signal_monitor = SignalMonitorView(frame, self._signal_buffer)
        self.signal_monitor.pack(fill="x", pady=(12, 0))
        ttk.Button(frame, text="등록 취소", command=self._stop.set).pack(anchor="e", pady=20)
        self._worker = threading.Thread(target=self._run_calibration, name="정자세-등록", daemon=True)
        self._worker.start()
        self.after(80, self.poll_calibration_messages)

    def _run_calibration(self) -> None:
        frames = None
        try:
            user, context = self._calibration_context
            guidance = GuidanceSession(STATE_ROOT, user, context, AlertSettings(), self._messages.put,
                                       registration_only=True)
            self._guidance_session = guidance
            port = self._calibration_port
            frames = serial_frames("auto" if port in {"", "자동 탐색"} else port, 115200, 2.0,
                                   stop_event=self._stop, on_status=lambda state: guidance.interrupt(state))
            first_time = previous_time = None
            signal_buffer = getattr(self, "_signal_buffer", None)
            for timestamp, values in frames:
                if self._stop.is_set():
                    break
                if first_time is None:
                    first_time = timestamp
                elapsed = timestamp - first_time
                if signal_buffer is not None:
                    signal_buffer.push(elapsed, values)
                if previous_time is not None and elapsed - previous_time > 1.0:
                    guidance.interrupt("input_error", elapsed)
                previous_time = elapsed
                guidance.push(elapsed, values)
                if guidance.completed_profile_path is not None:
                    self._calibration_result = guidance.completed_profile_path
                    break
        except Exception as error:
            self._messages.put(("error", ko(error)))
        finally:
            if frames is not None:
                try:
                    frames.close()
                except Exception as error:
                    self._messages.put(("error", ko(error)))
            self._messages.put(("finished", self._calibration_result))

    def poll_calibration_messages(self) -> None:
        if self._closing:
            return
        for _ in range(100):
            try:
                kind, payload = self._messages.get_nowait()
            except queue.Empty:
                break
            if kind == "guidance_status":
                self.state_text.set(ko(payload["state"]))
                self.personal_text.set(ko(payload["message"]))
            elif kind == "error":
                self._session_error = str(payload)
            elif kind == "finished":
                self._session_finished = True
                self._guidance_session = None
                frame = self.clear()
                self.heading(frame, "정자세 등록 완료" if payload else "정자세 등록이 종료되었습니다")
                if payload:
                    ttk.Label(frame, text=f"개인 기준 저장 위치: {payload}", wraplength=820).pack(anchor="w", pady=10)
                    ttk.Label(frame, text="측정 전용 모드에서 같은 사용자와 설치 정보를 사용하면 이 기준을 불러옵니다.").pack(anchor="w")
                else:
                    ttk.Label(frame, text=self._session_error or "등록을 취소했습니다. 새 개인 기준은 저장하지 않았습니다.", wraplength=820).pack(anchor="w")
                ttk.Button(frame, text="측정 방식 선택으로", command=self.show_mode_select).pack(anchor="w", pady=20)
                return
        self.after(80, self.poll_calibration_messages)

    def show_realtime(self) -> None:
        frame = self.clear()
        self.heading(frame, "실시간 측정", "개인 기준 2단계 판별 · 실험용 · 3초 윈도우 / 1초 간격" if self._session_personal_model else "센서 입력은 한 번만 읽어 원시 데이터 기록·인공지능 예측·화면 표시에 함께 사용합니다.")
        self.state_text = tk.StringVar(value="센서 연결 중")
        self.prediction_text = tk.StringVar(value="자세 예측 준비 중")
        self.confidence_text = tk.StringVar(value="신뢰도: -")
        self.time_text = tk.StringVar(value="측정 시간: 00:00")
        self.rows_text = tk.StringVar(value="저장 행: 0")
        self.risk_text = tk.StringVar(value="위험 수준: 정상")
        top = ttk.Frame(frame); top.pack(fill="x")
        source_label = "저장 파일 재생" if self.input_path.get().strip() else "실시간 센서"
        mode_name = {"labeled": "라벨 데이터 수집", "measurement": "측정 전용", "calibration": "정자세 등록"}.get(self.mode.get(), self.mode.get())
        ttk.Label(top, text=f"입력: {source_label} · 방식: {mode_name}", foreground="#475569").pack(side="left")
        ttk.Button(top, text="측정 종료", command=self.stop_measurement).pack(side="right")
        if self.mode.get() == "labeled":
            ttk.Label(frame, text=f"실제 자세: {self.ground_truth.get()}", font=("Apple SD Gothic Neo", 14, "bold")).pack(anchor="w")
        ttk.Label(frame, textvariable=self.state_text, font=("Apple SD Gothic Neo", 18, "bold")).pack(anchor="w", pady=(20, 8))
        self.personal_text = tk.StringVar(value="개인별 자세 안내를 시작하려면 빈 좌석을 확인하세요." if self._session_guidance_config else "선택한 모델로 자세를 분석합니다.")
        ttk.Label(frame, textvariable=self.personal_text, wraplength=880).pack(anchor="w")
        if self._session_guidance_config:
            user, _, settings = self._session_guidance_config
            ttk.Label(frame, text=f"사용자: {user} · {ko(settings.mode)} · 최초 / 반복 알림: " + ("12초 / 24초" if settings.mode == "development" else "60초 / 120초")).pack(anchor="w")
            actions = ttk.Frame(frame); actions.pack(fill="x", pady=5)
            ttk.Button(actions, text="1. 빈 좌석 확인", command=self.check_empty_seat).pack(side="left")
        self.avatar = tk.Canvas(
            frame,
            height=220,
            background="#eef2f8",
            highlightthickness=1,
            highlightbackground="#d9e2ef",
        )
        self.avatar.pack(fill="x")
        self._load_avatar_images()
        self.avatar.bind("<Configure>", lambda _event: self.draw_avatar(self._current_posture))
        self.draw_avatar(None)
        ttk.Label(frame, textvariable=self.prediction_text, font=("Apple SD Gothic Neo", 24, "bold")).pack(pady=(18, 4))
        ttk.Label(frame, textvariable=self.confidence_text).pack()
        info = ttk.Frame(frame); info.pack(fill="x", pady=16)
        for index, item in enumerate((self.time_text, self.rows_text, self.risk_text)):
            ttk.Label(info, textvariable=item).grid(row=0, column=index, padx=20, sticky="w")
        self.alert_text = tk.StringVar()
        ttk.Label(frame, textvariable=self.alert_text, foreground="#b71c1c").pack(anchor="w")
        self.signal_monitor = SignalMonitorView(frame, self._signal_buffer)
        self.signal_monitor.pack(fill="x", pady=(12, 0))

    def start_labeled_capture(self):
        self._capture_requested.set()
        if self._capture_button is not None:
            self._capture_button.configure(state="disabled", text="라벨 기록 중")

    def check_empty_seat(self):
        if self.mode.get() == "labeled" and self._recorder and self._recorder.rows:
            messagebox.showinfo("라벨 기록 중", "현재 기록을 종료한 뒤 새 측정에서 빈 좌석을 확인하세요."); return
        if self._guidance_session and messagebox.askokcancel("빈 좌석 기준 확인", "좌석 위의 물건을 모두 치우고 자리에서 일어나세요. 센서는 측정 위치에 둔 채 확인을 누르면 3초 동안 빈 좌석 기준을 측정합니다."):
            self._guidance_session.command("baseline")

    def register_correct_posture(self):
        if self.mode.get() != "calibration":
            return
        if self.mode.get() == "labeled" and self._recorder and self._recorder.rows:
            messagebox.showinfo("라벨 기록 중", "현재 기록을 종료한 뒤 새 측정에서 정자세를 등록하세요."); return
        text = ("좌석 중앙에 앉아 골반을 수평으로 유지하고 상체를 세우세요. 어깨의 힘을 빼고 양발을 편안하게 받치세요. "
                "등받이를 적절히 사용하고 다리를 꼬지 마세요. 가능하면 다른 사람에게 자세를 확인받으세요.\n\n"
                "압력값이 안정적이라는 이유만으로 정자세라고 판단할 수는 없습니다. 안내한 자세인지 확인한 뒤 10초 동안 움직이지 마세요. "
                "의자나 센서 설치 위치를 변경한 뒤에는 다시 등록하세요.")
        if self._guidance_session and messagebox.askokcancel("정자세 확인", text):
            self._guidance_session.command("register")

    def dismiss_warning(self, notify=True):
        popup = self._warning_popup
        self._warning_popup = None
        if popup is not None and popup.winfo_exists(): popup.destroy()
        if notify and self._guidance_session:
            self._guidance_session.command("close_popup")

    def show_warning(self, event):
        if self._warning_popup is not None and self._warning_popup.winfo_exists():
            return
        popup = tk.Toplevel(self)
        self._warning_popup = popup
        popup.title("자세 교정 안내")
        popup.geometry("560x220")
        popup.transient(self)
        self._warning_text = tk.StringVar()
        ttk.Label(popup, textvariable=self._warning_text, wraplength=510, padding=20, font=("Apple SD Gothic Neo", 14)).pack(fill="both", expand=True)
        ttk.Button(popup, text="닫기", command=self.dismiss_warning).pack(pady=12)
        popup.protocol("WM_DELETE_WINDOW", self.dismiss_warning)
        self._update_warning(event)

    def _update_warning(self, event):
        if self._warning_popup is not None and self._warning_popup.winfo_exists():
            self._warning_text.set(f"{event.posture or '기준 정자세에서 벗어났습니다'}\n지속 시간: {event.incorrect_duration_s:.1f}초\n{ko(event.correction_guidance)}")

    def _run_session(self) -> None:
        frames = None
        guidance = None
        session_error = None

        def report_error(error):
            nonlocal session_error
            session_error = ko(error)
            self._messages.put(("error", session_error))

        try:
            assert self.selected_model and self._recorder
            personal_model = getattr(self, "_session_personal_model", None)
            engine = None
            if personal_model is None:
                predictor = PosturePredictor(self.selected_model.path, self.session_snapshot.selected_window_seconds, registry=self.registry)
                engine = RealtimePosturePredictor(predictor, LiveConfig(window_seconds=self.session_snapshot.selected_window_seconds, minimum_window_seconds=predictor.minimum_window_seconds))
            input_path, port = self._session_input
            profile = None if getattr(self, "_session_guidance_config", None) else occupancy_profile_for_input(
                self.session_snapshot.input_source,
                DATA_DIRECTORY / "standard",
            )
            detector = OccupancyDetector(profile) if profile is not None else None
            risk = RiskAssessment()
            guidance_config = getattr(self, "_session_guidance_config", None)
            if guidance_config:
                user, context, settings = guidance_config
                guidance = GuidanceSession(STATE_ROOT, user, context, settings, self._messages.put, self._recorder,
                                           allow_registration=getattr(self, "_allow_registration", True), personal_model=personal_model)
                self._guidance_session = guidance
                detector = None
            first_time: float | None = None
            previous_elapsed: float | None = None
            latest_prediction = None

            def input_status(state):
                nonlocal latest_prediction, detector
                if engine: engine.reset(state)
                latest_prediction = None
                risk.reset()
                detector = OccupancyDetector(profile) if profile is not None else None
                if guidance:
                    now = time.monotonic() - first_time if first_time is not None else 0.
                    guidance.interrupt(state, now)
                self._messages.put(("input_status", state))

            frames = replay_csv(Path(input_path), 1.0, stop_event=self._stop) if input_path else serial_frames(port, 115200, 2.0, stop_event=self._stop, on_status=input_status)
            # Display-only copy; a missing monitor must never affect recording.
            signal_buffer = getattr(self, "_signal_buffer", None)
            for raw_time, values in frames:
                if self._stop.is_set(): break
                if first_time is None: first_time = raw_time
                elapsed = float(raw_time - first_time)
                if signal_buffer is not None: signal_buffer.push(elapsed, values)
                if not guidance or self._recorder.mode != "labeled":
                    self._recorder.write_frame(elapsed, values)
                if previous_elapsed is not None and elapsed - previous_elapsed > 1.0:
                    if engine: engine.reset("센서 연결 공백")
                    detector = OccupancyDetector(profile) if profile is not None else None
                    risk.reset()
                    latest_prediction = None
                    if guidance: guidance.interrupt("input_error", elapsed - 1e-6)
                    self._messages.put(("warming", None))
                previous_elapsed = elapsed
                if guidance:
                    self._messages.put(("frame", (elapsed, self._recorder.rows, True)))
                    if not guidance.push(elapsed, values, latest_prediction):
                        if engine: engine.reset("Personal guidance not ready / empty seat")
                        latest_prediction = None
                        continue
                    if self._recorder.mode == "labeled" and self._capture_requested.is_set():
                        self._recorder.write_frame(elapsed, values)
                if personal_model is not None:
                    continue
                changed = detector.update(elapsed, values) if detector is not None else None
                occupied = detector.occupied if detector is not None else True
                if not guidance:
                    self._messages.put(("frame", (elapsed, self._recorder.rows, profile is not None)))
                if changed is False:
                    engine.reset("빈 좌석 감지")
                    risk.reset()
                    self._messages.put(("empty", elapsed))
                    continue
                if not occupied:
                    self._messages.put(("empty", elapsed)); continue
                if changed is True:
                    self._messages.put(("warming", None))
                event = engine.push(elapsed, values)
                if event is not None:
                    latest_prediction = event
                    record = event.as_dict(); record.update({"session_elapsed_s": elapsed, "session_id": self.session_snapshot.session_id, "model_snapshot_id": self.session_snapshot.model_snapshot_id, "window_seconds": self.session_snapshot.selected_window_seconds, "class_scores": record.pop("probabilities")})
                    self._recorder.write_prediction(record)
                    if not guidance: risk.update(elapsed, event.posture, event.confidence)
                    self._messages.put(("prediction", (event, risk.level, risk.message)))
        except Exception as error:
            report_error(error)
        finally:
            for cleanup in (
                lambda: guidance.interrupt("session_stopped", guidance.last_timestamp + 1e-6) if guidance else None,
                lambda: frames.close() if frames is not None else None,
            ):
                try:
                    cleanup()
                except Exception as error:
                    report_error(error)
            if self._recorder:
                try:
                    self._last_manifest = self._recorder.close(error=session_error, completion_reason="stopped" if self._stop.is_set() else "completed")
                except Exception as error:
                    report_error(f"측정 결과 저장 실패: {error}")
            self._messages.put(("finished", getattr(self, "_last_manifest", None)))

    def poll_messages(self) -> None:
        if self._closing:
            return
        for _ in range(100):
            try: kind, payload = self._messages.get_nowait()
            except queue.Empty: break
            if kind == "frame":
                elapsed, rows, calibrated = payload
                self.time_text.set(f"측정 시간: {int(elapsed)//60:02d}:{int(elapsed)%60:02d}")
                self.rows_text.set(f"저장 행: {rows}")
                if not calibrated:
                    if self.session_snapshot.input_source == "csv_replay":
                        self.state_text.set("저장 파일의 자세 분석 중")
                    else:
                        self.state_text.set("측정 중 (빈 좌석 기준 데이터 없음)")
            elif kind == "empty":
                self.state_text.set("현재 상태: 앉아 있는 사람이 없습니다")
                self.prediction_text.set("자세 예측 없음")
                self.confidence_text.set("신뢰도: -")
                self.risk_text.set("위험 수준: 정상"); self.alert_text.set(""); self.draw_avatar(None)
            elif kind == "warming":
                self.state_text.set("현재 상태: 새 데이터 수집 중")
                self.prediction_text.set("자세 예측 준비 중")
                self.confidence_text.set("신뢰도: -")
                self.risk_text.set("위험 수준: 정상"); self.alert_text.set(""); self.draw_avatar(None)
            elif kind == "prediction":
                event, level, alert = payload
                if getattr(self, "_session_guidance_config", None):
                    self.confidence_text.set(f"모델 예측: {event.posture or '판단 유보'} · {(event.confidence or 0) * 100:.0f}%")
                    continue
                if event.posture:
                    self.state_text.set("현재 상태: 착석")
                    self.prediction_text.set(f"예측 자세: {event.posture}")
                    label = "확률" if self.selected_model.confidence_semantics == "calibrated_probability" else "신뢰 점수"
                    self.confidence_text.set(f"{label}: {event.confidence * 100:.0f}%")
                    self.risk_text.set(f"위험 수준: {level}"); self.alert_text.set(alert); self.draw_avatar(event.posture)
                else:
                    self.prediction_text.set("예측 자세: 판단 유보")
                    self.confidence_text.set("신뢰도: -")
                    self.risk_text.set("위험 수준: 정상"); self.alert_text.set(""); self.draw_avatar(None)
            elif kind == "error":
                self._session_error = str(payload)
                self.state_text.set("측정 오류"); self.alert_text.set(ko(payload))
                self.dismiss_warning(notify=False)
            elif kind == "input_status":
                self.state_text.set("입력 오류 / 센서 연결 끊김")
                self.prediction_text.set("자세 판단 중지")
                self.confidence_text.set("신뢰도: -")
                self.risk_text.set("위험 수준: 정상"); self.alert_text.set(""); self.draw_avatar(None)
                self.dismiss_warning(notify=False)
            elif kind == "guidance_status":
                self.personal_text.set(ko(payload["message"]))
                self.state_text.set(ko(payload["state"]))
                if payload.get("popup_action") == "dismiss": self.dismiss_warning(notify=False)
                button = self._capture_button
                if button is not None and button.winfo_exists() and not self._capture_requested.is_set():
                    button.configure(state="normal" if payload["state"] == "profile_loaded" else "disabled")
            elif kind == "guidance":
                self.state_text.set(ko(payload.state))
                self.personal_text.set(ko(payload.message))
                self.prediction_text.set(ko(payload.message))
                self.risk_text.set(f"잘못된 자세 지속: {payload.incorrect_duration_s:.1f}초")
                self.draw_avatar("정자세" if payload.state == "correct" else payload.posture)
                if payload.model_identity is not None:
                    self.confidence_text.set("비정상 점수: " + (f"{payload.gate_score:.2f}" if payload.gate_score is not None else "-")
                                             + (f" · 세부 자세 점수: {payload.detail_score:.2f}" if payload.detail_score is not None else "")
                                             + " · 확률 보정 전")
                if payload.popup_action == "dismiss": self.dismiss_warning(notify=False)
                elif payload.popup_action == "show": self.show_warning(payload)
                self._update_warning(payload)
            elif kind == "finished":
                self._session_finished = True
                self.finish_measurement(payload)
                return
        if not self._session_finished: self.after(80, self.poll_messages)

    def _load_avatar_images(self) -> None:
        if self._avatar_images or missing_avatar_assets(AVATAR_ASSET_DIRECTORY):
            return
        try:
            filenames = {avatar_filename(None)}
            filenames.update(avatar_filename(label) for label in self.selected_model.labels)
            for filename in filenames:
                source = tk.PhotoImage(master=self, file=str(AVATAR_ASSET_DIRECTORY / filename))
                scale = max(1, (source.height() + 255) // 256)
                self._avatar_images[filename] = source.subsample(scale, scale)
        except tk.TclError:
            self._avatar_images.clear()

    def draw_avatar(self, posture: str | None) -> None:
        self._current_posture = posture
        self.avatar.delete("all")
        width = max(self.avatar.winfo_width(), 700)
        height = max(self.avatar.winfo_height(), 280)
        cx = width // 2
        image = self._avatar_images.get(avatar_filename(posture))
        if image is not None:
            self.avatar.create_image(cx, height // 2, image=image, anchor="center")
            return

        # Keep a dependency-free fallback if an external package drops the PNGs.
        offset = -38 if posture and "왼쪽" in posture else 38 if posture and "오른쪽" in posture else 0
        lean = -25 if posture == "앞으로 숙이기" else 25 if posture == "뒤로 기대기" else 0
        self.avatar.create_oval(cx - 18 + offset, 30, cx + 18 + offset, 66, fill="#536dfe", outline="")
        self.avatar.create_line(cx + offset, 68, cx + offset + lean, 140, width=15, fill="#78909c")
        self.avatar.create_line(cx + offset + lean, 102, cx - 48 + offset, 128, width=10, fill="#78909c")
        self.avatar.create_line(cx + offset + lean, 102, cx + 48 + offset, 128, width=10, fill="#78909c")
        self.avatar.create_line(cx + offset + lean, 140, cx - 35 + offset, 190, width=12, fill="#78909c")
        self.avatar.create_line(cx + offset + lean, 140, cx + 35 + offset, 190, width=12, fill="#78909c")
        self.avatar.create_text(cx, 260, text=posture or "착석 상태를 확인하는 중", font=("Apple SD Gothic Neo", 11), fill="#455a64")

    def stop_measurement(self) -> None:
        self._stop.set()
        self.dismiss_warning(notify=False)
        self.state_text.set("측정 결과를 저장하고 있습니다.")

    def finish_measurement(self, manifest: Path | None) -> None:
        self.dismiss_warning(notify=False)
        if not manifest or not manifest.exists():
            if self._worker and self._worker.is_alive():
                self.after(150, lambda: self.finish_measurement(self._last_manifest))
                return
            messagebox.showerror("저장 오류", self._session_error or "측정 결과 파일을 생성하지 못했습니다.")
            self.show_mode_select()
            return
        frame = self.clear()
        result = json.loads(manifest.read_text(encoding="utf-8"))
        error = result.get("session_error") or self._session_error
        self.heading(frame, "측정 오류 · 수집된 데이터는 저장했습니다" if error else "측정이 저장되었습니다", f"저장 위치: {manifest.parent}")
        if error:
            ttk.Label(frame, text=f"오류 원인: {ko(error)}", wraplength=850, foreground="#a52424").pack(anchor="w", pady=8)
        quality = result["qc"]
        ttk.Label(frame, text=f"품질 검사: {ko(quality['status'])} · {'; '.join(quality.get('issues', [])) or '기록 형식 정상'} (저장 행: {quality.get('row_count', 0)})").pack(anchor="w", pady=8)
        if self.mode.get() == "labeled":
            ttk.Label(frame, text="데이터 품질을 확인한 뒤 처리 방식을 선택하세요.").pack(anchor="w", pady=8)
            current = self.dataset_status_for(manifest)
            self.finish_status_text = tk.StringVar(value=f"현재 상태: {ko(current) if current else '미지정'}")
            ttk.Label(frame, textvariable=self.finish_status_text, font=("Apple SD Gothic Neo", 12, "bold")).pack(anchor="w", pady=(0, 6))
            for title in ("저장만", "학습 대기열에 추가", "평가 전용으로 보관", "학습에서 제외"):
                ttk.Button(frame, text=title, command=lambda value=title: self.set_dataset_status(manifest, value)).pack(anchor="w", pady=4)
            ttk.Button(frame, text="지금 학습 시작", command=lambda: self.queue_and_train(manifest)).pack(anchor="w", pady=4)
            ttk.Button(frame, text="다시 측정", command=self.show_mode_select).pack(anchor="w", pady=4)
        if self.mode.get() != "labeled":
            ttk.Button(frame, text="다시 측정", command=self.show_mode_select).pack(anchor="w", pady=4)
        ttk.Button(frame, text="데이터 및 모델 관리", command=self.show_management).pack(anchor="w", pady=18)
        ttk.Button(frame, text="처음으로", command=self.show_model_select).pack(anchor="w")

    def set_dataset_status(self, manifest: Path, status: str) -> bool:
        record = next((item for item in self.dataset_service.records() if item.manifest_path == manifest), None)
        if record is None:
            messagebox.showerror("데이터 오류", "측정 기록을 찾을 수 없습니다."); return False
        try:
            self.dataset_service.set_status(record, status)
        except (OSError, ValueError) as error:
            messagebox.showerror("상태 저장 실패", ko(error)); return False
        status_text = getattr(self, "finish_status_text", None)
        if status_text is not None:
            try:
                status_text.set(f"현재 상태: {status}")
            except tk.TclError:
                pass
        messagebox.showinfo("저장 완료", f"데이터 상태를 ‘{status}’로 저장했습니다.")
        return True

    def dataset_status_for(self, manifest: Path) -> str | None:
        record = next((item for item in self.dataset_service.records() if item.manifest_path == manifest), None)
        return record.status if record else None

    def queue_and_train(self, manifest: Path) -> None:
        if not self.set_dataset_status(manifest, "학습 대기열에 추가"):
            return
        csv_name = json.loads(manifest.read_text(encoding="utf-8"))["csv"]
        csv_path = (manifest.parent / csv_name).resolve()
        records = [record for record in self.dataset_service.records() if record.csv_path.resolve() == csv_path]
        self.start_training_wizard(records)

    def start_training_wizard(self, preselected_records=None) -> None:
        self.selected_records = [record for record in (preselected_records or [])
                                 if self.dataset_service.is_training_eligible(record)]
        self.training_wizard_state = TrainingWizardState()
        self.training_wizard_state.remember_records(self.selected_records)
        self.show_training_wizard_setup()

    def show_management(self) -> None:
        frame = self.clear()
        self.heading(frame, "데이터 및 모델 관리", "모델을 검토하고 승인하세요. 기본 모델은 별도로 선택합니다.")
        navigation = ttk.Frame(frame); navigation.pack(fill="x", pady=(0, 10))
        ttk.Button(navigation, text="처음으로", command=self.show_model_select).pack(side="left")
        ttk.Button(navigation, text="+ 모델 추가", command=lambda: self.start_training_wizard(self.selected_records)).pack(side="right")
        ttk.Button(navigation, text="학습 결과 비교", command=self.show_results).pack(side="right", padx=6)
        notebook = ttk.Notebook(frame); notebook.pack(fill="both", expand=True)
        models_frame = ttk.Frame(notebook, padding=10)
        data_frame = ttk.Frame(notebook, padding=10)
        notebook.add(models_frame, text="모델 관리")
        notebook.add(data_frame, text="데이터 관리")
        self.management_notebook = notebook
        self.model_management = ModelManagementView(models_frame, self)
        self.model_management.pack(fill="both", expand=True)
        self._build_dataset_selector(data_frame, allow_status_changes=True)
        self.refresh_dataset_table()

    def _build_dataset_selector(self, frame: ttk.Frame, *, allow_status_changes: bool) -> None:
        self.training_selection_only = not allow_status_changes
        self._build_dataset_locations(frame)
        filters = ttk.Frame(frame); filters.pack(fill="x", pady=6)
        self.filter_subject = tk.StringVar(); self.filter_posture = tk.StringVar(); self.filter_status = tk.StringVar(value="전체"); self.filter_date = tk.StringVar(); self.filter_min_rows = tk.StringVar(); self.filter_min_duration = tk.StringVar()
        from src.posture_core import CLASS_NAMES
        choices = {"자세": ["", *CLASS_NAMES], "상태": ["전체", *sorted({record.status for record in self.dataset_service.records()})]}
        for index, (label, variable) in enumerate((("실명", self.filter_subject), ("자세", self.filter_posture), ("날짜", self.filter_date), ("최소 행", self.filter_min_rows), ("최소 시간", self.filter_min_duration), ("상태", self.filter_status))):
            row, column = divmod(index, 3)
            ttk.Label(filters, text=label).grid(row=row, column=column * 2, padx=(0, 5), pady=3, sticky="w")
            if label in choices:
                entry = ttk.Combobox(filters, textvariable=variable, values=choices[label], state="readonly", width=12)
                entry.bind("<<ComboboxSelected>>", lambda _event: self.refresh_dataset_table())
            else:
                entry = ttk.Entry(filters, textvariable=variable, width=12)
            entry.grid(row=row, column=column * 2 + 1, padx=(0, 12), sticky="ew")
            entry.bind("<Return>", lambda _event: self.refresh_dataset_table())
            filters.columnconfigure(column * 2 + 1, weight=1)
        filter_actions = ttk.Frame(filters); filter_actions.grid(row=2, column=0, columnspan=6, sticky="e", pady=3)
        ttk.Button(filter_actions, text="필터 초기화", command=self.reset_dataset_filters).pack(side="right")
        ttk.Button(filter_actions, text="필터 적용", command=self.refresh_dataset_table).pack(side="right", padx=6)
        ttk.Label(filter_actions, text="날짜는 YYYY-MM-DD 일부 입력 가능 · 열 제목을 누르면 정렬", foreground="#475569").pack(side="right", padx=8)
        table_frame = ttk.Frame(frame); table_frame.pack(fill="both", expand=True)
        self.dataset_tree = ttk.Treeview(table_frame, columns=("선택", "실명", "자세", "행", "시간", "품질", "상태"), show="headings", height=7)
        self.dataset_sort = getattr(self, "dataset_sort", (None, False))
        for key, label, width in (("선택", "선택", 45), ("실명", "실명", 100), ("자세", "자세", 120), ("행", "행", 60), ("시간", "시간(초)", 80), ("품질", "품질", 100), ("상태", "상태", 130)):
            if key == "선택":
                self.dataset_tree.heading(key, text=label)
            else:
                self.dataset_tree.heading(key, text=label, command=lambda column=key: self.sort_dataset_table(column))
            self.dataset_tree.column(key, width=width, anchor="center")
        self.dataset_tree.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.dataset_tree.yview)
        scroll.pack(side="right", fill="y"); self.dataset_tree.configure(yscrollcommand=scroll.set)
        self.dataset_tree.bind("<Button-1>", self.click_dataset_checkbox)
        self.dataset_tree.bind("<Double-1>", self.double_click_dataset)
        self.dataset_tree.bind("<space>", self.toggle_dataset)
        selection_actions = ttk.Frame(frame); selection_actions.pack(fill="x", pady=4)
        ttk.Button(selection_actions, text="현재 목록 전체 선택", command=self.select_visible_datasets).pack(side="left")
        ttk.Button(selection_actions, text="전체 선택 해제", command=self.clear_dataset_selection).pack(side="left", padx=5)
        self.dataset_selection_count = tk.StringVar()
        ttk.Label(selection_actions, textvariable=self.dataset_selection_count).pack(side="left", padx=8)
        ttk.Button(selection_actions, text="분포맵 보기", command=self.show_record_distribution).pack(side="right")
        ttk.Label(frame, text="체크박스 클릭 또는 스페이스 키로 선택·해제 · 행을 고른 뒤 ‘분포맵 보기’로 센서 분포 확인").pack(anchor="w")
        if allow_status_changes:
            choices = ttk.LabelFrame(frame, text="체크한 기록의 상태 변경", padding=6); choices.pack(fill="x", pady=6)
            for title in ("저장만", "학습 대기열에 추가", "학습에 포함", "학습에서 제외", "평가 전용", "폐기"):
                ttk.Button(choices, text=title, command=lambda value=title: self.confirm_selected_status(value)).pack(side="left", padx=2)

    def _build_dataset_locations(self, frame) -> None:
        locations = "\n".join(str(path) for path in self.dataset_service.data_directories)
        ttk.Label(frame, text=f"조회 위치: {locations}", wraplength=760).pack(anchor="w", pady=(2, 4))
        self.dataset_list_summary = tk.StringVar()
        ttk.Label(frame, textvariable=self.dataset_list_summary, wraplength=760).pack(anchor="w", pady=(0, 4))

    def reset_dataset_filters(self) -> None:
        for variable in (self.filter_subject, self.filter_posture, self.filter_date, self.filter_min_rows, self.filter_min_duration):
            variable.set("")
        self.filter_status.set("전체")
        self.refresh_dataset_table()

    SORT_KEYS = {"실명": lambda r: r.subject, "자세": lambda r: r.posture, "행": lambda r: r.rows,
                 "시간": lambda r: r.duration_s, "품질": lambda r: r.quality, "상태": lambda r: r.status}

    def sort_dataset_table(self, column: str) -> None:
        current, descending = getattr(self, "dataset_sort", (None, False))
        self.dataset_sort = (column, not descending if current == column else False)
        self.refresh_dataset_table()

    def _update_sort_headings(self) -> None:
        column, descending = getattr(self, "dataset_sort", (None, False))
        labels = {"실명": "실명", "자세": "자세", "행": "행", "시간": "시간(초)", "품질": "품질", "상태": "상태"}
        for key, label in labels.items():
            self.dataset_tree.heading(key, text=label + ((" ▼" if descending else " ▲") if key == column else ""))

    def confirm_selected_status(self, status: str) -> None:
        count = len(self.selected_records)
        if not count:
            messagebox.showerror("선택 필요", "상태를 변경할 데이터의 체크박스를 선택하세요."); return
        warning = "\n\n‘폐기’한 기록은 학습에 쓰이지 않습니다. 원본 파일은 삭제되지 않습니다." if status == "폐기" else ""
        if not messagebox.askyesno("상태 변경 확인", f"선택한 기록 {count}개의 상태를 ‘{status}’(으)로 바꿀까요?{warning}"):
            return
        remaining = self.set_selected_status(status)
        if remaining == 0:
            messagebox.showinfo("상태 변경 완료", f"기록 {count}개의 상태를 ‘{status}’(으)로 저장했습니다.")

    def refresh_dataset_table(self) -> None:
        records = self.dataset_service.records()
        subject, posture, status, date = self.filter_subject.get().strip(), self.filter_posture.get().strip(), self.filter_status.get().strip(), self.filter_date.get().strip()
        try:
            minimum_rows = int(self.filter_min_rows.get() or 0)
            minimum_duration = float(self.filter_min_duration.get() or 0)
            if minimum_rows < 0 or not np.isfinite(minimum_duration) or minimum_duration < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("필터 입력 확인", "최소 행은 0 이상의 정수, 최소 시간은 0 이상의 숫자로 입력하세요.")
            return
        selected_paths = {record.csv_path for record in self.selected_records}
        self.selected_records = [record for record in records if record.csv_path in selected_paths
                                 and (not self.training_selection_only or self.dataset_service.is_training_eligible(record))]
        self.visible_records = [record for record in records if (not getattr(self, "training_selection_only", False) or self.dataset_service.is_training_eligible(record)) and (not subject or subject in record.subject) and (not posture or posture in record.posture) and (not date or date in record.created_at) and record.rows >= minimum_rows and record.duration_s >= minimum_duration and (status in {"", "전체"} or status == record.status)]
        eligible_count = sum(self.dataset_service.is_training_eligible(record) for record in records)
        sort_column, descending = getattr(self, "dataset_sort", (None, False))
        if sort_column in self.SORT_KEYS:
            self.visible_records.sort(key=self.SORT_KEYS[sort_column], reverse=descending)
        self._update_sort_headings()
        summary = f"전체 {len(records)}개 · 학습 가능 {eligible_count}개 · 현재 표시 {len(self.visible_records)}개"
        if not records:
            summary += " · 조회 위치에 측정 CSV가 없습니다."
        elif not self.visible_records:
            summary += " · 검색 조건" + ("과 학습 가능 여부" if self.training_selection_only else "") + "를 확인하세요."
        self.dataset_list_summary.set(summary)
        for item in self.dataset_tree.get_children(): self.dataset_tree.delete(item)
        selected_paths = {record.csv_path for record in self.selected_records}
        for index, record in enumerate(self.visible_records):
            self.dataset_tree.insert("", "end", iid=str(index), values=("☑" if record.csv_path in selected_paths else "☐", record.subject, record.posture or "-", record.rows, f"{record.duration_s:.1f}", ko(record.quality), ko(record.status)))
        self.update_dataset_checkboxes()

    def show_record_distribution(self):
        item_id = self.dataset_tree.focus() or next(iter(self.dataset_tree.selection()), "")
        if not item_id:
            messagebox.showinfo("분포맵", "목록에서 확인할 기록을 먼저 고르세요.")
            return None
        record = self.visible_records[int(item_id)]
        try:
            window = DistributionWindow(self, record, self.dataset_service.records())
        except (OSError, ValueError) as error:
            messagebox.showerror("분포맵을 열 수 없음", ko(error))
            return None
        self._distribution_window = window
        return window

    def click_dataset_checkbox(self, event):
        if self.dataset_tree.identify_region(event.x, event.y) != "cell" or self.dataset_tree.identify_column(event.x) != "#1":
            return
        item = self.dataset_tree.identify_row(event.y)
        if item:
            self.dataset_tree.focus_set()
            self.dataset_tree.focus(item)
            self.dataset_tree.selection_set(item)
            self.toggle_dataset(item_id=item)
        return "break"

    def double_click_dataset(self, event):
        # The first click already toggles a checkbox; do not toggle it twice.
        if self.dataset_tree.identify_column(event.x) == "#1":
            return "break"
        item = self.dataset_tree.identify_row(event.y)
        if item:
            self.toggle_dataset(item_id=item)
        return "break"

    def toggle_dataset(self, _event=None, *, item_id=None):
        item_id = item_id or self.dataset_tree.focus()
        if not item_id: return "break"
        record = self.visible_records[int(item_id)]
        self.selected_records = [item for item in self.selected_records if item.csv_path != record.csv_path] if any(item.csv_path == record.csv_path for item in self.selected_records) else [*self.selected_records, record]
        self.update_dataset_checkboxes()
        return "break"

    def update_dataset_checkboxes(self) -> None:
        selected_paths = {record.csv_path for record in self.selected_records}
        for item in self.dataset_tree.get_children():
            record = self.visible_records[int(item)]
            self.dataset_tree.set(item, "선택", "☑" if record.csv_path in selected_paths else "☐")
        visible_count = sum(record.csv_path in selected_paths for record in self.visible_records)
        self.dataset_selection_count.set(f"선택 {len(selected_paths)}개 (현재 목록 {visible_count}개)")

    def select_visible_datasets(self) -> None:
        selected_paths = {record.csv_path for record in self.selected_records}
        self.selected_records.extend(record for record in self.visible_records if record.csv_path not in selected_paths)
        self.update_dataset_checkboxes()

    def clear_dataset_selection(self) -> None:
        self.selected_records = []
        self.update_dataset_checkboxes()

    def set_selected_status(self, status: str) -> None:
        if not self.selected_records:
            messagebox.showerror("선택 필요", "상태를 변경할 데이터의 체크박스를 선택하세요."); return
        for record in list(self.selected_records):
            try:
                self.dataset_service.set_status(record, status)
            except (OSError, ValueError) as error:
                self.refresh_dataset_table()
                messagebox.showerror("상태 저장 실패", f"{record.csv_path.name}: {ko(error)}")
                return len(self.selected_records)
            self.selected_records.remove(record)
        self.selected_records = []; self.refresh_dataset_table()
        return 0

    def activate_model(self, model: ModelDescriptor) -> None:
        try:
            self.registry.activate(model)
        except (ValueError, OSError) as error:
            messagebox.showerror("기본 모델 저장 실패", ko(error)); return
        messagebox.showinfo("기본 모델 저장", f"‘{ko(model.name)} ({model.window_label})’을 기본 모델로 저장했습니다.")
        self.show_management()

    def selected_managed_model(self) -> ModelDescriptor | None:
        selected = self.managed_model_tree.selection()
        model = self._managed_models.get(selected[0]) if selected else None
        if model is None:
            messagebox.showerror("모델 선택", "처리할 모델을 목록에서 선택하세요.")
        return model

    def update_managed_review_actions(self, _event=None) -> None:
        selected = self.managed_model_tree.selection()
        model = self._managed_models.get(selected[0]) if selected else None
        reviewable = model is not None and model.approval_status == "pending" and model.revocation_status == "active"
        ready = reviewable and model.validation_status == model.deployment_validation_status == "passed" and model.runtime_compatible and model.sensor_compatible
        self.managed_approve_button.configure(state="normal" if ready else "disabled")
        self.managed_reject_button.configure(state="normal" if reviewable else "disabled")

    def validate_managed_runtime(self) -> None:
        model = self.selected_managed_model()
        if model is None: return
        try:
            updated = self.registry.validate_runtime(model)
        except (ValueError, OSError) as error:
            messagebox.showerror("실행 호환성 검증 실패", ko(error)); return
        if updated.validation_status != "passed":
            messagebox.showerror("실행 호환성 검증 실패", ko(updated.lifecycle_reason or "검증에 실패했습니다."))
        else:
            messagebox.showinfo("실행 호환성 검증 완료", "해시·런타임·센서·입출력 계약을 통과했습니다.")
        self.show_management()

    def validate_managed_deployment(self) -> None:
        model = self.selected_managed_model()
        if model is None: return
        try:
            updated = self.registry.validate_deployment(model)
        except (OSError, ValueError) as error:
            messagebox.showerror("배포 검증 실패", ko(error)); return
        if updated.deployment_validation_status != "passed":
            messagebox.showerror("배포 검증 실패", ko(updated.lifecycle_reason or "검증에 실패했습니다."))
        else:
            messagebox.showinfo("배포 검증 완료", "기준 데이터 재생·평가 데이터 분리·자세 의미 연결·장치 허용 오차 검증을 통과했습니다.")
        self.show_management()

    def approve_managed_model(self) -> None:
        model = self.selected_managed_model()
        if model is None: return
        try:
            self.registry.approve(model, approved_by=__import__("getpass").getuser())
        except (OSError, ValueError) as error:
            messagebox.showerror("승인 실패", ko(error)); return
        self.show_management()

    def reject_managed_model(self) -> None:
        model = self.selected_managed_model()
        if model is None: return
        try:
            self.registry.reject(model, reason="사용자가 모델 관리에서 거절함")
        except (OSError, ValueError) as error:
            messagebox.showerror("거절 저장 실패", ko(error)); return
        self.show_management()

    def revoke_managed_model(self) -> None:
        model = self.selected_managed_model()
        if model is None: return
        reason = simpledialog.askstring("모델 폐기", "폐기 사유를 입력하세요.", parent=self)
        if not reason: return
        try:
            self.registry.revoke(model, reason=reason)
        except (OSError, ValueError) as error:
            messagebox.showerror("모델 폐기 실패", ko(error)); return
        self.show_management()

    def activate_managed_model(self) -> None:
        model = self.selected_managed_model()
        if model is not None: self.activate_model(model)

    def rollback_model(self):
        try:
            self.selected_model = self.registry.rollback()
            self.selected_window = self.selected_model.window.default_seconds
            self.show_management()
        except (OSError, ValueError) as error:
            messagebox.showerror("이전 모델 복원 불가", ko(error))

    def show_personal_report(self, path=None):
        path = path or getattr(self, "_personal_choices", {}).get(self.personal_model_choice.get())
        if path is None:
            messagebox.showinfo("개인 기준 모델", "개인 기준 2단계 학습을 완료한 모델이 없습니다.")
            return
        try:
            PersonalPostureModel.load(path)
            content = (Path(path).parent / "personalized.report.md").read_text(encoding="utf-8")
        except (OSError, ValueError) as error:
            messagebox.showerror("평가 결과 오류", str(error)); return
        dialog = tk.Toplevel(self); dialog.title("개인 기준 모델 · 세 방식 비교")
        dialog.geometry("900x600")
        text = tk.Text(dialog, wrap="word", padx=15, pady=15)
        text.pack(fill="both", expand=True); text.insert("1.0", content); text.configure(state="disabled")

    def start_training(self) -> None:
        if self.training_job and self.training_job.is_running:
            return
        from src.experiment_workflow import build_configuration, validate_registration
        selected_paths = set(self.training_wizard_state.selected_record_paths)
        records = [r for r in self.dataset_service.records() if r.csv_path.resolve() in selected_paths]
        try:
            if not records or len(records) != len(selected_paths):
                raise ValueError("선택 데이터가 변경되었거나 없습니다. 데이터 단계에서 다시 확인하세요.")
            config = build_configuration(self.training_wizard_state, allow_retrospective=self.personal_retrospective.get())
            validate_registration(records, config)
            self.training_job = TrainingJob(PROJECT_ROOT, records, config['seed'],
                windows=tuple(config['windows']), models=tuple(config['models']), state_root=STATE_ROOT,
                frozen=bool(getattr(sys, 'frozen', False)), personalized=config['personalized'],
                allow_retrospective=config['allow_retrospective'], soft_voting_members=tuple(config['soft_voting_members']),
                experiment_config=config, training_method=config['training_method'],
                selection_metric=config['selection_metric'] or 'loso')
            self.training_job.start()
        except (OSError, ValueError) as error:
            messagebox.showerror("실험 시작 오류", ko(error)); return
        self.training_wizard_state.current_step = 4
        self.show_training_progress()

    def show_training_progress(self) -> None:
        frame = self.clear()
        self.heading(frame, "모델 실험 · 실행 중", "등록 기록 분리 → 피험자 분할 → 세 방식 학습 → 독립 평가 → 실험용 모델 저장 검증"
                     if self.training_job and self.training_job.personalized else "데이터 검증 → 창 생성 → 특징 추출 → 피험자 분할 → 학습 → 평가 → 후보 저장 → 실행 호환성 검증 → 배포 검증")
        self.training_status = tk.StringVar(value="학습 프로세스를 시작했습니다.")
        ttk.Label(frame, textvariable=self.training_status).pack(anchor="w")
        self.training_log = tk.Text(frame, height=20, state="disabled", font=("Apple SD Gothic Neo", 11)); self.training_log.pack(fill="both", expand=True, pady=12)
        self.training_actions = ttk.Frame(frame); self.training_actions.pack(fill="x")
        ttk.Button(self.training_actions, text="학습 취소", command=self.cancel_training).pack(side="left")
        self.after(150, self.poll_training)

    def poll_training(self) -> None:
        if self._closing or self.training_job is None: return
        while True:
            try: kind, payload = self.training_job.events.get_nowait()
            except queue.Empty: break
            if kind == "log":
                self.training_log.configure(state="normal"); self.training_log.insert("end", ko(payload) + "\n"); self.training_log.see("end"); self.training_log.configure(state="disabled")
            elif kind == "status":
                self.training_status.set(str(payload))
            elif kind == "finished":
                if self.training_job.personalized:
                    self.training_status.set("개인 기준 모델 학습·저장 검증 완료 · 실험용" if payload == "0" else "개인 기준 학습이 실패했거나 취소되었습니다.")
                    for button in self.training_actions.winfo_children(): button.destroy()
                    if payload == "0":
                        ttk.Button(self.training_actions, text="실험 결과 보기", command=lambda: self.show_results(self.training_job.comparison_report_path)).pack(side="left")
                    else:
                        ttk.Button(self.training_actions, text="학습 설정으로", command=self.show_training_wizard_setup).pack(side="left")
                    ttk.Button(self.training_actions, text="측정 방식 선택", command=self.show_mode_select).pack(side="right")
                    return
                failed = sum(not item["passed"] for item in self.training_job.validation_results)
                self.training_status.set((f"학습·자동 검증 완료 · 검증 미통과 {failed}개 (사유를 확인하세요)." if failed else "학습·자동 검증 완료 · 모델 관리에서 승인 또는 거절하세요.") if payload == "0" else "학습이 실패했거나 취소되었습니다.")
                for button in self.training_actions.winfo_children(): button.destroy()
                ttk.Button(self.training_actions, text="모델 관리 · 승인 / 거절", command=self.show_management).pack(side="left")
                if payload != "0":
                    ttk.Button(self.training_actions, text="학습 설정으로", command=self.show_training_wizard_setup).pack(side="left", padx=8)
                ttk.Button(self.training_actions, text="실험 결과 보기", command=lambda: self.show_results(self.training_job.comparison_report_path)).pack(side="right")
                return
        self.after(150, self.poll_training)

    def cancel_training(self) -> None:
        if self.training_job: self.training_job.cancel(); self.training_status.set("학습 취소를 요청했습니다.")

    def show_results(self, preferred_report=None) -> None:
        frame = self.clear(); self.heading(frame, "모델 결과 비교", "신규 후보는 승인 대기로만 보관되며 현재 모델과 기본 모델은 바뀌지 않습니다.")
        report_roots = {STATE_ROOT / "artifacts" / "models", PROJECT_ROOT / "artifacts" / "models"}
        reports = sorted({path for root in report_roots for path in root.glob("**/model_comparison.json")},
                         key=lambda path: path.stat().st_mtime, reverse=True)
        personal_roots = {STATE_ROOT / "artifacts" / "personalized", PROJECT_ROOT / "artifacts" / "personalized"}
        reports += sorted({p for root in personal_roots for p in root.glob("**/personalized.report.json")}, key=lambda p: p.stat().st_mtime, reverse=True)
        reports = sorted(reports, key=lambda p: p.stat().st_mtime, reverse=True)
        if preferred_report and Path(preferred_report).is_file():
            reports = [Path(preferred_report), *(p for p in reports if p != Path(preferred_report))]
        if reports:
            self.show_comparison_report(frame, reports)
            return
        tree = ttk.Treeview(frame, columns=("상태", "창", "기록 단위 Macro F1", "학습 피험자", "분석 시작 대기"), show="headings", height=8)
        for key, label in (("상태", "상태"), ("창", "창"), ("기록 단위 Macro F1", "기록 단위 Macro F1"), ("학습 피험자", "학습 피험자"), ("분석 시작 대기", "분석 시작 대기")):
            tree.heading(key, text=label); tree.column(key, width=150, anchor="center")
        active = self.registry.active_model(self.registry.discover())
        candidate_details = []
        for manifest_path in TrainingJob.pending_candidate_manifests(STATE_ROOT):
            try:
                candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
                metadata = candidate.get("metadata", {})
                summary = metadata.get("fixed_configuration_group_cv_summary", {})
                seconds = candidate["window"]["values_seconds"][0]
                tree.insert("", "end", values=("승인 대기", f"{seconds:g}초", summary.get("record_macro_f1", "기록 없음"), metadata.get("training_subject_count", "기록 없음"), f"{metadata.get('minimum_window_seconds', seconds):g}초"))
                candidate_details.append((seconds, summary, manifest_path))
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        for model in self.registry.discover():
            summary = model.metadata.get("fixed_configuration_group_cv_summary", {})
            tree.insert("", "end", values=("활성" if active and active.path == model.path else "보관", model.window_label, summary.get("record_macro_f1", "기록 없음"), model.metadata.get("training_subject_count", "기록 없음"), f"{model.window_seconds:g}초"))
        tree.pack(fill="both", expand=True)
        detail = tk.Text(frame, height=8, state="normal", font=("Apple SD Gothic Neo", 11))
        detail.pack(fill="x", pady=10)
        detail.insert("end", "세부 평가 결과\n")
        for seconds, summary, manifest_path in candidate_details:
            detail.insert("end", f"• 승인 대기 {seconds:g}초: 창 Macro F1={summary.get('window_macro_f1', '기록 없음')}, 기록 Macro F1={summary.get('record_macro_f1', '기록 없음')} · {manifest_path.parent.parent.parent}\n")
        for model in self.registry.discover():
            summary = model.metadata.get("fixed_configuration_group_cv_summary", {})
            detail.insert("end", f"• {model.window_label}: 창 Macro F1={summary.get('window_macro_f1', '기록 없음')}, 기록 Macro F1={summary.get('record_macro_f1', '기록 없음')}, 분석 시작 대기={model.minimum_window_seconds if hasattr(model, 'minimum_window_seconds') else model.window_seconds:g}초\n")
        detail.insert("end", "자세별 F1와 Confusion Matrix은 재학습 결과 폴더에 표와 그림으로 함께 저장됩니다.")
        detail.configure(state="disabled")
        ttk.Button(frame, text="데이터 및 모델 관리", command=self.show_management).pack(anchor="w")

    def show_comparison_report(self, frame, reports):
        os.environ.setdefault("MPLCONFIGDIR", str(STATE_ROOT / ".cache" / "matplotlib"))
        from software.comparison_view import ComparisonView

        self.comparison_view = ComparisonView(frame, reports, clone_callback=self.clone_experiment)
        self.comparison_view.pack(fill="both", expand=True)
        ttk.Button(frame, text="데이터 및 모델 관리", command=self.show_management).pack(anchor="w")

    def _close(self) -> None:
        management = getattr(self, "model_management", None)
        if management is not None and management.busy:
            self.after(100, self._close)
            return
        self._closing = True
        self._stop.set()
        if self.training_job:
            self.training_job.cancel()
        self._finish_close()

    def _finish_close(self) -> None:
        training = self.training_job
        if ((self._worker and self._worker.is_alive())
                or (training and training.is_running)):
            self.after(80, self._finish_close)
            return
        self.destroy()


def datetime_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def run_application(application_class: type[PostureApplication] = PostureApplication) -> None:
    """Run a product entrypoint while preserving shared worker/self-check modes."""

    if "--training-worker" in sys.argv:
        sys.argv = [sys.argv[0], *sys.argv[sys.argv.index("--training-worker") + 1:]]
        from train_window_models import main as train_main
        train_main()
    elif "--gui-self-check" in sys.argv:
        from validate_integrated_gui import main as gui_check_main
        gui_check_main(sys.argv[sys.argv.index("--gui-self-check") + 1:])
    elif "--charts-self-check" in sys.argv:
        from validate_comparison_gui import main as charts_check_main
        sys.argv = [sys.argv[0], *sys.argv[sys.argv.index("--charts-self-check") + 1:]]
        charts_check_main()
    elif "--self-check" in sys.argv:
        from packaging_self_check import main as check_main
        raise SystemExit(check_main(PROJECT_ROOT, argv=sys.argv[sys.argv.index("--self-check") + 1:]))
    else:
        application_class().mainloop()


if __name__ == "__main__":
    run_application()
