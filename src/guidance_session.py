"""Single-worker orchestration for baseline checks, registration and guidance."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import queue
import uuid

import numpy as np

from .app_services import OccupancyDetector, StandardProfile
from .personal_guidance import (
    AlertSettings, CalibrationContext, CalibrationSession, CalibrationStore,
    GuidanceEngine,
)


class GuidanceSession:
    """UI commands are queued; all sensor and profile state belongs to the worker."""

    def __init__(self, root: Path, user_id: str, context: CalibrationContext,
                 settings: AlertSettings, emit, recorder=None, *,
                 allow_registration=True, registration_only=False, personal_model=None):
        self.personal_model = personal_model
        self._last_prediction_timestamp = None
        self.store = CalibrationStore(root / "calibration_profiles")
        self.user_id, self.context, self.settings = user_id, context, settings
        self.emit, self.recorder = emit, recorder
        self.allow_registration = allow_registration
        self.registration_only = registration_only
        self.completed_profile_path = None
        self.commands = queue.Queue()
        self.detector = None
        self.engine = None
        self.registration = None
        self.baseline_frames = None
        self.baseline_times = []
        self.last_timestamp = 0.0
        self.last_event = None
        self._message("baseline_required", "Leave the seat empty, then select Check empty seat.")

    def _make_engine(self, profile):
        if self.personal_model is not None:
            from .personalized_guidance import PersonalGuidanceEngine
            return PersonalGuidanceEngine(profile, self.personal_model, settings=self.settings, context=self.context)
        return GuidanceEngine(profile, settings=self.settings, context=self.context)

    def _log(self, payload):
        if self.recorder:
            self.recorder.write_guidance({
                **payload, "user_id": self.user_id, "alert_settings": asdict(self.settings),
                "model_snapshot": self.recorder.model_snapshot.as_dict(),
            })

    def _message(self, state, message, **extra):
        payload = {"state": state, "message": message, "timestamp_s": self.last_timestamp, **extra}
        self.emit(("guidance_status", payload))
        return payload

    def _publish(self, event):
        payload = event.as_dict()
        self.last_event = event
        self._log({"kind": "guidance", **payload})
        self.emit(("guidance", event))
        if self.personal_model is not None and self.recorder and event.window_timestamp_s is not None and event.window_timestamp_s != self._last_prediction_timestamp:
            self._last_prediction_timestamp = event.window_timestamp_s
            self.recorder.write_prediction({"kind": "personalized_hierarchy", **payload,
                                           "model_snapshot": self.recorder.model_snapshot.as_dict()})

    def command(self, kind):
        self.commands.put(kind)

    def _commands(self):
        while True:
            try:
                kind = self.commands.get_nowait()
            except queue.Empty:
                return
            if kind == "close_popup":
                if self.engine:
                    self.engine.close_popup(self.last_timestamp)
                    self._log({"kind": "popup_closed", "timestamp_s": self.last_timestamp})
            elif kind == "baseline":
                self.engine = None
                self.registration = None
                self.detector = None
                self.baseline_frames, self.baseline_times = [], []
                self._message("baseline_collecting", "Keep the seat empty and still for 3 seconds.", popup_action="dismiss")
            elif kind == "register":
                if not self.allow_registration:
                    self._message("profile_required", "개인 기준은 측정을 종료한 뒤 ‘정자세 등록’ 모드에서 등록하세요.")
                    continue
                if self.detector is None or not self.detector.occupied:
                    self._message("profile_required", "Check the empty seat first, then sit down before registration.")
                    continue
                self.engine = None
                self.registration = CalibrationSession(
                    self.user_id, self.context, guidance_confirmed=True, empty_seat_checked=True,
                )
                self._message("registering", "Maintain the instructed correct posture for 10 seconds.", popup_action="dismiss")

    def interrupt(self, state="input_error", timestamp_s=None):
        self._commands()
        self.last_timestamp = max(self.last_timestamp + 1e-6, timestamp_s if timestamp_s is not None else self.last_timestamp)
        self.registration = None
        self.baseline_frames = None
        self.baseline_times = []
        if self.detector is not None:
            self.detector = OccupancyDetector(self.detector.profile, self.detector.config)
        if self.engine:
            self._publish(self.engine.update(self.last_timestamp, [0.] * 16, valid=False, connected=False))
        self._message(state, "Sensor input unavailable. Classification and alerts are paused.", popup_action="dismiss")

    def push(self, timestamp_s, values, prediction=None):
        self.last_timestamp = timestamp_s
        self._commands()
        values = np.asarray(values, dtype=float)
        if values.shape != (16,) or not np.isfinite(values).all() or np.any(values < 0):
            self.interrupt(timestamp_s=timestamp_s)
            return False
        if self.baseline_frames is not None:
            self.baseline_frames.append(values.copy())
            self.baseline_times.append(timestamp_s)
            if timestamp_s - self.baseline_times[0] < 3:
                return False
            samples = np.stack(self.baseline_frames)
            self.baseline_frames = None
            center = np.median(samples, axis=0)
            scale = max(float(np.max(np.abs(center))), 1e-6)
            if len(samples) < 20 or np.max(np.std(samples, axis=0)) > scale * .03:
                self._message("baseline_required", "The empty-seat reading was unstable. Repeat the check.")
                return False
            deviation = np.maximum(np.std(samples, axis=0), scale * .002)
            signature = hashlib.sha256(center.tobytes() + deviation.tobytes()).hexdigest()
            self.detector = OccupancyDetector(StandardProfile(center, deviation, signature, 1))
            self.context = replace(self.context, empty_seat_center=tuple(center))
            baseline = {"kind": "empty_seat_baseline", "timestamp_s": timestamp_s,
                        "context": asdict(self.context), "deviation": deviation.tolist(), "signature": signature}
            self._log(baseline)
            baseline_root = self.store.root if hasattr(self.store, "root") else None
            # Baseline evidence is kept separately from personal profile records.
            if baseline_root is not None:
                baseline_root = Path(baseline_root) / "empty_seat"
                baseline_root.mkdir(parents=True, exist_ok=True)
                (baseline_root / f"{uuid.uuid4().hex}.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
            if self.registration_only:
                self._message("profile_required", "빈 좌석 확인 완료. 정자세로 앉은 뒤 ‘정자세 등록’을 누르세요.")
                return False
            try:
                profile = self.store.load(self.user_id, self.context)
                if profile is None:
                    raise ValueError("No saved reference for this user.")
                saved_center = np.asarray(profile.context.empty_seat_center)
                if np.max(np.abs(saved_center - center)) > scale * .05:
                    raise ValueError("The baseline changed. Re-register the correct posture.")
                self.engine = self._make_engine(profile)
                self._message("profile_loaded", "Personal reference loaded. Sit down to begin.")
            except (OSError, ValueError, KeyError, TypeError) as error:
                self._message("profile_required", f"개인 기준을 사용할 수 없습니다. 측정을 종료한 뒤 ‘정자세 등록’ 모드에서 다시 등록하세요: {error}")
            return False
        if self.detector is None:
            return False
        self.detector.update(timestamp_s, values)
        seated = self.detector.occupied
        if self.registration is not None:
            try:
                progress = self.registration.push(timestamp_s, values, seated=seated)
                self._message("registering", f"Registering correct posture: {progress * 100:.0f}%")
                if progress >= 1:
                    profile = self.registration.finish()
                    path = self.store.save(profile)
                    self.engine = None if self.registration_only else self._make_engine(profile)
                    self.registration = None
                    self._log({"kind": "registration_complete", "timestamp_s": timestamp_s, "profile": profile.as_dict()})
                    if self.registration_only:
                        self.completed_profile_path = path
                        self._message("registration_complete", "정자세 기준을 저장했습니다.", profile_path=str(path))
                    else:
                        self._message("profile_loaded", "Reference saved. Personal posture guidance is active.", profile_path=str(path))
            except ValueError as error:
                self.registration = None
                self._message("profile_required", f"Registration failed. Repeat the measurement: {error}")
            return False
        if self.engine:
            self._publish(self.engine.update(
                timestamp_s, values, seated=seated,
                model_posture=prediction.posture if prediction else None,
                model_confidence=prediction.confidence if prediction else None,
                model_timestamp_s=prediction.timestamp_s if prediction else None,
            ))
        elif not seated:
            self._message("empty_seat", "The seat is empty.", popup_action="dismiss")
        return (seated and self.engine is not None and self.last_event is not None
                and self.last_event.state not in {"input_error", "empty", "profile_required"})
