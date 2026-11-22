"""Live shared-window inference and alert policy for the personalized hierarchy."""
from collections import deque
import math

import numpy as np

from .personal_guidance import GuidanceEvent, PostureAlertTimer, AlertSettings, _CORRECTION
from .personalized_model import WINDOW_SECONDS, SAMPLING_RATE, TARGET_SAMPLES
from .posture_core import CLASS_NAMES


class PersonalGuidanceEngine:
    def __init__(self, profile, model, *, settings=None, context=None):
        profile.validate(context)
        self.profile, self.model = profile, model
        # Use the freshly measured offset with the saved, compatible reference.
        from dataclasses import replace
        if context is not None:
            self.profile = replace(profile, context=context)
        self.settings = settings or AlertSettings()
        self.timer = PostureAlertTimer(self.settings)
        self.frames = deque()
        self.last_time = self.last_inference = None
        self.prediction = None
        self.candidate = None
        self.candidate_count = 0
        self.confirmed = None
        self.last_state = None
        self.last_window_timestamp = None

    def close_popup(self, timestamp_s=None):
        self.timer.close_popup(timestamp_s)

    def reset(self):
        self.frames.clear()
        self.last_inference = None
        self.prediction = None
        self.candidate = self.confirmed = None
        self.candidate_count = 0
        self.last_window_timestamp = None

    def update(self, timestamp_s, values, *, seated=True, valid=True, connected=True, moving=False, **unused):
        try:
            now = float(timestamp_s)
            raw = np.asarray(values, dtype=float)
            valid = (valid and raw.shape == (16,) and np.isfinite(raw).all() and not (raw < 0).any()
                     and math.isfinite(now) and (self.last_time is None or 0 < now - self.last_time <= 1.0))
        except (ValueError, TypeError):
            valid = False; now = float("nan")
        if not math.isfinite(now) or (self.last_time is not None and now <= self.last_time):
            now = self.last_time if self.last_time is not None else 0.
        self.last_time = now
        state, posture, detail = "insufficient", None, None
        if not valid or not connected:
            self.reset(); state = "input_error"
        elif seated is False:
            self.reset(); state = "empty"
        elif seated is None:
            self.reset()
        else:
            self.frames.append((now, raw.copy()))
            # Retain one sample at/before the window start for interpolation.
            while len(self.frames) > 2 and self.frames[1][0] <= now - WINDOW_SECONDS:
                self.frames.popleft()
            recent = np.stack([x for t, x in self.frames if t >= now - .5])
            pressure = self.profile.context.pressure(recent)
            total = pressure.sum(axis=1)
            if np.any(total <= 1e-6):
                self.reset(); state = "input_error"
            else:
                ratios = pressure / total[:, None]
                moving = moving or (len(recent) >= 3 and (np.ptp(ratios, axis=0).sum() > .15
                                                         or np.ptp(total) / np.mean(total) > .25))
                if moving:
                    self.reset(); state = "movement"
                elif now - self.frames[0][0] + 1e-8 >= WINDOW_SECONDS:
                    if self.last_inference is None or now - self.last_inference >= 1. - 1e-8:
                        times = np.array([t for t, _ in self.frames])
                        values = np.stack([x for _, x in self.frames])
                        grid = now - WINDOW_SECONDS + np.arange(TARGET_SAMPLES) / SAMPLING_RATE
                        window = np.column_stack([np.interp(grid, times, values[:, j]) for j in range(16)])
                        self.prediction = self.model.predict(window, self.profile)
                        self.last_inference = self.last_window_timestamp = now
                        candidate = self.prediction.state
                        if candidate == "uncertain":
                            self.candidate = self.confirmed = None; self.candidate_count = 0
                        else:
                            self.candidate_count = self.candidate_count + 1 if candidate == self.candidate else 1
                            self.candidate = candidate
                            self.confirmed = candidate if self.candidate_count >= 2 else None
                    state = self.confirmed or "uncertain"
                    if self.prediction and self.prediction.state == state:
                        posture = self.prediction.posture
        decision = self.timer.update(now, state)
        if state == "correct" and self.timer.incorrect_since_s is not None and not decision.returned_to_correct:
            state, posture = "uncertain", None
            detail = "정자세 복귀를 확인하고 있습니다."
        messages = {"correct": "정자세", "incorrect": posture or "개인 기준에서 벗어남 · 자세 종류 확인 중",
                    "uncertain": "개인 기준과 비교 중 · 판단 유보", "movement": "움직임 감지 · 판단 유보",
                    "insufficient": "개인 기준 분석 준비 중 · 3초 데이터 수집", "empty": "빈 좌석 · 판단 중지",
                    "input_error": "센서 입력 오류 · 판단 중지"}
        events = list(decision.events)
        if state != self.last_state: events.insert(0, f"state_transition:{self.last_state or 'initial'}->{state}")
        self.last_state = state
        result = self.prediction
        return GuidanceEvent(timestamp_s=now, state=state, message=messages[state], posture=posture,
            reference_score=None, incorrect_duration_s=decision.incorrect_duration_s,
            popup_action=decision.popup_action, popup_open=decision.popup_open,
            correction_guidance=_CORRECTION.get(posture, "등록한 정자세로 돌아오세요."),
            profile_version=self.profile.profile_version, mode=self.settings.mode, alert_settings=self.settings.as_dict(),
            events=tuple(events), detail=detail,
            gate_score=result.incorrect_score if result else None,
            detail_score=result.detail_score if result else None,
            stage2_executed=result.stage2_executed if result else False,
            model_identity=self.model.identity, window_timestamp_s=self.last_window_timestamp)
