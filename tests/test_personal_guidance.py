from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.personal_guidance import (
    AlertSettings,
    CalibrationConfig,
    CalibrationContext,
    CalibrationProfile,
    CalibrationSession,
    CalibrationStore,
    GuidanceConfig,
    GuidanceEngine,
    PostureAlertTimer,
)
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS


def reference(user: str = "test-user", context: CalibrationContext | None = None) -> CalibrationProfile:
    context = context or CalibrationContext()
    session = CalibrationSession(user, context, guidance_confirmed=True, empty_seat_checked=True)
    for timestamp in np.arange(0, 10.01, 0.1):
        session.push(timestamp, np.asarray(context.empty_seat_center) + np.ones(16))
    return session.finish()


def bad_pressure(reverse: bool = False) -> np.ndarray:
    values = np.ones(16)
    values[:8] = 2.0
    return values[::-1].copy() if reverse else values


class CalibrationTests(unittest.TestCase):
    def test_registration_requires_guidance_and_separate_empty_seat_check(self) -> None:
        for guided, empty in ((False, True), (True, False), (False, False)):
            with self.subTest(guided=guided, empty=empty), self.assertRaises(ValueError):
                CalibrationSession("a", CalibrationContext(), guidance_confirmed=guided, empty_seat_checked=empty)

    def test_ten_second_stable_registration_preserves_raw_pressure_and_ratios(self) -> None:
        context = CalibrationContext(empty_seat_center=(1.2,) * 16, chair_id="chair-a", placement_id="pad-v2")
        session = CalibrationSession(" person ", context, guidance_confirmed=True, empty_seat_checked=True)
        raw = np.full(16, 2.2)
        for timestamp in np.arange(0, 10, 0.1):
            self.assertLess(session.push(timestamp, raw), 1.0)
        with self.assertRaisesRegex(ValueError, "10 seconds"):
            session.finish()
        self.assertEqual(session.push(10, raw), 1.0)
        profile = session.finish()
        self.assertEqual(profile.user_id, "person")
        np.testing.assert_allclose(profile.pressure_center, raw)
        np.testing.assert_allclose(profile.ratio_center, np.full(16, 1 / 16))
        self.assertEqual(profile.stability_metrics["sample_count"], 101)
        self.assertEqual(profile.stability_metrics["duration_s"], 10)
        self.assertIn("provisional", profile.thresholds_status)
        self.assertEqual(CalibrationProfile.from_dict(json.loads(json.dumps(profile.as_dict()))), profile)

    def test_movement_invalid_input_and_sensor_gap_invalidate_entire_attempt(self) -> None:
        for case in ("movement", "nan", "empty", "gap", "ratio_jump", "load_jump", "negative"):
            with self.subTest(case=case):
                session = CalibrationSession("a", CalibrationContext(), guidance_confirmed=True, empty_seat_checked=True)
                session.push(0, np.ones(16))
                values = np.ones(16)
                options = {}
                timestamp = 0.1
                if case == "movement": options["moving"] = True
                elif case == "nan": values[0] = np.nan
                elif case == "empty": options["seated"] = False
                elif case == "gap": timestamp = 2
                elif case == "ratio_jump": values[:8], values[8:] = 1.7, 0.3
                elif case == "load_jump": values *= 2
                elif case == "negative": values[0] = -0.1
                with self.assertRaisesRegex(ValueError, "Repeat"):
                    session.push(timestamp, values, **options)
                with self.assertRaises(ValueError):
                    session.push(2.1, np.ones(16))
                with self.assertRaises(ValueError):
                    session.finish()

    def test_slow_drift_is_rejected_even_without_large_frame_steps(self) -> None:
        session = CalibrationSession("a", CalibrationContext(), guidance_confirmed=True, empty_seat_checked=True)
        for timestamp in np.arange(0, 10.01, 0.1):
            values = np.ones(16)
            values[:8] += timestamp * 0.075
            values[8:] -= timestamp * 0.075
            session.push(timestamp, values)
        with self.assertRaisesRegex(ValueError, "not stable"):
            session.finish()

    def test_reregistration_is_versioned_and_context_changes_require_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CalibrationStore(directory)
            initial = reference("../user")
            saved = store.save(initial)
            self.assertTrue(saved.is_relative_to(directory))
            self.assertEqual(store.load("../user", CalibrationContext()), initial)
            updated = reference("../user")
            self.assertNotEqual(updated.profile_version, initial.profile_version)
            store.save(updated)
            self.assertEqual(store.load("../user"), updated)
            self.assertTrue(saved.with_name(f"{initial.profile_version}.json").is_file())
            self.assertEqual(store.list_profiles(), [updated])
            for context in (CalibrationContext(chair_id="new"), CalibrationContext(placement_id="moved"), CalibrationContext(sensor_configuration="new-sensor")):
                with self.subTest(context=context), self.assertRaisesRegex(ValueError, "register"):
                    store.load("../user", context)
            self.assertIsNone(store.load("missing"))

    def test_invalid_profiles_are_rejected(self) -> None:
        profile = reference()
        invalid = (
            replace(profile, schema_version=999),
            replace(profile, ratio_center=(0.0,) * 16),
            replace(profile, ratio_tolerance=(0.0,) * 16),
            replace(profile, ratio_tolerance=(999.0,) * 16),
            replace(profile, pressure_lower=(2.0,) * 16),
            replace(profile, guidance_confirmed=False),
            replace(profile, stability_metrics={**profile.stability_metrics, "total_cv": 0.99}),
            replace(profile, stability_metrics={**profile.stability_metrics, "duration_s": 1}),
            replace(profile, stability_metrics={**profile.stability_metrics, "sample_count": 5}),
            replace(profile, profile_version="../../outside"),
        )
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                candidate.validate()
        with self.assertRaises(ValueError):
            CalibrationContext(sensor_channels=tuple(reversed(SENSOR_COLUMNS)))

    def test_corrupt_saved_json_never_becomes_a_usable_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CalibrationStore(directory)
            profile = reference()
            path = store.save(profile)
            # A non-finite sensor value remains invalid even though Python's JSON
            # reader accepts NaN tokens by default.
            record = profile.as_dict()
            record["ratio_center"] = [float("nan")] + list(profile.ratio_center[1:])
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "register again"):
                store.load(profile.user_id)
            self.assertEqual(store.list_profiles(), [])

    def test_baseline_pressure_direction_is_explicit(self) -> None:
        raw = np.full(16, 1.0)
        for mode, expected in (("absolute_delta", 1), ("decreasing", 1), ("increasing", 0)):
            context = CalibrationContext(empty_seat_center=(2.0,) * 16, pressure_direction=mode)
            np.testing.assert_allclose(context.pressure(raw), np.full(16, expected))


class AlertTimerTests(unittest.TestCase):
    def test_default_deployment_and_development_exact_initial_and_realert_times(self) -> None:
        self.assertEqual(AlertSettings().mode, "deployment")
        self.assertEqual((AlertSettings().initial_alert_s, AlertSettings().realert_interval_s), (60, 120))
        for mode, initial, repeat in (("development", 12, 24), ("deployment", 60, 120)):
            with self.subTest(mode=mode):
                timer = PostureAlertTimer(AlertSettings.for_mode(mode))
                timer.update(0, "incorrect")
                self.assertIsNone(timer.update(initial - 0.001, "incorrect").popup_action)
                first = timer.update(initial, "incorrect")
                self.assertEqual(first.popup_action, "show")
                self.assertEqual(first.incorrect_duration_s, initial)
                self.assertIsNone(timer.update(initial + 1, "incorrect").popup_action)
                timer.close_popup(initial + 1)
                self.assertEqual(timer.incorrect_duration_s, initial + 1)
                self.assertEqual(timer.last_popup_s, initial)
                self.assertIsNone(timer.update(initial + repeat - 0.001, "incorrect").popup_action)
                self.assertEqual(timer.update(initial + repeat, "incorrect").popup_action, "show")

    def test_correct_return_has_independent_confirmation_and_dismisses_popup(self) -> None:
        timer = PostureAlertTimer(AlertSettings.for_mode("development", correct_return_confirmation_s=2, uncertainty_tolerance_s=5))
        timer.update(0, "incorrect")
        timer.update(12, "incorrect")
        self.assertIsNone(timer.update(13, "correct").popup_action)
        self.assertEqual(timer.update(14.999, "correct").incorrect_duration_s, 12)
        returned = timer.update(15, "correct")
        self.assertEqual(returned.popup_action, "dismiss")
        self.assertEqual(returned.incorrect_duration_s, 0)
        self.assertIn("returned_to_correct", returned.events)
        timer.update(16, "incorrect")
        self.assertIsNone(timer.update(28, "incorrect").popup_action)  # Previous popup cooldown survives recovery.
        self.assertEqual(timer.update(36, "incorrect").popup_action, "show")

    def test_short_uncertainty_pauses_and_long_uncertainty_resets(self) -> None:
        timer = PostureAlertTimer(AlertSettings.for_mode("development"))
        timer.update(0, "incorrect")
        self.assertEqual(timer.update(10, "incorrect").incorrect_duration_s, 10)
        self.assertEqual(timer.update(11, "uncertain").incorrect_duration_s, 10)
        self.assertEqual(timer.update(12, "movement").incorrect_duration_s, 10)
        self.assertEqual(timer.update(13, "incorrect").incorrect_duration_s, 10)
        self.assertEqual(timer.update(15, "incorrect").popup_action, "show")
        timer.update(16, "uncertain")
        timeout = timer.update(19, "uncertain")
        self.assertEqual(timeout.incorrect_duration_s, 0)
        self.assertEqual(timeout.popup_action, "dismiss")
        self.assertIn("uncertainty_timeout", timeout.events)
        self.assertEqual(timer.update(20, "incorrect").incorrect_duration_s, 0)

    def test_uncertainty_timeout_also_checked_when_next_state_resumes(self) -> None:
        timer = PostureAlertTimer(AlertSettings.for_mode("development"))
        timer.update(0, "incorrect")
        timer.update(8, "incorrect")
        timer.update(9, "uncertain")
        resumed = timer.update(12, "incorrect")
        self.assertEqual(resumed.incorrect_duration_s, 0)
        self.assertIn("uncertainty_timeout", resumed.events)

    def test_empty_disconnect_and_invalid_input_clear_duration_and_popup(self) -> None:
        for reset_state in ("empty", "disconnected", "input_error", "profile_required"):
            with self.subTest(state=reset_state):
                timer = PostureAlertTimer(AlertSettings.for_mode("development"))
                timer.update(0, "incorrect")
                timer.update(12, "incorrect")
                reset = timer.update(13, reset_state)
                self.assertEqual(reset.incorrect_duration_s, 0)
                self.assertEqual(reset.popup_action, "dismiss")
                self.assertFalse(reset.popup_open)


class GuidanceEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = reference()

    def engine(self, **config) -> GuidanceEngine:
        return GuidanceEngine(self.profile, AlertSettings.for_mode("development"), GuidanceConfig(**config))

    def feed(self, engine: GuidanceEngine, begin: float, end: float, values: np.ndarray, **options):
        return [engine.update(round(float(t), 5), values, **options) for t in np.arange(begin, end + 0.00001, 0.1)]

    def test_reference_and_model_conflict_allows_general_alert_and_keeps_raw_data(self) -> None:
        engine = self.engine()
        values = bad_pressure()
        original = values.copy()
        events = self.feed(engine, 0, 13, values, model_posture=CLASS_NAMES[0], model_confidence=0.99)
        self.assertEqual(events[-1].state, "incorrect")
        self.assertIn("Verifying posture type", events[-1].message)
        self.assertIsNone(events[-1].posture)
        shows = [event for event in events if event.popup_action == "show"]
        self.assertEqual(len(shows), 1)
        self.assertAlmostEqual(shows[0].incorrect_duration_s, 12)
        self.assertLessEqual(shows[0].timestamp_s, 12.3)
        np.testing.assert_array_equal(values, original)
        self.assertEqual(events[-1].profile_version, self.profile.profile_version)
        json.dumps(events[-1].as_dict(), allow_nan=False)

    def test_type_changes_keep_duration_and_stale_or_low_confidence_types_are_hidden(self) -> None:
        engine = self.engine()
        first = self.feed(engine, 0, 6, bad_pressure(), model_posture=CLASS_NAMES[3], model_confidence=0.9)[-1]
        second = self.feed(engine, 6.1, 12.2, bad_pressure(), model_posture=CLASS_NAMES[5], model_confidence=0.9)[-1]
        self.assertEqual(first.posture, CLASS_NAMES[3])
        self.assertEqual(second.posture, CLASS_NAMES[5])
        self.assertGreater(second.incorrect_duration_s, first.incorrect_duration_s)
        self.assertEqual(second.popup_action, "show")
        stale = engine.update(12.3, bad_pressure(), model_posture=CLASS_NAMES[3], model_confidence=0.9, model_timestamp_s=4)
        self.assertIsNone(stale.posture)
        low = engine.update(12.4, bad_pressure(), model_posture=CLASS_NAMES[3], model_confidence=0.1)
        self.assertIsNone(low.posture)

    def test_engine_recovery_keeps_sampling_then_dismisses_single_popup(self) -> None:
        engine = self.engine()
        initial = self.feed(engine, 0, 13, bad_pressure())
        recovered = self.feed(engine, 13.1, 17, np.ones(16))
        self.assertEqual(sum(event.popup_action == "show" for event in initial), 1)
        self.assertEqual(sum(event.popup_action == "dismiss" for event in recovered), 1)
        self.assertEqual(recovered[-1].state, "correct")
        self.assertEqual(recovered[-1].message, "Posture is correct.")
        self.assertEqual(recovered[-1].incorrect_duration_s, 0)
        self.assertTrue(any(event.state == "movement" for event in recovered))
        self.assertTrue(any(event.state == "uncertain" for event in recovered))
        self.assertEqual(len(recovered), 40)

    def test_invalid_seat_status_empty_and_disconnect_do_not_classify(self) -> None:
        for options, expected in (({"seated": False}, "empty"), ({"seated": None}, "insufficient"),
                                  ({"valid": False}, "input_error"), ({"connected": False}, "input_error")):
            with self.subTest(options=options):
                engine = self.engine()
                self.feed(engine, 0, 13, bad_pressure())
                event = engine.update(13.1, bad_pressure(), **options)
                self.assertEqual(event.state, expected)
                self.assertIsNone(event.posture)
                if expected != "insufficient":
                    self.assertEqual(event.incorrect_duration_s, 0)
                    self.assertEqual(event.popup_action, "dismiss")

    def test_gap_resets_and_missing_profile_has_explicit_state(self) -> None:
        engine = self.engine()
        self.feed(engine, 0, 13, bad_pressure())
        event = engine.update(20, bad_pressure())
        self.assertEqual(event.state, "input_error")
        self.assertEqual(event.incorrect_duration_s, 0)
        self.assertEqual(event.popup_action, "dismiss")
        self.assertEqual(GuidanceEngine(None).update(0, np.ones(16)).state, "profile_required")

    def test_bad_timestamp_returns_error_and_dismisses_without_accumulating_time(self) -> None:
        for timestamp in (float("nan"), float("inf"), 2, "bad"):
            with self.subTest(timestamp=timestamp):
                engine = self.engine()
                self.feed(engine, 0, 13, bad_pressure())
                event = engine.update(timestamp, bad_pressure())
                self.assertEqual(event.state, "input_error")
                self.assertEqual(event.incorrect_duration_s, 0)
                self.assertEqual(event.popup_action, "dismiss")
                self.assertEqual(event.timestamp_s, 13)
                self.assertEqual(engine.update(13.1, bad_pressure()).state, "insufficient")

    def test_hysteresis_retains_established_state_and_requires_consecutive_transitions(self) -> None:
        engine = self.engine(movement_ratio_range=1, consecutive_observations=3)
        self.assertEqual(self.feed(engine, 0, 1, np.ones(16))[-1].state, "correct")
        between = np.ones(16)
        # Keep total=16, shift one channel by .16 => ratio deviation=.01, score=1.25.
        between[0] += 0.16
        between[1] -= 0.16
        self.assertEqual(self.feed(engine, 1.1, 2, between)[-1].state, "correct")
        incorrect = self.feed(engine, 2.1, 4, bad_pressure())
        self.assertTrue(any(event.state == "uncertain" for event in incorrect))
        self.assertEqual(incorrect[-1].state, "incorrect")
        self.assertEqual(self.feed(engine, 4.1, 5, between)[-1].state, "incorrect")

    def test_long_movement_cannot_backfill_previous_incorrect_time(self) -> None:
        engine = self.engine()
        self.feed(engine, 0, 10, bad_pressure())
        movement = self.feed(engine, 10.1, 14, bad_pressure(), moving=True)
        self.assertEqual(movement[-1].incorrect_duration_s, 0)
        resumed = self.feed(engine, 14.1, 15, bad_pressure())
        self.assertLess(resumed[-1].incorrect_duration_s, 1)
        self.assertFalse(any(event.popup_action == "show" for event in resumed))

    def test_current_baseline_context_is_independent_of_model_input(self) -> None:
        baseline = CalibrationContext(empty_seat_center=(1.2,) * 16)
        profile = reference(context=baseline)
        updated_baseline = replace(baseline, empty_seat_center=(1.3,) * 16)
        engine = GuidanceEngine(profile, context=updated_baseline)
        raw = np.full(16, 2.3)
        events = self.feed(engine, 0, 1, raw)
        self.assertEqual(events[-1].state, "correct")
        np.testing.assert_array_equal(raw, np.full(16, 2.3))


if __name__ == "__main__":
    unittest.main()
