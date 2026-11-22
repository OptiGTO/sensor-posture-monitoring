import threading
import unittest

import numpy as np

from src.signal_monitor import CommonScale, SignalBuffer, format_signal, sweep_segments


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class SignalBufferTest(unittest.TestCase):
    def test_keeps_only_the_display_span_and_copies_values(self):
        buffer = SignalBuffer(span_s=2.0)
        frame = np.arange(16, dtype=float)
        for step in range(50):
            self.assertTrue(buffer.push(step * 0.1, frame + step))
        frame[:] = -1  # caller mutation must not reach the buffer
        snapshot = buffer.snapshot()
        self.assertAlmostEqual(snapshot.times[-1], 4.9)
        self.assertGreaterEqual(snapshot.times[0], 4.9 - 2.0 - 1e-9)
        self.assertEqual(snapshot.values.shape[1], 16)
        np.testing.assert_allclose(snapshot.latest, np.arange(16) + 49)
        self.assertEqual(snapshot.total_samples, 50)
        self.assertEqual(snapshot.origin_s, 0.0)
        self.assertAlmostEqual(snapshot.sample_rate_hz(), 10.0)

    def test_invalid_or_out_of_order_frames_are_ignored_without_raising(self):
        buffer = SignalBuffer()
        self.assertTrue(buffer.push(1.0, np.ones(16)))
        self.assertFalse(buffer.push(1.0, np.ones(16)))           # not increasing
        self.assertFalse(buffer.push(0.5, np.ones(16)))           # backwards
        self.assertFalse(buffer.push(2.0, np.ones(15)))           # wrong channel count
        self.assertFalse(buffer.push(2.0, [np.nan] + [1.0] * 15))  # non-finite
        self.assertFalse(buffer.push("x", np.ones(16)))
        self.assertFalse(buffer.push(float("inf"), np.ones(16)))
        self.assertEqual(buffer.snapshot().total_samples, 1)

    def test_rate_ignores_start_up_burst(self):
        # Same shape as example/sample_input.csv: 26 buffered lines ~0.6 ms apart, then 12 Hz.
        buffer = SignalBuffer()
        times = list(np.arange(26) * 0.0006) + list(0.02 + np.arange(1, 14) * 0.082)
        for t in times:
            buffer.push(t, np.ones(16))
        self.assertAlmostEqual(buffer.snapshot().sample_rate_hz(), 1 / 0.082, places=3)

    def test_age_reports_time_since_last_frame(self):
        clock = FakeClock()
        buffer = SignalBuffer(clock=clock)
        self.assertIsNone(buffer.snapshot().age_s)
        buffer.push(0.0, np.ones(16))
        clock.now += 2.5
        self.assertAlmostEqual(buffer.snapshot().age_s, 2.5)

    def test_concurrent_writer_and_reader(self):
        buffer = SignalBuffer(span_s=1.0)
        stop = threading.Event()

        def writer():
            for step in range(3000):
                buffer.push(step / 100, np.full(16, step, dtype=float))
            stop.set()

        thread = threading.Thread(target=writer)
        thread.start()
        while not stop.is_set():
            snapshot = buffer.snapshot()
            self.assertEqual(len(snapshot.times), len(snapshot.values))
            if len(snapshot.times) > 1:
                self.assertTrue(np.all(np.diff(snapshot.times) > 0))
        thread.join()
        self.assertEqual(buffer.snapshot().total_samples, 3000)

    def test_rejects_invalid_span(self):
        for span in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                SignalBuffer(span_s=span)


class SweepSegmentsTest(unittest.TestCase):
    def test_single_segment_before_first_wrap(self):
        times = np.arange(0, 5, 0.1)
        positions, segments = sweep_segments(times, 10.0)
        self.assertEqual(segments, [(0, len(times))])
        np.testing.assert_allclose(positions, times / 10.0)

    def test_wrap_splits_line_and_hides_samples_inside_erase_gap(self):
        times = np.round(np.arange(0, 12.0, 0.1), 6)
        positions, segments = sweep_segments(times, 10.0, erase_fraction=0.05)
        # Only the latest 9.5 s are visible.
        self.assertGreater(times[segments[0][0]], times[-1] - 9.5)
        self.assertEqual(len(segments), 2)
        (a, b), (c, d) = segments
        self.assertEqual(b, c)
        self.assertEqual(d, len(times))
        self.assertTrue(np.all(np.diff(positions[a:b]) > 0))
        self.assertTrue(np.all(np.diff(positions[c:d]) > 0))
        self.assertLess(positions[c], positions[b - 1])  # new sweep restarts at the left
        self.assertAlmostEqual(positions[-1], (times[-1] % 10.0) / 10.0)

    def test_sensor_gap_breaks_the_trace(self):
        times = np.array([0.0, 0.1, 0.2, 1.8, 1.9, 2.0])
        _, segments = sweep_segments(times, 10.0, break_s=1.0)
        self.assertEqual(segments, [(0, 3), (3, 6)])

    def test_origin_starts_the_sweep_at_the_left_edge(self):
        positions, _ = sweep_segments(np.array([7.0, 7.5]), 10.0, origin_s=7.0)
        np.testing.assert_allclose(positions, [0.0, 0.05])

    def test_empty_input(self):
        positions, segments = sweep_segments(np.empty(0), 10.0)
        self.assertEqual(len(positions), 0)
        self.assertEqual(segments, [])


class CommonScaleTest(unittest.TestCase):
    def test_flat_empty_seat_signal_stays_flat(self):
        low, high = CommonScale().update(np.full((30, 16), 1.15) + np.random.default_rng(0).normal(0, 0.002, (30, 16)))
        self.assertLess(low, 1.15)
        self.assertGreater(high, 1.15)
        self.assertGreaterEqual(high - low, 1.15 * 0.25)  # noise uses a small part of the height

    def test_expands_immediately_and_contracts_gradually(self):
        scale = CommonScale(decay=0.2)
        first = scale.update(np.full((10, 16), 1.15))
        loaded = np.full((10, 16), 1.15)
        loaded[:, 5] = 3.1
        expanded = scale.update(loaded)
        self.assertGreater(expanded[1], 3.1)
        self.assertLess(expanded[0], 1.15)
        self.assertLessEqual(expanded[0] - first[0], 0.2 * (1.15 - first[0]))  # lower bound only drifts
        shrunk = scale.update(np.full((10, 16), 1.15))
        self.assertLess(shrunk[1], expanded[1])
        self.assertGreater(shrunk[1], first[1])  # still decaying, not a jump

    def test_all_zero_input_has_positive_span(self):
        low, high = CommonScale().update(np.zeros((3, 16)))
        self.assertGreater(high - low, 0)

    def test_format_signal_matches_units(self):
        self.assertEqual(format_signal(1.1534, 1.0, 3.2), "1.15")
        self.assertEqual(format_signal(512.4, 200.0, 900.0), "512")


class SignalMonitorViewTest(unittest.TestCase):
    """Real Tk drawing check; skipped only when no display is available."""

    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk 화면을 사용할 수 없습니다: {error}")
        self.root.geometry("900x420")

    def tearDown(self):
        self.root.destroy()

    def test_draws_all_sixteen_sweeping_traces_and_highlights_strongest_sensor(self):
        from software.signal_monitor_view import ACCENT, ALERT, STALE, SignalMonitorView
        clock = FakeClock()
        buffer = SignalBuffer(clock=clock)
        view = SignalMonitorView(self.root, buffer)
        view.pack(fill="both", expand=True)
        self.root.update()
        self.assertEqual(len(view._panels), 16)
        self.assertEqual([panel["name"] for panel in view._panels][:2], ["R11", "R12"])
        self.assertIn("기다리는 중", view.summary.get())
        for step in range(150):  # 12.5 s at 12 Hz: the sweep has wrapped once
            values = np.full(16, 1.15)
            values[9] = 2.8 + 0.2 * np.sin(step / 5)  # R32 is loaded
            buffer.push(step / 12, values)
        view.refresh()
        counts = view.trace_point_counts()
        self.assertTrue(all(count >= 100 for count in counts), counts)
        strongest = view._panels[9]
        self.assertEqual(view.canvas.itemcget(strongest["frame"], "outline"), ACCENT)
        self.assertIn("R32", view.summary.get())
        self.assertIn("Hz", view.summary.get())
        head = view.canvas.coords(strongest["head"])
        plot = strongest["plot"]
        self.assertTrue(plot[0] - 4 <= head[0] <= plot[2])
        # Loaded sensor draws higher within its panel than an unloaded one.
        def relative_height(panel):
            top, bottom = panel["plot"][1], panel["plot"][3]
            y = sum(view.canvas.coords(panel["head"])[1::2]) / 2
            return (bottom - y) / (bottom - top)
        self.assertGreater(relative_height(strongest), 0.6)
        self.assertLess(relative_height(view._panels[0]), 0.2)
        self.assertEqual(view.canvas.itemcget(view._panels[0]["value"], "text"), "1.15")
        # No new frames: the monitor reports the interruption and greys out.
        clock.now += 3.0
        view.refresh()
        self.assertIn("신호 끊김", view.summary.get())
        self.assertEqual(str(view.summary_label.cget("foreground")), ALERT)
        self.assertEqual(view.canvas.itemcget(strongest["lines"][0], "fill"), STALE)
        self.assertEqual(view.canvas.itemcget(strongest["head"], "state"), "hidden")

    def test_destroy_cancels_refresh_loop(self):
        from software.signal_monitor_view import SignalMonitorView
        view = SignalMonitorView(self.root, SignalBuffer())
        view.pack()
        self.root.update()
        self.assertIsNotNone(view._after_id)
        view.destroy()
        self.assertIsNone(view._after_id)
        self.root.update()  # no pending callback on a destroyed widget


if __name__ == "__main__":
    unittest.main()
