#!/usr/bin/env python3
"""Run live posture inference from an Arduino serial stream or a recorded CSV."""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_core import PosturePredictor  # noqa: E402
from src.posture_core import SENSOR_COLUMNS, read_sensor_csv  # noqa: E402
from src.realtime import LiveConfig, PredictionEvent, RealtimePosturePredictor  # noqa: E402


MODEL_PATHS = {
    3: PROJECT_ROOT / "artifacts" / "posture_classifier_window_3s.joblib",
    5: PROJECT_ROOT / "artifacts" / "posture_classifier_window_5s.joblib",
    20: PROJECT_ROOT / "artifacts" / "posture_classifier.joblib",
}


def parse_sensor_line(line: str) -> np.ndarray | None:
    """Parse one Arduino line containing exactly sixteen comma-separated numbers."""

    parts = line.strip().split(",")
    if len(parts) != len(SENSOR_COLUMNS):
        return None
    try:
        values = np.asarray([float(value) for value in parts], dtype=float)
    except ValueError:
        return None
    return values if np.isfinite(values).all() else None


def replay_csv(path: Path, speed: float, *, stop_event: threading.Event | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield a saved CSV as a sensor stream, optionally respecting its timing."""
    if not np.isfinite(speed) or speed < 0:
        raise ValueError("CSV replay speed must be finite and nonnegative")
    frame = read_sensor_csv(path)
    timestamps = frame["elapsed_s"].to_numpy(dtype=float)
    values = frame[SENSOR_COLUMNS].to_numpy(dtype=float)
    started = time.monotonic()
    for timestamp, sensor_values in zip(timestamps, values):
        if stop_event is not None and stop_event.is_set():
            return
        if speed > 0:
            # Processing a yielded frame takes time too. Anchor playback to a
            # clock instead of adding that processing time to every CSV interval.
            delay = max(0.0, (timestamp - timestamps[0]) / speed - (time.monotonic() - started))
            if stop_event is None:
                time.sleep(delay)
            elif stop_event.wait(delay):
                return
        yield float(timestamp), sensor_values


def find_serial_port(requested_port: str, ports: list[object]) -> str | None:
    """Resolve an explicit port or find a likely Arduino port."""

    if requested_port != "auto":
        return requested_port
    return next(
        (
            item.device
            for item in ports
            if "usbmodem" in item.device.lower()
            or "usbserial" in item.device.lower()
            or "arduino" in item.description.lower()
        ),
        None,
    )


def serial_frames(
    port: str,
    baud_rate: int,
    reconnect_delay_seconds: float,
    *,
    stop_event: threading.Event | None = None,
    on_status: Callable[[str], None] | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield valid frames and reconnect when the serial device disappears."""

    try:
        import serial
        import serial.tools.list_ports
    except ImportError as error:  # pragma: no cover - depends on local hardware setup
        raise RuntimeError("Serial 입력에는 pyserial 설치가 필요합니다.") from error

    def wait_to_reconnect() -> None:
        if stop_event is None:
            time.sleep(reconnect_delay_seconds)
        else:
            stop_event.wait(reconnect_delay_seconds)

    while stop_event is None or not stop_event.is_set():
        actual_port = find_serial_port(port, list(serial.tools.list_ports.comports()))
        if actual_port is None:
            if on_status: on_status("disconnected")
            print(
                "Arduino Serial 포트를 찾지 못했습니다. "
                f"{reconnect_delay_seconds:g}초 후 다시 찾습니다.",
                file=sys.stderr,
            )
            wait_to_reconnect()
            continue

        try:
            with serial.Serial(actual_port, baud_rate, timeout=0.25) as connection:
                print(f"Serial 연결됨: {actual_port} ({baud_rate} baud)", file=sys.stderr)
                last_valid = time.monotonic()
                while stop_event is None or not stop_event.is_set():
                    raw = connection.readline().decode("utf-8", errors="ignore")
                    values = parse_sensor_line(raw)
                    if values is not None:
                        last_valid = time.monotonic()
                        yield time.monotonic(), values
                    elif on_status and (raw.strip() or time.monotonic() - last_valid > 1.0):
                        on_status("invalid_input" if raw.strip() else "disconnected")
        except (serial.SerialException, OSError) as error:
            if on_status: on_status("disconnected")
            print(f"Serial 연결 끊김: {error}", file=sys.stderr)
            print(
                "케이블과 Arduino IDE Serial Monitor를 확인하세요. "
                f"{reconnect_delay_seconds:g}초 후 자동 재연결합니다.",
                file=sys.stderr,
            )
            wait_to_reconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="재생할 센서 CSV 파일")
    source.add_argument("--port", help="Arduino 포트. 자동 탐색은 auto")
    model = parser.add_mutually_exclusive_group()
    model.add_argument(
        "--model",
        type=Path,
        help="사용자 지정 모델 번들 경로",
    )
    model.add_argument(
        "--window-model",
        type=int,
        choices=sorted(MODEL_PATHS),
        help="내장 3초·5초·20초 모델 선택. 기본값은 20초",
    )
    parser.add_argument("--baud-rate", type=int, default=115200)
    parser.add_argument("--reconnect-delay-seconds", type=float, default=2.0)
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="고급 옵션: 모델 번들의 창 길이와 같은 값만 허용",
    )
    parser.add_argument(
        "--minimum-window-seconds",
        type=float,
        default=None,
        help="첫 예측에 필요한 최소 데이터 길이. 기본값은 모델 메타데이터 사용",
    )
    parser.add_argument("--prediction-interval-seconds", type=float, default=1.0)
    parser.add_argument("--max-gap-seconds", type=float, default=1.0)
    parser.add_argument("--smoothing-history", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=0.0,
        help="CSV 재생 배율. 0이면 지연 없이 재생, 1이면 실제 시간",
    )
    parser.add_argument("--jsonl-output", type=Path, help="예측 이벤트를 저장할 JSONL 파일")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="확률 목록 없이 가장 높은 자세 하나만 GUI에 표시",
    )
    return parser


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model is not None:
        return args.model
    return MODEL_PATHS[args.window_model or 20]


def emit(event: PredictionEvent, output: TextIO | None, *, echo: bool = True) -> None:
    record = event.as_dict()
    line = json.dumps(record, ensure_ascii=False)
    if echo:
        print(line)
    if output is not None:
        output.write(line + "\n")
        output.flush()


def prediction_display_text(event: PredictionEvent) -> str:
    """Return the single posture text shown in GUI mode."""

    return event.posture or "판단 유보"


def _put_latest(messages: queue.Queue[tuple[str, object]], item: tuple[str, object]) -> None:
    try:
        messages.get_nowait()
    except queue.Empty:
        pass
    messages.put_nowait(item)


def run_gui(
    engine: RealtimePosturePredictor,
    frames: Iterator[tuple[float, np.ndarray]],
    output_path: Path | None,
    *,
    stop_event: threading.Event | None = None,
) -> None:
    """Run sensor inference in a worker and render only the winning posture."""

    import tkinter as tk

    root = tk.Tk()
    root.tk.call("tk", "appname", "실시간 자세 예측")
    root.title("실시간 자세 예측")
    root.geometry("620x320")
    root.minsize(520, 280)
    root.configure(background="#f4f6f9")

    tk.Label(
        root,
        text="현재 자세",
        font=("Apple SD Gothic Neo", 20, "bold"),
        foreground="#263238",
        background="#f4f6f9",
    ).pack(pady=(42, 18))

    posture_text = tk.StringVar(value="센서 연결 중")
    posture_label = tk.Label(
        root,
        textvariable=posture_text,
        font=("Apple SD Gothic Neo", 38, "bold"),
        foreground="#1565c0",
        background="#ffffff",
        relief="solid",
        borderwidth=1,
        padx=36,
        pady=24,
    )
    posture_label.pack(fill="x", padx=48)

    status_text = tk.StringVar(
        value=(
            "첫 예측을 위해 약 "
            f"{engine.config.window_seconds:g}초간 데이터를 수집합니다."
        )
    )
    tk.Label(
        root,
        textvariable=status_text,
        font=("Apple SD Gothic Neo", 12),
        foreground="#607d8b",
        background="#f4f6f9",
    ).pack(pady=18)

    messages: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
    stop_event = stop_event if stop_event is not None else threading.Event()

    def worker() -> None:
        output: TextIO | None = None
        last_warmup_second = -1
        prediction_made = False
        try:
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output = output_path.open("a", encoding="utf-8")
            for timestamp, values in frames:
                if stop_event.is_set():
                    return
                event = engine.push(timestamp, values)
                if event is not None:
                    prediction_made = True
                    emit(event, output, echo=False)
                    _put_latest(messages, ("prediction", event))
                    continue

                warmup_second = int(engine.buffer.duration_seconds)
                if warmup_second != last_warmup_second:
                    last_warmup_second = warmup_second
                    _put_latest(messages, ("warming", warmup_second))
            if not prediction_made:
                _put_latest(messages, ("error", "예측에 필요한 데이터 길이가 부족합니다."))
        except Exception as error:  # pragma: no cover - UI safety net
            _put_latest(messages, ("error", str(error)))
        finally:
            frames.close()
            if output is not None:
                output.close()

    def poll_messages() -> None:
        try:
            kind, payload = messages.get_nowait()
        except queue.Empty:
            pass
        else:
            if kind == "prediction":
                posture_text.set(prediction_display_text(payload))
                status_text.set("가장 높은 확률의 자세를 표시하고 있습니다.")
                posture_label.configure(foreground="#1565c0")
            elif kind == "warming":
                elapsed = min(int(payload), int(engine.config.window_seconds))
                posture_text.set("데이터 수집 중")
                status_text.set(f"{elapsed} / {int(engine.config.window_seconds)}초")
                posture_label.configure(foreground="#607d8b")
            else:
                posture_text.set("오류")
                status_text.set(str(payload))
                posture_label.configure(foreground="#c62828")
        if not stop_event.is_set():
            root.after(100, poll_messages)

    def close_window() -> None:
        stop_event.set()
        if worker_thread.is_alive():
            root.after(50, close_window)
            return
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_window)
    worker_thread = threading.Thread(target=worker, name="posture-inference", daemon=True)
    worker_thread.start()
    root.after(100, poll_messages)
    root.after(150, root.lift)
    root.mainloop()


def main() -> None:
    args = build_parser().parse_args()
    if args.replay_speed < 0:
        raise SystemExit("--replay-speed는 0 이상이어야 합니다.")
    if args.reconnect_delay_seconds < 0:
        raise SystemExit("--reconnect-delay-seconds는 0 이상이어야 합니다.")

    predictor = PosturePredictor(resolve_model_path(args))
    window_seconds = (
        predictor.window_seconds
        if args.window_seconds is None
        else args.window_seconds
    )
    if not math.isclose(window_seconds, predictor.window_seconds):
        raise SystemExit(
            "--window-seconds는 선택한 모델의 학습 창 길이 "
            f"({predictor.window_seconds:g}초)와 같아야 합니다."
        )
    minimum_window_seconds = (
        predictor.minimum_window_seconds
        if args.minimum_window_seconds is None
        else args.minimum_window_seconds
    )
    config = LiveConfig(
        window_seconds=window_seconds,
        minimum_window_seconds=minimum_window_seconds,
        prediction_interval_seconds=args.prediction_interval_seconds,
        max_gap_seconds=args.max_gap_seconds,
        smoothing_history=args.smoothing_history,
        confidence_threshold=args.confidence_threshold,
    )
    engine = RealtimePosturePredictor(predictor, config)
    stop_event = threading.Event()
    frames = (
        replay_csv(args.input, args.replay_speed, stop_event=stop_event)
        if args.input
        else serial_frames(args.port, args.baud_rate, args.reconnect_delay_seconds, stop_event=stop_event)
    )

    if args.gui:
        run_gui(engine, frames, args.jsonl_output, stop_event=stop_event)
        return

    output: TextIO | None = None
    try:
        if args.jsonl_output:
            args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
            output = args.jsonl_output.open("a", encoding="utf-8")
        for timestamp, values in frames:
            event = engine.push(timestamp, values)
            if event is not None:
                emit(event, output)
    except KeyboardInterrupt:
        print("실시간 추론을 중지했습니다.", file=sys.stderr)
    finally:
        if output is not None:
            output.close()


if __name__ == "__main__":
    main()
