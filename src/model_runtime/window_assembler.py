"""One time-axis window implementation shared by every inference backend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import ModelDescriptor


class WindowAssembler:
    def assemble(self, frame: pd.DataFrame, descriptor: ModelDescriptor, window_seconds: float) -> np.ndarray:
        selected = descriptor.window.validate(window_seconds)
        timestamps = frame["elapsed_s"].to_numpy(dtype=float)
        values = frame[list(descriptor.input_channels)].to_numpy(dtype=np.float32)
        if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
            raise ValueError("elapsed_s는 엄격히 증가해야 합니다.")
        duration = float(timestamps[-1] - timestamps[0])
        if duration + 1e-9 < selected:
            raise ValueError("선택한 윈도우 길이만큼의 센서 데이터가 필요합니다.")
        end = timestamps[-1]
        start = end - selected
        offsets = np.arange(descriptor.window.target_samples(selected, descriptor.sampling_rate), dtype=float) / descriptor.sampling_rate
        grid = start + offsets
        sequence = np.column_stack([np.interp(grid, timestamps, values[:, index]) for index in range(values.shape[1])]).astype(np.float32)
        return sequence[None, :, :]
