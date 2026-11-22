from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .posture_core import SENSOR_COLUMNS, normalize_text, read_sensor_csv, recording_label, training_csv_paths


def analyze_dataset(root: str | Path) -> dict[str, object]:
    root = Path(root)
    files = training_csv_paths(root)

    record_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, str]] = []
    channel_values: dict[str, list[np.ndarray]] = {name: [] for name in SENSOR_COLUMNS}
    file_hashes: dict[str, list[str]] = {}
    missing_count = 0
    infinite_count = 0
    duplicate_rows = 0
    sensor_min = float("inf")
    sensor_max = float("-inf")

    for path in files:
        frame = read_sensor_csv(path)
        raw_label, standardized_label = recording_label(frame)
        label_rows.append(
            {"raw_label": raw_label, "standardized_label": standardized_label}
        )
        subject = normalize_text(path.parent.name)
        match = re.search(r"_(\d{3})\.csv$", normalize_text(path.name))
        session = match.group(1) if match else "미상"
        numeric = frame[["elapsed_s", *SENSOR_COLUMNS]].to_numpy(dtype=float)
        sensor = frame[SENSOR_COLUMNS].to_numpy(dtype=float)
        missing_count += int(frame.isna().sum().sum())
        infinite_count += int(np.isinf(numeric).sum())
        duplicate_rows += int(frame.duplicated().sum())
        sensor_min = min(sensor_min, float(sensor.min()))
        sensor_max = max(sensor_max, float(sensor.max()))
        for channel_index, channel in enumerate(SENSOR_COLUMNS):
            channel_values[channel].append(sensor[:, channel_index])
        elapsed = frame["elapsed_s"].to_numpy(dtype=float)
        deltas = np.diff(elapsed)
        record_rows.append(
            {
                "file": normalize_text(path.name),
                "subject": subject,
                "raw_label": raw_label,
                "label": standardized_label,
                "session": session,
                "rows": int(len(frame)),
                "duration_s": float(elapsed[-1] - elapsed[0]),
                "median_dt_s": float(np.median(deltas)),
                "fast_start_interval_count": int((deltas < 0.01).sum()),
            }
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        file_hashes.setdefault(digest, []).append(str(path))

    records = pd.DataFrame(record_rows)
    label_map = pd.DataFrame(label_rows).drop_duplicates().sort_values(
        ["standardized_label", "raw_label"]
    )
    duplicate_file_groups = [paths for paths in file_hashes.values() if len(paths) > 1]

    channel_rows = []
    for channel in SENSOR_COLUMNS:
        values = np.concatenate(channel_values[channel]).astype(float)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        channel_rows.append(
            {
                "channel": channel,
                "count": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "q1": float(q1),
                "median": float(np.median(values)),
                "q3": float(q3),
                "max": float(values.max()),
                "iqr_outlier_rate": float(((values < lower) | (values > upper)).mean()),
            }
        )
    channels = pd.DataFrame(channel_rows)

    session_table = (
        records.groupby(["subject", "label", "session"], as_index=False)
        .size()
        .rename(columns={"size": "recording_count"})
    )
    combination_counts = (
        records.groupby(["subject", "label"], as_index=False)
        .agg(recording_count=("file", "size"), unique_sessions=("session", "nunique"))
    )
    session_anomalies = combination_counts[
        (combination_counts["recording_count"] != 2)
        | (combination_counts["unique_sessions"] != 2)
    ].copy()
    if not session_anomalies.empty:
        session_anomalies = session_anomalies.merge(
            session_table.groupby(["subject", "label"])["session"]
            .apply(lambda values: " | ".join(map(str, values)))
            .rename("observed_sessions"),
            on=["subject", "label"],
            how="left",
        )

    summary = {
        "file_count": int(len(records)),
        "subject_count": int(records["subject"].nunique()),
        "class_count": int(records["label"].nunique()),
        "total_rows": int(records["rows"].sum()),
        "recordings_per_subject": sorted(
            map(int, records.groupby("subject").size().unique())
        ),
        "recordings_per_class": {
            str(key): int(value)
            for key, value in records["label"].value_counts().sort_index().items()
        },
        "session_counts": {
            str(key): int(value)
            for key, value in records["session"].value_counts().sort_index().items()
        },
        "row_count_min": int(records["rows"].min()),
        "row_count_max": int(records["rows"].max()),
        "row_count_mean": float(records["rows"].mean()),
        "duration_min_s": float(records["duration_s"].min()),
        "duration_max_s": float(records["duration_s"].max()),
        "duration_mean_s": float(records["duration_s"].mean()),
        "median_dt_s": float(records["median_dt_s"].median()),
        "inferred_sampling_rate_hz": float(1.0 / records["median_dt_s"].median()),
        "fast_start_intervals_mean": float(
            records["fast_start_interval_count"].mean()
        ),
        "fast_start_intervals_min": int(
            records["fast_start_interval_count"].min()
        ),
        "fast_start_intervals_max": int(
            records["fast_start_interval_count"].max()
        ),
        "sensor_global_min": float(sensor_min),
        "sensor_global_max": float(sensor_max),
        "missing_value_count": int(missing_count),
        "infinite_value_count": int(infinite_count),
        "duplicate_complete_row_count": int(duplicate_rows),
        "duplicate_file_group_count": int(len(duplicate_file_groups)),
        "raw_label_variant_count": int(label_map["raw_label"].nunique()),
        "sensor_channel_count": len(SENSOR_COLUMNS),
        "columns": ["elapsed_s", "label", *SENSOR_COLUMNS],
        "class_names": records["label"].drop_duplicates().tolist(),
        "session_anomaly_count": int(len(session_anomalies)),
    }
    return {
        "summary": summary,
        "records": records,
        "label_map": label_map,
        "channel_summary": channels,
        "session_anomalies": session_anomalies,
        "duplicate_file_groups": duplicate_file_groups,
    }
