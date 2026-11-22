"""Read-only frozen-input audit with synthetic interpolation boundary evidence."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iterative_improvement.supplemental import _feature_source_compatibility
from src.posture_core import SENSOR_COLUMNS
from src.window_pipeline import WindowSpec, extract_window_features, segment_sensor_frame


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(ZoneInfo("Asia/Seoul"))


def inspect_headers(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"Unsupported audit NPY format: {version}")
            result[name] = {"shape": list(shape), "dtype": str(dtype), "fortran_order": fortran}
    return result


def synthetic_boundary():
    time = np.concatenate([np.arange(29, dtype=float) / 10, [3.2, 3.9, 4.8, 5.7, 6.0]])
    frame = pd.DataFrame({"elapsed_s": time, **{c: np.full(len(time), 100.) for c in SENSOR_COLUMNS}})
    changed = frame.copy()
    changed.loc[changed.elapsed_s == 3.2, "R11"] = 200.
    before, metadata = segment_sensor_frame(frame, WindowSpec(3))
    after, _ = segment_sensor_frame(changed, WindowSpec(3))
    return {"synthetic_only": True, "elapsed_s": time.tolist(), "baseline_all_sensors": 100.,
            "change": {"elapsed_s": 3.2, "sensor": "R11", "new_value": 200.},
            "first_window_metadata": metadata.iloc[0].to_dict(),
            "first_window_changed_grid_indices": np.flatnonzero(np.any(before[0] != after[0], axis=1)).tolist(),
            "first_window_R11_before": before[0, :, 0].astype(float).tolist(),
            "first_window_R11_after": after[0, :, 0].astype(float).tolist(),
            "first_window_max_abs_difference": float(np.max(np.abs(before[0] - after[0]))),
            "first_window_features_also_change": bool(not np.array_equal(
                extract_window_features(before[:1]), extract_window_features(after[:1]))),
            "interpretation": "A raw sample later than nominal 3s end can affect the first window via interpolation"}


def audit(output):
    started = now()
    output = output.resolve()
    if output.exists() or output.is_relative_to(ROOT / "data"):
        raise ValueError("Use a new isolated output directory outside data/")
    origin = ROOT / "research/results/2026-09-21_200011_improvement_retest"
    protocol_path = ROOT / "research/results/2026-09-22_004500_iterative_001/protocol.json"
    manifest_path = origin / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    cache_path = origin / "data.npz"
    cache_sha = digest(cache_path)
    if cache_sha != manifest["cache_sha256"]:
        raise ValueError("Frozen cache hash changed")
    compatibility = _feature_source_compatibility(ROOT, origin, manifest)
    for relative, row in compatibility.items():
        if digest(origin / "source_snapshot" / relative) != row["original_sha256"]:
            raise ValueError("Origin feature snapshot hash changed")
    protocol = json.loads(protocol_path.read_text())
    partition = protocol["partitions"][0]
    if protocol["cache_sha256"] != cache_sha or protocol["dataset_sha256"] != digest(manifest_path):
        raise ValueError("Development split is not bound to this frozen input")
    with np.load(cache_path, allow_pickle=False) as cache:
        # The label array and all test predictions remain unopened. Only row
        # membership selects development_1 TRAIN inputs for feature parity.
        groups = cache["groups"]
        rows = np.flatnonzero(np.isin(groups, partition["train"]))
        sample = rows[np.unique(np.linspace(0, len(rows) - 1, 24).astype(int))]
        sequences, features = cache["seq"][sample], cache["X"][sample]
        rebuilt = extract_window_features(sequences)
        parity = {"split": partition["id"], "partition": "train_only", "train_windows": len(rows),
                  "sample_count": len(sample), "sample_rule": "24 evenly spaced training-row offsets via linspace then integer cast",
                  "global_sample_indices": sample.tolist(), "sample_subjects": sorted(set(groups[sample].tolist())),
                  "bitwise_equal": bool(np.array_equal(rebuilt, features)),
                  "maximum_absolute_difference": float(np.max(np.abs(rebuilt - features)))}
    if not parity["bitwise_equal"]:
        raise ValueError("Training feature reconstruction differs from frozen X")
    current_runs = []
    for run in sorted((ROOT / "research/results").glob("*iterative_007")):
        path = run / "protocol.json"
        if not path.exists():
            continue
        p = json.loads(path.read_text())
        current_runs.append({"path": str(run.relative_to(ROOT)), "protocol_sha256": digest(path),
                             "cache_sha256": digest(run / "data.npz"),
                             "same_cache": p["cache_sha256"] == cache_sha == digest(run / "data.npz")})
    sources = ["src/posture_core.py", "src/window_pipeline.py", "src/extended_models.py",
               "src/_vendor/minirocket_multi.py", "src/model_catalog.py",
               "experiments/improvement_retest/prepare.py",
               "experiments/iterative_improvement/supplemental.py",
               "experiments/iterative_improvement/sequence_input_audit.py"]
    result = {"started_at": started.isoformat(), "finished_at": now().isoformat(),
              "scope": "input metadata, source review, training-only feature parity, synthetic causality evidence; no fitting or score evaluation",
              "measurement_acquisition_time": "Not inferred; audit timestamps are execution times only",
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "dirty_state": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
              "command": [sys.executable, str(Path(__file__).resolve()), "--output", str(output)],
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "scipy": scipy.__version__},
              "dataset": {"origin": str(origin.relative_to(ROOT)), "dataset_sha256": digest(manifest_path),
                          "cache_sha256": cache_sha, "npy_headers": inspect_headers(cache_path),
                          "sensor_order": SENSOR_COLUMNS,
                          **{k: manifest[k] for k in ("window_seconds", "sampling_rate_hz", "stride_seconds",
                                                     "min_tail_fraction", "feature_dimension")}},
              "source_compatibility": compatibility, "current_source_hashes": {p: digest(ROOT / p) for p in sources},
              "sample_protocol_sha256": digest(protocol_path), "current_iteration_007": current_runs,
              "train_only_feature_parity": parity, "synthetic_boundary_reproduction": synthetic_boundary(),
              "unchanged_frozen_input": digest(cache_path) == cache_sha,
              "labels_array_read": False, "raw_measurements_read": False, "existing_files_written": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return output, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or ROOT / "research/results" / f"{now():%Y-%m-%d_%H%M%S}_sequence_input_audit"
    destination, result = audit(destination)
    print(json.dumps({"output": str(destination), "parity": result["train_only_feature_parity"],
                      "synthetic_first_window_difference": result["synthetic_boundary_reproduction"]["first_window_max_abs_difference"]}, ensure_ascii=False))
