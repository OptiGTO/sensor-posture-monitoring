"""Describe fixed TRAIN feature tails without fitting, predicting, or scoring.

Only selected TRAIN rows of X/y are converted from NPZ bytes into arrays.
Full groups/record_ids are identity metadata for disjointness checks; compressed
streams may internally decompress skipped bytes without interpreting their values.
"""
from __future__ import annotations

import argparse
from collections import Counter
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

VERSION = "feature_tail_audit_v1"
QUANTILES = [.01, .99]


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([value.dtype.str, list(value.shape)]).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_evidence(path):
    path = Path(path)
    stat = path.stat()
    return {"sha256": file_hash(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def selected_rows(path, key, indices, expected_rows):
    """Decode only sorted unique selected rows of a C-order numeric NPY member."""
    take = np.asarray(indices)
    require(take.ndim == 1 and take.dtype.kind in "iu" and len(take) > 0
        and np.all(np.diff(take) > 0), "Rows must be nonempty sorted unique integer indices")
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        require(version in ((1, 0), (2, 0)), "Unsupported NPY header version")
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        require(not fortran and not dtype.hasobject and dtype.kind in "fiu"
            and len(shape) in (1, 2) and shape[0] == expected_rows, "Unsupported numeric row schema")
        require(take[0] >= 0 and take[-1] < shape[0], "Selected row outside archive")
        header_end = stream.tell()
        row_values = int(np.prod(shape[1:]))
        row_bytes = row_values * dtype.itemsize
        result = np.empty((len(take), *shape[1:]), dtype=dtype)
        boundaries = np.r_[0, np.flatnonzero(np.diff(take) != 1) + 1, len(take)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            stream.seek(header_end + int(take[left]) * row_bytes)
            buffer = stream.read(int(right - left) * row_bytes)
            require(len(buffer) == int(right - left) * row_bytes, "Truncated NPY row block")
            result[left:right] = np.frombuffer(buffer, dtype=dtype).reshape((int(right - left), *shape[1:]))
    return result


def feature_groups(names):
    statistics = ["mean", "std", "median", "min", "max", "range", "iqr", "rms", "mean_abs_diff",
        "slope", "skew", "kurtosis", "dominant_frequency", "spectral_entropy", "spectral_centroid", "low_frequency_ratio"]
    sensor_names = [f"R{row}{column}" for row in range(1, 5) for column in range(1, 5)]
    groups = {}
    for prefix in ("abs", "rel"):
        for statistic in statistics:
            groups[f"{prefix}/sensor/{statistic}"] = [names.index(f"{prefix}_{statistic}_{sensor}") for sensor in sensor_names]
        for geometry in ("row", "column"):
            for statistic in ("mean", "std", "mean_abs_diff"):
                groups[f"{prefix}/{geometry}/{statistic}"] = [names.index(f"{prefix}_{geometry}_{statistic}_{index}") for index in range(1, 5)]
        for statistic in ("mean", "std", "mean_abs_diff"):
            groups[f"{prefix}/derived/{statistic}"] = [names.index(f"{prefix}_{statistic}_{name}")
                for name in ("global", "left_right", "front_back", "diagonal", "outer_row", "outer_column")]
    flattened = [index for indices in groups.values() for index in indices]
    require(sorted(flattened) == list(range(596)), "Statistic groups must partition exactly 596 features")
    return groups


def ratios_summary(ratios, variable, central_zero):
    valid = ratios[variable]
    count = len(ratios)
    return {"features": count, "constant_features": int((~variable).sum()),
        "variable_features": int(variable.sum()), "zero_central_variable_features": int((variable & central_zero).sum()),
        "median_ratio_over_variable_features": None if not len(valid) else float(np.median(valid)),
        **{f"ratio_below_{str(cutoff).replace('.', '_')}": {
            "count": int(np.count_nonzero(variable & (ratios < cutoff))),
            "fraction_all_features": float(np.count_nonzero(variable & (ratios < cutoff)) / count),
            "fraction_variable_features": None if not len(valid) else float(np.count_nonzero(valid < cutoff) / len(valid))}
            for cutoff in (.1, .25)}}


def concentration(ids, row_tail_cells, total_cells_per_row, *, record_subjects=None):
    unique, reverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    tails = np.bincount(reverse, weights=row_tail_cells, minlength=len(unique)).astype(np.int64)
    total_tail = int(tails.sum())
    rows = []
    for index, identity in enumerate(unique):
        row = {"identity_sha256": hashlib.sha256(str(identity).encode("utf-8")).hexdigest(),
            "windows": int(counts[index]), "tail_cells": int(tails[index]),
            "tail_cell_rate": float(tails[index] / (counts[index] * total_cells_per_row)),
            "window_fraction": float(counts[index] / len(ids)),
            "tail_fraction": 0. if total_tail == 0 else float(tails[index] / total_tail)}
        if record_subjects is not None:
            row["subject_sha256"] = hashlib.sha256(str(record_subjects[str(identity)]).encode("utf-8")).hexdigest()
        rows.append(row)
    rows.sort(key=lambda row: (-row["tail_cells"], row["identity_sha256"]))
    top_count = max(1, int(np.ceil(.1 * len(rows))))
    return {"identities": len(rows), "identities_with_tail_cells": int(np.count_nonzero(tails)),
        "top_one_tail_fraction": rows[0]["tail_fraction"],
        "top_three_tail_fraction": sum(row["tail_fraction"] for row in rows[:3]),
        "top_ten_percent_identity_count": top_count,
        "top_ten_percent_tail_fraction": sum(row["tail_fraction"] for row in rows[:top_count]),
        "top_ten_percent_window_fraction": sum(row["window_fraction"] for row in rows[:top_count]),
        "tail_cell_rate_min": min(row["tail_cell_rate"] for row in rows),
        "tail_cell_rate_median": float(np.median([row["tail_cell_rate"] for row in rows])),
        "tail_cell_rate_max": max(row["tail_cell_rate"] for row in rows), "rows": rows}


def describe(values, people, records, names):
    require(values.ndim == 2 and values.shape[1] == 596 and values.dtype == np.float32
        and np.isfinite(values).all(), "Expected finite selected TRAIN float32 N x 596")
    x = values.astype(np.float64)
    low, high = np.quantile(x, QUANTILES, axis=0, method="linear")
    minimum, maximum = x.min(axis=0), x.max(axis=0)
    full, central = maximum - minimum, high - low
    variable = full > 0
    ratios = np.full(596, np.nan)
    np.divide(central, full, out=ratios, where=variable)
    lower, upper = x < low, x > high
    tail = lower | upper
    row_tail = tail.sum(axis=1)
    columns = []
    for index, name in enumerate(names):
        columns.append({"index": index, "name": name, "minimum": float(minimum[index]), "q01": float(low[index]),
            "q99": float(high[index]), "maximum": float(maximum[index]), "central_range": float(central[index]),
            "full_range": float(full[index]), "central_full_ratio": None if not variable[index] else float(ratios[index]),
            "constant": bool(not variable[index]), "zero_central_variable": bool(variable[index] and central[index] == 0),
            "lower_tail_cells": int(lower[:, index].sum()), "upper_tail_cells": int(upper[:, index].sum())})
    summaries = {}
    for name, indices in feature_groups(names).items():
        summaries[name] = {**ratios_summary(ratios[indices], variable[indices], central[indices] == 0),
            "lower_tail_cells": int(lower[:, indices].sum()), "upper_tail_cells": int(upper[:, indices].sum()),
            "tail_cells": int(tail[:, indices].sum()), "total_cells": len(values) * len(indices),
            "tail_cell_fraction": float(tail[:, indices].mean())}
    unique_records, record_counts = np.unique(records, return_counts=True)
    record_subjects = {str(record): str(person) for record, person in zip(records, people)}
    return {"training_windows": len(values), "training_records": len(unique_records),
        "training_subjects": len(np.unique(people)), "record_window_counts": dict(sorted(Counter(record_counts.tolist()).items())),
        "summary": ratios_summary(ratios, variable, central == 0),
        "tail_cells": int(tail.sum()), "total_cells": int(tail.size), "tail_cell_fraction": float(tail.mean()),
        "windows_with_any_tail": int(np.count_nonzero(row_tail)),
        "subjects": concentration(people, row_tail, 596),
        "records": concentration(records, row_tail, 596, record_subjects=record_subjects),
        "statistic_groups": summaries, "features": columns}


def execute(root, run, output):
    started = now()
    require(not output.exists(), "Output must be a new directory; no audit overwrite")
    protocol_path = run / "protocol.json"
    protocol = read_json(protocol_path)
    protocol_hash = file_hash(protocol_path)
    require(protocol_hash == read_json(run / "protocol_lock.json")["sha256"], "Protocol seal changed")
    input_paths = [protocol_path, run / "protocol_lock.json", run / "data.npz", run / "dataset.json",
        run / "supplemental_training.npz", run / "supplemental_manifest.json", run / "protected_before.json",
        run / "audits/support_fitted_identity.json"]
    inputs = {str(path.relative_to(root)): file_evidence(path) for path in input_paths}
    require(file_hash(run / "data.npz") == protocol["cache_sha256"], "Frozen feature cache changed")
    require(file_hash(run / "dataset.json") == protocol["dataset_sha256"], "Origin manifest changed")
    supplement = read_json(run / "supplemental_manifest.json")
    for key, name in (("manifest_sha256", "supplemental_manifest.json"), ("cache_sha256", "supplemental_training.npz")):
        require(file_hash(run / name) == protocol["supplemental_training"][key], "Supplement seal changed")
    require(supplement["cache_sha256"] == file_hash(run / "supplemental_training.npz")
        and supplement["personal_reference_used"] is False, "Supplement identity or personal reference differs")
    source_files = ["experiments/iterative_improvement/record_training.py",
        "experiments/iterative_improvement/supplemental.py", "src/posture_core.py"]
    sources = {}
    for rel in source_files:
        expected = protocol["source_hashes"][rel]
        require(file_hash(root / rel) == expected == file_hash(run / "source_snapshot" / rel), "Frozen helper changed: " + rel)
        sources[rel] = file_evidence(root / rel)
    from .record_training import append_record_training
    from src.posture_core import feature_names, normalize_text
    names = feature_names()
    require(names == supplement["feature_names"] and len(names) == 596, "Feature schema changed")
    protected = read_json(run / "protected_before.json")
    protected_before = {rel: file_evidence(root / rel) for rel in protected}
    require(protected_before == protected, "Protected originals differ from iteration015 freeze")
    identity = read_json(run / "audits/support_fitted_identity.json")
    require(identity["passed"] and identity["protocol_sha256"] == protocol_hash, "Fitted TRAIN identity audit missing")
    expected_train = {row["fold"]: row for row in identity["results"] if row["candidate"] == "SUPPORT_identity"}
    caches = {}
    for kind, filename in (("base", "data.npz"), ("supplement", "supplemental_training.npz")):
        with np.load(run / filename, allow_pickle=False) as archive:
            groups, records = archive["groups"], archive["record_ids"]
        require(groups.ndim == 1 and records.shape == groups.shape, "Identity metadata shape differs")
        caches[kind] = {"groups": np.asarray([normalize_text(str(value)) for value in groups]),
            "record_ids": np.asarray([normalize_text(str(value)) for value in records]), "path": run / filename}
    require(not set(caches["base"]["record_ids"]) & set(caches["supplement"]["record_ids"]), "Base/supplement record overlap")
    eligible = {normalize_text(row["record_id"]): row for row in supplement["records"]}
    for record in set(caches["supplement"]["record_ids"]):
        row = eligible[record]
        require(row["label"] == "정자세" and row["eligibility"]["explicit_registration"] is False
            and row["eligibility"]["globally_excluded"] is False and row["eligibility"]["training_role"] == "include",
            "Nonordinary or ineligible supplement record")
    rows, chosen_sets = [], []
    for partition in protocol["partitions"]:
        train_subjects = sorted(normalize_text(value) for value in partition["train"])
        other_subjects = {normalize_text(value) for key in ("validation", "test") for value in partition[key]}
        require(not set(train_subjects) & other_subjects, "TRAIN/held-out subjects overlap")
        chosen = {}
        for kind, cache in caches.items():
            take = np.flatnonzero(np.isin(cache["groups"], train_subjects))
            chosen[kind] = {"X": selected_rows(cache["path"], "X", take, len(cache["groups"])),
                "y": selected_rows(cache["path"], "y", take, len(cache["groups"])),
                "groups": cache["groups"][take], "record_ids": cache["record_ids"][take], "indices": take}
            require(set(chosen[kind]["groups"]) == set(train_subjects), "Incomplete TRAIN subject selection")
        base, extra = chosen["base"], chosen["supplement"]
        expected = partition["counts"]["train"]
        require(len(base["X"]) == expected["windows"] and len(np.unique(base["record_ids"])) == expected["records"]
            and np.bincount(base["y"], minlength=7).tolist() == expected["class_counts"], "Frozen TRAIN row selection differs")
        for record in set(extra["record_ids"]):
            require(int(np.count_nonzero(extra["record_ids"] == record)) == eligible[record]["windows"], "Partial restored recording")
        values, labels, people, records, restoration = append_record_training(base, extra, np.arange(len(base["X"])))
        fitted = expected_train[partition["id"]]
        train_hashes = {"training_input_sha256": array_hash(values), "training_labels_sha256": array_hash(labels),
            "training_groups_sha256": json_hash(people.tolist())}
        require(all(fitted[key] == value for key, value in train_hashes.items()), "Rows differ from fitted iteration015 TRAIN identity")
        result = describe(values, people, records, names)
        result.update(partition=partition["id"], input_hashes=train_hashes,
            base_train_indices=base["indices"].tolist(), supplemental_train_indices=extra["indices"].tolist(),
            record_ids_sha256=array_hash(records), restored_windows=restoration["added_windows"],
            restored_records=restoration["added_records"], class_window_counts=np.bincount(labels, minlength=7).tolist(),
            fitted_train_hashes_match=True)
        rows.append(result)
        chosen_sets.append({f"{kind}:{int(index)}" for kind in chosen for index in chosen[kind]["indices"]})
    require({rel: file_evidence(root / rel) for rel in protected} == protected_before, "Protected originals changed during audit")
    require({rel: file_evidence(root / rel) for rel in inputs} == inputs, "Frozen inputs changed during audit")
    require({rel: file_evidence(root / rel) for rel in sources} == sources, "Helpers changed during audit")
    script = Path(__file__).resolve()
    sources[str(script.relative_to(root))] = file_evidence(script)
    result = {"schema_version": 1, "version": VERSION, "started_at": started, "finished_at": now(),
        "timezone": "Asia/Seoul", "run": str(run.relative_to(root)), "protocol_sha256": protocol_hash,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "command": [sys.executable, "-B", "-m", "experiments.iterative_improvement.feature_tail_audit",
            "--input-run", str(run), "--output", str(output)],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "executable": sys.executable},
        "inputs": inputs, "sources": sources, "script_source": script.read_text(encoding="utf-8"),
        "protected_originals": {"count": len(protected_before), "before_equals_iteration015": True,
            "before_equals_after": True, "manifest_sha256": json_hash(protected_before), "files": protected_before},
        "method": {"quantiles": QUANTILES, "quantile_method": "numpy.quantile float64 linear, per TRAIN feature",
            "ratio": "(q99-q01)/(max-min); full range zero => null, excluded from variable denominator; zero central variable => 0",
            "tails": "strict X<q01 or X>q99; tied boundary values are not tails",
            "concentration": "tail cells per identity; exposure normalized by windows*596; identities SHA256-pseudonymized",
            "row_loading": "only TRAIN X/y blocks decoded; full group/record identity metadata used for partition checks",
            "compressed_io_caveat": "hashing and ZIP seeking can read/decompress skipped bytes; non-TRAIN X/y values never converted to arrays or analyzed",
            "selection_basis": "fixed diagnostic quantiles; no score or holdout driven threshold choice"},
        "overlap": {"split_window_observations": sum(len(values) for values in chosen_sets),
            "unique_source_windows": len(set.union(*chosen_sets)),
            "pairwise_shared_source_windows": {f"{rows[i]['partition']}__{rows[j]['partition']}": len(chosen_sets[i] & chosen_sets[j])
                for i in range(len(rows)) for j in range(i + 1, len(rows))},
            "note": "TRAIN partitions overlap; statistics are not independent replication. A person may train in one partition and validate in another; each summary uses only its own TRAIN."},
        "partitions": rows, "new_training": False, "new_prediction": False, "new_evaluation": False,
        "heldout_feature_or_label_arrays_decoded": False, "raw_data_mutated": False,
        "limitations": ["Finite tails do not establish data errors, physical units, sensor faults or harmful observations.",
            "About 2% strict tail cells per continuous feature is expected by construction; cell count alone is not evidence of heavy tails.",
            "Whole-TRAIN range ratios are not ExtraTrees node-local threshold ranges and cannot establish a mechanism or expected F1 gain.",
            "Training-fitted clipping would change genuine extremes and tree geometry; only a separate preregistered model experiment could assess effects."]}
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "feature_tail_audit.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    destination.chmod(0o444)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    destination = execute(root, args.input_run.resolve(), args.output.resolve())
    print(json.dumps({"path": str(destination), "sha256": file_hash(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
