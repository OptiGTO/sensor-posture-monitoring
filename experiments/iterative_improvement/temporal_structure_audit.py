"""Frozen TRAIN-only signed temporal-functional and descriptive variance audit.

No classifier, scaler, learned feature selector, prediction or score is fitted.
The only projection matrices learned from a matrix SVD use fixed36-point
analytic temporal kernels, not measured data. Record-position coefficients are
descriptive within-record least-squares summaries and never inference features.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version as package_version
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile
from zoneinfo import ZoneInfo

import numpy as np

ORIGIN = "research/results/2026-09-22_055131_iterative_019"
VERSION = "temporal_structure_audit_v1"
SCRIPT = "experiments/iterative_improvement/temporal_structure_audit.py"
VIEWS = ("absolute_float64", "core_relative_float32_eps1e-6", "study_relative_float64_eps1e-8")
FREQUENCIES = (2, 3, 4)
COMPONENTS = ("posture", "person", "person_posture", "record_within_person_posture", "window_position_linear", "window_position_remainder")
RTOL, EPS_FACTOR = 1e-10, 512.


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reference(path, root):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.relative_to(root)), "sha256": digest(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o444)


def array_hash(value):
    value = np.ascontiguousarray(value)
    result = hashlib.sha256(json.dumps([value.dtype.str, list(value.shape)]).encode())
    result.update(value.tobytes())
    return result.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def environment():
    return {"python": platform.python_version(), "executable": sys.executable,
        **{name: package_version(name) for name in ("numpy", "scipy", "scikit-learn", "pandas", "threadpoolctl")}}


def selected_rows(path, key, indices, expected_rows):
    """Decode only requested C-order numeric rows; bounded header, no uint wrap."""
    take = np.asarray(indices)
    require(type(expected_rows) is int and expected_rows > 0 and take.ndim == 1 and take.dtype.kind in "iu"
        and len(take) > 0 and np.all(take[1:] > take[:-1])
        and int(take[0]) >= 0 and int(take[-1]) < expected_rows, "Expected sorted unique in-range integer rows")
    take = take.astype(np.int64)
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        require(version in ((1, 0), (2, 0)), "Unsupported NPY version")
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream, max_header_size=10000)
        require(not fortran and not dtype.hasobject and dtype.kind in "fiu" and dtype.itemsize <= 8
            and len(shape) in (1, 2, 3) and shape[0] == expected_rows
            and all(type(n) is int and 0 < n <= 10_000_000 for n in shape)
            and int(np.prod(shape[1:], dtype=np.int64)) <= 10000, "Unsupported bounded numeric row schema")
        row_size = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        start = stream.tell()
        require(archive.getinfo(key + ".npy").file_size == start + expected_rows * row_size, "NPY payload length differs from shape")
        result = np.empty((len(take), *shape[1:]), dtype=dtype)
        boundaries = np.r_[0, np.flatnonzero(np.diff(take) != 1) + 1, len(take)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            stream.seek(start + int(take[left]) * row_size)
            payload = stream.read(int(right - left) * row_size)
            require(len(payload) == int(right - left) * row_size, "Truncated selected NPY rows")
            result[left:right] = np.frombuffer(payload, dtype=dtype).reshape((int(right - left), *shape[1:]))
    return result


def temporal_kernels():
    from threadpoolctl import threadpool_limits
    time = np.arange(36, dtype=np.float64)
    u = np.linspace(-1., 1., 36)
    basis = np.zeros((6, 36), dtype=np.float64)
    basis[0], basis[1] = 1 / 36., u / (np.sum(u ** 2) + 36e-9)
    basis[2, 24:], basis[3, 12:] = 1 / 12., 1 / 24.
    basis[4, 35], basis[4, 24], basis[5, 35], basis[5, 12] = 1., -1., 1., -1.
    with threadpool_limits(limits=1):
        _, singular, right = np.linalg.svd(basis, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(basis.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    projector = right[:rank].T @ right[:rank]
    raw = np.asarray([np.sqrt(2 / 36.) * np.cos(np.pi * k * (time + .5) / 36.) for k in FREQUENCIES])
    residual = raw - raw @ projector
    gram = residual @ residual.T
    norms = np.linalg.norm(residual, axis=1)
    correlation = np.divide(gram, norms[:, None] * norms[None], out=np.full_like(gram, np.nan), where=(norms[:, None] * norms[None]) != 0)
    require(np.max(np.abs(residual @ basis.T)) <= 1e-12, "Fixed residual temporal basis is not orthogonal to old signed span")
    details = {"kernel_names": ["full_mean", "core_slope", "tail12_mean", "tail24_mean", "tail12_end_difference", "tail24_end_difference"],
        "basis": basis.tolist(), "basis_sha256": array_hash(basis), "rank": rank, "singular_values": singular.tolist(), "rank_tolerance": tolerance,
        "raw_kernels": raw.tolist(), "residual_kernels": residual.tolist(), "raw_kernels_sha256": array_hash(raw), "residual_kernels_sha256": array_hash(residual),
        "raw_norms": np.linalg.norm(raw, axis=1).tolist(), "residual_norms": norms.tolist(),
        "residual_norm_squared_fractions": (norms ** 2 / np.square(raw).sum(axis=1)).tolist(),
        "residual_gram": gram.tolist(), "residual_correlation_gram": [[None if not np.isfinite(v) else float(v) for v in row] for row in correlation],
        "orthogonality_max_abs_error": float(np.max(np.abs(residual @ basis.T))),
        "residual_axes_are_not_assumed_orthogonal": True, "squared_coefficients_not_summed_as_independent_energy": True}
    return raw, residual, details


def representations(sequence):
    require(sequence.dtype == np.float32 and sequence.ndim == 3 and sequence.shape[1:] == (36, 16)
        and np.isfinite(sequence).all(), "Expected finite TRAIN float32 sequences(N,36,16)")
    core = (sequence - sequence.mean(axis=2, keepdims=True)) / (sequence.std(axis=2, keepdims=True) + 1e-6)
    absolute = np.asarray(sequence, dtype=np.float64)
    study = (absolute - absolute.mean(axis=2, keepdims=True)) / (absolute.std(axis=2, keepdims=True) + 1e-8)
    return (absolute, core.astype(np.float64), study)


def coefficients(sequence, raw, residual):
    views = representations(sequence)
    # Ordering: view, frequency2/3/4, row-major sensor. Always retain every axis.
    return tuple(np.concatenate([np.einsum("nts,kt->nks", values, kernel, optimize=False).reshape(len(sequence), -1)
                                for values in views], axis=1) for kernel in (raw, residual))


def balanced_decomposition(values, labels, people, records, starts):
    require(values.ndim == 2 and len(values) == len(labels) == len(people) == len(records) == len(starts)
        and np.isfinite(values).all() and np.isfinite(starts).all(), "Invalid aligned TRAIN diagnostic rows")
    subjects = sorted(set(people.tolist()))
    require(set(labels.tolist()) == set(range(7)), "All seven posture classes required")
    metadata, record_values, window_weights = [], [], np.zeros(len(values), dtype=np.float64)
    index_by_cell = {(subject, label): [] for subject in subjects for label in range(7)}
    for record in sorted(set(records.tolist())):
        idx = np.flatnonzero(records == record)
        require(len(set(people[idx])) == len(set(labels[idx])) == 1, "Mixed-subject or mixed-label recording")
        person, label = str(people[idx[0]]), int(labels[idx[0]])
        index_by_cell[(person, label)].append(len(metadata))
        metadata.append({"record_id": record, "subject": person, "class_id": label, "indices": idx, "windows": len(idx)})
        record_values.append(values[idx].mean(axis=0, dtype=np.float64))
    require(all(index_by_cell.values()), "Incomplete person by posture cells; no imputation")
    record_values = np.asarray(record_values)
    cells = np.asarray([[record_values[index_by_cell[(person, label)]].mean(axis=0) for label in range(7)] for person in subjects])
    grand = cells.mean(axis=(0, 1))
    posture = cells.mean(axis=0) - grand
    person_effect = cells.mean(axis=1) - grand
    interaction = cells - grand - posture[None] - person_effect[:, None]
    components = {"posture": np.square(posture).mean(axis=0), "person": np.square(person_effect).mean(axis=0),
        "person_posture": np.square(interaction).mean(axis=(0, 1)),
        "record_within_person_posture": np.zeros(values.shape[1]), "window_position_linear": np.zeros(values.shape[1]),
        "window_position_remainder": np.zeros(values.shape[1])}
    beta_records, beta_cells, position_available_mass, missing_reasons = [], [], 0., {"single_window": 0, "zero_span": 0}
    for number, row in enumerate(metadata):
        idx = row["indices"]
        person_index = subjects.index(row["subject"])
        class_id = row["class_id"]
        weight = 1 / (len(subjects) * 7 * len(index_by_cell[(row["subject"], class_id)]))
        window_weights[idx] = weight / len(idx)
        components["record_within_person_posture"] += weight * np.square(record_values[number] - cells[person_index, class_id])
        within = values[idx] - record_values[number]
        span = float(np.ptp(starts[idx]))
        available = len(idx) > 1 and span > 0
        if available:
            q = (starts[idx] - starts[idx].mean()) / span
            beta = np.einsum("n,nf->f", q, within, optimize=False) / np.square(q).sum()
            linear = q[:, None] * beta
            remainder = within - linear
            position_available_mass += weight
        else:
            beta, linear, remainder = None, np.zeros_like(within), within
            missing_reasons["single_window" if len(idx) == 1 else "zero_span"] += 1
        components["window_position_linear"] += weight * np.square(linear).mean(axis=0)
        components["window_position_remainder"] += weight * np.square(remainder).mean(axis=0)
        beta_records.append({"record_sha256": hashlib.sha256(row["record_id"].encode()).hexdigest(),
            "subject_sha256": hashlib.sha256(row["subject"].encode()).hexdigest(), "class_id": class_id, "windows": len(idx),
            "position_available": available, "balanced_record_mass": weight, "beta": None if beta is None else beta.tolist()})
    for person in subjects:
        for label in range(7):
            indices = index_by_cell[(person, label)]
            available = [beta_records[i] for i in indices if beta_records[i]["position_available"]]
            beta_cells.append({"subject_sha256": hashlib.sha256(person.encode()).hexdigest(), "class_id": label,
                "total_records": len(indices), "available_records": len(available),
                "available_record_mass": sum(row["balanced_record_mass"] for row in available),
                "beta_equal_available_records": None if not available else np.mean([row["beta"] for row in available], axis=0).tolist()})
    beta_classes = []
    for label in range(7):
        cells_available = [row for row in beta_cells if row["class_id"] == label and row["available_records"] > 0]
        beta_classes.append({"class_id": label, "total_people": len(subjects), "available_people": len(cells_available),
            "available_records": sum(row["available_records"] for row in cells_available),
            "available_record_mass": sum(row["available_record_mass"] for row in cells_available),
            "beta_equal_available_people": None if not cells_available else np.mean([row["beta_equal_available_records"] for row in cells_available], axis=0).tolist()})
    require(np.isclose(window_weights.sum(), 1., atol=1e-12, rtol=0), "Balanced weight mass is not1")
    total = np.einsum("n,nf->f", window_weights, np.square(values - grand), optimize=False)
    second = np.einsum("n,nf->f", window_weights, np.square(values), optimize=False)
    reconstructed = sum(components.values())
    errors = reconstructed - total
    tolerance = RTOL * total + EPS_FACTOR * np.finfo(np.float64).eps * np.maximum(second, np.finfo(np.float64).tiny)
    require(np.isfinite(reconstructed).all() and np.all(np.abs(errors) <= tolerance), "Variance decomposition fails frozen floating-point tolerance; no clipping")
    require(all(np.all(part >= 0) for part in components.values()), "Negative squared component; no clipping")
    fractions = {key: [None if total[i] == 0 else float(value[i] / total[i]) for i in range(len(total))] for key, value in components.items()}
    axes = [{"axis_index": i, "weighted_mean": float(grand[i]), "variance": float(total[i]),
        "components": {key: float(value[i]) for key, value in components.items()},
        "fractions": {key: value[i] for key, value in fractions.items()},
        "reconstruction_error": float(errors[i]), "reconstruction_tolerance": float(tolerance[i]),
        "relative_reconstruction_error": None if total[i] == 0 else float(errors[i] / total[i])} for i in range(values.shape[1])]
    summaries = []
    for vi, view in enumerate(VIEWS):
        for ki, frequency in enumerate(FREQUENCIES):
            indices = list(range(vi * 48 + ki * 16, vi * 48 + (ki + 1) * 16))
            if indices[-1] >= values.shape[1]:
                continue
            valid = [i for i in indices if total[i] > 0]
            summaries.append({"view": view, "frequency": frequency, "axes": 16, "nonzero_variance_axes": len(valid),
                "equal_axis_mean_fraction": {key: None if not valid else float(np.mean([fractions[key][i] for i in valid])) for key in COMPONENTS}})
    return {"windows": len(values), "people": len(subjects), "records": len(metadata), "balanced_window_mass": float(window_weights.sum()),
        "axes": axes, "block_summaries": summaries, "maximum_absolute_reconstruction_error": float(np.max(np.abs(errors))),
        "position": {"available_records": sum(row["position_available"] for row in beta_records), "total_records": len(metadata),
            "available_balanced_mass": position_available_mass, "excluded_reasons": missing_reasons,
            "available_person_class_cells": sum(row["available_records"] > 0 for row in beta_cells), "total_person_class_cells": len(beta_cells),
            "beta_record": beta_records, "beta_person_class": beta_cells, "beta_class": beta_classes,
            "missing_beta_not_imputed_as_zero": True}}, window_weights


def freeze(root, output):
    require(not output.exists(), "Use a fresh independent diagnostic run")
    origin = root / ORIGIN
    protocol = read(origin / "protocol.json")
    require(digest(origin / "protocol.json") == read(origin / "protocol_lock.json")["sha256"], "019 protocol changed")
    names = [origin / name for name in ("protocol.json", "protocol_lock.json", "dataset.json", "data.npz",
        "supplemental_manifest.json", "supplemental_training.npz", "protected_before.json", "audits/structured_fitted_identity.json")]
    names += [root / "research/models/checkpoints/research_baseline.json", root / "src/validated_checkpoints.json",
        root / "experiments/posture_study/classical.py"]
    require(digest(origin / "data.npz") == protocol["cache_sha256"] and digest(origin / "dataset.json") == protocol["dataset_sha256"], "019 cache changed")
    manifest = read(origin / "supplemental_manifest.json")
    require(digest(origin / "supplemental_training.npz") == protocol["supplemental_training"]["cache_sha256"] == manifest["cache_sha256"]
        and digest(origin / "supplemental_manifest.json") == protocol["supplemental_training"]["manifest_sha256"]
        and manifest["personal_reference_used"] is False, "019 ordinary cache linkage changed")
    require(read(names[8])["current_research_baseline"]["pipeline"] == protocol["baseline"], "Research baseline differs from019")
    for relative, expected in protocol["source_hashes"].items():
        require(digest(origin / "source_snapshot" / relative) == expected, "Frozen019 source changed")
    protected = read(origin / "protected_before.json")
    for relative, expected in protected.items():
        current = reference(root / relative, root)
        require({key: current[key] for key in expected} == expected, "Protected original changed")
    config = {"schema_version": 1, "version": VERSION, "frozen_at": now(), "timezone": "Asia/Seoul", "origin_run": ORIGIN,
        "origin_protocol_sha256": digest(origin / "protocol.json"), "origin_source_hashes": protocol["source_hashes"],
        "script_sha256": digest(__file__), "reader_source_sha256": hashlib.sha256(inspect.getsource(selected_rows).encode()).hexdigest(),
        "inputs": [reference(path, root) for path in names], "partitions": protocol["partitions"], "views": list(VIEWS), "frequencies": list(FREQUENCIES),
        "temporal_rule": "DCT-II orthonormal k2,3,4; residual against fixed36point fullmean/corelinear-slope/tail12mean/tail24mean/tail12enddiff/tail24enddiff span",
        "fixed_basis_rank_tolerance": "eps64*max(6,36)*largest singular value; all retained singular directions above tolerance; no measured-data SVD",
        "aggregation": "window weight1/(people*7*records_in_person_class*windows_in_record); complete person-class grid required",
        "variance_components": list(COMPONENTS), "decomposition_tolerance": {"relative_total_variance": RTOL, "roundoff_second_moment_factor": EPS_FACTOR,
            "formula": "1e-10*total +512*eps64*max(weighted_second_moment,tiny64); no negative clipping"},
        "position": "q=(start_s-mean_record_start)/ptp_record_start; beta=sum(q*withinrecord coefficient)/sum(q^2); 1window or zero span beta=null, omitted means with explicit counts/mass",
        "nonorthogonal_residual_axes": "Report individual raw/residual kernel norms and residualGram; never sum coefficient squares as independent frequency energy",
        "selection": "All144 raw and144 residual axes reported; no model/frequency/dimension/threshold selection",
        "scope": {"classifier_fits": 0, "scaler_fits": 0, "predictions": 0, "performance_evaluations": 0, "outer_validation_test_values_used": False},
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "environment": environment()}
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot/temporal_structure_audit.py"
    snapshot.parent.mkdir()
    with snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    snapshot.chmod(0o444)
    write(output / "diagnostic_config.json", config)
    write(output / "diagnostic_lock.json", {"config_sha256": digest(output / "diagnostic_config.json"), "script_sha256": digest(snapshot),
        "reader_source_sha256": config["reader_source_sha256"], "frozen_at": config["frozen_at"], "statistics_before_freeze": False})
    print(json.dumps({"output": str(output), "config_sha256": digest(output / "diagnostic_config.json"), "script_sha256": digest(snapshot)}))


def execute(root, output):
    config, lock = read(output / "diagnostic_config.json"), read(output / "diagnostic_lock.json")
    require(digest(output / "diagnostic_config.json") == lock["config_sha256"], "Diagnostic config changed")
    frozen = output / "source_snapshot/temporal_structure_audit.py"
    require(digest(frozen) == lock["script_sha256"] == config["script_sha256"], "Diagnostic source changed")
    require(environment() == config["environment"], "Frozen package environment changed")
    if Path(__file__).resolve() != frozen.resolve():
        subprocess.run([sys.executable, str(frozen), "run", "--root", str(root), "--output", str(output)], check=True)
        return
    require(not (output / "execution_start.json").exists(), "Diagnostic already attempted; preserve its evidence")
    origin = root / ORIGIN
    protected = read(origin / "protected_before.json")
    def verify_inputs():
        for evidence in config["inputs"]:
            require(reference(root / evidence["path"], root) == evidence, "Frozen input changed: " + evidence["path"])
        for relative, expected in protected.items():
            current = reference(root / relative, root)
            require({key: current[key] for key in expected} == expected, "Protected original changed: " + relative)
        for relative, expected in config["origin_source_hashes"].items():
            require(digest(origin / "source_snapshot" / relative) == expected, "Frozen019 source changed")
    verify_inputs()
    sys.path.insert(0, str(origin / "source_snapshot"))
    from experiments.iterative_improvement.record_training import append_record_training
    from src.posture_core import normalize_text, SENSOR_COLUMNS
    from src.window_pipeline import extract_window_features
    require(Path(inspect.getsourcefile(append_record_training)).resolve() == (origin / "source_snapshot/experiments/iterative_improvement/record_training.py").resolve(), "Unfrozen restoration helper import")
    write(output / "execution_start.json", {"started_at": now(), "config_sha256": lock["config_sha256"]})
    raw_kernel, residual_kernel, kernel_details = temporal_kernels()
    write(output / "temporal_kernels.json", kernel_details)
    metadata = {}
    for key, filename in (("base", "data.npz"), ("ordinary", "supplemental_training.npz")):
        with np.load(origin / filename, allow_pickle=False) as archive:
            metadata[key] = {"path": origin / filename, "groups": np.asarray([normalize_text(str(v)) for v in archive["groups"]]),
                "record_ids": np.asarray([normalize_text(str(v)) for v in archive["record_ids"]])}
    require(not set(metadata["base"]["record_ids"]) & set(metadata["ordinary"]["record_ids"]), "Base/ordinary record overlap")
    proof = read(origin / "audits/structured_fitted_identity.json")
    require(proof["protocol_sha256"] == config["origin_protocol_sha256"], "019 TRAIN proof differs")
    results, coverage = [], []
    axis_names = [f"{view}:k{k}:{sensor}" for view in VIEWS for k in FREQUENCIES for sensor in SENSOR_COLUMNS]
    for partition in config["partitions"]:
        part_start = now()
        people = sorted(normalize_text(v) for v in partition["train"])
        others = {normalize_text(v) for key in ("validation", "test") for v in partition[key]}
        require(not set(people) & others, "OuterTRAIN subject overlap")
        selected = {}
        for key, cache in metadata.items():
            take = np.flatnonzero(np.isin(cache["groups"], people))
            selected[key] = {name: selected_rows(cache["path"], name, take, len(cache["groups"])) for name in ("seq", "X", "y", "start_s")}
            selected[key].update(groups=cache["groups"][take], record_ids=cache["record_ids"][take], indices=take)
        base, extra = selected["base"], selected["ordinary"]
        X, labels, groups, records, restoration = append_record_training(base, extra, np.arange(len(base["X"])))
        sequence = np.concatenate((base["seq"], extra["seq"]))
        starts = np.concatenate((base["start_s"], extra["start_s"]))
        require(np.array_equal(np.concatenate((base["y"], extra["y"])), labels)
            and np.array_equal(np.concatenate((base["record_ids"], extra["record_ids"])), records), "TRAIN sequence/record row alignment differs")
        scope = {"training_input_sha256": array_hash(X), "training_labels_sha256": array_hash(labels.astype(np.int64)),
            "training_groups_sha256": json_hash(groups.tolist()), "training_records_sha256": json_hash(records.tolist())}
        previous = next(row for row in proof["train_scopes"] if row["partition"] == partition["id"])
        require(all(previous[key] == value for key, value in scope.items()) and previous["restoration"] == restoration, "Reconstructed TRAIN differs from019 actual fit")
        require(base["indices"].tolist() == previous["base_train_indices"] and extra["indices"].tolist() == previous["ordinary_train_indices"], "Actual019 TRAIN index mapping differs")
        untouched = {"seq": array_hash(sequence), "X": array_hash(X), "y": array_hash(labels), "start_s": array_hash(starts)}
        rebuilt = extract_window_features(sequence)
        require(rebuilt.dtype == X.dtype and rebuilt.shape == X.shape and rebuilt.tobytes() == X.tobytes(), "Actual TRAIN seq/features fail bitwise parity")
        require(set(groups) == set(people), "Incomplete TRAIN people")
        raw_values, residual_values = coefficients(sequence, raw_kernel, residual_kernel)
        summaries = {}
        for name, values in (("raw", raw_values), ("residual", residual_values)):
            summaries[name], weights = balanced_decomposition(values, labels, groups, records, starts)
        require(untouched == {"seq": array_hash(sequence), "X": array_hash(X), "y": array_hash(labels), "start_s": array_hash(starts)}, "Diagnostic mutated selected input arrays")
        folder = output / "train_temporal" / partition["id"]
        folder.mkdir(parents=True)
        arrays = {"raw_coefficients": raw_values, "residual_coefficients": residual_values, "weights": weights,
            "labels": labels, "groups": groups, "record_ids": records, "start_s": starts, "axis_names": np.asarray(axis_names)}
        path = folder / "coefficients.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        path.chmod(0o444)
        origins = {(key, int(index)) for key, data in selected.items() for index in data["indices"]}
        coverage.append(origins)
        detail = {"partition": partition["id"], "started_at": part_start, "finished_at": now(), **scope,
            "sequence_sha256": array_hash(sequence), "start_s_sha256": array_hash(starts), "feature_parity_bitwise": True,
            "windows": len(labels), "records": len(set(records)), "people": len(set(groups)),
            "base_train_indices": base["indices"].tolist(), "ordinary_train_indices": extra["indices"].tolist(), "restoration": restoration,
            "arrays": reference(path, root), "array_hashes": {key: array_hash(value) for key, value in arrays.items()},
            "axis_names": axis_names, "summaries": summaries}
        write(folder / "temporal_structure.json", detail)
        write(folder / "temporal_seal.json", {"sha256": digest(folder / "temporal_structure.json"), "arrays_sha256": digest(path), "config_sha256": lock["config_sha256"]})
        results.append({key: value for key, value in detail.items() if key != "summaries"})
        results[-1]["summary"] = {name: {key: value for key, value in summary.items() if key not in ("axes", "position")}
            for name, summary in summaries.items()}
        results[-1]["position_scope"] = {key: value for key, value in summaries["raw"]["position"].items() if not key.startswith("beta_")}
        results[-1]["detail"] = reference(folder / "temporal_structure.json", root)
        print(json.dumps({"completed_train_partition": partition["id"], "windows": len(labels), "axes_per_representation": raw_values.shape[1]}), flush=True)
    verify_inputs()
    result = {"schema_version": 1, "version": VERSION, "started_at": read(output / "execution_start.json")["started_at"], "finished_at": now(),
        "config_sha256": lock["config_sha256"], "script_sha256": lock["script_sha256"], "reader_source_sha256": lock["reader_source_sha256"],
        "origin_protocol_sha256": config["origin_protocol_sha256"], "kernels": reference(output / "temporal_kernels.json", root),
        "partitions": results, "split_times_train_windows": sum(map(len, coverage)), "unique_train_source_windows": len(set.union(*coverage)),
        "protected_originals_unchanged": True, "protected_originals_count": len(protected), "origin_sources_unchanged": True,
        "origin_source_count": len(config["origin_source_hashes"]), "inputs_unchanged": True, "classifier_fits": 0,
        "scaler_fits": 0, "predictions": 0, "performance_evaluations": 0, "outer_validation_test_numeric_rows_used": False,
        "limitations": ["Temporal kernel residuals remove only a fixed signed linear summary span, not information in all596 nonlinear features or H07std.",
            "Residual coefficient axes are correlated; their squared values are not summed as independent spectral energy.",
            "Balanced components describe TRAIN mean association, not causation, discriminative power or out-of-person generalization.",
            "Small posture mean component does not exclude class-dependent covariance, variance or phase information.",
            "Person/record/position association cannot identify sensor drift, physiology, intended movement or independent acquisition sessions.",
            "Nominal3second windows do not overlap, but records and outerTRAIN partitions are dependent; interpolation can share raw support.",
            "No frequencies, representation, rank, model or performance threshold were selected from this diagnostic."]}
    write(output / "temporal_structure_audit.json", result)
    write(output / "audit_seal.json", {"sha256": digest(output / "temporal_structure_audit.json"), "config_sha256": lock["config_sha256"], "sealed_at": now()})
    print(json.dumps({"output": str(output), "sha256": digest(output / "temporal_structure_audit.json")}))


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.npz"
        X = np.arange(7 * 36 * 16, dtype=np.float32).reshape(7, 36, 16)
        np.savez_compressed(path, seq=X, y=np.arange(7), X=X.reshape(7, -1), F=np.asfortranarray(X))
        for key, values in (("seq", X), ("y", np.arange(7)), ("X", X.reshape(7, -1))):
            require(np.array_equal(selected_rows(path, key, np.array([0, 3, 4, 6]), 7), values[[0, 3, 4, 6]]), "Selected reader mapping failed")
        for take in (np.array([-1]), np.array([7]), np.array([3, 1]), np.array([1, 1]), np.array([1.]),
                np.array([3, 1], dtype=np.uint64), np.array([2**64 - 1], dtype=np.uint64), np.array([], dtype=int)):
            try:
                selected_rows(path, "seq", take, 7)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid selected indices accepted")
        try:
            selected_rows(path, "F", np.array([0]), 7)
        except ValueError:
            pass
        else:
            raise AssertionError("Fortran NPY accepted")
    raw, residual, details = temporal_kernels()
    require(np.allclose(raw @ raw.T, np.eye(3), atol=1e-12), "DCT kernels are not orthonormal")
    require(np.allclose(residual @ np.asarray(details["basis"]).T, 0., atol=1e-12), "Residual span test failed")
    synthetic = np.arange(3 * 36 * 16, dtype=np.float32).reshape(3, 36, 16)
    original = synthetic.tobytes()
    a, b = coefficients(synthetic, raw, residual)
    for index in range(3):
        one_a, one_b = coefficients(synthetic[index:index+1], raw, residual)
        require(a[index].tobytes() == one_a[0].tobytes() and b[index].tobytes() == one_b[0].tobytes(), "Coefficient batch invariance failed")
    require(original == synthetic.tobytes(), "Coefficient input changed")
    labels, people, records, starts, values = [], [], [], [], []
    for person in range(3):
        for label in range(7):
            for record in range(1 + person):
                for window in range(3):
                    labels.append(label); people.append(str(person)); records.append(f"{person}:{label}:{record}")
                    starts.append(window * 3.); values.append([label * 2 + person * 5 + window - 1.])
    result, weights = balanced_decomposition(np.asarray(values), np.asarray(labels), np.asarray(people), np.asarray(records), np.asarray(starts))
    axis = result["axes"][0]
    require(np.isclose(axis["components"]["posture"], 16.) and np.isclose(axis["components"]["person"], 50/3), "Balanced person/posture decomposition failed")
    require(np.isclose(axis["components"]["window_position_linear"], 2/3) and axis["components"]["window_position_remainder"] == 0, "Known position trend decomposition failed")
    for subject in range(3):
        require(np.isclose(weights[np.asarray(people) == str(subject)].sum(), 1/3), "Unequal record count reweighted person")
    constant, _ = balanced_decomposition(np.zeros_like(values, dtype=float), np.asarray(labels), np.asarray(people), np.asarray(records), np.zeros(len(starts)))
    require(constant["axes"][0]["variance"] == 0 and constant["axes"][0]["fractions"]["posture"] is None,
        "Constant axis must have null fractions")
    require(constant["position"]["available_records"] == 0 and all(row["beta"] is None for row in constant["position"]["beta_record"]), "Missing position was imputed")
    print(json.dumps({"synthetic_tests": "passed", "real_data_used": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "self-test"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "self-test":
        self_test()
        return
    root = args.root.resolve()
    output = args.output or Path("research/results") / (datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H%M%S") + "_temporal_structure_audit")
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    require(output.is_relative_to(root / "research/results") and not output.is_relative_to(root / ORIGIN), "Independent diagnostic output required")
    if args.action == "freeze":
        freeze(root, output)
    else:
        require(args.output is not None, "Specify frozen output")
        execute(root, output)


if __name__ == "__main__":
    main()
