"""Describe pairwise class support using only frozen TRAIN rows; never fit or predict."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile
from zoneinfo import ZoneInfo

import numpy as np


VERSION = "pair_support_audit_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(path):
    path = Path(path)
    stat = path.stat()
    return {"sha256": file_hash(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def array_hash(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(json.dumps([values.dtype.str, list(values.shape)]).encode("utf-8"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def selected_rows(path, key, indices, expected_rows):
    """Read only chosen rows into numeric arrays, including from compressed NPZ."""
    take = np.asarray(indices)
    require(take.ndim == 1 and take.dtype.kind in "iu" and len(take) > 0
        and np.all(take[1:] > take[:-1]), "Selected rows must be sorted unique integers")
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        require(version in ((1, 0), (2, 0)), "Unsupported NPY header version")
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        require(not fortran and not dtype.hasobject and dtype.kind in "fiu"
            and len(shape) in (1, 2) and shape[0] == expected_rows, "Unsupported numeric TRAIN schema")
        require(int(take[0]) >= 0 and int(take[-1]) < shape[0], "Selected row outside archive")
        header_end = stream.tell()
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        result = np.empty((len(take), *shape[1:]), dtype=dtype)
        boundaries = np.r_[0, np.flatnonzero(take[1:] != take[:-1] + 1) + 1, len(take)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            stream.seek(header_end + int(take[left]) * row_bytes)
            count = int(right - left)
            buffer = stream.read(count * row_bytes)
            require(len(buffer) == count * row_bytes, "Truncated selected TRAIN block")
            result[left:right] = np.frombuffer(buffer, dtype=dtype).reshape((count, *shape[1:]))
    return result


def summarize(labels, people, records, class_names):
    """Count windows/recordings/people; no classifier, features or scores used."""
    from .record_training import validate_record_identities
    labels, people, records = map(np.asarray, (labels, people, records))
    validate_record_identities(labels, people, records)
    require(np.array_equal(np.unique(labels), np.arange(7)) and len(class_names) == 7,
        "Pair support requires all seven original classes")
    subjects = sorted(set(people.tolist()))
    per_class, subject_sets = [], []
    for class_id in range(7):
        mask = labels == class_id
        present = set(people[mask].tolist())
        subject_sets.append(present)
        windows_by_subject = [int(np.count_nonzero(mask & (people == person))) for person in subjects]
        records_by_subject = [len(np.unique(records[mask & (people == person)])) for person in subjects]
        per_class.append({"class_id": class_id, "class_name": class_names[class_id],
            "windows": int(mask.sum()), "records": len(np.unique(records[mask])), "subjects": len(present),
            "missing_training_subjects": len(subjects) - len(present),
            "windows_per_training_subject": {"minimum": min(windows_by_subject), "median": float(np.median(windows_by_subject)), "maximum": max(windows_by_subject)},
            "records_per_training_subject": {"minimum": min(records_by_subject), "median": float(np.median(records_by_subject)), "maximum": max(records_by_subject)}})
    per_subject = []
    for person in subjects:
        mask = people == person
        per_subject.append({"subject_sha256": hashlib.sha256(person.encode("utf-8")).hexdigest(),
            "class_windows": [int(np.count_nonzero(mask & (labels == c))) for c in range(7)],
            "class_records": [len(np.unique(records[mask & (labels == c)])) for c in range(7)]})
    pairs = []
    for pair_index, (a, b) in enumerate(combinations(range(7), 2)):
        first, second = per_class[a], per_class[b]
        n = first["windows"] + second["windows"]
        both = subject_sets[a] & subject_sets[b]
        joined = subject_sets[a] | subject_sets[b]
        pairs.append({"pair_index": pair_index, "classes": [a, b], "class_names": [class_names[a], class_names[b]],
            "class_windows": [first["windows"], second["windows"]],
            "class_records": [first["records"], second["records"]],
            "class_subjects": [first["subjects"], second["subjects"]],
            "windows": n, "records": first["records"] + second["records"],
            "subjects_union": len(joined), "subjects_with_both_classes": len(both),
            "subjects_with_only_first": len(subject_sets[a] - subject_sets[b]),
            "subjects_with_only_second": len(subject_sets[b] - subject_sets[a]),
            "window_count_ratio_max_min": max(first["windows"], second["windows"]) / min(first["windows"], second["windows"]),
            "proposed_balanced_binary_weights": [n / (2 * first["windows"]), n / (2 * second["windows"])],
            "windows_per_shared_subject_by_class": [{"subject_sha256": hashlib.sha256(person.encode("utf-8")).hexdigest(),
                "windows": [int(np.count_nonzero((people == person) & (labels == c))) for c in (a, b)],
                "records": [len(np.unique(records[(people == person) & (labels == c)])) for c in (a, b)]}
                for person in sorted(both)]})
    require(len(pairs) == 21 and sum(pair["windows"] for pair in pairs) == 6 * len(labels),
        "Pair enumeration must include every window exactly six times")
    _, record_counts = np.unique(records, return_counts=True)
    return {"training_windows": len(labels), "training_records": len(np.unique(records)),
        "training_subjects": len(subjects), "record_window_counts": dict(sorted(Counter(record_counts.tolist()).items())),
        "classes": per_class, "subjects": per_subject, "pairs": pairs,
        "pair_window_exposures": sum(pair["windows"] for pair in pairs),
        "tree_row_work_proxy": {"definition": "number_of_trees * supplied_rows; not a measured time or memory estimate",
            "proposed_32_per_pair": 32 * 6 * len(labels), "multiclass_200": 200 * len(labels), "ratio": .96}}


def execute(root, run, output):
    started = now()
    require(not output.exists() and not output.is_relative_to(run) and not output.is_relative_to(root / "data"),
        "Output must be a new independent directory outside the input run and raw data")
    require(output.is_relative_to(root / "research/results") or output.is_relative_to(Path("/private/tmp")),
        "Diagnostic output must be under research/results or /private/tmp")
    protocol = read_json(run / "protocol.json")
    protocol_hash = file_hash(run / "protocol.json")
    require(protocol_hash == read_json(run / "protocol_lock.json")["sha256"], "Protocol seal changed")
    input_paths = [run / name for name in ("protocol.json", "protocol_lock.json", "data.npz", "dataset.json",
        "supplemental_training.npz", "supplemental_manifest.json", "protected_before.json", "audits/alternatives_fitted_identity.json")]
    inputs = {str(path.relative_to(root)): evidence(path) for path in input_paths}
    require(file_hash(run / "data.npz") == protocol["cache_sha256"]
        and file_hash(run / "dataset.json") == protocol["dataset_sha256"], "Frozen input cache or manifest changed")
    supplement = read_json(run / "supplemental_manifest.json")
    for key, name in (("manifest_sha256", "supplemental_manifest.json"), ("cache_sha256", "supplemental_training.npz")):
        require(file_hash(run / name) == protocol["supplemental_training"][key], "Frozen supplement changed")
    require(protocol["supplemental_training"]["policy"] == "include_ordinary_reserved_training_records_v1"
        and supplement["personal_reference_used"] is False
        and supplement["cache_sha256"] == file_hash(run / "supplemental_training.npz"), "Unexpected supplement policy")
    sources = {}
    for rel in ("experiments/iterative_improvement/record_training.py", "experiments/iterative_improvement/supplemental.py", "src/posture_core.py"):
        require(file_hash(root / rel) == protocol["source_hashes"][rel] == file_hash(run / "source_snapshot" / rel),
            "Frozen helper changed: " + rel)
        sources[rel] = evidence(root / rel)
    from .record_training import append_record_training
    from src.posture_core import CLASS_NAMES, normalize_text
    protected = read_json(run / "protected_before.json")
    require({rel: evidence(root / rel) for rel in protected} == protected, "Protected originals differ from016 freeze")
    fitted = read_json(run / "audits/alternatives_fitted_identity.json")
    require(fitted["passed"] and fitted["protocol_sha256"] == protocol_hash, "Existing fitted TRAIN audit is invalid")
    expected_train = {row["fold"]: row for row in fitted["results"] if row["candidate"] == "TAIL_identity"}
    caches = {}
    for kind, filename in (("base", "data.npz"), ("supplement", "supplemental_training.npz")):
        with np.load(run / filename, allow_pickle=False) as archive:
            groups, records = archive["groups"], archive["record_ids"]
        require(groups.ndim == 1 and records.shape == groups.shape, "Invalid identity metadata shape")
        caches[kind] = {"path": run / filename,
            "groups": np.asarray([normalize_text(str(value)) for value in groups]),
            "record_ids": np.asarray([normalize_text(str(value)) for value in records])}
    require(not set(caches["base"]["record_ids"]) & set(caches["supplement"]["record_ids"]), "Base/supplement recording overlap")
    eligible = {normalize_text(row["record_id"]): row for row in supplement["records"]}
    for record in set(caches["supplement"]["record_ids"]):
        row = eligible[record]
        require(row["label"] == "정자세" and row["eligibility"]["explicit_registration"] is False
            and row["eligibility"]["globally_excluded"] is False and row["eligibility"]["training_role"] == "include",
            "Ineligible ordinary supplement")
    partitions, window_sets = [], []
    for partition in protocol["partitions"]:
        require(partition["id"] in ("development_1", "development_2", "development_3"), "Only fixed development TRAIN partitions are allowed")
        train_subjects = {normalize_text(person) for person in partition["train"]}
        excluded_subjects = {normalize_text(person) for key in ("validation", "test") for person in partition[key]}
        require(not train_subjects & excluded_subjects, "TRAIN/held-out subject overlap")
        chosen = {}
        for kind, cache in caches.items():
            take = np.flatnonzero(np.isin(cache["groups"], sorted(train_subjects)))
            chosen[kind] = {"X": selected_rows(cache["path"], "X", take, len(cache["groups"])),
                "y": selected_rows(cache["path"], "y", take, len(cache["groups"])),
                "groups": cache["groups"][take], "record_ids": cache["record_ids"][take], "indices": take}
            require(set(chosen[kind]["groups"]) == train_subjects, "Incomplete TRAIN subject selection")
        base, extra = chosen["base"], chosen["supplement"]
        expected = partition["counts"]["train"]
        require(len(base["X"]) == expected["windows"] and len(np.unique(base["record_ids"])) == expected["records"]
            and np.bincount(base["y"], minlength=7).tolist() == expected["class_counts"], "Frozen base TRAIN counts changed")
        for record in set(extra["record_ids"]):
            require(int(np.count_nonzero(extra["record_ids"] == record)) == eligible[record]["windows"], "Incomplete ordinary recording")
        values, labels, people, records, restoration = append_record_training(base, extra, np.arange(len(base["X"])))
        hashes = {"training_input_sha256": array_hash(values), "training_labels_sha256": array_hash(labels),
            "training_groups_sha256": json_hash(people.tolist())}
        require(all(expected_train[partition["id"]][key] == digest for key, digest in hashes.items()),
            "Selected rows differ from fitted016 TRAIN identity")
        require(values.dtype == np.float32 and values.shape == (len(labels), 596), "Frozen feature dtype or shape changed")
        row = summarize(labels, people, records, CLASS_NAMES)
        row.update(partition=partition["id"], input_hashes=hashes, record_ids_sha256=array_hash(records),
            base_train_indices=base["indices"].tolist(), supplemental_train_indices=extra["indices"].tolist(),
            restored_windows=restoration["added_windows"], restored_records=restoration["added_records"], fitted_train_hashes_match=True)
        partitions.append(row)
        window_sets.append({f"{kind}:{int(index)}" for kind in chosen for index in chosen[kind]["indices"]})
    require({rel: evidence(root / rel) for rel in protected} == protected, "Protected originals changed during diagnostic")
    require({rel: evidence(root / rel) for rel in inputs} == inputs, "Frozen inputs changed during diagnostic")
    require({rel: evidence(root / rel) for rel in sources} == sources, "Frozen helpers changed during diagnostic")
    script = Path(__file__).resolve()
    sources[str(script.relative_to(root))] = evidence(script)
    result = {"schema_version": 1, "version": VERSION, "started_at": started, "finished_at": now(),
        "timezone": "Asia/Seoul", "input_run": str(run.relative_to(root)), "protocol_sha256": protocol_hash,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "command": [sys.executable, "-B", "-m", "experiments.iterative_improvement.pair_support_audit",
            "--input-run", str(run), "--output", str(output)],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "executable": sys.executable},
        "inputs": inputs, "sources": sources, "script_source": script.read_text(encoding="utf-8"),
        "protected_originals": {"count": len(protected), "before_equals016_freeze": True, "before_equals_after": True,
            "manifest_sha256": json_hash(protected), "files": protected},
        "method": {"pairs": "lexicographic itertools.combinations(range(7),2), pair_index0..20",
            "support": "window, distinct recording and distinct subject counts; shared-subject intersection separately",
            "feature_use": "selected TRAIN X only to reconstruct existing append_training input and verify its hash; no feature distributions analyzed",
            "identity_metadata": "full groups/record IDs used solely to select TRAIN and check disjointness",
            "compressed_io_caveat": "hashing/ZIP seeking may read or decompress skipped bytes; held-out X/y values never converted to arrays or analyzed",
            "seed_proposal_only": "pair index k seed=(primary_seed+k)%2**32; no estimator constructed",
            "aggregation_proposal_only": "sum six aligned incident pair probabilities per class, divide by21; valid scores, not posterior calibration; every score<=2/7"},
        "overlap": {"split_window_observations": sum(map(len, window_sets)), "unique_source_windows": len(set.union(*window_sets)),
            "pairwise_shared_source_windows": {f"{partitions[i]['partition']}__{partitions[j]['partition']}": len(window_sets[i] & window_sets[j])
                for i in range(len(window_sets)) for j in range(i + 1, len(window_sets))},
            "note": "TRAIN partitions overlap; repeated windows/subjects are not independent replications"},
        "partitions": partitions, "new_training": False, "new_prediction": False, "new_evaluation": False,
        "heldout_feature_or_label_arrays_decoded": False, "originals_mutated": False,
        "limitations": ["Counts establish presence, not pairwise separability or model accuracy.",
            "Recording windows share subjects and are not independent sample size.",
            "Current runtime confidence thresholds are not compatible with scores capped at2/7 without a separately reviewed contract."]}
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "pair_support_audit.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    destination.chmod(0o444)
    seal = output / "pair_support_seal.json"
    with seal.open("x", encoding="utf-8") as stream:
        json.dump({"sha256": file_hash(destination), "source_sha256": file_hash(script)}, stream, indent=2)
        stream.write("\n")
    seal.chmod(0o444)
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
