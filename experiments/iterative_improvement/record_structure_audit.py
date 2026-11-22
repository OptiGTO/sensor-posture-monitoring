"""Describe training-record contributions without fitting or scoring models."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .record_training import append_record_training
from .run import digest, indices, load_data, now, verify


def audit(out):
    protocol, data = verify(out), load_data(out)
    with np.load(out / "supplemental_training.npz", allow_pickle=False) as archive:
        supplement = dict(archive)
    rows = []
    for part in protocol["partitions"]:
        train, unused_validation = indices(data, part)
        _, labels, people, records, augmentation = append_record_training(data, supplement, train)
        unique, counts = np.unique(records, return_counts=True)
        record_classes = {record: int(labels[np.flatnonzero(records == record)[0]]) for record in unique}
        records_per_class = np.bincount(list(record_classes.values()), minlength=7)
        windows_per_class = np.bincount(labels, minlength=7)
        record_sizes = dict(zip(unique.tolist(), counts.tolist()))
        ratios = np.array([windows_per_class[label] / (records_per_class[label] * record_sizes[record])
                           for label, record in zip(labels, records)])
        contributions = sorted(Counter(people.tolist()).values(), reverse=True)
        rows.append({"partition": part["id"], "training_subjects": sorted(set(people.tolist())),
            "training_records": len(unique), "training_windows": len(labels),
            "record_window_count_distribution": dict(sorted(Counter(counts.tolist()).items())),
            "class_record_counts": records_per_class.tolist(), "class_window_counts": windows_per_class.tolist(),
            "record_weight_over_baseline_class_weight": {"min": float(ratios.min()), "max": float(ratios.max()),
                "mean_absolute_change": float(np.abs(ratios - 1).mean())},
            "subject_window_range": [min(contributions), max(contributions)],
            "top_three_subject_window_fraction": sum(contributions[:3]) / len(labels),
            "supplemental_training_audit": augmentation})
    return {"created_at": now(), "scope": "Training rows only; no model fitting, predictions or evaluation scores",
        "protocol_sha256": digest(out / "protocol.json"), "cache_sha256": digest(out / "data.npz"),
        "supplemental_cache_sha256": digest(out / "supplemental_training.npz"),
        "weight_ratio_formula": "class_window_count / (class_record_count * windows_in_record)",
        "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    result = audit(out)
    destination = out / "audits/training_record_structure.json"
    destination.parent.mkdir(exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(destination)


if __name__ == "__main__":
    main()
