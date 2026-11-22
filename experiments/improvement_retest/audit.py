"""Independent, immutable evidence gates around the frozen experiment runner.

Run development before selection, selection before final fitting, and final
after summary creation. Audits never fit, select, or activate a model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.improvement_retest import run
from src.posture_core import CLASS_NAMES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, message):
    """Exact structural equality, tolerance only for finite floating metrics."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), message)
        for key in expected:
            equal(actual[key], expected[key], f"{message}: {key}")
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), message)
        for index, value in enumerate(expected):
            equal(actual[index], value, f"{message}: {index}")
    elif isinstance(expected, (float, np.floating)):
        require(isinstance(actual, (int, float)) and np.isfinite(actual)
                and np.isclose(actual, expected, rtol=1e-12, atol=1e-12), message)
    else:
        require(actual == expected, message)


def _files(out, paths):
    return {str(Path(path).relative_to(out)): run.digest(path) for path in sorted(set(paths))}


def _verify_saved_audit(out, stage):
    path = out / "audits" / f"{stage}.json"
    require(run.digest(path) == run.read(path.with_suffix(".sha256.json"))["sha256"], f"{stage} audit changed")
    value = run.read(path)
    require(value["audit_source_sha256"] == run.digest(Path(__file__)), "Audit implementation changed after evidence lock")
    for relative, expected in value["files"].items():
        require(run.digest(out / relative) == expected, f"{stage} audited evidence changed: {relative}")
    return value


def _save_audit(out, stage, paths, details):
    source_copy = out / "audits" / "audit_source.py"
    if not source_copy.exists():
        source_copy.parent.mkdir(parents=True, exist_ok=True)
        source_copy.write_bytes(Path(__file__).read_bytes())
    require(run.digest(source_copy) == run.digest(Path(__file__)), "Frozen audit source differs")
    paths = [*paths, source_copy]
    files = _files(out, paths)
    path = out / "audits" / f"{stage}.json"
    if path.exists():
        value = _verify_saved_audit(out, stage)
        equal(value["files"], files, f"{stage} audit file membership changed")
        equal(value["details"], details, f"{stage} audit conclusions changed")
        return value
    value = {"schema_version": 1, "stage": stage, "passed": True, "completed_at": run.now(),
             "audit_source_sha256": run.digest(Path(__file__)), "files": files, "details": details}
    run.write_json(path, value)
    run.write_json(path.with_suffix(".sha256.json"), {"sha256": run.digest(path)})
    return value


def _context(out):
    protocol = run.verify(out)
    data = dict(np.load(out / "data.npz", allow_pickle=False))
    expected_subjects, previous_holdout = set(data["groups"]), None
    for part in run.partitions(protocol):
        sets = [set(part[name]) for name in ("train", "validation", "test")]
        require(all(sets), "Empty subject partition")
        require(not any(sets[i] & sets[j] for i in range(3) for j in range(i)), "Subject partitions overlap")
        require(set.union(*sets) == expected_subjects, "Partitions do not exactly cover frozen subjects")
        require(previous_holdout is None or sets[2] == previous_holdout, "Historical holdout differs across development splits")
        previous_holdout = sets[2]
        records = []
        for name, identities in zip(("train", "validation", "test"), sets):
            mask = np.isin(data["groups"], list(identities))
            require(set(data["y"][mask]) == set(range(7)), f"Class missing in {part['id']}/{name}")
            records.append(set(data["record_ids"][mask]))
        require(not any(records[i] & records[j] for i in range(3) for j in range(i)), "Recording partitions overlap")
    return protocol, data


def _prediction(path, data, indices):
    prediction = dict(np.load(path, allow_pickle=False))
    for key, value in {"indices": indices, "truth": data["y"][indices], "groups": data["groups"][indices]}.items():
        require(np.array_equal(prediction[key], value), f"Prediction {key} differs from frozen partition: {path}")
    probability = prediction["probability"]
    require(probability.shape == (len(indices), 7) and np.isfinite(probability).all(), "Invalid probability shape/values")
    require(np.all((probability >= 0) & (probability <= 1))
        and np.allclose(probability.sum(1), 1, rtol=0, atol=1e-6), "Probabilities are not normalized")
    require(np.array_equal(prediction["prediction"], probability.argmax(1)), "Prediction differs from probability argmax")
    return prediction


def _metrics(row, prediction):
    equal(row["metrics"], run.scores(prediction["truth"], prediction["prediction"]), "Metrics differ from frozen predictions")
    expected = {subject: run.scores(prediction["truth"][prediction["groups"] == subject],
        prediction["prediction"][prediction["groups"] == subject]) for subject in sorted(set(prediction["groups"]))}
    equal(row["per_subject"], expected, "Per-subject metrics differ from frozen predictions")


def audit_development(output):
    out = Path(output)
    if not (out / "audits/development.json").exists():
        require(not (out / "selection_lock.json").exists() and not (out / "final").exists(),
                "Development audit must precede selection and holdout evaluation")
    protocol, data = _context(out)
    paths = [out / name for name in ("protocol.json", "protocol_lock.json", "dataset.json", "data.npz")]
    count = 0
    for family, methods in protocol["candidates"].items():
        for method, grid in methods.items():
            for part in run.partitions(protocol):
                indices = np.flatnonzero(np.isin(data["groups"], part["validation"]))
                for index, params in enumerate(grid):
                    stem = out / "development" / family / method / part["id"] / f"{index:02}"
                    metadata, predictions = stem.with_suffix(".json"), stem.with_suffix(".npz")
                    row = run.read(metadata)
                    for key, expected in {"family": family, "method": method, "split": part["id"],
                        "candidate_index": index, "params": params,
                        "protocol_sha256": run.digest(out / "protocol.json"),
                        "prediction_sha256": run.digest(predictions)}.items():
                        equal(row[key], expected, f"Development {key} differs")
                    prediction = _prediction(predictions, data, indices)
                    _metrics(row, prediction)
                    paths.extend([metadata, predictions])
                    count += 1
    return _save_audit(out, "development", paths, {"fits": count, "partitions_disjoint": True,
        "metrics_recomputed": True, "historical_holdout_untouched_by_development": True})


def _expected_selection(out, protocol):
    selected = {}
    criteria = protocol["criteria"]
    for family, methods in protocol["candidates"].items():
        for method, grid in methods.items():
            candidates = []
            for index, params in enumerate(grid):
                rows = [run.read(out / "development" / family / method / part["id"] / f"{index:02}.json")
                        for part in run.partitions(protocol)]
                values = [row["metrics"]["macro_f1"] for row in rows]
                epochs = [row["training_details"].get("best_epoch") for row in rows]
                candidates.append({"candidate_index": index, "params": params, "development_f1": values,
                    "mean_macro_f1": float(np.mean(values)),
                    "mean_worst_f1": float(np.mean([row["metrics"]["worst_class_f1"] for row in rows])),
                    "refit_epochs": int(np.median(epochs)) if all(value is not None for value in epochs) else None})
            best = max(candidates, key=lambda row: (row["mean_macro_f1"], row["mean_worst_f1"], -row["candidate_index"]))
            selected[f"{family}/{method}"] = {"family": family, "method": method, **best}
    for row in selected.values():
        baseline = selected[f"{row['family']}/standard"]
        delta = np.array(row["development_f1"]) - baseline["development_f1"]
        row.update(development_deltas=delta.tolist(), mean_development_delta=float(delta.mean()),
            development_qualified=bool(row["method"] != "standard"
                and delta.mean() >= criteria["minimum_development_mean_delta"]
                and int(np.sum(delta > 0)) >= criteria["minimum_positive_development_splits"]))
    return selected


def audit_selection(output):
    out = Path(output)
    if not (out / "audits/selection.json").exists():
        require(not (out / "final").exists(), "Selection audit must precede holdout evaluation")
    protocol, _ = _context(out)
    _verify_saved_audit(out, "development")
    selection = run.load_selection(out)
    equal(selection["protocol_sha256"], run.digest(out / "protocol.json"), "Selection protocol binding differs")
    require(selection["heldout_predictions_exist"] is False, "Selection records previous holdout exposure")
    equal(selection["selected"], _expected_selection(out, protocol), "Selected candidates or criteria differ from development evidence")
    return _save_audit(out, "selection", [out / name for name in (
        "protocol.json", "selection_lock.json", "selection_lock.sha256.json",
        "audits/development.json", "audits/development.sha256.json")],
        {"selection_recomputed": True, "holdout_not_used_for_selection": True,
         "selected_methods": len(selection["selected"])})


def audit_final(output):
    out = Path(output)
    protocol, data = _context(out)
    _verify_saved_audit(out, "development")
    _verify_saved_audit(out, "selection")
    selection = run.load_selection(out)
    equal(selection["protocol_sha256"], run.digest(out / "protocol.json"), "Final protocol binding differs")
    summary = run.read(out / "summary.json")
    for key, expected in {"protocol_sha256": run.digest(out / "protocol.json"),
        "selection_sha256": run.digest(out / "selection_lock.json"), "criteria": protocol["criteria"]}.items():
        equal(summary[key], expected, f"Summary {key} differs")
    holdout = run.partitions(protocol)[0]["test"]
    indices = np.flatnonzero(np.isin(data["groups"], holdout))
    paths = [out / name for name in ("summary.json", "protocol.json", "selection_lock.json",
        "audits/development.json", "audits/development.sha256.json", "audits/selection.json", "audits/selection.sha256.json")]
    finals = {}
    for key, selected in selection["selected"].items():
        if selected["method"] != "standard" and not selected["development_qualified"]:
            require(not (out / "final" / key).exists(), "Unqualified method was evaluated on holdout")
            continue
        directory = out / "final" / key
        row = run.read(directory / "metrics.json")
        params = dict(selected["params"])
        if selected["refit_epochs"] is not None:
            params["epochs"] = selected["refit_epochs"]
        for field, expected in {"family": selected["family"], "method": selected["method"], "params": params,
            "model_sha256": run.digest(directory / "model.joblib"),
            "model_size_bytes": (directory / "model.joblib").stat().st_size,
            "prediction_sha256": run.digest(directory / "predictions.npz"), "prediction_parity": True,
            "training_subjects": sorted(set(data["groups"]) - set(holdout)), "evaluation_subjects": sorted(holdout)}.items():
            equal(row[field], expected, f"Final {field} differs")
        prediction = _prediction(directory / "predictions.npz", data, indices)
        require(np.array_equal(prediction["record_ids"], data["record_ids"][indices]), "Final record IDs differ")
        _metrics(row, prediction)
        finals[key] = (row, prediction)
        paths.extend(directory / name for name in ("model.joblib", "metrics.json", "predictions.npz"))
    comparisons, enabled = [], []
    criteria = protocol["criteria"]
    for key, row in selection["selected"].items():
        if row["method"] == "standard":
            continue
        result = {**row, "improved": False, "heldout_evaluated": False}
        if row["development_qualified"]:
            baseline, bp = finals[f"{row['family']}/standard"]
            candidate, cp = finals[key]
            delta = candidate["metrics"]["macro_f1"] - baseline["metrics"]["macro_f1"]
            class_deltas = {label: candidate["metrics"]["per_class_f1"][label] - baseline["metrics"]["per_class_f1"][label]
                            for label in CLASS_NAMES}
            result.update(heldout_evaluated=True, baseline=baseline["metrics"], candidate=candidate["metrics"],
                heldout_delta=delta, per_class_deltas=class_deltas,
                paired_subject_95_interval=run.paired_interval(bp["truth"], bp["prediction"], cp["prediction"], bp["groups"]),
                baseline_inference_ms=baseline["inference_ms_median"], candidate_inference_ms=candidate["inference_ms_median"],
                improved=bool(delta >= criteria["minimum_holdout_delta"] and min(class_deltas.values()) >= -criteria["maximum_per_class_f1_drop"]))
        comparisons.append(result)
        if result["improved"]:
            enabled.append({"family": row["family"], "method": row["method"]})
    equal(summary["comparisons"], comparisons, "Final comparisons/gates differ from evidence")
    equal(summary["enabled_methods"], enabled, "Enabled methods differ from verified improvement gates")
    return _save_audit(out, "final", paths, {"final_models": len(finals), "metrics_recomputed": True,
        "selection_and_gates_recomputed": True, "enabled_methods": enabled})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "selection", "final"), required=True)
    args = parser.parse_args()
    result = globals()[f"audit_{args.stage}"](args.output)
    print(args.stage.upper(), "AUDIT PASSED", result["details"], flush=True)


if __name__ == "__main__":
    main()
