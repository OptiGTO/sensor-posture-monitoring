"""Preregister, run parallel branches, and confirm only meaningful improvements.

This is retrospective research. The historical holdout is already exposed at
project level; a successful iteration is not prospective sensor validation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

from src.model_comparison import scores
from src.pipeline_recipes import (build_pipeline, fit_pipeline, validate_pipeline,
    input_representation, pipeline_sources, requires_record_ids, combination_family, validate_identity_control)
from src.window_pipeline import aligned_predict_proba

ROOT = Path(os.environ.get("POSTURE_RESEARCH_ROOT", Path(__file__).resolve().parents[2])).resolve()
ORIGIN = ROOT / "research/results/2026-09-21_200011_improvement_retest"
SEEDS = [20260921, 20260922, 20260923]
CRITERIA = {"minimum_development_delta": .01, "minimum_positive_splits": 2,
    "maximum_per_class_drop": .05, "minimum_holdout_delta": .01,
    "bootstrap_lower_must_exceed": 0., "bootstrap_replicates": 10000,
    "all_confirmation_seed_deltas_positive": True, "refit_probability_atol": 1e-12,
    "scope": "retrospective_historical_holdout", "prospective_validation": False}


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def spec(family, params, mode="all", weighting="class_balanced", scaler="standard", post="argmax"):
    return validate_pipeline({"family": family, "base_method": "standard", "window_seconds": 3,
        "feature_engineering": {"mode": mode}, "preprocessing": {"scaler": scaler},
        "model_params": params, "training": {"seed": SEEDS[0], "weighting": weighting},
        "postprocessing": {"kind": post}})


def first_plan():
    trees = dict(n_estimators=200, min_samples_leaf=8, max_depth=8, max_features=.5)
    svm = {"C": 1., "gamma": .0005}
    baseline = spec("extra_trees", trees)
    return baseline, {
        "static_trees": {"ET_static": spec("extra_trees", trees, "static"),
            "ET_means": spec("extra_trees", trees, "means"),
            "ET_relative_static": spec("extra_trees", trees, "relative_static")},
        "subject_balance": {"ET_subject": spec("extra_trees", trees, weighting="subject_class_balanced"),
            "RF_subject": spec("random_forest", {**trees, "max_depth": None}, weighting="subject_class_balanced")},
        "svm_geometry": {"SVM_reference": spec("rbf_svm", svm),
            "SVM_whiten": spec("rbf_svm", {"C": 1., "gamma": "scale"}, scaler="shrinkage_whiten"),
            "SVM_decision": spec("rbf_svm", svm, post="decision_softmax"),
            "SVM_means": spec("rbf_svm", {"C": 1., "gamma": "scale"}, "means")}}


def snapshot_protected():
    result = {}
    for folder in (ROOT / "data", ROOT / "research/models"):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                stat = path.stat()
                result[path.relative_to(ROOT).as_posix()] = {"sha256": digest(path),
                    "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def check_originals(out):
    for rel, expected in read(out / "protected_before.json").items():
        path = ROOT / rel
        stat = path.stat()
        actual = {"sha256": digest(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if actual != expected:
            raise ValueError(f"Protected original changed: {rel}")


def initialize(out, plan_path=None):
    if out.exists():
        raise ValueError("Use a new output directory; immutable iterations cannot be overwritten")
    dataset = read(ORIGIN / "dataset.json")
    if digest(ORIGIN / "data.npz") != dataset["cache_sha256"]:
        raise ValueError("Original frozen cache changed")
    for rel, expected in dataset["input_originals"].items():
        if digest(ROOT / rel) != expected["sha256"]:
            raise ValueError(f"Original data changed since cohort freeze: {rel}")
    baseline, branches = first_plan()
    plan = {}
    if plan_path:
        plan = read(plan_path)
        baseline, branches = validate_pipeline(plan["baseline"]), plan["branches"]
        branches = {b: {k: validate_pipeline(v) for k, v in candidates.items()} for b, candidates in branches.items()}
    ids = [k for branch in branches.values() for k in branch]
    if len(ids) != len(set(ids)) or "baseline" in ids:
        raise ValueError("Candidate IDs must be globally unique and not baseline")
    identity_controls = plan.get("identity_controls", [])
    if (not isinstance(identity_controls, list) or any(not isinstance(name, str) or name not in ids for name in identity_controls)
            or len(identity_controls) != len(set(identity_controls))):
        raise ValueError("Identity controls must name unique preregistered candidates")
    for name in identity_controls:
        recipe = next(branch[name] for branch in branches.values() if name in branch)
        validate_identity_control(baseline, recipe)
    baseline_reference = plan.get("baseline_checkpoint")
    if baseline_reference:
        from src.checkpoint_registry import validate_checkpoint
        checkpoint_path = ROOT / baseline_reference["path"]
        if digest(checkpoint_path) != baseline_reference["sha256"]:
            raise ValueError("Research baseline checkpoint changed")
        checkpoint = read(checkpoint_path)
        validate_checkpoint(checkpoint, root=ROOT)
        if validate_pipeline(checkpoint["pipeline"]) != baseline:
            raise ValueError("Research baseline differs from selected checkpoint recipe")
    criteria = deepcopy(CRITERIA)
    authorization = None
    if plan.get("user_authorization"):
        authorization_path = ROOT / plan["user_authorization"]["path"]
        if digest(authorization_path) != plan["user_authorization"]["sha256"]:
            raise ValueError("User adoption authorization changed")
        authorization = read(authorization_path)
        if (authorization.get("kind") != "explicit_user_research_adoption"
                or ids != [authorization.get("candidate_id")]
                or validate_pipeline(next(iter(next(iter(branches.values())).values()))) != validate_pipeline(authorization.get("pipeline"))):
            raise ValueError("User adoption must name exactly the authorized pipeline")
        previous = ROOT / authorization["source_run"]
        if digest(previous / "selection_lock.json") != authorization["source_selection_sha256"]:
            raise ValueError("Original development selection changed")
        if digest(previous / "protocol.json") != authorization["source_protocol_sha256"]:
            raise ValueError("Original development protocol changed")
        previous_protocol = read(previous / "protocol.json")
        if (baseline != previous_protocol["baseline"] or dataset["partitions"] != previous_protocol["partitions"]
                or digest(ORIGIN / "dataset.json") != previous_protocol["dataset_sha256"]
                or digest(ORIGIN / "data.npz") != previous_protocol["cache_sha256"]):
            raise ValueError("User adoption baseline or frozen evaluation differs from the named experiment")
        source_candidate = next(row for row in read(previous / "selection_lock.json")["candidates"] if row["id"] == ids[0])
        if source_candidate["pipeline"] != authorization["pipeline"] or source_candidate["mean_delta"] != authorization["observed_development_delta"]:
            raise ValueError("User adoption differs from observed candidate")
        # A user-directed near-threshold exception is local to this one recipe.
        # Subsequent autonomous iterations retain the original CRITERIA.
        threshold = authorization["minimum_development_delta"]
        if not 0 < threshold <= source_candidate["mean_delta"] < CRITERIA["minimum_development_delta"]:
            raise ValueError("Invalid near-threshold adoption exception")
        criteria["minimum_development_delta"] = threshold
    out.mkdir(parents=True)
    if authorization is not None:
        write(out / "user_authorization.json", authorization)
    for name in ("dataset.json", "data.npz"):
        (out / name).symlink_to(os.path.relpath(ORIGIN / name, out))
    write(out / "protected_before.json", snapshot_protected())
    supplemental = None
    policies = {recipe["training"].get("data_policy") for branch in branches.values() for recipe in branch.values()}
    policies.add(baseline["training"].get("data_policy"))
    if "include_ordinary_reserved_training_records_v1" in policies:
        recipes = [baseline, *(recipe for branch in branches.values() for recipe in branch.values())]
        if any(input_representation(recipe) == "sensor_sequence" for recipe in recipes):
            from .supplemental_sequence import prepare_supplement_sequences as prepare_supplement
        else:
            from .supplemental import prepare_supplement
        supplement_manifest = prepare_supplement(out, ROOT, ORIGIN)
        supplemental = {"manifest_sha256": digest(out / "supplemental_manifest.json"),
            "cache_sha256": digest(out / "supplemental_training.npz"), "policy": supplement_manifest["policy"],
            "records": supplement_manifest["record_count"], "windows": supplement_manifest["window_count"]}
    source_hashes = {}
    paths = list((ROOT / "src").rglob("*.py")) + list((ROOT / "experiments/iterative_improvement").glob("*.py"))
    paths += [ROOT / "experiments/__init__.py", ROOT / "AGENTS.md"]
    for recipe in [baseline, *(recipe for branch in branches.values() for recipe in branch.values())]:
        paths += [ROOT / relative for relative in pipeline_sources(recipe)]
    paths = sorted(set(paths))
    for path in paths:
        relative = path.relative_to(ROOT)
        dest = out / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        source_hashes[str(relative)] = digest(dest)
    protocol = {"schema_version": 1, "started_at": now(), "baseline": baseline, "branches": branches,
        "criteria": criteria, "seeds": SEEDS, "dataset_sha256": digest(out / "dataset.json"),
        "cache_sha256": digest(out / "data.npz"), "partitions": dataset["partitions"],
        "source_hashes": source_hashes, "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "command": [sys.executable, *sys.argv], "environment": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
        "cohort": dataset["cohort"], "registration_policy": dataset["registration_policy"],
        "supplemental_training": supplemental,
        "previous_iteration": plan.get("previous_iteration"), "hypotheses": plan.get("hypotheses", {}),
        "identity_controls": identity_controls,
        "user_adoption": ({"authorization_sha256": digest(out / "user_authorization.json"),
            "candidate_id": authorization["candidate_id"], "checkpoint_id": authorization["checkpoint_id"], "original_criteria": CRITERIA,
            "scope": "explicit research-baseline adoption; retain all measurements and reproduce; statistical gates remain separately reported"}
            if authorization is not None else None),
        "baseline_checkpoint": baseline_reference,
        "incumbent_basis": ("Explicit user-adopted research checkpoint; original autonomous gates were not met; prior comparison and active models preserved"
            if baseline_reference and checkpoint["status"] == "user_adopted_research" else
            "Validated checkpoint selected by the preceding iteration" if baseline_reference else
            "Best selected final Macro F1 on identical frozen 692-record retest cohort; original advantage was uncertain; never overwrite active model"),
        "selection": "Development mean Macro F1, then worst split, then candidate ID; eligible changes only; lock exactly one finalist before historical holdout",
        "combination": "Equal soft voting uses at most the top three development-qualified compatible single estimators; never nest ensembles; require development gain over the best member and rank against all qualified candidates",
        "exposure": "All subjects and historical fixed evaluation already inspected at project level; no fresh confirmatory population inference",
        "repeated_testing": "Descriptive cluster bootstrap only; adaptive multi-iteration selection is not multiplicity-corrected; freeze all attempted candidates and report failures",
        "raw_data_mutated": False, "physical_sensor_validation": False}
    write(out / "protocol.json", protocol)
    write(out / "protocol_lock.json", {"sha256": digest(out / "protocol.json"), "locked_at": now()})
    print(json.dumps({"output": str(out), "cohort": dataset["cohort"], "branches": {k:len(v) for k,v in branches.items()}}, ensure_ascii=False), flush=True)


def verify(out):
    p = read(out / "protocol.json")
    if digest(out / "protocol.json") != read(out / "protocol_lock.json")["sha256"]:
        raise ValueError("Protocol lock changed")
    if digest(out / "dataset.json") != p["dataset_sha256"] or digest(out / "data.npz") != p["cache_sha256"]:
        raise ValueError("Frozen cohort changed")
    if p.get("user_adoption") and digest(out / "user_authorization.json") != p["user_adoption"]["authorization_sha256"]:
        raise ValueError("User adoption authorization changed")
    if p.get("supplemental_training"):
        extra = p["supplemental_training"]
        if digest(out / "supplemental_manifest.json") != extra["manifest_sha256"] or digest(out / "supplemental_training.npz") != extra["cache_sha256"]:
            raise ValueError("Frozen supplemental training changed")
    for relative, expected in p["source_hashes"].items():
        if digest(out / "source_snapshot" / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
        executing = Path(__file__).resolve().parents[2] / relative
        if digest(executing) != expected:
            raise ValueError(f"Executing source differs from frozen source: {relative}")
    return p


def load_data(out):
    return dict(np.load(out / "data.npz", allow_pickle=False))


def model_input(data, pipeline, rows):
    """Route identical frozen rows to the recipe's declared input contract."""
    key = "seq" if input_representation(pipeline) == "sensor_sequence" else "X"
    values = data[key]
    if len(values) != len(data["y"]):
        raise ValueError("Model input rows differ from frozen labels")
    return values[rows]


def indices(data, part, final=False):
    memberships = [part[name] for name in ("train", "validation", "test")]
    if any(len(group) != len(set(group)) for group in memberships):
        raise ValueError("Duplicate subject membership")
    if any(set(memberships[i]) & set(memberships[j]) for i in range(3) for j in range(i)):
        raise ValueError("Subject leakage across train/validation/test")
    if set().union(*map(set, memberships)) != set(data["groups"]):
        raise ValueError("Partitions must cover exactly the frozen cohort")
    train_subjects = part["train"] + part["validation"] if final else part["train"]
    eval_subjects = part["test"] if final else part["validation"]
    if set(train_subjects) & set(eval_subjects):
        raise ValueError("Subject leakage")
    train = np.flatnonzero(np.isin(data["groups"], train_subjects))
    valid = np.flatnonzero(np.isin(data["groups"], eval_subjects))
    if set(data["record_ids"][train]) & set(data["record_ids"][valid]):
        raise ValueError("Recording leakage")
    for idx in (train, valid):
        if set(data["y"][idx]) != set(range(7)):
            raise ValueError("Missing class")
    return train, valid


def verify_fit(out, destination, pipeline=None, seed=None):
    row = read(destination / "metrics.json")
    if digest(destination / "metrics.json") != read(destination / "metrics_seal.json")["sha256"]:
        raise ValueError("Fit metrics changed")
    if row["protocol_sha256"] != digest(out / "protocol.json"):
        raise ValueError("Fit protocol mismatch")
    if pipeline is not None and row["pipeline"] != pipeline or seed is not None and row["seed"] != seed:
        raise ValueError("Fit recipe or seed mismatch")
    if digest(destination / "predictions.npz") != row["predictions_sha256"] or digest(destination / "model.joblib") != row["model_sha256"]:
        raise ValueError("Fit predictions or model changed")
    values = dict(np.load(destination / "predictions.npz", allow_pickle=False))
    if not np.array_equal(values["prediction"], values["probability"].argmax(1)) or scores(values["truth"], values["prediction"]) != row["metrics"]:
        raise ValueError("Saved metrics do not reproduce from predictions")
    return row


def fit_one(out, data, pipeline, train, valid, destination, seed, *, overwrite=False):
    destination.mkdir(parents=True, exist_ok=True)
    metrics_file = destination / "metrics.json"
    if metrics_file.exists() and not overwrite:
        return verify_fit(out, destination, pipeline, seed)
    started, tick = now(), time.perf_counter()
    model = build_pipeline(pipeline, seed=seed)
    train_X, train_y, train_groups = model_input(data, pipeline, train), data["y"][train], data["groups"][train]
    train_records = data["record_ids"][train] if requires_record_ids(pipeline) else None
    training_audit = None
    if pipeline["training"].get("data_policy") == "include_ordinary_reserved_training_records_v1":
        supplement = dict(np.load(out / "supplemental_training.npz", allow_pickle=False))
        if requires_record_ids(pipeline) and input_representation(pipeline) == "sensor_sequence":
            from .record_sequence_training import append_record_sequence_training
            train_X, train_y, train_groups, train_records, training_audit = append_record_sequence_training(data, supplement, train)
        elif requires_record_ids(pipeline):
            from .record_training import append_record_training
            train_X, train_y, train_groups, train_records, training_audit = append_record_training(data, supplement, train)
        else:
            if input_representation(pipeline) == "sensor_sequence":
                from .supplemental_sequence import append_sequence_training as append_training
            else:
                from .supplemental import append_training
            train_X, train_y, train_groups, training_audit = append_training(data, supplement, train)
    fit_kwargs = {"groups": train_groups}
    if train_records is not None:
        fit_kwargs["record_ids"] = train_records
    fit_pipeline(model, pipeline, train_X, train_y, **fit_kwargs)
    elapsed = time.perf_counter() - tick
    predict_tick = time.perf_counter()
    probability = aligned_predict_proba(model, model_input(data, pipeline, valid), 7)
    predict_seconds = time.perf_counter() - predict_tick
    if probability.shape != (len(valid), 7) or not np.isfinite(probability).all() or np.any(probability < 0) or not np.allclose(probability.sum(1), 1):
        raise ValueError("Invalid probability contract")
    prediction = probability.argmax(1)
    joblib.dump(model, destination / "model.joblib", compress=3)
    np.savez_compressed(destination / "predictions.npz", indices=valid, truth=data["y"][valid],
        groups=data["groups"][valid], record_ids=data["record_ids"][valid], probability=probability, prediction=prediction)
    row = {"pipeline": pipeline, "seed": seed, "input_representation": input_representation(pipeline), "started_at": started, "finished_at": now(),
        "training_seconds": elapsed, "batch_inference_ms_per_window": predict_seconds * 1000 / len(valid),
        "metrics": scores(data["y"][valid], prediction),
        "per_subject": {str(g): scores(data["y"][valid][data["groups"][valid] == g], prediction[data["groups"][valid] == g]) for g in sorted(set(data["groups"][valid]))},
        "training_subjects": sorted(set(data["groups"][train].tolist())), "evaluation_subjects": sorted(set(data["groups"][valid].tolist())),
        "model_sha256": digest(destination / "model.joblib"), "predictions_sha256": digest(destination / "predictions.npz"),
        "protocol_sha256": digest(out / "protocol.json"), "training_details": getattr(model, "training_details_", {})}
    if training_audit is not None:
        row["supplemental_training_audit"] = training_audit
    write(metrics_file, row)
    write(destination / "metrics_seal.json", {"sha256": digest(metrics_file)})
    return row


def development(out, branch):
    p, data = verify(out), load_data(out)
    candidates = {"baseline": p["baseline"]} if branch == "baseline" else p["branches"][branch]
    for name, pipeline in candidates.items():
        for part in p["partitions"]:
            train, valid = indices(data, part)
            row = fit_one(out, data, pipeline, train, valid, out / "development" / name / part["id"], p["seeds"][0])
            print(f"{name}/{part['id']} F1={row['metrics']['macro_f1']:.6f}", flush=True)


def dispatch(out):
    p = verify(out)
    def one(branch):
        log = out / "logs" / f"{branch}.log"
        log.parent.mkdir(exist_ok=True)
        with log.open("a") as stream:
            done = subprocess.run([sys.executable, "-m", "experiments.iterative_improvement.run", "--output", str(out), "--phase", "worker", "--branch", branch],
                cwd=out / "source_snapshot", env={**os.environ, "POSTURE_RESEARCH_ROOT": str(ROOT)}, stdout=stream, stderr=subprocess.STDOUT)
        if done.returncode:
            raise RuntimeError(f"Worker {branch} failed: {log}")
        print(f"COMPLETED {branch}", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(one, name) for name in ["baseline", *p["branches"]]]
        for future in as_completed(futures):
            future.result()


def development_summary(out, name, baseline="baseline"):
    p = verify(out)
    recipes = {"baseline": p["baseline"], **{k:v for branch in p["branches"].values() for k,v in branch.items()}}
    if name == "combined":
        recipes[name] = read(out / "combination_lock.json")["pipeline"]
    rows = [verify_fit(out, out / "development" / name / part["id"], recipes[name], p["seeds"][0]) for part in p["partitions"]]
    reference = [verify_fit(out, out / "development" / baseline / part["id"], recipes[baseline], p["seeds"][0]) for part in p["partitions"]]
    f1 = np.array([r["metrics"]["macro_f1"] for r in rows])
    delta = f1 - [r["metrics"]["macro_f1"] for r in reference]
    class_delta = np.array([list(r["metrics"]["per_class_f1"].values()) for r in rows]) - np.array([list(r["metrics"]["per_class_f1"].values()) for r in reference])
    qualified = delta.mean() >= p["criteria"]["minimum_development_delta"] and (delta > 0).sum() >= p["criteria"]["minimum_positive_splits"] and class_delta.mean(0).min() >= -p["criteria"]["maximum_per_class_drop"]
    return {"id": name, "baseline": baseline, "pipeline": rows[0]["pipeline"], "f1": f1.tolist(),
        "mean_f1": float(f1.mean()), "worst_split": float(f1.min()), "deltas": delta.tolist(),
        "mean_delta": float(delta.mean()), "mean_class_deltas": class_delta.mean(0).tolist(), "qualified": bool(qualified)}


def verify_identity_controls(out, protocol):
    """A no-op representation path must reproduce baseline before selection."""
    controls = protocol.get("identity_controls", [])
    if not controls:
        return
    recipes = {name: recipe for branch in protocol["branches"].values() for name, recipe in branch.items()}
    rows = []
    for name in controls:
        validate_identity_control(protocol["baseline"], recipes[name])
        for part in protocol["partitions"]:
            reference = out / "development/baseline" / part["id"]
            candidate = out / "development" / name / part["id"]
            base_metrics = verify_fit(out, reference, protocol["baseline"], protocol["seeds"][0])
            new_metrics = verify_fit(out, candidate, recipes[name], protocol["seeds"][0])
            with np.load(reference / "predictions.npz", allow_pickle=False) as base, np.load(candidate / "predictions.npz", allow_pickle=False) as new:
                equal = set(base.files) == set(new.files) and all(base[key].dtype == new[key].dtype
                    and base[key].shape == new[key].shape and base[key].tobytes(order="C") == new[key].tobytes(order="C") for key in base.files)
            if not equal or base_metrics["metrics"] != new_metrics["metrics"]:
                raise ValueError(f"Identity control does not reproduce baseline: {name}/{part['id']}")
            rows.append({"candidate": name, "partition": part["id"], "all_prediction_arrays_bitwise_equal": True,
                "baseline_predictions_sha256": digest(reference / "predictions.npz"),
                "control_predictions_sha256": digest(candidate / "predictions.npz")})
    evidence = {"protocol_sha256": digest(out / "protocol.json"), "seed": protocol["seeds"][0], "rows": rows}
    path = out / "audits/identity_controls.json"
    if path.exists() and read(path) != evidence:
        raise ValueError("Existing identity control evidence changed")
    if not path.exists():
        write(path, evidence)


def select(out):
    p = verify(out)
    if (out / "selection_lock.json").exists():
        raise ValueError("Selection already locked")
    verify_identity_controls(out, p)
    names = [name for branch in p["branches"].values() for name in branch]
    rows = [development_summary(out, name) for name in names]
    accepted = sorted([r for r in rows if r["qualified"]], key=lambda r: (-r["mean_f1"], -r["worst_split"], r["id"]))
    # Additional combinations use individually qualified single estimators.
    # An existing ensemble candidate cannot be nested.
    single = [r for r in accepted if r["pipeline"]["family"] not in {"equal_soft_voting", "sequence_soft_voting", "record_soft_voting", "spatial_soft_voting", "gain_soft_voting", "local_soft_voting", "interaction_soft_voting", "support_soft_voting", "alternatives_soft_voting", "projection_soft_voting", "hardness_soft_voting", "structured_soft_voting"}]
    if len(single) >= 2:
        policy = single[0]["pipeline"]["training"].get("data_policy")
        representation = input_representation(single[0]["pipeline"])
        combo_family = combination_family(single[0]["pipeline"])
        compatible = [r for r in single if r["pipeline"]["training"].get("data_policy") == policy
            and input_representation(r["pipeline"]) == representation
            and combination_family(r["pipeline"]) == combo_family][:3]
    else:
        compatible = []
    if len(compatible) >= 2:
        combo_training = {"seed": p["seeds"][0]}
        if policy is not None:
            combo_training["data_policy"] = policy
        combo = validate_pipeline({"family": combo_family,
            "members": [r["pipeline"] for r in compatible], "training": combo_training})
        write(out / "combination_lock.json", {"pipeline": combo, "members": [r["id"] for r in compatible], "locked_at": now()})
        data = load_data(out)
        for part in p["partitions"]:
            train, valid = indices(data, part)
            fit_one(out, data, combo, train, valid, out / "development/combined" / part["id"], p["seeds"][0])
        row = development_summary(out, "combined")
        versus_member = development_summary(out, "combined", compatible[0]["id"])
        row["versus_best_member"] = versus_member
        row["qualified"] = row["qualified"] and versus_member["qualified"]
        rows.append(row)
        if row["qualified"]:
            accepted.append(row)
            accepted.sort(key=lambda r: (-r["mean_f1"], -r["worst_split"], r["id"]))
    selection = {"locked_at": now(), "protocol_sha256": digest(out / "protocol.json"), "historical_holdout_opened_this_iteration": False,
        "candidates": rows, "selected": accepted[0] if accepted else None,
        "reason": "Highest development-qualified candidate" if accepted else "No meaningful development gain; preserve incumbent; holdout remains unopened this iteration"}
    write(out / "selection_lock.json", selection)
    write(out / "selection_seal.json", {"sha256": digest(out / "selection_lock.json"), "sealed_at": now()})
    print(json.dumps({"selected": selection["selected"]["id"] if accepted else None, "candidates": [{k:r[k] for k in ("id","mean_f1","mean_delta","qualified")} for r in rows]}, ensure_ascii=False), flush=True)


def cluster_bootstrap(y, base, candidate, groups, repeats=10000):
    subjects = sorted(set(groups))
    a = np.stack([confusion_matrix(y[groups == g], base[groups == g], labels=np.arange(7)) for g in subjects])
    b = np.stack([confusion_matrix(y[groups == g], candidate[groups == g], labels=np.arange(7)) for g in subjects])
    draw = np.random.default_rng(20260922).integers(0, len(subjects), (repeats, len(subjects)))
    def f1(c):
        c = c[draw].sum(1)
        den = c.sum(1) + c.sum(2)
        return np.divide(2 * np.diagonal(c, axis1=1, axis2=2), den, out=np.zeros_like(den, dtype=float), where=den != 0).mean(1)
    values = f1(b) - f1(a)
    return {"lower": float(np.quantile(values, .025)), "upper": float(np.quantile(values, .975)), "replicates": repeats,
        "unit": "subject", "interpretation": "Descriptive conditional interval; historical/adaptive comparisons are not independent confirmatory inference"}


def confirm(out):
    p, data = verify(out), load_data(out)
    lock = read(out / "selection_lock.json")
    if digest(out / "selection_lock.json") != read(out / "selection_seal.json")["sha256"] or lock["protocol_sha256"] != digest(out / "protocol.json"):
        raise ValueError("Selection seal or protocol mismatch")
    if (out / "evaluation.json").exists():
        raise ValueError("Evaluation already sealed")
    if p.get("identity_controls"):
        if not (out / "audits/identity_controls.json").is_file():
            raise ValueError("Preregistered identity control evidence is missing")
        verify_identity_controls(out, p)
    if lock["selected"] is None:
        check_originals(out)
        write(out / "evaluation.json", {"finished_at": now(), "improved": False, "meaningful_gain": False,
            "reason": lock["reason"], "holdout_evaluated": False, "baseline_preserved": True})
        write(out / "evaluation_seal.json", {"sha256": digest(out / "evaluation.json")})
        return
    selected = lock["selected"]
    recomputed = development_summary(out, selected["id"])
    if not recomputed["qualified"] or any(recomputed[k] != selected[k] for k in ("pipeline", "f1", "mean_delta")):
        raise ValueError("Selected candidate no longer matches qualified development evidence")
    # Validate random-seed robustness on development before opening the fixed evaluation.
    robust = []
    for seed in p["seeds"][1:]:
        deltas = []
        for part in p["partitions"]:
            train, valid = indices(data, part)
            ref = fit_one(out, data, p["baseline"], train, valid, out / "robustness" / str(seed) / "baseline" / part["id"], seed)
            row = fit_one(out, data, selected["pipeline"], train, valid, out / "robustness" / str(seed) / selected["id"] / part["id"], seed)
            deltas.append(row["metrics"]["macro_f1"] - ref["metrics"]["macro_f1"])
        robust.append({"seed": seed, "deltas": deltas, "mean_delta": float(np.mean(deltas))})
    if any(r["mean_delta"] <= 0 for r in robust) and not p.get("user_adoption"):
        check_originals(out)
        write(out / "evaluation.json", {"finished_at": now(), "improved": False, "meaningful_gain": False,
            "reason": "Development gain did not reproduce across seeds", "robustness": robust, "holdout_evaluated": False, "baseline_preserved": True})
        write(out / "evaluation_seal.json", {"sha256": digest(out / "evaluation.json")})
        return
    train, valid = indices(data, p["partitions"][0], final=True)
    final = []
    for seed in p["seeds"]:
        ref = fit_one(out, data, p["baseline"], train, valid, out / "final" / str(seed) / "baseline", seed)
        row = fit_one(out, data, selected["pipeline"], train, valid, out / "final" / str(seed) / selected["id"], seed)
        final.append({"seed": seed, "baseline": ref["metrics"], "candidate": row["metrics"],
            "delta": row["metrics"]["macro_f1"] - ref["metrics"]["macro_f1"]})
    first = out / "final" / str(p["seeds"][0])
    baseline_npz = dict(np.load(first / "baseline/predictions.npz", allow_pickle=False))
    candidate_npz = dict(np.load(first / selected["id"] / "predictions.npz", allow_pickle=False))
    interval = cluster_bootstrap(data["y"][valid], baseline_npz["prediction"], candidate_npz["prediction"], data["groups"][valid], p["criteria"]["bootstrap_replicates"])
    repeat = out / "reproduction" / selected["id"]
    fit_one(out, data, selected["pipeline"], train, valid, repeat, p["seeds"][0])
    repro = dict(np.load(repeat / "predictions.npz", allow_pickle=False))
    max_error = float(np.max(np.abs(repro["probability"] - candidate_npz["probability"])))
    reloaded = joblib.load(first / selected["id"] / "model.joblib")
    load_error = float(np.max(np.abs(aligned_predict_proba(reloaded, model_input(data, selected["pipeline"], valid), 7) - candidate_npz["probability"])))
    class_delta = np.array(list(final[0]["candidate"]["per_class_f1"].values())) - np.array(list(final[0]["baseline"]["per_class_f1"].values()))
    reproduced = max_error <= p["criteria"]["refit_probability_atol"] and load_error <= p["criteria"]["refit_probability_atol"]
    passed = all(r["mean_delta"] > 0 for r in robust) and final[0]["delta"] >= p["criteria"]["minimum_holdout_delta"] and min(r["delta"] for r in final) > 0 and interval["lower"] > 0 and class_delta.min() >= -p["criteria"]["maximum_per_class_drop"] and reproduced
    write(out / "reproducibility.json", {"reproduced": reproduced, "max_probability_error": max_error, "reload_max_error": load_error,
        "seed_robustness": robust, "finished_at": now(), "protocol_sha256": digest(out / "protocol.json"),
        "selected_id": selected["id"], "selected_model_sha256": digest(first / selected["id"] / "model.joblib"),
        "selected_predictions_sha256": digest(first / selected["id"] / "predictions.npz"),
        "final_metrics_sha256": digest(first / selected["id"] / "metrics.json"),
        "selection_sha256": digest(out / "selection_lock.json")})
    check_originals(out)
    write(out / "evaluation.json", {"finished_at": now(), "improved": bool(passed), "meaningful_gain": bool(passed), "reproduced": reproduced,
        "selected_id": selected["id"], "holdout_evaluated": True, "final": final, "paired_subject_interval": interval,
        "class_deltas": class_delta.tolist(), "development_seed_robustness": robust, "baseline_preserved": True,
        "scope": "retrospective_historical_holdout", "selection_sha256": digest(out / "selection_lock.json"),
        "user_adoption": p.get("user_adoption"),
        "original_autonomous_criteria_met": bool(passed and recomputed["mean_delta"] >= CRITERIA["minimum_development_delta"])})
    write(out / "evaluation_seal.json", {"sha256": digest(out / "evaluation.json")})
    print(json.dumps({"improved": bool(passed), "delta": final[0]["delta"], "interval": interval, "reproduced": reproduced}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["init", "run", "worker", "select", "confirm", "verify"], required=True)
    parser.add_argument("--branch")
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.phase != "init":
        frozen_script = out / "source_snapshot/experiments/iterative_improvement/run.py"
        if Path(__file__).resolve() != frozen_script:
            os.chdir(out / "source_snapshot")
            os.execve(sys.executable, [sys.executable, "-m", "experiments.iterative_improvement.run", "--output", str(out),
                "--phase", args.phase, *(["--branch", args.branch] if args.branch else [])],
                {**os.environ, "POSTURE_RESEARCH_ROOT": str(ROOT)})
    if args.phase == "init": initialize(out, args.plan)
    elif args.phase == "run": dispatch(out)
    elif args.phase == "worker": development(out, args.branch)
    elif args.phase == "select": select(out)
    elif args.phase == "confirm": confirm(out)
    else:
        verify(out)
        check_originals(out)
        print("Frozen sources, cohort and protected originals verified")


if __name__ == "__main__":
    main()
