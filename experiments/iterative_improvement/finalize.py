"""Publish a winning, reproducible recipe without moving or copying model bytes.

All checks precede creation of the canonical checkpoint directory. A temporary
descriptor is validated against the original model before the final symlink and
append-only registry entry are created. Non-winners are a successful no-op.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import joblib
import numpy as np

from src.checkpoint_registry import PIPELINE_SOURCES, USER_ADOPTION_STATUS, register_checkpoint, validate_checkpoint
from src.pipeline_recipes import validate_pipeline, pipeline_sources, input_representation, requires_record_ids, is_spatial_recipe, is_gain_recipe, is_local_recipe, is_interaction_recipe, is_support_recipe, is_alternatives_recipe, is_projection_recipe, is_hardness_recipe, is_structured_recipe
from src.model_comparison import scores
from src.window_pipeline import aligned_predict_proba

ROOT = Path(__file__).resolve().parents[2]
CORE_SOURCES = tuple(PIPELINE_SOURCES)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def ref(path, root):
    path = Path(path).absolute()
    path.resolve().relative_to(root.resolve())
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


def check_seal(path, seal, required=True):
    if not seal.is_file():
        if required:
            raise ValueError(f"Missing evidence seal: {seal}")
        return
    if read(seal).get("sha256") != digest(path):
        raise ValueError(f"Evidence seal differs: {path}")


def fit_evidence(out, folder, pipeline, seed, data, indices):
    row = read(folder / "metrics.json")
    check_seal(folder / "metrics.json", folder / "metrics_seal.json", required=False)
    if (row["protocol_sha256"] != digest(out / "protocol.json") or row["seed"] != seed
            or validate_pipeline(row["pipeline"]) != validate_pipeline(pipeline)):
        raise ValueError(f"Fit identity mismatch: {folder}")
    for name, key in (("model.joblib", "model_sha256"), ("predictions.npz", "predictions_sha256")):
        if digest(folder / name) != row[key]:
            raise ValueError(f"Fit artifact changed: {folder / name}")
    values = dict(np.load(folder / "predictions.npz", allow_pickle=False))
    if (not np.array_equal(values["indices"], indices)
            or not np.array_equal(values["truth"], data["y"][indices])
            or not np.array_equal(values["groups"], data["groups"][indices])
            or not np.array_equal(values["record_ids"], data["record_ids"][indices])):
        raise ValueError(f"Prediction identities differ: {folder}")
    probability = values["probability"]
    if (probability.shape != (len(indices), 7) or not np.isfinite(probability).all()
            or (probability < 0).any() or not np.allclose(probability.sum(1), 1)
            or not np.array_equal(values["prediction"], probability.argmax(1))
            or row["metrics"] != scores(values["truth"], values["prediction"])):
        raise ValueError(f"Prediction metrics do not reproduce: {folder}")
    return row, values


def _verify_fitted_identity(model, pipeline, seed):
    recipe = validate_pipeline(pipeline)
    if validate_pipeline(getattr(model, "spec_", {})) != recipe or getattr(model, "effective_seed_", None) != seed:
        raise ValueError("Serialized model recipe/seed differs from the selected checkpoint")
    if is_structured_recipe(recipe):
        from src.structured_forest_pipeline_v1 import verify_fitted_structured_identity
        verify_fitted_structured_identity(model, recipe, seed)
        return
    if is_hardness_recipe(recipe):
        from src.hardness_pipeline_v1 import verify_fitted_hardness_identity
        verify_fitted_hardness_identity(model, recipe, seed)
        return
    if is_projection_recipe(recipe):
        from src.learned_projection_pipeline_v1 import verify_fitted_projection_identity
        verify_fitted_projection_identity(model, recipe, seed)
        return
    if is_alternatives_recipe(recipe):
        from src.forest_alternatives_pipeline_v1 import verify_fitted_alternatives_identity
        verify_fitted_alternatives_identity(model, recipe, seed)
    elif is_support_recipe(recipe):
        from src.forest_support_pipeline_v1 import verify_fitted_support_identity
        verify_fitted_support_identity(model, recipe, seed)
    elif is_interaction_recipe(recipe):
        from src.interaction_pipeline_v1 import verify_fitted_interaction_identity
        verify_fitted_interaction_identity(model, recipe, seed)
    elif is_gain_recipe(recipe):
        from src.augmentation_pipeline_v1 import verify_fitted_augmentation_identity
        verify_fitted_augmentation_identity(model, recipe, seed)
    elif is_spatial_recipe(recipe):
        from src.spatial_pipeline_v1 import verify_fitted_spatial_identity
        verify_fitted_spatial_identity(model, recipe, seed)
    elif is_local_recipe(recipe):
        from src.local_geometry_pipeline_v1 import verify_fitted_local_identity
        verify_fitted_local_identity(model, recipe, seed)
    elif requires_record_ids(recipe):
        from src.record_pipeline_v1 import verify_fitted_record_identity
        verify_fitted_record_identity(model, recipe, seed)
    elif recipe["family"] in {"equal_soft_voting", "sequence_soft_voting"}:
        members = getattr(model, "members_", ())
        if len(members) != len(recipe["members"]):
            raise ValueError("Serialized voting member count differs from the selected checkpoint")
        for member, member_recipe in zip(members, recipe["members"]):
            _verify_fitted_identity(member, member_recipe, seed)
    elif recipe["family"] in {"cnn1d", "minirocket"}:
        from src.sequence_pipeline_v1 import SequenceIterativePipeline
        from src.extended_models import MiniRocketClassifier, SmallNeuralClassifier
        expected = (SmallNeuralClassifier(architecture="cnn1d", random_state=seed, **recipe["model_params"])
            if recipe["family"] == "cnn1d" else MiniRocketClassifier(random_state=seed, **recipe["model_params"]))
        fitted = getattr(model, "estimator_", None)
        if (not isinstance(model, SequenceIterativePipeline) or type(fitted) is not type(expected)
                or fitted.get_params(deep=False) != expected.get_params(deep=False)
                or not np.array_equal(getattr(fitted, "classes_", ()), np.arange(7))):
            raise ValueError("Serialized sequence estimator parameters, seed or classes differ")
    elif recipe["postprocessing"]["kind"] == "macro_f1_bias":
        base_recipe = deepcopy(recipe)
        base_recipe["postprocessing"] = {"kind": "argmax"}
        _verify_fitted_identity(getattr(model, "estimator_", None), base_recipe, seed)
    elif recipe["family"] == "sagittal_hierarchy":
        from src.sagittal_hierarchy import SagittalHierarchyClassifier, GATE_CLASSES, SPECIALIST_CLASSES
        expected = SagittalHierarchyClassifier(family=recipe["model_params"]["base_family"],
            model_params=recipe["model_params"]["base_params"], random_state=seed)
        fitted = model.named_steps["model"]
        if not isinstance(fitted, SagittalHierarchyClassifier) or fitted.get_params(deep=False) != expected.get_params(deep=False):
            raise ValueError("Serialized hierarchy recipe differs from the selected checkpoint")
        params = expected._configuration()
        for attribute, head_seed, classes in (("gate_", seed, GATE_CLASSES),
                ("specialist_", (seed + 1) % 2**32, SPECIALIST_CLASSES)):
            head = getattr(fitted, attribute, None)
            expected_head = expected._estimator(params, head_seed)
            if (type(head) is not type(expected_head)
                    or head.get_params(deep=False) != expected_head.get_params(deep=False)
                    or not np.array_equal(np.sort(getattr(head, "classes_", ())), classes)):
                raise ValueError("Serialized hierarchy head parameters, seed or classes differ")


def verified_model_probabilities(model, pipeline, seed, features):
    """Check fitted recipe identity before replaying a serialized checkpoint."""
    _verify_fitted_identity(model, pipeline, seed)
    probability = aligned_predict_proba(model, features, 7)
    if (probability.shape != (len(features), 7) or not np.isfinite(probability).all()
            or (probability < 0).any() or not np.allclose(probability.sum(1), 1)):
        raise ValueError("Serialized model replay produced invalid probabilities")
    return probability


def _verify_selected(out, root, *, user_adopted=False):
    # The current evaluator is used only for numerical checks; all training
    # algorithm files below must exactly match the frozen execution sources.
    from experiments.iterative_improvement.run import indices, cluster_bootstrap, model_input, verify_identity_controls
    p, selection, evaluation = (read(out / name) for name in ("protocol.json", "selection_lock.json", "evaluation.json"))
    for name, seal in (("protocol.json", "protocol_lock.json"), ("selection_lock.json", "selection_seal.json")):
        check_seal(out / name, out / seal)
    check_seal(out / "evaluation.json", out / "evaluation_seal.json", required=False)
    if selection["protocol_sha256"] != digest(out / "protocol.json") or evaluation["selection_sha256"] != digest(out / "selection_lock.json"):
        raise ValueError("Evaluation/selection/protocol linkage differs")
    for name, key in (("dataset.json", "dataset_sha256"), ("data.npz", "cache_sha256")):
        if digest(out / name) != p[key]:
            raise ValueError(f"Frozen input changed: {name}")
    if p.get("supplemental_training"):
        for name, key in (("supplemental_manifest.json", "manifest_sha256"), ("supplemental_training.npz", "cache_sha256")):
            if digest(out / name) != p["supplemental_training"][key]:
                raise ValueError(f"Frozen supplemental input changed: {name}")
    for relative, expected in p["source_hashes"].items():
        if digest(out / "source_snapshot" / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    if p.get("identity_controls"):
        if not (out / "audits/identity_controls.json").is_file():
            raise ValueError("Preregistered identity control evidence is missing")
        verify_identity_controls(out, p)
    selected = selection["selected"]
    if not selected:
        raise ValueError("No locked development-qualified final candidate")
    recipe_sources = pipeline_sources(selected["pipeline"])
    for relative in recipe_sources:
        if relative not in p["source_hashes"] or digest(root / relative) != p["source_hashes"][relative]:
            raise ValueError(f"Reusable algorithm source differs from validated source: {relative}")
    for relative, expected in read(out / "protected_before.json").items():
        path = root / relative
        stat = path.stat()
        if {"sha256": digest(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
            raise ValueError(f"Protected original changed: {relative}")
    selected = selection["selected"]
    if not selected or not selected["qualified"] or evaluation.get("selected_id") != selected["id"] or evaluation.get("holdout_evaluated") is not True:
        raise ValueError("No locked development-qualified final candidate")
    pipeline, primary = validate_pipeline(selected["pipeline"]), p["seeds"][0]
    criteria = p["criteria"]
    data = dict(np.load(out / "data.npz", allow_pickle=False))
    development_deltas, class_deltas = [], []
    for part in p["partitions"]:
        _, valid = indices(data, part)
        base, _ = fit_evidence(out, out / "development/baseline" / part["id"], p["baseline"], primary, data, valid)
        candidate, _ = fit_evidence(out, out / "development" / selected["id"] / part["id"], pipeline, primary, data, valid)
        development_deltas.append(candidate["metrics"]["macro_f1"] - base["metrics"]["macro_f1"])
        class_deltas.append([candidate["metrics"]["per_class_f1"][key] - value for key, value in base["metrics"]["per_class_f1"].items()])
    if (np.mean(development_deltas) < criteria["minimum_development_delta"]
            or sum(value > 0 for value in development_deltas) < criteria["minimum_positive_splits"]
            or np.asarray(class_deltas).mean(0).min() < -criteria["maximum_per_class_drop"]
            or not np.allclose(development_deltas, selected["deltas"], atol=1e-12, rtol=0)
            or abs(float(np.mean(development_deltas)) - selected["mean_delta"]) > 1e-12
            or not np.allclose(np.asarray(class_deltas).mean(0), selected["mean_class_deltas"], atol=1e-12, rtol=0)):
        raise ValueError("Development improvement does not satisfy the locked gates")
    observed_robustness = []
    for seed in p["seeds"][1:]:
        deltas = []
        for part in p["partitions"]:
            _, valid = indices(data, part)
            base, _ = fit_evidence(out, out / "robustness" / str(seed) / "baseline" / part["id"], p["baseline"], seed, data, valid)
            candidate, _ = fit_evidence(out, out / "robustness" / str(seed) / selected["id"] / part["id"], pipeline, seed, data, valid)
            deltas.append(candidate["metrics"]["macro_f1"] - base["metrics"]["macro_f1"])
        if not user_adopted and np.mean(deltas) <= 0:
            raise ValueError("Development improvement did not reproduce across seeds")
        observed_robustness.append({"seed": seed, "deltas": deltas, "mean_delta": float(np.mean(deltas))})
    _, valid = indices(data, p["partitions"][0], final=True)
    primary_base = primary_candidate = None
    for number, seed in enumerate(p["seeds"]):
        folder = out / "final" / str(seed)
        base, base_values = fit_evidence(out, folder / "baseline", p["baseline"], seed, data, valid)
        candidate, values = fit_evidence(out, folder / selected["id"], pipeline, seed, data, valid)
        observed = candidate["metrics"]["macro_f1"] - base["metrics"]["macro_f1"]
        expected = evaluation["final"][number]
        if (expected["seed"] != seed or expected["baseline"] != base["metrics"] or expected["candidate"] != candidate["metrics"]
                or abs(expected["delta"] - observed) > 1e-12
                or not user_adopted and (observed <= 0 or number == 0 and observed < criteria["minimum_holdout_delta"])):
            raise ValueError("Final improvement does not satisfy the locked gates")
        if number == 0:
            primary_base, primary_candidate = base_values, values
            if not user_adopted and min(candidate["metrics"]["per_class_f1"][key] - value for key, value in base["metrics"]["per_class_f1"].items()) < -criteria["maximum_per_class_drop"]:
                raise ValueError("Final class performance regressed beyond the limit")
    interval = cluster_bootstrap(data["y"][valid], primary_base["prediction"], primary_candidate["prediction"], data["groups"][valid], criteria["bootstrap_replicates"])
    recorded = evaluation["paired_subject_interval"]
    if (not user_adopted and interval["lower"] <= criteria["bootstrap_lower_must_exceed"]
            or any(abs(interval[key] - recorded[key]) > 1e-12 for key in ("lower", "upper"))):
        raise ValueError("Paired-subject uncertainty gate failed")
    original_reproduction = read(out / "reproducibility.json")
    if (original_reproduction.get("seed_robustness") != observed_robustness
            or evaluation.get("development_seed_robustness") != observed_robustness):
        raise ValueError("Recorded seed robustness differs from saved development predictions")
    if original_reproduction.get("reproduced") is not True:
        raise ValueError("Original reproduction did not pass")
    repeat_folder = out / "reproduction" / selected["id"]
    _, repeated = fit_evidence(out, repeat_folder, pipeline, primary, data, valid)
    original_model = out / "final" / str(primary) / selected["id"] / "model.joblib"
    max_error = float(np.max(np.abs(repeated["probability"] - primary_candidate["probability"])))
    loaded = joblib.load(original_model)
    reloaded_error = float(np.max(np.abs(verified_model_probabilities(loaded, pipeline, primary, model_input(data, pipeline, valid)) - primary_candidate["probability"])))
    repeat_model = joblib.load(repeat_folder / "model.joblib")
    repeat_reload_error = float(np.max(np.abs(verified_model_probabilities(repeat_model, pipeline, primary, model_input(data, pipeline, valid)) - repeated["probability"])))
    if any(value > criteria["refit_probability_atol"] for value in (max_error, reloaded_error, repeat_reload_error)):
        raise ValueError("Independent saved-refit/reload verification failed")
    evidence = {**original_reproduction, "reproduced": True, "independently_verified_at": datetime.now().astimezone().isoformat(),
        "max_probability_error": max_error, "reload_max_error": reloaded_error, "refit_reload_max_error": repeat_reload_error,
        "protocol_sha256": digest(out / "protocol.json"), "selection_sha256": digest(out / "selection_lock.json"),
        "evaluation_sha256": digest(out / "evaluation.json"), "selected_id": selected["id"],
        "selected_model_sha256": digest(original_model),
        "selected_predictions_sha256": digest(original_model.parent / "predictions.npz"),
        "final_metrics_sha256": digest(original_model.parent / "metrics.json"),
        "original_reproducibility_sha256": digest(out / "reproducibility.json"),
        "source_hashes": {relative: p["source_hashes"][relative] for relative in recipe_sources},
        "verification": ("Recomputed reported metrics and uncertainty, retaining failed statistical gates under explicit user research adoption; "
            if user_adopted else "Recomputed all gates from saved predictions; ")
            + "reloaded primary and independently refitted serialized models; original evidence unchanged"}
    return p, selected, evaluation, evidence, original_model


def verify_winner(out, root):
    """The autonomous winner path always enforces every statistical gate."""
    return _verify_selected(out, root)


def _finalize(out, report, checkpoint_id, name, version, *, root=ROOT, registry_path=None, user_adopted=False):
    root, out, report = Path(root).resolve(), Path(out).resolve(), Path(report).absolute()
    evaluation = read(out / "evaluation.json")
    required_flags = ("reproduced",) if user_adopted else ("improved", "meaningful_gain", "reproduced")
    if any(evaluation.get(key) is not True for key in required_flags):
        return {"registered": False, "checkpoint_created": False, "reason": "No fully validated meaningful improvement; incumbent preserved"}
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,95}", checkpoint_id) or not name.strip() or not version.strip():
        raise ValueError("Invalid checkpoint identity")
    out.relative_to(root / "research/results")
    report.relative_to(root / "docs")
    report_text = report.read_text(encoding="utf-8")
    if name not in report_text or version not in report_text or "신규 검증 체크포인트 없음" in report_text:
        raise ValueError("Detailed report must identify the exact winning checkpoint name and version")
    if "런타임 승인·활성화는 별도 단계" not in report_text:
        raise ValueError("Report must explicitly distinguish checkpoint validation from runtime approval and activation")
    if user_adopted and "사용자 채택" not in report_text:
        raise ValueError("Report must explicitly identify user research adoption")
    destination = root / "research/models/checkpoints" / checkpoint_id
    destination.resolve().relative_to(root)
    registry = Path(registry_path or root / "src/validated_checkpoints.json")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Existing checkpoint directories are immutable")
    if registry.exists() and any(row.get("id") == checkpoint_id for row in read(registry)["checkpoints"]):
        raise ValueError("Existing checkpoint IDs are immutable")
    p, selected, evaluation, reproduction, original_model = _verify_selected(out, root, user_adopted=user_adopted)
    created = []
    with tempfile.TemporaryDirectory(prefix=".checkpoint-finalize-", dir=out) as temporary:
        stage = Path(temporary)
        write_new(stage / "pipeline.json", selected["pipeline"])
        write_new(stage / "checkpoint_reproducibility.json", reproduction)
        # Older frozen runners did not write every seal. Produce separate
        # audited seals here only after recomputing the underlying evidence.
        write_new(stage / "checkpoint_evaluation_seal.json", {"sha256": digest(out / "evaluation.json")})
        write_new(stage / "checkpoint_final_metrics_seal.json", {"sha256": digest(original_model.parent / "metrics.json")})
        summary = f"고정 평가 Macro F1 {evaluation['final'][0]['baseline']['macro_f1']*100:.2f}% → {evaluation['final'][0]['candidate']['macro_f1']*100:.2f}%; 기존 자료의 후향적 검증"
        if user_adopted:
            summary = "사용자 채택 · 연구 기준선; 기존 자동 채택 기준 미통과; " + summary
        entry = {"id": checkpoint_id, "name": name, "version": version,
            "status": USER_ADOPTION_STATUS if user_adopted else "validated_retrospective",
            "pipeline": selected["pipeline"], "evidence": {"improved": evaluation["improved"],
                "meaningful_gain": evaluation["meaningful_gain"], "reproduced": True,
                **({"adoption_basis": "explicit_user_request", "standard_validation_passed": False} if user_adopted else {}),
                "scope": evaluation.get("scope", "retrospective_historical_holdout"),
                "summary": summary,
                "source_revision": p["source_revision"], "source_hashes": reproduction["source_hashes"],
                "protocol_sha256": digest(out / "protocol.json"), "selection_sha256": digest(out / "selection_lock.json"),
                "active_model_changed": False},
            "files": {"model": ref(original_model, root), "pipeline": ref(stage / "pipeline.json", root),
                "report": ref(report, root), "protocol": ref(out / "protocol.json", root),
                "protocol_lock": ref(out / "protocol_lock.json", root),
                "evaluation": ref(out / "evaluation.json", root), "reproducibility": ref(stage / "checkpoint_reproducibility.json", root),
                "evaluation_seal": ref(stage / "checkpoint_evaluation_seal.json", root),
                "selection": ref(out / "selection_lock.json", root), "selection_seal": ref(out / "selection_seal.json", root),
                "final_metrics": ref(original_model.parent / "metrics.json", root),
                "final_metrics_seal": ref(stage / "checkpoint_final_metrics_seal.json", root),
                "final_predictions": ref(original_model.parent / "predictions.npz", root)}}
        for relative in pipeline_sources(selected["pipeline"]):
            role = relative.replace("/", "__").replace(".", "_") if input_representation(selected["pipeline"]) == "sensor_sequence" or requires_record_ids(selected["pipeline"]) or is_spatial_recipe(selected["pipeline"]) or is_local_recipe(selected["pipeline"]) or is_support_recipe(selected["pipeline"]) or is_alternatives_recipe(selected["pipeline"]) or is_projection_recipe(selected["pipeline"]) else Path(relative).stem
            entry["files"]["source_" + role] = ref(out / "source_snapshot" / relative, root)
        if p.get("supplemental_training"):
            entry["files"]["supplemental_manifest"] = ref(out / "supplemental_manifest.json", root)
            entry["files"]["supplemental_cache"] = ref(out / "supplemental_training.npz", root)
            entry["files"]["source_supplemental"] = ref(out / "source_snapshot/experiments/iterative_improvement/supplemental.py", root)
            if input_representation(selected["pipeline"]) == "sensor_sequence":
                entry["files"]["source_supplemental_sequence"] = ref(out / "source_snapshot/experiments/iterative_improvement/supplemental_sequence.py", root)
            if is_gain_recipe(selected["pipeline"]):
                entry["files"]["source_record_sequence_training"] = ref(out / "source_snapshot/experiments/iterative_improvement/record_sequence_training.py", root)
            elif requires_record_ids(selected["pipeline"]):
                entry["files"]["source_record_training"] = ref(out / "source_snapshot/experiments/iterative_improvement/record_training.py", root)
        if p.get("identity_controls"):
            entry["files"]["identity_controls"] = ref(out / "audits/identity_controls.json", root)
        if user_adopted:
            authorization_path = out / "user_authorization.json"
            authorization = read(authorization_path)
            source_run = root / authorization["source_run"]
            source_run.resolve().relative_to(root / "research/results")
            entry["files"].update(user_authorization=ref(authorization_path, root),
                adoption_source_protocol=ref(source_run / "protocol.json", root),
                adoption_source_selection=ref(source_run / "selection_lock.json", root))
        # Validate before creating a canonical checkpoint or touching registry.
        validate_checkpoint(entry, root=root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir()
        try:
            for filename in ("pipeline.json", "checkpoint_reproducibility.json",
                    "checkpoint_evaluation_seal.json", "checkpoint_final_metrics_seal.json"):
                target = destination / filename
                (stage / filename).rename(target)
                created.append(target)
            target_model = destination / "model.joblib"
            target_model.symlink_to(os.path.relpath(original_model, destination))
            created.append(target_model)
            entry["files"].update(model=ref(target_model, root), pipeline=ref(destination / "pipeline.json", root),
                reproducibility=ref(destination / "checkpoint_reproducibility.json", root),
                evaluation_seal=ref(destination / "checkpoint_evaluation_seal.json", root),
                final_metrics_seal=ref(destination / "checkpoint_final_metrics_seal.json", root))
            validate_checkpoint(entry, root=root)
            manifest = destination / "checkpoint.json"
            write_new(manifest, entry)
            created.append(manifest)
            register_checkpoint(entry, registry_path=registry, root=root)
        except BaseException:
            # Do not undo a successful atomic registry write if a later delivery
            # error occurs; in all other failures remove only our new files.
            published = registry.exists() and any(row.get("id") == checkpoint_id for row in read(registry)["checkpoints"])
            if not published:
                for path in reversed(created):
                    path.unlink()
                destination.rmdir()
            raise
    return {"registered": True, "checkpoint_created": True, "id": checkpoint_id,
        "status": entry["status"],
        "checkpoint": str(destination), "model_storage": "relative symlink to original final candidate; no duplicated model bytes", "active_model_changed": False}


def finalize(out, report, checkpoint_id, name, version, *, root=ROOT, registry_path=None):
    return _finalize(out, report, checkpoint_id, name, version, root=root, registry_path=registry_path)


def finalize_user_adoption(out, report, checkpoint_id, name, version, *, root=ROOT, registry_path=None):
    """Publish only an explicitly authorized, reproducible research adoption."""
    return _finalize(out, report, checkpoint_id, name, version, root=root, registry_path=registry_path, user_adopted=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--user-adoption", action="store_true", help="Require linked explicit user research adoption; preserve actual gate outcomes")
    args = parser.parse_args()
    publish = finalize_user_adoption if args.user_adoption else finalize
    print(json.dumps(publish(args.output, args.report, args.checkpoint_id, args.name, args.version), ensure_ascii=False))


if __name__ == "__main__":
    main()
