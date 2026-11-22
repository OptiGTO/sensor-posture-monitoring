"""Dispatch versioned research recipes while retaining accepted legacy recipes.

Legacy implementations and their eight source pins remain unchanged. A new
input contract uses its own module and source set; extending that set never
adds requirements retroactively to an already accepted legacy checkpoint.
"""
from __future__ import annotations


LEGACY_PIPELINE_SOURCES = (
    "src/iterative_pipeline.py", "src/posture_core.py", "src/window_pipeline.py",
    "src/pressure_contrast_features.py", "src/domain_feature_selection.py",
    "src/group_probability_bias.py", "src/extended_models.py", "src/sagittal_hierarchy.py",
)
SEQUENCE_FAMILIES = frozenset({"cnn1d", "minirocket", "sequence_soft_voting"})
RECORD_FAMILIES = frozenset({"record_balanced_extra_trees", "record_bagged_extra_trees", "record_soft_voting"})
SPATIAL_FAMILIES = frozenset({"ordinal_extra_trees", "shrinkage_qda", "spatial_soft_voting"})
GAIN_FAMILIES = frozenset({"record_gain_extra_trees", "gain_soft_voting"})
LOCAL_FAMILIES = frozenset({"record_prototype", "class_conditional_mixture", "local_soft_voting"})
INTERACTION_FAMILIES = frozenset({"interaction_extra_trees", "interaction_soft_voting"})
SUPPORT_FAMILIES = frozenset({"support_extra_trees", "support_soft_voting"})
PROJECTION_FAMILIES = frozenset({"projection_extra_trees", "projection_soft_voting"})
HARDNESS_FAMILIES = frozenset({"hardness_extra_trees", "hardness_random_forest", "hardness_soft_voting"})
STRUCTURED_FAMILIES = frozenset({"coupled_extra_trees", "subject_residual_extra_trees", "structured_soft_voting"})
ALTERNATIVES_FAMILIES = frozenset({"tail_extra_trees", "ovr_extra_trees", "alternatives_soft_voting"})
# Dispatch may grow without changing an accepted recipe. Pin each versioned
# algorithm and its input dependencies, as legacy recipes do; finalization also
# verifies the fitted class, normalized recipe and effective seed.
RECORD_PIPELINE_SOURCES = (
    "src/record_pipeline_v1.py", "src/posture_core.py",
    "src/window_pipeline.py", "src/__init__.py",
)
SPATIAL_PIPELINE_SOURCES = (
    "src/spatial_pipeline_v1.py", "src/spatial_features_v1.py", "src/posture_core.py",
    "src/window_pipeline.py", "src/__init__.py",
)
GAIN_PIPELINE_SOURCES = (
    "src/augmentation_pipeline_v1.py", "src/record_gain_features_v1.py", "src/posture_core.py",
    "src/window_pipeline.py", "src/__init__.py",
)
LOCAL_PIPELINE_SOURCES = (
    "src/local_geometry_pipeline_v1.py", "src/record_prototype_v1.py", "src/class_conditional_mixture_v1.py",
    "src/record_gain_features_v1.py", "src/posture_core.py", "src/window_pipeline.py", "src/__init__.py",
)
INTERACTION_PIPELINE_SOURCES = (
    "src/interaction_pipeline_v1.py", "src/interaction_features_v1.py", "src/record_gain_features_v1.py",
    "src/posture_core.py", "src/window_pipeline.py", "src/__init__.py",
)
SUPPORT_PIPELINE_SOURCES = (
    "src/forest_support_pipeline_v1.py", "src/leaf_support_v1.py", "src/posture_core.py",
    "src/window_pipeline.py", "src/__init__.py",
)
PROJECTION_PIPELINE_SOURCES = (
    "src/learned_projection_pipeline_v1.py", "src/learned_projection_v1.py",
    "src/posture_core.py", "src/window_pipeline.py", "src/__init__.py",
)
HARDNESS_PIPELINE_SOURCES = (
    "src/hardness_pipeline_v1.py", "src/posture_core.py",
    "src/window_pipeline.py", "src/__init__.py",
)
STRUCTURED_PIPELINE_SOURCES = (
    "src/structured_forest_pipeline_v1.py", "src/pairwise_coupling_v1.py", "src/subject_residual_v1.py",
    "src/posture_core.py", "src/window_pipeline.py", "src/__init__.py",
)
ALTERNATIVES_PIPELINE_SOURCES = (
    "src/forest_alternatives_pipeline_v1.py", "src/tail_clipping_v1.py", "src/binary_forest_v1.py",
    "src/posture_core.py", "src/window_pipeline.py", "src/__init__.py",
)
SEQUENCE_PIPELINE_SOURCES = (
    "src/sequence_pipeline_v1.py", "src/extended_models.py",
    "src/posture_core.py", "src/window_pipeline.py", "src/__init__.py",
)
MINIROCKET_SOURCES = (
    "src/_vendor/minirocket_multi.py", "src/_vendor/__init__.py", "src/_vendor/LICENSE.sktime",
)


def _sequence(spec):
    return isinstance(spec, dict) and spec.get("family") in SEQUENCE_FAMILIES


def _record(spec):
    return isinstance(spec, dict) and spec.get("family") in RECORD_FAMILIES


def _spatial(spec):
    return isinstance(spec, dict) and spec.get("family") in SPATIAL_FAMILIES


def _gain(spec):
    return isinstance(spec, dict) and spec.get("family") in GAIN_FAMILIES


def _local(spec):
    return isinstance(spec, dict) and spec.get("family") in LOCAL_FAMILIES


def _interaction(spec):
    return isinstance(spec, dict) and spec.get("family") in INTERACTION_FAMILIES


def _support(spec):
    return isinstance(spec, dict) and spec.get("family") in SUPPORT_FAMILIES


def _projection(spec):
    return isinstance(spec, dict) and spec.get("family") in PROJECTION_FAMILIES


def _hardness(spec):
    return isinstance(spec, dict) and spec.get("family") in HARDNESS_FAMILIES


def _structured(spec):
    return isinstance(spec, dict) and spec.get("family") in STRUCTURED_FAMILIES


def _alternatives(spec):
    return isinstance(spec, dict) and spec.get("family") in ALTERNATIVES_FAMILIES


def validate_pipeline(spec: dict) -> dict:
    if _structured(spec):
        from .structured_forest_pipeline_v1 import validate_structured_pipeline
        return validate_structured_pipeline(spec)
    if _hardness(spec):
        from .hardness_pipeline_v1 import validate_hardness_pipeline
        return validate_hardness_pipeline(spec)
    if _projection(spec):
        from .learned_projection_pipeline_v1 import validate_projection_pipeline
        return validate_projection_pipeline(spec)
    if _alternatives(spec):
        from .forest_alternatives_pipeline_v1 import validate_alternatives_pipeline
        return validate_alternatives_pipeline(spec)
    if _support(spec):
        from .forest_support_pipeline_v1 import validate_support_pipeline
        return validate_support_pipeline(spec)
    if _interaction(spec):
        from .interaction_pipeline_v1 import validate_interaction_pipeline
        return validate_interaction_pipeline(spec)
    if _local(spec):
        from .local_geometry_pipeline_v1 import validate_local_pipeline
        return validate_local_pipeline(spec)
    if _gain(spec):
        from .augmentation_pipeline_v1 import validate_augmentation_pipeline
        return validate_augmentation_pipeline(spec)
    if _spatial(spec):
        from .spatial_pipeline_v1 import validate_spatial_pipeline
        return validate_spatial_pipeline(spec)
    if _record(spec):
        from .record_pipeline_v1 import validate_record_pipeline
        return validate_record_pipeline(spec)
    if _sequence(spec):
        from .sequence_pipeline_v1 import validate_sequence_pipeline
        return validate_sequence_pipeline(spec)
    from .iterative_pipeline import validate_pipeline as validate_legacy
    return validate_legacy(spec)


def build_pipeline(spec: dict, seed: int | None = None):
    if _structured(spec):
        from .structured_forest_pipeline_v1 import build_structured_pipeline
        return build_structured_pipeline(spec, seed=seed)
    if _hardness(spec):
        from .hardness_pipeline_v1 import build_hardness_pipeline
        return build_hardness_pipeline(spec, seed=seed)
    if _projection(spec):
        from .learned_projection_pipeline_v1 import build_projection_pipeline
        return build_projection_pipeline(spec, seed=seed)
    if _alternatives(spec):
        from .forest_alternatives_pipeline_v1 import build_alternatives_pipeline
        return build_alternatives_pipeline(spec, seed=seed)
    if _support(spec):
        from .forest_support_pipeline_v1 import build_support_pipeline
        return build_support_pipeline(spec, seed=seed)
    if _interaction(spec):
        from .interaction_pipeline_v1 import build_interaction_pipeline
        return build_interaction_pipeline(spec, seed=seed)
    if _local(spec):
        from .local_geometry_pipeline_v1 import build_local_pipeline
        return build_local_pipeline(spec, seed=seed)
    if _gain(spec):
        from .augmentation_pipeline_v1 import build_augmentation_pipeline
        return build_augmentation_pipeline(spec, seed=seed)
    if _spatial(spec):
        from .spatial_pipeline_v1 import build_spatial_pipeline
        return build_spatial_pipeline(spec, seed=seed)
    if _record(spec):
        from .record_pipeline_v1 import build_record_pipeline
        return build_record_pipeline(spec, seed=seed)
    if _sequence(spec):
        from .sequence_pipeline_v1 import build_sequence_pipeline
        return build_sequence_pipeline(spec, seed=seed)
    from .iterative_pipeline import build_pipeline as build_legacy
    return build_legacy(spec, seed=seed)


def fit_pipeline(model, spec: dict, X, y, *, groups=None, record_ids=None, validation_data=None):
    if _structured(spec):
        from .structured_forest_pipeline_v1 import fit_structured_pipeline
        return fit_structured_pipeline(model, spec, X, y, groups=groups, record_ids=record_ids,
                                       validation_data=validation_data)
    if _hardness(spec):
        from .hardness_pipeline_v1 import fit_hardness_pipeline
        return fit_hardness_pipeline(model, spec, X, y, groups=groups, record_ids=record_ids,
                                     validation_data=validation_data)
    if _projection(spec):
        from .learned_projection_pipeline_v1 import fit_projection_pipeline
        return fit_projection_pipeline(model, spec, X, y, groups=groups, validation_data=validation_data)
    if _alternatives(spec):
        from .forest_alternatives_pipeline_v1 import fit_alternatives_pipeline
        return fit_alternatives_pipeline(model, spec, X, y, groups=groups, validation_data=validation_data)
    if _support(spec):
        from .forest_support_pipeline_v1 import fit_support_pipeline
        return fit_support_pipeline(model, spec, X, y, groups=groups, validation_data=validation_data)
    if _interaction(spec):
        from .interaction_pipeline_v1 import fit_interaction_pipeline
        return fit_interaction_pipeline(model, spec, X, y, groups=groups, validation_data=validation_data)
    if _local(spec):
        from .local_geometry_pipeline_v1 import fit_local_pipeline, local_requires_record_ids
        return fit_local_pipeline(model, spec, X, y, groups=groups,
            record_ids=record_ids if local_requires_record_ids(spec) else None, validation_data=validation_data)
    if _gain(spec):
        from .augmentation_pipeline_v1 import fit_augmentation_pipeline
        return fit_augmentation_pipeline(model, spec, X, y, groups=groups, record_ids=record_ids,
                                         validation_data=validation_data)
    if _spatial(spec):
        from .spatial_pipeline_v1 import fit_spatial_pipeline
        return fit_spatial_pipeline(model, spec, X, y, groups=groups, validation_data=validation_data)
    if _record(spec):
        from .record_pipeline_v1 import fit_record_pipeline
        return fit_record_pipeline(model, spec, X, y, groups=groups, record_ids=record_ids,
                                   validation_data=validation_data)
    if _sequence(spec):
        from .sequence_pipeline_v1 import fit_sequence_pipeline
        return fit_sequence_pipeline(model, spec, X, y, groups=groups, validation_data=validation_data)
    from .iterative_pipeline import fit_pipeline as fit_legacy
    return fit_legacy(model, spec, X, y, groups=groups, validation_data=validation_data)


def input_representation(spec: dict) -> str:
    recipe = validate_pipeline(spec)
    return "sensor_sequence" if _sequence(recipe) or _gain(recipe) or _interaction(recipe) else "feature_vector"


def requires_record_ids(spec: dict) -> bool:
    recipe = validate_pipeline(spec)
    if _local(recipe):
        from .local_geometry_pipeline_v1 import local_requires_record_ids
        return local_requires_record_ids(recipe)
    return _record(recipe) or _gain(recipe) or _hardness(recipe) or _structured(recipe)


def is_local_recipe(spec: dict) -> bool:
    return _local(validate_pipeline(spec))


def is_interaction_recipe(spec: dict) -> bool:
    return _interaction(validate_pipeline(spec))


def is_support_recipe(spec: dict) -> bool:
    return _support(validate_pipeline(spec))


def is_projection_recipe(spec: dict) -> bool:
    return _projection(validate_pipeline(spec))


def is_hardness_recipe(spec: dict) -> bool:
    return _hardness(validate_pipeline(spec))


def is_structured_recipe(spec: dict) -> bool:
    return _structured(validate_pipeline(spec))


def is_alternatives_recipe(spec: dict) -> bool:
    return _alternatives(validate_pipeline(spec))


def is_gain_recipe(spec: dict) -> bool:
    return _gain(validate_pipeline(spec))


def validate_identity_control(reference: dict, control: dict) -> None:
    """Accept declared zero-gain, identity-feature and zero-processing ET checks.

    Other no-op families need their own declared equivalence rule. Legacy ET
    defaults are not inferred: all four parameters must be explicit, including
    the distinction between integer feature counts and fractional max_features.
    """
    baseline, candidate = validate_pipeline(reference), validate_pipeline(control)
    if (baseline["family"] != "extra_trees" or candidate["family"] not in {
                "record_gain_extra_trees", "interaction_extra_trees", "support_extra_trees", "tail_extra_trees", "projection_extra_trees", "hardness_extra_trees", "subject_residual_extra_trees"}
            or baseline["feature_engineering"] != {"mode": "all"}
            or baseline["preprocessing"] != {"scaler": "standard"}
            or baseline["postprocessing"] != {"kind": "argmax"}
            or baseline["window_seconds"] != 3):
        raise ValueError("Identity control requires full-feature standard-scaled legacy Extra Trees and an explicitly supported no-op leaf")
    if candidate["family"] == "subject_residual_extra_trees":
        if (candidate["feature_engineering"] != {"mode": "identity"}
                or candidate["preprocessing"] != {"scaler": "standard"}
                or candidate["postprocessing"] != {"kind": "argmax"}):
            raise ValueError("Subject residual identity must preserve every standardized original feature")
    elif candidate["family"] == "hardness_extra_trees":
        if (candidate["feature_engineering"] != {"mode": "all"}
                or candidate["preprocessing"] != {"scaler": "standard"}
                or candidate["postprocessing"] != {"kind": "argmax"}
                or candidate["training"]["hardness"]["coefficient"] != 0):
            raise ValueError("Hardness identity must disable OOF reweighting on the original features")
    elif candidate["family"] == "projection_extra_trees":
        if (candidate["feature_engineering"] != {"mode": "identity"}
                or candidate["preprocessing"] != {"scaler": "standard"}
                or candidate["postprocessing"] != {"kind": "argmax"}):
            raise ValueError("Projection identity must explicitly preserve every standardized original feature")
    elif candidate["family"] == "tail_extra_trees":
        if (candidate["feature_engineering"] != {"mode": "all"}
                or candidate["preprocessing"] != {"scaler": "standard", "tail_fraction": 0.0}
                or candidate["postprocessing"] != {"kind": "argmax"}):
            raise ValueError("Tail identity control must explicitly disable clipping on all features")
    elif candidate["family"] == "record_gain_extra_trees":
        augmentation = candidate["training"]["augmentation"]
        if augmentation["mode"] != "identity" or augmentation["strength"] != 0:
            raise ValueError("Identity control must explicitly use identity augmentation with zero strength")
    elif candidate["family"] == "support_extra_trees":
        if (candidate["feature_engineering"] != {"mode": "all"}
                or candidate["postprocessing"] != {"kind": "uniform_leaf_shrinkage", "alpha": 0.0}):
            raise ValueError("Support identity control must explicitly use all features and zero leaf shrinkage")
    elif candidate["feature_engineering"] != {"mode": "identity"}:
        raise ValueError("Interaction identity control must explicitly use identity features")
    parameters = {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
    old, new = baseline["model_params"], candidate["model_params"]
    if (set(old) != parameters or set(new) != parameters
            or any(type(old[key]) is not type(new[key]) or old[key] != new[key] for key in parameters)):
        raise ValueError("Identity control requires the exact four explicitly specified Extra Trees parameters")
    if (baseline["training"]["weighting"] != "class_balanced"
            or candidate["training"]["weighting"] != "class_balanced"
            or baseline["training"]["seed"] != candidate["training"]["seed"]
            or baseline["training"].get("data_policy") != candidate["training"].get("data_policy")):
        raise ValueError("Identity control class balancing, seed and training-record policy must match the reference")


def is_spatial_recipe(spec: dict) -> bool:
    return _spatial(validate_pipeline(spec))


def combination_family(spec: dict) -> str:
    recipe = validate_pipeline(spec)
    if _structured(recipe):
        return "structured_soft_voting"
    if _hardness(recipe):
        return "hardness_soft_voting"
    if _projection(recipe):
        return "projection_soft_voting"
    if _alternatives(recipe):
        return "alternatives_soft_voting"
    if _support(recipe):
        return "support_soft_voting"
    if _interaction(recipe):
        return "interaction_soft_voting"
    if _local(recipe):
        return "local_soft_voting"
    if _gain(recipe):
        return "gain_soft_voting"
    if _spatial(recipe):
        return "spatial_soft_voting"
    if _record(recipe):
        return "record_soft_voting"
    return "sequence_soft_voting" if _sequence(recipe) else "equal_soft_voting"


def pipeline_sources(spec: dict) -> tuple[str, ...]:
    recipe = validate_pipeline(spec)
    if _structured(recipe):
        return STRUCTURED_PIPELINE_SOURCES
    if _hardness(recipe):
        return HARDNESS_PIPELINE_SOURCES
    if _projection(recipe):
        return PROJECTION_PIPELINE_SOURCES
    if _alternatives(recipe):
        return ALTERNATIVES_PIPELINE_SOURCES
    if _support(recipe):
        return SUPPORT_PIPELINE_SOURCES
    if _interaction(recipe):
        return INTERACTION_PIPELINE_SOURCES
    if _local(recipe):
        return LOCAL_PIPELINE_SOURCES
    if _gain(recipe):
        return GAIN_PIPELINE_SOURCES
    if _spatial(recipe):
        return SPATIAL_PIPELINE_SOURCES
    if _record(recipe):
        return RECORD_PIPELINE_SOURCES
    if not _sequence(recipe):
        return LEGACY_PIPELINE_SOURCES
    members = recipe.get("members", ())
    uses_rocket = recipe["family"] == "minirocket" or any(member["family"] == "minirocket" for member in members)
    return SEQUENCE_PIPELINE_SOURCES + (MINIROCKET_SOURCES if uses_rocket else ())
