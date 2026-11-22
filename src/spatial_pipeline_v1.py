"""Versioned spatial feature and shrinkage-discriminant research recipes.

Each serialized estimator keeps its exact feature mapping, train-only scaler,
class order and fitted model. External validation never selects parameters.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .posture_core import CLASS_NAMES, feature_names
from .spatial_features_v1 import SpatialFeatureTransformer


ORDINAL = "ordinal_extra_trees"
QDA = "shrinkage_qda"
VOTING = "spatial_soft_voting"
SPATIAL_FAMILIES = frozenset({ORDINAL, QDA, VOTING})
WIDTHS = {"all_plus_mean_order": 716, "all_plus_mean_median_order": 836, "abs_rel_sensor_means": 32}
DATA_POLICY = "include_ordinary_reserved_training_records_v1"
DEFAULT_SEED = 20260921
TREE_DEFAULTS = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 8, "max_features": .5}
QDA_FIXED = {"solver": "svd", "shrinkage": None, "covariance_estimator": None,
    "priors": [1 / 7] * 7, "tol": 1e-4, "store_covariance": True}
FEATURE_ATTRIBUTES = ("mean_indices_", "median_indices_", "selected_indices_", "pair_indices_",
    "feature_names_in_", "feature_names_out_")


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("Spatial pipeline seed must be a nonnegative uint32 integer")
    return value


def validate_spatial_pipeline(spec: dict) -> dict:
    """Return a complete, JSON-safe spatial recipe with explicit QDA semantics."""
    if not isinstance(spec, dict):
        raise ValueError("Spatial pipeline specification must be an object")
    value = deepcopy(spec)
    family = value.get("family")
    allowed = {"family", "base_method", "window_seconds", "feature_engineering", "preprocessing",
        "model_params", "training", "postprocessing", "members"}
    if set(value) - allowed or not isinstance(family, str) or family not in SPATIAL_FAMILIES:
        raise ValueError("Unknown spatial pipeline field or family")
    mode = "all_plus_mean_order" if family == ORDINAL else "abs_rel_sensor_means" if family == QDA else "per_member"
    value.setdefault("base_method", "standard")
    value.setdefault("window_seconds", 3)
    value.setdefault("feature_engineering", {"mode": mode})
    value.setdefault("preprocessing", {"scaler": "standard"})
    value.setdefault("model_params", {})
    value.setdefault("training", {})
    value.setdefault("postprocessing", {"kind": "argmax"})
    for key in ("feature_engineering", "preprocessing", "model_params", "training", "postprocessing"):
        if not isinstance(value[key], dict):
            raise ValueError(f"{key} must be an object")
    if value["base_method"] not in ("standard", "regularized"):
        raise ValueError("Unknown spatial base method")
    if type(value["window_seconds"]) is not int or value["window_seconds"] != 3:
        raise ValueError("Spatial pipeline v1 requires the 3-second feature contract")
    feature = value["feature_engineering"]
    if set(feature) != {"mode"}:
        raise ValueError("Spatial feature mode must be fixed before fitting")
    supported_modes = ("all_plus_mean_order", "all_plus_mean_median_order") if family == ORDINAL else (mode,)
    if feature["mode"] not in supported_modes:
        raise ValueError("Spatial feature mode is incompatible with the selected family")
    if value["preprocessing"] != {"scaler": "standard"} or value["postprocessing"] != {"kind": "argmax"}:
        raise ValueError("Spatial pipeline v1 requires train StandardScaler and argmax")
    training = value["training"]
    if set(training) - {"seed", "weighting", "data_policy"}:
        raise ValueError("Unknown spatial training fields")
    expected_weighting = "class_balanced" if family == ORDINAL else "uniform_class_prior" if family == QDA else "per_member"
    training.setdefault("seed", DEFAULT_SEED)
    training.setdefault("weighting", expected_weighting)
    _seed(training["seed"])
    if training["weighting"] != expected_weighting:
        raise ValueError(f"{family} requires weighting={expected_weighting}")
    if training.get("data_policy") not in (None, DATA_POLICY):
        raise ValueError("Unknown training-record policy")
    params = value["model_params"]
    if family == VOTING:
        members = value.get("members")
        if params or value["base_method"] != "standard" or not isinstance(members, list) or len(members) < 2:
            raise ValueError("Spatial voting requires two or more complete members and no outer model parameters")
        if any(not isinstance(member, dict) or member.get("family") not in (ORDINAL, QDA) for member in members):
            raise ValueError("Spatial voting accepts only nonnested ordinal ET or shrinkage QDA members")
        value["members"] = [validate_spatial_pipeline(member) for member in members]
        if any(member["training"].get("data_policy") != training.get("data_policy")
                or member["training"]["seed"] != training["seed"] for member in value["members"]):
            raise ValueError("Spatial voting members must share training-record policy and primary seed")
        if len({json.dumps(member, sort_keys=True) for member in value["members"]}) != len(members):
            raise ValueError("Spatial voting members must have distinct recipes")
    else:
        if "members" in value:
            raise ValueError("Only spatial_soft_voting accepts members")
        defaults = TREE_DEFAULTS if family == ORDINAL else {"reg_param": .5, **QDA_FIXED}
        if set(params) - set(defaults):
            raise ValueError("Unknown spatial model parameters; weights and seed are controlled by the recipe")
        for key, default in defaults.items():
            params.setdefault(key, deepcopy(default))
        if family == ORDINAL:
            for key in ("n_estimators", "min_samples_leaf"):
                if type(params[key]) is not int or params[key] < 1:
                    raise ValueError(f"{key} must be a positive integer")
            if params["max_depth"] is not None and (type(params["max_depth"]) is not int or params["max_depth"] < 1):
                raise ValueError("max_depth must be null or a positive integer")
            count = params["max_features"]
            if not ((type(count) is int and 1 <= count <= WIDTHS[feature["mode"]])
                    or (type(count) is float and np.isfinite(count) and 0 < count <= 1)
                    or (isinstance(count, str) and count in ("sqrt", "log2"))):
                raise ValueError("max_features must be a supported feature count or fraction")
        else:
            strength = params["reg_param"]
            if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not np.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError("QDA reg_param must be finite and within [0, 1]")
            if (not isinstance(params["priors"], list) or params["priors"] != QDA_FIXED["priors"]
                    or params["store_covariance"] is not True
                    or any(params[key] != expected for key, expected in QDA_FIXED.items() if key not in {"priors", "store_covariance"})):
                raise ValueError("QDA requires fixed SVD solver, uniform seven-class priors, stored covariance and tolerance")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Spatial recipe must contain finite JSON values") from error
    return value


def _features(X):
    if hasattr(X, "columns") and list(X.columns) != feature_names():
        raise ValueError("Named input columns differ from the frozen 596-feature order")
    values = np.asarray(X)
    if (values.ndim != 2 or values.shape[1] != 596 or not len(values)
            or values.dtype.kind not in "iuf" or not np.isfinite(values).all()):
        raise ValueError("Expected nonempty finite numeric features with shape (N, 596)")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(values, dtype=np.float32 if values.dtype == np.dtype(np.float32) else np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Features exceed the supported finite numeric range")
    return result


def _digest_arrays(values):
    digest = hashlib.sha256()
    for key in sorted(values):
        array = np.ascontiguousarray(values[key])
        digest.update(json.dumps([key, array.dtype.str, list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _scaler_digest(scaler):
    return _digest_arrays({key: getattr(scaler, key) for key in ("mean_", "scale_", "var_", "n_samples_seen_")})


def _qda_digest(estimator):
    values = {key: getattr(estimator, key) for key in ("means_", "priors_", "classes_")}
    for key in ("covariance_", "scalings_", "rotations_"):
        values.update({f"{key}{index}": array for index, array in enumerate(getattr(estimator, key))})
    return _digest_arrays(values)


def _estimator(spec, seed):
    params = deepcopy(spec["model_params"])
    if spec["family"] == ORDINAL:
        return ExtraTreesClassifier(**params, class_weight="balanced", random_state=seed, n_jobs=1,
            criterion="gini", bootstrap=False, ccp_alpha=0., min_impurity_decrease=0.)
    return QuadraticDiscriminantAnalysis(**params)


class SpatialIterativePipeline(ClassifierMixin, BaseEstimator):
    def __init__(self, spec, seed=None):
        self.spec = spec
        self.seed = seed

    def fit(self, X, y, *, groups=None, validation_data=None):
        spec = validate_spatial_pipeline(self.spec)
        seed = _seed(spec["training"]["seed"] if self.seed is None else self.seed)
        values, labels = _features(X), np.asarray(y)
        if (labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu"
                or not np.array_equal(np.unique(labels), np.arange(7))):
            raise ValueError("Spatial training requires matching labels with all seven posture IDs 0..6")
        labels = labels.astype(np.int64, copy=True)
        if spec["family"] == QDA and np.any(np.bincount(labels, minlength=7) < 32):
            raise ValueError("Full 32-axis SVD QDA requires at least 32 training rows in every posture class")
        subjects = None
        if groups is not None:
            identities = np.asarray(groups)
            if identities.ndim != 1 or len(identities) != len(values) or any(
                    not isinstance(item, str) or not item.strip() for item in identities.tolist()):
                raise ValueError("Spatial training groups must be aligned nonempty subject strings")
            subjects = sorted(set(identities.tolist()))
        family = spec["family"]
        with threadpool_limits(limits=1):
            if family == VOTING:
                members = [build_spatial_pipeline(member, seed=self.seed).fit(values, labels, groups=groups)
                    for member in spec["members"]]
            else:
                features = SpatialFeatureTransformer(mode=spec["feature_engineering"]["mode"])
                engineered = features.fit_transform(values, labels)
                scaler = StandardScaler()
                transformed = scaler.fit_transform(engineered)
                estimator = _estimator(spec, seed).fit(transformed, labels)
        self.spec_, self.effective_seed_ = spec, seed
        self.classes_, self.label_names_ = np.arange(7), list(CLASS_NAMES)
        self.n_features_in_, self.feature_names_in_ = 596, np.asarray(feature_names())
        self.training_details_ = {"pipeline": deepcopy(spec), "effective_seed": seed,
            "sklearn_version": sklearn.__version__, "input_contract": "feature_vector",
            "training_windows": len(values), "training_feature_dtype": str(values.dtype),
            "training_subjects": subjects, "class_counts": np.bincount(labels, minlength=7).tolist(),
            "class_names": list(CLASS_NAMES), "external_validation_used": False,
            "preprocessing_fit": "predeclared spatial feature mapping, then unweighted StandardScaler fitted on training rows only",
            "training_record_policy": spec["training"].get("data_policy"),
            "training_record_policy_application": "caller supplies eligible training rows; historical caches are never imported by estimator",
            "threads": 1}
        if family == VOTING:
            self.members_ = members
            self.training_details_.update(combination="equal aligned probability mean",
                weights=[1 / len(members)] * len(members), members=[deepcopy(member.training_details_) for member in members])
        else:
            self.features_, self.scaler_, self.estimator_ = features, scaler, estimator
            self.n_transformed_features_ = engineered.shape[1]
            self.scaler_state_sha256_ = _scaler_digest(scaler)
            self.training_details_.update(model_input_features=self.n_transformed_features_,
                model_input_dtype=str(transformed.dtype), transformed_feature_names=features.get_feature_names_out().tolist(),
                feature_mapping={key: np.asarray(getattr(features, key)).tolist() for key in FEATURE_ATTRIBUTES},
                scaler_state_sha256=self.scaler_state_sha256_, effective_model_params=estimator.get_params(deep=False))
            if family == QDA:
                self.qda_state_sha256_ = _qda_digest(estimator)
                self.training_details_.update(qda_state_sha256=self.qda_state_sha256_,
                    class_priors=estimator.priors_.tolist(),
                    weighting_semantics="uniform class priors in QDA posterior; class covariance/mean fitted to each class's unweighted training rows",
                    covariance_regularization="SVD scaling=(1-reg_param)*singular_value_squared/(class_count-1)+reg_param; no solver fallback",
                    covariance_shapes=[list(matrix.shape) for matrix in estimator.covariance_],
                    covariance_min_scalings=[float(scaling.min()) for scaling in estimator.scalings_],
                    qda_uses_random_seed=False)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "spec_")
        if validate_spatial_pipeline(self.spec) != self.spec_ or _seed(
                self.spec_["training"]["seed"] if self.seed is None else self.seed) != self.effective_seed_:
            raise ValueError("Spatial recipe or seed changed after fitting")
        values = _features(X)
        with threadpool_limits(limits=1):
            if self.spec_["family"] == VOTING:
                result = np.mean([member.predict_proba(values) for member in self.members_], axis=0)
            else:
                transformed = self.scaler_.transform(self.features_.transform(values))
                classes = np.asarray(self.estimator_.classes_)
                if classes.dtype.kind not in "iu" or not np.array_equal(np.sort(classes), self.classes_):
                    raise ValueError("Spatial estimator posture classes changed")
                raw = np.asarray(self.estimator_.predict_proba(transformed), dtype=float)
                if raw.shape != (len(values), 7):
                    raise ValueError("Invalid spatial estimator probability shape")
                result = np.empty_like(raw)
                result[:, classes] = raw
        if (result.shape != (len(values), 7) or not np.isfinite(result).all() or (result < 0).any()
                or not np.allclose(result.sum(1), 1., atol=1e-12, rtol=0)):
            raise ValueError("Invalid spatial estimator probabilities")
        return result

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def build_spatial_pipeline(spec: dict, seed: int | None = None):
    if seed is not None:
        _seed(seed)
    return SpatialIterativePipeline(validate_spatial_pipeline(spec), seed=seed)


def fit_spatial_pipeline(model, spec: dict, X, y, *, groups=None, validation_data=None):
    normalized = validate_spatial_pipeline(spec)
    if not isinstance(model, SpatialIterativePipeline) or validate_spatial_pipeline(model.spec) != normalized:
        raise ValueError("Training specification differs from the constructed spatial pipeline")
    return model.fit(X, y, groups=groups, validation_data=validation_data)


def verify_fitted_spatial_identity(model, spec: dict, seed: int):
    """Verify recipe, mapping, model parameters and persisted fitted state."""
    recipe, seed = validate_spatial_pipeline(spec), _seed(seed)
    if (not isinstance(model, SpatialIterativePipeline) or getattr(model, "spec_", None) != recipe
            or validate_spatial_pipeline(model.spec) != recipe or getattr(model, "effective_seed_", None) != seed
            or (recipe["training"]["seed"] if model.seed is None else model.seed) != seed
            or getattr(model, "n_features_in_", None) != 596
            or not np.array_equal(getattr(model, "classes_", ()), np.arange(7))
            or getattr(model, "label_names_", None) != list(CLASS_NAMES)
            or not np.array_equal(getattr(model, "feature_names_in_", ()), feature_names())):
        raise ValueError("Fitted spatial recipe, seed or input contract changed")
    details = getattr(model, "training_details_", {})
    count = details.get("training_windows")
    if (details.get("pipeline") != recipe or details.get("effective_seed") != seed
            or details.get("sklearn_version") != sklearn.__version__ or details.get("external_validation_used") is not False
            or type(count) is not int or count < 7 or details.get("class_names") != list(CLASS_NAMES)
            or details.get("training_record_policy") != recipe["training"].get("data_policy")):
        raise ValueError("Fitted spatial training audit differs from its recipe or sklearn version")
    if recipe["family"] == VOTING:
        members = getattr(model, "members_", ())
        if len(members) != len(recipe["members"]):
            raise ValueError("Fitted spatial voting member count changed")
        for member, member_recipe in zip(members, recipe["members"]):
            verify_fitted_spatial_identity(member, member_recipe, seed)
        return True
    mode = recipe["feature_engineering"]["mode"]
    expected_features = SpatialFeatureTransformer(mode=mode).fit(np.zeros((1, 596), dtype=np.float32))
    features = getattr(model, "features_", None)
    if (type(features) is not SpatialFeatureTransformer or features.get_params(deep=False) != {"mode": mode}
            or getattr(features, "mode_", None) != mode or getattr(features, "n_features_in_", None) != 596
            or getattr(features, "n_features_out_", None) != WIDTHS[mode]
            or any(not np.array_equal(getattr(features, key, ()), getattr(expected_features, key)) for key in FEATURE_ATTRIBUTES)
            or getattr(model, "n_transformed_features_", None) != WIDTHS[mode]
            or details.get("model_input_features") != WIDTHS[mode]
            or details.get("transformed_feature_names") != expected_features.get_feature_names_out().tolist()
            or details.get("feature_mapping") != {key: np.asarray(getattr(expected_features, key)).tolist() for key in FEATURE_ATTRIBUTES}):
        raise ValueError("Fitted spatial feature mapping differs from the predeclared mode")
    scaler = getattr(model, "scaler_", None)
    if (type(scaler) is not StandardScaler or scaler.get_params(deep=False) != StandardScaler().get_params(deep=False)
            or getattr(scaler, "n_features_in_", None) != WIDTHS[mode] or getattr(scaler, "n_samples_seen_", None) != count
            or _scaler_digest(scaler) != getattr(model, "scaler_state_sha256_", None)
            or details.get("scaler_state_sha256") != model.scaler_state_sha256_):
        raise ValueError("Fitted spatial scaler or learned statistics changed")
    estimator, expected = getattr(model, "estimator_", None), _estimator(recipe, seed)
    if (type(estimator) is not type(expected) or estimator.get_params(deep=False) != expected.get_params(deep=False)
            or details.get("effective_model_params") != expected.get_params(deep=False)
            or getattr(estimator, "n_features_in_", None) != WIDTHS[mode]
            or not np.array_equal(getattr(estimator, "classes_", ()), np.arange(7))):
        raise ValueError("Fitted spatial backend parameters, seed or classes changed")
    if recipe["family"] == ORDINAL:
        if len(getattr(estimator, "estimators_", ())) != recipe["model_params"]["n_estimators"]:
            raise ValueError("Fitted ordinal forest tree count changed")
    else:
        if (not np.array_equal(estimator.priors_, QDA_FIXED["priors"])
                or len(estimator.covariance_) != 7 or any(np.asarray(matrix).shape != (32, 32) for matrix in estimator.covariance_)
                or _qda_digest(estimator) != getattr(model, "qda_state_sha256_", None)
                or details.get("qda_state_sha256") != model.qda_state_sha256_):
            raise ValueError("Fitted QDA priors, covariance or discriminant state changed")
    return True
