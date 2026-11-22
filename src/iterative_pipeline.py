"""Serializable, explicit 596-feature recipes for controlled model studies.

This module builds recipes; it never decides whether a recipe is an improvement
and never registers, approves or activates one. All learned preprocessing is
fitted inside the estimator on the supplied training partition only.
"""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .posture_core import CLASS_NAMES, feature_names
from .pressure_contrast_features import PressureContrastFeatures, contrast_feature_names
from .domain_feature_selection import SubjectInvariantSelector, ShrinkageDiscriminantProjection


FAMILIES = frozenset({"rbf_svm", "logistic_regression", "random_forest",
    "extra_trees", "xgboost", "catboost", "mlp", "sagittal_hierarchy"})
FEATURE_MODES = frozenset({"all", "static", "means", "relative_static", "contrast", "all_plus_contrast",
    "subject_invariant", "shrinkage_lda"})
DEFAULT_SEED = 20260922


def validate_pipeline(spec: dict) -> dict:
    """Validate a complete reproducible recipe, without importing native models."""
    if not isinstance(spec, dict):
        raise ValueError("Pipeline specification must be an object")
    value = deepcopy(spec)
    allowed = {"family", "base_method", "window_seconds", "feature_engineering",
        "preprocessing", "model_params", "training", "postprocessing", "members"}
    if set(value) - allowed:
        raise ValueError(f"Unknown pipeline fields: {sorted(set(value) - allowed)}")
    family = value.get("family")
    if family not in FAMILIES | {"equal_soft_voting"}:
        raise ValueError(f"Unsupported iterative pipeline family: {family}")
    value.setdefault("base_method", "standard")
    value.setdefault("window_seconds", 3)
    value.setdefault("feature_engineering", {"mode": "all"})
    value.setdefault("preprocessing", {"scaler": "standard"})
    value.setdefault("model_params", {})
    value.setdefault("training", {})
    value.setdefault("postprocessing", {"kind": "argmax"})
    for key in ("feature_engineering", "preprocessing", "model_params", "training", "postprocessing"):
        if not isinstance(value[key], dict):
            raise ValueError(f"{key} must be an object")
    value["feature_engineering"].setdefault("mode", "all")
    value["preprocessing"].setdefault("scaler", "standard")
    value["training"].setdefault("seed", DEFAULT_SEED)
    value["training"].setdefault("weighting", "class_balanced")
    value["postprocessing"].setdefault("kind", "argmax")
    mode = value["feature_engineering"]["mode"]
    feature_keys = ({"mode", "k", "strength"} if mode == "subject_invariant" else
        {"mode", "n_components"} if mode == "shrinkage_lda" else {"mode"})
    post_kind = value["postprocessing"]["kind"]
    post_keys = ({"kind", "inner_folds", "max_sweeps", "regularization"} if post_kind == "macro_f1_bias"
        else {"kind", "temperature"})
    for key, keys in (("feature_engineering", feature_keys), ("preprocessing", {"scaler"}),
            ("training", {"seed", "weighting", "data_policy"}), ("postprocessing", post_keys)):
        if set(value[key]) - keys:
            raise ValueError(f"Unknown {key} fields")
    if value["base_method"] not in {"standard", "regularized"}:
        raise ValueError("base_method must be standard or regularized; model_params fixes the exact recipe")
    if type(value["window_seconds"]) is not int or value["window_seconds"] != 3:
        raise ValueError("Iterative pipeline evidence currently supports the 3-second feature contract only")
    if value["feature_engineering"]["mode"] not in FEATURE_MODES:
        raise ValueError("Unknown feature engineering mode")
    if mode == "subject_invariant":
        feature = value["feature_engineering"]
        if set(feature) != {"mode", "k", "strength"}:
            raise ValueError("subject_invariant requires explicit k and strength")
        if type(feature["k"]) is not int or not 1 <= feature["k"] <= 596:
            raise ValueError("subject_invariant k must be an integer between 1 and 596")
        if (isinstance(feature["strength"], bool) or not isinstance(feature["strength"], (int, float))
                or not np.isfinite(feature["strength"]) or feature["strength"] < 0):
            raise ValueError("subject_invariant strength must be nonnegative and finite")
    elif mode == "shrinkage_lda":
        feature = value["feature_engineering"]
        if "n_components" not in feature:
            raise ValueError("shrinkage_lda requires explicit n_components (integer or null)")
        count = feature["n_components"]
        if count is not None and (type(count) is not int or not 1 <= count <= len(CLASS_NAMES) - 1):
            raise ValueError("shrinkage_lda n_components must be null or an integer from 1 to 6")
    if value["preprocessing"]["scaler"] not in {"standard", "shrinkage_whiten", "quantile_normal"}:
        raise ValueError("Unknown preprocessing scaler")
    if type(value["training"]["seed"]) is not int or not 0 <= value["training"]["seed"] < 2**32:
        raise ValueError("Training seed must be a nonnegative uint32 integer")
    if value["training"]["weighting"] not in {"class_balanced", "subject_class_balanced"}:
        raise ValueError("Unknown training weighting")
    if value["training"].get("data_policy") not in {None, "include_ordinary_reserved_training_records_v1"}:
        raise ValueError("Unknown training data policy")
    kind = value["postprocessing"]["kind"]
    if kind not in {"argmax", "decision_softmax", "macro_f1_bias"}:
        raise ValueError("Unknown postprocessing kind")
    if kind == "decision_softmax" and family != "rbf_svm":
        raise ValueError("decision_softmax is supported only for RBF SVM")
    if kind == "macro_f1_bias":
        post = value["postprocessing"]
        if family == "equal_soft_voting" or set(post) != post_keys:
            raise ValueError("macro_f1_bias requires explicit folds, sweeps and regularization on a single estimator")
        if type(post["inner_folds"]) is not int or not 2 <= post["inner_folds"] <= 10:
            raise ValueError("inner_folds must be an integer from 2 to 10")
        if type(post["max_sweeps"]) is not int or not 1 <= post["max_sweeps"] <= 3:
            raise ValueError("max_sweeps must be an integer from 1 to 3")
        penalty = post["regularization"]
        if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or not np.isfinite(penalty) or penalty < 0:
            raise ValueError("Bias regularization must be nonnegative and finite")
    if kind == "decision_softmax":
        value["postprocessing"].setdefault("temperature", 1.0)
        temperature = value["postprocessing"]["temperature"]
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Softmax temperature must be positive and finite")
    elif "temperature" in value["postprocessing"]:
        raise ValueError("Temperature is only applicable to decision_softmax")
    if set(value["model_params"]) & {"random_state", "random_seed", "class_weight",
            "auto_class_weights", "n_jobs", "thread_count", "probability", "decision_function_shape"}:
        raise ValueError("Seed, weighting, probability and thread settings are controlled by the pipeline contract")
    if family == "sagittal_hierarchy":
        params = value["model_params"]
        if set(params) != {"base_family", "base_params"} or params["base_family"] not in {"extra_trees", "rbf_svm"}:
            raise ValueError("Hierarchy requires one supported base family and explicit shared base_params")
        if (mode != "all" or value["preprocessing"]["scaler"] != "standard"
                or value["training"]["weighting"] != "class_balanced" or kind != "argmax"):
            raise ValueError("Hierarchy currently requires full features, standard scaling, class balancing and argmax")
        validate_pipeline({"family": params["base_family"], "model_params": params["base_params"],
            "training": value["training"]})
    if family == "mlp":
        defaults = {"hidden_size": 32, "epochs": 40, "patience": 8, "batch_size": 64,
            "learning_rate": .001, "weight_decay": .01, "dropout": .2, "gain_std": 0., "noise_fraction": 0.}
        params = value["model_params"]
        if set(params) - set(defaults):
            raise ValueError("Unknown MLP model parameters; architecture is fixed to mlp")
        for key, default in defaults.items():
            params.setdefault(key, default)
        for key in ("hidden_size", "epochs", "patience", "batch_size"):
            if type(params[key]) is not int or params[key] < 1:
                raise ValueError(f"MLP {key} must be a positive integer")
        for key in ("learning_rate", "weight_decay", "dropout", "gain_std", "noise_fraction"):
            parameter = params[key]
            if isinstance(parameter, bool) or not isinstance(parameter, (int, float)) or not np.isfinite(parameter):
                raise ValueError(f"MLP {key} must be finite and numeric")
        if params["learning_rate"] <= 0 or params["weight_decay"] < 0 or not 0 <= params["dropout"] < 1:
            raise ValueError("MLP learning rate, weight decay or dropout is outside its valid range")
        if params["gain_std"] != 0 or params["noise_fraction"] != 0:
            raise ValueError("Raw sensor augmentation is unsupported for the MLP feature-vector input")
        if value["training"]["weighting"] != "class_balanced":
            raise ValueError("MLP currently supports class_balanced weighting only")
        if value["preprocessing"]["scaler"] != "standard":
            raise ValueError("MLP currently requires outer StandardScaler and internal training standardization")
    if value["preprocessing"]["scaler"] == "shrinkage_whiten" and value["training"]["weighting"] != "class_balanced":
        raise ValueError("Shrinkage whitening currently requires class_balanced weighting")
    if value["preprocessing"]["scaler"] == "quantile_normal" and value["training"]["weighting"] != "class_balanced":
        raise ValueError("Quantile scaling currently requires class_balanced weighting")
    if family == "equal_soft_voting":
        members = value.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError("Equal soft voting requires at least two complete member recipes")
        value["members"] = [validate_pipeline(member) for member in members]
        if any(member["training"].get("data_policy") != value["training"].get("data_policy") for member in value["members"]):
            raise ValueError("Voting members must share one training data policy")
        if any(member["family"] == "equal_soft_voting" for member in value["members"]):
            raise ValueError("Nested voting ensembles are not supported")
        if len({json.dumps(member, sort_keys=True) for member in value["members"]}) != len(members):
            raise ValueError("Voting member recipes must be distinct")
        if (value["feature_engineering"]["mode"] != "all" or value["preprocessing"]["scaler"] != "standard"
                or value["training"]["weighting"] != "class_balanced" or value["model_params"]
                or value["base_method"] != "standard"):
            raise ValueError("Voting preprocessing and weighting must be configured separately in each member")
    elif "members" in value:
        raise ValueError("Only equal_soft_voting accepts members")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Pipeline recipe must contain finite JSON values") from error
    return value


def selected_feature_names(mode: str) -> list[str]:
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature mode: {mode}")
    if mode == "all":
        return feature_names()
    if mode in {"subject_invariant", "shrinkage_lda"}:
        raise ValueError("Learned feature names require the fitted training-only transformer")
    if mode in {"contrast", "all_plus_contrast"}:
        return ([] if mode == "contrast" else feature_names()) + contrast_feature_names()
    retained = []
    for name in feature_names():
        prefix, bare = name.split("_", 1)
        if mode == "relative_static" and prefix != "rel":
            continue
        sensor_stats = ("mean", "median") if mode == "means" else (
            "mean", "std", "median", "min", "max", "range", "iqr", "rms")
        aggregate_stats = ("mean",) if mode == "means" else ("mean", "std")
        if any(bare.startswith(f"{stat}_R") for stat in sensor_stats):
            retained.append(name)
        elif any(bare == f"{axis}_{stat}_{index}" for axis in ("row", "column")
                for stat in aggregate_stats for index in range(1, 5)):
            retained.append(name)
        elif any(bare == f"{stat}_{derived}" for stat in aggregate_stats
                for derived in ("global", "left_right", "front_back", "diagonal", "outer_row", "outer_column")):
            retained.append(name)
    return retained


class NamedFeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, mode="all"):
        self.mode = mode

    def fit(self, X, y=None):
        if self.mode in {"contrast", "all_plus_contrast"}:
            raise ValueError("Derived contrasts require PressureContrastFeatures")
        if self.mode in {"subject_invariant", "shrinkage_lda"}:
            raise ValueError("Learned features require their dedicated training-only transformer")
        values = check_array(X)
        names = feature_names()
        if values.shape[1] != len(names):
            raise ValueError("Expected original 596-feature runtime input")
        self.n_features_in_ = values.shape[1]
        self.feature_names_out_ = selected_feature_names(self.mode)
        selected = set(self.feature_names_out_)
        self.indices_ = np.asarray([i for i, name in enumerate(names) if name in selected])
        return self

    def transform(self, X):
        check_is_fitted(self, "indices_")
        values = check_array(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Feature input differs from training contract")
        return values if self.mode == "all" else values[:, self.indices_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)


class ShrinkageWhitening(TransformerMixin, BaseEstimator):
    """Train-fitted Ledoit-Wolf covariance metric after standard scaling."""
    def fit(self, X, y=None):
        values = check_array(X)
        covariance = LedoitWolf().fit(values)
        eigenvalues, vectors = np.linalg.eigh(covariance.covariance_)
        self.location_ = covariance.location_
        self.whitening_ = (vectors * (1 / np.sqrt(np.maximum(eigenvalues, 1e-9)))) @ vectors.T
        self.shrinkage_ = float(covariance.shrinkage_)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "whitening_")
        values = check_array(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Whitening input differs from training contract")
        return (values - self.location_) @ self.whitening_


def _subject_class_weights(y, groups):
    if groups is None:
        raise ValueError("Subject-class balancing requires training subject IDs")
    subjects = np.asarray(groups)
    if subjects.ndim != 1 or len(subjects) != len(y):
        raise ValueError("Training subject IDs must match labels")
    if any(subject is None or not str(subject).strip() for subject in subjects):
        raise ValueError("Training subject IDs must not be empty")
    unique = np.unique(subjects)
    weights = np.zeros(len(y), dtype=float)
    for subject in unique:
        mask = subjects == subject
        labels, counts = np.unique(y[mask], return_counts=True)
        for label, count in zip(labels, counts):
            weights[mask & (y == label)] = len(y) / (len(unique) * len(labels) * count)
    return weights


def _estimator(spec, seed):
    params, family = dict(spec["model_params"]), spec["family"]
    balanced = spec["training"]["weighting"] == "class_balanced"
    if family == "sagittal_hierarchy":
        from .sagittal_hierarchy import SagittalHierarchyClassifier
        return SagittalHierarchyClassifier(family=params["base_family"],
            model_params=params["base_params"], random_state=seed)
    if family == "rbf_svm":
        return SVC(**params, kernel="rbf", probability=spec["postprocessing"]["kind"] == "argmax",
            decision_function_shape="ovr", class_weight="balanced" if balanced else None, random_state=seed)
    if family == "logistic_regression":
        params.setdefault("max_iter", 5000)
        return LogisticRegression(**params, class_weight="balanced" if balanced else None, random_state=seed)
    if family in {"random_forest", "extra_trees"}:
        cls = RandomForestClassifier if family == "random_forest" else ExtraTreesClassifier
        return cls(**params, class_weight="balanced" if balanced else None, random_state=seed, n_jobs=1)
    if family == "xgboost":
        from xgboost import XGBClassifier
        params.setdefault("objective", "multi:softprob")
        params.setdefault("eval_metric", "mlogloss")
        params.setdefault("tree_method", "hist")
        return XGBClassifier(**params, random_state=seed, n_jobs=1)
    if family == "catboost":
        from catboost import CatBoostClassifier
        params.setdefault("loss_function", "MultiClass")
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        return CatBoostClassifier(**params, auto_class_weights="Balanced" if balanced else None,
            random_seed=seed, thread_count=1)
    if family == "mlp":
        from .extended_models import SmallNeuralClassifier
        return SmallNeuralClassifier(architecture="mlp", **params, random_state=seed)
    raise ValueError(f"Unsupported estimator family: {family}")


class IterativePipeline(ClassifierMixin, BaseEstimator):
    def __init__(self, spec, seed=None):
        self.spec = spec
        self.seed = seed

    def fit(self, X, y, *, groups=None, validation_data=None):
        # No model in this module uses validation_data to fit or stop training.
        # The argument allows a common app training signature without leakage.
        self.spec_ = validate_pipeline(self.spec)
        values, labels = check_X_y(X, y)
        if values.shape[1] != len(feature_names()):
            raise ValueError("Expected original 596-feature runtime input")
        if not np.isin(labels, np.arange(len(CLASS_NAMES))).all() or len(np.unique(labels)) < 2:
            raise ValueError("Training labels must contain at least two supported posture classes")
        seed = self.spec_["training"]["seed"] if self.seed is None else self.seed
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seed override must be a nonnegative uint32 integer")
        self.effective_seed_ = seed
        if self.spec_["postprocessing"]["kind"] == "macro_f1_bias":
            self._fit_probability_bias(values, labels, groups, seed)
        elif self.spec_["family"] == "equal_soft_voting":
            self.members_ = [build_pipeline(member, seed=self.seed).fit(values, labels, groups=groups)
                for member in self.spec_["members"]]
            self.training_details_ = {"pipeline": self.spec_, "effective_seed": seed,
                "members": [member.training_details_ for member in self.members_],
                "input_contract": "feature_vector", "combination": "equal aligned probability mean"}
        else:
            mode = self.spec_["feature_engineering"]["mode"]
            feature_spec = self.spec_["feature_engineering"]
            if mode == "subject_invariant":
                features = SubjectInvariantSelector(k=feature_spec["k"], strength=feature_spec["strength"])
            elif mode == "shrinkage_lda":
                features = ShrinkageDiscriminantProjection(n_components=feature_spec["n_components"])
            elif mode in {"contrast", "all_plus_contrast"}:
                features = PressureContrastFeatures(mode)
            else:
                features = NamedFeatureSelector(mode)
            scale = (QuantileTransformer(n_quantiles=min(256, len(values)), output_distribution="normal",
                subsample=len(values), random_state=seed, copy=True)
                if self.spec_["preprocessing"]["scaler"] == "quantile_normal" else StandardScaler())
            steps = [("features", features), ("scale", scale)]
            if self.spec_["preprocessing"]["scaler"] == "shrinkage_whiten":
                steps.append(("whiten", ShrinkageWhitening()))
            steps.append(("model", _estimator(self.spec_, seed)))
            self.estimator_ = Pipeline(steps)
            options = {"features__groups": groups} if mode == "subject_invariant" else {}
            weights = None
            if self.spec_["training"]["weighting"] == "subject_class_balanced":
                weights = _subject_class_weights(labels, groups)
                options.update(scale__sample_weight=weights, model__sample_weight=weights)
            elif self.spec_["family"] == "xgboost":
                options["model__sample_weight"] = compute_sample_weight("balanced", labels)
            elif self.spec_["family"] == "mlp":
                # External validation must not choose a neural epoch in this
                # controlled comparison. The fixed budget uses training loss.
                options["model__validation_data"] = None
            self.estimator_.fit(values, labels, **options)
            self.training_details_ = {"pipeline": self.spec_, "effective_seed": seed,
                "selected_feature_names": self.estimator_.named_steps["features"].get_feature_names_out().tolist(),
                "input_contract": "feature_vector", "preprocessing_fit": "training subjects only"}
            if self.spec_["family"] == "mlp":
                self.training_details_["neural_training"] = deepcopy(self.estimator_.named_steps["model"].training_details_)
                self.training_details_["neural_normalization"] = {
                    "outer": "StandardScaler fitted on training features only",
                    "inner": "SmallNeuralClassifier training mean/std after outer scaling, float32; both serialized",
                    "epoch_policy": "Run fixed epochs; keep minimum training-loss weights; no external validation or patience stopping",
                    "loss": "class-balanced CrossEntropyLoss", "raw_sensor_augmentation": False}
            if self.spec_["family"] == "sagittal_hierarchy":
                self.training_details_["hierarchy_training"] = deepcopy(self.estimator_.named_steps["model"].training_details_)
            if self.spec_["preprocessing"]["scaler"] == "quantile_normal":
                self.training_details_["quantile_scaling"] = {"n_quantiles": scale.n_quantiles_,
                    "output_distribution": "normal", "subsample": len(values), "random_state": seed,
                    "evaluation_outliers": "clipped to fitted train quantile range by sklearn QuantileTransformer"}
            if mode == "subject_invariant":
                self.training_details_["feature_learning"] = {"mode": mode, "k": features.k, "strength": features.strength,
                    "training_samples": features.n_samples_seen_, "training_subjects": features.n_subjects_seen_,
                    "training_classes": features.n_classes_seen_, "selected_indices": features.indices_.tolist(),
                    "ranking": "unweighted training F_posture / (1 + strength * F_subject); finite cap 1e12; no p-value claim"}
            elif mode == "shrinkage_lda":
                self.training_details_["feature_learning"] = {"mode": mode, "n_components": features.n_components_,
                    "training_samples": features.n_samples_seen_, "retained_input_features": len(features.retained_indices_),
                    "algorithm": "training StandardScaler; remove constant columns; LDA(solver=eigen,shrinkage=auto); downstream StandardScaler on projected axes"}
            if weights is not None:
                self.training_details_["weighting"] = {"minimum": float(weights.min()), "maximum": float(weights.max()),
                    "subject_count": int(len(np.unique(groups))), "mean": float(weights.mean())}
        self.classes_ = np.arange(len(CLASS_NAMES))
        self.n_features_in_ = values.shape[1]
        return self

    def _fit_probability_bias(self, values, labels, groups, seed):
        from .group_probability_bias import GroupProbabilityBias
        subjects = np.asarray(groups)
        config = self.spec_["postprocessing"]
        if groups is None or subjects.ndim != 1 or len(subjects) != len(labels):
            raise ValueError("Group-held-out bias fitting requires training subject IDs")
        if any(v is None or not str(v).strip() or isinstance(v, (float, np.floating)) and not np.isfinite(v) for v in subjects):
            raise ValueError("Training subject IDs must be nonempty and finite")
        if len(np.unique(subjects)) < config["inner_folds"]:
            raise ValueError("Not enough training subjects for the fixed inner folds")
        base_spec = deepcopy(self.spec_)
        base_spec["postprocessing"] = {"kind": "argmax"}
        probability = np.zeros((len(labels), len(CLASS_NAMES)))
        seen = np.zeros(len(labels), dtype=int)
        folds = []
        for train, held in GroupKFold(config["inner_folds"]).split(values, labels, subjects):
            if set(labels[train]) != set(labels):
                raise ValueError("An inner training fold is missing a posture class")
            model = build_pipeline(base_spec, seed=seed).fit(values[train], labels[train], groups=subjects[train])
            probability[held] = model.predict_proba(values[held])
            seen[held] += 1
            folds.append({"training_subjects": sorted(map(str, set(subjects[train]))),
                "held_out_subjects": sorted(map(str, set(subjects[held]))),
                "training_windows": len(train), "held_out_windows": len(held)})
        if not np.all(seen == 1):
            raise ValueError("Every training window needs exactly one out-of-fold prediction")
        self.probability_bias_ = GroupProbabilityBias(regularization=config["regularization"],
            max_sweeps=config["max_sweeps"]).fit(probability, labels, groups=subjects)
        self.estimator_ = build_pipeline(base_spec, seed=seed).fit(values, labels, groups=subjects)
        self.training_details_ = {"pipeline": self.spec_, "effective_seed": seed,
            "input_contract": "feature_vector", "preprocessing_fit": "inner training subjects only, then all training subjects",
            "base_training": self.estimator_.training_details_, "bias_training": self.probability_bias_.training_details_,
            "inner_group_folds": folds, "out_of_fold_windows": len(labels),
            "postprocessing_output": "renormalized class decision scores; not calibrated confidence"}

    @property
    def named_steps(self):
        check_is_fitted(self, "estimator_")
        return self.estimator_.named_steps

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        values = check_array(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Feature input differs from training contract")
        if self.spec_["postprocessing"]["kind"] == "macro_f1_bias":
            return self.probability_bias_.transform(self.estimator_.predict_proba(values))
        if self.spec_["family"] == "equal_soft_voting":
            return np.mean([member.predict_proba(values) for member in self.members_], axis=0)
        classes = np.asarray(self.estimator_.classes_, dtype=int)
        if self.spec_["postprocessing"]["kind"] == "decision_softmax":
            decision = self.estimator_.decision_function(values)
            if decision.ndim == 1:
                decision = np.column_stack((-decision, decision))
            logits = decision / self.spec_["postprocessing"]["temperature"]
            logits -= logits.max(axis=1, keepdims=True)
            raw = np.exp(logits)
            raw /= raw.sum(axis=1, keepdims=True)
        else:
            raw = self.estimator_.predict_proba(values)
        result = np.zeros((len(values), len(CLASS_NAMES)), dtype=float)
        result[:, classes] = raw
        return result

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def build_pipeline(spec: dict, seed: int | None = None):
    return IterativePipeline(validate_pipeline(spec), seed=seed)


def fit_pipeline(model, spec: dict, X, y, *, groups=None, validation_data=None):
    normalized = validate_pipeline(spec)
    if not isinstance(model, IterativePipeline) or validate_pipeline(model.spec) != normalized:
        raise ValueError("Training specification differs from the constructed pipeline")
    return model.fit(X, y, groups=groups, validation_data=validation_data)
