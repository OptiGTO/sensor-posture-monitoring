"""RBF-SVM model and prediction smoothing for posture classification."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import feature_names


@dataclass(frozen=True)
class SVMConfig:
    """Hyperparameters for the standardized RBF-SVM classifier."""

    c: float = 5.0
    gamma: str | float = "scale"
    class_weight: str | dict[str, float] | None = "balanced"
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.c <= 0:
            raise ValueError("c must be positive")


def make_pipeline(config: SVMConfig) -> Pipeline:
    """Create the exact preprocessing + classifier pipeline used in every fold."""

    base_svm = SVC(
        C=config.c,
        gamma=config.gamma,
        kernel="rbf",
        class_weight=config.class_weight,
        random_state=config.random_state,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                CalibratedClassifierCV(
                    estimator=base_svm,
                    method="sigmoid",
                    cv=3,
                    ensemble=False,
                ),
            ),
        ]
    )


class PostureSVM:
    """A serializable standardized RBF-SVM with feature-schema validation."""

    def __init__(self, config: SVMConfig | None = None) -> None:
        self.config = config or SVMConfig()
        self.pipeline = make_pipeline(self.config)
        self.feature_names = feature_names()
        self.classes_: tuple[str, ...] | None = None

    def fit(self, features: np.ndarray, labels: Iterable[str]) -> "PostureSVM":
        features = np.asarray(features, dtype=float)
        labels = np.asarray(list(labels), dtype=str)
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError(f"features must have shape (samples, {len(self.feature_names)})")
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same number of rows")
        if len(np.unique(labels)) < 2:
            raise ValueError("at least two posture labels are required")
        if not np.isfinite(features).all():
            raise ValueError("features must be finite")

        self.pipeline.fit(features, labels)
        self.classes_ = tuple(str(label) for label in self.pipeline.classes_)
        return self

    def _require_fitted(self) -> None:
        if self.classes_ is None:
            raise RuntimeError("The model must be fitted before prediction or saving")

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self.pipeline.predict(np.asarray(features, dtype=float))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self.pipeline.predict_proba(np.asarray(features, dtype=float))

    def save(self, path: Path) -> None:
        """Save the fitted model, preprocessing parameters, and feature schema."""

        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "PostureSVM":
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return loaded


class ProbabilitySmoother:
    """Average recent model probabilities to make live predictions less jumpy."""

    def __init__(self, labels: Iterable[str], history_size: int = 5) -> None:
        self.labels = tuple(labels)
        if not self.labels:
            raise ValueError("at least one label is required")
        if history_size < 1:
            raise ValueError("history_size must be positive")
        self._history: deque[np.ndarray] = deque(maxlen=history_size)

    def update(self, probabilities: np.ndarray) -> tuple[str, np.ndarray]:
        """Add one prediction and return the smoothed label and probabilities."""

        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(self.labels),):
            raise ValueError(f"probabilities must have shape ({len(self.labels)},)")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError("probabilities must be finite and non-negative")
        total = probabilities.sum()
        if total <= 0:
            raise ValueError("probabilities must sum to a positive value")

        self._history.append(probabilities / total)
        averaged = np.mean(np.stack(tuple(self._history)), axis=0)
        return self.labels[int(np.argmax(averaged))], averaged
