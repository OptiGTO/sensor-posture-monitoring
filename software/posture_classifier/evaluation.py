"""Leakage-safe evaluation utilities for the posture classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data import CANONICAL_LABELS, Recording
from .features import FeatureConfig, build_feature_dataset
from .model import SVMConfig, make_pipeline


@dataclass(frozen=True)
class FoldResult:
    held_out_subject: str
    window_count: int
    accuracy: float
    macro_f1: float


@dataclass(frozen=True)
class EvaluationResult:
    labels: tuple[str, ...]
    accuracy: float
    macro_f1: float
    confusion_matrix: np.ndarray
    folds: tuple[FoldResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": list(self.labels),
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "folds": [asdict(fold) for fold in self.folds],
        }


def evaluate_leave_one_subject_out(
    recordings: Iterable[Recording],
    *,
    feature_config: FeatureConfig | None = None,
    svm_config: SVMConfig | None = None,
) -> EvaluationResult:
    """Evaluate new-person generalization by holding out one subject per fold."""

    dataset = build_feature_dataset(recordings, feature_config)
    subjects = tuple(sorted(np.unique(dataset.subjects)))
    if len(subjects) < 2:
        raise ValueError("Leave-One-Subject-Out evaluation requires at least two subjects")

    observed_labels = set(dataset.labels)
    labels = tuple(label for label in CANONICAL_LABELS if label in observed_labels)
    # ``dtype=str`` without an explicit width produces one-character unicode
    # storage, which silently corrupts Korean class names. Object dtype preserves
    # the complete predicted label in every fold.
    predictions = np.empty(dataset.labels.shape, dtype=object)
    folds: list[FoldResult] = []
    config = svm_config or SVMConfig()

    for subject in subjects:
        test_mask = dataset.subjects == subject
        train_mask = ~test_mask
        pipeline = make_pipeline(config)
        pipeline.fit(dataset.features[train_mask], dataset.labels[train_mask])
        fold_predictions = pipeline.predict(dataset.features[test_mask])
        predictions[test_mask] = fold_predictions
        folds.append(
            FoldResult(
                held_out_subject=subject,
                window_count=int(test_mask.sum()),
                accuracy=float(accuracy_score(dataset.labels[test_mask], fold_predictions)),
                macro_f1=float(
                    f1_score(dataset.labels[test_mask], fold_predictions, labels=labels, average="macro", zero_division=0)
                ),
            )
        )

    return EvaluationResult(
        labels=labels,
        accuracy=float(accuracy_score(dataset.labels, predictions)),
        macro_f1=float(f1_score(dataset.labels, predictions, labels=labels, average="macro", zero_division=0)),
        confusion_matrix=confusion_matrix(dataset.labels, predictions, labels=labels),
        folds=tuple(folds),
    )
