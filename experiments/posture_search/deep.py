"""Small neural hypotheses with a train-only, fixed-epoch evaluation contract.

Inputs are nonnegative raw sensor windows ``(N, frames, 16)``. Every model is
initialized from scratch. No validation/test arrays or labels are accepted by
``fit``. Normalization, class weights, and the optional SVM are fitted using
only the supplied training partition. The caller owns subject/session splitting.

The spatial model tests row-major 4x4 adjacency as a modeling assumption; it
does not claim the wiring has been physically validated. Hybrid probabilities
are normalized decision scores, not calibrated probability estimates.
"""

from __future__ import annotations

import platform
import time
from typing import Any

import numpy as np
import torch
from torch import nn


MODEL_VERSION = "posture-search-deep-v1"
SUPPORTED_KINDS = (
    "cnn1d", "cnn2d", "gru", "lstm", "cnn_lstm", "attention",
    "transformer", "tcn", "cnn_svm",
)
_ALIASES = {
    "1dcnn": "cnn1d", "1d_cnn": "cnn1d", "cnn_1d": "cnn1d",
    "2dcnn": "cnn2d", "2d_cnn": "cnn2d", "cnn_2d": "cnn2d",
    "spatial2dcnn": "cnn2d", "self_attention": "attention",
    "multiscale_tcn": "tcn", "hybrid": "cnn_svm", "hybrid_svm": "cnn_svm",
}


class _DilatedBranch(nn.Module):
    def __init__(self, dilation: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Conv1d(48, 24, 1)
        self.convolutions = nn.Sequential(
            nn.Conv1d(24, 24, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(24, 24, 3, padding=dilation, dilation=dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.projection(x)
        return nn.functional.gelu(h + self.convolutions(h))


class _PostureNetwork(nn.Module):
    """Independent small encoders; all expose a 64-dimensional embedding."""

    def __init__(self, kind: str, frames: int, dropout: float) -> None:
        super().__init__()
        self.kind = "cnn1d" if kind == "cnn_svm" else kind
        if self.kind == "cnn1d":
            self.encoder = nn.Sequential(
                nn.Conv1d(48, 32, 5, padding=2), nn.GELU(),
                nn.Conv1d(32, 48, 3, padding=1), nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(48, 48, 3, padding=1), nn.GELU(),
            )
            self.pool = nn.Sequential(nn.Linear(96, 64), nn.GELU())
        elif self.kind == "cnn2d":
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.GELU(),
                nn.Conv2d(16, 24, 3, padding=1), nn.GELU(),
                nn.Flatten(), nn.Linear(24 * 4 * 4, 64), nn.GELU(),
            )
        elif self.kind in ("gru", "lstm", "cnn_lstm"):
            if self.kind == "cnn_lstm":
                self.convolution = nn.Sequential(
                    nn.Conv1d(48, 32, 5, padding=2), nn.GELU(),
                    nn.Dropout(dropout),
                )
            recurrence = nn.GRU if self.kind == "gru" else nn.LSTM
            self.encoder = recurrence(
                input_size=32 if self.kind == "cnn_lstm" else 48,
                hidden_size=48, num_layers=1, batch_first=True,
            )
            self.pool = nn.Sequential(nn.Linear(96, 64), nn.GELU())
        elif self.kind == "attention":
            self.projection = nn.Linear(48, 64)
            self.attention = nn.MultiheadAttention(
                embed_dim=64, num_heads=4, dropout=dropout, batch_first=True,
            )
            self.normalization = nn.LayerNorm(64)
            self.pool = nn.Sequential(nn.Linear(128, 64), nn.GELU())
        elif self.kind == "transformer":
            self.projection = nn.Linear(48, 48)
            self.position = nn.Parameter(torch.empty(1, frames, 48))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=48, nhead=4, dim_feedforward=96, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=2, enable_nested_tensor=False,
            )
            self.pool = nn.Sequential(nn.LayerNorm(48), nn.Linear(48, 64), nn.GELU())
        elif self.kind == "tcn":
            self.branches = nn.ModuleList([_DilatedBranch(d, dropout) for d in (1, 2, 4)])
            self.pool = nn.Sequential(nn.Linear(144, 64), nn.GELU())
        else:
            raise ValueError(f"Unsupported neural architecture: {kind!r}")
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, 7))

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "cnn2d":
            n, frames, _ = x.shape
            h = self.encoder(x.reshape(n * frames, 3, 4, 4))
            return h.reshape(n, frames, 64).mean(dim=1)
        if self.kind in ("gru", "lstm", "cnn_lstm"):
            if self.kind == "cnn_lstm":
                x = self.convolution(x.transpose(1, 2)).transpose(1, 2)
            h, _ = self.encoder(x)
            return self.pool(torch.cat((h.mean(dim=1), h[:, -1]), dim=1))
        if self.kind == "attention":
            x = self.projection(x)
            attended, _ = self.attention(x, x, x, need_weights=False)
            h = self.normalization(x + attended)
            return self.pool(torch.cat((h.mean(dim=1), h.amax(dim=1)), dim=1))
        if self.kind == "transformer":
            h = self.encoder(self.projection(x) + self.position)
            return self.pool(h.mean(dim=1))
        x = x.transpose(1, 2)
        if self.kind == "tcn":
            pooled = []
            for branch in self.branches:
                h = branch(x)
                pooled.extend((h.mean(dim=2), h.amax(dim=2)))
            return self.pool(torch.cat(pooled, dim=1))
        h = self.encoder(x)
        return self.pool(torch.cat((h.mean(dim=2), h.amax(dim=2)), dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedding(x))


class DeepEstimator:
    """Fixed training recipe with deterministic CPU operations and no early stop.

    ``groups`` is checked and recorded only. The fixed-epoch procedure never
    creates a random window-level validation split. Use ``epochs=1`` for fast
    synthetic contract checks; actual experiments use the default 45 epochs.
    """

    def __init__(self, kind: str, seed: int = 42, epochs: int = 45,
                 batch_size: int = 64, learning_rate: float = 0.001,
                 weight_decay: float = 0.01, dropout: float = 0.15,
                 svm_c: float = 3.0) -> None:
        canonical_kind = _ALIASES.get(kind, kind)
        if canonical_kind not in SUPPORTED_KINDS:
            raise ValueError(f"Unknown kind {kind!r}; expected one of {SUPPORTED_KINDS}")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if learning_rate <= 0 or weight_decay < 0 or not 0 <= dropout < 1 or svm_c <= 0:
            raise ValueError("Invalid optimizer, dropout, or SVM configuration")
        self.kind = canonical_kind
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.svm_c = float(svm_c)
        self.classes_ = np.arange(7)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {key: getattr(self, key) for key in (
            "kind", "seed", "epochs", "batch_size", "learning_rate",
            "weight_decay", "dropout", "svm_c",
        )}

    @staticmethod
    def _raw(X: np.ndarray) -> np.ndarray:
        raw = np.asarray(X, dtype=np.float32)
        if raw.ndim != 3 or raw.shape[1] < 2 or raw.shape[2] != 16:
            raise ValueError("X must have shape (windows, at_least_2_frames, 16)")
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError("Raw pressure windows must be finite and nonnegative")
        return np.ascontiguousarray(raw)

    @staticmethod
    def _representation(raw: torch.Tensor) -> torch.Tensor:
        proportions = raw / raw.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.cat((raw / 1023.0, torch.log1p(raw) / np.log(1024.0), proportions), dim=-1)

    def _transform(self, raw: torch.Tensor) -> torch.Tensor:
        return (self._representation(raw) - self.normalizer_mean_) / self.normalizer_std_

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> "DeepEstimator":
        started = time.perf_counter()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        raw = torch.from_numpy(self._raw(X).copy())
        supplied_labels = np.asarray(y)
        if supplied_labels.ndim != 1 or not np.issubdtype(supplied_labels.dtype, np.integer):
            raise ValueError("y must be a one-dimensional array of integer labels")
        labels = supplied_labels.astype(np.int64, copy=False)
        if len(raw) == 0 or len(raw) != len(labels):
            raise ValueError("X and y must have the same nonzero sample count")
        if np.any(labels < 0) or np.any(labels > 6):
            raise ValueError("y labels must be in 0..6")
        counts = np.bincount(labels, minlength=7)
        if np.any(counts == 0):
            raise ValueError("Training partition must contain all seven classes")
        if groups is not None and (np.asarray(groups).ndim != 1 or len(groups) != len(raw)):
            raise ValueError("groups must have one entry per training window")
        represented = self._representation(raw)
        self.normalizer_mean_ = represented.mean(dim=(0, 1), keepdim=True)
        self.normalizer_std_ = represented.std(dim=(0, 1), keepdim=True, unbiased=False).clamp(min=1e-5)
        self.input_shape_ = list(raw.shape[1:])
        self.model_ = _PostureNetwork(self.kind, raw.shape[1], self.dropout)
        targets = torch.from_numpy(labels.copy())
        self.class_weights_ = len(labels) / (7.0 * counts)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights_, dtype=torch.float32))
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        # Transformation has no stochastic operations, so fit it once for efficiency.
        transformed = self._transform(raw)
        self.training_history_ = []
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.perf_counter()
            self.model_.train()
            total_loss = 0.0
            correct = 0
            permutation = rng.permutation(len(labels))
            for offset in range(0, len(labels), self.batch_size):
                indexes = torch.from_numpy(permutation[offset:offset + self.batch_size])
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(transformed[indexes])
                loss = criterion(logits, targets[indexes])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=2.0)
                optimizer.step()
                total_loss += loss.detach().item() * len(indexes)
                correct += int((logits.detach().argmax(dim=1) == targets[indexes]).sum())
            self.training_history_.append({
                "epoch": epoch, "train_loss": total_loss / len(labels),
                "train_online_accuracy": correct / len(labels),
                "seconds": time.perf_counter() - epoch_start,
            })
        self.model_.eval()
        self.architecture_ = str(self.model_)
        self.config_ = {
            **self.get_params(), "model_version": MODEL_VERSION,
            "architecture": self.architecture_, "input_shape": self.input_shape_,
            "device": "cpu", "threads": 1, "deterministic_algorithms": True,
            "optimizer": "AdamW", "gradient_clip_norm": 2.0,
            "loss": "inverse_train_class_frequency_weighted_cross_entropy",
            "class_weights": self.class_weights_.tolist(),
            "epoch_selection": "fixed_45_default_no_validation_or_test_access",
            "input_representation": ["raw_div_1023", "log1p_raw_div_log1024", "per_frame_sensor_proportions"],
            "normalization": "training_only_channel_mean_and_population_std_min_1e-5",
            "spatial_geometry": "assumed_row_major_4x4_not_physically_verified",
            "training_sample_count": len(labels), "training_class_counts": counts.tolist(),
            "training_groups": sorted({str(group) for group in groups}) if groups is not None else None,
            "group_usage": "audit_only_no_internal_split",
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "platform": platform.platform(),
        }
        self.complexity_ = {
            "trainable_parameters": sum(p.numel() for p in self.model_.parameters() if p.requires_grad),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in self.model_.parameters()),
            "embedding_dimensions": 64, "input_shape": self.input_shape_,
            "epochs_trained": len(self.training_history_), "model_version": MODEL_VERSION,
        }
        if self.kind == "cnn_svm":
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC

            self.embedding_scaler_ = StandardScaler()
            features = self.embedding_scaler_.fit_transform(self.embedding(X))
            self.svm_ = SVC(C=self.svm_c, gamma="scale", kernel="rbf", class_weight="balanced",
                            probability=False, decision_function_shape="ovr", random_state=self.seed)
            self.svm_.fit(features, labels)
            self.config_["hybrid"] = {
                "encoder": "cnn1d_supervised_45_epoch_default", "embedding_scaler": "train_only_StandardScaler",
                "svm": self.svm_.get_params(),
                "probability_method": "uncalibrated_softmax_of_ovr_decision_scores_temperature_1",
            }
            self.complexity_["support_vector_count"] = int(self.svm_.n_support_.sum())
            self.complexity_["support_vector_bytes"] = int(self.svm_.support_vectors_.nbytes)
        self.fit_seconds_ = time.perf_counter() - started
        self.complexity_["fit_seconds"] = self.fit_seconds_
        return self

    def _output(self, X: np.ndarray, embedding: bool) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before prediction")
        torch.set_num_threads(1)
        raw = torch.from_numpy(self._raw(X))
        if list(raw.shape[1:]) != self.input_shape_:
            raise ValueError(f"Expected fitted window shape {self.input_shape_}, got {list(raw.shape[1:])}")
        self.model_.eval()
        results = []
        with torch.inference_mode():
            for offset in range(0, len(raw), self.batch_size):
                transformed = self._transform(raw[offset:offset + self.batch_size])
                output = (self.model_.embedding(transformed) if embedding else
                          torch.softmax(self.model_(transformed), dim=1))
                results.append(output.numpy())
        return np.concatenate(results) if results else np.empty((0, 64 if embedding else 7), dtype=np.float32)

    def embedding(self, X: np.ndarray) -> np.ndarray:
        return self._output(X, embedding=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.kind != "cnn_svm":
            return self._output(X, embedding=False)
        features = self.embedding(X)
        if not len(features):
            return np.empty((0, 7), dtype=np.float64)
        scores = self.svm_.decision_function(self.embedding_scaler_.transform(features))
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def build_deep(kind: str, seed: int = 42, **kwargs: Any) -> DeepEstimator:
    """Build an unfitted estimator; keyword overrides are recorded in config_."""
    return DeepEstimator(kind=kind, seed=seed, **kwargs)
