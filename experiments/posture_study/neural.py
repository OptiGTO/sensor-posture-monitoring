"""Small, reproducible neural estimators for the isolated posture study.

No production modules are modified or imported. The sequence input is a raw ADC
array of shape (windows, frames, 16); physical front/back orientation is never
assumed. Spatial CNNs use only the documented row-major 4 by 4 sensor adjacency.
All normalizers are fitted on training data. Validation labels select an epoch
only; test data must never be passed to ``fit``.

Objects, including their CPU Torch state, can be persisted with ``joblib.dump``.
"""

from __future__ import annotations

import copy
import platform
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


ARCHITECTURE_VERSION = "posture-study-neural-v1.1-unit-aware-augmentation"
SUPPORTED_KINDS = (
    "1dcnn", "spatial2dcnn", "lstm", "gru", "cnn_lstm", "attention",
    "transformer", "tcn", "mlp",
)


class _TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class _Network(nn.Module):
    """Architectures written for this experiment, with no pretrained weights."""

    def __init__(self, kind: str, input_dim: int, frames: int, classes: int,
                 dropout: float) -> None:
        super().__init__()
        self.kind = kind
        self.embedding_dim = 64
        self.pool_project: nn.Module = nn.Identity()
        if kind == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(128, 64), nn.GELU(),
            )
        elif kind == "1dcnn":
            self.encoder = nn.Sequential(
                nn.Conv1d(32, 48, 5, padding=2), nn.GELU(),
                nn.Conv1d(48, 64, 3, padding=1), nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(64, 64, 3, padding=1), nn.GELU(),
            )
            self.pool_project = nn.Sequential(nn.Linear(128, 64), nn.GELU())
        elif kind == "spatial2dcnn":
            self.encoder = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1), nn.GELU(),
                nn.Conv2d(16, 32, 3, padding=1), nn.GELU(),
                nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.GELU(),
            )
        elif kind in ("lstm", "gru", "cnn_lstm"):
            if kind == "cnn_lstm":
                self.temporal_conv = nn.Sequential(
                    nn.Conv1d(32, 48, 5, padding=2), nn.GELU(),
                    nn.Dropout(dropout),
                )
            rnn_type = nn.GRU if kind == "gru" else nn.LSTM
            self.encoder = rnn_type(
                input_size=48 if kind == "cnn_lstm" else 32,
                hidden_size=48, num_layers=1, batch_first=True,
            )
            self.pool_project = nn.Sequential(nn.Linear(96, 64), nn.GELU())
        elif kind == "attention":
            self.encoder = nn.Sequential(nn.Linear(32, 64), nn.GELU())
            self.attention_score = nn.Sequential(
                nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1),
            )
        elif kind == "transformer":
            self.input_project = nn.Linear(32, 48)
            self.position = nn.Parameter(torch.zeros(1, frames, 48))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=48, nhead=4, dim_feedforward=96, dropout=dropout,
                batch_first=True, norm_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=2, enable_nested_tensor=False,
            )
            self.pool_project = nn.Sequential(
                nn.LayerNorm(48), nn.Linear(48, 64), nn.GELU(),
            )
        elif kind == "tcn":
            self.encoder = nn.Sequential(
                nn.Conv1d(32, 48, 1), nn.GELU(),
                *[_TemporalBlock(48, d, dropout) for d in (1, 2, 4)],
            )
            self.pool_project = nn.Sequential(nn.Linear(96, 64), nn.GELU())
        else:
            raise ValueError(f"Unknown architecture {kind!r}; expected {SUPPORTED_KINDS}")
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, classes))

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "mlp":
            return self.encoder(x)
        if self.kind == "spatial2dcnn":
            n, t, _ = x.shape
            # Channels are [ADC cells 0..15, proportional cells 0..15].
            h = self.encoder(x.reshape(n * t, 2, 4, 4)).reshape(n, t, 64)
            return h.mean(dim=1)
        if self.kind in ("lstm", "gru", "cnn_lstm"):
            if self.kind == "cnn_lstm":
                x = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
            h, _ = self.encoder(x)
            return self.pool_project(torch.cat((h.mean(dim=1), h[:, -1]), dim=1))
        if self.kind == "attention":
            h = self.encoder(x)
            weights = torch.softmax(self.attention_score(h), dim=1)
            return (weights * h).sum(dim=1)
        if self.kind == "transformer":
            if x.shape[1] > self.position.shape[1]:
                raise ValueError("Inference window exceeds the fitted position embedding")
            h = self.encoder(self.input_project(x) + self.position[:, :x.shape[1]])
            return self.pool_project(h.mean(dim=1))
        h = self.encoder(x.transpose(1, 2))
        if self.kind == "tcn":
            pooled = torch.cat((h.mean(dim=2), h[:, :, -1]), dim=1)
        else:
            pooled = torch.cat((h.mean(dim=2), h.amax(dim=2)), dim=1)
        return self.pool_project(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedding(x))


def _macro_f1(y: np.ndarray, predicted: np.ndarray, classes: int) -> float:
    confusion = np.bincount(y * classes + predicted, minlength=classes ** 2).reshape(classes, classes)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    per_class = np.divide(2 * np.diag(confusion), denominator,
                          out=np.zeros(classes, dtype=float), where=denominator > 0)
    return float(per_class.mean())


class NeuralEstimator:
    """Deterministic CPU estimator with explicit train-only preprocessing.

    Parameters default to one shared training recipe across architecture tests.
    Augmentation applies independent, window-constant cell gains plus small ADC
    noise to training sequences only, then recomputes pressure proportions.
    ``validation=None`` trains every requested epoch without selecting on labels
    outside the training partition. ``validation=(Xdict, y)`` enables early
    stopping on validation Macro F1 (ties use validation cross-entropy).
    """

    def __init__(self, kind: str, seed: int = 42, epochs: int = 60,
                 patience: int = 10, batch_size: int = 64,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-3,
                 dropout: float = 0.15, augmentation: bool = False,
                 gain_std: float = 0.10, noise_std: float = 0.003,
                 class_weight: bool = True, num_classes: int = 7,
                 threads: int = 2, feature_key: str = "features") -> None:
        if kind not in SUPPORTED_KINDS:
            raise ValueError(f"Unknown architecture {kind!r}; expected {SUPPORTED_KINDS}")
        if epochs < 1 or patience < 1 or batch_size < 1:
            raise ValueError("epochs, patience, and batch_size must be positive")
        self.kind = kind
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.augmentation = bool(augmentation)
        self.gain_std = float(gain_std)
        self.noise_std = float(noise_std)
        self.class_weight = bool(class_weight)
        self.num_classes = int(num_classes)
        self.threads = int(threads)
        self.feature_key = feature_key
        self.training_history: list[dict[str, Any]] = []

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (
            "kind", "seed", "epochs", "patience", "batch_size", "learning_rate",
            "weight_decay", "dropout", "augmentation", "gain_std", "noise_std",
            "class_weight", "num_classes", "threads", "feature_key",
        )}

    def get_config(self) -> dict[str, Any]:
        return {
            **self.get_params(), "architecture_version": ARCHITECTURE_VERSION,
            "optimizer": "AdamW", "gradient_clip_norm": 2.0,
            "selection": "validation_macro_f1_then_validation_loss",
            "device": "cpu", "deterministic_algorithms": True,
            "input_representation": ("train_zscored_features" if self.kind == "mlp"
                                     else "raw_ADC_div_1023_and_per_frame_cell_proportions"),
            "normalization": "training_partition_channel_mean_and_population_std_min_1e-5",
            "sequence_geometry": "row_major_4x4; physical_orientation_unspecified",
            "raw_units": "recorded sensor units; 1023 is a fixed divisor cancelled by train z-scoring, not an asserted ADC range",
            "augmentation_noise_basis": "noise_std times training-only per-cell raw population standard deviation; clamp nonnegative only",
        }

    def _raw(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        key = self.feature_key if self.kind == "mlp" else "seq"
        values = np.asarray(X[key], dtype=np.float32)
        if self.kind == "mlp":
            if values.ndim != 2:
                raise ValueError("MLP features must have shape (windows, features)")
        elif values.ndim != 3 or values.shape[2] != 16 or values.shape[1] < 2:
            raise ValueError("Sequences must have shape (windows, at_least_2_frames, 16)")
        if not np.isfinite(values).all():
            raise ValueError("Neural input must contain finite values only")
        if self.kind != "mlp" and np.any(values < 0):
            raise ValueError("Pressure ADC values must be nonnegative")
        return values

    def _representation(self, raw: torch.Tensor) -> torch.Tensor:
        if self.kind == "mlp":
            return raw
        proportions = raw / raw.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.cat((raw / 1023.0, proportions), dim=-1)

    def _transform(self, raw: torch.Tensor, augment: bool = False) -> torch.Tensor:
        if augment and self.kind != "mlp":
            gains = 1.0 + self.gain_std * torch.randn(raw.shape[0], 1, 16)
            raw = (raw * gains + self.noise_std * self.augmentation_noise_scale_ * torch.randn_like(raw)).clamp(min=0)
        represented = self._representation(raw)
        return (represented - self.normalizer_mean_) / self.normalizer_std_

    def fit(self, X: Mapping[str, np.ndarray], y: np.ndarray,
            validation: tuple[Mapping[str, np.ndarray], np.ndarray] | None = None) -> "NeuralEstimator":
        started = time.perf_counter()
        torch.set_num_threads(self.threads)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        raw = torch.from_numpy(self._raw(X).copy())
        self.augmentation_noise_scale_ = raw.std(dim=(0,1),keepdim=True,unbiased=False).clamp(min=1e-8) if self.kind!='mlp' else None
        labels = np.asarray(y, dtype=np.int64)
        if len(raw) != len(labels) or len(labels) == 0:
            raise ValueError("Training inputs and labels must have the same nonzero length")
        if labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError("Class labels must be zero-based integers within num_classes")
        label_tensor = torch.from_numpy(labels.copy())
        represented = self._representation(raw)
        axes = (0,) if self.kind == "mlp" else (0, 1)
        self.normalizer_mean_ = represented.mean(dim=axes, keepdim=True)
        self.normalizer_std_ = represented.std(dim=axes, keepdim=True, unbiased=False).clamp(min=1e-5)
        self.input_shape_ = list(raw.shape[1:])
        self.model_ = _Network(self.kind, raw.shape[-1],
                               raw.shape[1] if self.kind != "mlp" else 1,
                               self.num_classes, self.dropout)
        counts = np.bincount(labels, minlength=self.num_classes)
        if np.any(counts == 0):
            raise ValueError("Training partition must contain every configured class")
        weights = len(labels) / (self.num_classes * counts)
        self.class_weights_ = weights.tolist() if self.class_weight else None
        loss_function = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32) if self.class_weight else None,
        )
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        validation_raw = validation_labels = None
        if validation is not None:
            validation_raw = torch.from_numpy(self._raw(validation[0]).copy())
            validation_labels = np.asarray(validation[1], dtype=np.int64)
            if len(validation_raw) == 0 or len(validation_raw) != len(validation_labels):
                raise ValueError("Validation inputs and labels must have the same nonzero length")
            if validation_labels.min() < 0 or validation_labels.max() >= self.num_classes:
                raise ValueError("Validation labels are outside num_classes")
        self.training_history = []
        self.best_epoch_ = 0
        self.best_validation_macro_f1_ = None
        best_score, best_loss, stale = -float("inf"), float("inf"), 0
        best_state = None
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.perf_counter()
            self.model_.train()
            train_loss = 0.0
            permutation = rng.permutation(len(labels))
            for start in range(0, len(labels), self.batch_size):
                indexes = torch.from_numpy(permutation[start:start + self.batch_size])
                inputs = self._transform(raw[indexes], augment=self.augmentation)
                targets = label_tensor[indexes]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(self.model_(inputs), targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=2.0)
                optimizer.step()
                train_loss += float(loss.detach()) * len(indexes)
            row: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss / len(labels)}
            if validation_raw is not None:
                self.model_.eval()
                probability_batches = []
                validation_loss = 0.0
                with torch.inference_mode():
                    for start in range(0, len(validation_raw), self.batch_size):
                        logits = self.model_(self._transform(validation_raw[start:start + self.batch_size]))
                        targets = torch.from_numpy(validation_labels[start:start + self.batch_size])
                        # Unweighted CE is used only to break identical Macro F1 ties.
                        validation_loss += float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
                        probability_batches.append(logits.argmax(dim=1).numpy())
                score = _macro_f1(validation_labels, np.concatenate(probability_batches), self.num_classes)
                validation_loss /= len(validation_labels)
                row.update(validation_macro_f1=score, validation_loss=validation_loss)
                improved = (score > best_score + 1e-10 or
                            (abs(score - best_score) <= 1e-10 and validation_loss < best_loss - 1e-8))
                if improved:
                    best_score, best_loss, stale = score, validation_loss, 0
                    self.best_epoch_ = epoch
                    self.best_validation_macro_f1_ = score
                    best_state = copy.deepcopy(self.model_.state_dict())
                else:
                    stale += 1
            else:
                self.best_epoch_ = epoch
            row["seconds"] = time.perf_counter() - epoch_start
            self.training_history.append(row)
            if validation_raw is not None and stale >= self.patience:
                break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.classes_ = np.arange(self.num_classes)
        self.fit_seconds_ = time.perf_counter() - started
        self.epochs_trained_ = len(self.training_history)
        self.architecture_ = str(self.model_)
        self.environment_ = {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "device": "cpu", "torch_num_threads": torch.get_num_threads(),
        }
        self.complexity_ = {
            "trainable_parameters": sum(p.numel() for p in self.model_.parameters() if p.requires_grad),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in self.model_.parameters()),
            "embedding_dimensions": 64, "input_shape": self.input_shape_,
            "epochs_trained": self.epochs_trained_, "best_epoch": self.best_epoch_,
            "fit_seconds": self.fit_seconds_, "architecture_version": ARCHITECTURE_VERSION,
        }
        return self

    def _output(self, X: Mapping[str, np.ndarray], embedding: bool) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit before prediction")
        raw = torch.from_numpy(self._raw(X))
        self.model_.eval()
        results = []
        with torch.inference_mode():
            for start in range(0, len(raw), self.batch_size):
                inputs = self._transform(raw[start:start + self.batch_size])
                output = self.model_.embedding(inputs) if embedding else torch.softmax(self.model_(inputs), dim=1)
                results.append(output.numpy())
        columns = 64 if embedding else self.num_classes
        return np.concatenate(results) if results else np.empty((0, columns), dtype=np.float32)

    def predict_proba(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        return self._output(X, embedding=False)

    def predict(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def embedding(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        return self._output(X, embedding=True)

    @property
    def complexity(self) -> dict[str, Any]:
        return dict(self.complexity_)
