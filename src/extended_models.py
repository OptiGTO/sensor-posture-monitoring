"""Serializable estimators for the shared training and window inference path.

Optional native libraries are imported only when their model is selected.
Neural checkpoints contain NumPy weights; joblib never needs an experiment module.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _finite_input(values, ndim):
    data = np.asarray(values, dtype=np.float32)
    if data.ndim != ndim or not len(data) or not np.isfinite(data).all():
        raise ValueError(f"Expected nonempty finite {ndim}-dimensional model input")
    return data


class MiniRocketClassifier(ClassifierMixin, BaseEstimator):
    """Standard multivariate MiniRocket with train-fitted logistic readout."""

    def __init__(self, num_kernels=1008, C=1.0, random_state=0):
        self.num_kernels, self.C, self.random_state = num_kernels, C, random_state

    def _sequence(self, X):
        values = _finite_input(X, 3)
        if values.shape[1] < 9 or values.shape[2] != 16:
            raise ValueError("MiniRocket requires at least 9 time samples and 16 sensors")
        if hasattr(self, "input_shape_") and values.shape[1:] != self.input_shape_:
            raise ValueError("MiniRocket input window differs from its training window")
        return np.ascontiguousarray(values.transpose(0, 2, 1))

    def fit(self, X, y):
        from numba import set_num_threads
        from ._vendor.minirocket_multi import _fit_multi, _transform_multi
        set_num_threads(1)
        values = self._sequence(X)
        self.input_shape_ = tuple(np.asarray(X).shape[1:])
        state = np.random.get_state()
        try:
            self.parameters_ = _fit_multi(values, self.num_kernels, 32, self.random_state)
        finally:
            np.random.set_state(state)
        features = _transform_multi(values, self.parameters_)
        self.readout_ = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced", random_state=self.random_state)),
        ]).fit(features, y)
        self.classes_ = self.readout_.classes_
        self.n_features_in_ = 16
        self.training_details_ = {"implementation": "sktime 0.40.1 MiniRocketMultivariate kernel",
                                  "transform_features": int(features.shape[1]), "readout": "LogisticRegression"}
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "readout_")
        from numba import set_num_threads
        from ._vendor.minirocket_multi import _transform_multi
        set_num_threads(1)
        return self.readout_.predict_proba(_transform_multi(self._sequence(X), self.parameters_))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class SmallNeuralClassifier(ClassifierMixin, BaseEstimator):
    """CPU MLP/CNN with normalization fit on training subjects only.

    Validation is supplied by the caller's subject split, never a random window
    split. Its labels select an epoch only and never enter gradient updates.
    """

    def __init__(self, architecture="cnn1d", hidden_size=32, epochs=40, patience=8,
                 batch_size=64, learning_rate=0.001, weight_decay=0.01,
                 dropout=0.2, random_state=0, gain_std=0.0, noise_fraction=0.0):
        self.architecture, self.hidden_size, self.epochs = architecture, hidden_size, epochs
        self.patience, self.batch_size, self.learning_rate = patience, batch_size, learning_rate
        self.weight_decay, self.dropout, self.random_state = weight_decay, dropout, random_state
        self.gain_std, self.noise_fraction = gain_std, noise_fraction

    def _input(self, X):
        if self.architecture not in {"cnn1d", "mlp"}:
            raise ValueError("Unknown neural architecture")
        values = _finite_input(X, 3 if self.architecture == "cnn1d" else 2)
        if self.architecture == "cnn1d" and values.shape[2] != 16:
            raise ValueError("CNN requires 16 sensor channels")
        if hasattr(self, "input_shape_") and values.shape[1:] != self.input_shape_:
            raise ValueError("Neural input shape differs from its training contract")
        return values

    def _build_network(self):
        from torch import nn
        width = int(self.hidden_size)
        count = len(self.classes_)
        if self.architecture == "cnn1d":
            return nn.Sequential(
                nn.Conv1d(16, width, 3, padding=1), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Conv1d(width, width * 2, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width * 2, count),
            )
        return nn.Sequential(
            nn.Linear(self.input_shape_[0], width * 2), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(width * 2, width), nn.ReLU(), nn.Linear(width, count),
        )

    def _tensor(self, X):
        import torch
        values = (self._input(X) - self.mean_) / self.scale_
        if self.architecture == "cnn1d":
            values = values.transpose(0, 2, 1)
        return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))

    def _augment_training_values(self, values, generator):
        """Perturb a training batch in raw sensor space; never used for scoring."""
        augmented = np.asarray(values, dtype=np.float32).copy()
        if self.gain_std:
            gain = generator.normal(1.0, self.gain_std, (len(augmented), 1, augmented.shape[-1]))
            augmented *= np.clip(gain, 0.5, 1.5).astype(np.float32)
        if self.noise_fraction:
            augmented += (generator.normal(size=augmented.shape)
                * self.augmentation_noise_scale_ * self.noise_fraction).astype(np.float32)
        # ADC pressure measurements cannot become negative.
        return np.maximum(augmented, 0)

    def fit(self, X, y, *, validation_data=None):
        import torch
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("Epochs, patience and batch size must be positive")
        augmentation = bool(self.gain_std or self.noise_fraction)
        if (not np.isfinite(self.gain_std) or not np.isfinite(self.noise_fraction)
                or self.gain_std < 0 or self.noise_fraction < 0):
            raise ValueError("Augmentation strengths must be finite and nonnegative")
        if augmentation and self.architecture != "cnn1d":
            raise ValueError("Raw sensor augmentation is supported only for CNN sequences")
        values = self._input(X)
        self.input_shape_ = tuple(values.shape[1:])
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(encoded) != len(values) or len(self.classes_) < 2:
            raise ValueError("Training labels must match the input and contain at least two classes")
        axes = (0, 1) if values.ndim == 3 else 0
        self.mean_ = values.mean(axis=axes, dtype=np.float64).astype(np.float32)
        deviation = values.std(axis=axes, dtype=np.float64).astype(np.float32)
        self.scale_ = np.where(deviation > 1e-7, deviation, 1).astype(np.float32)
        if augmentation:
            self.augmentation_noise_scale_ = deviation.copy()
        self.n_features_in_ = values.shape[-1]
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            network = self._build_network()
            train_x, train_y = self._tensor(values), torch.as_tensor(encoded, dtype=torch.long)
            validation_x = self._tensor(validation_data[0]) if validation_data is not None else None
            validation_y = np.asarray(validation_data[1]) if validation_data is not None else None
            if validation_y is not None and (len(validation_y) != len(validation_x) or not set(validation_y).issubset(self.classes_)):
                raise ValueError("Validation labels do not match the trained classes")
            counts = np.bincount(encoded, minlength=len(self.classes_))
            criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(y) / (len(counts) * counts), dtype=torch.float32))
            optimizer = torch.optim.AdamW(network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            generator = torch.Generator().manual_seed(self.random_state)
            augmentation_generator = np.random.default_rng(self.random_state)
            history, best_score, best_epoch, stale = [], -np.inf, 0, 0
            for epoch in range(int(self.epochs)):
                network.train()
                order = torch.randperm(len(train_x), generator=generator)
                loss_sum = 0.0
                for index in order.split(int(self.batch_size)):
                    optimizer.zero_grad(set_to_none=True)
                    batch = (self._tensor(self._augment_training_values(values[index.numpy()], augmentation_generator))
                        if augmentation else train_x[index])
                    loss = criterion(network(batch), train_y[index])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite neural training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 2.0)
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(index)
                network.eval()
                row = {"epoch": epoch + 1, "training_loss": loss_sum / len(train_x)}
                if validation_x is not None:
                    with torch.inference_mode():
                        predicted = self.classes_[network(validation_x).argmax(dim=1).numpy()]
                    score = float(f1_score(validation_y, predicted, labels=self.classes_, average="macro", zero_division=0))
                    row["validation_macro_f1"] = score
                else:
                    score = -row["training_loss"]
                history.append(row)
                if score > best_score:
                    best_score, best_epoch, stale = score, epoch + 1, 0
                    self.weights_ = {key: value.detach().numpy().copy() for key, value in network.state_dict().items()}
                else:
                    stale += 1
                if validation_x is not None and stale >= self.patience:
                    break
            network.load_state_dict({key: torch.from_numpy(value) for key, value in self.weights_.items()})
            self._network = network.eval()
        self.training_details_ = {"architecture": str(network), "best_epoch": best_epoch,
            "epochs_run": len(history), "history": history, "device": "cpu", "threads": 1,
            "epoch_selection": "subject-disjoint validation Macro F1" if validation_data is not None else "training loss",
            "parameter_count": sum(value.size for value in self.weights_.values())}
        if augmentation:
            self.training_details_["augmentation"] = {"gain_std": float(self.gain_std),
                "noise_fraction": float(self.noise_fraction), "space": "raw sensor values before normalization",
                "noise_scale": "per-sensor training standard deviation", "training_only": True,
                "channel_gain_bounds": [0.5, 1.5], "negative_pressure_policy": "clip to zero"}
        return self

    def predict_proba(self, X):
        import torch
        check_is_fitted(self, "weights_")
        if not hasattr(self, "_network"):
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                self._network = self._build_network()
            self._network.load_state_dict({key: torch.from_numpy(value) for key, value in self.weights_.items()})
            self._network.eval()
        values = self._tensor(X)
        with torch.inference_mode():
            return np.concatenate([self._network(batch).softmax(dim=1).numpy()
                                   for batch in values.split(int(self.batch_size))]).astype(np.float64)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key != "_network"}
