from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "딥러닝 비교를 실행하려면 PyTorch를 설치해야 합니다."
    ) from exc

from .posture_core import CLASS_NAMES, make_subject_pairs

DEFAULT_SEED = 20260826


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool = False):
        self.x = x
        self.y = y
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        signal = self.x[index].copy()
        if self.augment:
            # Small jitter and short time shifts preserve the static-posture semantics.
            if np.random.random() < 0.8:
                signal += np.random.normal(0.0, 0.02, signal.shape).astype(np.float32)
            if np.random.random() < 0.5:
                signal = np.roll(signal, np.random.randint(-3, 4), axis=0)
        return torch.from_numpy(signal), torch.tensor(self.y[index], dtype=torch.long)


class MLP(nn.Module):
    def __init__(self, window_length: int = 60, channels: int = 16, classes: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_length * channels, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 96),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(96, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, channels: int = 16, classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.30), nn.Linear(128, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x.transpose(1, 2)))


class BiGRU(nn.Module):
    def __init__(self, channels: int = 16, classes: int = 7):
        super().__init__()
        self.gru = nn.GRU(
            channels,
            48,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.20,
        )
        self.classifier = nn.Sequential(nn.Dropout(0.30), nn.Linear(96, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.classifier(output.mean(dim=1))


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TCN(nn.Module):
    def __init__(self, channels: int = 16, classes: int = 7):
        super().__init__()
        self.input_projection = nn.Conv1d(channels, 64, kernel_size=1)
        self.blocks = nn.Sequential(
            ResidualTemporalBlock(64, 1),
            ResidualTemporalBlock(64, 2),
            ResidualTemporalBlock(64, 4),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = self.input_projection(x.transpose(1, 2))
        return self.classifier(self.blocks(signal))


class TransformerClassifier(nn.Module):
    def __init__(self, window_length: int = 60, channels: int = 16, classes: int = 7):
        super().__init__()
        dimension = 48
        self.projection = nn.Linear(channels, dimension)
        self.position = nn.Parameter(torch.zeros(1, window_length, dimension))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=4,
            dim_feedforward=128,
            dropout=0.20,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dimension), nn.Dropout(0.25), nn.Linear(dimension, classes)
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.projection(x) + self.position)
        return self.classifier(encoded.mean(dim=1))


@dataclass
class WindowedData:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    record_ids: np.ndarray


def make_windows(
    sequences: np.ndarray,
    record_labels: np.ndarray,
    record_groups: np.ndarray,
    window_length: int = 60,
    stride: int = 30,
) -> WindowedData:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    record_ids: list[int] = []
    for record_id, sequence in enumerate(sequences):
        for start in range(0, len(sequence) - window_length + 1, stride):
            windows.append(sequence[start : start + window_length])
            labels.append(int(record_labels[record_id]))
            groups.append(str(record_groups[record_id]))
            record_ids.append(record_id)
    return WindowedData(
        np.stack(windows).astype(np.float32),
        np.asarray(labels, dtype=int),
        np.asarray(groups, dtype=str),
        np.asarray(record_ids, dtype=int),
    )


def _probabilities(
    model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 128
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            outputs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(outputs)


def _aggregate_record_predictions(
    probabilities: np.ndarray,
    selected_window_indices: np.ndarray,
    window_record_ids: np.ndarray,
    record_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    record_probabilities: list[np.ndarray] = []
    selected_record_ids = window_record_ids[selected_window_indices]
    for record_id in np.unique(selected_record_ids):
        mask = selected_record_ids == record_id
        mean_probability = probabilities[mask].mean(axis=0)
        true_labels.append(int(record_labels[record_id]))
        predicted_labels.append(int(mean_probability.argmax()))
        record_probabilities.append(mean_probability)
    return (
        np.asarray(true_labels, dtype=int),
        np.asarray(predicted_labels, dtype=int),
        np.stack(record_probabilities),
    )


def train_one_model(
    model_factory: Callable[[], nn.Module],
    windowed: WindowedData,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
    max_epochs: int = 45,
) -> tuple[nn.Module, np.ndarray, list[dict[str, float]]]:
    set_seed(seed)
    mean = windowed.x[train_indices].mean(axis=(0, 1), keepdims=True)
    std = windowed.x[train_indices].std(axis=(0, 1), keepdims=True) + 1e-6
    normalized = ((windowed.x - mean) / std).astype(np.float32)
    loader = DataLoader(
        WindowDataset(normalized[train_indices], windowed.y[train_indices], augment=True),
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3
    )
    loss_function = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    best_state = None
    best_loss = float("inf")
    patience = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_count = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = loss_function(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(y_batch)
            train_correct += int((logits.argmax(dim=1) == y_batch).sum().item())
            train_count += len(y_batch)

        model.eval()
        validation_loss = 0.0
        validation_correct = 0
        validation_count = 0
        with torch.no_grad():
            for start in range(0, len(validation_indices), 128):
                indices = validation_indices[start : start + 128]
                x_batch = torch.from_numpy(normalized[indices]).to(device)
                y_batch = torch.from_numpy(windowed.y[indices]).to(device)
                logits = model(x_batch)
                loss = loss_function(logits, y_batch)
                validation_loss += float(loss.item()) * len(indices)
                validation_correct += int((logits.argmax(dim=1) == y_batch).sum().item())
                validation_count += len(indices)
        current_validation_loss = validation_loss / validation_count
        scheduler.step(current_validation_loss)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss / train_count,
                "validation_loss": current_validation_loss,
                "train_accuracy": train_correct / train_count,
                "validation_window_accuracy": validation_correct / validation_count,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if current_validation_loss < best_loss - 1e-4:
            best_loss = current_validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if patience >= 7:
            break
    if best_state is None:
        raise RuntimeError("딥러닝 모델의 최적 상태를 저장하지 못했습니다.")
    model.load_state_dict(best_state)
    return model, normalized, history


def run_deep_screen(
    sequences: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    subject_codes: dict[str, str],
    seed: int = DEFAULT_SEED,
    include_recurrent: bool = False,
) -> dict[str, object]:
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = detect_device()
    print(f"딥러닝 장치: {device}", flush=True)
    windowed = make_windows(sequences, labels, groups)
    pairs = make_subject_pairs(groups, seed=seed)
    factories: dict[str, Callable[[], nn.Module]] = {
        "MLP": MLP,
        "1D CNN": CNN1D,
        "TCN": TCN,
        "Transformer": TransformerClassifier,
    }
    if include_recurrent:
        factories["BiGRU"] = BiGRU

    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    histories: dict[str, pd.DataFrame] = {}
    for model_name, factory in factories.items():
        start = time.perf_counter()
        current_folds = []
        for fold_index, test_subjects in enumerate(pairs, start=1):
            validation_subjects = pairs[fold_index % len(pairs)]
            training_subjects = [
                subject
                for pair in pairs
                for subject in pair
                if subject not in test_subjects and subject not in validation_subjects
            ]
            train_indices = np.where(np.isin(windowed.groups, training_subjects))[0]
            validation_indices = np.where(
                np.isin(windowed.groups, validation_subjects)
            )[0]
            test_indices = np.where(np.isin(windowed.groups, test_subjects))[0]
            assert set(training_subjects).isdisjoint(set(test_subjects))
            model, normalized, history = train_one_model(
                factory,
                windowed,
                train_indices,
                validation_indices,
                device,
                seed + fold_index - 1,
            )
            probability = _probabilities(model, normalized[test_indices], device)
            y_true, y_pred, _ = _aggregate_record_predictions(
                probability,
                test_indices,
                windowed.record_ids,
                labels,
            )
            row = {
                "model": model_name,
                "fold": fold_index,
                "test_subjects": " | ".join(subject_codes[s] for s in test_subjects),
                "validation_subjects": " | ".join(
                    subject_codes[s] for s in validation_subjects
                ),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                "epochs": len(history),
            }
            current_folds.append(row)
            fold_rows.append(row)
            if fold_index == 1:
                histories[model_name] = pd.DataFrame(history)
        current = pd.DataFrame(current_folds)
        summary_rows.append(
            {
                "model": model_name,
                "category": "딥러닝",
                "accuracy_mean": float(current["accuracy"].mean()),
                "accuracy_std": float(current["accuracy"].std(ddof=0)),
                "balanced_accuracy_mean": float(
                    current["balanced_accuracy"].mean()
                ),
                "balanced_accuracy_std": float(
                    current["balanced_accuracy"].std(ddof=0)
                ),
                "macro_f1_mean": float(current["macro_f1"].mean()),
                "macro_f1_std": float(current["macro_f1"].std(ddof=0)),
                "training_seconds": float(time.perf_counter() - start),
            }
        )
        print(
            f"{model_name}: Macro F1 "
            f"{summary_rows[-1]['macro_f1_mean']:.3f} ± "
            f"{summary_rows[-1]['macro_f1_std']:.3f}",
            flush=True,
        )
    return {
        "summary": pd.DataFrame(summary_rows).sort_values(
            "macro_f1_mean", ascending=False
        ),
        "folds": pd.DataFrame(fold_rows),
        "histories": histories,
        "device": str(device),
        "window_length": 60,
        "stride": 30,
    }
