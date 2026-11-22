"""Safe state_dict-based PyTorch inference. Importing this module does not require torch."""

from __future__ import annotations

import numpy as np

from .contracts import ModelDescriptor


class PyTorchAdapter:
    def __init__(self, descriptor: ModelDescriptor) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch를 사용할 수 없습니다.") from error
        from .pytorch_models import build_model
        self.torch = torch
        self.descriptor = descriptor
        params = dict(descriptor.metadata.get("factory_parameters", {}))
        self.model = build_model(str(descriptor.metadata["factory_key"]), **params)
        state = torch.load(descriptor.artifact_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.device = self._select_device()
        try:
            self.model.to(self.device).eval()
        except RuntimeError:
            self.device = torch.device("cpu")
            self.model.to(self.device).eval()
        normalization = descriptor.preprocessing.get("normalization", {})
        self.mean = np.asarray(normalization.get("mean", []), dtype=np.float32)
        self.std = np.asarray(normalization.get("std", []), dtype=np.float32)

    def _select_device(self):
        torch = self.torch
        allowed = set(self.descriptor.metadata.get("supported_devices", ["cpu", "mps", "cuda"]))
        if "cuda" in allowed and torch.cuda.is_available(): return torch.device("cuda")
        if "mps" in allowed and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    def validate_runtime(self, window_seconds: float) -> float:
        return self.descriptor.window.validate(window_seconds)

    def predict_proba(self, sequence: np.ndarray, window_seconds: float | None = None) -> np.ndarray:
        self.validate_runtime(window_seconds if window_seconds is not None else self.descriptor.window.default_seconds)
        data = np.asarray(sequence, dtype=np.float32)
        if data.ndim != 3 or data.shape[-1] != len(self.descriptor.input_channels):
            raise ValueError("PyTorch 입력 형태가 매니페스트와 다릅니다.")
        if self.mean.size:
            data = (data - self.mean) / np.where(self.std == 0, 1.0, self.std)
        with self.torch.inference_mode():
            output = self.model(self.torch.as_tensor(data, device=self.device, dtype=self.torch.float32)).detach().cpu().numpy()
        if self.descriptor.output_type == "logits": output = np.exp(output - output.max(axis=1, keepdims=True)); output /= output.sum(axis=1, keepdims=True)
        if output.shape != (len(data), len(self.descriptor.labels)) or not np.isfinite(output).all() or (output < 0).any():
            raise ValueError("PyTorch 출력 계약 검증에 실패했습니다.")
        return output
