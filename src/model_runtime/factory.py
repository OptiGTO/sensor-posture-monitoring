"""Backend selection point; no GUI code needs to know model implementations."""

from __future__ import annotations

from .contracts import ModelDescriptor
from .pytorch_adapter import PyTorchAdapter
from .sklearn_adapter import SklearnAdapter


class AdapterFactory:
    @staticmethod
    def create(descriptor: ModelDescriptor):
        if descriptor.backend == "sklearn":
            return SklearnAdapter(descriptor)
        if descriptor.backend == "pytorch":
            return PyTorchAdapter(descriptor)
        raise ValueError(f"지원하지 않는 모델 백엔드: {descriptor.backend}")
