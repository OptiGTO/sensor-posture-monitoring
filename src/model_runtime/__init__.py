"""Manifest-first model discovery and backend-neutral posture inference."""

from .contracts import ModelDescriptor, SessionModelSnapshot, WindowContract
from .catalog import ArtifactCatalog
from .factory import AdapterFactory
from .lifecycle import DeploymentRecord, DeploymentStore, ValidationCache
from .registry import ModelRegistry
from .validation import ArtifactValidator, ValidationResult

__all__ = [
    "AdapterFactory", "ArtifactCatalog", "ArtifactValidator", "DeploymentRecord",
    "DeploymentStore", "ModelDescriptor", "ModelRegistry", "SessionModelSnapshot",
    "ValidationCache", "ValidationResult", "WindowContract",
]
