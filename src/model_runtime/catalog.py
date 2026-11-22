"""Cheap manifest discovery. This module never opens a model artifact."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


class ArtifactCatalog:
    """Find manifests under one or more artifact roots without opening weights."""

    def __init__(self, roots: Path | Iterable[Path]) -> None:
        if isinstance(roots, (str, Path)):
            roots = (Path(roots),)
        self.roots = tuple(Path(root) for root in roots)
        # Kept for callers that used the original single-root attribute.
        self.manifest_root = self.roots[0] if self.roots else Path()

    def manifests(self) -> list[Path]:
        found: set[Path] = set()
        for root in self.roots:
            if not root.is_dir():
                continue
            candidates = list(root.glob("*.json")) if root.name == "manifests" else [
                *root.rglob("manifests/*.json"), *root.rglob("manifest.json")
            ]
            for candidate in candidates:
                relative_parts = candidate.relative_to(root).parts
                if any(part.startswith(".") or part.endswith((".tmp", ".part", ".partial")) for part in relative_parts):
                    continue
                if any((parent / ".incomplete").exists() for parent in (candidate.parent, *candidate.parents) if parent == root or root in parent.parents):
                    continue
                found.add(candidate.resolve())
        return sorted(found)
