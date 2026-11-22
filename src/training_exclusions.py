from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path


def _normalize(value: object) -> str:
    return unicodedata.normalize("NFC", str(value)).strip()


def default_exclusion_manifest(data_root: Path) -> Path:
    data_root = data_root.resolve()
    project_root = data_root.parent if data_root.name == "data" else data_root
    return project_root / "research" / "models" / "training_exclusions.json"


def load_training_exclusions(
    data_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    data_root = Path(data_root).resolve()
    manifest = Path(manifest_path).resolve() if manifest_path else default_exclusion_manifest(data_root)
    if not manifest.is_file():
        return {}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("exclusions"), list):
        raise ValueError(f"잘못된 학습 제외 manifest: {manifest}")
    exclusions: dict[str, dict[str, str]] = {}
    required = {"source_file", "source_sha256", "reason", "requested_at"}
    for value in payload["exclusions"]:
        if not isinstance(value, dict) or not required.issubset(value) or not all(isinstance(value[key], str) for key in required):
            raise ValueError(f"잘못된 학습 제외 항목: {manifest}")
        relative = Path(value["source_file"])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.casefold() != ".csv":
            raise ValueError(f"안전하지 않은 학습 제외 경로: {value['source_file']}")
        key = _normalize(relative.as_posix())
        if key in exclusions:
            raise ValueError(f"중복 학습 제외 항목: {value['source_file']}")
        source = data_root / relative
        if not source.is_file():
            raise ValueError(f"학습 제외 원본을 찾을 수 없습니다: {value['source_file']}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != value["source_sha256"]:
            raise ValueError(f"학습 제외 원본 해시가 달라졌습니다: {value['source_file']}")
        exclusions[key] = {name: str(value[name]) for name in required}
    return exclusions


def is_training_excluded(path: str | Path, data_root: str | Path, exclusions: dict[str, dict[str, str]]) -> bool:
    relative = Path(path).resolve().relative_to(Path(data_root).resolve())
    return _normalize(relative.as_posix()) in exclusions
