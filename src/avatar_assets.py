"""Posture-avatar asset mapping shared by the desktop UI and its tests."""

from __future__ import annotations

from pathlib import Path


POSTURE_AVATAR_FILENAMES: dict[str, str] = {
    "정자세": "upright.png",
    "앞으로 숙이기": "lean-forward.png",
    "뒤로 기대기": "lean-backward.png",
    "왼쪽 기울임": "lean-left.png",
    "오른쪽 기울임": "lean-right.png",
    "왼쪽 다리 꼬기": "cross-left.png",
    "오른쪽 다리 꼬기": "cross-right.png",
}
WAITING_AVATAR_FILENAME = "waiting.png"


def avatar_filename(posture: str | None) -> str:
    """Return a safe avatar filename, falling back to the waiting pose."""
    return POSTURE_AVATAR_FILENAMES.get(posture or "", WAITING_AVATAR_FILENAME)


def missing_avatar_assets(asset_directory: Path) -> tuple[str, ...]:
    """List missing required PNGs in stable filename order."""
    filenames = {*POSTURE_AVATAR_FILENAMES.values(), WAITING_AVATAR_FILENAME}
    return tuple(filename for filename in sorted(filenames) if not (asset_directory / filename).is_file())
