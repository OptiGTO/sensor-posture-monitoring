"""Locate writable datasets independently of bundled read-only resources."""

from pathlib import Path
import os


def data_directory(source_root: Path, *, executable: Path | None = None) -> Path:
    """Use project-relative data for source and the standard dist layouts.

    An app moved outside dist uses data beside the app. Never use the process
    working directory or PyInstaller's internal resource directory for records.
    """
    if executable is None:
        return source_root.resolve() / "data"
    executable = executable.resolve()
    bundle = next((parent for parent in executable.parents if parent.suffix == ".app"), None)
    base = bundle.parent if bundle else executable.parent
    # Both dist/App.app and dist/<version>/App.app (and onedir variants)
    # must retain the same dataset when a new build is installed locally.
    distribution = next((parent for parent in (base, *base.parents) if parent.name == "dist"), None)
    if distribution is not None:
        base = distribution.parent
    return base / "data"


def writable_data_directory(preferred: Path, state_root: Path) -> Path:
    """Keep portable data when writable; installed apps use durable user state."""
    fallback = state_root / "data"
    for candidate in dict.fromkeys((preferred, fallback)):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            # Measurement storage may allow new files but deliberately deny
            # deletion. A TemporaryFile probe then leaves an undeletable file
            # and incorrectly rejects a perfectly usable recording directory.
            # The recorder's exclusive open remains the authoritative check.
            if not os.access(candidate, os.W_OK | os.X_OK):
                raise PermissionError(f"Dataset directory is not writable: {candidate}")
            return candidate
        except OSError:
            if candidate == fallback:
                raise
    raise OSError("No writable dataset directory is available")


def dataset_directories(recording_directory: Path, source_root: Path, *,
                        executable: Path | None = None, isolated: bool = False) -> tuple[Path, ...]:
    """Read legacy project/portable records alongside this product's recordings.

    An explicit state root is an isolation boundary (including GUI self-checks).
    Never discover another product's user state or copy/migrate original data.
    """
    directories = [recording_directory.resolve()]
    if not isolated:
        existing = data_directory(source_root, executable=executable)
        if existing.is_dir() and existing.resolve() not in directories:
            directories.append(existing.resolve())
    return tuple(directories)
