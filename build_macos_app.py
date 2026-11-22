#!/usr/bin/env python3
"""Build the double-clickable macOS application with the deployed model assets."""

from __future__ import annotations

import subprocess
import sys
import os
import argparse
import hashlib
import json
import shutil
import uuid
import plistlib
from datetime import datetime, timezone
from pathlib import Path

from src.app_products import AppProduct, GTO_PRODUCT
from src.checkpoint_registry import checkpoint_resource_files, checkpoint_required_packages


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
CACHE = ROOT / ".pyinstaller-cache"
# Backward-compatible aliases for the established GTO release target.
APP_NAME = GTO_PRODUCT.display_name
EXECUTABLE_NAME = GTO_PRODUCT.executable_name
BUNDLE_IDENTIFIER = GTO_PRODUCT.bundle_identifier
# Runtime validation reads distribution versions via importlib.metadata.
# Including importable modules alone does not preserve their dist-info files.
RUNTIME_METADATA_PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "joblib", "jsonschema", "packaging",
)
EXTENDED_METADATA_PACKAGES = ("catboost", "xgboost", "numba", "llvmlite")


def build_command(*, distpath: Path | None = None, workpath: Path | None = None,
                  artifacts_root: Path | None = None, specpath: Path | None = None,
                  include_pytorch: bool = False, include_extended_models: bool = False,
                  product: AppProduct = GTO_PRODUCT,
                  identity_path: Path | None = None) -> list[str]:
    """Return a clean, reproducible PyInstaller command for the macOS bundle."""

    include_pytorch = include_pytorch or include_extended_models
    required_checkpoint_packages = checkpoint_required_packages(ROOT)
    if required_checkpoint_packages & set(EXTENDED_METADATA_PACKAGES) and not include_extended_models:
        raise ValueError("검증 체크포인트의 구성원 의존성을 포함하려면 --include-extended-models로 빌드하세요.")
    return [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed",
        "--name", product.executable_name, "--osx-bundle-identifier", product.bundle_identifier,
        "--add-data", f"{artifacts_root or ROOT / 'artifacts'}:artifacts",
        "--add-data", f"{ROOT / 'schemas'}:schemas",
        "--add-data", f"{ROOT / 'example'}:example",
        "--add-data", f"{ROOT / 'software' / 'assets'}:software/assets",
        # Keep the recipe evidence beside its module for source/frozen parity.
        # A checkout without a registry safely exposes only standard training.
        *(["--add-data", f"{ROOT / 'src' / 'training_improvements.json'}:src"]
          if (ROOT / "src" / "training_improvements.json").is_file() else []),
        *(["--add-data", f"{ROOT / 'src' / 'validated_checkpoints.json'}:src"]
          if (ROOT / "src" / "validated_checkpoints.json").is_file() else []),
        *[argument for path in checkpoint_resource_files(ROOT)
          for argument in ("--add-data", f"{path}:{path.relative_to(ROOT).parent}")],
        # PyInstaller treats the add-data destination as a directory.  Using the
        # filename here would create build_identity.json/build_identity.json,
        # while the frozen self-check intentionally reads it from the resource
        # root as build_identity.json.
        *(["--add-data", f"{identity_path}:."] if identity_path is not None else []),
        "--hidden-import", "serial", "--hidden-import", "serial.tools.list_ports", "--hidden-import", "train_window_models",
        "--hidden-import", "packaging_self_check",
        "--hidden-import", "software.posture_app_base",
        "--hidden-import", "src.training_methods",
        "--hidden-import", "src.checkpoint_registry", "--hidden-import", "src.iterative_pipeline",
        "--hidden-import", "validate_integrated_gui", "--hidden-import", "validate_comparison_gui",
        *([] if include_pytorch else ["--exclude-module", "torch"]),
        *([] if include_extended_models else [argument for package in EXTENDED_METADATA_PACKAGES
                                              for argument in ("--exclude-module", package)]),
        *(["--copy-metadata", "torch"] if include_pytorch else []),
        *([argument for package in EXTENDED_METADATA_PACKAGES
           for argument in ("--copy-metadata", package)] if include_extended_models else []),
        *(["--collect-all", "catboost", "--collect-all", "xgboost",
           "--collect-all", "numba", "--collect-all", "llvmlite",
           "--add-data", f"{ROOT / 'src' / '_vendor'}:src/_vendor"] if include_extended_models else []),
        *[argument for package in RUNTIME_METADATA_PACKAGES for argument in ("--copy-metadata", package)],
        "--paths", str(ROOT), "--distpath", str(distpath or DIST), "--workpath", str(workpath or ROOT / "build"),
        *(["--specpath", str(specpath)] if specpath is not None else []),
        str(ROOT / product.entrypoint),
    ]


def stage_runtime_artifacts(source: Path, destination: Path, *, model_runs: tuple[Path, ...] = ()) -> dict[str, object]:
    """Copy immutable runtime dependencies, never recursively copy training runs.

    Model run selection is explicit. Manifests retain their bytes and relative
    paths, so runtime hash and deployment-evidence checks remain meaningful.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError("The staging directory must be separate from source artifacts.")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("The runtime staging directory must be empty.")
    destination.mkdir(parents=True, exist_ok=True)
    copied: set[Path] = set()
    identities: set[tuple[str, str, str]] = set()

    def copy_file(path: Path) -> None:
        resolved = path.resolve()
        # Report prose has one canonical copy under docs. Only the exact mirrored
        # Markdown path is allowed outside artifacts; model components stay confined.
        if (not resolved.is_relative_to(source) and path.suffix == '.md'
                and source == (ROOT / 'research/models').resolve()
                and path.is_relative_to(source)
                and '..' not in path.relative_to(source).parts
                and resolved.is_relative_to((ROOT / 'docs/research/models').resolve())
                and resolved == (ROOT / 'docs/research/models' / path.relative_to(source)).resolve()
                and resolved.is_file()):
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved, target)
            copied.add(relative)
            return
        if not resolved.is_relative_to(source) or not resolved.is_file():
            raise ValueError(f"Missing runtime component or path outside artifacts: {path}")
        relative = resolved.relative_to(source)
        if any(part in {"training_data", "data", "measurements", "calibration_profiles"} for part in relative.parts):
            raise ValueError(f"A runtime manifest must not include raw collection data: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative not in copied:
            shutil.copy2(resolved, target)
            copied.add(relative)

    def add_manifest(path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = tuple(str(payload[key]) for key in ("artifact_id", "version", "artifact_sha256"))
        identities.add(identity)
        copy_file(path)
        component_names = [payload["artifact_file"], *[item["file"] for item in payload.get("components", [])]]
        for name in component_names:
            if not isinstance(name, str) or Path(name).is_absolute():
                raise ValueError("Packaged model component paths must be relative.")
            copy_file(path.parent / name)

    for window in (3, 5, 20):
        add_manifest(source / "manifests" / f"posture_classifier_{window}s.json")
    copy_file(source / "golden" / "legacy_baseline_v1.json")
    for name in ("window_3s_model_config.json", "window_5s_model_config.json", "final_model_config.json"):
        if (source / name).is_file():
            copy_file(source / name)

    selected_runs: list[str] = []
    for supplied in model_runs:
        run = Path(supplied).resolve()
        if not run.is_relative_to(source / "models") or not run.is_dir():
            raise ValueError("Selected model runs must be existing directories under artifacts/models.")
        runtime_root = run / "artifacts"
        manifests = sorted((runtime_root / "manifests").glob("*.json"))
        if not manifests or (run / ".incomplete").exists() or (runtime_root / ".incomplete").exists():
            raise ValueError(f"The selected model run has no completed runtime manifests: {run.name}")
        for manifest in manifests:
            add_manifest(manifest)
        for path in (runtime_root / "model_comparison.json", runtime_root / "model_comparison.md", run / "training.manifest.json"):
            if path.is_file():
                copy_file(path)
        selected_runs.append(run.relative_to(source).as_posix())

    # Experimental personal hierarchies have their own checked contract. Bundle
    # model/evaluation files only; registration profiles and training CSVs stay local.
    from src.personalized_model import PersonalPostureModel
    for path in sorted((source / "personalized").glob("*/personalized.manifest.json")):
        try:
            PersonalPostureModel.load(path)
        except (OSError, ValueError):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        copy_file(path)
        for role in ("model", "report"):
            copy_file(path.parent / payload[role]["file"])
        report_md = path.parent / "personalized.report.md"
        if report_md.is_file(): copy_file(report_md)

    for path in sorted((source / "deployment_records").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = tuple(str(payload.get(key, "")) for key in ("artifact_id", "version", "artifact_hash"))
        if identity in identities:
            copy_file(path)

    active = source / "active_model.json"
    if active.is_file():
        payload = json.loads(active.read_text(encoding="utf-8"))
        artifact_id = payload.get("artifact_id")
        if artifact_id and artifact_id not in {identity[0] for identity in identities}:
            raise ValueError("The active model is not included; pass its --model-run explicitly.")
        copy_file(active)
    release = source / "release_selection.json"
    if release.is_file():
        payload = json.loads(release.read_text(encoding="utf-8"))
        relative = Path(payload["manifest_file"])
        if relative.is_absolute() or relative not in copied:
            raise ValueError("The bundled release selection does not identify a staged manifest.")
        if payload.get("artifact_id") not in {identity[0] for identity in identities}:
            raise ValueError("The bundled release selection is not part of this package.")
        copy_file(release)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "default_alert_mode": "deployment",
        "initial_alert_s": 60,
        "realert_interval_s": 120,
        "selected_model_runs": selected_runs,
        "raw_training_data_included": False,
        "files": [{"path": path.as_posix(), "sha256": hashlib.sha256((destination / path).read_bytes()).hexdigest(),
                   "size_bytes": (destination / path).stat().st_size} for path in sorted(copied)],
    }
    (destination / "package_inventory.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def sign_and_verify(app: Path) -> None:
    """Seal the completed bundle so Finder sees one valid application identity."""
    # Sign nested framework bundles first. Recursive --deep signing can revisit
    # resource symlinks; it is appropriate for verification, not this signing pass.
    frameworks = {path.resolve() for path in (app / "Contents" / "Frameworks").rglob("*.framework") if path.is_dir()}
    for framework in sorted(frameworks, key=lambda path: (-len(path.parts), str(path))):
        subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(framework)], cwd=ROOT, check=True)
    subprocess.run(
        ["/usr/bin/codesign", "--force", "--sign", "-", str(app)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["/usr/bin/codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)],
        cwd=ROOT,
        check=True,
    )


def finalize_bundle(distpath: Path, product: AppProduct = GTO_PRODUCT) -> Path:
    """Keep the Korean display name with an ASCII CFBundleExecutable.

    macOS 27 can misidentify a Unicode executable as nested code and seal its
    own old signature, causing recursive signing crashes or invalid resources.
    """
    generated = distpath / f"{product.executable_name}.app"
    app = distpath / f"{product.display_name}.app"
    if app.exists():
        raise ValueError("The destination application already exists.")
    info_path = generated / "Contents" / "Info.plist"
    info = plistlib.loads(info_path.read_bytes())
    if info.get("CFBundleExecutable") != product.executable_name:
        raise ValueError("Unexpected packaged executable identity.")
    info.update(CFBundleName=product.display_name, CFBundleDisplayName=product.display_name)
    info_path.write_bytes(plistlib.dumps(info))
    generated.rename(app)
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distpath", type=Path, help="New output directory; existing applications are never overwritten.")
    parser.add_argument("--workpath", type=Path, help="Empty build work directory.")
    parser.add_argument("--model-run", type=Path, action="append", default=[], help="Explicit completed artifacts/models run to include (repeatable).")
    parser.add_argument("--include-pytorch", action="store_true", help="Include the optional PyTorch inference backend; core four-model training uses sklearn.")
    parser.add_argument("--include-extended-models", action="store_true", help="Include all nine training choices (CatBoost, XGBoost, MiniRocket, Small 1D-CNN and Small MLP).")
    return parser


def resolve_model_runs(source: Path, explicit: tuple[Path, ...] = ()) -> tuple[Path, ...]:
    """Default to the already selected release; never guess a newest training run."""
    if explicit:
        return explicit
    source = Path(source).resolve()
    release = source / "release_selection.json"
    if not release.is_file():
        return ()
    payload = json.loads(release.read_text(encoding="utf-8"))
    relative = Path(payload["manifest_file"])
    if (relative.is_absolute() or len(relative.parts) < 5 or relative.parts[0] != "models"
            or relative.parts[2:4] != ("artifacts", "manifests")):
        raise ValueError("The selected release must identify a manifest under artifacts/models/<run>/artifacts/manifests.")
    manifest = (source / relative).resolve()
    run = (source / "models" / relative.parts[1]).resolve()
    if not manifest.is_relative_to(run) or not run.is_relative_to(source / "models") or not manifest.is_file():
        raise ValueError("The selected release manifest is missing or escapes its model run.")
    return (run,)


def write_build_identity(path: Path, product: AppProduct) -> dict[str, object]:
    """Embed enough provenance to reject a bundle built from the wrong UI target."""

    source = ROOT / product.entrypoint
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    identity = {
        "schema_version": 1,
        "product_variant": product.variant,
        "owner_name": product.owner_name,
        "output_directory_prefix": product.output_directory_prefix,
        "display_name": product.display_name,
        "executable_name": product.executable_name,
        "bundle_identifier": product.bundle_identifier,
        "state_directory_name": product.state_directory_name,
        "entrypoint": product.entrypoint,
        "entrypoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_revision": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
        "source_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return identity


def require_product_labeled_directory(path: Path, product: AppProduct, option: str) -> None:
    """Reject ambiguous build folders before PyInstaller writes any output."""

    expected = f"{product.output_directory_prefix}-"
    if not path.name.startswith(expected) or path.name == expected:
        raise ValueError(
            f"{option} must identify the {product.owner_name} ({product.variant}) app "
            f"with a directory named {expected}<release>: {path}"
        )


def main(argv: list[str] | None = None, *, product: AppProduct = GTO_PRODUCT) -> None:
    args = build_parser().parse_args(argv)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    distpath = (args.distpath or DIST / f"{product.output_directory_prefix}-{tag}").resolve()
    workpath = (args.workpath or ROOT / "build" / f"{product.output_directory_prefix}-{tag}").resolve()
    require_product_labeled_directory(distpath, product, "--distpath")
    require_product_labeled_directory(workpath, product, "--workpath")
    if any((distpath / name).exists() for name in (
            f"{product.display_name}.app", product.display_name,
            f"{product.executable_name}.app", product.executable_name)):
        raise ValueError("The output already contains this application; choose a new --distpath.")
    if workpath.exists() and any(workpath.iterdir()):
        raise ValueError("Choose an empty --workpath to preserve existing build output.")
    staging = workpath.parent / f"{workpath.name}-runtime-assets" / "artifacts"
    specpath = workpath.parent / f"{workpath.name}-spec"
    model_runs = resolve_model_runs(ROOT / "artifacts", tuple(args.model_run))
    stage_runtime_artifacts(ROOT / "artifacts", staging, model_runs=model_runs)
    identity_path = staging.parent / "build_identity.json"
    write_build_identity(identity_path, product)
    specpath.mkdir(parents=True, exist_ok=False)
    # PyInstaller's clean build deletes its work tree. Keep the binary-signing
    # cache outside that tree so COLLECT can still read cached extensions.
    environment = os.environ | {"PYINSTALLER_CONFIG_DIR": str(CACHE)}
    command = build_command(distpath=distpath, workpath=workpath, artifacts_root=staging, specpath=specpath,
                            include_pytorch=args.include_pytorch,
                            include_extended_models=args.include_extended_models,
                            product=product, identity_path=identity_path)
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    app = finalize_bundle(distpath, product)
    if not app.is_dir():
        raise RuntimeError("macOS 앱 생성 결과를 찾지 못했습니다.")
    sign_and_verify(app)
    print(app)


if __name__ == "__main__":
    main()
