"""Headless executable check for the exact frozen application's runtime assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


METADATA_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "jsonschema", "packaging")


def run_self_check(root: Path, state_root: Path) -> dict[str, object]:
    """Load the packaged selection through its real registry and execute inference."""
    from src.inference_core import PosturePredictor
    from src.model_runtime import ModelRegistry
    from src.personal_guidance import AlertSettings, PostureAlertTimer
    from src.posture_core import CLASS_NAMES, read_sensor_csv
    from src.app_products import select_product
    import numpy as np

    root, state_root = Path(root).resolve(), Path(state_root).resolve()
    checks: dict[str, object] = {}
    report: dict[str, object] = {
        "status": "failed", "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen": bool(getattr(sys, "frozen", False)), "executable": sys.executable,
        "resource_root": str(root), "state_root": str(state_root), "checks": checks,
        "scope": "Import metadata, selected model, actual CSV inference, inventory and default alert policy; no physical sensor or GUI interaction.",
    }
    try:
        product = select_product()
        identity_path = root / "build_identity.json"
        if identity_path.is_file():
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            expected = {
                "product_variant": product.variant,
                "owner_name": product.owner_name,
                "output_directory_prefix": product.output_directory_prefix,
                "display_name": product.display_name,
                "executable_name": product.executable_name,
                "bundle_identifier": product.bundle_identifier,
                "state_directory_name": product.state_directory_name,
                "entrypoint": product.entrypoint,
            }
            mismatches = {
                key: {"expected": value, "actual": identity.get(key)}
                for key, value in expected.items() if identity.get(key) != value
            }
            if mismatches:
                raise ValueError(f"The frozen product identity does not match its entrypoint: {mismatches}")
            checks["build_identity"] = identity
        elif report["frozen"]:
            raise ValueError("The frozen application is missing build_identity.json.")
        else:
            checks["build_identity"] = {
                "product_variant": product.variant,
                "bundle_identifier": product.bundle_identifier,
                "state_directory_name": product.state_directory_name,
                "status": "source-run",
            }
        checks["package_versions"] = {package: importlib.metadata.version(package) for package in METADATA_PACKAGES}
        required_modules = ("serial", "serial.tools.list_ports", "tkinter", "sklearn.svm", "sklearn.linear_model", "sklearn.ensemble", "train_window_models", "src.guidance_session")
        for name in required_modules:
            importlib.import_module(name)
        checks["imports"] = list(required_modules)
        settings = AlertSettings()
        checks["default_alert_settings"] = settings.as_dict()
        if (settings.mode, settings.initial_alert_s, settings.realert_interval_s) != ("deployment", 60.0, 120.0):
            raise ValueError("The frozen deployment alert defaults are not 60/120 seconds.")
        for mode, initial, repeat in (("development", 12.0, 24.0), ("deployment", 60.0, 120.0)):
            timer = PostureAlertTimer(AlertSettings.for_mode(mode))
            timer.update(0, "incorrect")
            if timer.update(initial - 0.001, "incorrect").popup_action is not None or timer.update(initial, "incorrect").popup_action != "show":
                raise ValueError(f"The {mode} initial popup boundary failed.")
            timer.close_popup()
            if timer.update(initial + repeat - 0.001, "incorrect").popup_action is not None or timer.update(initial + repeat, "incorrect").popup_action != "show":
                raise ValueError(f"The {mode} repeated popup boundary failed.")
        checks["alert_boundary_tests"] = "passed"
        inventory_path = root / "artifacts" / "package_inventory.json"
        if inventory_path.is_file():
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            for record in inventory["files"]:
                path = (root / "artifacts" / record["path"]).resolve()
                if not path.is_relative_to((root / "artifacts").resolve()):
                    raise ValueError("The package inventory contains an escaping path.")
                if path.stat().st_size != record["size_bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError(f"Packaged component changed: {record['path']}")
            prohibited = {"training_data", "measurements", "calibration_profiles"}
            unexpected = [str(path.relative_to(root)) for path in (root / "artifacts").rglob("*") if path.is_dir() and path.name in prohibited]
            if unexpected or inventory.get("raw_training_data_included") is not False:
                raise ValueError(f"Raw data unexpectedly included in the runtime package: {unexpected}")
            checks["inventory"] = {"status": "passed", "file_count": len(inventory["files"]), "raw_training_data_included": False}
        elif report["frozen"]:
            raise ValueError("The frozen package is missing its runtime inventory.")
        registry = ModelRegistry(root, state_root, use_bundled_release=True)
        models = registry.discover()
        selected = registry.active_model(models)
        checks["available_model_count"] = len(models)
        checks["discovery_results"] = [{"status": item.status, "reason": item.reason,
                                          "artifact_id": item.descriptor.artifact_id if item.descriptor else None}
                                         for item in registry.last_results]
        if selected is None:
            raise ValueError("No approved compatible model is available in this package.")
        checks["active_model_id"] = selected.artifact_id
        release_path = root / "artifacts" / "release_selection.json"
        if release_path.is_file():
            release = json.loads(release_path.read_text(encoding="utf-8"))
            if (selected.artifact_id, selected.artifact_hash, selected.manifest_hash) != tuple(release[key] for key in ("artifact_id", "artifact_hash", "manifest_hash")):
                raise ValueError("The application did not select the exact packaged release model.")
        checks["selected_model"] = {
            "artifact_id": selected.artifact_id, "version": selected.version,
            "artifact_hash": selected.artifact_hash, "manifest_hash": selected.manifest_hash,
            "model_family": selected.model_family, "window_seconds": selected.window.default_seconds,
            "runtime_compatible": selected.runtime_compatible, "production_selectable": selected.production_selectable,
        }
        predictor = PosturePredictor(selected.path, registry=registry)
        probabilities = predictor.predict_frame(read_sensor_csv(root / "example" / "sample_input.csv"))
        if probabilities.shape != (len(CLASS_NAMES),) or not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("Packaged-model inference returned invalid posture scores.")
        checks["csv_inference"] = {
            "status": "passed", "label": selected.labels[int(np.argmax(probabilities))],
            "scores": probabilities.tolist(), "label_order": list(selected.labels),
        }
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def main(root: Path | None = None, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-root", type=Path)
    args = parser.parse_args(argv)
    resource_root = root or Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    if args.state_root is None:
        with tempfile.TemporaryDirectory(prefix="posture-package-check-") as temporary:
            report = run_self_check(resource_root, Path(temporary))
    else:
        args.state_root.mkdir(parents=True, exist_ok=True)
        report = run_self_check(resource_root, args.state_root)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
