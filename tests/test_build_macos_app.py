from __future__ import annotations

import unittest
import json
import tempfile
import plistlib
from pathlib import Path
from unittest.mock import call, patch

import build_macos_app


class MacOSBuildTests(unittest.TestCase):
    def test_gto_command_bundles_recipe_evidence_when_present(self):
        from src.app_products import GTO_PRODUCT
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "src/training_improvements.json"
            registry.parent.mkdir()
            with patch.object(build_macos_app, "ROOT", root):
                missing = build_macos_app.build_command()
                self.assertNotIn(f"{registry}:src", missing)
                registry.write_text('{"schema_version": 1, "methods": []}')
                command = build_macos_app.build_command()
                data = [command[i + 1] for i, value in enumerate(command[:-1]) if value == "--add-data"]
                self.assertIn(f"{registry}:src", data)
                hidden = [command[i + 1] for i, value in enumerate(command[:-1]) if value == "--hidden-import"]
                self.assertIn("src.training_methods", hidden)
                self.assertEqual(command[-1], str(root / GTO_PRODUCT.entrypoint))

    def test_extended_training_build_copies_optional_runtimes_and_vendor_license(self):
        command = build_macos_app.build_command(include_extended_models=True)
        self.assertNotIn("--exclude-module", command)
        copied = [command[i + 1] for i, value in enumerate(command[:-1]) if value == "--copy-metadata"]
        for name in ("torch", *build_macos_app.EXTENDED_METADATA_PACKAGES):
            self.assertIn(name, copied)
        self.assertIn(f"{build_macos_app.ROOT / 'src' / '_vendor'}:src/_vendor", command)

    def test_build_is_clean_and_uses_a_stable_bundle_identifier(self) -> None:
        command = build_macos_app.build_command()
        self.assertIn("--clean", command)
        identifier_index = command.index("--osx-bundle-identifier")
        self.assertEqual(command[identifier_index + 1], build_macos_app.BUNDLE_IDENTIFIER)
        self.assertNotEqual(build_macos_app.CACHE.parent / "build", build_macos_app.CACHE)
        self.assertNotIn(build_macos_app.ROOT / "build", build_macos_app.CACHE.parents)

    def test_isolated_build_paths_preserve_existing_spec_and_copy_metadata(self) -> None:
        command = build_macos_app.build_command(distpath=Path("/tmp/new-dist"), workpath=Path("/tmp/new-work"),
                                                artifacts_root=Path("/tmp/runtime-only"), specpath=Path("/tmp/new-spec"))
        for option, value in (("--distpath", "/tmp/new-dist"), ("--workpath", "/tmp/new-work"), ("--specpath", "/tmp/new-spec")):
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertIn("/tmp/runtime-only:artifacts", command)
        self.assertIn("packaging_self_check", command)
        copied_metadata = [command[index + 1] for index, item in enumerate(command[:-1]) if item == "--copy-metadata"]
        self.assertEqual(copied_metadata, list(build_macos_app.RUNTIME_METADATA_PACKAGES))

    def test_core_package_omits_optional_torch_but_can_include_it_explicitly(self):
        command = build_macos_app.build_command()
        self.assertEqual(command[command.index('--exclude-module') + 1], 'torch')
        extended = build_macos_app.build_command(include_pytorch=True)
        excluded = [extended[index + 1] for index, value in enumerate(extended[:-1]) if value == '--exclude-module']
        self.assertNotIn('torch', excluded)
        self.assertIn('xgboost', excluded)
        copied = [extended[index + 1] for index, value in enumerate(extended[:-1]) if value == '--copy-metadata']
        self.assertIn('torch', copied)

    @staticmethod
    def write(path: Path, content="fixture") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content) if isinstance(content, dict) else content, encoding="utf-8")

    def artifacts(self, root: Path) -> Path:
        source = root / "source"
        for window in (3, 5, 20):
            self.write(source / f"model-{window}.joblib")
            self.write(source / "manifests" / f"posture_classifier_{window}s.json", {
                "artifact_id": f"builtin-{window}", "version": "1", "artifact_sha256": f"hash-{window}",
                "artifact_file": f"../model-{window}.joblib",
            })
        self.write(source / "golden" / "legacy_baseline_v1.json", {})
        return source

    def test_staging_reads_only_exact_consolidated_report_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self.artifacts(root)
            source = root / 'research/models'
            source.parent.mkdir()
            original.rename(source)
            run = source / 'models/selected'
            runtime = run / 'artifacts'
            self.write(runtime / 'candidate.joblib')
            self.write(runtime / 'manifests/candidate.json', {
                'artifact_id': 'candidate', 'version': '1', 'artifact_sha256': 'hash',
                'artifact_file': '../candidate.joblib',
            })
            prose = root / 'docs/research/models/models/selected/artifacts/model_comparison.md'
            self.write(prose, 'report content')
            alias = runtime / 'model_comparison.md'
            alias.symlink_to(prose)
            with patch.object(build_macos_app, 'ROOT', root):
                build_macos_app.stage_runtime_artifacts(source, root / 'staged', model_runs=(run,))
                self.assertEqual((root / 'staged/models/selected/artifacts/model_comparison.md').read_text(), 'report content')
                alias.unlink()
                outside = root / 'outside.md'
                self.write(outside, 'must not package')
                alias.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, 'outside artifacts'):
                    build_macos_app.stage_runtime_artifacts(source, root / 'rejected', model_runs=(run,))

    def test_staging_includes_only_selected_runtime_components_and_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.artifacts(root)
            run = source / "models" / "selected"
            runtime = run / "artifacts"
            manifest = runtime / "manifests" / "candidate.json"
            self.write(runtime / "model.joblib")
            self.write(runtime / "evaluation.json", {"scores": [1, 2]})
            self.write(manifest, {"artifact_id": "candidate", "version": "2", "artifact_sha256": "newhash",
                                  "artifact_file": "../model.joblib", "components": [{"file": "../evaluation.json"}]})
            self.write(runtime / "model_comparison.json", {"selected": "candidate"})
            self.write(run / "training_data" / "person.csv", "private")
            self.write(source / "models" / "failed" / "training_data" / "person.csv", "private")
            self.write(source / "calibration_profiles" / "user.json", {"private": True})
            self.write(source / "deployment_records" / "included.json", {"artifact_id": "candidate", "version": "2", "artifact_hash": "newhash"})
            self.write(source / "deployment_records" / "excluded.json", {"artifact_id": "old", "version": "2", "artifact_hash": "oldhash"})
            self.write(source / "release_selection.json", {"artifact_id": "candidate", "manifest_file": "models/selected/artifacts/manifests/candidate.json"})
            destination = root / "staged"
            inventory = build_macos_app.stage_runtime_artifacts(source, destination, model_runs=(run,))
            self.assertEqual((destination / manifest.relative_to(source)).read_bytes(), manifest.read_bytes())
            self.assertTrue((destination / "deployment_records" / "included.json").is_file())
            self.assertTrue((destination / "release_selection.json").is_file())
            self.assertFalse((destination / "deployment_records" / "excluded.json").exists())
            self.assertFalse((destination / "models" / "failed").exists())
            self.assertFalse((destination / "models" / "selected" / "training_data").exists())
            self.assertFalse((destination / "calibration_profiles").exists())
            self.assertEqual((inventory["default_alert_mode"], inventory["initial_alert_s"], inventory["realert_interval_s"]), ("deployment", 60, 120))
            self.assertFalse(inventory["raw_training_data_included"])
            with self.assertRaisesRegex(ValueError, "empty"):
                build_macos_app.stage_runtime_artifacts(source, destination, model_runs=(run,))

    def test_personal_runtime_package_excludes_registration_and_training_data(self):
        import shutil
        from src.personalized_model import PersonalPostureModel
        project = Path(__file__).resolve().parents[1]
        model = project / "artifacts/personalized/20260911_initial"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = self.artifacts(root)
            target = source / "personalized" / "experiment"
            shutil.copytree(model, target)
            self.write(target / "training_data" / "person.csv", "private data")
            self.write(target / "calibration_profiles" / "person.json", {})
            inventory = build_macos_app.stage_runtime_artifacts(source, root / "staged")
            output = root / "staged/personalized/experiment/personalized.manifest.json"
            PersonalPostureModel.load(output)
            self.assertFalse((output.parent / "training_data").exists())
            self.assertFalse((output.parent / "calibration_profiles").exists())
            self.assertFalse(inventory["raw_training_data_included"])

    def test_component_path_escape_and_unselected_active_model_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.artifacts(root)
            self.write(source / "active_model.json", {"artifact_id": "unselected"})
            with self.assertRaisesRegex(ValueError, "active model"):
                build_macos_app.stage_runtime_artifacts(source, root / "staged1")
            self.write(source / "active_model.json", {"artifact_id": "builtin-3"})
            self.write(root / "outside.joblib")
            self.write(source / "manifests" / "posture_classifier_3s.json", {
                "artifact_id": "builtin-3", "version": "1", "artifact_sha256": "hash-3",
                "artifact_file": "../../outside.joblib",
            })
            with self.assertRaisesRegex(ValueError, "outside artifacts"):
                build_macos_app.stage_runtime_artifacts(source, root / "staged2")

    def test_default_model_run_is_resolved_from_selected_release_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self.artifacts(Path(temporary))
            self.assertEqual(build_macos_app.resolve_model_runs(source), ())
            manifest = source / "models" / "reviewed" / "artifacts" / "manifests" / "candidate.json"
            self.write(manifest, {})
            self.write(source / "release_selection.json", {"manifest_file": "models/reviewed/artifacts/manifests/candidate.json"})
            self.assertEqual(build_macos_app.resolve_model_runs(source), (manifest.parents[2].resolve(),))
            self.write(source / "release_selection.json", {"manifest_file": "../../outside.json"})
            with self.assertRaises(ValueError):
                build_macos_app.resolve_model_runs(source)

    @patch("build_macos_app.subprocess.run")
    def test_completed_bundle_is_signed_before_strict_verification(self, run) -> None:
        app = Path("/tmp/posture-test.app")
        build_macos_app.sign_and_verify(app)
        self.assertEqual(
            run.call_args_list,
            [
                call(
                    ["/usr/bin/codesign", "--force", "--sign", "-", str(app)],
                    cwd=build_macos_app.ROOT,
                    check=True,
                ),
                call(
                    ["/usr/bin/codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)],
                    cwd=build_macos_app.ROOT,
                    check=True,
                ),
            ],
        )

    def test_korean_display_name_does_not_change_ascii_executable_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            generated = dist / f"{build_macos_app.EXECUTABLE_NAME}.app"
            info = generated / "Contents/Info.plist"
            info.parent.mkdir(parents=True)
            info.write_bytes(plistlib.dumps({'CFBundleExecutable': build_macos_app.EXECUTABLE_NAME}))
            app = build_macos_app.finalize_bundle(dist)
            data = plistlib.loads((app / 'Contents/Info.plist').read_bytes())
            self.assertEqual(app.name, build_macos_app.APP_NAME + '.app')
            self.assertEqual(data['CFBundleDisplayName'], build_macos_app.APP_NAME)
            self.assertEqual(data['CFBundleExecutable'], 'PostureMonitor')
            command = build_macos_app.build_command()
            self.assertEqual(command[command.index('--name') + 1], 'PostureMonitor')

    @patch('build_macos_app.subprocess.run')
    def test_nested_framework_is_signed_before_app_and_strict_verification(self, run):
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp) / 'Posture.app'
            framework = app / 'Contents/Frameworks/Python.framework'
            framework.mkdir(parents=True)
            build_macos_app.sign_and_verify(app)
            commands = [entry.args[0] for entry in run.call_args_list]
            self.assertEqual(commands[0][-1], str(framework.resolve()))
            self.assertEqual(commands[1][-1], str(app))
            self.assertNotIn('--deep', commands[0])
            self.assertIn('--verify', commands[2])


if __name__ == "__main__":
    unittest.main()
