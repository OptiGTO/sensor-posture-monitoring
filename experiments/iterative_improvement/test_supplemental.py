"""Synthetic-only checks for ordinary-reservation training augmentation."""
from __future__ import annotations

from copy import deepcopy
import ast
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.iterative_improvement import supplemental as subject
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS, feature_names, read_sensor_csv


class SupplementalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.origin, self.out = self.root / "origin", self.root / "output"
        self.origin.mkdir()
        self.base = {"X": np.arange(21 * 596, dtype=float).reshape(21, 596),
            "y": np.tile(np.arange(7), 3), "groups": np.repeat(["a", "b", "c"], 7),
            "record_ids": np.repeat(["a/base.csv", "b/base.csv", "c/base.csv"], 7)}
        np.savez_compressed(self.origin / "data.npz", **self.base)
        sources = {}
        for relative, module in subject.CORE_SOURCES.items():
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(module.__file__, destination)
            sources[relative] = subject._digest(destination)
        reserved, originals = [], {}
        rng = np.random.default_rng(290)
        for name in ("a", "b", "c"):
            path = self.root / "data" / name / "correct.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(rng.uniform(10, 15, (73, 16)), columns=SENSOR_COLUMNS)
            frame.insert(0, "label", "정자세")
            frame.insert(0, "elapsed_s", np.arange(73) / 12)
            frame.to_csv(path, index=False)
            reserved.append({"source_relative_path": path.relative_to(self.root / "data").as_posix(),
                "subject": name, "label": "정자세", "sha256": subject._digest(path),
                "sensor_sha256": subject._sensor_digest(read_sensor_csv(path)),
                "reference_mean_first_10s": [987654321.] * 16})
            originals[path.relative_to(self.root).as_posix()] = {"sha256": subject._digest(path)}
        self.dataset = {"cache_sha256": subject._digest(self.origin / "data.npz"),
            "window_seconds": 3, "sampling_rate_hz": 12, "stride_seconds": 3, "min_tail_fraction": .95,
            "feature_dimension": 596, "class_names": CLASS_NAMES, "source_snapshot_hashes": sources,
            "cohort": {"subjects": 3}, "input_originals": originals,
            "manifest_snapshots": {name: {"exists": False, "sha256": None}
                for name in ("training_exclusions.json", "training_label_corrections.json")},
            "registrations_excluded": reserved,
            "records": [{"source_relative_path": f"{name}/base.csv", "sha256": f"{index + 1:064x}",
                "sensor_sha256": f"{index + 11:064x}"} for index, name in enumerate(("a", "b", "c"))]}
        self.save_dataset()

    def save_dataset(self):
        (self.origin / "dataset.json").write_text(json.dumps(self.dataset, ensure_ascii=False))

    def set_sidecar(self, payload, *, frozen=True):
        path = self.root / "data/a/correct.manifest.json"
        path.write_text(json.dumps(payload))
        if frozen:
            self.dataset["input_originals"][path.relative_to(self.root).as_posix()] = {"sha256": subject._digest(path)}
            self.save_dataset()

    def prepare(self):
        manifest = subject.prepare_supplement(self.out, self.root, self.origin)
        with np.load(self.out / "supplemental_training.npz", allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        return manifest, arrays

    def make_pre_helper_snapshot(self):
        current = ast.parse((self.root / "src/posture_core.py").read_text())
        current.body = [node for node in current.body if not (
            isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == "hashlib"
            or isinstance(node, ast.FunctionDef) and node.name == "sensor_identity")]
        path = self.origin / "source_snapshot/src/posture_core.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ast.unparse(current) + "\n")
        self.dataset["source_snapshot_hashes"]["src/posture_core.py"] = subject._digest(path)
        self.save_dataset()
        return path

    def test_preparation_is_audited_without_raw_copy_reference_or_unknown_subject(self):
        self.dataset["registrations_excluded"].append({"source_relative_path": "unknown/not_read.csv", "subject": "unknown"})
        self.save_dataset()
        originals = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (self.root / "data").rglob("*") if path.is_file()}
        manifest, arrays = self.prepare()
        self.assertEqual((manifest["record_count"], manifest["window_count"]), (3, 6))
        self.assertEqual(arrays["X"].shape, (6, 596))
        self.assertEqual(arrays["seq"].shape, (6, 36, 16))
        self.assertEqual(set(arrays["groups"]), {"a", "b", "c"})
        self.assertEqual(manifest["omitted"][0]["reason"], "not_in_frozen_subject_cohort")
        self.assertEqual(manifest["cache_sha256"], subject._digest(self.out / "supplemental_training.npz"))
        self.assertFalse(manifest["personal_reference_used"])
        self.assertNotIn("reference_mean_first_10s", json.dumps(manifest))
        self.assertEqual({path.name for path in self.out.iterdir()}, {"supplemental_training.npz", "supplemental_manifest.json"})
        for path, (content, modified) in originals.items():
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(path.stat().st_mtime_ns, modified)
        with self.assertRaisesRegex(ValueError, "immutable"):
            subject.prepare_supplement(self.out, self.root, self.origin)

    def test_only_selected_complete_training_subjects_are_appended(self):
        _, arrays = self.prepare()
        before = {key: value.copy() for key, value in self.base.items()}
        before_extra = {key: value.copy() for key, value in arrays.items()}
        X, y, groups, audit = subject.append_training(self.base, arrays, np.arange(7))
        self.assertEqual(X.shape, (9, 596))
        self.assertEqual(set(groups), {"a"})
        np.testing.assert_array_equal(y[-2:], [0, 0])
        np.testing.assert_array_equal(X[:7], self.base["X"][:7])
        self.assertEqual(audit["added_record_paths"], ["a/correct.csv"])
        self.assertEqual((audit["added_records"], audit["added_windows"]), (1, 2))
        self.assertEqual(audit["class_counts_after"], [3, 1, 1, 1, 1, 1, 1])
        for key, value in before.items():
            np.testing.assert_array_equal(self.base[key], value)
        for key, value in before_extra.items():
            np.testing.assert_array_equal(arrays[key], value)
        _, _, final_groups, final_audit = subject.append_training(self.base, arrays, np.arange(14))
        self.assertEqual(set(final_groups), {"a", "b"})
        self.assertEqual(final_audit["added_windows"], 4)

    def test_explicit_registration_and_calibration_are_never_restored(self):
        for key, value in (("mode", "registration"), ("source_type", "calibration"),
                ("purpose", "personal_registration"), ("training_role", "registration")):
            self.set_sidecar({key: value, "label_status": "labeled"})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare()
        self.assertFalse(self.out.exists())

    def test_noneligible_sidecar_and_added_or_changed_sidecar_fail_closed(self):
        self.set_sidecar({"training_role": "exclude"})
        with self.assertRaisesRegex(ValueError, "eligible"):
            self.prepare()
        self.set_sidecar({"training_role": "include"})
        (self.root / "data/a/correct.manifest.json").write_text('{"training_role":"include", "comment":"changed"}')
        with self.assertRaisesRegex(ValueError, "sidecar"):
            self.prepare()
        del self.dataset["input_originals"]["data/a/correct.manifest.json"]
        self.save_dataset()
        with self.assertRaisesRegex(ValueError, "sidecar"):
            self.prepare()

    def test_source_core_cache_and_policy_hash_changes_fail_closed(self):
        original = (self.root / "data/a/correct.csv").read_bytes()
        with (self.root / "data/a/correct.csv").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "source hash"):
            self.prepare()
        (self.root / "data/a/correct.csv").write_bytes(original)
        source = self.root / "src/posture_core.py"
        original_core = source.read_bytes()
        source.write_bytes(original_core + b"\n")
        with self.assertRaisesRegex(ValueError, "Core feature source"):
            self.prepare()
        source.write_bytes(original_core)
        manifest = self.root / "research/models/training_exclusions.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{"schema_version":1,"exclusions":[]}')
        with self.assertRaisesRegex(ValueError, "Eligibility manifest changed"):
            self.prepare()
        manifest.unlink()
        with (self.origin / "data.npz").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Origin feature cache"):
            self.prepare()

    def test_hash_bound_global_exclusion_cannot_be_restored(self):
        record = self.dataset["registrations_excluded"][0]
        path = self.root / "research/models/training_exclusions.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"schema_version": 1, "exclusions": [{
            "source_file": record["source_relative_path"], "source_sha256": record["sha256"],
            "reason": "synthetic exclusion", "requested_at": "2026-09-22T00:00:00+09:00"}]}))
        self.dataset["manifest_snapshots"][path.name] = {"exists": True, "sha256": subject._digest(path)}
        self.save_dataset()
        with self.assertRaisesRegex(ValueError, "eligible"):
            self.prepare()

    def test_duplicate_sensor_content_and_base_source_path_are_rejected(self):
        self.dataset["records"][0]["sensor_sha256"] = self.dataset["registrations_excluded"][0]["sensor_sha256"]
        self.save_dataset()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.prepare()
        self.dataset["records"][0]["sensor_sha256"] = "f" * 64
        a, b = self.dataset["registrations_excluded"][:2]
        shutil.copyfile(self.root / "data" / a["source_relative_path"], self.root / "data" / b["source_relative_path"])
        b["sha256"], b["sensor_sha256"] = a["sha256"], a["sensor_sha256"]
        self.save_dataset()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.prepare()

    def test_append_rejects_evaluation_overlap_unknown_subject_and_partial_person(self):
        _, supplement = self.prepare()
        for indices in (np.arange(6), np.array([0, 0]), np.array([999]), np.array([.5])):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                subject.append_training(self.base, supplement, indices)
        bad = deepcopy(supplement)
        bad["record_ids"][0] = "c/base.csv"
        with self.assertRaisesRegex(ValueError, "duplicates"):
            subject.append_training(self.base, bad, np.arange(7))
        bad = deepcopy(supplement)
        bad["groups"][0] = "z"
        with self.assertRaises(ValueError):
            subject.append_training(self.base, bad, np.arange(7))
        bad = deepcopy(supplement)
        bad["y"][0] = 4
        with self.assertRaisesRegex(ValueError, "correct-posture"):
            subject.append_training(self.base, bad, np.arange(7))

    def test_output_cannot_be_placed_under_raw_data_or_origin(self):
        for out in (self.root / "data/new-run", self.origin):
            with self.subTest(out=out), self.assertRaisesRegex(ValueError, "isolated"):
                subject.prepare_supplement(out, self.root, self.origin)

    def test_only_additive_identity_helper_accepts_verified_ast_compatible_core(self):
        original = self.make_pre_helper_snapshot()
        manifest, arrays = self.prepare()
        detail = manifest["feature_source_compatibility"]["src/posture_core.py"]
        self.assertEqual(detail["compatibility"], "additive_sensor_identity_helper_only_ast_identical_features")
        self.assertEqual(detail["original_sha256"], subject._digest(original))
        self.assertEqual(detail["current_sha256"], subject._digest(self.root / "src/posture_core.py"))
        self.assertNotEqual(detail["original_sha256"], detail["current_sha256"])
        self.assertEqual(manifest["feature_source_compatibility"]["src/window_pipeline.py"]["compatibility"], "exact_source_hash")
        self.assertEqual(arrays["X"].shape, (6, 596))

    def test_feature_code_change_is_rejected_even_with_added_identity_helper(self):
        self.make_pre_helper_snapshot()
        path = self.root / "src/posture_core.py"
        module = ast.parse(path.read_text())
        extraction = next(node for node in module.body if isinstance(node, ast.FunctionDef)
            and node.name == "extract_feature_vector")
        returned = next(node for node in ast.walk(extraction) if isinstance(node, ast.Return))
        returned.value = ast.BinOp(left=returned.value, op=ast.Mult(), right=ast.Constant(2.))
        path.write_text(ast.unparse(ast.fix_missing_locations(module)))
        with patch.object(subject.posture_core, "__file__", str(path)):
            with self.assertRaisesRegex(ValueError, "Core feature source differs"):
                self.prepare()
        self.assertFalse(self.out.exists())

    def test_compatibility_requires_authentic_origin_and_loaded_source(self):
        original = self.make_pre_helper_snapshot()
        original.write_text(original.read_text() + "\n# changed original snapshot\n")
        with self.assertRaisesRegex(ValueError, "original core source snapshot"):
            self.prepare()
        self.make_pre_helper_snapshot()
        current = self.root / "src/posture_core.py"
        current.write_text(current.read_text() + "\n# different from imported source\n")
        with self.assertRaisesRegex(ValueError, "loaded by Python"):
            self.prepare()

    def test_added_helper_decorators_or_default_expressions_are_rejected(self):
        original = self.make_pre_helper_snapshot()
        path = self.root / "src/posture_core.py"
        initial = path.read_text()
        for change in ("decorator", "default"):
            module = ast.parse(initial)
            helper = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                and node.name == "sensor_identity")
            if change == "decorator":
                helper.decorator_list = [ast.Name(id="staticmethod", ctx=ast.Load())]
            else:
                helper.args.defaults = [ast.Constant(None)]
            path.write_text(ast.unparse(ast.fix_missing_locations(module)))
            with self.subTest(change=change):
                self.assertFalse(subject._additive_sensor_identity_only(original, path))


if __name__ == "__main__":
    unittest.main()
