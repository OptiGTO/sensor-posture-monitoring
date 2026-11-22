"""Synthetic-only sequence augmentation guards; no project measurements used."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments.iterative_improvement import supplemental as base
from experiments.iterative_improvement import supplemental_sequence as subject
from experiments.iterative_improvement import test_supplemental as fixtures
from src.window_pipeline import extract_window_features


class SupplementalSequenceTests(unittest.TestCase):
    save_dataset = fixtures.SupplementalTests.save_dataset
    set_sidecar = fixtures.SupplementalTests.set_sidecar

    def setUp(self):
        fixtures.SupplementalTests.setUp(self)
        rng = np.random.default_rng(901)
        self.base["seq"] = rng.uniform(20, 90, (21, 36, 16)).astype(np.float32)
        self.base["X"] = extract_window_features(self.base["seq"])
        self.base["start_s"] = np.tile(np.arange(7) * 3., 3)
        np.savez_compressed(self.origin / "data.npz", **self.base)
        self.dataset["cache_sha256"] = base._digest(self.origin / "data.npz")
        self.save_dataset()

    def prepare(self, out=None):
        out = self.out if out is None else out
        manifest = subject.prepare_supplement_sequences(out, self.root, self.origin)
        with np.load(out / "supplemental_training.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        return manifest, arrays

    def test_preparation_keeps_cache_bytes_and_proves_all_sequence_rows(self):
        original = base.prepare_supplement
        produced = []

        def capture(*args):
            result = original(*args)
            produced.append((self.out / "supplemental_training.npz").read_bytes())
            return result

        raw_before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in (self.root / "data").rglob("*") if p.is_file()}
        origin_before = (self.origin / "data.npz").read_bytes()
        with patch.object(base, "prepare_supplement", side_effect=capture):
            manifest, arrays = self.prepare()
        self.assertEqual((self.out / "supplemental_training.npz").read_bytes(), produced[0])
        self.assertEqual(manifest["cache_sha256"], base._digest(self.out / "supplemental_training.npz"))
        evidence = manifest["sequence_evidence"]
        self.assertEqual(evidence["shape"], [6, 36, 16])
        self.assertTrue(evidence["feature_parity"]["bitwise_equal"])
        self.assertEqual(evidence["feature_parity"]["checked_windows"], 6)
        self.assertTrue(evidence["cache_bytes_reused_without_rewrite"])
        np.testing.assert_array_equal(extract_window_features(arrays["seq"]), arrays["X"])
        self.assertEqual(manifest["supplemental_source_hashes"]["experiments/iterative_improvement/supplemental_sequence.py"],
                         base._digest(subject.__file__))
        self.assertEqual(set(p.name for p in self.out.iterdir()), {"supplemental_training.npz", "supplemental_manifest.json"})
        self.assertEqual((self.origin / "data.npz").read_bytes(), origin_before)
        for path, (content, modified) in raw_before.items():
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(path.stat().st_mtime_ns, modified)

    def test_sequence_and_feature_training_rows_audits_are_identical(self):
        _, supplement = self.prepare()
        original = deepcopy(self.base)
        original_supplement = deepcopy(supplement)
        for train in (np.arange(7), np.arange(14), np.arange(13, -1, -1)):
            features, y, groups, expected_audit = base.append_training(self.base, supplement, train)
            seq, actual_y, actual_groups, audit = subject.append_sequence_training(self.base, supplement, train)
            np.testing.assert_array_equal(extract_window_features(seq), features)
            np.testing.assert_array_equal(actual_y, y)
            np.testing.assert_array_equal(actual_groups, groups)
            np.testing.assert_array_equal(seq[:len(train)], self.base["seq"][train])
            self.assertEqual(audit, expected_audit)
            self.assertTrue(set(groups).issubset(set(self.base["groups"][train])))
        for key in original:
            np.testing.assert_array_equal(self.base[key], original[key])
        for key in original_supplement:
            np.testing.assert_array_equal(supplement[key], original_supplement[key])

    def test_existing_output_and_sealed_run_are_not_modified(self):
        self.prepare()
        before = {p: p.read_bytes() for p in self.out.iterdir()}
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.prepare()
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        for marker in subject.SEALED_MARKERS:
            out = self.root / ("sealed_" + marker)
            out.mkdir()
            (out / marker).write_text("frozen")
            with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, "unsealed"):
                self.prepare(out)
            self.assertEqual([p.name for p in out.iterdir()], [marker])

    def test_original_eligibility_and_hash_checks_remain_required(self):
        self.set_sidecar({"mode": "registration", "label_status": "labeled"})
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.out.exists())
        path = self.root / "data/a/correct.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "source hash"):
            self.prepare()

    def test_preparation_rejects_misaligned_sequence_feature_and_identity_rows(self):
        original = base.prepare_supplement
        for field in ("seq", "X", "groups", "record_ids", "start_s"):
            out = self.root / ("corrupt_" + field)

            def corrupted(*args):
                manifest = original(*args)
                cache = out / "supplemental_training.npz"
                with np.load(cache, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                if field in {"seq", "X"}:
                    arrays[field] = arrays[field][::-1].copy()
                elif field == "start_s":
                    arrays[field][0] += 1
                else:
                    arrays[field][0] = arrays[field][-1]
                np.savez_compressed(cache, **arrays)
                manifest["cache_sha256"] = base._digest(cache)
                (out / "supplemental_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
                return manifest

            with self.subTest(field=field), patch.object(base, "prepare_supplement", side_effect=corrupted):
                with self.assertRaises(ValueError):
                    self.prepare(out)
            self.assertNotIn("sequence_evidence", base._read(out / "supplemental_manifest.json"))

    def test_append_retains_leakage_guards_and_rejects_invalid_sequences(self):
        _, supplement = self.prepare()
        for train in (np.arange(6), np.array([0, 0]), np.array([999]), np.array([.5])):
            with self.subTest(train=train), self.assertRaises(ValueError):
                subject.append_sequence_training(self.base, supplement, train)
        for field, value in (("record_ids", "c/base.csv"), ("groups", "z"), ("y", 4)):
            bad = deepcopy(supplement)
            bad[field][0] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject.append_sequence_training(self.base, bad, np.arange(7))
        for changed in (supplement["seq"][:, :-1], supplement["seq"].astype(np.float64),
                        supplement["seq"][::-1].copy(), supplement["seq"] * np.nan):
            bad = {**supplement, "seq": changed}
            with self.subTest(shape=changed.shape, dtype=changed.dtype), self.assertRaises(ValueError):
                subject.append_sequence_training(self.base, bad, np.arange(7))
        bad_base = {k: v for k, v in self.base.items() if k != "seq"}
        with self.assertRaisesRegex(ValueError, "missing seq"):
            subject.append_sequence_training(bad_base, supplement, np.arange(7))


if __name__ == "__main__":
    unittest.main()
