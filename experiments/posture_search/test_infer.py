"""Unlabeled inference contract and lock validation; synthetic data only."""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from .data import digest, write_json
from .infer import infer_recording, make_inputs
from .run import Arm, Fusion, Temporal
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS


class InferenceTests(unittest.TestCase):
    def _csv(self, path, value, label=None):
        elapsed = np.arange(145) / 12.0
        frame = pd.DataFrame(np.full((len(elapsed), 16), value), columns=SENSOR_COLUMNS)
        frame.insert(0, "elapsed_s", elapsed)
        if label is not None:
            frame["label"] = label
        frame.to_csv(path, index=False)

    def test_input_has_no_labels_and_reference_matches_study_formula(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._csv(root / "query.csv", 200, label="untrusted invalid label")
            self._csv(root / "reference.csv", 100, label="also not read")
            data, metadata, provenance = make_inputs(root / "query.csv", root / "reference.csv", True)
            self.assertNotIn("y", data)
            self.assertNotIn("label", data)
            self.assertEqual(data["seq"].shape, (4, 36, 16))
            np.testing.assert_allclose(data["personal"][:, :16], 1 / 16)
            np.testing.assert_allclose(data["personal"][:, 16:32], 0)
            np.testing.assert_allclose(data["personal"][:, 32:48], np.log(201 / 101))
            self.assertFalse(provenance["labels_used"])
            with self.assertRaisesRegex(ValueError, "requires --reference"):
                make_inputs(root / "query.csv", require_reference=True)
            with self.assertRaisesRegex(ValueError, "separate recording"):
                make_inputs(root / "query.csv", root / "query.csv", True)

    def test_locked_composite_infers_without_labels_and_rejects_checkpoint_change(self):
        rng = np.random.default_rng(313)
        data = {"seq": rng.uniform(10, 500, (21, 36, 16)).astype(np.float32),
                "y": np.tile(np.arange(7), 3), "groups": np.repeat(["a", "b", "c"], 7)}
        spec = {"id": "H02", "family": "classical", "kind": "baseline", "features": "spatial"}
        arm = Arm(spec, 8).fit(data)
        model = Temporal(Fusion([arm, arm]), "ema")
        selected = {"id": "C01", "family": "temporal", "kind": "ema"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "C01/primary").mkdir(parents=True)
            checkpoint = root / "C01/primary/model.joblib"
            joblib.dump(model, checkpoint)
            write_json(root / "dataset.json", {
                "fingerprint": "synthetic", "class_names": CLASS_NAMES,
                "window_seconds": 3, "sampling_rate": 12, "stride_seconds": 3, "min_tail_fraction": .95,
            })
            write_json(root / "protocol.json", {"test": "synthetic_only"})
            write_json(root / "selection_lock.json", {
                "selected_id": "C01", "selected_spec": selected,
                "test_used_for_selection": False, "dataset_fingerprint": "synthetic",
                "protocol_sha256": digest(root / "protocol.json"), "source_hashes": {},
                "checkpoints": {"C01/primary/model.joblib": digest(checkpoint)},
            })
            self._csv(root / "query.csv", 200)
            result = infer_recording(root / "query.csv", study_dir=root)
            self.assertEqual(result["window_count"], 4)
            self.assertFalse(result["runtime_model_activated"])
            self.assertFalse(result["input"]["labels_used"])
            json.dumps(result, allow_nan=False)
            checkpoint.write_bytes(checkpoint.read_bytes() + b"modified")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                infer_recording(root / "query.csv", study_dir=root)


if __name__ == "__main__":
    unittest.main()
