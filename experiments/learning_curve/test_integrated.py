import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from experiments.learning_curve.integrated import assert_no_leakage
from experiments.learning_curve.integrated_report import collect
from experiments.learning_curve.resume_frozen import verify_evidence, verify_originals
from experiments.learning_curve.run import digest


class IntegratedEvaluationTests(unittest.TestCase):
    def test_recovery_rejects_modified_frozen_source(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp); frozen=out/'source_snapshot/src/example.py'
            frozen.parent.mkdir(parents=True); frozen.write_text('original')
            (out/'data.npz').write_bytes(b'cached data')
            p={'cache_sha256':digest(out/'data.npz'),'source_hashes':{'src/example.py':digest(frozen)}}
            (out/'protocol.json').write_text(json.dumps(p))
            (out/'protocol_lock.json').write_text(json.dumps({'sha256':digest(out/'protocol.json')}))
            verify_evidence(out)
            frozen.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Frozen evidence changed'):
                verify_evidence(out)

    def test_recovery_rejects_modified_original_data(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp); (out/'protected_before.json').write_text(json.dumps({'data/raw.csv':{'sha256':'original'}}))
            with patch('experiments.learning_curve.resume_frozen.protected_snapshot',return_value={'data/raw.csv':{'sha256':'changed'}}):
                with self.assertRaisesRegex(ValueError,'original protected file changed'):
                    verify_originals(out)

    def test_reject_same_person_even_when_rows_differ(self):
        data = {"groups": np.repeat(["a", "b"], 14), "record_ids": np.array([f"r{i}" for i in range(28)]),
                "y": np.tile(np.arange(7), 4)}
        with self.assertRaisesRegex(ValueError, "person overlap"):
            assert_no_leakage(data, np.arange(7), np.arange(7,14))
        assert_no_leakage(data, np.arange(14), np.arange(14,28))

    def test_reject_duplicate_record_identity_across_subjects(self):
        data = {"groups": np.repeat(["a", "b"], 7), "record_ids": np.tile([f"r{i}" for i in range(7)],2),
                "y": np.tile(np.arange(7), 2)}
        with self.assertRaisesRegex(ValueError, "recording overlap"):
            assert_no_leakage(data, np.arange(7), np.arange(7,14))

    def test_training_score_cannot_enter_generalization_report(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp); (out/'training_complete.json').write_text('{}')
            np.savez(out/'data.npz', groups=np.array(['a','b']), y=np.array([0,1]))
            dest=out/'fits/xgboost/s/subjects_04'; dest.mkdir(parents=True)
            (dest/'metrics.json').write_text(json.dumps({'score_kind':'training_resubstitution_only_not_generalization'}))
            p={'splits':[{'id':'s','arms':[{'id':'subjects_04'}]}]}
            with patch('experiments.learning_curve.integrated_report.verify',return_value=p):
                with self.assertRaisesRegex(ValueError,'Resubstitution metrics'):
                    collect(out)


if __name__=='__main__': unittest.main()
