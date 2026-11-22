from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data_review import audit, inspect_record, verify_unchanged
from src.posture_core import SENSOR_COLUMNS


class DataReviewTests(unittest.TestCase):
    def make(self, root, subject='person', name='person_정자세_001.csv', label='정자세', gap=False):
        path=Path(root)/subject/name;path.parent.mkdir(parents=True,exist_ok=True)
        rng=np.random.default_rng(41)
        data=rng.normal(2,.01,(120,16))+np.arange(16)*.03
        frame=pd.DataFrame(data,columns=SENSOR_COLUMNS)
        time=np.arange(120)/12
        if gap:time[60:]+=2
        frame.insert(0,'label',label);frame.insert(0,'elapsed_s',time)
        frame.to_csv(path,index=False)
        return path

    def test_gap_blocks_model_windows_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.make(root,gap=True);before=p.read_bytes()
            r,_=inspect_record(p,root)
            self.assertTrue(r['blocking']);self.assertFalse(r['eligible'])
            self.assertIn('long_gap',[f['code'] for f in r['flags']]);self.assertEqual(p.read_bytes(),before)

    def test_typographical_label_is_a_proposal_not_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.make(root,name="person_'뒤로 기대기_001.csv",label="'뒤로 기대기")
            r,_=inspect_record(p,root)
            self.assertEqual(r['label'],'미확인');self.assertEqual(r['suggested_label'],'뒤로 기대기')
            self.assertFalse(r['eligible']);self.assertTrue(r['blocking'])

    def test_duplicates_across_subjects_are_flagged_and_measurements_stay_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.make(root)
            copy=root/'other'/'other_정자세_001.csv';copy.parent.mkdir();copy.write_bytes(p.read_bytes())
            self.make(root,subject='measurements')
            report,_=audit(root);verify_unchanged(report)
            self.assertEqual(len(report['duplicate_groups']),1)
            self.assertEqual(report['summary']['excluded_records'],1)
            self.assertEqual(report['summary']['eligible_records'],0)
            self.assertTrue(report['source_unchanged'])

    def test_constant_sensor_nominates_review_but_does_not_delete_or_exclude(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.make(root);f=pd.read_csv(p);f['R11']=1;f.to_csv(p,index=False)
            r,_=inspect_record(p,root)
            self.assertTrue(r['eligible']);self.assertIn('constant_channel',[f['code'] for f in r['flags']])

    def test_hash_check_rejects_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.make(root);report,_=audit(root);p.write_text('changed')
            with self.assertRaises(RuntimeError):verify_unchanged(report)

    def test_prepared_windows_keep_subject_and_record_linkage(self):
        from audit_sensor_data import prepare_windows
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';p=self.make(root);before=p.read_bytes()
            report,frames=audit(root);out=Path(tmp)/'output';out.mkdir()
            summary=prepare_windows(report,frames,out)
            with np.load(out/'prepared_windows_5s.npz',allow_pickle=False) as data:
                self.assertEqual(data['sequences'].shape,(2,60,16))
                self.assertEqual(data['subjects'].tolist(),['person','person'])
                self.assertEqual(set(data['record_ids']),{report['records'][0]['id']})
                self.assertEqual(data['start_seconds'].tolist(),[0,5])
            self.assertEqual(summary['source_records'],1);self.assertEqual(before,p.read_bytes())

    def test_source_snapshot_detects_new_files_during_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make(root);report,_=audit(root)
            self.make(root,name='person_정자세_002.csv')
            with self.assertRaises(RuntimeError):verify_unchanged(report)


if __name__=='__main__':unittest.main()
