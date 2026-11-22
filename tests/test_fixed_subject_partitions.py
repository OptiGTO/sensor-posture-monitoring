from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.model_comparison import _dataset, train_comparison
from src.posture_core import CLASS_NAMES, SENSOR_COLUMNS


class FixedSubjectPartitionsTests(unittest.TestCase):
    def make_data(self, root):
        rng=np.random.default_rng(81)
        for subject in range(4):
            folder=root/f's{subject}';folder.mkdir(parents=True)
            for i,label in enumerate(CLASS_NAMES):
                frame=pd.DataFrame(rng.normal(2+i,.02,(37,16)),columns=SENSOR_COLUMNS)
                frame.insert(0,'label',label);frame.insert(0,'elapsed_s',np.arange(37)/12)
                frame.to_csv(folder/f'{i}.csv',index=False)

    def test_explicit_partition_persists_without_reshuffling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_data(root/'data')
            parts={'train':['s0','s3'],'validation':['s2'],'final_evaluation':['s1']}
            with mock.patch('src.model_comparison.MODEL_CANDIDATES',{'rbf_svm':[{'C':1.,'gamma':'scale'}]}):
                report=train_comparison(root/'data',root/'run',(3,),999,model_families=('rbf_svm',),
                                        fixed_subject_partitions=parts,progress=lambda _:None)
            actual=report['datasets']['3']['partitions']
            for name,subjects in parts.items():self.assertEqual(set(actual[name]['subjects']),set(subjects))
            self.assertTrue(report['selection_locked']);self.assertFalse(report['final_evaluation_used_for_selection'])

    def test_overlap_missing_subject_and_empty_partition_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_data(root)
            invalid=[{'train':['s0','s1'],'validation':['s2'],'final_evaluation':['s1','s3']},
                     {'train':['s0'],'validation':['s2'],'final_evaluation':['s3']},
                     {'train':['s0','s1','s2'],'validation':[],'final_evaluation':['s3']}]
            for parts in invalid:
                with self.subTest(parts=parts),self.assertRaises(ValueError):_dataset(root,3,17,parts)


if __name__=='__main__':unittest.main()
