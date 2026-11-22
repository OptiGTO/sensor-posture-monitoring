import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
from src.model_comparison import train_comparison, _dataset, aligned_predict_proba
from src.soft_voting import aligned_indices, evaluate, validate_members
from src.app_services import TrainingJob
import test_model_comparison as comparison_tests
from test_model_comparison import SMALL_GRID, ROOT


class SoftVotingTests(unittest.TestCase):
    def test_members_reject_duplicates_unknown_and_single(self):
        for members in [('rbf_svm:2',), ('rbf_svm:2', 'rbf_svm:2'), ('rbf_svm:2','extra_trees:3')]:
            with self.assertRaises(ValueError):
                validate_members(members, ('rbf_svm',), (2,3))

    def test_mixed_windows_lock_average_and_no_ensemble_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comparison_tests.ModelComparisonTests().make_data(root/'data', rows=145)
            members = ('rbf_svm:2', 'extra_trees:3')
            final_inputs = []
            for seconds in (2,3):
                data = _dataset(root/'data', seconds, 41)
                final_inputs.append(data['X'][data['masks']['final_evaluation']])
            def probability(model, features, count):
                if any(np.array_equal(features, x) for x in final_inputs):
                    locked = json.loads((root/'run/artifacts/model_comparison.json').read_text())
                    self.assertTrue(locked['selection_locked'])
                    self.assertTrue(locked['soft_voting']['selection_locked_at'])
                    self.assertEqual(locked['soft_voting']['members'], list(members))
                return aligned_predict_proba(model, features, count)
            with mock.patch('src.model_comparison.MODEL_CANDIDATES', SMALL_GRID), mock.patch('src.model_comparison.aligned_predict_proba', side_effect=probability):
                report = train_comparison(root/'data', root/'run', (2,3), 41,
                    ('rbf_svm','extra_trees'), soft_voting_members=members, progress=lambda _:None)
            soft = report['soft_voting']
            self.assertEqual(soft['weights'], [.5,.5])
            self.assertEqual(len(soft['models']), 3)
            for split in ('validation','final_evaluation'):
                evidence = json.loads((root/f'run/artifacts/soft_voting_{split}.json').read_text())
                expected = np.mean(list(evidence['member_probabilities'].values()), axis=0)
                np.testing.assert_allclose(expected, evidence['mean_probabilities'])
                np.testing.assert_array_equal(expected.argmax(axis=1), evidence['predictions'])
                self.assertEqual({r['nominal_end_s'] for r in evidence['identities']}, {6.,12.})
                self.assertEqual(soft['counts'][split], 14)
            job = TrainingJob(ROOT, (), 41, state_root=root)
            job.run_dir = root/'run'
            self.assertEqual(len(job._write_pending_candidate_manifests()), 4)

    def test_no_common_points_fails_before_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            comparison_tests.ModelComparisonTests().make_data(root/'data', rows=49)
            with mock.patch('src.model_comparison.build_model') as build, self.assertRaisesRegex(ValueError, '공통 평가 시점'):
                train_comparison(root/'data', root/'run', (2,3), 41, ('rbf_svm',),
                    soft_voting_members=('rbf_svm:2','rbf_svm:3'), progress=lambda _:None)
            build.assert_not_called()

    def test_alignment_is_identity_based_and_rejects_different_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            comparison_tests.ModelComparisonTests().make_data(root/'data', rows=73)
            datasets={seconds:_dataset(root/'data', seconds, 41) for seconds in (2,3)}
            members=('rbf_svm:2','extra_trees:3')
            indices, identities=aligned_indices(datasets,members,'validation')
            self.assertEqual(len(identities),7)
            self.assertTrue(np.all(indices[2] % 3 == 2))
            datasets[3]['partitions']['train']['subjects']=['wrong']
            with self.assertRaisesRegex(ValueError, '피험자 분할'):
                aligned_indices(datasets,members,'validation')
