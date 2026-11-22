import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from src.app_services import TrainingWizardState
from src.experiment_workflow import CONFIGURATIONS, build_configuration, apply_configuration, adapt_report, validate_registration
from src.comparison_charts import matrix_values, posture_labels
from src.posture_core import CLASS_NAMES


class ExperimentWorkflowTests(unittest.TestCase):
    def test_personal_excludes_inactive_model_window_and_ensemble_drafts(self):
        state = TrainingWizardState(configuration_type='personal', windows=(20,), models=('xgboost',),
                                    soft_voting_members=('xgboost:20', 'extra_trees:20'))
        config = build_configuration(state)
        self.assertEqual(config['windows'], [3])
        self.assertEqual(config['models'], ['rbf_svm', 'logistic_regression'])
        self.assertEqual(config['soft_voting_members'], [])
        self.assertTrue(config['personalized'])
        self.assertFalse(config['allow_retrospective'])
        with self.assertRaisesRegex(ValueError, '해제'):
            build_configuration(state, allow_retrospective=True)
        self.assertEqual(state.windows, (20,))
        self.assertEqual(state.models, ('xgboost',))

    def test_ensemble_requires_explicit_valid_members_and_ignores_personal_policy(self):
        state = TrainingWizardState(configuration_type='ensemble')
        with self.assertRaisesRegex(ValueError, '2개 이상'):
            build_configuration(state)
        state.soft_voting_members = ('rbf_svm:3', 'extra_trees:3')
        config = build_configuration(state, allow_retrospective=True)
        self.assertFalse(config['allow_retrospective'])
        self.assertFalse(config['personalized'])
        self.assertEqual(config['soft_voting_members'], list(state.soft_voting_members))
        state.soft_voting_members = ('rbf_svm:3', 'rbf_svm:3')
        with self.assertRaises(ValueError): build_configuration(state)

    def test_registry_contains_only_existing_configurations(self):
        self.assertEqual(set(CONFIGURATIONS), {'individual','ensemble','personal'})
        state = TrainingWizardState(seed=-1)
        with self.assertRaises(ValueError): build_configuration(state)
        state.seed=41; state.experiment_name=' '
        with self.assertRaises(ValueError): build_configuration(state)

    def test_clone_restores_configuration_without_editing_saved_snapshot(self):
        source = TrainingWizardState(configuration_type='ensemble', windows=(2,3),
            soft_voting_members=('rbf_svm:3','extra_trees:2'), experiment_name='원본')
        config = build_configuration(source)
        original = copy.deepcopy(config)
        target = TrainingWizardState()
        apply_configuration(target, config)
        self.assertEqual(target.soft_voting_members, source.soft_voting_members)
        self.assertEqual(target.windows, (2,3))
        self.assertEqual(target.experiment_name, '원본 · 복제')
        self.assertEqual(config, original)

    def test_registration_missing_and_mixed_are_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(csv_path=Path(tmp)/'a.csv',subject='A')
            b=SimpleNamespace(csv_path=Path(tmp)/'b.csv',subject='A')
            config=build_configuration(TrainingWizardState(configuration_type='personal'))
            with self.assertRaisesRegex(ValueError,'등록 정보가 없습니다'):
                validate_registration([a,b],config)
            a.csv_path.with_suffix('.manifest.json').write_text(json.dumps({'guidance':{'reference_profile':{'user_id':'A'}}}))
            config['allow_retrospective']=True
            with self.assertRaisesRegex(ValueError,'함께 사용할 수 없습니다'):
                validate_registration([a,b],config)
            validate_registration([a],config)

    def test_personal_adapter_preserves_abstentions_and_original_report(self):
        labels=[*CLASS_NAMES,'판단 유보 / 종류 미확인']
        metric={'confusion_labels':labels,'confusion_matrix':[[int(i==j) for j in range(8)] for i in range(8)],
                'per_class':{name:{'f1':.5,'support':1} for name in CLASS_NAMES},'accuracy':.5,'macro_f1':.5,'unknown_rate':.2}
        report={'registration_records':[], 'results':{split:{key:metric for key in ('single_7class','generic_hierarchy','personal_hierarchy')} for split in ('validation','final_evaluation')}}
        original=copy.deepcopy(report); adapted=adapt_report(report)
        self.assertEqual(len(adapted['models']),3)
        row=adapted['models'][-1]['final_evaluation']
        self.assertEqual(len(posture_labels(adapted,row)),7)
        names,matrix=matrix_values(adapted,row)
        self.assertEqual(len(names),8); self.assertEqual(matrix.shape,(8,8))
        self.assertEqual(row['unknown_rate'],.2)
        self.assertEqual(report,original)
