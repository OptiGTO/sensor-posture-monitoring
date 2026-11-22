"""Evidence-gated recipes survive the UI/service/CLI boundaries without bypasses."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import unicodedata
from unittest import mock

from src.app_services import DatasetService, TrainingJob, TrainingWizardState
from src.experiment_workflow import apply_configuration, build_configuration
from src.training_methods import available_training_methods
import test_model_comparison as comparison_tests


ROOT = Path(__file__).resolve().parents[1]


class TrainingMethodWorkflowTests(unittest.TestCase):
    def setUp(self):
        # These fixtures exercise the legacy method registry independently of
        # whichever immutable checkpoints are installed in the real checkout.
        checkpoints = mock.patch('src.training_methods.checkpoint_method_entries', return_value=[])
        checkpoints.start()
        self.addCleanup(checkpoints.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = self.root / 'methods.json'
        self.registry.write_text(json.dumps({'schema_version': 1, 'methods': [
            {'id': 'selected', 'label': '핵심 특징 96개', 'families': ['rbf_svm', 'extra_trees'],
             'window_seconds': [3], 'evidence': {'improved': True, 'summary': '검증용 재실험 근거'}},
            {'id': 'relative', 'families': ['rbf_svm'], 'window_seconds': [3],
             'evidence': {'improved': False}},
        ]}))
        self.patch = mock.patch('src.training_methods.REGISTRY_PATH', self.registry)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_only_positive_evidence_is_public_and_configuration_clone_keeps_method(self):
        self.assertEqual([entry['id'] for entry in available_training_methods()], ['standard', 'selected'])
        state = TrainingWizardState(models=('rbf_svm',), training_method='selected')
        config = build_configuration(state)
        original = copy.deepcopy(config)
        copied = TrainingWizardState()
        apply_configuration(copied, config)
        self.assertEqual(copied.training_method, 'selected')
        self.assertEqual(build_configuration(copied)['training_method_evidence'], config['training_method_evidence'])
        self.assertEqual(config, original)
        config.pop('training_method')
        apply_configuration(copied, config)
        self.assertEqual(copied.training_method, 'standard')

    def test_unsupported_family_window_and_removed_recipe_fail_before_job_writes(self):
        for models, windows, method in ((('random_forest',), (3,), 'selected'),
                                        (('rbf_svm',), (5,), 'selected'),
                                        (('rbf_svm',), (3,), 'relative')):
            with self.subTest(models=models, windows=windows, method=method):
                state = TrainingWizardState(models=models, windows=windows, training_method=method)
                with self.assertRaises(ValueError):
                    build_configuration(state)
                job = TrainingJob(ROOT, (), 41, models=models, windows=windows,
                                  training_method=method, state_root=self.root)
                with mock.patch('subprocess.Popen') as process, self.assertRaises(ValueError):
                    job.start()
                process.assert_not_called()
                self.assertFalse(job.run_dir.exists())

    def test_personal_configuration_ignores_inactive_draft_but_job_rejects_tampering(self):
        state = TrainingWizardState(configuration_type='personal', training_method='selected')
        config = build_configuration(state)
        self.assertEqual(config['training_method'], 'standard')
        self.assertEqual(state.training_method, 'selected')
        config['training_method'] = 'selected'
        with self.assertRaisesRegex(ValueError, '개인 표준화'):
            apply_configuration(TrainingWizardState(), config)
        job = TrainingJob(ROOT, (), 41, models=('rbf_svm',), personalized=True,
                          training_method='selected', state_root=self.root)
        with self.assertRaisesRegex(ValueError, '개인 표준화'):
            job.start()
        self.assertFalse(job.run_dir.exists())

    def test_soft_voting_retains_supported_recipe_and_members(self):
        state = TrainingWizardState(configuration_type='ensemble', models=('rbf_svm', 'extra_trees'),
                                    training_method='selected', soft_voting_members=('rbf_svm:3', 'extra_trees:3'))
        config = build_configuration(state)
        self.assertEqual(config['training_method'], 'selected')
        self.assertEqual(config['soft_voting_members'], list(state.soft_voting_members))

    def test_training_job_passes_recipe_to_worker_and_records_exact_snapshot(self):
        comparison_tests.ModelComparisonTests().make_data(self.root / 'data')
        records = DatasetService(self.root).records()
        config = build_configuration(TrainingWizardState(models=('rbf_svm',), training_method='selected'))
        job = TrainingJob(ROOT, records, 41, models=('rbf_svm',), training_method='selected',
                          state_root=self.root, experiment_config=config)
        with mock.patch('src.app_services.subprocess.Popen') as process, \
             mock.patch('src.app_services.threading.Thread.start'):
            job.start()
        command = process.call_args.args[0]
        self.assertEqual(command[command.index('--training-method') + 1], 'selected')
        snapshot = json.loads((job.run_dir / 'experiment.json').read_text())
        self.assertEqual(snapshot['training_method'], 'selected')
        self.assertEqual(snapshot['training_method_evidence']['improved'], True)

    def test_cli_propagates_recipe_and_rejects_incompatible_modes(self):
        from train_window_models import main, train_deployment
        base = ['train_window_models.py', '--data', str(self.root / 'data'),
                '--output-dir', str(self.root / 'out'), '--windows', '3', '--training-method', 'selected']
        with mock.patch('sys.argv', [*base, '--deployment', '--models', 'rbf_svm']), \
             mock.patch('src.model_comparison.train_comparison') as train:
            main()
        self.assertEqual(train.call_args.kwargs['training_method'], 'selected')
        for extra in ([], ['--deployment'], ['--deployment', '--models', 'rbf_svm', '--personalized']):
            with self.subTest(extra=extra), mock.patch('sys.argv', [*base, *extra]), self.assertRaises(ValueError):
                main()
        with self.assertRaises(ValueError):
            train_deployment(self.root / 'data', self.root / 'out', (3,), 41, training_method='selected')
        self.assertFalse((self.root / 'out').exists())

    def test_global_exclusions_refresh_cached_eligibility_and_block_stale_selection(self):
        comparison_tests.ModelComparisonTests().make_data(self.root / 'data')
        service = DatasetService(self.root)
        stale = service.records()[0]
        source_bytes = stale.csv_path.read_bytes()
        source_mtime = stale.csv_path.stat().st_mtime_ns
        policy = self.root / 'research/models/training_exclusions.json'
        policy.parent.mkdir(parents=True)
        policy.write_text(json.dumps({'schema_version': 1, 'exclusions': [{
            'source_file': stale.csv_path.relative_to(self.root / 'data').as_posix(),
            'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
            'reason': 'test exclusion', 'requested_at': '2026-09-21T20:00:00+09:00'}]}))
        current = next(record for record in service.records() if record.csv_path == stale.csv_path)
        self.assertFalse(service.is_training_eligible(current))
        self.assertEqual(current.status, '학습 제외 목록')
        with self.assertRaisesRegex(ValueError, '학습에 사용할 수 없는 기록'):
            DatasetService(self.root / 'isolated_state').stage_for_training([stale], self.root / 'stage')
        self.assertEqual(stale.csv_path.read_bytes(), source_bytes)
        self.assertEqual(stale.csv_path.stat().st_mtime_ns, source_mtime)
        policy.write_text(policy.read_text().replace(hashlib.sha256(source_bytes).hexdigest(), '0' * 64))
        with self.assertRaisesRegex(ValueError, '해시가 달라졌습니다'):
            service.records()

    def test_direct_cli_rejects_canonical_exclusions_before_loading_or_writing(self):
        from src.training_input_policy import validate_cli_training_input
        from train_window_models import main, train_deployment
        source = self.root / 'data/person/excluded.csv'
        source.parent.mkdir(parents=True)
        source.write_bytes(b'original source; no CSV parsing should occur')
        before = (source.read_bytes(), source.stat().st_mtime_ns)
        manifest = self.root / 'research/models/training_exclusions.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'schema_version': 1, 'exclusions': [{
            'source_file': 'person/excluded.csv', 'source_sha256': hashlib.sha256(before[0]).hexdigest(),
            'reason': 'test', 'requested_at': '2026-09-21T20:00:00+09:00'}]}))
        for root in (self.root / 'data', source.parent):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, '별도 학습 폴더'):
                validate_cli_training_input(root)
        linked = self.root / 'linked'
        linked.mkdir()
        (linked / 'alias.csv').symlink_to(source)
        with self.assertRaisesRegex(ValueError, '학습 제외 목록'):
            validate_cli_training_input(linked)
        safe_stage = self.root / 'safe_stage'
        safe_stage.mkdir()
        validate_cli_training_input(safe_stage)
        with mock.patch('src.model_comparison.train_comparison') as train:
            with self.assertRaisesRegex(ValueError, '학습 제외 목록'):
                train_deployment(self.root / 'data', self.root / 'output', (3,), 41, ('rbf_svm',))
            with mock.patch('sys.argv', ['train_window_models.py', '--data', str(self.root / 'data'),
                                        '--output-dir', str(self.root / 'output'), '--deployment',
                                        '--models', 'rbf_svm']), self.assertRaisesRegex(ValueError, '학습 제외 목록'):
                main()
            train.assert_not_called()
        self.assertFalse((self.root / 'output').exists())
        self.assertEqual((source.read_bytes(), source.stat().st_mtime_ns), before)

    def test_direct_input_guard_matches_nfd_paths_to_nfc_manifest(self):
        from src.training_input_policy import validate_cli_training_input
        relative_nfc = '참여자/정자세.csv'
        relative_nfd = unicodedata.normalize('NFD', relative_nfc)
        source = self.root / 'data' / relative_nfd
        source.parent.mkdir(parents=True)
        source.write_bytes(b'canonical source')
        manifest = self.root / 'research/models/training_exclusions.json'
        manifest.parent.mkdir(parents=True)
        item = {'source_file': relative_nfc, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'reason': 'test NFC manifest', 'requested_at': '2026-09-21T20:00:00+09:00'}
        manifest.write_text(json.dumps({'schema_version': 1, 'exclusions': [item]}))
        # The loader's hash/path validation is covered separately. Supplying its
        # validated result also exercises identity matching on case-sensitive
        # Linux filesystems that do not alias NFC and NFD filenames like macOS.
        with mock.patch('src.training_input_policy.load_training_exclusions', return_value={relative_nfc: item}), \
             self.assertRaisesRegex(ValueError, '등록된 원본 1개'):
            validate_cli_training_input(self.root / 'data')



if __name__ == '__main__':
    unittest.main()
