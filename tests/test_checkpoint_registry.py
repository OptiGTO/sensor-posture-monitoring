"""Checkpoint recipes stay exact and unavailable when their evidence changes."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np

from src import checkpoint_registry as registry
from src.app_services import TrainingWizardState
from src.experiment_workflow import apply_configuration, build_configuration
from src.pipeline_recipes import (build_pipeline, validate_pipeline, pipeline_sources,
    input_representation, requires_record_ids, combination_family, is_spatial_recipe, is_gain_recipe,
    validate_identity_control, is_local_recipe, is_interaction_recipe, is_support_recipe,
    is_alternatives_recipe, is_projection_recipe, is_hardness_recipe, is_structured_recipe)
from functools import partial
from src.model_comparison import train_comparison as _train_comparison

# These contracts count fits on the single validation split; LOSO selection has
# its own tests in test_loso_selection.py.
train_comparison = partial(_train_comparison, selection_metric="validation")
from src.posture_core import feature_names, CLASS_NAMES
from src.training_methods import (available_training_methods, build_method_model,
    fit_method_model, method_candidates, validate_training_method)


class CheckpointRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "src/validated_checkpoints.json"
        for name, value in (("PROJECT_ROOT", self.root), ("REGISTRY_PATH", self.path)):
            patch = mock.patch.object(registry, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.spec = validate_pipeline({"family": "rbf_svm", "window_seconds": 3,
            "feature_engineering": {"mode": "means"}, "model_params": {"C": 1.0, "gamma": 0.0005},
            "training": {"seed": 53, "weighting": "class_balanced"}})
        self.write_linked_fixture(self.spec)
        self.method = "checkpoint:synthetic-v1"

    def write_linked_fixture(self, pipeline):
        self.spec = validate_pipeline(pipeline)
        files = {}
        def write(role, content):
            relative = f"docs/reports/{role}.md" if role == "report" else f"research/results/fixture/{role}.json"
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content))
            files[role] = {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        sources = {}
        for relative in pipeline_sources(self.spec):
            original = Path(__file__).resolve().parents[1] / relative
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original.read_bytes())
            sources[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
        seeds = [self.spec['training']['seed'] + index for index in range(3)]
        write('model', {'synthetic_test_fixture': True, 'family': self.spec['family']})
        write('report', {'synthetic_test_fixture': True})
        write('pipeline', self.spec)
        supplemental = None
        training_audit = None
        if self.spec['training'].get('data_policy'):
            policy = self.spec['training']['data_policy']
            write('supplemental_cache', {'synthetic_test_fixture': True})
            write('source_supplemental', {'synthetic_source_fixture': True})
            sources['experiments/iterative_improvement/supplemental.py'] = files['source_supplemental']['sha256']
            if requires_record_ids(self.spec):
                helper = 'record_sequence_training' if is_gain_recipe(self.spec) else 'record_training'
                write('source_' + helper, {'synthetic_record_source_fixture': True})
                sources['experiments/iterative_improvement/' + helper + '.py'] = files['source_' + helper]['sha256']
            records = [{'subject': name, 'record_id': f'{name}/ordinary.csv', 'label': '정자세', 'windows': 2}
                       for name in ('a', 'b', 'c')]
            manifest = {'policy': policy, 'cache_sha256': files['supplemental_cache']['sha256'],
                'origin_dataset_sha256': 'd' * 64, 'origin_cache_sha256': 'c' * 64,
                'records': records, 'record_count': 3, 'window_count': 6}
            if input_representation(self.spec) == 'sensor_sequence':
                from src.posture_core import SENSOR_COLUMNS
                write('source_supplemental_sequence', {'synthetic_sequence_source_fixture': True})
                sources['experiments/iterative_improvement/supplemental_sequence.py'] = files['source_supplemental_sequence']['sha256']
                for row in records:
                    row['window_start_s'] = [0., 3.]
                manifest['supplemental_source_hashes'] = {path: sources[path] for path in (
                    'experiments/iterative_improvement/supplemental.py',
                    'experiments/iterative_improvement/supplemental_sequence.py')}
                manifest['sequence_evidence'] = {'schema_version': 1, 'shape': [6, 36, 16],
                    'dtype': 'float32', 'axis_order': ['windows', 'time', 'sensors'],
                    'sensor_order': list(SENSOR_COLUMNS), 'aligned_keys': ['seq', 'X', 'y', 'groups', 'record_ids', 'start_s'],
                    'record_alignment': 'All rows match manifest record order, subject, label and window_start_s',
                    'cache_bytes_reused_without_rewrite': True, 'sequence_array_sha256': 'a' * 64,
                    'feature_parity': {'method': 'extract_window_features(seq); equal dtype, np.array_equal and C-order bytes',
                        'bitwise_equal': True, 'checked_windows': 6, 'feature_dimension': 596,
                        'maximum_absolute_difference': 0.0}}
            write('supplemental_manifest', manifest)
            supplemental = {'policy': policy, 'manifest_sha256': files['supplemental_manifest']['sha256'],
                'cache_sha256': files['supplemental_cache']['sha256'], 'records': 3, 'windows': 6}
            training_audit = {'policy': policy, 'training_subjects': ['a', 'b'],
                'added_record_paths': ['a/ordinary.csv', 'b/ordinary.csv'],
                'added_windows_by_record': {'a/ordinary.csv': 2, 'b/ordinary.csv': 2},
                'added_records': 2, 'added_windows': 4, 'evaluation_rows_unchanged': True,
                'personal_reference_used': False, 'base_training_windows': 14,
                'class_counts_before': [2] * 7, 'class_counts_after': [6, *([2] * 6)]}
        recipes = ({f'member{index}': member for index, member in enumerate(self.spec['members'])}
                   if self.spec['family'] in {'equal_soft_voting', 'sequence_soft_voting', 'record_soft_voting', 'spatial_soft_voting', 'gain_soft_voting', 'local_soft_voting', 'interaction_soft_voting', 'support_soft_voting', 'alternatives_soft_voting', 'projection_soft_voting', 'hardness_soft_voting', 'structured_soft_voting'} else {'fixture_candidate': self.spec})
        write('protocol', {'source_hashes': sources, 'seeds': seeds, 'branches': {'fixture': recipes},
            'supplemental_training': supplemental, 'dataset_sha256': 'd' * 64, 'cache_sha256': 'c' * 64,
            'partitions': [{'id': str(index)} for index in range(3)],
            'criteria': {'scope': 'retrospective_subject_disjoint', 'minimum_development_delta': .01,
                'minimum_holdout_delta': .01, 'maximum_per_class_drop': .05, 'refit_probability_atol': 1e-12,
                'minimum_positive_splits': 2, 'bootstrap_lower_must_exceed': 0, 'bootstrap_replicates': 10000}})
        write('protocol_lock', {'sha256': files['protocol']['sha256']})
        selected = {'id': 'fixture_candidate', 'pipeline': self.spec, 'qualified': True,
            'deltas': [.03] * 3, 'mean_delta': .03, 'mean_class_deltas': [.03] * 7}
        candidates = [{'id': name, 'pipeline': recipe, 'qualified': True} for name, recipe in recipes.items()]
        if self.spec['family'] in {'equal_soft_voting', 'sequence_soft_voting', 'record_soft_voting', 'spatial_soft_voting', 'gain_soft_voting', 'local_soft_voting', 'interaction_soft_voting', 'support_soft_voting', 'alternatives_soft_voting', 'projection_soft_voting', 'hardness_soft_voting', 'structured_soft_voting'}:
            selected['versus_best_member'] = {'qualified': True}
        write('selection', {'selected': selected, 'protocol_sha256': files['protocol']['sha256'],
            'historical_holdout_opened_this_iteration': False, 'candidates': candidates})
        write('selection_seal', {'sha256': files['selection']['sha256']})
        baseline = {'macro_f1': .6, 'per_class_f1': dict.fromkeys(CLASS_NAMES, .6)}
        candidate = {'macro_f1': .63, 'per_class_f1': dict.fromkeys(CLASS_NAMES, .63)}
        write('final_metrics', {'pipeline': self.spec, 'seed': seeds[0], 'metrics': candidate,
            'protocol_sha256': files['protocol']['sha256'], 'model_sha256': files['model']['sha256'],
            'predictions_sha256': 'f' * 64, 'training_subjects': ['a', 'b'], 'evaluation_subjects': ['c'],
            **({'supplemental_training_audit': training_audit} if training_audit is not None else {})})
        write('final_metrics_seal', {'sha256': files['final_metrics']['sha256']})
        robustness = [{'seed': seed, 'mean_delta': .02, 'deltas': [.02] * 3} for seed in seeds[1:]]
        write('reproducibility', {'reproduced': True, 'protocol_sha256': files['protocol']['sha256'],
            'selected_id': selected['id'], 'selected_model_sha256': files['model']['sha256'],
            'selected_predictions_sha256': 'f' * 64, 'final_metrics_sha256': files['final_metrics']['sha256'],
            'selection_sha256': files['selection']['sha256'], 'max_probability_error': 0.,
            'reload_max_error': 0., 'seed_robustness': robustness})
        write('evaluation', {'improved': True, 'meaningful_gain': True, 'reproduced': True,
            'selected_id': selected['id'], 'selection_sha256': files['selection']['sha256'],
            'scope': 'retrospective_subject_disjoint', 'holdout_evaluated': True,
            'final': [{'seed': seed, 'candidate': candidate, 'baseline': baseline, 'delta': .03} for seed in seeds],
            'paired_subject_interval': {'unit': 'subject', 'lower': .005, 'upper': .05, 'replicates': 10000},
            'development_seed_robustness': robustness})
        write('evaluation_seal', {'sha256': files['evaluation']['sha256']})
        self.entry = {"id": "synthetic-v1", "name": "합성 검증 픽스처", "version": "1",
            "status": "validated_retrospective", "pipeline": self.spec,
            "evidence": {"improved": True, "meaningful_gain": True, "reproduced": True,
                "scope": "retrospective_subject_disjoint", "summary": "synthetic unit fixture"},
            "files": files}

    def rewrite_evidence(self, role, update):
        reference = self.entry['files'][role]
        path = self.root / reference['path']
        content = json.loads(path.read_text())
        update(content)
        path.write_text(json.dumps(content))
        reference['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        seals = {'protocol': 'protocol_lock', 'selection': 'selection_seal',
                 'final_metrics': 'final_metrics_seal', 'evaluation': 'evaluation_seal'}
        if role in seals:
            seal = self.entry['files'][seals[role]]
            seal_path = self.root / seal['path']
            seal_path.write_text(json.dumps({'sha256': reference['sha256']}))
            seal['sha256'] = hashlib.sha256(seal_path.read_bytes()).hexdigest()

    def write_user_adoption_fixture(self):
        self.write_linked_fixture(self.spec)
        files = self.entry['files']
        protocol = json.loads((self.root / files['protocol']['path']).read_text())
        protocol['baseline'] = deepcopy(self.spec)
        original_criteria = deepcopy(protocol['criteria'])
        source_run = 'research/results/earlier_failed_trial'
        def write(role, relative, value):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(value))
            files[role] = {'path': relative, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        write('adoption_source_protocol', source_run + '/protocol.json', protocol)
        source_candidate = {'id': 'fixture_candidate', 'pipeline': self.spec, 'qualified': False,
                            'mean_delta': .0094, 'deltas': [.0094] * 3}
        write('adoption_source_selection', source_run + '/selection_lock.json',
              {'protocol_sha256': files['adoption_source_protocol']['sha256'], 'candidates': [source_candidate]})
        authorization = {'schema_version': 1, 'kind': 'explicit_user_research_adoption',
            'instruction': '합성 테스트에 한정하여 이 정확한 연구 구성을 채택합니다.',
            'authorized_at': '2026-09-22T01:00:00+09:00', 'checkpoint_id': self.entry['id'],
            'candidate_id': 'fixture_candidate', 'pipeline': self.spec, 'source_run': source_run,
            'source_protocol_sha256': files['adoption_source_protocol']['sha256'],
            'source_selection_sha256': files['adoption_source_selection']['sha256'],
            'observed_development_delta': .0094, 'minimum_development_delta': .009,
            'scope': 'research_checkpoint_only', 'active_model_changed': False}
        write('user_authorization', 'research/results/fixture/user_authorization.json', authorization)
        directive = {'authorization_sha256': files['user_authorization']['sha256'],
            'checkpoint_id': self.entry['id'], 'candidate_id': 'fixture_candidate',
            'original_criteria': original_criteria, 'scope': 'research_checkpoint_only'}
        self.rewrite_evidence('protocol', lambda row: row.update(baseline=self.spec, user_adoption=directive,
            criteria={**original_criteria, 'minimum_development_delta': .009}))
        self.rewrite_evidence('selection', lambda row: (row.update(protocol_sha256=files['protocol']['sha256']),
            row['selected'].update(mean_delta=.0094, deltas=[.0094] * 3, mean_class_deltas=[.0094] * 7)))
        candidate = {'macro_f1': .59, 'per_class_f1': dict.fromkeys(CLASS_NAMES, .59)}
        self.rewrite_evidence('final_metrics', lambda row: row.update(metrics=candidate,
            protocol_sha256=files['protocol']['sha256']))
        seeds = [self.spec['training']['seed'] + index for index in range(3)]
        robustness = [{'seed': seed, 'mean_delta': -.003, 'deltas': [-.003] * 3} for seed in seeds[1:]]
        self.rewrite_evidence('reproducibility', lambda row: row.update(protocol_sha256=files['protocol']['sha256'],
            final_metrics_sha256=files['final_metrics']['sha256'], selection_sha256=files['selection']['sha256'],
            seed_robustness=robustness))
        baseline = {'macro_f1': .6, 'per_class_f1': dict.fromkeys(CLASS_NAMES, .6)}
        self.rewrite_evidence('evaluation', lambda row: row.update(improved=False, meaningful_gain=False,
            selection_sha256=files['selection']['sha256'], development_seed_robustness=robustness,
            user_adoption=directive, original_autonomous_criteria_met=False,
            final=[{'seed': seed, 'baseline': baseline, 'candidate': candidate, 'delta': -.01} for seed in seeds],
            paired_subject_interval={'unit': 'subject', 'lower': -.03, 'upper': .01, 'replicates': 10000}))
        self.entry['status'] = registry.USER_ADOPTION_STATUS
        self.entry['evidence'].update(improved=False, meaningful_gain=False, adoption_basis='explicit_user_request',
            standard_validation_passed=False, active_model_changed=False,
            summary='사용자 채택 · 연구 기준선; 기존 자동 채택 기준 미통과')

    def test_explicit_research_adoption_preserves_failed_metrics_and_exact_recipe(self):
        self.write_user_adoption_fixture()
        with mock.patch('joblib.load', side_effect=AssertionError('Discovery must not unpickle')):
            registry.register_checkpoint(self.entry)
        choice = next(row for row in available_training_methods() if row['id'] == self.method)
        self.assertIn('사용자 채택 · 연구 기준선', choice['label'])
        self.assertFalse(choice['evidence']['improved'])
        self.assertFalse(choice['evidence']['meaningful_gain'])
        self.assertEqual(method_candidates(self.spec['family'], self.method), [self.spec['model_params']])
        strict = deepcopy(self.entry)
        strict['status'] = 'validated_retrospective'
        with self.assertRaises(ValueError):
            registry.validate_checkpoint(strict)
        self.assertFalse((self.root / 'artifacts/active_model.json').exists())

    def test_user_adoption_does_not_waive_authorization_integrity_or_reproduction(self):
        mutations = {
            'missing explicit authority': lambda: self.entry['files'].pop('user_authorization'),
            'changed user instruction': lambda: self.rewrite_evidence('user_authorization', lambda row: row.update(instruction='changed')),
            'another adopted model': lambda: self.rewrite_evidence('model', lambda row: row.update(other=True)),
            'another original recipe': lambda: self.rewrite_evidence('adoption_source_selection',
                lambda row: row['candidates'][0]['pipeline']['model_params'].update(C=999)),
            'unreproduced': lambda: self.rewrite_evidence('reproducibility', lambda row: row.update(reproduced=False)),
            'fake success flags': lambda: self.rewrite_evidence('evaluation', lambda row: row.update(improved=True, meaningful_gain=True)),
            'another runtime scope': lambda: self.entry['evidence'].update(active_model_changed=True),
        }
        for name, mutation in mutations.items():
            with self.subTest(case=name):
                self.write_user_adoption_fixture()
                mutation()
                with self.assertRaises(ValueError):
                    registry.validate_checkpoint(self.entry)

    def test_autonomous_finalizer_does_not_publish_user_selected_statistical_failures(self):
        from experiments.iterative_improvement.finalize import finalize
        self.write_user_adoption_fixture()
        output = self.root / 'research/results/fixture'
        (output / 'evaluation.json').write_text(json.dumps({'improved': False, 'meaningful_gain': False, 'reproduced': True}))
        result = finalize(output, self.root / 'docs/not-created.md', 'user-only', 'user', '1', root=self.root)
        self.assertFalse(result['registered'])
        self.assertFalse((self.root / 'research/models/checkpoints/user-only').exists())

    def test_registration_is_append_only_and_does_not_deserialize_or_activate(self):
        with mock.patch("joblib.load", side_effect=AssertionError("Must not unpickle discovery")):
            registry.register_checkpoint(self.entry)
            self.assertEqual(registry.available_checkpoints(), [self.entry])
        previous = self.path.read_bytes()
        for candidate in (self.entry, {**self.entry, "version": "2"}):
            with self.assertRaisesRegex(ValueError, "덮어쓸"):
                registry.register_checkpoint(candidate)
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertFalse((self.root / "artifacts/active_model.json").exists())
        next_entry = {**self.entry, "id": "synthetic-v2", "version": "2"}
        registry.register_checkpoint(next_entry)
        self.assertEqual([entry["id"] for entry in registry.available_checkpoints()], ["synthetic-v1", "synthetic-v2"])

    def test_incomplete_degraded_unreproduced_and_altered_evidence_fail_closed(self):
        for field in ("improved", "meaningful_gain", "reproduced"):
            candidate = deepcopy(self.entry)
            candidate["evidence"][field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                registry.register_checkpoint(candidate)
        incomplete = deepcopy(self.entry)
        del incomplete["files"]["reproducibility"]
        with self.assertRaises(ValueError):
            registry.register_checkpoint(incomplete)
        registry.register_checkpoint(self.entry)
        (self.root / self.entry["files"]["model"]["path"]).write_text("changed")
        self.assertEqual(registry.available_checkpoints(), [])
        with self.assertRaises(ValueError):
            validate_training_method(self.method, ["rbf_svm"], [3])

    def test_concurrent_registration_does_not_lose_existing_checkpoints(self):
        registry.register_checkpoint(self.entry)
        original = self.path.read_bytes()
        lock = self.path.with_name(self.path.name + ".lock")
        lock.write_text("another writer")
        with self.assertRaisesRegex(ValueError, "등록이 진행"):
            registry.register_checkpoint({**self.entry, "id": "synthetic-v2"})
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(lock.read_text(), "another writer")

    def test_recipe_and_validation_decisions_are_checked_against_file_contents(self):
        changed = deepcopy(self.entry)
        changed["pipeline"]["model_params"]["C"] = 9
        with self.assertRaisesRegex(ValueError, "파이프라인 파일이 다릅니다"):
            registry.register_checkpoint(changed)
        for role in ("evaluation", "reproducibility"):
            entry = deepcopy(self.entry)
            path = self.root / entry["files"][role]["path"]
            original = path.read_bytes()
            path.write_text(json.dumps({"reproduced": False, "improved": True, "meaningful_gain": True}))
            entry["files"][role]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.subTest(role=role), self.assertRaises(ValueError):
                registry.register_checkpoint(entry)
            path.write_bytes(original)

    def test_path_escape_missing_report_and_duplicate_entries_are_unavailable(self):
        for path in ("../outside", "/tmp/outside", "research/report.md"):
            entry = deepcopy(self.entry)
            entry["files"]["report"]["path"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                registry.register_checkpoint(entry)
        registry.register_checkpoint(self.entry)
        self.path.write_text(json.dumps({"schema_version": 1, "checkpoints": [self.entry, self.entry]}))
        self.assertEqual(registry.available_checkpoints(), [])

    def test_exact_recipe_seed_params_and_configuration_survive_model_add(self):
        registry.register_checkpoint(self.entry)
        choice = next(entry for entry in available_training_methods() if entry["id"] == self.method)
        self.assertEqual(choice["checkpoint"]["pipeline"], self.spec)
        self.assertEqual(method_candidates("rbf_svm", self.method), [self.spec["model_params"]])
        with self.assertRaises(ValueError):
            validate_training_method(self.method, ["extra_trees"], [3])
        with self.assertRaises(ValueError):
            validate_training_method(self.method, ["rbf_svm"], [5])
        state = TrainingWizardState(models=("rbf_svm",), training_method=self.method, seed=41)
        config = build_configuration(state)
        restored = TrainingWizardState()
        apply_configuration(restored, config)
        self.assertEqual(build_configuration(restored)["checkpoint"], config["checkpoint"])
        X = np.random.default_rng(31).normal(size=(28, len(feature_names())))
        y = np.tile(np.arange(7), 4)
        model = build_method_model("rbf_svm", self.spec["model_params"], 999, self.method)
        fit_method_model(model, "rbf_svm", self.method, X, y)
        expected = build_pipeline(self.spec).fit(X, y)
        self.assertEqual(model.effective_seed_, 53)
        np.testing.assert_array_equal(model.predict_proba(X), expected.predict_proba(X))
        self.assertEqual(model.n_features_in_, 596)
        with self.assertRaisesRegex(ValueError, "하이퍼파라미터"):
            build_method_model("rbf_svm", {"C": 100}, 999, self.method)

    def test_synthetic_training_reuses_full_pipeline_and_exports_runtime_bundle(self):
        from tests.test_model_comparison import ModelComparisonTests
        registry.register_checkpoint(self.entry)
        ModelComparisonTests().make_data(self.root / "data")
        report = train_comparison(self.root / "data", self.root / "run", [3], 41,
            ["rbf_svm"], training_method=self.method, progress=lambda _: None)
        self.assertEqual(len(report["candidates"]), 1)
        self.assertTrue(report["selection_locked"])
        self.assertEqual(report["checkpoint"], self.entry)
        row = report["models"][0]
        artifact = self.root / "run/artifacts" / row["artifact_file"]
        bundle = joblib.load(artifact)
        self.assertEqual(bundle["model"].effective_seed_, 53)
        self.assertEqual(bundle["training_method_details"]["pipeline"], self.spec)
        self.assertEqual(bundle["training_method_details"]["checkpoint"]["files"], self.entry["files"])
        self.assertEqual(len(bundle["feature_names"]), 596)
        self.assertEqual(bundle["model"].n_features_in_, 596)

    def test_model_add_learned_selectors_receive_only_actual_training_subjects(self):
        from src.domain_feature_selection import SubjectInvariantSelector
        from src.model_comparison import _dataset
        from tests.test_model_comparison import ModelComparisonTests
        ModelComparisonTests().make_data(self.root / "data")
        partitions = {"train": ["subject0", "subject1"], "validation": ["subject2"],
                      "final_evaluation": ["subject3"]}
        data = _dataset(self.root / "data", 3, 41, partitions)
        mask = data["masks"]["train"]
        expected_groups = data["meta"].loc[mask, "subject"].to_numpy()
        learned = deepcopy(self.spec)
        learned["feature_engineering"] = {"mode": "subject_invariant", "k": 32, "strength": 1.0}
        ensemble = validate_pipeline({"family": "equal_soft_voting", "members": [learned, self.spec],
                                      "training": {"seed": 53}})
        original_fit = SubjectInvariantSelector.fit
        captures = []
        def capture(selector, X, y, *, groups=None):
            captures.append((np.array(X), np.array(y), np.array(groups)))
            return original_fit(selector, X, y, groups=groups)
        for pipeline in (learned, ensemble):
            with self.subTest(family=pipeline["family"]):
                captures.clear()
                self.write_linked_fixture(pipeline)
                # Keep each synthetic registry isolated, never touch the project registry.
                self.path.write_text(json.dumps({"schema_version": 1, "checkpoints": []}))
                registry.register_checkpoint(self.entry)
                with mock.patch.object(SubjectInvariantSelector, "fit", capture):
                    report = train_comparison(self.root / "data", self.root / pipeline["family"], [3], 41,
                        [pipeline["family"]], training_method=self.method, progress=lambda _: None,
                        fixed_subject_partitions=partitions)
                self.assertEqual(report["failures"], [])
                self.assertEqual(len(captures), 1)
                observed_X, observed_y, observed_groups = captures[0]
                np.testing.assert_array_equal(observed_X, data["X"][mask])
                np.testing.assert_array_equal(observed_y, data["y"][mask])
                np.testing.assert_array_equal(observed_groups, expected_groups)
                self.assertEqual(set(observed_groups), set(partitions["train"]))
                bundle = joblib.load(self.root / pipeline["family"] / "artifacts" / report["models"][0]["artifact_file"])
                model = bundle["model"]
                member = model.members_[0] if pipeline["family"] == "equal_soft_voting" else model
                selector = member.named_steps["features"]
                self.assertEqual(selector.n_subjects_seen_, 2)
                self.assertEqual(selector.n_samples_seen_, int(mask.sum()))
                expected_selector = SubjectInvariantSelector(k=32, strength=1.0).fit(
                    data["X"][mask], data["y"][mask], groups=expected_groups)
                np.testing.assert_array_equal(selector.indices_, expected_selector.indices_)

    def test_checkpoint_voting_members_must_declare_the_validated_primary_seed(self):
        pipeline = self.ensemble_spec()
        pipeline["members"][0]["training"]["seed"] += 1
        self.write_linked_fixture(pipeline)
        with self.assertRaisesRegex(ValueError, "구성원 검증 시드"):
            registry.validate_checkpoint(self.entry)

    def test_finalizer_checks_each_fitted_member_and_finite_replay(self):
        from experiments.iterative_improvement.finalize import verified_model_probabilities
        pipeline = self.ensemble_spec()
        X = np.random.default_rng(9).normal(size=(28, 596))
        y = np.tile(np.arange(7), 4)
        model = build_pipeline(pipeline).fit(X, y)
        probability = verified_model_probabilities(model, pipeline, 53, X)
        np.testing.assert_array_equal(probability, model.predict_proba(X))
        model.members_[1].effective_seed_ = 54
        with self.assertRaisesRegex(ValueError, "recipe/seed"):
            verified_model_probabilities(model, pipeline, 53, X)
        model.members_[1].effective_seed_ = 53
        model.members_[1].spec_["feature_engineering"]["mode"] = "static"
        with self.assertRaisesRegex(ValueError, "recipe/seed"):
            verified_model_probabilities(model, pipeline, 53, X)
        model.members_[1].spec_ = deepcopy(pipeline["members"][1])
        with mock.patch.object(model, "predict_proba", return_value=np.full((len(X), 7), np.nan)):
            with self.assertRaisesRegex(ValueError, "invalid probabilities"):
                verified_model_probabilities(model, pipeline, 53, X)

    def test_probability_bias_checkpoint_reuses_group_fit_and_verifies_inner_model(self):
        from experiments.iterative_improvement.finalize import verified_model_probabilities
        pipeline = deepcopy(self.spec)
        pipeline["postprocessing"] = {"kind": "macro_f1_bias", "inner_folds": 4,
                                      "max_sweeps": 2, "regularization": .01}
        self.write_linked_fixture(pipeline)
        registry.register_checkpoint(self.entry)
        X = np.random.default_rng(11).normal(size=(56, 596))
        y = np.tile(np.arange(7), 8)
        groups = np.repeat(["training-a", "training-b", "training-c", "training-d"], 14)
        model = build_method_model("rbf_svm", pipeline["model_params"], 999, self.method)
        fit_method_model(model, "rbf_svm", self.method, X, y, groups=groups)
        expected_base = deepcopy(pipeline)
        expected_base["postprocessing"] = {"kind": "argmax"}
        self.assertEqual(model.estimator_.spec_, validate_pipeline(expected_base))
        folds = model.training_details_["inner_group_folds"]
        self.assertEqual(len(folds), 4)
        for fold in folds:
            self.assertFalse(set(fold["training_subjects"]) & set(fold["held_out_subjects"]))
            self.assertEqual(set(fold["training_subjects"]) | set(fold["held_out_subjects"]), set(groups))
        artifact = self.root / "bias-wrapper.joblib"
        joblib.dump(model, artifact)
        loaded = joblib.load(artifact)
        np.testing.assert_array_equal(verified_model_probabilities(loaded, pipeline, 53, X), model.predict_proba(X))
        loaded.estimator_.effective_seed_ = 54
        with self.assertRaisesRegex(ValueError, "recipe/seed"):
            verified_model_probabilities(loaded, pipeline, 53, X)

    def test_packaging_keeps_registry_and_only_referenced_evidence(self):
        import build_macos_app
        registry.register_checkpoint(self.entry)
        with mock.patch.object(build_macos_app, "ROOT", self.root):
            command = build_macos_app.build_command()
        items = [command[index + 1] for index, item in enumerate(command[:-1]) if item == "--add-data"]
        self.assertIn(f"{self.path}:src", items)
        for reference in self.entry["files"].values():
            path = self.root / reference["path"]
            self.assertIn(f"{path}:{Path(reference['path']).parent}", items)
        self.assertFalse(any(":data" in item for item in items))

    def test_unrelated_artifact_recipe_seed_protocol_scope_and_metrics_are_rejected(self):
        def change_recipe_and_entry():
            self.rewrite_evidence('pipeline', lambda content: content['model_params'].update(C=99))
            self.entry['pipeline'] = deepcopy(self.entry['pipeline'])
            self.entry['pipeline']['model_params']['C'] = 99
        mutations = {
            'different model': lambda: self.rewrite_evidence('model', lambda content: content.update(other_model=True)),
            'different recipe': lambda: self.rewrite_evidence('pipeline', lambda content: content['model_params'].update(C=99)),
            'matching new recipe and entry but old evidence': change_recipe_and_entry,
            'different seed': lambda: self.rewrite_evidence('final_metrics', lambda content: content.update(seed=54)),
            'different protocol': lambda: self.rewrite_evidence('final_metrics', lambda content: content.update(protocol_sha256='a' * 64)),
            'different selection': lambda: self.rewrite_evidence('evaluation', lambda content: content.update(selected_id='another')),
            'different scope': lambda: self.rewrite_evidence('evaluation', lambda content: content.update(scope='prospective')),
            'different reproduced model': lambda: self.rewrite_evidence('reproducibility', lambda content: content.update(selected_model_sha256='b' * 64)),
            'different final result': lambda: self.rewrite_evidence('final_metrics', lambda content: content['metrics'].update(macro_f1=.99)),
            'failed real threshold': lambda: self.rewrite_evidence('evaluation', lambda content: content['paired_subject_interval'].update(lower=-.01)),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                self.write_linked_fixture(self.spec)
                mutation()
                with self.assertRaises(ValueError):
                    registry.register_checkpoint(self.entry)
                self.assertFalse(self.path.exists())

    def test_algorithm_source_drift_hides_checkpoint_without_unpickling(self):
        registry.register_checkpoint(self.entry)
        source = self.root / 'src/iterative_pipeline.py'
        source.write_bytes(source.read_bytes() + b'\n# changed algorithm\n')
        with mock.patch('joblib.load', side_effect=AssertionError('No unpickling discovery')):
            self.assertEqual(registry.available_checkpoints(), [])

    def identity_control_fixture(self, family='record_gain_extra_trees'):
        self.write_linked_fixture(self.spec)
        parameters = {'n_estimators': 200, 'max_depth': 8, 'min_samples_leaf': 8, 'max_features': .5}
        baseline = validate_pipeline({'family': 'extra_trees', 'model_params': parameters, 'training': {'seed': 53}})
        control = validate_pipeline({'family': family, 'model_params': parameters,
            **({'feature_engineering': {'mode': 'identity'}} if family in {'interaction_extra_trees', 'subject_residual_extra_trees'} else {}),
            **({'preprocessing': {'scaler': 'standard', 'tail_fraction': 0.0}} if family == 'tail_extra_trees' else {}),
            **({'postprocessing': {'kind': 'uniform_leaf_shrinkage', 'alpha': 0.0}}
                if family == 'support_extra_trees' else {}),
            'training': {'seed': 53, **({'augmentation': {'mode': 'identity', 'strength': 0.}}
                if family == 'record_gain_extra_trees' else {}),
                **({'hardness': {'coefficient': 0.0, 'n_splits': 4, 'seed_offset': 1000}}
                   if family == 'hardness_extra_trees' else {})}})
        self.rewrite_evidence('protocol', lambda row: row.update(baseline=baseline, identity_controls=['identity_a', 'identity_b'],
            branches={**row['branches'], 'identity': {'identity_a': control, 'identity_b': control}}))
        protocol_hash = self.entry['files']['protocol']['sha256']
        self.rewrite_evidence('selection', lambda row: row.update(protocol_sha256=protocol_hash))
        self.rewrite_evidence('final_metrics', lambda row: row.update(protocol_sha256=protocol_hash))
        selection_hash = self.entry['files']['selection']['sha256']
        self.rewrite_evidence('evaluation', lambda row: row.update(selection_sha256=selection_hash))
        self.rewrite_evidence('reproducibility', lambda row: row.update(protocol_sha256=protocol_hash,
            selection_sha256=selection_hash, final_metrics_sha256=self.entry['files']['final_metrics']['sha256']))
        relative = 'research/results/fixture/audits/identity_controls.json'
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'protocol_sha256': protocol_hash, 'seed': self.spec['training']['seed'],
            'rows': [{'candidate': control, 'partition': str(part), 'all_prediction_arrays_bitwise_equal': True,
                'baseline_predictions_sha256': 'a' * 64, 'control_predictions_sha256': 'b' * 64}
                for control in ('identity_a', 'identity_b') for part in range(3)]}))
        self.entry['files']['identity_controls'] = {'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_preregistered_identity_controls_require_complete_bound_bitwise_audit(self):
        self.identity_control_fixture()
        registry.validate_checkpoint(self.entry)
        missing = deepcopy(self.entry)
        del missing['files']['identity_controls']
        with self.assertRaisesRegex(ValueError, '동일 경로 대조군'):
            registry.validate_checkpoint(missing)
        mutations = {
            'protocol': lambda row: row.update(protocol_sha256='c' * 64),
            'seed': lambda row: row.update(seed=54),
            'missing partition': lambda row: row['rows'].pop(),
            'duplicate partition': lambda row: row['rows'].__setitem__(1, deepcopy(row['rows'][0])),
            'wrong candidate': lambda row: row['rows'][0].update(candidate='unplanned'),
            'not bitwise equal': lambda row: row['rows'][0].update(all_prediction_arrays_bitwise_equal=False),
            'missing hash': lambda row: row['rows'][0].pop('control_predictions_sha256'),
            'invalid hash': lambda row: row['rows'][0].update(baseline_predictions_sha256='not a hash'),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                self.identity_control_fixture()
                self.rewrite_evidence('identity_controls', mutation)
                with self.assertRaisesRegex(ValueError, '동일 경로 대조군'):
                    registry.validate_checkpoint(self.entry)
        self.identity_control_fixture()
        protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
        for controls in (['identity_a', 'identity_a'], ['unplanned'], 'identity_a'):
            broken = deepcopy(protocol)
            broken['identity_controls'] = controls
            with self.assertRaisesRegex(ValueError, '동일 경로 대조군'):
                registry._validate_identity_controls(self.entry, self.root, broken)

    def test_identity_control_rejects_recipe_drift_despite_equal_prediction_audit(self):
        self.identity_control_fixture()
        protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
        baseline = protocol['baseline']
        control = protocol['branches']['identity']['identity_a']
        validate_identity_control(baseline, control)
        mutations = {
            'nonzero gain': lambda ref, row: row['training']['augmentation'].update(mode='per_sensor', strength=.02),
            'common zero mode': lambda ref, row: row['training']['augmentation'].update(mode='common', strength=0.),
            'tree parameters': lambda ref, row: row['model_params'].update(n_estimators=201),
            'missing reference parameter': lambda ref, row: ref['model_params'].pop('max_depth'),
            'implicit reference defaults': lambda ref, row: ref.update(model_params={}),
            'integer versus fraction': lambda ref, row: (ref['model_params'].update(max_features=1), row['model_params'].update(max_features=1.0)),
            'extra reference parameter': lambda ref, row: ref['model_params'].update(bootstrap=False),
            'seed': lambda ref, row: row['training'].update(seed=54),
            'policy': lambda ref, row: row['training'].update(data_policy='include_ordinary_reserved_training_records_v1'),
            'reference weighting': lambda ref, row: ref['training'].update(weighting='subject_class_balanced'),
            'reference features': lambda ref, row: ref['feature_engineering'].update(mode='means'),
            'reference scaler': lambda ref, row: ref['preprocessing'].update(scaler='shrinkage_whiten'),
            'reference family': lambda ref, row: ref.update(family='random_forest'),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = deepcopy(protocol)
                mutate(changed['baseline'], changed['branches']['identity']['identity_a'])
                # The audit still claims complete bitwise equality. Semantically
                # different recipes cannot become identity controls via that flag.
                with self.assertRaises(ValueError):
                    validate_identity_control(changed['baseline'], changed['branches']['identity']['identity_a'])
                with self.assertRaisesRegex(ValueError, '동일 경로 대조군'):
                    registry._validate_identity_controls(self.entry, self.root, changed)

    def test_version_dispatch_preserves_legacy_source_contract_and_saved_type(self):
        from src.iterative_pipeline import IterativePipeline
        from src.pipeline_recipes import LEGACY_PIPELINE_SOURCES, input_representation
        self.assertEqual(registry.PIPELINE_SOURCES, LEGACY_PIPELINE_SOURCES)
        self.assertEqual(len(pipeline_sources(self.spec)), 8)
        self.assertEqual(input_representation(self.spec), 'feature_vector')
        self.assertIsInstance(build_pipeline(self.spec), IterativePipeline)
        registry.register_checkpoint(self.entry)
        original = self.path.read_bytes()
        for relative in ('src/pipeline_recipes.py', 'src/sequence_pipeline_v1.py'):
            (self.root / relative).write_text('# An unrelated new implementation need not exist for legacy recipes.\n')
        self.assertEqual(registry.available_checkpoints(), [self.entry])
        self.assertEqual(self.path.read_bytes(), original)

    def test_record_metadata_is_not_forwarded_into_frozen_legacy_or_sequence_fitters(self):
        from src.pipeline_recipes import fit_pipeline
        for pipeline, target in ((self.spec, 'src.iterative_pipeline.fit_pipeline'),
                ({'family': 'cnn1d'}, 'src.sequence_pipeline_v1.fit_sequence_pipeline')):
            with self.subTest(family=pipeline['family']), mock.patch(target) as fit:
                fit_pipeline(object(), pipeline, 'input', 'labels', groups='subjects', record_ids='record identities')
                self.assertNotIn('record_ids', fit.call_args.kwargs)
                self.assertEqual(fit.call_args.kwargs['groups'], 'subjects')
        self.assertFalse(requires_record_ids(self.spec))
        self.assertEqual(combination_family(self.spec), 'equal_soft_voting')
        self.assertFalse(requires_record_ids({'family': 'cnn1d'}))
        self.assertEqual(combination_family({'family': 'cnn1d'}), 'sequence_soft_voting')

    def test_versioned_checkpoints_survive_dispatch_growth_but_reject_algorithm_drift(self):
        for pipeline, algorithm in (({'family': 'cnn1d'}, 'src/sequence_pipeline_v1.py'),
                ({'family': 'record_balanced_extra_trees'}, 'src/record_pipeline_v1.py'),
                ({'family': 'ordinal_extra_trees'}, 'src/spatial_pipeline_v1.py'),
                (self.gain_specs()[0], 'src/augmentation_pipeline_v1.py'),
                ({'family': 'record_prototype'}, 'src/local_geometry_pipeline_v1.py'),
                ({'family': 'interaction_extra_trees'}, 'src/interaction_pipeline_v1.py'),
                ({'family': 'support_extra_trees'}, 'src/forest_support_pipeline_v1.py')):
            with self.subTest(family=pipeline['family']):
                self.write_linked_fixture(pipeline)
                self.assertNotIn('src/pipeline_recipes.py', pipeline_sources(pipeline))
                self.assertIn(algorithm, pipeline_sources(pipeline))
                dispatch = self.root / 'src/pipeline_recipes.py'
                dispatch.write_text('# Dispatcher before an unrelated family was added.\n')
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                original_registry = self.path.read_bytes()
                dispatch.write_text('# Dispatcher after an unrelated family was added.\n')
                with mock.patch('joblib.load', side_effect=AssertionError('No unpickling discovery')):
                    self.assertEqual(registry.available_checkpoints(), [self.entry])
                    implementation = self.root / algorithm
                    implementation.write_bytes(implementation.read_bytes() + b'\n# Changed fitted algorithm.\n')
                    self.assertEqual(registry.available_checkpoints(), [])
                self.assertEqual(self.path.read_bytes(), original_registry)

    def spatial_specs(self):
        ordinal = validate_pipeline({'family': 'ordinal_extra_trees',
            'feature_engineering': {'mode': 'all_plus_mean_order'},
            'model_params': {'n_estimators': 4, 'min_samples_leaf': 1},
            'training': {'seed': 53, 'weighting': 'class_balanced'}})
        qda = validate_pipeline({'family': 'shrinkage_qda',
            'feature_engineering': {'mode': 'abs_rel_sensor_means'},
            'model_params': {'reg_param': .5},
            'training': {'seed': 53, 'weighting': 'uniform_class_prior'}})
        ensemble = validate_pipeline({'family': 'spatial_soft_voting', 'members': [ordinal, qda],
            'feature_engineering': {'mode': 'per_member'},
            'training': {'seed': 53, 'weighting': 'per_member'}})
        return ordinal, qda, ensemble

    def test_spatial_checkpoints_reuse_exact_recipe_and_only_train_rows_through_csv_replay(self):
        from src import spatial_pipeline_v1
        from src.app_services import TrainingJob
        from src.model_catalog import MODEL_NAMES, pipeline_runtime_packages
        from src.model_comparison import _dataset
        from src.model_runtime import ModelRegistry
        from src.model_runtime.sklearn_adapter import SklearnAdapter
        from src.posture_core import read_sensor_csv
        from tests.test_model_comparison import ModelComparisonTests
        source = Path(__file__).resolve().parents[1]
        # Two training people produce 34 independent synthetic windows per class,
        # exceeding the QDA SVD dimensionality contract for 32 spatial features.
        ModelComparisonTests().make_data(self.root / 'data', rows=613)
        partitions = {'train': ['subject0', 'subject1'], 'validation': ['subject2'], 'final_evaluation': ['subject3']}
        data = _dataset(self.root / 'data', 3, 41, partitions)
        mask = data['masks']['train']
        self.assertTrue(np.all(np.bincount(data['y'][mask], minlength=7) >= 32))
        original_fit = spatial_pipeline_v1.fit_spatial_pipeline
        captures = []
        def capture(model, recipe, X, y, *, groups=None, validation_data=None):
            captures.append((np.array(X), np.array(y), np.array(groups)))
            return original_fit(model, recipe, X, y, groups=groups, validation_data=validation_data)
        for pipeline in self.spatial_specs():
            with self.subTest(family=pipeline['family']):
                captures.clear()
                self.write_linked_fixture(pipeline)
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                family = pipeline['family']
                self.assertEqual(len(MODEL_NAMES), 9)
                self.assertNotIn(family, MODEL_NAMES)
                self.assertTrue(is_spatial_recipe(pipeline))
                self.assertFalse(requires_record_ids(pipeline))
                self.assertEqual(combination_family(pipeline), 'spatial_soft_voting')
                self.assertEqual(input_representation(pipeline), 'feature_vector')
                self.assertEqual(pipeline_runtime_packages(pipeline), {})
                with self.assertRaises(ValueError):
                    TrainingWizardState().configure([3], 41, [family])
                state = TrainingWizardState(training_method=self.method)
                state.configure([3], 41, [family])
                restored = TrainingWizardState()
                apply_configuration(restored, build_configuration(state))
                self.assertEqual(restored.models, (family,))
                self.assertEqual(method_candidates(family, self.method), [pipeline['model_params']])
                job = TrainingJob(source, (), 41, models=(family,), state_root=self.root / family,
                                  training_method=self.method)
                # This strict signature also proves record IDs are not forwarded
                # into spatial fitting despite being available in the GUI dataset.
                with mock.patch.object(spatial_pipeline_v1, 'fit_spatial_pipeline', capture):
                    report = train_comparison(self.root / 'data', job.run_dir, [3], 41, [family],
                        training_method=self.method, progress=lambda _: None, fixed_subject_partitions=partitions)
                self.assertEqual(report['failures'], [])
                self.assertEqual(len(captures), 1)
                observed_X, observed_y, observed_groups = captures[0]
                np.testing.assert_array_equal(observed_X, data['X'][mask])
                np.testing.assert_array_equal(observed_y, data['y'][mask])
                np.testing.assert_array_equal(observed_groups, data['meta'].loc[mask, 'subject'].to_numpy())
                bundle = joblib.load(job.run_dir / 'artifacts' / report['models'][0]['artifact_file'])
                self.assertEqual(bundle['model'].effective_seed_, 53)
                self.assertEqual(bundle['model'].spec_, pipeline)
                self.assertEqual(bundle['input_representation'], 'feature_vector')
                self.assertEqual(len(bundle['feature_names']), 596)
                spatial_pipeline_v1.verify_fitted_spatial_identity(bundle['model'], pipeline, 53)
                leaves = bundle['model'].members_ if family == 'spatial_soft_voting' else [bundle['model']]
                for leaf in leaves:
                    expected_features = leaf.features_.transform(data['X'][mask])
                    np.testing.assert_allclose(leaf.scaler_.mean_, expected_features.mean(axis=0, dtype=np.float64))
                    self.assertEqual(leaf.scaler_.n_samples_seen_, int(mask.sum()))
                manifest, = job._write_pending_candidate_manifests()
                runtime_registry = ModelRegistry(source, self.root / family)
                descriptor = runtime_registry.validator.discover_manifest(manifest).descriptor
                validated = runtime_registry.validate_runtime(descriptor)
                self.assertEqual(validated.validation_status, 'passed', validated.lifecycle_reason)
                replay = SklearnAdapter(validated).predict_proba(read_sensor_csv(source / 'example/sample_input.csv'), 3)
                self.assertEqual(replay.shape[1], 7)
                self.assertEqual(validated.approval_status, 'pending')

    def test_spatial_members_need_independent_gains_and_feature_implementation_is_pinned(self):
        self.write_linked_fixture(self.spatial_specs()[-1])
        registry.validate_checkpoint(self.entry)
        self.assertIn('src/spatial_features_v1.py', pipeline_sources(self.spec))
        self.rewrite_evidence('selection', lambda row: row['candidates'][0].update(qualified=False))
        selection_hash = self.entry['files']['selection']['sha256']
        self.rewrite_evidence('evaluation', lambda row: row.update(selection_sha256=selection_hash))
        self.rewrite_evidence('reproducibility', lambda row: row.update(selection_sha256=selection_hash))
        with self.assertRaisesRegex(ValueError, '검증된 개선 후보만'):
            registry.validate_checkpoint(self.entry)
        self.write_linked_fixture(self.spatial_specs()[0])
        registry.register_checkpoint(self.entry)
        feature_source = self.root / 'src/spatial_features_v1.py'
        feature_source.write_bytes(feature_source.read_bytes() + b'\n# Changed spatial feature mapping.\n')
        with mock.patch('joblib.load', side_effect=AssertionError('No unpickling discovery')):
            self.assertEqual(registry.available_checkpoints(), [])

    def gain_specs(self):
        leaves = [validate_pipeline({'family': 'record_gain_extra_trees',
            'feature_engineering': {'mode': 'all_from_sequence'},
            'model_params': {'n_estimators': 4, 'min_samples_leaf': 1},
            'training': {'seed': 53, 'weighting': 'class_balanced', 'augmentation': {
                'mode': 'per_sensor', 'strength': strength, 'distribution': 'uniform',
                'row_policy': 'replace', 'record_seed_policy': 'sha256_record_v1'}}})
            for strength in (.02, .05)]
        ensemble = validate_pipeline({'family': 'gain_soft_voting', 'members': leaves,
            'feature_engineering': {'mode': 'all_from_sequence'},
            'training': {'seed': 53, 'weighting': 'per_member'}})
        return leaves[0], ensemble

    def local_specs(self):
        prototype = validate_pipeline({'family': 'record_prototype',
            'model_params': {'n_neighbors': 1, 'epsilon': 1e-12},
            'training': {'seed': 53, 'weighting': 'equal_class_distance'}})
        mixture = validate_pipeline({'family': 'class_conditional_mixture',
            'model_params': {'n_components': 1, 'reg_covar': .1, 'n_init': 3, 'max_iter': 200, 'tol': .001},
            'training': {'seed': 53, 'weighting': 'uniform_class_prior'}})
        another = deepcopy(mixture)
        another['model_params']['n_components'] = 2
        def ensemble(members):
            return validate_pipeline({'family': 'local_soft_voting', 'members': members,
                                      'training': {'seed': 53, 'weighting': 'per_member'}})
        return prototype, mixture, ensemble([prototype, mixture]), ensemble([mixture, another])

    def test_local_checkpoint_train_metadata_contracts_and_full_csv_replay(self):
        from src import local_geometry_pipeline_v1
        from src.app_services import TrainingJob
        from src.model_catalog import MODEL_NAMES, pipeline_runtime_packages
        from src.model_comparison import _dataset
        from src.model_runtime import ModelRegistry
        from src.model_runtime.sklearn_adapter import SklearnAdapter
        from src.posture_core import read_sensor_csv
        from tests.test_model_comparison import ModelComparisonTests
        source = Path(__file__).resolve().parents[1]
        ModelComparisonTests().make_data(self.root / 'data', rows=73)
        partitions = {'train': ['subject0', 'subject1'], 'validation': ['subject2'], 'final_evaluation': ['subject3']}
        data = _dataset(self.root / 'data', 3, 41, partitions)
        mask = data['masks']['train']
        records = data['meta'].loc[mask, 'record_id'].to_numpy()
        groups = data['meta'].loc[mask, 'subject'].to_numpy()
        original_fit = local_geometry_pipeline_v1.fit_local_pipeline
        captures = []
        def capture(model, recipe, X, y, *, groups=None, record_ids=None, validation_data=None):
            captures.append((np.array(X), np.array(y), np.array(groups),
                             None if record_ids is None else np.array(record_ids)))
            return original_fit(model, recipe, X, y, groups=groups, record_ids=record_ids, validation_data=validation_data)
        for index, (pipeline, needs_records) in enumerate(zip(self.local_specs(), (True, False, True, False))):
            with self.subTest(family=pipeline['family'], needs_records=needs_records):
                captures.clear()
                self.write_linked_fixture(pipeline)
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                family = pipeline['family']
                self.assertTrue(is_local_recipe(pipeline))
                self.assertFalse(is_gain_recipe(pipeline))
                self.assertEqual(requires_record_ids(pipeline), needs_records)
                self.assertEqual(input_representation(pipeline), 'feature_vector')
                self.assertEqual(combination_family(pipeline), 'local_soft_voting')
                self.assertEqual(len(MODEL_NAMES), 9)
                self.assertNotIn(family, MODEL_NAMES)
                self.assertEqual(pipeline_runtime_packages(pipeline), {})
                with self.assertRaises(ValueError):
                    TrainingWizardState().configure([3], 41, [family])
                state = TrainingWizardState(training_method=self.method)
                state.configure([3], 41, [family])
                restored = TrainingWizardState()
                apply_configuration(restored, build_configuration(state))
                self.assertEqual(restored.models, (family,))
                self.assertEqual(method_candidates(family, self.method), [pipeline['model_params']])
                state_root = self.root / f'{index}-{family}'
                job = TrainingJob(source, (), 41, models=(family,), state_root=state_root, training_method=self.method)
                with mock.patch.object(local_geometry_pipeline_v1, 'fit_local_pipeline', capture):
                    report = train_comparison(self.root / 'data', job.run_dir, [3], 41, [family],
                        training_method=self.method, progress=lambda _: None, fixed_subject_partitions=partitions)
                self.assertEqual(report['failures'], [])
                self.assertEqual(len(captures), 1)
                observed_X, observed_y, observed_groups, observed_records = captures[0]
                np.testing.assert_array_equal(observed_X, data['X'][mask])
                np.testing.assert_array_equal(observed_y, data['y'][mask])
                np.testing.assert_array_equal(observed_groups, groups)
                if needs_records:
                    np.testing.assert_array_equal(observed_records, records)
                    self.assertFalse(set(observed_records) & set(data['meta'].loc[~mask, 'record_id']))
                else:
                    self.assertIsNone(observed_records)
                bundle = joblib.load(job.run_dir / 'artifacts' / report['models'][0]['artifact_file'])
                self.assertEqual(bundle['model'].effective_seed_, 53)
                self.assertEqual(bundle['model'].spec_, pipeline)
                self.assertEqual(bundle['input_representation'], 'feature_vector')
                self.assertEqual(len(bundle['feature_names']), 596)
                local_geometry_pipeline_v1.verify_fitted_local_identity(bundle['model'], pipeline, 53)
                leaves = bundle['model'].members_ if family == 'local_soft_voting' else [bundle['model']]
                for leaf in leaves:
                    np.testing.assert_allclose(leaf.scaler_.mean_, data['X'][mask].mean(axis=0, dtype=np.float64))
                    self.assertEqual(leaf.scaler_.n_samples_seen_, int(mask.sum()))
                manifest, = job._write_pending_candidate_manifests()
                runtime_registry = ModelRegistry(source, state_root)
                descriptor = runtime_registry.validator.discover_manifest(manifest).descriptor
                validated = runtime_registry.validate_runtime(descriptor)
                self.assertEqual(validated.validation_status, 'passed', validated.lifecycle_reason)
                replay = SklearnAdapter(validated).predict_proba(read_sensor_csv(source / 'example/sample_input.csv'), 3)
                self.assertEqual(replay.shape[1], 7)
                self.assertTrue(np.isfinite(replay).all())
                np.testing.assert_allclose(replay.sum(1), 1, atol=1e-6)
                self.assertEqual(validated.approval_status, 'pending')
                if needs_records:
                    unfitted = build_method_model(family, pipeline['model_params'], 41, self.method)
                    with self.assertRaises(ValueError):
                        fit_method_model(unfitted, family, self.method, data['X'][mask], data['y'][mask], groups=groups)

    def test_local_supplement_requires_record_helper_only_for_prototype_members(self):
        for pipeline, needs_records in zip(self.local_specs(), (True, False, True, False)):
            with self.subTest(family=pipeline['family'], needs_records=needs_records):
                pipeline['training']['data_policy'] = 'include_ordinary_reserved_training_records_v1'
                for member in pipeline.get('members', ()):
                    member['training']['data_policy'] = pipeline['training']['data_policy']
                self.write_linked_fixture(pipeline)
                registry.validate_checkpoint(self.entry)
                self.assertEqual('source_record_training' in self.entry['files'], needs_records)
                self.assertNotIn('source_record_sequence_training', self.entry['files'])
                self.assertNotIn('source_supplemental_sequence', self.entry['files'])
                if needs_records:
                    broken = deepcopy(self.entry)
                    del broken['files']['source_record_training']
                    with self.assertRaisesRegex(ValueError, '기록 ID'):
                        registry.validate_checkpoint(broken)

    def test_local_combinations_need_individually_qualified_members_and_versioned_backends(self):
        self.write_linked_fixture(self.local_specs()[2])
        registry.validate_checkpoint(self.entry)
        self.assertEqual(len(pipeline_sources(self.spec)), 7)
        self.assertNotIn('src/pipeline_recipes.py', pipeline_sources(self.spec))
        self.rewrite_evidence('selection', lambda row: row['candidates'][0].update(qualified=False))
        selection_hash = self.entry['files']['selection']['sha256']
        self.rewrite_evidence('evaluation', lambda row: row.update(selection_sha256=selection_hash))
        self.rewrite_evidence('reproducibility', lambda row: row.update(selection_sha256=selection_hash))
        with self.assertRaisesRegex(ValueError, '검증된 개선 후보만'):
            registry.validate_checkpoint(self.entry)
        for source in ('src/record_prototype_v1.py', 'src/class_conditional_mixture_v1.py'):
            self.write_linked_fixture(self.local_specs()[2])
            self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
            registry.register_checkpoint(self.entry)
            implementation = self.root / source
            implementation.write_bytes(implementation.read_bytes() + b'\n# Changed local backend.\n')
            with mock.patch('joblib.load', side_effect=AssertionError('No unpickling discovery')):
                self.assertEqual(registry.available_checkpoints(), [])

    def test_gain_checkpoints_need_training_sequences_and_record_ids_then_replay_without_ids(self):
        from src import augmentation_pipeline_v1
        from src.app_services import TrainingJob
        from src.model_catalog import MODEL_NAMES, pipeline_runtime_packages
        from src.model_comparison import _dataset
        from src.model_runtime import ModelRegistry
        from src.model_runtime.sklearn_adapter import SklearnAdapter
        from src.posture_core import read_sensor_csv
        from tests.test_model_comparison import ModelComparisonTests
        source = Path(__file__).resolve().parents[1]
        ModelComparisonTests().make_data(self.root / 'data', rows=73)
        partitions = {'train': ['subject0', 'subject1'], 'validation': ['subject2'], 'final_evaluation': ['subject3']}
        data = _dataset(self.root / 'data', 3, 41, partitions)
        mask = data['masks']['train']
        records = data['meta'].loc[mask, 'record_id'].to_numpy()
        groups = data['meta'].loc[mask, 'subject'].to_numpy()
        self.assertLess(len(set(records)), len(records))
        original_fit = augmentation_pipeline_v1.fit_augmentation_pipeline
        captures = []
        def capture(model, recipe, X, y, *, groups=None, record_ids=None, validation_data=None):
            captures.append((np.array(X), np.array(y), np.array(groups), np.array(record_ids)))
            return original_fit(model, recipe, X, y, groups=groups, record_ids=record_ids, validation_data=validation_data)
        for pipeline in self.gain_specs():
            with self.subTest(family=pipeline['family']):
                captures.clear()
                self.write_linked_fixture(pipeline)
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                family = pipeline['family']
                self.assertTrue(is_gain_recipe(pipeline))
                self.assertFalse(is_spatial_recipe(pipeline))
                self.assertTrue(requires_record_ids(pipeline))
                self.assertEqual(input_representation(pipeline), 'sensor_sequence')
                self.assertEqual(combination_family(pipeline), 'gain_soft_voting')
                self.assertEqual(len(MODEL_NAMES), 9)
                self.assertNotIn(family, MODEL_NAMES)
                self.assertEqual(pipeline_runtime_packages(pipeline), {})
                with self.assertRaises(ValueError):
                    TrainingWizardState().configure([3], 41, [family])
                state = TrainingWizardState(training_method=self.method)
                state.configure([3], 41, [family])
                restored = TrainingWizardState()
                apply_configuration(restored, build_configuration(state))
                self.assertEqual(restored.models, (family,))
                self.assertEqual(method_candidates(family, self.method), [pipeline['model_params']])
                job = TrainingJob(source, (), 41, models=(family,), state_root=self.root / family,
                                  training_method=self.method)
                with mock.patch.object(augmentation_pipeline_v1, 'fit_augmentation_pipeline', capture):
                    report = train_comparison(self.root / 'data', job.run_dir, [3], 41, [family],
                        training_method=self.method, progress=lambda _: None, fixed_subject_partitions=partitions)
                self.assertEqual(report['failures'], [])
                self.assertEqual(len(captures), 1)
                observed_X, observed_y, observed_groups, observed_records = captures[0]
                np.testing.assert_array_equal(observed_X, data['sequences'][mask])
                np.testing.assert_array_equal(observed_y, data['y'][mask])
                np.testing.assert_array_equal(observed_groups, groups)
                np.testing.assert_array_equal(observed_records, records)
                self.assertFalse(set(observed_records) & set(data['meta'].loc[~mask, 'record_id']))
                bundle = joblib.load(job.run_dir / 'artifacts' / report['models'][0]['artifact_file'])
                self.assertEqual(bundle['model'].effective_seed_, 53)
                self.assertEqual(bundle['model'].spec_, pipeline)
                self.assertEqual(bundle['input_representation'], 'sensor_sequence')
                self.assertEqual(bundle['feature_names'], [])
                self.assertEqual(bundle['model'].input_shape_, (36, 16))
                augmentation_pipeline_v1.verify_fitted_augmentation_identity(bundle['model'], pipeline, 53)
                leaves = bundle['model'].members_ if family == 'gain_soft_voting' else [bundle['model']]
                for leaf in leaves:
                    self.assertEqual(leaf.scaler_.n_samples_seen_, int(mask.sum()))
                manifest, = job._write_pending_candidate_manifests()
                runtime_registry = ModelRegistry(source, self.root / family)
                descriptor = runtime_registry.validator.discover_manifest(manifest).descriptor
                validated = runtime_registry.validate_runtime(descriptor)
                self.assertEqual(validated.validation_status, 'passed', validated.lifecycle_reason)
                replay = SklearnAdapter(validated).predict_proba(read_sensor_csv(source / 'example/sample_input.csv'), 3)
                self.assertEqual(replay.shape[1], 7)
                self.assertTrue(np.isfinite(replay).all())
                np.testing.assert_allclose(replay.sum(1), 1, atol=1e-6)
                self.assertEqual(validated.approval_status, 'pending')
                unfitted = build_method_model(family, pipeline['model_params'], 41, self.method)
                for missing in ('groups', 'record_ids'):
                    inputs = {'groups': groups, 'record_ids': records}
                    inputs[missing] = None
                    with self.assertRaises(ValueError):
                        fit_method_model(unfitted, family, self.method, data['sequences'][mask], data['y'][mask], **inputs)

    def test_gain_restoration_requires_sequence_alignment_and_joint_record_helper(self):
        from src.checkpoint_registry import _validate_supplement
        pipeline = self.gain_specs()[0]
        pipeline['training']['data_policy'] = 'include_ordinary_reserved_training_records_v1'
        self.write_linked_fixture(pipeline)
        registry.validate_checkpoint(self.entry)
        required = {'source_record_sequence_training', 'source_supplemental', 'source_supplemental_sequence'}
        self.assertTrue(required.issubset(self.entry['files']))
        self.assertNotIn('source_record_training', self.entry['files'])
        for role in required:
            with self.subTest(missing=role):
                broken = deepcopy(self.entry)
                del broken['files'][role]
                with self.assertRaises(ValueError):
                    registry.validate_checkpoint(broken)
        protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
        metrics = json.loads((self.root / self.entry['files']['final_metrics']['path']).read_text())
        protocol['source_hashes']['experiments/iterative_improvement/record_sequence_training.py'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, '기록 ID'):
            _validate_supplement(self.entry, self.root, pipeline, protocol, metrics)

    def test_gain_members_need_independent_gains_and_augmentation_source_is_pinned(self):
        self.write_linked_fixture(self.gain_specs()[-1])
        registry.validate_checkpoint(self.entry)
        self.assertIn('src/record_gain_features_v1.py', pipeline_sources(self.spec))
        self.rewrite_evidence('selection', lambda row: row['candidates'][0].update(qualified=False))
        selection_hash = self.entry['files']['selection']['sha256']
        self.rewrite_evidence('evaluation', lambda row: row.update(selection_sha256=selection_hash))
        self.rewrite_evidence('reproducibility', lambda row: row.update(selection_sha256=selection_hash))
        with self.assertRaisesRegex(ValueError, '검증된 개선 후보만'):
            registry.validate_checkpoint(self.entry)
        self.write_linked_fixture(self.gain_specs()[0])
        registry.register_checkpoint(self.entry)
        feature_source = self.root / 'src/record_gain_features_v1.py'
        feature_source.write_bytes(feature_source.read_bytes() + b'\n# Changed record gain procedure.\n')
        with mock.patch('joblib.load', side_effect=AssertionError('No unpickling discovery')):
            self.assertEqual(registry.available_checkpoints(), [])

    def interaction_specs(self):
        leaves = [validate_pipeline({'family': 'interaction_extra_trees',
            'feature_engineering': {'mode': mode}, 'model_params': {'n_estimators': 4, 'min_samples_leaf': 1},
            'training': {'seed': 53}}) for mode in ('identity', 'hadamard512', 'temporal_corr120')]
        ensemble = validate_pipeline({'family': 'interaction_soft_voting', 'members': leaves[1:],
            'feature_engineering': {'mode': 'per_member'}, 'training': {'seed': 53, 'weighting': 'per_member'}})
        return (*leaves, ensemble)

    def test_interaction_checkpoints_train_only_sequences_then_pending_runtime_csv_replay(self):
        from src import interaction_pipeline_v1
        from src.app_services import TrainingJob
        from src.interaction_features_v1 import extract_interaction_features
        from src.model_catalog import MODEL_NAMES, pipeline_runtime_packages
        from src.model_comparison import _dataset
        from src.model_runtime import ModelRegistry
        from src.model_runtime.sklearn_adapter import SklearnAdapter
        from src.posture_core import read_sensor_csv
        from tests.test_model_comparison import ModelComparisonTests
        source = Path(__file__).resolve().parents[1]
        ModelComparisonTests().make_data(self.root / 'data', rows=73)
        partitions = {'train': ['subject0', 'subject1'], 'validation': ['subject2'], 'final_evaluation': ['subject3']}
        data = _dataset(self.root / 'data', 3, 41, partitions)
        mask = data['masks']['train']
        groups = data['meta'].loc[mask, 'subject'].to_numpy()
        original_fit = interaction_pipeline_v1.fit_interaction_pipeline
        captures = []
        # This strict signature also verifies that unrelated record metadata is
        # not forwarded into the no-record-ID interaction fitting contract.
        def capture(model, recipe, X, y, *, groups=None, validation_data=None):
            captures.append((np.array(X), np.array(y), np.array(groups)))
            return original_fit(model, recipe, X, y, groups=groups, validation_data=validation_data)
        for index, pipeline in enumerate(self.interaction_specs()):
            with self.subTest(family=pipeline['family'], features=pipeline['feature_engineering']):
                captures.clear()
                self.write_linked_fixture(pipeline)
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                family = pipeline['family']
                self.assertTrue(is_interaction_recipe(pipeline))
                self.assertFalse(is_gain_recipe(pipeline))
                self.assertFalse(requires_record_ids(pipeline))
                self.assertEqual(input_representation(pipeline), 'sensor_sequence')
                self.assertEqual(combination_family(pipeline), 'interaction_soft_voting')
                self.assertEqual(len(MODEL_NAMES), 9)
                self.assertNotIn(family, MODEL_NAMES)
                self.assertEqual(pipeline_runtime_packages(pipeline), {})
                with self.assertRaises(ValueError):
                    TrainingWizardState().configure([3], 41, [family])
                state = TrainingWizardState(training_method=self.method)
                state.configure([3], 41, [family])
                restored = TrainingWizardState()
                apply_configuration(restored, build_configuration(state))
                self.assertEqual(restored.models, (family,))
                self.assertEqual(method_candidates(family, self.method), [pipeline['model_params']])
                state_root = self.root / f'interaction-{index}'
                job = TrainingJob(source, (), 41, models=(family,), state_root=state_root, training_method=self.method)
                with mock.patch.object(interaction_pipeline_v1, 'fit_interaction_pipeline', capture):
                    report = train_comparison(self.root / 'data', job.run_dir, [3], 41, [family],
                        training_method=self.method, progress=lambda _: None, fixed_subject_partitions=partitions)
                self.assertEqual(report['failures'], [])
                self.assertEqual(len(captures), 1)
                observed_X, observed_y, observed_groups = captures[0]
                np.testing.assert_array_equal(observed_X, data['sequences'][mask])
                np.testing.assert_array_equal(observed_y, data['y'][mask])
                np.testing.assert_array_equal(observed_groups, groups)
                self.assertFalse(set(observed_groups) & set(data['meta'].loc[~mask, 'subject']))
                bundle = joblib.load(job.run_dir / 'artifacts' / report['models'][0]['artifact_file'])
                model = bundle['model']
                self.assertEqual(model.effective_seed_, 53)
                self.assertEqual(model.spec_, pipeline)
                self.assertEqual(model.input_shape_, (36, 16))
                self.assertEqual(bundle['input_representation'], 'sensor_sequence')
                self.assertEqual(bundle['feature_names'], [])
                interaction_pipeline_v1.verify_fitted_interaction_identity(model, pipeline, 53)
                leaves = model.members_ if family == 'interaction_soft_voting' else [model]
                for leaf in leaves:
                    self.assertEqual(leaf.scaler_.n_samples_seen_, int(mask.sum()))
                    extracted = extract_interaction_features(data['sequences'][mask], leaf.spec_['feature_engineering']['mode'])
                    np.testing.assert_allclose(leaf.scaler_.mean_, np.mean(extracted, axis=0, dtype=np.float64))
                manifest, = job._write_pending_candidate_manifests()
                runtime_registry = ModelRegistry(source, state_root)
                descriptor = runtime_registry.validator.discover_manifest(manifest).descriptor
                validated = runtime_registry.validate_runtime(descriptor)
                self.assertEqual(validated.validation_status, 'passed', validated.lifecycle_reason)
                replay = SklearnAdapter(validated).predict_proba(read_sensor_csv(source / 'example/sample_input.csv'), 3)
                self.assertEqual(replay.shape[1], 7)
                self.assertTrue(np.isfinite(replay).all())
                np.testing.assert_allclose(replay.sum(1), 1, atol=1e-6)
                self.assertEqual(validated.approval_status, 'pending')

    def test_interaction_identity_control_rejects_semantic_changes_despite_bitwise_audit(self):
        self.identity_control_fixture('interaction_extra_trees')
        registry.validate_checkpoint(self.entry)
        protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
        validate_identity_control(protocol['baseline'], protocol['branches']['identity']['identity_a'])
        mutations = {
            'hadamard transform': lambda ref, row: row['feature_engineering'].update(mode='hadamard512'),
            'temporal correlations': lambda ref, row: row['feature_engineering'].update(mode='temporal_corr120'),
            'trees': lambda ref, row: row['model_params'].update(n_estimators=201),
            'reference missing parameter': lambda ref, row: ref['model_params'].pop('max_depth'),
            'integer versus fraction': lambda ref, row: (ref['model_params'].update(max_features=1), row['model_params'].update(max_features=1.0)),
            'seed': lambda ref, row: row['training'].update(seed=54),
            'policy': lambda ref, row: row['training'].update(data_policy='include_ordinary_reserved_training_records_v1'),
            'reference weighting': lambda ref, row: ref['training'].update(weighting='subject_class_balanced'),
            'reference subset': lambda ref, row: ref['feature_engineering'].update(mode='means'),
            'reference scaler': lambda ref, row: ref['preprocessing'].update(scaler='robust'),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = deepcopy(protocol)
                mutate(changed['baseline'], changed['branches']['identity']['identity_a'])
                with self.assertRaises(ValueError):
                    validate_identity_control(changed['baseline'], changed['branches']['identity']['identity_a'])
                with self.assertRaisesRegex(ValueError, '동일 경로 대조군'):
                    registry._validate_identity_controls(self.entry, self.root, changed)
        with self.assertRaises(ValueError):
            validate_identity_control(protocol['baseline'], self.interaction_specs()[-1])

    def test_interaction_supplement_requires_sequence_evidence_without_record_helper(self):
        for pipeline in (self.interaction_specs()[1], self.interaction_specs()[-1]):
            with self.subTest(family=pipeline['family']):
                pipeline['training']['data_policy'] = 'include_ordinary_reserved_training_records_v1'
                for member in pipeline.get('members', []):
                    member['training']['data_policy'] = pipeline['training']['data_policy']
                self.write_linked_fixture(pipeline)
                registry.validate_checkpoint(self.entry)
                self.assertNotIn('source_record_training', self.entry['files'])
                self.assertNotIn('source_record_sequence_training', self.entry['files'])
                self.assertIn('source_supplemental_sequence', self.entry['files'])
                for role in ('source_supplemental', 'source_supplemental_sequence', 'supplemental_manifest', 'supplemental_cache'):
                    broken = deepcopy(self.entry)
                    del broken['files'][role]
                    with self.assertRaises(ValueError):
                        registry.validate_checkpoint(broken)

    def test_interaction_members_need_independent_gains_and_all_feature_sources_are_pinned(self):
        self.write_linked_fixture(self.interaction_specs()[-1])
        registry.validate_checkpoint(self.entry)
        self.assertEqual(set(pipeline_sources(self.spec)), {'src/interaction_pipeline_v1.py',
            'src/interaction_features_v1.py', 'src/record_gain_features_v1.py', 'src/posture_core.py',
            'src/window_pipeline.py', 'src/__init__.py'})
        self.rewrite_evidence('selection', lambda row: row['candidates'][0].update(qualified=False))
        selection_hash = self.entry['files']['selection']['sha256']
        self.rewrite_evidence('evaluation', lambda row: row.update(selection_sha256=selection_hash))
        self.rewrite_evidence('reproducibility', lambda row: row.update(selection_sha256=selection_hash))
        with self.assertRaisesRegex(ValueError, '검증된 개선 후보만'):
            registry.validate_checkpoint(self.entry)
        for relative in ('src/interaction_features_v1.py', 'src/record_gain_features_v1.py'):
            with self.subTest(source=relative):
                self.write_linked_fixture(self.interaction_specs()[1])
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                feature_source = self.root / relative
                feature_source.write_bytes(feature_source.read_bytes() + b'\n# Changed interaction input procedure.\n')
                with mock.patch('joblib.load', side_effect=AssertionError('No unpickling discovery')):
                    self.assertEqual(registry.available_checkpoints(), [])

    def support_specs(self):
        leaves = [validate_pipeline({'family': 'support_extra_trees',
            'model_params': {'n_estimators': 4, 'min_samples_leaf': 1},
            'training': {'seed': 53}, 'postprocessing': {'kind': 'uniform_leaf_shrinkage', 'alpha': alpha}})
            for alpha in (0.0, 8.0, 32.0)]
        ensemble = validate_pipeline({'family': 'support_soft_voting', 'members': leaves[1:],
            'training': {'seed': 53, 'weighting': 'per_member'}, 'postprocessing': {'kind': 'argmax'}})
        return leaves[0], leaves[1], ensemble

    def test_support_checkpoints_train_only_feature_rows_then_pending_runtime_csv_replay(self):
        from src import forest_support_pipeline_v1
        from src.app_services import TrainingJob
        from src.model_catalog import MODEL_NAMES, pipeline_runtime_packages
        from src.model_comparison import _dataset
        from src.model_runtime import ModelRegistry
        from src.model_runtime.sklearn_adapter import SklearnAdapter
        from src.posture_core import read_sensor_csv
        from tests.test_model_comparison import ModelComparisonTests
        source = Path(__file__).resolve().parents[1]
        ModelComparisonTests().make_data(self.root / 'data', rows=73)
        partitions = {'train': ['subject0', 'subject1'], 'validation': ['subject2'], 'final_evaluation': ['subject3']}
        data = _dataset(self.root / 'data', 3, 41, partitions)
        mask = data['masks']['train']
        groups = data['meta'].loc[mask, 'subject'].to_numpy()
        original_fit = forest_support_pipeline_v1.fit_support_pipeline
        captures = []
        def capture(model, recipe, X, y, *, groups=None, validation_data=None):
            captures.append((np.array(X), np.array(y), np.array(groups)))
            return original_fit(model, recipe, X, y, groups=groups, validation_data=validation_data)
        for index, pipeline in enumerate(self.support_specs()):
            with self.subTest(family=pipeline['family'], postprocessing=pipeline['postprocessing']):
                captures.clear()
                self.write_linked_fixture(pipeline)
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                family = pipeline['family']
                self.assertTrue(is_support_recipe(pipeline))
                self.assertFalse(is_interaction_recipe(pipeline))
                self.assertFalse(requires_record_ids(pipeline))
                self.assertEqual(input_representation(pipeline), 'feature_vector')
                self.assertEqual(combination_family(pipeline), 'support_soft_voting')
                self.assertEqual(len(MODEL_NAMES), 9)
                self.assertNotIn(family, MODEL_NAMES)
                self.assertEqual(pipeline_runtime_packages(pipeline), {})
                with self.assertRaises(ValueError):
                    TrainingWizardState().configure([3], 41, [family])
                state = TrainingWizardState(training_method=self.method)
                state.configure([3], 41, [family])
                restored = TrainingWizardState()
                apply_configuration(restored, build_configuration(state))
                self.assertEqual(restored.models, (family,))
                self.assertEqual(method_candidates(family, self.method), [pipeline['model_params']])
                state_root = self.root / f'support-{index}'
                job = TrainingJob(source, (), 41, models=(family,), state_root=state_root, training_method=self.method)
                with mock.patch.object(forest_support_pipeline_v1, 'fit_support_pipeline', capture):
                    report = train_comparison(self.root / 'data', job.run_dir, [3], 41, [family],
                        training_method=self.method, progress=lambda _: None, fixed_subject_partitions=partitions)
                self.assertEqual(report['failures'], [])
                self.assertEqual(len(captures), 1)
                observed_X, observed_y, observed_groups = captures[0]
                np.testing.assert_array_equal(observed_X, data['X'][mask])
                np.testing.assert_array_equal(observed_y, data['y'][mask])
                np.testing.assert_array_equal(observed_groups, groups)
                self.assertFalse(set(observed_groups) & set(data['meta'].loc[~mask, 'subject']))
                bundle = joblib.load(job.run_dir / 'artifacts' / report['models'][0]['artifact_file'])
                model = bundle['model']
                self.assertEqual(model.effective_seed_, 53)
                self.assertEqual(model.spec_, pipeline)
                self.assertEqual(bundle['input_representation'], 'feature_vector')
                self.assertEqual(bundle['feature_names'], feature_names())
                forest_support_pipeline_v1.verify_fitted_support_identity(model, pipeline, 53)
                leaves = model.members_ if family == 'support_soft_voting' else [model]
                for leaf in leaves:
                    self.assertEqual(leaf.scaler_.n_samples_seen_, int(mask.sum()))
                    np.testing.assert_allclose(leaf.scaler_.mean_, np.mean(data['X'][mask], axis=0, dtype=np.float64))
                manifest, = job._write_pending_candidate_manifests()
                runtime_registry = ModelRegistry(source, state_root)
                descriptor = runtime_registry.validator.discover_manifest(manifest).descriptor
                validated = runtime_registry.validate_runtime(descriptor)
                self.assertEqual(validated.validation_status, 'passed', validated.lifecycle_reason)
                replay = SklearnAdapter(validated).predict_proba(read_sensor_csv(source / 'example/sample_input.csv'), 3)
                self.assertEqual(replay.shape[1], 7)
                self.assertTrue(np.isfinite(replay).all())
                np.testing.assert_allclose(replay.sum(1), 1, atol=1e-6)
                self.assertEqual(validated.approval_status, 'pending')

    def test_support_identity_control_rejects_shrinkage_or_recipe_drift_despite_equal_audit(self):
        self.identity_control_fixture('support_extra_trees')
        registry.validate_checkpoint(self.entry)
        protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
        validate_identity_control(protocol['baseline'], protocol['branches']['identity']['identity_a'])
        mutations = {
            'alpha8': lambda ref, row: row['postprocessing'].update(alpha=8.0),
            'alpha32': lambda ref, row: row['postprocessing'].update(alpha=32.0),
            'argmax': lambda ref, row: row.update(postprocessing={'kind': 'argmax'}),
            'trees': lambda ref, row: row['model_params'].update(n_estimators=201),
            'missing explicit reference parameter': lambda ref, row: ref['model_params'].pop('max_depth'),
            'integer versus fraction': lambda ref, row: (ref['model_params'].update(max_features=1), row['model_params'].update(max_features=1.0)),
            'seed': lambda ref, row: row['training'].update(seed=54),
            'policy': lambda ref, row: row['training'].update(data_policy='include_ordinary_reserved_training_records_v1'),
            'reference weighting': lambda ref, row: ref['training'].update(weighting='subject_class_balanced'),
            'reference features': lambda ref, row: ref['feature_engineering'].update(mode='means'),
            'reference scaler': lambda ref, row: ref['preprocessing'].update(scaler='robust'),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = deepcopy(protocol)
                mutate(changed['baseline'], changed['branches']['identity']['identity_a'])
                with self.assertRaises(ValueError):
                    validate_identity_control(changed['baseline'], changed['branches']['identity']['identity_a'])
                with self.assertRaisesRegex(ValueError, '동일 경로 대조군'):
                    registry._validate_identity_controls(self.entry, self.root, changed)
        with self.assertRaises(ValueError):
            validate_identity_control(protocol['baseline'], self.support_specs()[-1])

    def test_support_supplement_uses_frozen_feature_helper_without_record_or_sequence_helpers(self):
        for pipeline in (self.support_specs()[1], self.support_specs()[-1]):
            with self.subTest(family=pipeline['family']):
                pipeline['training']['data_policy'] = 'include_ordinary_reserved_training_records_v1'
                for member in pipeline.get('members', []):
                    member['training']['data_policy'] = pipeline['training']['data_policy']
                self.write_linked_fixture(pipeline)
                registry.validate_checkpoint(self.entry)
                for role in ('source_record_training', 'source_record_sequence_training', 'source_supplemental_sequence'):
                    self.assertNotIn(role, self.entry['files'])
                for role in ('source_supplemental', 'supplemental_manifest', 'supplemental_cache'):
                    broken = deepcopy(self.entry)
                    del broken['files'][role]
                    with self.assertRaises(ValueError):
                        registry.validate_checkpoint(broken)

    def test_support_members_need_independent_gains_and_shrinkage_implementation_is_pinned(self):
        self.write_linked_fixture(self.support_specs()[-1])
        registry.validate_checkpoint(self.entry)
        self.assertEqual(set(pipeline_sources(self.spec)), {'src/forest_support_pipeline_v1.py',
            'src/leaf_support_v1.py', 'src/posture_core.py', 'src/window_pipeline.py', 'src/__init__.py'})
        self.rewrite_evidence('selection', lambda row: row['candidates'][0].update(qualified=False))
        selection_hash = self.entry['files']['selection']['sha256']
        self.rewrite_evidence('evaluation', lambda row: row.update(selection_sha256=selection_hash))
        self.rewrite_evidence('reproducibility', lambda row: row.update(selection_sha256=selection_hash))
        with self.assertRaisesRegex(ValueError, '검증된 개선 후보만'):
            registry.validate_checkpoint(self.entry)
        self.write_linked_fixture(self.support_specs()[1])
        registry.register_checkpoint(self.entry)
        helper = self.root / 'src/leaf_support_v1.py'
        helper.write_bytes(helper.read_bytes() + b'\n# Changed leaf shrinkage rule.\n')
        with mock.patch('joblib.load', side_effect=AssertionError('No unpickling discovery')):
            self.assertEqual(registry.available_checkpoints(), [])

    def alternatives_specs(self):
        tails = [validate_pipeline({'family': 'tail_extra_trees',
            'model_params': {'n_estimators': 4, 'min_samples_leaf': 1},
            'preprocessing': {'scaler': 'standard', 'tail_fraction': fraction},
            'training': {'seed': 53}}) for fraction in (0.0, .01)]
        binary = validate_pipeline({'family': 'ovr_extra_trees',
            'model_params': {'n_estimators': 4, 'min_samples_leaf': 1},
            'training': {'seed': 53, 'weighting': 'binary_class_balanced'}})
        ensemble = validate_pipeline({'family': 'alternatives_soft_voting', 'members': [tails[1], binary],
            'training': {'seed': 53, 'weighting': 'per_member'}})
        return tails[0], tails[1], binary, ensemble

    def test_alternatives_checkpoints_train_only_rows_with_correct_scaler_then_pending_csv_replay(self):
        from src import forest_alternatives_pipeline_v1
        from src.app_services import TrainingJob
        from src.binary_forest_v1 import BinaryForestV1, verify_binary_forest
        from src.model_catalog import MODEL_NAMES, pipeline_runtime_packages
        from src.model_comparison import _dataset
        from src.model_runtime import ModelRegistry
        from src.model_runtime.sklearn_adapter import SklearnAdapter
        from src.posture_core import read_sensor_csv
        from src.tail_clipping_v1 import verify_tail_clipper
        from tests.test_model_comparison import ModelComparisonTests
        source = Path(__file__).resolve().parents[1]
        ModelComparisonTests().make_data(self.root / 'data', rows=73)
        partitions = {'train': ['subject0', 'subject1'], 'validation': ['subject2'], 'final_evaluation': ['subject3']}
        data = _dataset(self.root / 'data', 3, 41, partitions)
        mask = data['masks']['train']
        groups = data['meta'].loc[mask, 'subject'].to_numpy()
        original_fit = forest_alternatives_pipeline_v1.fit_alternatives_pipeline
        captures = []
        # No record_ids parameter: forwarding metadata for unrelated estimators
        # is an integration failure, while validation is never used by fit.
        def capture(model, recipe, X, y, *, groups=None, validation_data=None):
            captures.append((np.array(X), np.array(y), np.array(groups)))
            return original_fit(model, recipe, X, y, groups=groups, validation_data=validation_data)
        for index, pipeline in enumerate(self.alternatives_specs()):
            with self.subTest(family=pipeline['family'], preprocessing=pipeline['preprocessing']):
                captures.clear()
                self.write_linked_fixture(pipeline)
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                family = pipeline['family']
                self.assertTrue(is_alternatives_recipe(pipeline))
                self.assertFalse(is_support_recipe(pipeline))
                self.assertFalse(requires_record_ids(pipeline))
                self.assertEqual(input_representation(pipeline), 'feature_vector')
                self.assertEqual(combination_family(pipeline), 'alternatives_soft_voting')
                self.assertEqual(len(MODEL_NAMES), 9)
                self.assertNotIn(family, MODEL_NAMES)
                self.assertEqual(pipeline_runtime_packages(pipeline), {})
                with self.assertRaises(ValueError):
                    TrainingWizardState().configure([3], 41, [family])
                state = TrainingWizardState(training_method=self.method)
                state.configure([3], 41, [family])
                restored = TrainingWizardState()
                apply_configuration(restored, build_configuration(state))
                self.assertEqual(restored.models, (family,))
                self.assertEqual(method_candidates(family, self.method), [pipeline['model_params']])
                state_root = self.root / f'alternatives-{index}'
                job = TrainingJob(source, (), 41, models=(family,), state_root=state_root, training_method=self.method)
                with mock.patch.object(forest_alternatives_pipeline_v1, 'fit_alternatives_pipeline', capture):
                    report = train_comparison(self.root / 'data', job.run_dir, [3], 41, [family],
                        training_method=self.method, progress=lambda _: None, fixed_subject_partitions=partitions)
                self.assertEqual(report['failures'], [])
                self.assertEqual(len(captures), 1)
                observed_X, observed_y, observed_groups = captures[0]
                np.testing.assert_array_equal(observed_X, data['X'][mask])
                np.testing.assert_array_equal(observed_y, data['y'][mask])
                np.testing.assert_array_equal(observed_groups, groups)
                self.assertFalse(set(observed_groups) & set(data['meta'].loc[~mask, 'subject']))
                bundle = joblib.load(job.run_dir / 'artifacts' / report['models'][0]['artifact_file'])
                model = bundle['model']
                self.assertEqual(model.effective_seed_, 53)
                self.assertEqual(model.spec_, pipeline)
                self.assertEqual(bundle['input_representation'], 'feature_vector')
                self.assertEqual(bundle['feature_names'], feature_names())
                self.assertTrue(forest_alternatives_pipeline_v1.verify_fitted_alternatives_identity(model, pipeline, 53))
                leaves = model.members_ if family == 'alternatives_soft_voting' else [model]
                for leaf in leaves:
                    self.assertFalse(leaf.training_details_['external_validation_used'])
                    self.assertFalse(leaf.training_details_['evaluation_population_normalization'])
                    self.assertEqual(leaf.training_subjects_, sorted(partitions['train']))
                    self.assertEqual(leaf.scaler_.n_samples_seen_, int(mask.sum()))
                    prepared = data['X'][mask]
                    if leaf.spec_['family'] == 'tail_extra_trees':
                        fraction = leaf.spec_['preprocessing']['tail_fraction']
                        self.assertTrue(verify_tail_clipper(leaf.clipper_, fraction))
                        self.assertEqual(leaf.clipper_.n_samples_seen_, int(mask.sum()))
                        if fraction:
                            bounds = np.quantile(prepared.astype(np.float64), [fraction, 1 - fraction], axis=0, method='linear')
                            np.testing.assert_array_equal(leaf.clipper_.lower_bounds_, bounds[0])
                            np.testing.assert_array_equal(leaf.clipper_.upper_bounds_, bounds[1])
                            prepared = np.clip(prepared, bounds[0], bounds[1]).astype(prepared.dtype)
                            self.assertFalse(np.array_equal(prepared, data['X'][mask]))
                        else:
                            self.assertIsNone(leaf.clipper_.lower_bounds_)
                            self.assertIsNone(leaf.clipper_.upper_bounds_)
                            self.assertEqual(leaf.prepared_state_sha256_, leaf.training_input_sha256_)
                    else:
                        self.assertFalse(hasattr(leaf, 'clipper_'))
                        self.assertIsInstance(leaf.estimator_, BinaryForestV1)
                        self.assertTrue(verify_binary_forest(leaf.estimator_, leaf.spec_['model_params'], 53))
                        self.assertEqual(leaf.prepared_state_sha256_, leaf.training_input_sha256_)
                        self.assertEqual(leaf.estimator_.effective_seeds_, list(range(53, 60)))
                    np.testing.assert_allclose(leaf.scaler_.mean_, np.mean(prepared, axis=0, dtype=np.float64), rtol=0, atol=0)
                manifest, = job._write_pending_candidate_manifests()
                runtime_registry = ModelRegistry(source, state_root)
                descriptor = runtime_registry.validator.discover_manifest(manifest).descriptor
                validated = runtime_registry.validate_runtime(descriptor)
                self.assertEqual(validated.validation_status, 'passed', validated.lifecycle_reason)
                replay = SklearnAdapter(validated).predict_proba(read_sensor_csv(source / 'example/sample_input.csv'), 3)
                self.assertEqual(replay.shape[1], 7)
                self.assertTrue(np.isfinite(replay).all())
                np.testing.assert_allclose(replay.sum(1), 1, atol=1e-6)
                self.assertEqual(validated.approval_status, 'pending')

    def test_alternatives_identity_control_rejects_clipping_binary_or_recipe_drift_despite_equal_audit(self):
        self.identity_control_fixture('tail_extra_trees')
        registry.validate_checkpoint(self.entry)
        protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
        control = protocol['branches']['identity']['identity_a']
        self.assertEqual(control['preprocessing'], {'scaler': 'standard', 'tail_fraction': 0.0})
        validate_identity_control(protocol['baseline'], control)
        def replace_with_binary(reference, row):
            row.clear()
            row.update(validate_pipeline({'family': 'ovr_extra_trees', 'model_params': reference['model_params'],
                'training': {'seed': 53, 'weighting': 'binary_class_balanced'}}))
        mutations = {
            'fraction0.005': lambda ref, row: row['preprocessing'].update(tail_fraction=.005),
            'fraction0.01': lambda ref, row: row['preprocessing'].update(tail_fraction=.01),
            'binary targets': replace_with_binary,
            'trees': lambda ref, row: row['model_params'].update(n_estimators=201),
            'seed': lambda ref, row: row['training'].update(seed=54),
            'policy': lambda ref, row: row['training'].update(data_policy='include_ordinary_reserved_training_records_v1'),
            'missing explicit reference parameter': lambda ref, row: ref['model_params'].pop('max_depth'),
            'reference weighting': lambda ref, row: ref['training'].update(weighting='subject_class_balanced'),
            'reference features': lambda ref, row: ref['feature_engineering'].update(mode='means'),
            'reference scaler': lambda ref, row: ref['preprocessing'].update(scaler='robust'),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = deepcopy(protocol)
                mutate(changed['baseline'], changed['branches']['identity']['identity_a'])
                with self.assertRaises(ValueError):
                    validate_identity_control(changed['baseline'], changed['branches']['identity']['identity_a'])
                with self.assertRaisesRegex(ValueError, '동일 경로 대조군'):
                    registry._validate_identity_controls(self.entry, self.root, changed)
        with self.assertRaises(ValueError):
            validate_identity_control(protocol['baseline'], self.alternatives_specs()[-1])

    def test_alternatives_supplement_requires_frozen_feature_evidence_and_train_only_counts(self):
        for pipeline in self.alternatives_specs()[1:]:
            with self.subTest(family=pipeline['family']):
                pipeline['training']['data_policy'] = 'include_ordinary_reserved_training_records_v1'
                for member in pipeline.get('members', []):
                    member['training']['data_policy'] = pipeline['training']['data_policy']
                self.write_linked_fixture(pipeline)
                registry.validate_checkpoint(self.entry)
                for role in ('source_record_training', 'source_record_sequence_training', 'source_supplemental_sequence'):
                    self.assertNotIn(role, self.entry['files'])
                for role in ('source_supplemental', 'supplemental_manifest', 'supplemental_cache'):
                    broken = deepcopy(self.entry)
                    del broken['files'][role]
                    with self.assertRaises(ValueError):
                        registry.validate_checkpoint(broken)
                protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
                metrics = json.loads((self.root / self.entry['files']['final_metrics']['path']).read_text())
                for changed in (
                        {'added_record_paths': ['a/ordinary.csv', 'c/ordinary.csv']},
                        {'added_windows': 6}, {'class_counts_after': [5, 3, *([2] * 5)]},
                        {'personal_reference_used': True}):
                    bad = deepcopy(metrics)
                    bad['supplemental_training_audit'].update(changed)
                    with self.assertRaises(ValueError):
                        registry._validate_supplement(self.entry, self.root, pipeline, protocol, bad)

    def test_alternatives_members_need_independent_gains_and_both_helpers_are_pinned(self):
        self.write_linked_fixture(self.alternatives_specs()[-1])
        registry.validate_checkpoint(self.entry)
        self.assertEqual(set(pipeline_sources(self.spec)), {'src/forest_alternatives_pipeline_v1.py',
            'src/tail_clipping_v1.py', 'src/binary_forest_v1.py', 'src/posture_core.py',
            'src/window_pipeline.py', 'src/__init__.py'})
        self.rewrite_evidence('selection', lambda row: row['candidates'][0].update(qualified=False))
        selection_hash = self.entry['files']['selection']['sha256']
        self.rewrite_evidence('evaluation', lambda row: row.update(selection_sha256=selection_hash))
        self.rewrite_evidence('reproducibility', lambda row: row.update(selection_sha256=selection_hash))
        with self.assertRaisesRegex(ValueError, '검증된 개선 후보만'):
            registry.validate_checkpoint(self.entry)
        for relative in ('src/tail_clipping_v1.py', 'src/binary_forest_v1.py'):
            with self.subTest(source=relative):
                self.write_linked_fixture(self.alternatives_specs()[-1])
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                helper = self.root / relative
                helper.write_bytes(helper.read_bytes() + b'\n# Changed alternatives algorithm.\n')
                with mock.patch('joblib.load', side_effect=AssertionError('No unpickling discovery')):
                    self.assertEqual(registry.available_checkpoints(), [])

    def projection_specs(self):
        leaves = [validate_pipeline({'family': 'projection_extra_trees',
            'feature_engineering': {'mode': mode},
            'model_params': {'n_estimators': 4, 'min_samples_leaf': 1},
            'training': {'seed': 53}}) for mode in ('identity', 'pca64', 'all_plus_lda6')]
        ensemble = validate_pipeline({'family': 'projection_soft_voting',
            'members': leaves[1:], 'training': {'seed': 53, 'weighting': 'per_member'}})
        return *leaves, ensemble

    def test_projection_checkpoints_train_only_axes_then_pending_runtime_and_actual_csv(self):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from src import learned_projection_pipeline_v1
        from src.app_services import TrainingJob
        from src.learned_projection_v1 import verify_learned_projection
        from src.model_catalog import MODEL_NAMES, pipeline_runtime_packages
        from src.model_comparison import _dataset
        from src.model_runtime import ModelRegistry
        from src.model_runtime.sklearn_adapter import SklearnAdapter
        from src.posture_core import SENSOR_COLUMNS, read_sensor_csv
        from experiments.iterative_improvement.finalize import _verify_fitted_identity
        source = Path(__file__).resolve().parents[1]
        rng = np.random.default_rng(15017)
        # Eight disjoint windows per posture/person give112 TRAIN rows. Sensor
        # noise varies independently, so PCA64 is tested without rank fallback.
        for subject in range(4):
            directory = self.root / 'data' / f'subject{subject}'
            directory.mkdir(parents=True)
            for label_id, label in enumerate(CLASS_NAMES):
                values = rng.uniform(10., 35., (289, len(SENSOR_COLUMNS)))
                values += label_id * 12 + subject * 2
                frame = pd.DataFrame(values, columns=SENSOR_COLUMNS)
                frame.insert(0, 'label', label)
                frame.insert(0, 'elapsed_s', np.arange(len(values)) / 12.)
                frame.to_csv(directory / f'posture{label_id}.csv', index=False)
        partitions = {'train': ['subject0', 'subject1'], 'validation': ['subject2'],
                      'final_evaluation': ['subject3']}
        data = _dataset(self.root / 'data', 3, 41, partitions)
        mask = data['masks']['train']
        train_X, train_y = data['X'][mask], data['y'][mask]
        groups = data['meta'].loc[mask, 'subject'].to_numpy()
        self.assertGreaterEqual(len(train_X), 65)
        expected_scaler = StandardScaler().fit(train_X)
        standardized = expected_scaler.transform(train_X)
        original_fit = learned_projection_pipeline_v1.fit_projection_pipeline
        captures = []
        # An unrelated record_ids argument must not leak through the dispatcher.
        def capture(model, recipe, X, y, *, groups=None, validation_data=None):
            captures.append((np.array(X), np.array(y), np.array(groups)))
            return original_fit(model, recipe, X, y, groups=groups, validation_data=validation_data)
        for index, pipeline in enumerate(self.projection_specs()):
            with self.subTest(mode=pipeline['feature_engineering']['mode']):
                # Policy documents the research cohort. Model-add fits only the
                # current selected data; the linked historical cache is JSON,
                # so reading it as training data would fail this integration.
                pipeline['training']['data_policy'] = 'include_ordinary_reserved_training_records_v1'
                for member in pipeline.get('members', []):
                    member['training']['data_policy'] = pipeline['training']['data_policy']
                captures.clear()
                self.write_linked_fixture(pipeline)
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                family = pipeline['family']
                self.assertTrue(is_projection_recipe(pipeline))
                self.assertFalse(is_alternatives_recipe(pipeline))
                self.assertFalse(requires_record_ids(pipeline))
                self.assertEqual(input_representation(pipeline), 'feature_vector')
                self.assertEqual(combination_family(pipeline), 'projection_soft_voting')
                self.assertEqual(len(MODEL_NAMES), 9)
                self.assertNotIn(family, MODEL_NAMES)
                self.assertEqual(pipeline_runtime_packages(pipeline), {})
                with self.assertRaises(ValueError):
                    TrainingWizardState().configure([3], 41, [family])
                state = TrainingWizardState(training_method=self.method)
                state.configure([3], 41, [family])
                restored = TrainingWizardState()
                apply_configuration(restored, build_configuration(state))
                self.assertEqual(restored.models, (family,))
                self.assertEqual(method_candidates(family, self.method), [pipeline['model_params']])
                state_root = self.root / f'projection-{index}'
                job = TrainingJob(source, (), 41, models=(family,), state_root=state_root,
                                  training_method=self.method)
                with mock.patch.object(learned_projection_pipeline_v1, 'fit_projection_pipeline', capture):
                    report = train_comparison(self.root / 'data', job.run_dir, [3], 41, [family],
                        training_method=self.method, progress=lambda _: None,
                        fixed_subject_partitions=partitions)
                self.assertEqual(report['failures'], [])
                self.assertEqual(len(captures), 1)
                observed_X, observed_y, observed_groups = captures[0]
                np.testing.assert_array_equal(observed_X, train_X)
                np.testing.assert_array_equal(observed_y, train_y)
                np.testing.assert_array_equal(observed_groups, groups)
                self.assertFalse(set(observed_groups) & set(data['meta'].loc[~mask, 'subject']))
                bundle = joblib.load(job.run_dir / 'artifacts' / report['models'][0]['artifact_file'])
                model = bundle['model']
                self.assertEqual(model.effective_seed_, 53)
                self.assertEqual(model.spec_, pipeline)
                self.assertEqual(bundle['input_representation'], 'feature_vector')
                self.assertEqual(bundle['feature_names'], feature_names())
                self.assertTrue(learned_projection_pipeline_v1.verify_fitted_projection_identity(model, pipeline, 53))
                _verify_fitted_identity(model, pipeline, 53)
                leaves = model.members_ if family == 'projection_soft_voting' else [model]
                for leaf in leaves:
                    mode = leaf.spec_['feature_engineering']['mode']
                    self.assertFalse(leaf.training_details_['external_validation_used'])
                    self.assertFalse(leaf.training_details_['evaluation_population_normalization'])
                    self.assertEqual(leaf.training_subjects_, sorted(partitions['train']))
                    self.assertEqual(leaf.scaler_.n_samples_seen_, len(train_X))
                    np.testing.assert_array_equal(leaf.scaler_.mean_, expected_scaler.mean_)
                    np.testing.assert_array_equal(leaf.scaler_.scale_, expected_scaler.scale_)
                    np.testing.assert_array_equal(leaf.scaler_.transform(train_X), standardized)
                    self.assertEqual(leaf.projector_.n_samples_seen_, len(train_X))
                    np.testing.assert_array_equal(leaf.projector_.class_counts_, np.bincount(train_y))
                    self.assertEqual(leaf.projector_.training_input_sha256_, leaf.standardized_state_sha256_)
                    self.assertEqual(leaf.projector_.training_labels_sha256_, leaf.training_labels_sha256_)
                    self.assertTrue(verify_learned_projection(leaf.projector_, mode))
                    width = {'identity': 596, 'pca64': 64, 'all_plus_lda6': 602}[mode]
                    self.assertEqual(leaf.estimator_.n_features_in_, width)
                    transformed = leaf.projector_.transform(standardized)
                    self.assertEqual(transformed.shape, (len(train_X), width))
                    if mode == 'identity':
                        self.assertIs(transformed, standardized)
                        self.assertEqual(leaf.standardized_state_sha256_, leaf.transformed_state_sha256_)
                    elif mode == 'pca64':
                        self.assertGreater(leaf.projector_.estimator_.singular_values_[-1],
                                           leaf.projector_.rank_tolerance_)
                        np.testing.assert_array_equal(leaf.projector_.estimator_.mean_,
                                                      standardized.mean(axis=0, dtype=np.float64))
                    else:
                        np.testing.assert_array_equal(transformed[:, :596], standardized.astype(np.float64))
                        np.testing.assert_array_equal(leaf.projector_.estimator_.priors_,
                                                      np.bincount(train_y) / len(train_y))
                # Fitted identity routes reject a changed learned axis, identity
                # input audit or ensemble member state before checkpoint use.
                damaged = deepcopy(model)
                damaged_leaf = damaged.members_[0] if family == 'projection_soft_voting' else damaged
                if damaged_leaf.spec_['feature_engineering']['mode'] == 'identity':
                    damaged_leaf.scaler_.mean_[0] += 1.
                else:
                    damaged_leaf.projector_.components_[0, 0] += 1.
                with self.assertRaises(ValueError):
                    _verify_fitted_identity(damaged, pipeline, 53)
                probability = model.predict_proba(data['X'][~mask][:9])
                np.testing.assert_array_equal(probability,
                    np.concatenate([model.predict_proba(row[None]) for row in data['X'][~mask][:9]]))
                manifest, = job._write_pending_candidate_manifests()
                runtime_registry = ModelRegistry(source, state_root)
                descriptor = runtime_registry.validator.discover_manifest(manifest).descriptor
                validated = runtime_registry.validate_runtime(descriptor)
                self.assertEqual(validated.validation_status, 'passed', validated.lifecycle_reason)
                replay = SklearnAdapter(validated).predict_proba(read_sensor_csv(source / 'example/sample_input.csv'), 3)
                self.assertEqual(replay.shape[1], 7)
                self.assertTrue(np.isfinite(replay).all())
                np.testing.assert_allclose(replay.sum(1), 1, atol=1e-6)
                self.assertEqual(validated.approval_status, 'pending')

    def test_projection_identity_control_rejects_learned_axes_or_parameter_drift(self):
        self.identity_control_fixture('projection_extra_trees')
        registry.validate_checkpoint(self.entry)
        protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
        control = protocol['branches']['identity']['identity_a']
        self.assertEqual(control['feature_engineering'], {'mode': 'identity'})
        validate_identity_control(protocol['baseline'], control)
        mutations = {
            'PCA64': lambda ref, row: row['feature_engineering'].update(mode='pca64'),
            'LDA6': lambda ref, row: row['feature_engineering'].update(mode='all_plus_lda6'),
            'trees': lambda ref, row: row['model_params'].update(n_estimators=201),
            'seed': lambda ref, row: row['training'].update(seed=54),
            'policy': lambda ref, row: row['training'].update(data_policy='include_ordinary_reserved_training_records_v1'),
            'inferred parameter': lambda ref, row: ref['model_params'].pop('max_depth'),
            'reference weighting': lambda ref, row: ref['training'].update(weighting='subject_class_balanced'),
            'reference features': lambda ref, row: ref['feature_engineering'].update(mode='means'),
            'reference scaler': lambda ref, row: ref['preprocessing'].update(scaler='robust'),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = deepcopy(protocol)
                mutate(changed['baseline'], changed['branches']['identity']['identity_a'])
                with self.assertRaises(ValueError):
                    validate_identity_control(changed['baseline'], changed['branches']['identity']['identity_a'])
                with self.assertRaisesRegex(ValueError, '동일 경로 대조군'):
                    registry._validate_identity_controls(self.entry, self.root, changed)
        with self.assertRaises(ValueError):
            validate_identity_control(protocol['baseline'], self.projection_specs()[-1])

    def test_projection_supplement_uses_frozen_feature_rows_without_sequence_or_record_helpers(self):
        for pipeline in self.projection_specs():
            with self.subTest(mode=pipeline['feature_engineering']['mode']):
                pipeline['training']['data_policy'] = 'include_ordinary_reserved_training_records_v1'
                for member in pipeline.get('members', []):
                    member['training']['data_policy'] = pipeline['training']['data_policy']
                self.write_linked_fixture(pipeline)
                registry.validate_checkpoint(self.entry)
                for role in ('source_record_training', 'source_record_sequence_training', 'source_supplemental_sequence'):
                    self.assertNotIn(role, self.entry['files'])
                for role in ('source_supplemental', 'supplemental_manifest', 'supplemental_cache'):
                    broken = deepcopy(self.entry)
                    del broken['files'][role]
                    with self.assertRaises(ValueError):
                        registry.validate_checkpoint(broken)
                protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
                metrics = json.loads((self.root / self.entry['files']['final_metrics']['path']).read_text())
                for changed in ({'added_record_paths': ['a/ordinary.csv', 'c/ordinary.csv']},
                        {'added_windows': 6}, {'class_counts_after': [5, 3, *([2] * 5)]},
                        {'personal_reference_used': True}):
                    bad = deepcopy(metrics)
                    bad['supplemental_training_audit'].update(changed)
                    with self.assertRaises(ValueError):
                        registry._validate_supplement(self.entry, self.root, pipeline, protocol, bad)

    def test_projection_members_require_individual_gains_and_five_sources_are_pinned(self):
        pipeline = self.projection_specs()[-1]
        self.write_linked_fixture(pipeline)
        registry.validate_checkpoint(self.entry)
        expected = {'src/learned_projection_pipeline_v1.py', 'src/learned_projection_v1.py',
                    'src/posture_core.py', 'src/window_pipeline.py', 'src/__init__.py'}
        self.assertEqual(set(pipeline_sources(self.spec)), expected)
        for update in (lambda row: row['candidates'][0].update(qualified=False),
                       lambda row: row['selected']['versus_best_member'].update(qualified=False)):
            self.write_linked_fixture(pipeline)
            self.rewrite_evidence('selection', update)
            selection_hash = self.entry['files']['selection']['sha256']
            self.rewrite_evidence('evaluation', lambda row: row.update(selection_sha256=selection_hash))
            self.rewrite_evidence('reproducibility', lambda row: row.update(selection_sha256=selection_hash))
            with self.assertRaisesRegex(ValueError, '검증된 개선 후보만'):
                registry.validate_checkpoint(self.entry)
        for relative in sorted(expected):
            with self.subTest(source=relative):
                self.write_linked_fixture(pipeline)
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                helper = self.root / relative
                helper.write_bytes(helper.read_bytes() + b'\n# Changed pinned projection dependency.\n')
                with mock.patch('joblib.load', side_effect=AssertionError('No unpickling discovery')):
                    self.assertEqual(registry.available_checkpoints(), [])

    def hardness_specs(self):
        leaves = [validate_pipeline({'family': family,
            'model_params': {'n_estimators': 4, 'min_samples_leaf': 1},
            'training': {'seed': 53, 'hardness': {'coefficient': coefficient,
                'n_splits': 4, 'seed_offset': 1000}}})
            for family, coefficient in (('hardness_extra_trees', 0.0),
                ('hardness_extra_trees', .5), ('hardness_random_forest', 0.0),
                ('hardness_random_forest', .5))]
        ensemble = validate_pipeline({'family': 'hardness_soft_voting',
            'members': [leaves[1], leaves[3]], 'training': {'seed': 53, 'weighting': 'per_member'}})
        return *leaves, ensemble

    def test_hardness_checkpoints_use_only_selected_train_record_ids_then_pending_csv_replay(self):
        import pandas as pd
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from src import hardness_pipeline_v1
        from src.app_services import TrainingJob
        from src.model_catalog import MODEL_NAMES, pipeline_runtime_packages
        from src.model_comparison import _dataset
        from src.model_runtime import ModelRegistry
        from src.model_runtime.sklearn_adapter import SklearnAdapter
        from src.posture_core import SENSOR_COLUMNS, read_sensor_csv
        from experiments.iterative_improvement.finalize import _verify_fitted_identity
        source = Path(__file__).resolve().parents[1]
        rng = np.random.default_rng(18053)
        for subject in range(6):
            directory = self.root / 'data' / f'subject{subject}'
            directory.mkdir(parents=True)
            for label_id, label in enumerate(CLASS_NAMES):
                # Unequal four/five-window records exercise window-weighted
                # normalization; all four inner TRAIN folds retain seven classes.
                rows = 145 + 36 * (subject % 2)
                values = rng.uniform(10., 35., (rows, len(SENSOR_COLUMNS))) + label_id * 2 + subject * .5
                frame = pd.DataFrame(values, columns=SENSOR_COLUMNS)
                frame.insert(0, 'label', label)
                frame.insert(0, 'elapsed_s', np.arange(rows) / 12.)
                frame.to_csv(directory / f'posture{label_id}.csv', index=False)
        partitions = {'train': [f'subject{i}' for i in range(4)],
                      'validation': ['subject4'], 'final_evaluation': ['subject5']}
        data = _dataset(self.root / 'data', 3, 41, partitions)
        mask = data['masks']['train']
        train_X, train_y = data['X'][mask], data['y'][mask]
        groups = data['meta'].loc[mask, 'subject'].to_numpy()
        records = data['meta'].loc[mask, 'record_id'].to_numpy()
        expected_scaler = StandardScaler().fit(train_X)
        original_fit = hardness_pipeline_v1.fit_hardness_pipeline
        captures = []
        def capture(model, recipe, X, y, *, groups=None, record_ids=None, validation_data=None):
            captures.append((np.array(X), np.array(y), np.array(groups),
                             None if record_ids is None else np.array(record_ids)))
            return original_fit(model, recipe, X, y, groups=groups,
                                record_ids=record_ids, validation_data=validation_data)
        for index, pipeline in enumerate(self.hardness_specs()):
            with self.subTest(family=pipeline['family'], index=index):
                pipeline['training']['data_policy'] = 'include_ordinary_reserved_training_records_v1'
                for member in pipeline.get('members', []):
                    member['training']['data_policy'] = pipeline['training']['data_policy']
                self.write_linked_fixture(pipeline)
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                captures.clear()
                family = pipeline['family']
                self.assertTrue(is_hardness_recipe(pipeline))
                self.assertTrue(requires_record_ids(pipeline))
                self.assertEqual(input_representation(pipeline), 'feature_vector')
                self.assertEqual(combination_family(pipeline), 'hardness_soft_voting')
                self.assertEqual(len(MODEL_NAMES), 9)
                self.assertNotIn(family, MODEL_NAMES)
                self.assertEqual(pipeline_runtime_packages(pipeline), {})
                with self.assertRaises(ValueError):
                    TrainingWizardState().configure([3], 41, [family])
                state = TrainingWizardState(training_method=self.method)
                state.configure([3], 41, [family])
                restored = TrainingWizardState()
                apply_configuration(restored, build_configuration(state))
                self.assertEqual(restored.models, (family,))
                self.assertEqual(method_candidates(family, self.method), [pipeline['model_params']])
                state_root = self.root / f'hardness-{index}'
                job = TrainingJob(source, (), 41, models=(family,), state_root=state_root,
                                  training_method=self.method)
                with mock.patch.object(hardness_pipeline_v1, 'fit_hardness_pipeline', capture):
                    report = train_comparison(self.root / 'data', job.run_dir, [3], 41, [family],
                        training_method=self.method, progress=lambda _: None,
                        fixed_subject_partitions=partitions)
                self.assertEqual(report['failures'], [])
                self.assertEqual(len(captures), 1)
                observed_X, observed_y, observed_groups, observed_records = captures[0]
                np.testing.assert_array_equal(observed_X, train_X)
                np.testing.assert_array_equal(observed_y, train_y)
                np.testing.assert_array_equal(observed_groups, groups)
                np.testing.assert_array_equal(observed_records, records)
                self.assertFalse(set(observed_groups) & set(data['meta'].loc[~mask, 'subject']))
                self.assertFalse(set(observed_records) & set(data['meta'].loc[~mask, 'record_id']))
                # The research supplemental cache in this isolated fixture is
                # JSON, not a training NPZ. Exact current rows prove it was not appended.
                bundle = joblib.load(job.run_dir / 'artifacts' / report['models'][0]['artifact_file'])
                model = bundle['model']
                self.assertEqual(model.spec_, pipeline)
                self.assertEqual(model.effective_seed_, 53)
                self.assertEqual(bundle['input_representation'], 'feature_vector')
                self.assertEqual(bundle['feature_names'], feature_names())
                self.assertTrue(hardness_pipeline_v1.verify_fitted_hardness_identity(model, pipeline, 53))
                _verify_fitted_identity(model, pipeline, 53)
                leaves = model.members_ if family == 'hardness_soft_voting' else [model]
                for leaf in leaves:
                    coefficient = leaf.spec_['training']['hardness']['coefficient']
                    self.assertFalse(leaf.training_details_['external_validation_used'])
                    self.assertEqual(leaf.scaler_.n_samples_seen_, len(train_X))
                    np.testing.assert_array_equal(leaf.scaler_.mean_, expected_scaler.mean_)
                    np.testing.assert_array_equal(leaf.scaler_.scale_, expected_scaler.scale_)
                    expected_backend = (ExtraTreesClassifier if leaf.spec_['family'] == 'hardness_extra_trees'
                                        else RandomForestClassifier)
                    self.assertIs(type(leaf.estimator_), expected_backend)
                    self.assertFalse(leaf.estimator_.bootstrap)
                    self.assertEqual(leaf.estimator_.class_weight, 'balanced')
                    self.assertEqual(leaf.estimator_.random_state, 53)
                    self.assertEqual(leaf.estimator_.n_jobs, 1)
                    if coefficient == 0:
                        self.assertIsNone(leaf.sample_weight_)
                        self.assertIsNone(leaf.oof_prediction_)
                        if leaf.spec_['family'] == 'hardness_extra_trees':
                            reference = build_pipeline({'family': 'extra_trees',
                                'model_params': deepcopy(leaf.spec_['model_params']),
                                'training': {'seed': 53, 'weighting': 'class_balanced',
                                    'data_policy': pipeline['training']['data_policy']}})
                            reference.fit(train_X, train_y, groups=groups)
                            np.testing.assert_array_equal(leaf.predict_proba(data['X']),
                                                          reference.predict_proba(data['X']))
                    else:
                        oof = np.asarray(leaf.oof_prediction_)
                        folds = np.asarray(leaf.oof_fold_)
                        self.assertEqual(oof.shape, train_y.shape)
                        self.assertEqual(folds.shape, train_y.shape)
                        self.assertEqual(len(np.unique(folds)), 4)
                        np.testing.assert_array_equal(leaf.oof_probability_.argmax(1), oof)
                        np.testing.assert_allclose(leaf.oof_probability_.sum(1), 1., rtol=0, atol=1e-12)
                        for fold_index, audit in enumerate(leaf.hardness_audit_['folds']):
                            inner = np.asarray(audit['train_indices'])
                            held = np.asarray(audit['heldout_indices'])
                            self.assertFalse(set(groups[inner]) & set(groups[held]))
                            self.assertFalse(set(records[inner]) & set(records[held]))
                            np.testing.assert_array_equal(np.unique(train_y[inner]), np.arange(7))
                            np.testing.assert_array_equal(folds[held], np.full(len(held), fold_index))
                            self.assertEqual(audit['seed'], 53 + 1000 + fold_index)
                            teacher = audit['model_params']
                            self.assertEqual((teacher['n_estimators'], teacher['max_depth'],
                                teacher['min_samples_leaf'], teacher['max_features']), (200, 8, 8, .5))
                            self.assertFalse(teacher['bootstrap'])
                            self.assertEqual(teacher['class_weight'], 'balanced')
                            self.assertEqual(teacher['n_jobs'], 1)
                            inner_scaler = StandardScaler().fit(train_X[inner])
                            self.assertEqual(audit['scaler']['samples'], len(inner))
                            np.testing.assert_array_equal(audit['scaler']['mean_'], inner_scaler.mean_)
                            np.testing.assert_array_equal(audit['scaler']['scale_'], inner_scaler.scale_)
                        raw_weights = np.empty(len(train_y), dtype=np.float64)
                        for record in sorted(set(records)):
                            rows = records == record
                            self.assertEqual(len(np.unique(folds[rows])), 1)
                            self.assertEqual(len(np.unique(train_y[rows])), 1)
                            raw_weights[rows] = 1 + coefficient * np.mean(oof[rows] != train_y[rows])
                        for person in sorted(set(groups)):
                            self.assertEqual(len(np.unique(folds[groups == person])), 1)
                        expected_weights = raw_weights.copy()
                        for class_id in range(7):
                            rows = train_y == class_id
                            expected_weights[rows] /= raw_weights[rows].mean()
                            self.assertAlmostEqual(leaf.sample_weight_[rows].sum(), int(rows.sum()), places=10)
                            balanced = len(train_y) / (7 * rows.sum())
                            self.assertAlmostEqual((leaf.sample_weight_[rows] * balanced).sum(), len(train_y) / 7, places=10)
                        np.testing.assert_allclose(leaf.sample_weight_, expected_weights, rtol=1e-14, atol=0)
                damaged = deepcopy(model)
                damaged_leaf = damaged.members_[0] if family == 'hardness_soft_voting' else damaged
                if damaged_leaf.sample_weight_ is None:
                    damaged_leaf.estimator_.bootstrap = True
                else:
                    damaged_leaf.sample_weight_[0] += .1
                with self.assertRaises(ValueError):
                    _verify_fitted_identity(damaged, pipeline, 53)
                probability = model.predict_proba(data['X'][~mask][:7])
                np.testing.assert_array_equal(probability,
                    np.concatenate([model.predict_proba(row[None]) for row in data['X'][~mask][:7]]))
                manifest, = job._write_pending_candidate_manifests()
                runtime_registry = ModelRegistry(source, state_root)
                descriptor = runtime_registry.validator.discover_manifest(manifest).descriptor
                validated = runtime_registry.validate_runtime(descriptor)
                self.assertEqual(validated.validation_status, 'passed', validated.lifecycle_reason)
                replay = SklearnAdapter(validated).predict_proba(read_sensor_csv(source / 'example/sample_input.csv'), 3)
                self.assertEqual(replay.shape[1], 7)
                self.assertTrue(np.isfinite(replay).all())
                np.testing.assert_allclose(replay.sum(1), 1, atol=1e-6)
                self.assertEqual(validated.approval_status, 'pending')
                for missing in ('groups', 'record_ids'):
                    unfitted = build_method_model(family, pipeline['model_params'], 41, self.method)
                    options = {'groups': groups, 'record_ids': records}
                    options.pop(missing)
                    with self.assertRaises(ValueError):
                        fit_method_model(unfitted, family, self.method, train_X, train_y, **options)

    def test_hardness_identity_control_rejects_positive_weight_or_rf_backend(self):
        self.identity_control_fixture('hardness_extra_trees')
        registry.validate_checkpoint(self.entry)
        protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
        control = protocol['branches']['identity']['identity_a']
        self.assertEqual(control['training']['hardness']['coefficient'], 0.0)
        validate_identity_control(protocol['baseline'], control)
        mutations = {
            'positive coefficient': lambda ref, row: row['training']['hardness'].update(coefficient=.5),
            'optimized threshold RF': lambda ref, row: row.update(family='hardness_random_forest'),
            'trees': lambda ref, row: row['model_params'].update(n_estimators=201),
            'seed': lambda ref, row: row['training'].update(seed=54),
            'policy': lambda ref, row: row['training'].update(data_policy='include_ordinary_reserved_training_records_v1'),
            'missing reference parameter': lambda ref, row: ref['model_params'].pop('max_depth'),
            'reference weighting': lambda ref, row: ref['training'].update(weighting='subject_class_balanced'),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = deepcopy(protocol)
                mutate(changed['baseline'], changed['branches']['identity']['identity_a'])
                with self.assertRaises(ValueError):
                    validate_identity_control(changed['baseline'], changed['branches']['identity']['identity_a'])
                with self.assertRaisesRegex(ValueError, '동일 경로 대조군'):
                    registry._validate_identity_controls(self.entry, self.root, changed)
        with self.assertRaises(ValueError):
            validate_identity_control(protocol['baseline'], self.hardness_specs()[-1])

    def test_hardness_supplement_requires_record_identity_source_and_only_train_records(self):
        for pipeline in self.hardness_specs():
            with self.subTest(family=pipeline['family']):
                pipeline['training']['data_policy'] = 'include_ordinary_reserved_training_records_v1'
                for member in pipeline.get('members', []):
                    member['training']['data_policy'] = pipeline['training']['data_policy']
                self.write_linked_fixture(pipeline)
                registry.validate_checkpoint(self.entry)
                for role in ('source_supplemental_sequence', 'source_record_sequence_training'):
                    self.assertNotIn(role, self.entry['files'])
                for role in ('source_record_training', 'source_supplemental', 'supplemental_manifest', 'supplemental_cache'):
                    broken = deepcopy(self.entry)
                    del broken['files'][role]
                    with self.assertRaises(ValueError):
                        registry.validate_checkpoint(broken)
                protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
                metrics = json.loads((self.root / self.entry['files']['final_metrics']['path']).read_text())
                for changed in ({'added_record_paths': ['a/ordinary.csv', 'c/ordinary.csv']},
                        {'added_windows': 6}, {'class_counts_after': [5, 3, *([2] * 5)]},
                        {'personal_reference_used': True}):
                    bad = deepcopy(metrics)
                    bad['supplemental_training_audit'].update(changed)
                    with self.assertRaises(ValueError):
                        registry._validate_supplement(self.entry, self.root, pipeline, protocol, bad)

    def test_hardness_members_require_independent_gains_and_four_sources_are_pinned(self):
        pipeline = self.hardness_specs()[-1]
        self.write_linked_fixture(pipeline)
        registry.validate_checkpoint(self.entry)
        expected = {'src/hardness_pipeline_v1.py', 'src/posture_core.py',
                    'src/window_pipeline.py', 'src/__init__.py'}
        self.assertEqual(set(pipeline_sources(self.spec)), expected)
        for update in (lambda row: row['candidates'][0].update(qualified=False),
                       lambda row: row['selected']['versus_best_member'].update(qualified=False)):
            self.write_linked_fixture(pipeline)
            self.rewrite_evidence('selection', update)
            selection_hash = self.entry['files']['selection']['sha256']
            self.rewrite_evidence('evaluation', lambda row: row.update(selection_sha256=selection_hash))
            self.rewrite_evidence('reproducibility', lambda row: row.update(selection_sha256=selection_hash))
            with self.assertRaisesRegex(ValueError, '검증된 개선 후보만'):
                registry.validate_checkpoint(self.entry)
        for relative in sorted(expected):
            with self.subTest(source=relative):
                self.write_linked_fixture(pipeline)
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                dependency = self.root / relative
                dependency.write_bytes(dependency.read_bytes() + b'\n# Changed pinned hardness dependency.\n')
                with mock.patch('joblib.load', side_effect=AssertionError('No unpickling discovery')):
                    self.assertEqual(registry.available_checkpoints(), [])

    def structured_specs(self):
        leaves = [validate_pipeline({'family': family, 'feature_engineering': {'mode': mode},
            'model_params': {'n_estimators': 4, 'min_samples_leaf': 1}, 'training': {'seed': 53}})
            for family, mode in (('subject_residual_extra_trees', 'identity'),
                ('subject_residual_extra_trees', 'remove_subject3'), ('coupled_extra_trees', 'all'))]
        ensemble = validate_pipeline({'family': 'structured_soft_voting',
            'members': [leaves[1], leaves[2]], 'training': {'seed': 53}})
        return *leaves, ensemble

    def test_structured_checkpoints_fit_actual_train_records_and_replay_pending_runtime(self):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from src import structured_forest_pipeline_v1 as implementation
        from src.app_services import TrainingJob
        from src.model_catalog import MODEL_NAMES, pipeline_runtime_packages
        from src.model_comparison import _dataset
        from src.model_runtime import ModelRegistry
        from src.model_runtime.sklearn_adapter import SklearnAdapter
        from src.pairwise_coupling_v1 import PAIR_INDICES, couple_pairwise
        from src.posture_core import SENSOR_COLUMNS, read_sensor_csv
        from experiments.iterative_improvement.finalize import _verify_fitted_identity
        source = Path(__file__).resolve().parents[1]
        rng = np.random.default_rng(19053)
        for subject in range(6):
            directory = self.root / 'data' / f'subject{subject}'
            directory.mkdir(parents=True)
            for label_id, label in enumerate(CLASS_NAMES):
                rows = 109 + 36 * (subject % 2)
                values = rng.uniform(10., 35., (rows, len(SENSOR_COLUMNS))) + label_id * 2 + subject * .5
                frame = pd.DataFrame(values, columns=SENSOR_COLUMNS)
                frame.insert(0, 'label', label)
                frame.insert(0, 'elapsed_s', np.arange(rows) / 12.)
                frame.to_csv(directory / f'posture{label_id}.csv', index=False)
        partitions = {'train': [f'subject{i}' for i in range(4)],
                      'validation': ['subject4'], 'final_evaluation': ['subject5']}
        data = _dataset(self.root / 'data', 3, 41, partitions)
        mask = data['masks']['train']
        X, y = data['X'][mask], data['y'][mask]
        groups, records = [data['meta'].loc[mask, column].to_numpy() for column in ('subject', 'record_id')]
        expected_scaler = StandardScaler().fit(X)
        captures = []
        original = implementation.fit_structured_pipeline
        def capture(model, recipe, features, labels, *, groups=None, record_ids=None, validation_data=None):
            captures.append(tuple(np.array(value) for value in (features, labels, groups, record_ids)))
            return original(model, recipe, features, labels, groups=groups, record_ids=record_ids,
                            validation_data=validation_data)
        for index, pipeline in enumerate(self.structured_specs()):
            with self.subTest(recipe=pipeline['family'], index=index):
                pipeline['training']['data_policy'] = 'include_ordinary_reserved_training_records_v1'
                for member in pipeline.get('members', []):
                    member['training']['data_policy'] = pipeline['training']['data_policy']
                self.write_linked_fixture(pipeline)
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                family = pipeline['family']
                self.assertTrue(is_structured_recipe(pipeline))
                self.assertTrue(requires_record_ids(pipeline))
                self.assertEqual(input_representation(pipeline), 'feature_vector')
                self.assertEqual(combination_family(pipeline), 'structured_soft_voting')
                self.assertEqual(len(MODEL_NAMES), 9)
                self.assertNotIn(family, MODEL_NAMES)
                self.assertEqual(pipeline_runtime_packages(pipeline), {})
                with self.assertRaises(ValueError):
                    TrainingWizardState().configure([3], 41, [family])
                state = TrainingWizardState(training_method=self.method)
                state.configure([3], 41, [family])
                restored = TrainingWizardState()
                apply_configuration(restored, build_configuration(state))
                self.assertEqual(restored.models, (family,))
                self.assertEqual(method_candidates(family, self.method), [pipeline['model_params']])
                job = TrainingJob(source, (), 41, models=(family,), state_root=self.root / f'structured-{index}',
                                  training_method=self.method)
                captures.clear()
                with mock.patch.object(implementation, 'fit_structured_pipeline', capture):
                    report = train_comparison(self.root / 'data', job.run_dir, [3], 41, [family],
                        training_method=self.method, progress=lambda _: None, fixed_subject_partitions=partitions)
                self.assertEqual(report['failures'], [])
                self.assertEqual(len(captures), 1)
                for actual, expected in zip(captures[0], (X, y, groups, records)):
                    np.testing.assert_array_equal(actual, expected)
                self.assertFalse(set(captures[0][2]) & set(data['meta'].loc[~mask, 'subject']))
                self.assertFalse(set(captures[0][3]) & set(data['meta'].loc[~mask, 'record_id']))
                bundle = joblib.load(job.run_dir / 'artifacts' / report['models'][0]['artifact_file'])
                model = bundle['model']
                self.assertEqual(model.spec_, pipeline)
                self.assertEqual(model.effective_seed_, 53)
                self.assertEqual(bundle['input_representation'], 'feature_vector')
                self.assertEqual(bundle['feature_names'], feature_names())
                _verify_fitted_identity(model, pipeline, 53)
                leaves = model.members_ if family == 'structured_soft_voting' else [model]
                query = data['X'][~mask][:7]
                for leaf in leaves:
                    self.assertFalse(leaf.training_details_['external_validation_used'])
                    self.assertEqual(leaf.scaler_.n_samples_seen_, len(X))
                    np.testing.assert_array_equal(leaf.scaler_.mean_, expected_scaler.mean_)
                    np.testing.assert_array_equal(leaf.scaler_.scale_, expected_scaler.scale_)
                    self.assertEqual(sorted(row['record_id'] for row in leaf.record_metadata_), sorted(set(records)))
                    if leaf.spec_['family'] == 'coupled_extra_trees':
                        scores = np.column_stack([head.predict_proba(leaf.scaler_.transform(query).astype(np.float32))[:, 0]
                                                  for head in leaf.heads_])
                        np.testing.assert_array_equal(leaf.predict_proba(query), couple_pairwise(scores))
                        self.assertEqual(leaf.training_details_['total_trees'], 21 * 4)
                        self.assertEqual(leaf.training_details_['pair_training_row_exposures'], 6 * len(X))
                        for number, ((i, j), head, audit) in enumerate(zip(PAIR_INDICES, leaf.heads_, leaf.pair_audits_)):
                            take = np.flatnonzero((y == i) | (y == j))
                            self.assertEqual(audit['row_indices'], take.tolist())
                            self.assertEqual(head.random_state, 53 + number)
                            self.assertEqual(audit['binary_counts'], [int(sum(y == i)), int(sum(y == j))])
                            self.assertFalse(set(audit['subjects']) - set(groups))
                            self.assertFalse(set(audit['records']) - set(records))
                    elif leaf.spec_['feature_engineering']['mode'] == 'identity':
                        baseline = build_pipeline({'family': 'extra_trees', 'model_params': leaf.spec_['model_params'],
                                                   'training': {'seed': 53}}).fit(X, y, groups=groups)
                        self.assertEqual(leaf.predict_proba(query).tobytes(), baseline.predict_proba(query).tobytes())
                    else:
                        self.assertEqual(leaf.projector_.components_.shape, (3, 596))
                        self.assertEqual(leaf.projector_.training_records_sha256_, leaf.training_records_sha256_)
                probability = model.predict_proba(query)
                self.assertEqual(probability.tobytes(), np.concatenate([model.predict_proba(row[None]) for row in query]).tobytes())
                if family == 'structured_soft_voting':
                    np.testing.assert_array_equal(probability, np.mean([leaf.predict_proba(query) for leaf in leaves], axis=0))
                damaged = deepcopy(model)
                target = damaged.members_[0] if family == 'structured_soft_voting' else damaged
                target.scaler_.mean_[0] += 1
                with self.assertRaises(ValueError):
                    _verify_fitted_identity(damaged, pipeline, 53)
                manifest, = job._write_pending_candidate_manifests()
                runtime = ModelRegistry(source, job.state_root)
                descriptor = runtime.validator.discover_manifest(manifest).descriptor
                validated = runtime.validate_runtime(descriptor)
                self.assertEqual(validated.validation_status, 'passed', validated.lifecycle_reason)
                replay = SklearnAdapter(validated).predict_proba(read_sensor_csv(source / 'example/sample_input.csv'), 3)
                self.assertEqual(replay.shape[1], 7)
                self.assertTrue(np.isfinite(replay).all())
                np.testing.assert_allclose(replay.sum(1), 1, atol=1e-6)
                self.assertEqual(validated.approval_status, 'pending')
                for missing in ('groups', 'record_ids'):
                    unfitted = build_method_model(family, pipeline['model_params'], 41, self.method)
                    options = {'groups': groups, 'record_ids': records}
                    options.pop(missing)
                    with self.assertRaises(ValueError):
                        fit_method_model(unfitted, family, self.method, X, y, **options)

    def test_structured_identity_control_rejects_projection_coupling_or_parameter_drift(self):
        self.identity_control_fixture('subject_residual_extra_trees')
        registry.validate_checkpoint(self.entry)
        protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
        control = protocol['branches']['identity']['identity_a']
        validate_identity_control(protocol['baseline'], control)
        mutations = {
            'projection': lambda ref, row: row['feature_engineering'].update(mode='remove_subject3'),
            'coupled head': lambda ref, row: row.update(family='coupled_extra_trees', feature_engineering={'mode': 'all'},
                postprocessing={'kind': 'pairwise_wu_kkt'}, training={'seed': 53, 'weighting': 'pair_class_balanced'}),
            'tree count': lambda ref, row: row['model_params'].update(n_estimators=201),
            'seed': lambda ref, row: row['training'].update(seed=54),
            'policy': lambda ref, row: row['training'].update(data_policy='include_ordinary_reserved_training_records_v1'),
            'missing reference parameter': lambda ref, row: ref['model_params'].pop('max_depth'),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = deepcopy(protocol)
                mutate(changed['baseline'], changed['branches']['identity']['identity_a'])
                with self.assertRaises(ValueError):
                    validate_identity_control(changed['baseline'], changed['branches']['identity']['identity_a'])
                with self.assertRaisesRegex(ValueError, '동일 경로 대조군'):
                    registry._validate_identity_controls(self.entry, self.root, changed)
        with self.assertRaises(ValueError):
            validate_identity_control(protocol['baseline'], self.structured_specs()[-1])

    def test_structured_supplement_requires_frozen_record_helper_and_train_only_audit(self):
        for pipeline in self.structured_specs():
            with self.subTest(family=pipeline['family']):
                pipeline['training']['data_policy'] = 'include_ordinary_reserved_training_records_v1'
                for member in pipeline.get('members', []):
                    member['training']['data_policy'] = pipeline['training']['data_policy']
                self.write_linked_fixture(pipeline)
                registry.validate_checkpoint(self.entry)
                self.assertNotIn('source_supplemental_sequence', self.entry['files'])
                for role in ('source_record_training', 'source_supplemental', 'supplemental_manifest', 'supplemental_cache'):
                    broken = deepcopy(self.entry)
                    del broken['files'][role]
                    with self.assertRaises(ValueError):
                        registry.validate_checkpoint(broken)
                protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
                metrics = json.loads((self.root / self.entry['files']['final_metrics']['path']).read_text())
                metrics['supplemental_training_audit']['added_record_paths'] = ['a/ordinary.csv', 'c/ordinary.csv']
                with self.assertRaises(ValueError):
                    registry._validate_supplement(self.entry, self.root, pipeline, protocol, metrics)

    def test_structured_members_require_gains_and_six_algorithm_sources_are_pinned(self):
        pipeline = self.structured_specs()[-1]
        self.write_linked_fixture(pipeline)
        registry.validate_checkpoint(self.entry)
        expected = {'src/structured_forest_pipeline_v1.py', 'src/pairwise_coupling_v1.py', 'src/subject_residual_v1.py',
                    'src/posture_core.py', 'src/window_pipeline.py', 'src/__init__.py'}
        self.assertEqual(set(pipeline_sources(self.spec)), expected)
        for update in (lambda row: row['candidates'][0].update(qualified=False),
                       lambda row: row['selected']['versus_best_member'].update(qualified=False)):
            self.write_linked_fixture(pipeline)
            self.rewrite_evidence('selection', update)
            selection_hash = self.entry['files']['selection']['sha256']
            self.rewrite_evidence('evaluation', lambda row: row.update(selection_sha256=selection_hash))
            self.rewrite_evidence('reproducibility', lambda row: row.update(selection_sha256=selection_hash))
            with self.assertRaisesRegex(ValueError, '검증된 개선 후보만'):
                registry.validate_checkpoint(self.entry)
        for relative in sorted(expected):
            with self.subTest(source=relative):
                self.write_linked_fixture(pipeline)
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                dependency = self.root / relative
                dependency.write_bytes(dependency.read_bytes() + b'\n# Changed pinned structured dependency.\n')
                with mock.patch('joblib.load', side_effect=AssertionError('No unpickling discovery')):
                    self.assertEqual(registry.available_checkpoints(), [])

    def _sequence_checkpoint_case(self, family):
        from src.app_services import TrainingJob
        from src.model_catalog import MODEL_NAMES, pipeline_runtime_packages
        from src.model_runtime import ModelRegistry
        from src.model_runtime.sklearn_adapter import SklearnAdapter
        from src.pipeline_recipes import input_representation
        from src.posture_core import read_sensor_csv
        from tests.test_model_comparison import ModelComparisonTests
        source = Path(__file__).resolve().parents[1]
        cnn = {'family': 'cnn1d', 'model_params': {'hidden_size': 8, 'epochs': 1, 'batch_size': 7, 'patience': 1},
               'training': {'seed': 53}}
        if family == 'sequence_soft_voting':
            other = deepcopy(cnn)
            other['model_params']['hidden_size'] = 12
            spec = {'family': family, 'members': [cnn, other], 'training': {'seed': 53}}
        else:
            spec = cnn if family == 'cnn1d' else {'family': family,
                'model_params': {'num_kernels': 84, 'C': .1}, 'training': {'seed': 53}}
        self.write_linked_fixture(validate_pipeline(spec))
        registry.register_checkpoint(self.entry)
        self.assertEqual(input_representation(self.spec), 'sensor_sequence')
        self.assertEqual(len(MODEL_NAMES), 9)
        if family == 'sequence_soft_voting':
            self.assertNotIn(family, MODEL_NAMES)
            with self.assertRaises(ValueError):
                TrainingWizardState().configure([3], 41, [family])
        state = TrainingWizardState(training_method=self.method)
        state.configure([3], 41, [family])
        restored = TrainingWizardState()
        apply_configuration(restored, build_configuration(state))
        self.assertEqual(restored.models, (family,))
        self.assertEqual(method_candidates(family, self.method), [self.spec['model_params']])
        ModelComparisonTests().make_data(self.root / 'data')
        job = TrainingJob(source, (), 41, models=(family,), state_root=self.root, training_method=self.method)
        report = train_comparison(self.root / 'data', job.run_dir, [3], 41, [family],
                                  training_method=self.method, progress=lambda _: None)
        self.assertEqual(report['failures'], [])
        self.assertEqual(len(report['candidates']), 1)
        bundle = joblib.load(job.run_dir / 'artifacts' / report['models'][0]['artifact_file'])
        self.assertEqual(bundle['input_representation'], 'sensor_sequence')
        self.assertEqual(bundle['feature_names'], [])
        self.assertEqual(bundle['runtime_packages'], pipeline_runtime_packages(self.spec))
        self.assertEqual(bundle['model'].effective_seed_, 53)
        self.assertEqual(bundle['model'].input_shape_, (36, 16))
        manifest, = job._write_pending_candidate_manifests()
        runtime_registry = ModelRegistry(source, self.root)
        descriptor = runtime_registry.validator.discover_manifest(manifest).descriptor
        validated = runtime_registry.validate_runtime(descriptor)
        self.assertEqual(validated.validation_status, 'passed', validated.lifecycle_reason)
        replay = SklearnAdapter(validated).predict_proba(read_sensor_csv(source / 'example/sample_input.csv'), 3)
        self.assertEqual(replay.shape[1], 7)
        self.assertTrue(np.isfinite(replay).all())
        np.testing.assert_allclose(replay.sum(1), 1, atol=1e-6)
        self.assertEqual(validated.approval_status, 'pending')
        for relative in pipeline_sources(self.spec):
            self.assertIn(self.root / relative, registry.checkpoint_resource_files(self.root))
        tracked = self.root / 'src/sequence_pipeline_v1.py'
        tracked.write_bytes(tracked.read_bytes() + b'\n# modified sequence implementation\n')
        self.assertEqual(registry.available_checkpoints(), [])

    def test_sequence_checkpoints_train_and_replay_in_isolated_native_processes(self):
        from importlib.util import find_spec
        if find_spec('torch') is None or find_spec('numba') is None:
            self.skipTest('Sequence runtime integration needs torch and numba')
        source = Path(__file__).resolve().parents[1]
        for family in ('cnn1d', 'minirocket', 'sequence_soft_voting'):
            with self.subTest(family=family):
                code = ('from tests.test_checkpoint_registry import CheckpointRegistryTests\n'
                    'case = CheckpointRegistryTests()\ncase.setUp()\n'
                    f'try:\n    case._sequence_checkpoint_case({family!r})\n'
                    'finally:\n    case.doCleanups()\n')
                environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                    'MKL_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1',
                    'NUMBA_CACHE_DIR': str(self.root / ('numba-' + family)), 'PYTHONPATH': str(source)}
                completed = subprocess.run([sys.executable, '-c', code], cwd=source, env=environment,
                    capture_output=True, text=True, timeout=120)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_sequence_ensemble_source_and_package_union_require_only_qualified_members(self):
        from src.model_catalog import pipeline_runtime_packages
        spec = {'family': 'sequence_soft_voting', 'training': {'seed': 53}, 'members': [
            {'family': 'cnn1d', 'training': {'seed': 53}},
            {'family': 'minirocket', 'training': {'seed': 53}}]}
        self.write_linked_fixture(spec)
        registry.validate_checkpoint(self.entry)
        self.assertEqual(pipeline_runtime_packages(self.spec), {'torch': '>=2.1', 'numba': '>=0.63'})
        self.assertIn('src/_vendor/minirocket_multi.py', pipeline_sources(self.spec))
        self.assertIn('src/_vendor/LICENSE.sktime', pipeline_sources(self.spec))
        self.assertNotIn('src/iterative_pipeline.py', pipeline_sources(self.spec))
        self.rewrite_evidence('selection', lambda row: row['candidates'][1].update(qualified=False))
        # First restore all hash links to show that the rejected member, rather
        # than an incidental stale selection hash, makes the combination invalid.
        digest = self.entry['files']['selection']['sha256']
        self.rewrite_evidence('evaluation', lambda row: row.update(selection_sha256=digest))
        self.rewrite_evidence('reproducibility', lambda row: row.update(selection_sha256=digest))
        with self.assertRaisesRegex(ValueError, '검증된 개선 후보만'):
            registry.validate_checkpoint(self.entry)

    def test_supplemental_policy_requires_bound_manifest_cache_and_source(self):
        policy = 'include_ordinary_reserved_training_records_v1'
        spec = deepcopy(self.spec)
        spec['training']['data_policy'] = policy
        for missing in ('supplemental_manifest', 'supplemental_cache', 'source_supplemental'):
            self.write_linked_fixture(spec)
            del self.entry['files'][missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                registry.validate_checkpoint(self.entry)
        self.write_linked_fixture(spec)
        self.assertEqual(registry.validate_checkpoint(self.entry)['pipeline']['training']['data_policy'], policy)
        for role in ('supplemental_cache', 'source_supplemental', 'supplemental_manifest'):
            self.write_linked_fixture(spec)
            self.rewrite_evidence(role, lambda value: value.update(unrelated_source=True))
            with self.subTest(unrelated=role), self.assertRaises(ValueError):
                registry.validate_checkpoint(self.entry)

    def test_supplemental_audit_is_derived_from_only_final_training_subjects(self):
        from src.checkpoint_registry import _validate_supplement
        spec = deepcopy(self.spec)
        spec['training']['data_policy'] = 'include_ordinary_reserved_training_records_v1'
        self.write_linked_fixture(spec)
        protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
        original = json.loads((self.root / self.entry['files']['final_metrics']['path']).read_text())
        changes = {
            'evaluation record appended': {'added_record_paths': ['a/ordinary.csv', 'b/ordinary.csv', 'c/ordinary.csv']},
            'wrong training subjects': {'training_subjects': ['a', 'c']},
            'wrong record count': {'added_records': 3},
            'wrong window count': {'added_windows': 6},
            'wrong per-record counts': {'added_windows_by_record': {'a/ordinary.csv': 1, 'b/ordinary.csv': 3}},
            'wrong class augmented': {'class_counts_after': [2, 6, 2, 2, 2, 2, 2]},
            'wrong base count': {'base_training_windows': 15},
        }
        for name, update in changes.items():
            metrics = deepcopy(original)
            metrics['supplemental_training_audit'].update(update)
            with self.subTest(name=name), self.assertRaises(ValueError):
                _validate_supplement(self.entry, self.root, spec, protocol, metrics)
        wrong_origin = deepcopy(protocol)
        wrong_origin['dataset_sha256'] = 'a' * 64
        with self.assertRaises(ValueError):
            _validate_supplement(self.entry, self.root, spec, wrong_origin, original)

    def test_sequence_supplement_requires_frozen_source_and_aligned_manifest(self):
        from src.checkpoint_registry import _validate_supplement
        pipeline = validate_pipeline({'family': 'cnn1d', 'training': {'seed': 53,
            'data_policy': 'include_ordinary_reserved_training_records_v1'}})
        self.write_linked_fixture(pipeline)
        with mock.patch('numpy.load', side_effect=AssertionError('Discovery must not load array caches')):
            registry.validate_checkpoint(self.entry)
        missing = deepcopy(self.entry)
        del missing['files']['source_supplemental_sequence']
        with self.assertRaisesRegex(ValueError, '시계열 복원 실행 소스'):
            registry.validate_checkpoint(missing)
        reference = self.entry['files']['supplemental_manifest']
        path = self.root / reference['path']
        original = json.loads(path.read_text())
        protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
        metrics = json.loads((self.root / self.entry['files']['final_metrics']['path']).read_text())
        changes = {
            'wrong sequence source': lambda m: m['supplemental_source_hashes'].update(
                {'experiments/iterative_improvement/supplemental_sequence.py': 'b' * 64}),
            'missing sequence evidence': lambda m: m.pop('sequence_evidence'),
            'wrong shape': lambda m: m['sequence_evidence'].update(shape=[6, 16, 36]),
            'wrong sensor order': lambda m: m['sequence_evidence']['sensor_order'].reverse(),
            'failed parity': lambda m: m['sequence_evidence']['feature_parity'].update(bitwise_equal=False),
            'wrong checked rows': lambda m: m['sequence_evidence']['feature_parity'].update(checked_windows=5),
            'bad array hash': lambda m: m['sequence_evidence'].update(sequence_array_sha256='missing'),
            'wrong starts': lambda m: m['records'][0].update(window_start_s=[0.]),
            'reordered starts': lambda m: m['records'][0].update(window_start_s=[3., 0.]),
        }
        for name, mutate in changes.items():
            with self.subTest(case=name):
                manifest = deepcopy(original)
                mutate(manifest)
                path.write_text(json.dumps(manifest))
                reference['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                linked_protocol = deepcopy(protocol)
                linked_protocol['supplemental_training']['manifest_sha256'] = reference['sha256']
                with self.assertRaises(ValueError):
                    _validate_supplement(self.entry, self.root, pipeline, linked_protocol, metrics)

    def ensemble_spec(self, second_family='extra_trees'):
        other = validate_pipeline({'family': second_family,
            'model_params': {'n_estimators': 8, 'max_depth': 3},
            'training': {'seed': 53}})
        return validate_pipeline({'family': 'equal_soft_voting', 'members': [self.spec, other],
                                  'training': {'seed': 53}})

    def test_combination_is_checkpoint_only_and_reuses_full_runtime_pipeline(self):
        from src.app_services import TrainingJob
        from src.inference_core import predict_probabilities
        from src.model_catalog import MODEL_NAMES, model_display_name
        from src.model_runtime import ModelRegistry
        from src.model_runtime.sklearn_adapter import SklearnAdapter
        from src.posture_core import read_sensor_csv
        from tests.test_model_comparison import ModelComparisonTests
        source = Path(__file__).resolve().parents[1]
        self.write_linked_fixture(self.ensemble_spec())
        registry.register_checkpoint(self.entry)
        self.assertEqual(len(MODEL_NAMES), 9)
        standard = next(row for row in available_training_methods() if row['id'] == 'standard')
        self.assertNotIn('equal_soft_voting', standard['families'])
        with self.assertRaises(ValueError):
            validate_training_method('standard', ['equal_soft_voting'], [3])
        state = TrainingWizardState(models=('equal_soft_voting',), training_method=self.method)
        config = build_configuration(state)
        restored = TrainingWizardState()
        apply_configuration(restored, config)
        self.assertEqual(restored.models, ('equal_soft_voting',))
        ModelComparisonTests().make_data(self.root / 'data')
        job = TrainingJob(source, (), 41, models=('equal_soft_voting',), state_root=self.root, training_method=self.method)
        report = train_comparison(self.root / 'data', job.run_dir, [3], 41, ['equal_soft_voting'],
                                  training_method=self.method, progress=lambda _: None)
        self.assertEqual(len(report['candidates']), 1)
        bundle = joblib.load(job.run_dir / 'artifacts' / report['models'][0]['artifact_file'])
        self.assertEqual(bundle['selected_model'], model_display_name('equal_soft_voting'))
        self.assertEqual(len(bundle['model'].members_), 2)
        X = np.zeros((1, len(feature_names())))
        np.testing.assert_allclose(predict_probabilities(bundle, X), bundle['model'].predict_proba(X))
        manifest, = job._write_pending_candidate_manifests()
        model_registry = ModelRegistry(source, self.root)
        descriptor = model_registry.validator.discover_manifest(manifest).descriptor
        validated = model_registry.validate_runtime(descriptor)
        self.assertEqual(validated.validation_status, 'passed', validated.lifecycle_reason)
        replay = SklearnAdapter(validated).predict_proba(read_sensor_csv(source / 'example/sample_input.csv'), 3)
        self.assertEqual(replay.shape[1], 7)
        self.assertEqual(validated.approval_status, 'pending')

    def test_combination_member_dependencies_are_required_before_training_and_packaging(self):
        import build_macos_app
        from src.app_services import TrainingJob
        from src.model_catalog import pipeline_runtime_packages
        self.write_linked_fixture(self.ensemble_spec('xgboost'))
        registry.register_checkpoint(self.entry)
        self.assertEqual(pipeline_runtime_packages(self.spec), {'xgboost': '>=2.0'})
        job = TrainingJob(self.root, (), 41, models=('equal_soft_voting',), state_root=self.root, training_method=self.method)
        with mock.patch('src.model_catalog.find_spec', return_value=None), self.assertRaisesRegex(ValueError, 'xgboost'):
            job.start()
        self.assertFalse(job.run_dir.exists())
        with mock.patch.object(build_macos_app, 'ROOT', self.root):
            with self.assertRaisesRegex(ValueError, 'include-extended-models'):
                build_macos_app.build_command()
            command = build_macos_app.build_command(include_extended_models=True)
        self.assertIn('xgboost', command)

    def test_hierarchy_checkpoint_trains_and_replays_without_entering_standard_choices(self):
        from experiments.iterative_improvement.finalize import verified_model_probabilities
        from src.app_services import TrainingJob
        from src.model_catalog import MODEL_NAMES, model_display_name, pipeline_runtime_packages
        from src.model_runtime import ModelRegistry
        from src.model_runtime.sklearn_adapter import SklearnAdapter
        from src.posture_core import read_sensor_csv
        from tests.test_model_comparison import ModelComparisonTests
        source = Path(__file__).resolve().parents[1]
        family = 'sagittal_hierarchy'
        self.assertEqual(len(MODEL_NAMES), 9)
        self.assertNotIn(family, TrainingWizardState.SUPPORTED_MODELS)
        with self.assertRaises(ValueError):
            TrainingWizardState().configure([3], 41, [family])
        ModelComparisonTests().make_data(self.root / 'data')
        for base_family, base_params in (
            ('extra_trees', {'n_estimators': 8, 'max_depth': 3, 'min_samples_leaf': 1}),
            ('rbf_svm', {'C': 1.0, 'gamma': .0005}),
        ):
            with self.subTest(base=base_family):
                pipeline = validate_pipeline({'family': family,
                    'model_params': {'base_family': base_family, 'base_params': base_params},
                    'training': {'seed': 53}})
                self.write_linked_fixture(pipeline)
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                state = TrainingWizardState(training_method=self.method)
                state.configure([3], 41, [family])
                configuration = build_configuration(state)
                restored = TrainingWizardState()
                apply_configuration(restored, configuration)
                self.assertEqual(restored.models, (family,))
                self.assertEqual(method_candidates(family, self.method), [pipeline['model_params']])
                self.assertEqual(pipeline_runtime_packages(pipeline), {})
                job = TrainingJob(source, (), 41, models=(family,), state_root=self.root / base_family,
                                  training_method=self.method)
                report = train_comparison(self.root / 'data', job.run_dir, [3], 41, [family],
                    training_method=self.method, progress=lambda _: None)
                self.assertEqual(report['failures'], [])
                self.assertEqual(len(report['candidates']), 1)
                bundle = joblib.load(job.run_dir / 'artifacts' / report['models'][0]['artifact_file'])
                self.assertEqual(bundle['selected_model'], model_display_name(family))
                self.assertEqual(bundle['model'].effective_seed_, 53)
                values = verified_model_probabilities(bundle['model'], pipeline, 53, np.zeros((2, 596)))
                self.assertEqual(values.shape, (2, 7))
                manifest, = job._write_pending_candidate_manifests()
                runtime_registry = ModelRegistry(source, self.root / base_family)
                descriptor = runtime_registry.validator.discover_manifest(manifest).descriptor
                validated = runtime_registry.validate_runtime(descriptor)
                self.assertEqual(validated.validation_status, 'passed', validated.lifecycle_reason)
                replay = SklearnAdapter(validated).predict_proba(read_sensor_csv(source / 'example/sample_input.csv'), 3)
                self.assertEqual(replay.shape[1], 7)
                self.assertEqual(validated.approval_status, 'pending')
                hierarchy = bundle['model'].named_steps['model']
                hierarchy.specialist_.random_state = 55
                with self.assertRaisesRegex(ValueError, 'hierarchy'):
                    verified_model_probabilities(bundle['model'], pipeline, 53, np.zeros((2, 596)))

    def record_specs(self):
        params = {'n_estimators': 10, 'max_depth': 3, 'min_samples_leaf': 1, 'max_features': .5}
        balanced = validate_pipeline({'family': 'record_balanced_extra_trees', 'model_params': params,
            'training': {'seed': 53, 'weighting': 'class_record_balanced'}})
        bagged = validate_pipeline({'family': 'record_bagged_extra_trees',
            'model_params': {**params, 'bag_count': 2}, 'training': {'seed': 53, 'weighting': 'class_balanced'}})
        ensemble = validate_pipeline({'family': 'record_soft_voting', 'members': [balanced, bagged],
            'training': {'seed': 53, 'weighting': 'class_balanced'}})
        return balanced, bagged, ensemble

    def test_record_checkpoint_model_add_routes_actual_training_record_ids_and_replays(self):
        from src import record_pipeline_v1
        from src.app_services import TrainingJob
        from src.model_catalog import MODEL_NAMES, pipeline_runtime_packages
        from src.model_comparison import _dataset
        from src.model_runtime import ModelRegistry
        from src.model_runtime.sklearn_adapter import SklearnAdapter
        from src.posture_core import read_sensor_csv
        from tests.test_model_comparison import ModelComparisonTests
        source = Path(__file__).resolve().parents[1]
        ModelComparisonTests().make_data(self.root / 'data', rows=73)
        partitions = {'train': ['subject0', 'subject1'], 'validation': ['subject2'], 'final_evaluation': ['subject3']}
        data = _dataset(self.root / 'data', 3, 41, partitions)
        mask = data['masks']['train']
        record_ids = data['meta'].loc[mask, 'record_id'].to_numpy()
        groups = data['meta'].loc[mask, 'subject'].to_numpy()
        self.assertLess(len(set(record_ids)), len(record_ids))
        original_fit = record_pipeline_v1.fit_record_pipeline
        captures = []
        def capture(model, recipe, X, y, *, groups=None, record_ids=None, validation_data=None):
            captures.append((np.array(X), np.array(y), np.array(groups), np.array(record_ids)))
            return original_fit(model, recipe, X, y, groups=groups, record_ids=record_ids, validation_data=validation_data)
        for pipeline in self.record_specs():
            with self.subTest(family=pipeline['family']):
                captures.clear()
                self.write_linked_fixture(pipeline)
                self.path.write_text(json.dumps({'schema_version': 1, 'checkpoints': []}))
                registry.register_checkpoint(self.entry)
                family = pipeline['family']
                self.assertEqual(len(MODEL_NAMES), 9)
                self.assertNotIn(family, MODEL_NAMES)
                self.assertTrue(requires_record_ids(pipeline))
                self.assertEqual(combination_family(pipeline), 'record_soft_voting')
                self.assertEqual(input_representation(pipeline), 'feature_vector')
                self.assertEqual(pipeline_runtime_packages(pipeline), {})
                with self.assertRaises(ValueError):
                    TrainingWizardState().configure([3], 41, [family])
                state = TrainingWizardState(training_method=self.method)
                state.configure([3], 41, [family])
                restored = TrainingWizardState()
                apply_configuration(restored, build_configuration(state))
                self.assertEqual(restored.models, (family,))
                self.assertEqual(method_candidates(family, self.method), [pipeline['model_params']])
                job = TrainingJob(source, (), 41, models=(family,), state_root=self.root / family,
                                  training_method=self.method)
                with mock.patch.object(record_pipeline_v1, 'fit_record_pipeline', capture):
                    report = train_comparison(self.root / 'data', job.run_dir, [3], 41, [family],
                        training_method=self.method, progress=lambda _: None, fixed_subject_partitions=partitions)
                self.assertEqual(report['failures'], [])
                self.assertEqual(len(captures), 1)
                observed_X, observed_y, observed_groups, observed_records = captures[0]
                np.testing.assert_array_equal(observed_X, data['X'][mask])
                np.testing.assert_array_equal(observed_y, data['y'][mask])
                np.testing.assert_array_equal(observed_groups, groups)
                np.testing.assert_array_equal(observed_records, record_ids)
                held_records = set(data['meta'].loc[~mask, 'record_id'])
                self.assertFalse(set(observed_records) & held_records)
                bundle = joblib.load(job.run_dir / 'artifacts' / report['models'][0]['artifact_file'])
                self.assertEqual(bundle['model'].effective_seed_, 53)
                self.assertEqual(bundle['input_representation'], 'feature_vector')
                self.assertEqual(len(bundle['feature_names']), 596)
                record_pipeline_v1.verify_fitted_record_identity(bundle['model'], pipeline, 53)
                manifest, = job._write_pending_candidate_manifests()
                runtime_registry = ModelRegistry(source, self.root / family)
                descriptor = runtime_registry.validator.discover_manifest(manifest).descriptor
                validated = runtime_registry.validate_runtime(descriptor)
                self.assertEqual(validated.validation_status, 'passed', validated.lifecycle_reason)
                replay = SklearnAdapter(validated).predict_proba(read_sensor_csv(source / 'example/sample_input.csv'), 3)
                self.assertEqual(replay.shape[1], 7)
                self.assertEqual(validated.approval_status, 'pending')
                # Missing record identities must never be replaced by window IDs.
                unfitted = build_method_model(family, pipeline['model_params'], 41, self.method)
                with self.assertRaises(ValueError):
                    fit_method_model(unfitted, family, self.method, data['X'][mask], data['y'][mask], groups=groups)

    def test_record_restoration_requires_its_record_identity_source(self):
        from src.checkpoint_registry import _validate_supplement
        pipeline = self.record_specs()[0]
        pipeline['training']['data_policy'] = 'include_ordinary_reserved_training_records_v1'
        self.write_linked_fixture(pipeline)
        registry.validate_checkpoint(self.entry)
        missing = deepcopy(self.entry)
        del missing['files']['source_record_training']
        with self.assertRaisesRegex(ValueError, '기록 ID'):
            registry.validate_checkpoint(missing)
        protocol = json.loads((self.root / self.entry['files']['protocol']['path']).read_text())
        metrics = json.loads((self.root / self.entry['files']['final_metrics']['path']).read_text())
        protocol['source_hashes']['experiments/iterative_improvement/record_training.py'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, '기록 ID'):
            _validate_supplement(self.entry, self.root, pipeline, protocol, metrics)

    def test_record_combinations_require_individually_qualified_members(self):
        self.write_linked_fixture(self.record_specs()[-1])
        registry.validate_checkpoint(self.entry)
        self.rewrite_evidence('selection', lambda row: row['candidates'][0].update(qualified=False))
        selection_hash = self.entry['files']['selection']['sha256']
        self.rewrite_evidence('evaluation', lambda row: row.update(selection_sha256=selection_hash))
        self.rewrite_evidence('reproducibility', lambda row: row.update(selection_sha256=selection_hash))
        with self.assertRaisesRegex(ValueError, '검증된 개선 후보만'):
            registry.validate_checkpoint(self.entry)


if __name__ == "__main__":
    unittest.main()
