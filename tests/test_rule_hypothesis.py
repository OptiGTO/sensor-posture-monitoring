from pathlib import Path
import json
from dataclasses import asdict
from unittest import mock
import tempfile
import unittest

import numpy as np

from validate_personalization import ReplayRecord
from validate_rule_hypothesis import (Case, choose_rule, metrics, ratios, registration_groups,
                                      rule_scores, temporal_predictions, frozen_records)
from src.posture_core import CLASS_NAMES


class RuleHypothesisTests(unittest.TestCase):
    def record(self, name, correct=True, sensor=None):
        return ReplayRecord(Path('/tmp') / name, name, 'S', CLASS_NAMES[0 if correct else 1],
                            240, 20., name, sensor or name)

    def test_multi_registration_excludes_identical_whole_records_for_all_counts(self):
        records = [self.record(f'correct{i}') for i in range(4)] + [self.record('incorrect', False)]
        evaluation_sets = []
        for count in (1, 2, 3):
            registration, excluded, evaluation = registration_groups(records, count, 3)
            self.assertEqual(len(registration), count)
            self.assertEqual(len(excluded), 3)
            evaluation_sets.append({r.relative_path for r in evaluation})
        self.assertEqual(evaluation_sets, [{'correct3', 'incorrect'}] * 3)

    def test_registration_cannot_reuse_correct_evaluation_or_hidden_duplicate(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            registration_groups([self.record('only'), self.record('wrong', False)])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            registration_groups([self.record('c0', sensor='same'), self.record('c1', sensor='same')])

    def test_pressure_ratios_ignore_common_gain_and_reject_empty_frames(self):
        values = np.arange(1, 17, dtype=float)
        np.testing.assert_allclose(ratios(values), ratios(values * 3))
        with self.assertRaisesRegex(ValueError, 'Zero'):
            ratios(np.zeros(16))

    def test_spatial_rule_recognizes_matching_reference_and_balanced_shift(self):
        center = np.full(16, 1 / 16)
        moved = center.reshape(4, 4).copy()
        moved[:, :2] += .02
        moved[:, 2:] -= .02
        score = rule_scores(np.stack([center, moved.ravel()]), center, np.full(16, .008))
        for rule in score:
            self.assertEqual(score[rule][0], 0.)
            self.assertGreater(score[rule][1], 0.)

    def test_always_incorrect_cannot_score_above_chance_balanced_accuracy(self):
        truth = np.array([0] + [1] * 99)
        result = metrics(truth, np.ones(100, dtype=int))
        self.assertEqual(result['accuracy'], .99)
        self.assertEqual(result['balanced_accuracy'], .5)
        self.assertEqual(result['false_positive_rate'], 1.)

    def test_uncertainty_is_not_dropped_or_counted_as_success(self):
        result = metrics(np.array([0, 0, 1, 1]), np.array([0, -1, 1, -1]))
        self.assertEqual(result['balanced_accuracy'], .5)
        self.assertEqual(result['uncertain_rate'], .5)
        self.assertEqual(result['false_positive_rate'], 0.)
        self.assertEqual(result['missed_as_correct_rate'], 0.)

    def test_held_subject_outcomes_do_not_change_selected_rule(self):
        def case(subject):
            return Case(subject, np.array([0, 1]), np.array([0, 1]), np.array(['a', 'b']), [],
                        {'maximum': np.array([.1, 2.])}, np.array([0, 1]))
        cases = [case('A'), case('B'), case('held')]
        before = choose_rule(cases, {'held'}, ('maximum',))
        cases[-1].truth[:] = [1, 0]
        cases[-1].scores['maximum'][:] = [90, 0]
        after = choose_rule(cases, {'held'}, ('maximum',))
        self.assertEqual(before, after)
        self.assertNotIn('held', before['development_subjects'])

    def test_temporal_confirmation_resets_at_record_boundary(self):
        output = temporal_predictions([4, 4, 4, 4, 0, 0], 2., ['a', 'a', 'b', 'b', 'b', 'b'])
        np.testing.assert_array_equal(output, [-1, 1, -1, 1, -1, 0])

    def test_frozen_source_rejects_changed_hash_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self.record('test.csv')
            (root / 'test.csv').write_text('modified')
            manifest = root / 'input.json'
            evidence = asdict(record)
            evidence.pop('path')
            manifest.write_text(json.dumps({'audit': {'records': [evidence], 'excluded_records': []}}))
            with mock.patch('validate_rule_hypothesis.training_csv_paths', return_value=[root / 'test.csv']):
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    frozen_records(root, manifest)


if __name__ == '__main__':
    unittest.main()
