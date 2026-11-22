import unittest
import numpy as np
from experiments.sensor_gain.et_followup import PARAMS, fixed_model_seed_partitions
from experiments.sensor_gain.run import build_model
from experiments.sensor_gain.summarize import matrix, metrics, subject_bootstrap


class RegularizedETContract(unittest.TestCase):
    def test_original_et_model_seed_without_changing_subject_split(self):
        source = [{"seed": 20260922, "train": ['a'], "validation": ['b'], "test": ['c']}]
        result = fixed_model_seed_partitions(source, 20260921)
        self.assertEqual(result[0]['seed'], 20260921)
        self.assertEqual(result[0]['split_seed'], 20260922)
        self.assertEqual(source[0]['seed'], 20260922)
        for key in ('train', 'validation', 'test'): self.assertEqual(result[0][key], source[0][key])

    def test_recipe_is_pre_restoration_regularized_et(self):
        model = build_model('extra_trees', PARAMS, 20260921)
        actual = model.named_steps['model'].get_params()
        for k, v in PARAMS.items(): self.assertEqual(actual[k], v)
        self.assertEqual(actual['class_weight'], 'balanced')
        self.assertEqual(list(model.named_steps), ['scale', 'model'])

    def test_independent_metrics_and_paired_bootstrap(self):
        y = np.tile(np.arange(7), 3); groups = np.repeat(['a', 'b', 'c'], 7)
        self.assertEqual(metrics(matrix(y, y))['macro_f1'], 1.)
        self.assertEqual(metrics(matrix(y, (y+1)%7))['macro_f1'], 0.)
        self.assertEqual(subject_bootstrap(y, y, y, groups, 20), [0., 0.])
        self.assertEqual(subject_bootstrap(y, (y+1)%7, y, groups, 20), [1., 1.])


if __name__ == '__main__': unittest.main()
