import unittest
import numpy as np
from experiments.sensor_gain.top_two_ensemble import align,equal_mean


class EnsembleContract(unittest.TestCase):
    def test_equal_probability_mean_not_hard_vote(self):
        a=np.array([[.51,.49,0,0,0,0,0]])
        b=np.array([[.01,.99,0,0,0,0,0]])
        p=equal_mean(a,b)
        np.testing.assert_allclose(p,[[.26,.74,0,0,0,0,0]])
        self.assertEqual(p.argmax(1)[0],1)

    def test_invalid_class_probabilities_rejected(self):
        with self.assertRaises(ValueError):equal_mean(np.ones((1,7)),np.ones((1,7)))

    def test_match_record_and_endpoint_not_row_position(self):
        data=[{'y':np.array([0,0,0]),'groups':np.array(['a']*3),'record_ids':np.array(['r']*3),'start_s':np.array([0,2,4])},
              {'y':np.array([0,0]),'groups':np.array(['a']*2),'record_ids':np.array(['r']*2),'start_s':np.array([0,3])}]
        preds=[]
        for d,ix in zip(data,[np.array([2,0,1]),np.array([1,0])]):
            preds.append({'indices':ix,'truth':d['y'][ix],'groups':d['groups'][ix],'record_ids':d['record_ids'][ix]})
        indices,keys=align(preds,data,[2,3])
        self.assertEqual(keys,[('r',6.)]);self.assertEqual([i.tolist() for i in indices],[[0],[0]])


if __name__=='__main__':unittest.main()
