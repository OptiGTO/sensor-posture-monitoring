import unittest
import numpy as np
from experiments.empty_seat_e2.run import transform, gate, audit_score


class E2Contract(unittest.TestCase):
    def test_signed_offset_before_gain_preserves_input(self):
        x = np.ones((2, 36, 16), dtype=np.float32)
        old = x.copy(); b = np.full(16, 1.25); g = np.full(16, 2.)
        np.testing.assert_array_equal(transform(x, "P0", b, g), x)
        np.testing.assert_array_equal(transform(x, "P1", b, g), np.full_like(x, -.25))
        np.testing.assert_array_equal(transform(x, "P2", b, g), np.full_like(x, 2.))
        np.testing.assert_array_equal(transform(x, "P3", b, g), np.full_like(x, -.5))
        np.testing.assert_array_equal(x, old)

    def test_invalid_calibration_rejected(self):
        for arm, b, g in [("bad", np.ones(16), np.ones(16)), ("P1", np.ones(15), np.ones(16)),
                          ("P2", np.ones(16), np.zeros(16)), ("P3", np.full(16, np.nan), np.ones(16))]:
            with self.assertRaises(ValueError):
                transform(np.ones((1,36,16)), arm, b, g)

    def test_positive_mean_cannot_override_class_guardrail(self):
        def metric(value, cls=None):
            return {"macro_f1": value, "per_class_f1": {str(i): value if cls is None else cls[i] for i in range(7)}}
        base = [metric(.60)] * 3
        self.assertTrue(gate([metric(.62)] * 3, base)["passed"])
        self.assertFalse(gate([metric(.609)] * 3, base)["passed"])
        self.assertFalse(gate([metric(.69), metric(.60), metric(.60)], base)["passed"])
        bad = [metric(.62), metric(.62, [.54, .64, .64, .64, .64, .64, .64]), metric(.62)]
        self.assertFalse(gate(bad, base)["passed"])
        with self.assertRaises(ValueError):
            gate(base[:2], base)

    def test_independent_metric_known_cases(self):
        y = np.arange(7)
        self.assertEqual(audit_score(y,y)["macro_f1"], 1.)
        self.assertEqual(audit_score(y,(y+1)%7)["macro_f1"], 0.)
        y = np.repeat(np.arange(7),2); pred = y.copy(); pred[0] = 1
        expected = ((2/3) + (4/5) + 5) / 7
        self.assertAlmostEqual(audit_score(y,pred)["macro_f1"], expected)


if __name__ == "__main__":
    unittest.main()
