import unittest

from experiments.learning_curve.full_cohort import ROOT, assert_registrations_included


class FullCohortTests(unittest.TestCase):
    def test_every_reserved_record_is_present_with_original_bytes(self):
        old = {"registrations_excluded": [{"relative_path": "sample/correct.csv", "sha256": "frozen"}]}
        staged = [{"source_file": str(ROOT / "data/sample/correct.csv"),
                   "source_sha256": "frozen", "staged_file": "sample/correct.csv"}]
        self.assertEqual(assert_registrations_included(old, staged), ["sample/correct.csv"])
        with self.assertRaisesRegex(ValueError, "registration missing"):
            assert_registrations_included(old, [])
        with self.assertRaisesRegex(ValueError, "content changed"):
            assert_registrations_included(old, [staged[0] | {"source_sha256": "modified"}])


if __name__ == "__main__":
    unittest.main()
