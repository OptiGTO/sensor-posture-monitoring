from __future__ import annotations

import unittest

from src.training_windows import merge_window_selection, parse_window_text, validate_training_windows


class TrainingWindowTests(unittest.TestCase):
    def test_custom_values_are_parsed_deduplicated_and_sorted(self) -> None:
        self.assertEqual(parse_window_text("12, 4 7,12"), (4, 7, 12))

    def test_full_supported_range_is_available(self) -> None:
        self.assertEqual(validate_training_windows((1, 60)), (1, 60))

    def test_quick_choices_and_custom_input_are_combined(self) -> None:
        self.assertEqual(merge_window_selection((3, 20), "4, 7 20"), (3, 4, 7, 20))

    def test_empty_fractional_and_out_of_range_values_are_rejected(self) -> None:
        for values in ((), (0,), (61,), (2.5,), (True,)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_training_windows(values)


if __name__ == "__main__":
    unittest.main()
