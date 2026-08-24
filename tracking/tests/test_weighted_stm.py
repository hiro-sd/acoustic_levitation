import unittest

import numpy as np

from core.weighted_stm import (
    expand_focus_offsets,
    make_weighted_dwell_pattern,
)


class WeightedDwellPatternTests(unittest.TestCase):
    def test_zero_bias_allocates_equal_slots(self):
        pattern = make_weighted_dwell_pattern(
            point_num=8,
            total_slots=32,
            bias_angle_rad=0.0,
            bias_level=0.0,
        )

        self.assertEqual(pattern.slot_counts, (4,) * 8)
        self.assertEqual(pattern.total_slots, 32)

    def test_bias_increases_dwell_on_named_side(self):
        positive_x = make_weighted_dwell_pattern(
            point_num=8,
            total_slots=32,
            bias_angle_rad=0.0,
            bias_level=0.4,
        )
        negative_x = make_weighted_dwell_pattern(
            point_num=8,
            total_slots=32,
            bias_angle_rad=np.pi,
            bias_level=0.4,
        )

        self.assertEqual(positive_x.total_slots, 32)
        self.assertEqual(negative_x.total_slots, 32)
        self.assertGreater(positive_x.slot_counts[0], positive_x.slot_counts[4])
        self.assertLess(negative_x.slot_counts[0], negative_x.slot_counts[4])

    def test_none_keeps_the_existing_focus_sequence(self):
        offsets = np.arange(24, dtype=np.float32).reshape(8, 3)

        expanded = expand_focus_offsets(offsets, None)

        np.testing.assert_array_equal(expanded, offsets)

    def test_expansion_repeats_each_focus_consecutively(self):
        offsets = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )

        expanded = expand_focus_offsets(offsets, (2, 3))

        np.testing.assert_array_equal(
            expanded[:, 0],
            np.asarray([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

    def test_invalid_slot_count_is_rejected(self):
        with self.assertRaises(ValueError):
            make_weighted_dwell_pattern(
                point_num=8,
                total_slots=4,
                bias_angle_rad=0.0,
                bias_level=0.2,
            )


if __name__ == "__main__":
    unittest.main()
