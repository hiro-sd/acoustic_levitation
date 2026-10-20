import unittest
from types import SimpleNamespace

from tracking.experiments.acoustic_force_step_response import (
    StepCondition,
    _make_schedule,
)


def make_args(*, repeats=3, no_randomize=False, seed=1234):
    return SimpleNamespace(
        repeats=repeats,
        intensities=(0.5, 0.8),
        x_offsets_mm=(0.0,),
        y_offsets_mm=(0.0,),
        z_offsets_mm=(-5.0, 5.0),
        no_randomize=no_randomize,
        seed=seed,
    )


class AcousticForceScheduleTests(unittest.TestCase):
    def test_each_randomized_repeat_is_a_complete_block(self):
        args = make_args(repeats=4)
        schedule = _make_schedule(args)
        expected = {
            StepCondition(intensity, 0.0, 0.0, z_offset)
            for intensity in args.intensities
            for z_offset in args.z_offsets_mm
        }
        block_size = len(expected)

        self.assertEqual(len(schedule), block_size * args.repeats)
        for start in range(0, len(schedule), block_size):
            self.assertEqual(set(schedule[start : start + block_size]), expected)

        counts = {condition: schedule.count(condition) for condition in expected}
        self.assertEqual(set(counts.values()), {args.repeats})

    def test_randomized_schedule_is_reproducible_for_the_same_seed(self):
        first = _make_schedule(make_args(seed=20261020))
        second = _make_schedule(make_args(seed=20261020))
        different = _make_schedule(make_args(seed=20261021))

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_no_randomize_repeats_the_same_condition_order(self):
        args = make_args(repeats=2, no_randomize=True)
        schedule = _make_schedule(args)
        block_size = len(args.intensities) * len(args.z_offsets_mm)

        self.assertEqual(schedule[:block_size], schedule[block_size:])


if __name__ == "__main__":
    unittest.main()
