import sys
import unittest
from unittest.mock import patch

from manual_release_tracking.experiments.mediapipe_auto_release_hold import (
    parse_args,
)


class StaticMultiFocusRunnerTests(unittest.TestCase):
    def parse(self, *arguments):
        with patch.object(sys, "argv", ["runner.py", *arguments]):
            return parse_args()

    def test_baseline_defaults_are_unchanged(self):
        args = self.parse()
        self.assertFalse(args.static_multi_focus)
        self.assertEqual(args.condition_id, "baseline")
        self.assertEqual(args.paper_data_dir, "./manual_release_tracking/si2026_data")

    def test_multi_focus_has_separate_default_data_directory(self):
        args = self.parse("--static-multi-focus")
        self.assertEqual(args.condition_id, "static_multi_focus_circle")
        self.assertEqual(
            args.paper_data_dir,
            "./manual_release_tracking/si2026_data_static_multi_focus",
        )

    def test_explicit_condition_and_data_directory_are_preserved(self):
        args = self.parse(
            "--static-multi-focus",
            "--condition-id",
            "pilot_a",
            "--paper-data-dir",
            "pilot_data",
        )
        self.assertEqual(args.condition_id, "pilot_a")
        self.assertEqual(args.paper_data_dir, "pilot_data")


if __name__ == "__main__":
    unittest.main()
