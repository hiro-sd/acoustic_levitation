import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from manual_release_tracking.core.mediapipe_release_trigger import (
    build_auto_release_event,
)


class AutoReleaseEventTest(unittest.TestCase):
    def test_fresh_release_transition_emits_one_event(self):
        event = build_auto_release_event(
            just_released=True,
            event_timestamp_ms=1000,
            now_ms=1042.5,
            pair_skew_ms=3.0,
            maximum_age_ms=80.0,
        )

        self.assertIsNotNone(event)
        self.assertEqual(event.timestamp_ms, 1000)
        self.assertAlmostEqual(event.result_age_ms, 42.5)
        self.assertAlmostEqual(event.result_pair_skew_ms, 3.0)

    def test_non_transition_does_not_emit(self):
        event = build_auto_release_event(
            just_released=False,
            event_timestamp_ms=1000,
            now_ms=1010,
            pair_skew_ms=2.0,
            maximum_age_ms=80.0,
        )

        self.assertIsNone(event)

    def test_stale_release_is_rejected(self):
        event = build_auto_release_event(
            just_released=True,
            event_timestamp_ms=1000,
            now_ms=1080.1,
            pair_skew_ms=2.0,
            maximum_age_ms=80.0,
        )

        self.assertIsNone(event)


if __name__ == "__main__":
    unittest.main()
