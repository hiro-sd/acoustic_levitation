import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from manual_release_tracking.core.mediapipe_hands import (
    BallFrameMetadata,
    BinaryErrorStats,
    HandTipConfig,
    TipContactState,
    TipContactStateConfig,
    TipContactStateMachine,
    build_hand_tip_observation,
)


class HandTipGeometryTest(unittest.TestCase):
    def setUp(self):
        self.metadata = BallFrameMetadata(
            frame_width=200,
            frame_height=200,
            ball_center=(100.0, 100.0),
            ball_radius_px=20.0,
        )
        self.cfg = HandTipConfig(
            fingertip_gap_normalized_max=0.45,
            min_opposition_angle_deg=60.0,
        )

    def test_opposite_fingertips_near_surface_are_contact(self):
        observation = build_hand_tip_observation(
            sequence=1,
            timestamp_ms=1000,
            received_time_ms=1010,
            metadata=self.metadata,
            thumb_tip_normalized=(0.39, 0.50),
            index_tip_normalized=(0.61, 0.50),
            cfg=self.cfg,
        )

        self.assertTrue(observation.contact_candidate)
        self.assertAlmostEqual(observation.thumb_gap_normalized, 0.10)
        self.assertAlmostEqual(observation.index_gap_normalized, 0.10)
        self.assertAlmostEqual(observation.opposition_angle_deg, 180.0)
        self.assertAlmostEqual(observation.result_latency_ms, 10.0)

    def test_two_fingertips_on_same_side_are_not_grasp(self):
        observation = build_hand_tip_observation(
            sequence=1,
            timestamp_ms=1000,
            received_time_ms=1010,
            metadata=self.metadata,
            thumb_tip_normalized=(0.61, 0.50),
            index_tip_normalized=(0.60, 0.52),
            cfg=self.cfg,
        )

        self.assertFalse(observation.contact_candidate)
        self.assertLess(observation.opposition_angle_deg, 60.0)

    def test_distant_fingertip_is_not_grasp(self):
        observation = build_hand_tip_observation(
            sequence=1,
            timestamp_ms=1000,
            received_time_ms=1010,
            metadata=self.metadata,
            thumb_tip_normalized=(0.30, 0.50),
            index_tip_normalized=(0.61, 0.50),
            cfg=self.cfg,
        )

        self.assertFalse(observation.contact_candidate)
        self.assertGreater(
            observation.thumb_gap_normalized,
            self.cfg.fingertip_gap_normalized_max,
        )


class TipContactStateMachineTest(unittest.TestCase):
    def setUp(self):
        self.machine = TipContactStateMachine(
            TipContactStateConfig(
                grasp_confirm_ms=40.0,
                release_confirm_ms=30.0,
                released_latch_ms=100.0,
            )
        )

    def test_grasp_release_and_rearm_cycle(self):
        self.assertEqual(
            self.machine.update(0.0, True),
            TipContactState.WAITING_FOR_GRASP,
        )
        self.assertEqual(
            self.machine.update(40.0, True),
            TipContactState.GRASPED,
        )
        self.assertEqual(
            self.machine.update(50.0, False),
            TipContactState.GRASPED,
        )
        self.assertEqual(
            self.machine.update(80.0, False),
            TipContactState.RELEASED,
        )
        self.assertTrue(self.machine.just_released)
        self.assertEqual(
            self.machine.update(180.0, False),
            TipContactState.WAITING_FOR_GRASP,
        )

    def test_invalid_observation_does_not_release(self):
        self.machine.update(0.0, True)
        self.machine.update(40.0, True)
        self.assertEqual(self.machine.state, TipContactState.GRASPED)
        self.machine.update(100.0, False, valid=False)
        self.machine.update(200.0, False, valid=False)
        self.assertEqual(self.machine.state, TipContactState.GRASPED)


class BinaryErrorStatsTest(unittest.TestCase):
    def test_false_positive_and_false_negative(self):
        stats = BinaryErrorStats()
        stats.update(predicted_grasped=True, actual_grasped=False)
        stats.update(predicted_grasped=False, actual_grasped=True)
        stats.update(predicted_grasped=True, actual_grasped=True)

        self.assertEqual(stats.samples, 3)
        self.assertEqual(stats.false_positive, 1)
        self.assertEqual(stats.false_negative, 1)
        self.assertAlmostEqual(stats.error_rate, 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
