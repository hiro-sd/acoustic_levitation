import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from manual_release_tracking.core.mediapipe_release_trigger import (
    build_auto_release_event,
    fused_contact_candidate,
    normalize_decision_camera,
    observation_is_explicit,
    should_stop_automatic_hold,
)
from manual_release_tracking.core.mediapipe_hands import (
    BallFrameMetadata,
    HandTipConfig,
    TipContactState,
    TipContactStateConfig,
    TipContactStateMachine,
    build_hand_tip_observation,
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


class AutomaticContactFusionTest(unittest.TestCase):
    def test_decision_camera_accepts_xy_z_and_both(self):
        self.assertEqual(normalize_decision_camera("XY"), "xy")
        self.assertEqual(normalize_decision_camera("z"), "z")
        self.assertEqual(normalize_decision_camera(" both "), "both")
        with self.assertRaises(ValueError):
            normalize_decision_camera("either")

    def test_grasp_requires_contact_in_both_cameras(self):
        self.assertTrue(
            fused_contact_candidate(
                TipContactState.WAITING_FOR_GRASP,
                xy_contact=True,
                z_contact=True,
            )
        )
        self.assertFalse(
            fused_contact_candidate(
                TipContactState.WAITING_FOR_GRASP,
                xy_contact=True,
                z_contact=False,
            )
        )

    def test_release_requires_separation_in_both_cameras(self):
        self.assertTrue(
            fused_contact_candidate(
                TipContactState.GRASPED,
                xy_contact=True,
                z_contact=False,
            )
        )
        self.assertFalse(
            fused_contact_candidate(
                TipContactState.GRASPED,
                xy_contact=False,
                z_contact=False,
            )
        )

    def test_one_camera_dropout_does_not_advance_release_timer(self):
        machine = TipContactStateMachine(
            TipContactStateConfig(
                grasp_confirm_ms=40.0,
                release_confirm_ms=30.0,
                released_latch_ms=100.0,
            )
        )
        machine.update(
            0.0,
            fused_contact_candidate(machine.state, True, True),
        )
        machine.update(
            40.0,
            fused_contact_candidate(machine.state, True, True),
        )
        self.assertEqual(machine.state, TipContactState.GRASPED)

        machine.update(
            50.0,
            fused_contact_candidate(machine.state, True, False),
        )
        machine.update(
            100.0,
            fused_contact_candidate(machine.state, True, False),
        )
        self.assertEqual(machine.state, TipContactState.GRASPED)

        machine.update(
            110.0,
            fused_contact_candidate(machine.state, False, False),
        )
        machine.update(
            140.0,
            fused_contact_candidate(machine.state, False, False),
        )
        self.assertEqual(machine.state, TipContactState.RELEASED)
        self.assertTrue(machine.just_released)

    def test_missing_hand_is_unknown_not_release_evidence(self):
        observation = build_hand_tip_observation(
            sequence=1,
            timestamp_ms=1000,
            received_time_ms=1010,
            metadata=BallFrameMetadata(
                frame_width=200,
                frame_height=200,
                ball_center=(100.0, 100.0),
                ball_radius_px=20.0,
            ),
            thumb_tip_normalized=None,
            index_tip_normalized=None,
            cfg=HandTipConfig(),
        )

        self.assertTrue(observation.ball_detected)
        self.assertFalse(observation.hand_detected)
        self.assertFalse(observation_is_explicit(observation))

    def test_only_regrasp_stops_an_automatic_hold(self):
        self.assertTrue(
            should_stop_automatic_hold(
                auto_hold_active=True,
                both_state=TipContactState.GRASPED,
            )
        )
        self.assertFalse(
            should_stop_automatic_hold(
                auto_hold_active=False,
                both_state=TipContactState.GRASPED,
            )
        )
        self.assertFalse(
            should_stop_automatic_hold(
                auto_hold_active=True,
                both_state=TipContactState.RELEASED,
            )
        )


if __name__ == "__main__":
    unittest.main()
