import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from manual_release_tracking.release_detector import (
    FingerDistanceResult,
    ReleaseDetectorConfig,
    ReleaseState,
    ReleaseStateMachine,
    detect_finger_distance,
)
from manual_release_tracking.experiments.release_detection_preview import (
    _combine_release_states,
    _resize_pair_to_same_size,
    _to_bgr,
)


def _observation(
    contact_ratio: float,
    near_area_normalized: float,
    normalized_gap: float | None,
) -> FingerDistanceResult:
    return FingerDistanceResult(
        distance_px=(
            None if normalized_gap is None else normalized_gap * 20.0
        ),
        finger_area_px=near_area_normalized * 400.0,
        contact_area_px=contact_ratio * 1000.0,
        contact_ratio=contact_ratio,
        near_area_normalized=near_area_normalized,
        normalized_gap=normalized_gap,
        nearest_ball_point=None,
        nearest_finger_point=None,
        contours=[],
        skin_mask_roi=np.zeros((1, 1), dtype=np.uint8),
        roi=(0, 0, 1, 1),
    )


class ReleaseStateMachineTest(unittest.TestCase):
    def setUp(self):
        self.cfg = ReleaseDetectorConfig(
            grasp_window_frames=7,
            grasp_required_votes=5,
            release_window_frames=5,
            release_required_votes=4,
            released_latch_frames=3,
            ball_lost_grace_frames=3,
        )
        self.machine = ReleaseStateMachine(self.cfg)
        self.grasp = _observation(0.080, 0.20, 0.05)
        self.release = _observation(0.005, 0.01, 0.30)

    def _arm(self):
        for _ in range(self.cfg.grasp_required_votes):
            state = self.machine.update(
                self.grasp,
                ball_detected=True,
                position_stable=False,
            )
        self.assertEqual(state, ReleaseState.GRASPED)

    def test_strong_contact_arms_and_stores_baseline(self):
        self._arm()
        self.assertAlmostEqual(self.machine.baseline_contact_ratio, 0.080)
        self.assertAlmostEqual(self.machine.baseline_gap_normalized, 0.05)

    def test_one_missing_skin_frame_does_not_release(self):
        self._arm()
        state = self.machine.update(
            _observation(0.0, 0.0, None),
            ball_detected=True,
        )
        self.assertEqual(state, ReleaseState.GRASPED)
        state = self.machine.update(self.grasp, ball_detected=True)
        self.assertEqual(state, ReleaseState.GRASPED)

    def test_relative_contact_drop_releases_after_vote_threshold(self):
        self._arm()
        for _ in range(self.cfg.release_required_votes):
            state = self.machine.update(self.release, ball_detected=True)
        self.assertEqual(state, ReleaseState.RELEASED)
        self.assertEqual(
            self.machine.transition_reason,
            "contact_ring_separated",
        )

    def test_short_ball_detection_dropout_keeps_grasped(self):
        self._arm()
        for _ in range(self.cfg.ball_lost_grace_frames):
            state = self.machine.update(None, ball_detected=False)
            self.assertEqual(state, ReleaseState.GRASPED)

        state = self.machine.update(None, ball_detected=False)
        self.assertEqual(state, ReleaseState.WAITING_FOR_GRASP)
        self.assertEqual(self.machine.transition_reason, "ball_lost_timeout")

    def test_release_rearms_and_allows_second_grasp(self):
        self._arm()
        for _ in range(self.cfg.release_required_votes):
            state = self.machine.update(self.release, ball_detected=True)
        self.assertEqual(state, ReleaseState.RELEASED)

        for _ in range(self.cfg.released_latch_frames - 1):
            state = self.machine.update(self.release, ball_detected=True)
            self.assertEqual(state, ReleaseState.RELEASED)

        state = self.machine.update(self.release, ball_detected=True)
        self.assertEqual(state, ReleaseState.WAITING_FOR_GRASP)
        self.assertEqual(self.machine.transition_reason, "ready_for_regrasp")

        self._arm()
        self.assertEqual(self.machine.state, ReleaseState.GRASPED)


class ContactRingFeatureTest(unittest.TestCase):
    def test_contact_ring_is_radius_normalized(self):
        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        skin_bgr = (120, 150, 200)
        cv2.rectangle(frame, (101, 70), (116, 90), skin_bgr, thickness=-1)

        result = detect_finger_distance(
            frame,
            ball_center=(80.0, 80.0),
            ball_radius_px=20.0,
            cfg=ReleaseDetectorConfig(),
        )

        self.assertGreater(result.contact_area_px, 0.0)
        self.assertGreater(result.contact_ratio, 0.0)
        self.assertIsNotNone(result.normalized_gap)
        self.assertLess(result.normalized_gap, 0.30)

    def test_rgb_bgr_swap_is_not_mistaken_for_skin(self):
        correct_bgr = np.zeros((160, 160, 3), dtype=np.uint8)
        swapped_rgb = np.zeros_like(correct_bgr)
        cv2.rectangle(correct_bgr, (101, 70), (116, 90), (120, 150, 200), -1)
        cv2.rectangle(swapped_rgb, (101, 70), (116, 90), (200, 150, 120), -1)

        correct = detect_finger_distance(
            correct_bgr,
            ball_center=(80.0, 80.0),
            ball_radius_px=20.0,
        )
        swapped = detect_finger_distance(
            swapped_rgb,
            ball_center=(80.0, 80.0),
            ball_radius_px=20.0,
        )

        self.assertGreater(correct.contact_ratio, 0.0)
        self.assertEqual(swapped.contact_ratio, 0.0)


class PreviewDisplayTest(unittest.TestCase):
    def test_color_frame_is_not_channel_swapped_for_display(self):
        frame_bgr = np.array([[[10, 20, 200]]], dtype=np.uint8)
        shown = _to_bgr(frame_bgr)
        np.testing.assert_array_equal(shown, frame_bgr)
        self.assertIsNot(shown, frame_bgr)

    def test_camera_panels_are_resized_to_identical_dimensions(self):
        frame_xy = np.zeros((480, 800, 3), dtype=np.uint8)
        frame_z = np.zeros((800, 480, 3), dtype=np.uint8)

        shown_xy, shown_z = _resize_pair_to_same_size(
            frame_xy,
            frame_z,
            max_display_height=720,
        )

        self.assertEqual(shown_xy.shape, shown_z.shape)
        self.assertEqual(shown_xy.shape[:2], (720, 720))

    def test_both_mode_requires_state_agreement(self):
        self.assertEqual(
            _combine_release_states(
                "both",
                ReleaseState.GRASPED,
                ReleaseState.WAITING_FOR_GRASP,
            ),
            ReleaseState.WAITING_FOR_GRASP,
        )
        self.assertEqual(
            _combine_release_states(
                "both",
                ReleaseState.GRASPED,
                ReleaseState.GRASPED,
            ),
            ReleaseState.GRASPED,
        )
        self.assertEqual(
            _combine_release_states(
                "both",
                ReleaseState.RELEASED,
                ReleaseState.RELEASED,
            ),
            ReleaseState.RELEASED,
        )


if __name__ == "__main__":
    unittest.main()
