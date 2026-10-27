import unittest

import cv2
import numpy as np

from tracking.core.ball_tracker import RoiBallTracker
from tracking.core.config import AppConfig
from tracking.core.vision import _refine_upper_arc, track_ball_cv


def shaded_sphere(radius=40, cutoff=-15, fingers=False, seed=0):
    y, x = np.indices((240, 240))
    center = (120, 130)
    ball = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
    frame = np.full((240, 240, 3), 20, np.uint8)
    if fingers:
        # Fingers join the shaded lower hemisphere, outside the central top arc.
        cv2.rectangle(frame, (45, 131), (195, 153), (145, 115, 95), -1)
        cv2.circle(frame, (79, 129), 12, (150, 125, 100), -1)
        cv2.circle(frame, (161, 129), 12, (150, 125, 100), -1)
    frame[ball] = (75, 79, 83)
    frame[ball & (y <= center[1] + cutoff)] = (225, 230, 235)
    noise = np.random.default_rng(seed).normal(0, 1, frame.shape)
    return np.clip(frame.astype(float) + noise, 0, 255).astype(np.uint8)


class UpperArcRefinementTests(unittest.TestCase):
    def test_shaded_sphere_center_is_recovered(self):
        cfg = AppConfig()
        for radius in (25, 40, 65):
            for cutoff_fraction in (-0.25, -0.4, -0.5):
                with self.subTest(radius=radius, cutoff=cutoff_fraction):
                    frame = shaded_sphere(radius, int(radius * cutoff_fraction))
                    original, original_mask = track_ball_cv(frame, (0, 0, 240, 240), cfg)
                    corrected, corrected_mask = track_ball_cv(
                        frame, (0, 0, 240, 240), cfg, refine_upper_arc=True
                    )
                    self.assertIsNotNone(corrected)
                    np.testing.assert_array_equal(original_mask, corrected_mask)
                    if original is not None:
                        self.assertLess(original[1], 130 - radius * 0.15)
                    np.testing.assert_allclose(corrected[:2], (120, 130), atol=3.0)
                    self.assertAlmostEqual(corrected[2], radius, delta=3.0)

    def test_fingers_at_sides_and_below_do_not_pull_center(self):
        for seed in range(3):
            frame = shaded_sphere(fingers=True, seed=seed)
            detection, _ = track_ball_cv(frame, (0, 0, 240, 240), AppConfig(), refine_upper_arc=True)
            self.assertIsNotNone(detection)
            np.testing.assert_allclose(detection[:2], (120, 130), atol=3)
            self.assertAlmostEqual(detection[2], 40, delta=3)

    def test_fully_lit_ball_keeps_original_result(self):
        frame = shaded_sphere(cutoff=100)
        before, _ = track_ball_cv(frame, (0, 0, 240, 240), AppConfig())
        after, _ = track_ball_cv(frame, (0, 0, 240, 240), AppConfig(), refine_upper_arc=True)
        np.testing.assert_allclose(after, before, atol=1e-4)

    def test_full_ball_near_roi_edge_keeps_original_result(self):
        frame = np.full((240, 240, 3), 20, np.uint8)
        cv2.circle(frame, (24, 120), 25, (230, 230, 230), -1)
        before, _ = track_ball_cv(frame, (0, 0, 240, 240), AppConfig())
        after, _ = track_ball_cv(frame, (0, 0, 240, 240), AppConfig(), refine_upper_arc=True)
        self.assertIsNotNone(before)
        np.testing.assert_allclose(after, before, atol=1e-4)

    def test_unreliable_straight_finger_arc_is_not_extrapolated(self):
        mask = np.zeros((240, 240), np.uint8)
        cv2.rectangle(mask, (65, 100), (175, 130), 255, -1)
        contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        self.assertEqual(_refine_upper_arc(contour, (cx, cy, radius)), (cx, cy, radius))

    def test_single_small_highlight_is_not_extrapolated(self):
        mask = np.zeros((240, 240), np.uint8)
        cv2.circle(mask, (120, 110), 7, 255, -1)
        contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        self.assertEqual(_refine_upper_arc(contour, (cx, cy, radius)), (cx, cy, radius))

    def test_bright_fingers_contaminating_arc_are_rejected(self):
        for box in ((40, 102, 90, 127), (150, 102, 205, 127), (112, 55, 128, 102)):
            frame = shaded_sphere()
            cv2.rectangle(frame, box[:2], box[2:], (200, 175, 160), -1)
            found, _ = track_ball_cv(frame, (0, 0, 240, 240), AppConfig(), refine_upper_arc=True)
            self.assertIsNone(found)

    def test_blank_frame_returns_missing_detection(self):
        frame = np.zeros((240, 240, 3), np.uint8)
        found, _ = track_ball_cv(frame, (0, 0, 240, 240), AppConfig(), refine_upper_arc=True)
        self.assertIsNone(found)

    def test_roi_offset_and_frame_pixels_are_preserved(self):
        frame = np.zeros((320, 360, 3), np.uint8)
        frame[30:270, 50:290] = shaded_sphere(fingers=True)
        untouched = frame.copy()
        found, _ = track_ball_cv(frame, (50, 30, 290, 270), AppConfig(), refine_upper_arc=True)
        np.testing.assert_allclose(found[:2], (170, 160), atol=3)
        np.testing.assert_array_equal(frame, untouched)

    def test_z_tracker_enables_correction_explicitly(self):
        cfg = AppConfig()
        frame = shaded_sphere()
        xy = RoiBallTracker(240, 240, cfg).detect(frame)
        z = RoiBallTracker(240, 240, cfg, refine_upper_arc=cfg.z_refine_upper_arc).detect(frame)
        original, _ = track_ball_cv(frame, (0, 0, 240, 240), cfg)
        self.assertEqual(xy.detected, original is not None)
        if original is not None:
            np.testing.assert_allclose(xy.center, original[:2])
        self.assertAlmostEqual(z.center[1], 130, delta=3)


if __name__ == "__main__":
    unittest.main()
