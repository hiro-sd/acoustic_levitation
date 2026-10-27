import unittest

import cv2
import numpy as np

from tracking.core.ball_tracker import RoiBallTracker
from tracking.core.config import AppConfig
from tracking.core.radius_prior import SphereRadiusPrior
from tracking.tests.test_upper_arc_refinement import shaded_sphere


def noisy_shaded_frame(rng, center=(120, 130), radius=40):
    y, x = np.indices((240, 240), dtype=float)
    edge = np.clip((radius - np.hypot(x - center[0], y - center[1]) + 1.5) / 3, 0, 1)
    light = np.clip(110 - 3 * (y - center[1]) + rng.normal(0, 4), 60, 245)
    gray = 20 * (1 - edge) + light * edge + rng.normal(0, 5, (240, 240))
    return np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


class RadiusStabilityTests(unittest.TestCase):
    def make_tracker(self, stable=True):
        return RoiBallTracker(240, 240, AppConfig(z_stabilize_radius=stable), refine_upper_arc=True)

    def test_size_history_rejects_outlier_and_resets_after_losses(self):
        model = SphereRadiusPrior()
        for radius in (39.8, 40.2, 40.0, 40.1):
            model.observe(radius)
        self.assertAlmostEqual(model.radius, 40, delta=0.2)
        model.observe(70)
        self.assertAlmostEqual(model.radius, 40, delta=0.2)
        for _ in range(5):
            model.missing()
        self.assertIsNone(model.radius)
        self.assertEqual(len(model.samples), 0)

    def test_noise_jitter_and_bias_improve_without_hiding_losses(self):
        rng = np.random.default_rng(52)
        trackers = (self.make_tracker(False), self.make_tracker(True))
        results = ([], [])
        for i in range(180):
            frame = noisy_shaded_frame(rng)
            for tracker, samples in zip(trackers, results):
                found = tracker.detect(frame)
                if i >= 20 and found.detected:
                    samples.append((*found.center, found.radius_px))
        old, new = map(np.asarray, results)
        # Count misses explicitly; low jitter achieved by dropping frames is not
        # acceptable. Tolerate at most two very dim cap frames in this sequence.
        self.assertGreaterEqual(len(new), 158)
        self.assertLess(np.std(new[:, 1]), 0.3 * np.std(old[:, 1]))
        self.assertLess(np.std(new[:, 2]), 0.3 * np.std(old[:, 2]))
        error = np.sqrt(np.mean(np.sum((new[:, :2] - (120, 130)) ** 2, axis=1)))
        self.assertLess(error, 0.5)

    def test_consistent_size_change_can_replace_the_prior(self):
        model = SphereRadiusPrior()
        for radius in (40.0, 40.0, 40.0):
            model.observe(radius)
        model.observe(55.1)
        self.assertEqual(model.radius, 40.0)
        model.observe(54.9)
        model.observe(55.0)
        self.assertAlmostEqual(model.radius, 55.0, delta=0.2)

    def test_abrupt_real_motion_is_followed_on_same_frame(self):
        rng = np.random.default_rng(8)
        tracker = self.make_tracker()
        for _ in range(15):
            tracker.detect(noisy_shaded_frame(rng))
        self.assertIsNotNone(tracker.radius_model.radius)
        moved = tracker.detect(noisy_shaded_frame(rng, center=(135, 152)))
        self.assertTrue(moved.detected)
        np.testing.assert_allclose(moved.center, (135, 152), atol=1)

    def test_no_stale_detection_when_ball_disappears(self):
        rng = np.random.default_rng(8)
        tracker = self.make_tracker()
        for _ in range(15):
            tracker.detect(noisy_shaded_frame(rng))
        for _ in range(5):
            self.assertFalse(tracker.detect(np.zeros((240, 240, 3), np.uint8)).detected)
        self.assertIsNone(tracker.radius_model.radius)
        reacquired = tracker.detect(noisy_shaded_frame(rng, radius=35))
        self.assertTrue(reacquired.detected)
        np.testing.assert_allclose(reacquired.center, (120, 130), atol=2)

    def test_bright_finger_does_not_get_accepted_using_prior(self):
        rng = np.random.default_rng(8)
        tracker = self.make_tracker()
        for _ in range(15):
            tracker.detect(noisy_shaded_frame(rng))
        frame = shaded_sphere()
        cv2.rectangle(frame, (112, 55), (128, 102), (200, 175, 160), -1)
        self.assertFalse(tracker.detect(frame).detected)

    def test_xy_has_no_radius_model(self):
        self.assertIsNone(RoiBallTracker(240, 240, AppConfig()).radius_model)

    def test_small_and_large_pixelated_balls_keep_center_accuracy(self):
        for radius in (25, 40, 65):
            for cutoff in (-0.25, -0.4, -0.5):
                tracker = self.make_tracker()
                for i in range(12):
                    found = tracker.detect(shaded_sphere(radius, int(cutoff * radius), seed=i))
                self.assertTrue(found.detected)
                np.testing.assert_allclose(found.center, (120, 130), atol=3)
                self.assertAlmostEqual(found.radius_px, radius, delta=3)

    def test_lighting_change_keeps_current_center_with_established_prior(self):
        rng = np.random.default_rng(8)
        tracker = self.make_tracker()
        for _ in range(15):
            tracker.detect(noisy_shaded_frame(rng))
        for cutoff in (-15, -10, 0, 10, 40):
            found = tracker.detect(shaded_sphere(cutoff=cutoff))
            self.assertTrue(found.detected)
            np.testing.assert_allclose(found.center, (120, 130), atol=2)


if __name__ == "__main__":
    unittest.main()
