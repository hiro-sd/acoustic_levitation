import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.config import AppConfig
from tracking.core.control.recovery import (
    apply_capture_intensity_slew,
    capture_force_command_for_velocity,
    capture_intensity_from_force,
    fall_detection_reason,
    measured_force_for_intensity_mN,
    predict_falling_position,
    required_capture_force_mN,
)
from tracking.core.models import HomePosition


class FallRecoveryForceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = AppConfig()

    def test_required_force_uses_downward_speed_and_safety_factor(self):
        cfg = AppConfig(
            ball_mass_kg=0.0005,
            fall_capture_time_sec=0.1,
            fall_capture_force_safety_factor=1.0,
        )

        # predicted_vz=-200 mm/s -> u=0.2 m/s
        force_mN = required_capture_force_mN(cfg, predicted_vz_mm_s=-200.0)

        expected_mN = 1000.0 * 0.0005 * (9.80665 + 0.2 / 0.1)
        self.assertAlmostEqual(force_mN, expected_mN, places=6)

    def test_upward_or_stationary_speed_needs_weight_only_before_safety(self):
        cfg = AppConfig(
            ball_mass_kg=0.0005,
            fall_capture_time_sec=0.1,
            fall_capture_force_safety_factor=1.0,
        )

        force_mN = required_capture_force_mN(cfg, predicted_vz_mm_s=100.0)

        self.assertAlmostEqual(force_mN, 1000.0 * 0.0005 * 9.80665, places=6)

    def test_loadcell_model_interpolates_force(self):
        self.assertAlmostEqual(measured_force_for_intensity_mN(self.cfg, 0.85), 6.5)

    def test_capture_intensity_is_staged(self):
        minimum = capture_intensity_from_force(self.cfg, 3.5)
        low_hold = capture_intensity_from_force(self.cfg, 4.5)
        low = capture_intensity_from_force(self.cfg, 5.5)
        mid = capture_intensity_from_force(self.cfg, 6.5)
        high = capture_intensity_from_force(self.cfg, 7.5)

        self.assertEqual(minimum.commanded_intensity, 0.6)
        self.assertEqual(low_hold.commanded_intensity, 0.7)
        self.assertEqual(low.commanded_intensity, 0.8)
        self.assertEqual(mid.commanded_intensity, 0.9)
        self.assertEqual(high.commanded_intensity, 0.9)
        self.assertTrue(high.saturated)
        self.assertFalse(low.saturated)

    def test_capture_intensity_saturates_but_continues_at_configured_max(self):
        cmd = capture_intensity_from_force(self.cfg, 20.0)

        self.assertEqual(cmd.commanded_intensity, 0.9)
        self.assertTrue(cmd.saturated)

    def test_velocity_force_command_uses_shifted_experiment_model(self):
        cfg = AppConfig(
            ball_mass_kg=0.0005,
            fall_capture_time_sec=0.12,
            fall_capture_force_safety_factor=1.0,
            fall_capture_max_intensity_ratio=0.8,
            fall_capture_intensity_levels=(0.5, 0.6, 0.7, 0.8),
            fall_capture_loadcell_model=(
                (0.5, 4.0),
                (0.6, 5.0),
                (0.7, 6.0),
                (0.8, 7.0),
            ),
        )

        hold = capture_force_command_for_velocity(cfg, 0.0)
        medium_drop = capture_force_command_for_velocity(cfg, -150.0)
        fast_drop = capture_force_command_for_velocity(cfg, -300.0)
        saturated_drop = capture_force_command_for_velocity(cfg, -600.0)

        self.assertAlmostEqual(hold.required_force_mN, 4.903325)
        self.assertEqual(hold.commanded_intensity, 0.6)
        self.assertEqual(medium_drop.commanded_intensity, 0.7)
        self.assertEqual(fast_drop.commanded_intensity, 0.8)
        self.assertEqual(saturated_drop.commanded_intensity, 0.8)
        self.assertTrue(saturated_drop.saturated)

    def test_force_command_decrease_has_boundary_hysteresis(self):
        cfg = AppConfig(
            ball_mass_kg=0.0005,
            fall_capture_time_sec=0.12,
            fall_capture_force_safety_factor=1.0,
            fall_capture_max_intensity_ratio=0.8,
            fall_capture_intensity_levels=(0.5, 0.6, 0.7, 0.8),
            fall_capture_loadcell_model=(
                (0.5, 4.0),
                (0.6, 5.0),
                (0.7, 6.0),
                (0.8, 7.0),
            ),
        )

        near_boundary = capture_force_command_for_velocity(
            cfg,
            -15.0,
            current_intensity=0.7,
            downward_hysteresis_mN=0.05,
        )
        settled = capture_force_command_for_velocity(
            cfg,
            0.0,
            current_intensity=0.7,
            downward_hysteresis_mN=0.05,
        )

        self.assertEqual(near_boundary.commanded_intensity, 0.7)
        self.assertEqual(settled.commanded_intensity, 0.6)

    def test_intensity_slew_rises_immediately_and_falls_slowly(self):
        cfg = AppConfig(fall_intensity_down_slew_per_sec=0.4)

        self.assertEqual(apply_capture_intensity_slew(0.8, 1.0, cfg, 0.01), 1.0)
        self.assertAlmostEqual(
            apply_capture_intensity_slew(1.0, 0.8, cfg, 0.1),
            0.96,
            places=6,
        )


class FallRecoveryStateConditionTests(unittest.TestCase):
    def test_fall_detection_requires_speed_frames_and_outside_hold_region(self):
        cfg = AppConfig(
            fall_vz_threshold_mm_s=-80.0,
            fall_descending_frames=5,
            fall_hold_region_z_mm=12.0,
            fall_hold_region_xy_mm=12.0,
        )
        home = HomePosition(0.0, 0.0, 400.0)

        self.assertIsNone(
            fall_detection_reason(cfg, home, 0.0, 0.0, 390.0, -100.0, 5)
        )
        self.assertIsNone(
            fall_detection_reason(cfg, home, 0.0, 0.0, 380.0, -100.0, 4)
        )
        self.assertIsNone(
            fall_detection_reason(cfg, home, 0.0, 0.0, 380.0, -50.0, 5)
        )
        self.assertIsNone(
            fall_detection_reason(cfg, home, 20.0, 0.0, 400.0, -100.0, 5)
        )

        reason = fall_detection_reason(cfg, home, 0.0, 0.0, 380.0, -100.0, 5)
        self.assertIsNotNone(reason)
        self.assertIn("fall_detected", reason)

    def test_prediction_uses_latency_and_gravity(self):
        cfg = AppConfig(fall_system_delay_sec=0.02, gravity_mm_s2=9800.0)

        px, py, pz, pvz = predict_falling_position(
            cfg,
            x_mm=10.0,
            y_mm=20.0,
            z_mm=400.0,
            vx_mm_s=100.0,
            vy_mm_s=-50.0,
            vz_mm_s=-200.0,
        )

        self.assertAlmostEqual(px, 12.0)
        self.assertAlmostEqual(py, 19.0)
        self.assertAlmostEqual(pz, 400.0 - 4.0 - 0.5 * 9800.0 * 0.02**2)
        self.assertAlmostEqual(pvz, -200.0 - 9800.0 * 0.02)


if __name__ == "__main__":
    unittest.main()
