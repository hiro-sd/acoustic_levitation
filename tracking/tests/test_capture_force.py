import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.config import AppConfig
from manual_release_tracking.core.auto_release_capture import (
    braking_reference_at,
    make_braking_plan,
)
from tracking.core.control.capture_force import (
    SI2026_CAPTURE_FORCE_MODEL,
    apply_capture_intensity_slew,
    capture_buoyancy_mN,
    capture_force_command_for_velocity,
    capture_intensity_interpolated_from_force,
    capture_intensity_from_force,
    measured_force_for_intensity_mN,
    required_capture_force_mN,
)


class CaptureForceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = AppConfig()

    def test_required_force_uses_downward_speed_and_safety_factor(self):
        cfg = AppConfig(
            ball_mass_kg=0.0005,
            capture_time_sec=0.1,
            capture_force_safety_factor=1.0,
        )

        # predicted_vz=-200 mm/s -> u=0.2 m/s
        force_mN = required_capture_force_mN(cfg, predicted_vz_mm_s=-200.0)

        expected_mN = (
            1000.0 * 0.0005 * (9.80665 + 0.2 / 0.1)
            - capture_buoyancy_mN(cfg)
        )
        self.assertAlmostEqual(force_mN, expected_mN, places=6)

    def test_fixed_deadline_keeps_ideal_required_force_constant(self):
        cfg = AppConfig(
            ball_mass_kg=0.0005,
            capture_time_sec=0.1,
            capture_force_safety_factor=1.0,
        )

        initial = required_capture_force_mN(
            cfg,
            predicted_vz_mm_s=-200.0,
            time_to_stop_sec=0.1,
        )
        halfway = required_capture_force_mN(
            cfg,
            predicted_vz_mm_s=-100.0,
            time_to_stop_sec=0.05,
        )

        self.assertAlmostEqual(initial, halfway, places=6)

    def test_fixed_deadline_uses_remaining_time_in_force_command(self):
        cfg = AppConfig(
            ball_mass_kg=0.0005,
            capture_time_sec=0.1,
            capture_force_safety_factor=1.0,
            capture_max_intensity_ratio=1.0,
            capture_loadcell_model=(
                (0.5, 4.0),
                (0.6, 5.0),
                (0.7, 6.0),
                (0.8, 7.0),
                (0.9, 8.0),
                (1.0, 9.0),
            ),
        )

        initial = capture_force_command_for_velocity(
            cfg,
            -200.0,
            time_to_stop_sec=0.1,
            interpolate_intensity=True,
            minimum_intensity=0.5,
        )
        halfway = capture_force_command_for_velocity(
            cfg,
            -100.0,
            time_to_stop_sec=0.05,
            interpolate_intensity=True,
            minimum_intensity=0.5,
        )

        self.assertAlmostEqual(
            initial.commanded_intensity,
            halfway.commanded_intensity,
            places=6,
        )

    def test_braking_reference_and_force_share_fixed_endpoint(self):
        cfg = AppConfig(
            ball_mass_kg=0.00053,
            ball_diameter_mm=38.0,
            capture_time_sec=0.1,
            capture_force_safety_factor=1.0,
            capture_max_intensity_ratio=1.0,
            capture_loadcell_model=SI2026_CAPTURE_FORCE_MODEL,
        )
        plan = make_braking_plan(
            start_time_sec=1.0,
            initial_z_mm=460.0,
            initial_vz_mm_s=-700.0,
            nominal_duration_sec=0.1,
            maximum_duration_sec=1.0,
            minimum_duration_sec=0.05,
            ball_mass_kg=cfg.ball_mass_kg,
            gravity_mm_s2=9806.65,
            maximum_force_mN=SI2026_CAPTURE_FORCE_MODEL[-1][1],
            z_min_mm=250.0,
            workspace_margin_mm=10.0,
            buoyancy_mN=capture_buoyancy_mN(cfg),
        )

        for fraction in (0.25, 0.5, 0.75):
            elapsed = plan.duration_sec * fraction
            reference = braking_reference_at(
                plan,
                now_sec=plan.start_time_sec + elapsed,
            )
            force_mn = required_capture_force_mN(
                cfg,
                reference.vz_mm_s,
                time_to_stop_sec=plan.duration_sec - elapsed,
            )
            self.assertAlmostEqual(
                force_mn,
                plan.required_force_mN,
                places=6,
            )

    def test_upward_or_stationary_speed_needs_weight_only_before_safety(self):
        cfg = AppConfig(
            ball_mass_kg=0.0005,
            capture_time_sec=0.1,
            capture_force_safety_factor=1.0,
        )

        force_mN = required_capture_force_mN(cfg, predicted_vz_mm_s=100.0)

        self.assertAlmostEqual(
            force_mN,
            1000.0 * 0.0005 * 9.80665 - capture_buoyancy_mN(cfg),
            places=6,
        )

    def test_si2026_camera_step_response_model_is_monotone(self):
        intensities = [point[0] for point in SI2026_CAPTURE_FORCE_MODEL]
        forces = [point[1] for point in SI2026_CAPTURE_FORCE_MODEL]

        self.assertEqual(intensities, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.assertTrue(all(b > a for a, b in zip(forces, forces[1:])))
        self.assertAlmostEqual(forces[0], 3.947338)
        self.assertAlmostEqual(forces[-1], 8.919526)

    def test_si2026_model_uses_full_intensity_range_for_fast_fall(self):
        cfg = AppConfig(
            static_intensity_ratio=0.6,
            ball_mass_kg=0.00053,
            ball_diameter_mm=38.0,
            capture_time_sec=0.12,
            capture_max_intensity_ratio=1.0,
            capture_loadcell_model=SI2026_CAPTURE_FORCE_MODEL,
        )

        command = capture_force_command_for_velocity(
            cfg,
            -800.0,
            interpolate_intensity=True,
            minimum_intensity=cfg.static_intensity_ratio,
        )

        # Correcting both the inferred force and the controller mass for the
        # same sphere preserves the acceleration-to-intensity command.
        self.assertAlmostEqual(command.commanded_intensity, 0.946915, places=5)
        self.assertFalse(command.saturated)

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
            capture_time_sec=0.12,
            capture_force_safety_factor=1.0,
            capture_max_intensity_ratio=0.8,
            capture_intensity_levels=(0.5, 0.6, 0.7, 0.8),
            capture_loadcell_model=(
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

        self.assertAlmostEqual(
            hold.required_force_mN,
            4.903325 - capture_buoyancy_mN(cfg),
        )
        self.assertEqual(hold.commanded_intensity, 0.6)
        self.assertEqual(medium_drop.commanded_intensity, 0.7)
        self.assertEqual(fast_drop.commanded_intensity, 0.7)
        self.assertEqual(saturated_drop.commanded_intensity, 0.8)
        self.assertTrue(saturated_drop.saturated)

    def test_force_command_decrease_has_boundary_hysteresis(self):
        cfg = AppConfig(
            ball_mass_kg=0.0005,
            capture_time_sec=0.12,
            capture_force_safety_factor=1.0,
            capture_max_intensity_ratio=0.8,
            capture_intensity_levels=(0.5, 0.6, 0.7, 0.8),
            capture_loadcell_model=(
                (0.5, 4.0),
                (0.6, 5.0),
                (0.7, 6.0),
                (0.8, 7.0),
            ),
        )

        near_boundary = capture_force_command_for_velocity(
            cfg,
            -95.0,
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

    def test_force_interpolation_returns_continuous_intensity(self):
        cfg = AppConfig(
            capture_max_intensity_ratio=0.8,
            capture_loadcell_model=(
                (0.5, 4.0),
                (0.6, 5.0),
                (0.7, 6.0),
                (0.8, 7.0),
            ),
        )

        command = capture_intensity_interpolated_from_force(cfg, 5.5)

        self.assertAlmostEqual(command.commanded_intensity, 0.65)
        self.assertFalse(command.saturated)

    def test_interpolated_velocity_command_keeps_static_floor(self):
        cfg = AppConfig(
            ball_mass_kg=0.0005,
            capture_time_sec=0.12,
            capture_force_safety_factor=1.0,
            capture_max_intensity_ratio=0.8,
            capture_loadcell_model=(
                (0.5, 4.0),
                (0.6, 5.0),
                (0.7, 6.0),
                (0.8, 7.0),
            ),
        )

        hold = capture_force_command_for_velocity(
            cfg,
            0.0,
            interpolate_intensity=True,
            minimum_intensity=0.6,
        )
        falling = capture_force_command_for_velocity(
            cfg,
            -150.0,
            interpolate_intensity=True,
            minimum_intensity=0.6,
        )

        self.assertEqual(hold.commanded_intensity, 0.6)
        expected = 0.6 + (
            falling.required_force_mN - 5.0
        ) / (6.0 - 5.0) * 0.1
        self.assertAlmostEqual(falling.commanded_intensity, expected)

    def test_interpolated_command_saturates_at_configured_maximum(self):
        cfg = AppConfig(
            ball_mass_kg=0.0005,
            capture_time_sec=0.12,
            capture_force_safety_factor=1.0,
            capture_max_intensity_ratio=0.8,
            capture_loadcell_model=(
                (0.5, 4.0),
                (0.6, 5.0),
                (0.7, 6.0),
                (0.8, 7.0),
            ),
        )

        command = capture_force_command_for_velocity(
            cfg,
            -600.0,
            interpolate_intensity=True,
            minimum_intensity=0.6,
        )

        self.assertEqual(command.commanded_intensity, 0.8)
        self.assertTrue(command.saturated)

    def test_interpolated_command_ignores_sub_deadband_change(self):
        cfg = AppConfig(
            ball_mass_kg=0.0005,
            capture_time_sec=0.12,
            capture_force_safety_factor=1.0,
            capture_max_intensity_ratio=0.8,
            capture_loadcell_model=(
                (0.5, 4.0),
                (0.6, 5.0),
                (0.7, 6.0),
                (0.8, 7.0),
            ),
        )

        command = capture_force_command_for_velocity(
            cfg,
            -150.0,
            current_intensity=0.62,
            interpolate_intensity=True,
            minimum_intensity=0.6,
            intensity_deadband=0.005,
        )

        self.assertEqual(command.commanded_intensity, 0.62)

    def test_intensity_slew_rises_immediately_and_falls_slowly(self):
        cfg = AppConfig(capture_intensity_down_slew_per_sec=0.4)

        self.assertEqual(apply_capture_intensity_slew(0.8, 1.0, cfg, 0.01), 1.0)
        self.assertAlmostEqual(
            apply_capture_intensity_slew(1.0, 0.8, cfg, 0.1),
            0.96,
            places=6,
        )
if __name__ == "__main__":
    unittest.main()
