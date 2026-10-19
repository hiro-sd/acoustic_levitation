import math
import unittest

import numpy as np

from tracking.core.acoustic_force_measurement import (
    ForceStepSample,
    acoustic_force_mN,
    buoyancy_force_mN,
    estimate_position_resolution,
    estimate_step_acceleration,
    step_safety_reason,
)


def sample(t, x, y, z, radius=38.0):
    return ForceStepSample(
        measurement_time_sec=t,
        x_mm=x,
        y_mm=y,
        z_mm=z,
        u_xy_px=300.0,
        v_xy_px=220.0,
        radius_xy_px=radius,
        u_z_px=310.0,
        v_z_px=230.0,
        radius_z_px=radius,
        sync_skew_sec=0.0005,
    )


class AcousticForceMeasurementTests(unittest.TestCase):
    def test_quadratic_fit_uses_only_post_delay_samples(self):
        samples = []
        for index in range(15):
            t = index * 0.005
            # Deliberately corrupt the samples in the ignored first 10 ms.
            z = 999.0 if t < 0.010 else 400.0 + 20.0 * t + 0.5 * 800.0 * t**2
            samples.append(
                sample(
                    t,
                    10.0 + 0.5 * 100.0 * t**2,
                    20.0 + 0.5 * -200.0 * t**2,
                    z,
                )
            )

        estimate = estimate_step_acceleration(
            samples,
            field_started_time_sec=0.0,
            analysis_delay_sec=0.010,
            field_stopped_time_sec=0.070,
            minimum_samples=6,
            minimum_span_sec=0.025,
        )

        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.x.acceleration_mm_s2, 100.0, places=5)
        self.assertAlmostEqual(estimate.y.acceleration_mm_s2, -200.0, places=5)
        self.assertAlmostEqual(estimate.z.acceleration_mm_s2, 800.0, places=5)
        self.assertLess(estimate.z.residual_rms_mm, 1e-8)

    def test_buoyancy_and_static_force_are_consistent(self):
        buoyancy = buoyancy_force_mN(
            diameter_mm=38.0,
            air_density_kg_m3=1.204,
        )
        force = acoustic_force_mN(
            mass_kg=0.0005,
            acceleration_mm_s2=0.0,
            gravity_mm_s2=9806.65,
            buoyancy_mN=buoyancy,
        )

        self.assertAlmostEqual(buoyancy, 0.3394, places=3)
        self.assertAlmostEqual(force, 4.5639, places=3)

    def test_resolution_uses_observed_sphere_radius(self):
        samples = [
            sample(index * 0.005, index % 2, 2.0, 400.0, radius=38.0)
            for index in range(10)
        ]
        resolution = estimate_position_resolution(
            samples,
            sphere_diameter_mm=38.0,
        )

        self.assertIsNotNone(resolution)
        self.assertAlmostEqual(resolution.approximate_scale_xy_mm_per_px, 0.5)
        self.assertAlmostEqual(resolution.approximate_scale_z_mm_per_px, 0.5)
        self.assertGreater(resolution.single_difference_95_x_mm, 0.0)

    def test_safety_stops_for_upward_or_downward_z_motion(self):
        upward = step_safety_reason(
            measurement_valid=True,
            initial_xyz=(0.0, 0.0, 400.0),
            current_xyz=(0.0, 0.0, 416.0),
            speed_mm_s=0.0,
            maximum_xy_displacement_mm=15.0,
            maximum_abs_z_displacement_mm=15.0,
            maximum_speed_mm_s=500.0,
            z_min_mm=250.0,
            z_max_mm=550.0,
        )
        downward = step_safety_reason(
            measurement_valid=True,
            initial_xyz=(0.0, 0.0, 400.0),
            current_xyz=(0.0, 0.0, 384.0),
            speed_mm_s=0.0,
            maximum_xy_displacement_mm=15.0,
            maximum_abs_z_displacement_mm=15.0,
            maximum_speed_mm_s=500.0,
            z_min_mm=250.0,
            z_max_mm=550.0,
        )

        self.assertEqual(upward, "z_displacement_limit")
        self.assertEqual(downward, "z_displacement_limit")


if __name__ == "__main__":
    unittest.main()
