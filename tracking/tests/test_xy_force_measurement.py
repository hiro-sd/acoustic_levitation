import unittest

from core.xy_force_measurement import (
    PulseSample,
    WindowedMotionEstimator3D,
    estimate_pulse_acceleration,
    pulse_safety_reason,
)


class XYForceMeasurementTests(unittest.TestCase):
    def test_windowed_estimator_recovers_velocity(self):
        estimator = WindowedMotionEstimator3D(
            window_sec=0.060,
            minimum_samples=3,
            minimum_span_sec=0.010,
        )
        estimate = None
        for index in range(5):
            t = index * 0.005
            estimate = estimator.update(
                20.0 + t,
                100.0 + 30.0 * t,
                200.0 - 40.0 * t,
                400.0 + 10.0 * t,
            )

        self.assertTrue(estimate.velocity_ready)
        self.assertAlmostEqual(estimate.vx_mm_s, 30.0, places=5)
        self.assertAlmostEqual(estimate.vy_mm_s, -40.0, places=5)
        self.assertAlmostEqual(estimate.vz_mm_s, 10.0, places=5)

    def test_quadratic_fit_recovers_acceleration(self):
        samples = []
        for index in range(9):
            t = index * 0.01
            samples.append(
                PulseSample(
                    measurement_time_sec=10.0 + t,
                    x_mm=100.0 + 20.0 * t + 0.5 * 400.0 * t**2,
                    y_mm=200.0 - 10.0 * t + 0.5 * -200.0 * t**2,
                    z_mm=400.0 + 0.5 * 50.0 * t**2,
                )
            )

        estimate = estimate_pulse_acceleration(samples)

        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.ax_mm_s2, 400.0, places=5)
        self.assertAlmostEqual(estimate.ay_mm_s2, -200.0, places=5)
        self.assertAlmostEqual(estimate.az_mm_s2, 50.0, places=5)

    def test_safety_stops_on_xy_displacement(self):
        reason = pulse_safety_reason(
            measurement_valid=True,
            initial_xyz=(0.0, 0.0, 400.0),
            current_xyz=(10.1, 0.0, 400.0),
            vz_mm_s=0.0,
            maximum_xy_displacement_mm=10.0,
            maximum_z_drop_mm=10.0,
            maximum_abs_vz_mm_s=150.0,
        )

        self.assertEqual(reason, "xy_displacement_limit")

    def test_safety_stops_on_lost_measurement(self):
        reason = pulse_safety_reason(
            measurement_valid=False,
            initial_xyz=(0.0, 0.0, 400.0),
            current_xyz=None,
            vz_mm_s=None,
            maximum_xy_displacement_mm=10.0,
            maximum_z_drop_mm=10.0,
            maximum_abs_vz_mm_s=150.0,
        )

        self.assertEqual(reason, "stereo_measurement_lost")


if __name__ == "__main__":
    unittest.main()
