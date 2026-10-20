import unittest

from tracking.core.config import AppConfig
from tracking.core.controller import PredictionPIDController
from tracking.core.models import HomePosition, Measurement3D
from tracking.core.prediction_timing import prediction_timing_from_timestamps


class PredictionTimingTests(unittest.TestCase):
    def test_horizon_adds_measurement_age_and_measured_actuation_delay(self):
        timing = prediction_timing_from_timestamps(
            measurement_time_sec=10.000,
            now_sec=10.006,
            actuation_delay_sec=0.003,
        )

        self.assertAlmostEqual(timing.measurement_age_sec, 0.006)
        self.assertAlmostEqual(timing.actuation_delay_sec, 0.003)
        self.assertAlmostEqual(timing.horizon_sec, 0.009)

    def test_pid_uses_timestamp_dependent_horizon_when_supplied(self):
        cfg = AppConfig(
            z_lpf_alpha=0.0,
            enable_delay_compensation=True,
            use_gravity_prediction_z=False,
        )
        controller = PredictionPIDController(cfg)
        home = HomePosition(0.0, 0.0, 0.0)
        controller.update(
            Measurement3D(
                detected_xy=True,
                detected_z=True,
                x=0.0,
                y=0.0,
                z=0.0,
                t_xy=1.000,
                t_z=1.000,
            ),
            home,
            now_sec=1.000,
            actuation_delay_sec=0.0,
        )
        _, debug = controller.update(
            Measurement3D(
                detected_xy=True,
                detected_z=True,
                x=1.0,
                y=-2.0,
                z=3.0,
                t_xy=1.010,
                t_z=1.010,
            ),
            home,
            now_sec=1.016,
            actuation_delay_sec=0.004,
        )

        # Filtered derivative is half the raw 10 ms finite difference.
        self.assertAlmostEqual(debug.vx, 50.0)
        self.assertAlmostEqual(debug.vy, -100.0)
        self.assertAlmostEqual(debug.vz, 150.0)
        # Horizon = 6 ms measurement age + 4 ms actuation delay.
        self.assertAlmostEqual(debug.x_pred, 1.5)
        self.assertAlmostEqual(debug.y_pred, -3.0)
        self.assertAlmostEqual(debug.z_pred, 4.5)


if __name__ == "__main__":
    unittest.main()
