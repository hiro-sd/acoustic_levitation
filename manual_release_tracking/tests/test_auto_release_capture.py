import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from manual_release_tracking.core.auto_release_capture import (
    AutoReleaseCaptureLogger,
    MotionEstimate3D,
    StereoMotionEstimator,
    capture_intensity_for_vz,
    predict_capture_position,
    update_capture_stability,
)
from tracking.core.models import Target3D


class StereoMotionEstimatorTest(unittest.TestCase):
    def test_estimates_filtered_downward_velocity_before_control(self):
        estimator = StereoMotionEstimator(
            position_current_weight=1.0,
            velocity_current_weight=1.0,
        )
        first = estimator.update(1.00, 10.0, 20.0, 400.0)
        second = estimator.update(1.01, 10.0, 20.0, 399.0)

        self.assertEqual(first.vz_mm_s, 0.0)
        self.assertAlmostEqual(second.vz_mm_s, -100.0)

    def test_invalid_measurement_keeps_last_estimate(self):
        estimator = StereoMotionEstimator()
        first = estimator.update(1.0, 10.0, 20.0, 400.0)
        self.assertIs(estimator.update(1.01, None, 20.0, 400.0), first)


class CapturePredictionTest(unittest.TestCase):
    def test_prediction_includes_measurement_age_and_ten_ms_actuation(self):
        estimate = MotionEstimate3D(
            measurement_time_sec=1.000,
            x_mm=100.0,
            y_mm=200.0,
            z_mm=400.0,
            vx_mm_s=10.0,
            vy_mm_s=-20.0,
            vz_mm_s=-100.0,
        )
        prediction = predict_capture_position(
            estimate,
            now_sec=1.005,
            actuation_delay_sec=0.010,
            gravity_mm_s2=9800.0,
        )

        self.assertAlmostEqual(prediction.horizon_sec, 0.015)
        self.assertAlmostEqual(prediction.x_mm, 100.15)
        self.assertAlmostEqual(prediction.y_mm, 199.70)
        self.assertAlmostEqual(
            prediction.z_mm,
            400.0 - 100.0 * 0.015 - 0.5 * 9800.0 * 0.015**2,
        )

    def test_velocity_schedule_is_capped_at_point_eight(self):
        self.assertEqual(capture_intensity_for_vz(10.0), 0.6)
        self.assertEqual(capture_intensity_for_vz(-50.0), 0.7)
        self.assertEqual(capture_intensity_for_vz(-150.0), 0.8)

    def test_local_hold_requires_confirmed_stable_velocity_duration(self):
        since, ready = update_capture_stability(
            now_sec=1.0,
            vz_mm_s=10.0,
            release_confirmed=False,
            stable_since_sec=None,
            maximum_abs_vz_mm_s=30.0,
            required_duration_sec=0.05,
        )
        self.assertIsNone(since)
        self.assertFalse(ready)

        since, ready = update_capture_stability(
            now_sec=1.0,
            vz_mm_s=10.0,
            release_confirmed=True,
            stable_since_sec=None,
            maximum_abs_vz_mm_s=30.0,
            required_duration_sec=0.05,
        )
        self.assertEqual(since, 1.0)
        self.assertFalse(ready)

        since, ready = update_capture_stability(
            now_sec=1.05,
            vz_mm_s=10.0,
            release_confirmed=True,
            stable_since_sec=since,
            maximum_abs_vz_mm_s=30.0,
            required_duration_sec=0.05,
        )
        self.assertTrue(ready)


class AutoReleaseCaptureLoggerTest(unittest.TestCase):
    def test_writes_release_timing_and_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.csv"
            logger = AutoReleaseCaptureLogger(str(path))
            estimate = MotionEstimate3D(
                1.0,
                10.0,
                20.0,
                400.0,
                1.0,
                2.0,
                -30.0,
            )
            prediction = predict_capture_position(
                estimate,
                now_sec=1.0,
                actuation_delay_sec=0.01,
                gravity_mm_s2=9800.0,
            )
            logger.write(
                event="capture_field_sent",
                release_timestamp_ms=990,
                event_time_sec=1.010,
                estimate=estimate,
                prediction=prediction,
                target=Target3D(10.0, 20.0, 399.0),
                intensity=0.7,
                command_seq=3,
            )

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event"], "capture_field_sent")
            self.assertEqual(rows[0]["elapsed_from_release_ms"], "20.000")
            self.assertEqual(rows[0]["intensity"], "0.700")
            self.assertEqual(rows[0]["command_seq"], "3")


if __name__ == "__main__":
    unittest.main()
