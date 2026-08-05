import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from manual_release_tracking.core.auto_release_capture import (
    AutoReleaseCaptureLogger,
    AutoReleaseDelayLogger,
    AutoReleaseTrajectoryLogger,
    MotionEstimate3D,
    PredictedCapture,
    SETTLE_INTENSITY_DOWNWARD_RESCUE,
    SETTLE_INTENSITY_NORMAL,
    SETTLE_INTENSITY_UPWARD_DAMPING,
    StereoMotionEstimator,
    capture_align_target_xy,
    capture_intensity_for_vz,
    braking_reference_at,
    limit_upward_capture_target_z,
    make_braking_plan,
    predict_capture_position,
    should_start_return_home,
    trajectory_target_z,
    update_brake_exit_stability,
    update_capture_stability,
    update_settle_intensity,
    xy_braking_reference_at,
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

    def test_upward_velocity_immediately_uses_point_five(self):
        self.assertEqual(capture_intensity_for_vz(20.0), 0.6)
        self.assertEqual(capture_intensity_for_vz(20.1), 0.5)
        self.assertEqual(capture_intensity_for_vz(200.0), 0.5)

    def test_settle_upward_damping_has_hysteresis(self):
        state, ratio = update_settle_intensity(
            SETTLE_INTENSITY_NORMAL,
            61.0,
        )
        self.assertEqual(state, SETTLE_INTENSITY_UPWARD_DAMPING)
        self.assertEqual(ratio, 0.5)

        state, ratio = update_settle_intensity(state, 30.0)
        self.assertEqual(state, SETTLE_INTENSITY_UPWARD_DAMPING)
        self.assertEqual(ratio, 0.5)

        state, ratio = update_settle_intensity(state, 19.0)
        self.assertEqual(state, SETTLE_INTENSITY_NORMAL)
        self.assertEqual(ratio, 0.6)

    def test_settle_downward_rescue_has_hysteresis(self):
        state, ratio = update_settle_intensity(
            SETTLE_INTENSITY_NORMAL,
            -81.0,
        )
        self.assertEqual(state, SETTLE_INTENSITY_DOWNWARD_RESCUE)
        self.assertEqual(ratio, 0.7)

        state, ratio = update_settle_intensity(state, -50.0)
        self.assertEqual(state, SETTLE_INTENSITY_DOWNWARD_RESCUE)
        self.assertEqual(ratio, 0.7)

        state, ratio = update_settle_intensity(state, -29.0)
        self.assertEqual(state, SETTLE_INTENSITY_NORMAL)
        self.assertEqual(ratio, 0.6)

    def test_settle_normal_does_not_switch_for_small_velocity_noise(self):
        for vz in (59.0, 30.0, 0.0, -30.0, -79.0):
            state, ratio = update_settle_intensity(
                SETTLE_INTENSITY_NORMAL,
                vz,
            )
            self.assertEqual(state, SETTLE_INTENSITY_NORMAL)
            self.assertEqual(ratio, 0.6)

    def test_automatic_local_hold_returns_after_configured_delay(self):
        self.assertFalse(
            should_start_return_home(
                now_sec=12.999,
                local_hold_started_sec=10.0,
                delay_sec=3.0,
                automatic_hold=True,
            )
        )
        self.assertTrue(
            should_start_return_home(
                now_sec=13.0,
                local_hold_started_sec=10.0,
                delay_sec=3.0,
                automatic_hold=True,
            )
        )

    def test_manual_or_unstarted_local_hold_does_not_auto_return(self):
        self.assertFalse(
            should_start_return_home(
                now_sec=20.0,
                local_hold_started_sec=10.0,
                delay_sec=3.0,
                automatic_hold=False,
            )
        )
        self.assertFalse(
            should_start_return_home(
                now_sec=20.0,
                local_hold_started_sec=None,
                delay_sec=3.0,
                automatic_hold=True,
            )
        )

    def test_upward_brake_prevents_target_z_from_increasing(self):
        self.assertEqual(
            limit_upward_capture_target_z(
                410.0,
                400.0,
                upward_brake_active=True,
            ),
            400.0,
        )
        self.assertEqual(
            limit_upward_capture_target_z(
                390.0,
                400.0,
                upward_brake_active=True,
            ),
            390.0,
        )
        self.assertEqual(
            limit_upward_capture_target_z(
                410.0,
                400.0,
                upward_brake_active=False,
            ),
            410.0,
        )

    def test_capture_align_keeps_initial_predicted_xy(self):
        initial = predict_capture_position(
            MotionEstimate3D(
                1.0,
                100.0,
                200.0,
                400.0,
                10.0,
                -20.0,
                0.0,
            ),
            now_sec=1.0,
            actuation_delay_sec=0.01,
            gravity_mm_s2=9800.0,
        )
        current = predict_capture_position(
            MotionEstimate3D(
                1.1,
                140.0,
                250.0,
                400.0,
                200.0,
                300.0,
                0.0,
            ),
            now_sec=1.1,
            actuation_delay_sec=0.01,
            gravity_mm_s2=9800.0,
        )

        self.assertEqual(
            capture_align_target_xy(initial, current),
            (initial.x_mm, initial.y_mm),
        )
        self.assertEqual(
            capture_align_target_xy(None, current),
            (current.x_mm, current.y_mm),
        )

    def test_xy_braking_reference_reduces_velocity_to_zero_in_120_ms(self):
        initial = PredictedCapture(
            horizon_sec=0.013,
            x_mm=100.0,
            y_mm=200.0,
            z_mm=400.0,
            vx_mm_s=200.0,
            vy_mm_s=-100.0,
            vz_mm_s=-50.0,
        )

        halfway = xy_braking_reference_at(
            initial,
            start_time_sec=1.0,
            now_sec=1.06,
            duration_sec=0.12,
        )
        stopped = xy_braking_reference_at(
            initial,
            start_time_sec=1.0,
            now_sec=1.12,
            duration_sec=0.12,
        )

        self.assertAlmostEqual(halfway.x_mm, 109.0)
        self.assertAlmostEqual(halfway.y_mm, 195.5)
        self.assertAlmostEqual(halfway.vx_mm_s, 100.0)
        self.assertAlmostEqual(halfway.vy_mm_s, -50.0)
        self.assertFalse(halfway.complete)
        self.assertAlmostEqual(stopped.x_mm, 112.0)
        self.assertAlmostEqual(stopped.y_mm, 194.0)
        self.assertEqual(stopped.vx_mm_s, 0.0)
        self.assertEqual(stopped.vy_mm_s, 0.0)
        self.assertTrue(stopped.complete)

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

    def test_braking_plan_expands_duration_to_available_force(self):
        plan = make_braking_plan(
            start_time_sec=1.0,
            initial_z_mm=400.0,
            initial_vz_mm_s=-500.0,
            nominal_duration_sec=0.12,
            maximum_duration_sec=1.0,
            minimum_duration_sec=0.05,
            ball_mass_kg=0.0005,
            gravity_mm_s2=9806.65,
            maximum_force_mN=6.0,
            z_min_mm=250.0,
            workspace_margin_mm=10.0,
        )

        self.assertTrue(plan.feasible)
        self.assertAlmostEqual(plan.duration_sec, 0.22796, places=4)
        self.assertAlmostEqual(plan.required_force_mN, 6.0, places=6)
        self.assertAlmostEqual(plan.stopping_distance_mm, 56.99, places=2)

    def test_braking_plan_marks_insufficient_workspace(self):
        plan = make_braking_plan(
            start_time_sec=1.0,
            initial_z_mm=280.0,
            initial_vz_mm_s=-500.0,
            nominal_duration_sec=0.12,
            maximum_duration_sec=1.0,
            minimum_duration_sec=0.05,
            ball_mass_kg=0.0005,
            gravity_mm_s2=9806.65,
            maximum_force_mN=6.0,
            z_min_mm=250.0,
            workspace_margin_mm=10.0,
        )

        self.assertFalse(plan.feasible)
        self.assertTrue(plan.workspace_limited)
        self.assertTrue(plan.force_saturated)
        self.assertAlmostEqual(plan.duration_sec, 0.08)
        self.assertAlmostEqual(plan.stop_z_mm, 260.0)

    def test_braking_reference_reaches_zero_velocity_at_stop(self):
        plan = make_braking_plan(
            start_time_sec=1.0,
            initial_z_mm=400.0,
            initial_vz_mm_s=-100.0,
            nominal_duration_sec=0.12,
            maximum_duration_sec=1.0,
            minimum_duration_sec=0.05,
            ball_mass_kg=0.0005,
            gravity_mm_s2=9806.65,
            maximum_force_mN=6.0,
            z_min_mm=250.0,
            workspace_margin_mm=10.0,
        )
        halfway = braking_reference_at(
            plan,
            now_sec=1.0 + plan.duration_sec / 2.0,
        )
        stopped = braking_reference_at(
            plan,
            now_sec=1.0 + plan.duration_sec,
        )

        self.assertAlmostEqual(halfway.vz_mm_s, -50.0)
        self.assertTrue(stopped.complete)
        self.assertAlmostEqual(stopped.vz_mm_s, 0.0)
        self.assertAlmostEqual(stopped.z_mm, plan.stop_z_mm)

    def test_trajectory_target_correction_is_bounded(self):
        target_z, correction = trajectory_target_z(
            reference_z_mm=390.0,
            reference_vz_mm_s=-100.0,
            measured_z_mm=370.0,
            measured_vz_mm_s=-500.0,
            position_gain=0.3,
            velocity_gain=0.01,
            maximum_correction_mm=5.0,
        )

        self.assertEqual(correction, 5.0)
        self.assertEqual(target_z, 395.0)

    def test_local_hold_rejects_unsettled_xy_or_incomplete_trajectory(self):
        since, ready = update_capture_stability(
            now_sec=1.0,
            vz_mm_s=0.0,
            release_confirmed=True,
            stable_since_sec=None,
            maximum_abs_vz_mm_s=30.0,
            required_duration_sec=0.05,
            trajectory_complete=False,
            vxy_mm_s=0.0,
            maximum_vxy_mm_s=30.0,
        )
        self.assertIsNone(since)
        self.assertFalse(ready)

        since, ready = update_capture_stability(
            now_sec=1.0,
            vz_mm_s=0.0,
            release_confirmed=True,
            stable_since_sec=None,
            maximum_abs_vz_mm_s=30.0,
            required_duration_sec=0.05,
            trajectory_complete=True,
            vxy_mm_s=70.0,
            maximum_vxy_mm_s=30.0,
        )
        self.assertIsNone(since)
        self.assertFalse(ready)

    def test_brake_exit_requires_confirmed_sustained_near_zero_vz(self):
        since, ready = update_brake_exit_stability(
            now_sec=1.0,
            vz_mm_s=-20.0,
            release_confirmed=False,
            stable_since_sec=None,
            minimum_vz_mm_s=-30.0,
            required_duration_sec=0.015,
        )
        self.assertIsNone(since)
        self.assertFalse(ready)

        since, ready = update_brake_exit_stability(
            now_sec=1.0,
            vz_mm_s=-20.0,
            release_confirmed=True,
            stable_since_sec=None,
            minimum_vz_mm_s=-30.0,
            required_duration_sec=0.015,
        )
        self.assertEqual(since, 1.0)
        self.assertFalse(ready)

        since, ready = update_brake_exit_stability(
            now_sec=1.016,
            vz_mm_s=40.0,
            release_confirmed=True,
            stable_since_sec=since,
            minimum_vz_mm_s=-30.0,
            required_duration_sec=0.015,
        )
        self.assertTrue(ready)

    def test_brake_exit_timer_resets_when_descent_accelerates_again(self):
        since, ready = update_brake_exit_stability(
            now_sec=1.010,
            vz_mm_s=-80.0,
            release_confirmed=True,
            stable_since_sec=1.0,
            minimum_vz_mm_s=-30.0,
            required_duration_sec=0.015,
        )
        self.assertIsNone(since)
        self.assertFalse(ready)

    def test_local_hold_rejects_non_normal_settle_intensity(self):
        since, ready = update_capture_stability(
            now_sec=1.0,
            vz_mm_s=0.0,
            release_confirmed=True,
            stable_since_sec=None,
            maximum_abs_vz_mm_s=60.0,
            required_duration_sec=0.025,
            vxy_mm_s=0.0,
            maximum_vxy_mm_s=60.0,
            xy_distance_mm=0.0,
            maximum_xy_distance_mm=15.0,
            z_tracking_error_mm=0.0,
            maximum_z_tracking_error_mm=10.0,
            intensity_stable=False,
        )
        self.assertIsNone(since)
        self.assertFalse(ready)


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

    def test_writes_decomposed_effective_delay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delay.csv"
            logger = AutoReleaseDelayLogger(str(path))
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
            sent = SimpleNamespace(
                sequence=3,
                dequeued_time_sec=1.007,
                send_started_time_sec=1.009,
                sent_time_sec=1.012,
                target=SimpleNamespace(
                    x=10.0,
                    y=20.0,
                    z=399.0,
                    intensity_ratio=0.7,
                    enqueued_time_sec=1.005,
                ),
            )

            logger.write(
                release_timestamp_ms=990,
                estimate=estimate,
                prediction=prediction,
                sent=sent,
                candidate_command_seq=3,
                configured_prediction_delay_sec=0.01,
            )

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["release_to_enqueue_ms"], "15.000")
            self.assertEqual(
                rows[0]["enqueue_to_send_complete_ms"],
                "7.000",
            )
            self.assertEqual(rows[0]["enqueue_to_dequeue_ms"], "2.000")
            self.assertEqual(rows[0]["dequeue_to_send_start_ms"], "2.000")
            self.assertEqual(rows[0]["autd_send_duration_ms"], "3.000")
            self.assertEqual(
                rows[0]["measurement_to_send_complete_ms"],
                "12.000",
            )
            self.assertEqual(rows[0]["prediction_shortfall_ms"], "2.000")
            self.assertEqual(rows[0]["command_superseded"], "0")

    def test_writes_trajectory_reference_and_tracking_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            logger = AutoReleaseTrajectoryLogger(str(path))
            plan = make_braking_plan(
                start_time_sec=1.0,
                initial_z_mm=400.0,
                initial_vz_mm_s=-100.0,
                nominal_duration_sec=0.12,
                maximum_duration_sec=1.0,
                minimum_duration_sec=0.05,
                ball_mass_kg=0.0005,
                gravity_mm_s2=9806.65,
                maximum_force_mN=6.0,
                z_min_mm=250.0,
                workspace_margin_mm=10.0,
            )
            reference = braking_reference_at(plan, now_sec=1.06)
            estimate = MotionEstimate3D(
                measurement_time_sec=1.06,
                x_mm=10.0,
                y_mm=20.0,
                z_mm=395.0,
                vx_mm_s=1.0,
                vy_mm_s=2.0,
                vz_mm_s=-40.0,
            )

            logger.write(
                release_timestamp_ms=990,
                estimate=estimate,
                plan=plan,
                reference=reference,
                target=Target3D(10.0, 20.0, 396.0),
                target_z_correction_mm=1.0,
                xy_distance_mm=2.0,
                commanded_intensity=0.7,
                upward_brake_active=False,
            )
            logger.close()

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["elapsed_from_field_ms"], "60.000")
            self.assertEqual(rows[0]["commanded_intensity"], "0.700")
            self.assertEqual(rows[0]["upward_brake_active"], "0")


if __name__ == "__main__":
    unittest.main()
