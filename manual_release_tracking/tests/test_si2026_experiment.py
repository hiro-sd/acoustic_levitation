import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from manual_release_tracking.analysis.analyze_si2026_trials import (
    add_centered_release_metrics,
    apply_analysis_exclusions,
    build_all_outputs,
    failure_reason_category,
    make_outcome_summary,
    resolve_release_center,
    wilson_interval,
)
from manual_release_tracking.analysis.manage_trial_exclusions import (
    read_rows,
    write_rows,
)
from manual_release_tracking.core.si2026_experiment_logging import (
    SI2026Condition,
    SI2026ExperimentLogger,
    release_hold_success_reached,
)


class SI2026ExperimentLoggerTest(unittest.TestCase):
    def test_success_requires_five_seconds_field_and_active_hold(self):
        self.assertFalse(
            release_hold_success_reached(
                elapsed_from_release_sec=4.999,
                required_hold_sec=5.0,
                field_started=True,
                automatic_hold_active=True,
            )
        )
        self.assertTrue(
            release_hold_success_reached(
                elapsed_from_release_sec=5.0,
                required_hold_sec=5.0,
                field_started=True,
                automatic_hold_active=True,
            )
        )
        self.assertFalse(
            release_hold_success_reached(
                elapsed_from_release_sec=5.0,
                required_hold_sec=5.0,
                field_started=False,
                automatic_hold_active=True,
            )
        )

    def make_config(self):
        return SimpleNamespace(
            ball_mass_kg=0.0005,
            ball_diameter_mm=38.0,
            intrinsic_xy_npz="",
            intrinsic_z_npz="",
            stereo_npz="",
            stereo_camera_to_autd_npz="",
            auto_release_gate_max_vz_mm_s=-100.0,
            auto_release_gate_min_drop_mm=2.5,
            auto_release_gate_consecutive_estimates=3,
            si2026_success_hold_sec=5.0,
            si2026_observation_sec=10.0,
        )

    def test_successful_trial_writes_timeseries_summary_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            logger = SI2026ExperimentLogger(
                output,
                config=self.make_config(),
                condition=SI2026Condition(
                    condition_id="delay_000ms",
                    operator="tester",
                    planned_release_x_mm=100.0,
                    planned_release_y_mm=200.0,
                    planned_release_z_mm=400.0,
                ),
                project_root=PROJECT_ROOT,
                session_id="test_session",
            )
            logger.set_reference_center(0.0, 0.0, 400.0)
            logger.start_trial(release_timestamp_ms=1000)
            logger.mark_motion_ready(1.020)
            gate_estimate = SimpleNamespace(vz_mm_s=-180.0)
            logger.update_release_gate_observation(
                origin_z_mm=400.0,
                estimate=gate_estimate,
                downward_displacement_mm=3.0,
                evidence_seen=True,
            )
            logger.mark_release_gate_passed(
                event_time_sec=1.030,
                estimate=gate_estimate,
                downward_displacement_mm=3.0,
                consecutive_count=3,
            )
            prediction = SimpleNamespace(
                x_mm=100.0,
                y_mm=200.0,
                z_mm=390.0,
                vz_mm_s=-300.0,
            )
            plan = SimpleNamespace(
                duration_sec=0.120,
                stop_z_mm=372.0,
                stopping_distance_mm=18.0,
                required_force_mN=6.15,
                maximum_force_mN=7.0,
                feasible=True,
                force_saturated=False,
                workspace_limited=False,
            )
            logger.mark_field_sent(
                field_time_sec=1.050,
                prediction=prediction,
                anchor=(100.0, 200.0),
                plan=plan,
            )

            target = SimpleNamespace(x=100.0, y=200.0, z=389.0)
            setpoint = SimpleNamespace(x=100.0, y=200.0, z=375.0)
            for index in range(4):
                measurement_time = 1.050 + index * 0.010
                estimate = SimpleNamespace(
                    measurement_time_sec=measurement_time,
                    x_mm=100.0 + index * 0.1,
                    y_mm=200.0,
                    z_mm=389.5 - index * 2.0,
                    vx_mm_s=10.0,
                    vy_mm_s=0.0,
                    vz_mm_s=-250.0 + index * 80.0,
                    velocity_ready=True,
                )
                logger.record_sample(
                    sample_time_sec=measurement_time + 0.002,
                    frame_xy_time_sec=measurement_time,
                    frame_z_time_sec=measurement_time - 0.001,
                    frame_sync_skew_sec=0.001,
                    detected_xy=True,
                    detected_z=True,
                    raw_x_mm=estimate.x_mm,
                    raw_y_mm=estimate.y_mm,
                    raw_z_mm=estimate.z_mm,
                    estimate=estimate,
                    control_mode=("CAPTURE_ALIGN" if index < 2 else "LOCAL_HOLD"),
                    reference=SimpleNamespace(
                        z_mm=390.0 - index * 2.0,
                        vz_mm_s=-300.0 + index * 100.0,
                    ),
                    anchor=(100.0, 200.0),
                    setpoint=setpoint,
                    target=target,
                    commanded_intensity=0.7,
                    xy_distance_to_target_mm=0.5,
                    required_force_mN=6.15,
                    force_saturated=False,
                    plan=plan,
                )
            logger.mark_local_hold_reached(1.070)
            logger.mark_braking_end(
                1.080,
                SimpleNamespace(
                    x_mm=100.0,
                    y_mm=200.0,
                    z_mm=373.0,
                    vx_mm_s=5.0,
                    vy_mm_s=0.0,
                    vz_mm_s=-20.0,
                ),
                "braking_reference_complete",
            )
            logger.mark_success_reached(6.000)
            logger.finish_trial(
                status="success",
                reason="post_success_observation_completed",
                event_time_sec=11.000,
            )

            logger.close()

            self.assertTrue((output / "session_test_session.json").is_file())
            with (output / "trial_timeseries.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                timeseries_rows = list(csv.DictReader(file))
            self.assertEqual(len(timeseries_rows), 4)
            self.assertNotIn(None, timeseries_rows[0])
            self.assertNotIn(None, timeseries_rows[0].values())
            self.assertEqual(timeseries_rows[0]["release_gate_passed"], "0")
            self.assertEqual(timeseries_rows[0]["success_criterion_reached"], "0")
            with (output / "trial_summary.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertNotIn(None, rows[0])
            self.assertNotIn(None, rows[0].values())
            self.assertEqual(rows[0]["status"], "success")
            self.assertEqual(rows[0]["condition_id"], "delay_000ms")
            self.assertEqual(rows[0]["sample_count"], "4")
            self.assertAlmostEqual(float(rows[0]["release_to_field_ms"]), 50.0)
            self.assertAlmostEqual(float(rows[0]["retained_duration_ms"]), 10000.0)
            self.assertAlmostEqual(float(rows[0]["release_x_mm"]), 100.0)
            self.assertAlmostEqual(float(rows[0]["autd_center_z_mm"]), 400.0)
            self.assertAlmostEqual(float(rows[0]["success_reached_elapsed_ms"]), 5000.0)
            self.assertAlmostEqual(float(rows[0]["actual_stopping_distance_mm"]), 16.5)

    def test_close_marks_active_trial_aborted(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            logger = SI2026ExperimentLogger(
                output,
                config=self.make_config(),
                project_root=PROJECT_ROOT,
                session_id="aborted_session",
            )
            logger.start_trial(release_timestamp_ms=1000)
            logger.close()
            summary = (output / "trial_summary.csv").read_text(encoding="utf-8")
            self.assertIn("aborted", summary)

    def test_reject_key_can_exclude_a_just_finished_false_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            logger = SI2026ExperimentLogger(
                output,
                config=self.make_config(),
                project_root=PROJECT_ROOT,
                session_id="reject_session",
            )
            logger.start_trial(release_timestamp_ms=1000)
            logger.mark_field_sent(
                field_time_sec=1.050,
                prediction=None,
                anchor=None,
                plan=None,
            )
            logger.finish_trial(
                status="failure",
                reason="regrasp_detected_during_auto_hold",
                event_time_sec=2.000,
            )
            trial_id = logger.exclude_recent_failed_trial(
                reason="operator_rejected_false_release_after_automatic_stop",
                now_sec=2.500,
            )
            self.assertIsNotNone(trial_id)
            exclusions = pd.read_csv(output / "trial_exclusions.csv")
            self.assertEqual(exclusions.iloc[0]["trial_id"], trial_id)
            self.assertIn("operator_rejected", exclusions.iloc[0]["reason"])
            logger.close()


class SI2026AnalysisTest(unittest.TestCase):
    def test_centered_release_metrics_and_outcome_summary(self):
        trials = pd.DataFrame(
            {
                "planned_release_x_mm": [100.0, 100.0, 100.0],
                "planned_release_y_mm": [200.0, 200.0, 200.0],
                "planned_release_z_mm": [400.0, 400.0, 400.0],
                "release_x_mm": [100.0, 103.0, 99.0],
                "release_y_mm": [200.0, 204.0, 200.0],
                "release_z_mm": [400.0, 390.0, 401.0],
                "onset_vx_mm_s": [0.0, 30.0, 0.0],
                "onset_vy_mm_s": [0.0, 40.0, 0.0],
                "capture_state_vz_mm_s": [-300.0, -500.0, -250.0],
                "status": ["success", "failure", "invalid"],
            }
        )
        center = resolve_release_center(trials)
        enriched = add_centered_release_metrics(trials, center)
        outcome = make_outcome_summary(enriched, center)
        self.assertEqual(center, (100.0, 200.0, 400.0))
        self.assertAlmostEqual(enriched.iloc[1]["release_xy_offset_mm"], 5.0)
        self.assertAlmostEqual(enriched.iloc[1]["capture_state_vxy_mm_s"], 50.0)
        self.assertEqual(int(outcome.iloc[0]["eligible_trials"]), 2)
        self.assertEqual(int(outcome.iloc[0]["successes"]), 1)
        self.assertAlmostEqual(outcome.iloc[0]["success_rate"], 0.5)
        self.assertEqual(int(outcome.iloc[0]["target_valid_trials"]), 100)
        self.assertEqual(int(outcome.iloc[0]["remaining_valid_trials"]), 98)

    def test_failure_reason_categories(self):
        self.assertEqual(
            failure_reason_category("capture_stereo_measurement_timeout"),
            "Tracking/estimation",
        )
        self.assertEqual(
            failure_reason_category("regrasp_detected_during_auto_hold"),
            "Regrasp/contact",
        )

    def test_logged_autd_center_has_priority_over_release_median(self):
        trials = pd.DataFrame(
            {
                "autd_center_x_mm": [0.0],
                "autd_center_y_mm": [5.0],
                "autd_center_z_mm": [400.0],
                "planned_release_x_mm": [100.0],
                "planned_release_y_mm": [200.0],
                "planned_release_z_mm": [390.0],
                "release_x_mm": [101.0],
                "release_y_mm": [201.0],
                "release_z_mm": [391.0],
                "status": ["success"],
            }
        )
        self.assertEqual(resolve_release_center(trials), (0.0, 5.0, 400.0))

    def test_false_triggers_and_manual_exclusions_are_not_eligible(self):
        trials = pd.DataFrame(
            {
                "trial_id": ["normal", "legacy_regrasp", "real_regrasp", "manual"],
                "schema_version": [4, 3, 4, 4],
                "release_gate_passed_time_sec": [1.0, float("nan"), 2.0, 3.0],
                "status": ["success", "failure", "failure", "failure"],
                "reason": [
                    "post_success_observation_completed",
                    "regrasp_detected_during_auto_hold",
                    "regrasp_detected_during_auto_hold",
                    "capture_stereo_measurement_timeout",
                ],
            }
        )
        exclusions = pd.DataFrame(
            {"trial_id": ["manual"], "reason": ["operator_confirmed_false_trigger"]}
        )
        marked = apply_analysis_exclusions(trials, exclusions)
        self.assertEqual(
            marked["analysis_eligible"].tolist(),
            [True, False, True, False],
        )
        self.assertEqual(
            marked.loc[3, "analysis_exclusion_source"],
            "manual_file",
        )

    def test_exclusions_file_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "trial_exclusions.csv"
            rows = [{"trial_id": "session_T0003", "reason": "false_trigger"}]
            write_rows(path, rows)
            self.assertEqual(read_rows(path), rows)

    def test_wilson_interval_contains_observed_rate(self):
        lower, upper = wilson_interval(8, 10)
        self.assertLess(lower, 0.8)
        self.assertGreater(upper, 0.8)

    def test_analysis_generates_all_minimum_figures(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data = root / "data"
            figures = root / "figures"
            config = SimpleNamespace(
                ball_mass_kg=0.0005,
                ball_diameter_mm=38.0,
                intrinsic_xy_npz="",
                intrinsic_z_npz="",
                stereo_npz="",
                stereo_camera_to_autd_npz="",
                auto_release_gate_max_vz_mm_s=-100.0,
                auto_release_gate_min_drop_mm=2.5,
                auto_release_gate_consecutive_estimates=3,
                si2026_success_hold_sec=5.0,
                si2026_observation_sec=10.0,
            )
            logger = SI2026ExperimentLogger(
                data,
                config=config,
                project_root=PROJECT_ROOT,
                session_id="plot_session",
            )
            logger.start_trial(release_timestamp_ms=1000)
            prediction = SimpleNamespace(
                x_mm=100.0,
                y_mm=200.0,
                z_mm=390.0,
                vz_mm_s=-300.0,
            )
            plan = SimpleNamespace(
                duration_sec=0.120,
                stop_z_mm=372.0,
                stopping_distance_mm=18.0,
                required_force_mN=6.15,
                maximum_force_mN=7.0,
                feasible=True,
                force_saturated=False,
                workspace_limited=False,
            )
            logger.mark_field_sent(
                field_time_sec=1.050,
                prediction=prediction,
                anchor=(100.0, 200.0),
                plan=plan,
            )
            for index, mode in enumerate(
                ["CAPTURE_ALIGN", "CAPTURE_ALIGN", "CAPTURE_SETTLE", "LOCAL_HOLD"]
            ):
                timestamp = 1.040 + index * 0.020
                estimate = SimpleNamespace(
                    measurement_time_sec=timestamp,
                    x_mm=100.0,
                    y_mm=200.0,
                    z_mm=392.0 - index * 4.0,
                    vx_mm_s=0.0,
                    vy_mm_s=0.0,
                    vz_mm_s=-350.0 + index * 110.0,
                    velocity_ready=True,
                )
                logger.record_sample(
                    sample_time_sec=timestamp + 0.003,
                    frame_xy_time_sec=timestamp,
                    frame_z_time_sec=timestamp,
                    frame_sync_skew_sec=0.0,
                    detected_xy=True,
                    detected_z=True,
                    raw_x_mm=estimate.x_mm,
                    raw_y_mm=estimate.y_mm,
                    raw_z_mm=estimate.z_mm,
                    estimate=estimate,
                    control_mode=mode,
                    reference=SimpleNamespace(
                        z_mm=391.0 - index * 3.0,
                        vz_mm_s=-300.0 + index * 100.0,
                    ),
                    anchor=(100.0, 200.0),
                    setpoint=SimpleNamespace(x=100.0, y=200.0, z=380.0),
                    target=SimpleNamespace(x=100.0, y=200.0, z=390.0 - index * 3.0),
                    commanded_intensity=0.7,
                    xy_distance_to_target_mm=0.0,
                    required_force_mN=6.15,
                    force_saturated=False,
                    plan=plan,
                )
            logger.mark_local_hold_reached(1.100)
            logger.mark_braking_end(
                1.100,
                SimpleNamespace(
                    x_mm=100.0,
                    y_mm=200.0,
                    z_mm=379.0,
                    vx_mm_s=0.0,
                    vy_mm_s=0.0,
                    vz_mm_s=-20.0,
                ),
                "braking_reference_complete",
            )
            logger.mark_success_reached(6.000)
            logger.finish_trial(
                status="success",
                reason="post_success_observation_completed",
                event_time_sec=11.000,
            )

            logger.start_trial(release_timestamp_ms=12000)
            logger.mark_field_sent(
                field_time_sec=12.050,
                prediction=prediction,
                anchor=(100.0, 200.0),
                plan=plan,
            )
            for index in range(2):
                timestamp = 12.040 + index * 0.020
                estimate = SimpleNamespace(
                    measurement_time_sec=timestamp,
                    x_mm=108.0 + index,
                    y_mm=204.0,
                    z_mm=390.0 - index * 8.0,
                    vx_mm_s=45.0,
                    vy_mm_s=20.0,
                    vz_mm_s=-420.0,
                    velocity_ready=True,
                )
                logger.record_sample(
                    sample_time_sec=timestamp + 0.003,
                    frame_xy_time_sec=timestamp,
                    frame_z_time_sec=timestamp,
                    frame_sync_skew_sec=0.0,
                    detected_xy=True,
                    detected_z=True,
                    raw_x_mm=estimate.x_mm,
                    raw_y_mm=estimate.y_mm,
                    raw_z_mm=estimate.z_mm,
                    estimate=estimate,
                    control_mode="CAPTURE_ALIGN",
                    reference=SimpleNamespace(z_mm=390.0, vz_mm_s=-300.0),
                    anchor=(100.0, 200.0),
                    setpoint=SimpleNamespace(x=100.0, y=200.0, z=380.0),
                    target=SimpleNamespace(x=100.0, y=200.0, z=390.0),
                    commanded_intensity=0.8,
                    xy_distance_to_target_mm=8.0,
                    required_force_mN=7.0,
                    force_saturated=True,
                    plan=plan,
                )
            logger.finish_trial(
                status="failure",
                reason="capture_stereo_measurement_timeout",
                event_time_sec=12.500,
            )
            logger.close()

            representative, trials, outcome = build_all_outputs(
                data,
                figures,
            )
            self.assertTrue(representative.startswith("plot_session"))
            self.assertEqual(len(trials), 2)
            self.assertEqual(len(outcome), 2)
            self.assertEqual(outcome.iloc[0]["scope"], "all")
            self.assertEqual(int(outcome.iloc[0]["failures"]), 1)
            for filename in (
                "successful_release_trajectory.png",
                "successful_trials_mean_response.png",
                "successful_trials_mean_response.csv",
                "failed_trials_mean_response.png",
                "failed_trials_mean_response.csv",
                "failed_trial_metric_summary.csv",
                "outcome_trajectory_comparison.png",
                "release_outcomes_3d.png",
                "capture_success_vs_vxy.png",
                "capture_success_vs_vxy_summary.csv",
                "capture_success_vs_vxy_bins.csv",
                "capture_success_vs_vxy_logistic.csv",
                "capture_success_vs_vxy_predictions.csv",
                "success_failure_comparison.png",
                "success_failure_effect_sizes.png",
                "failure_reason_counts.png",
                "control_mechanism_validation.png",
                "braking_distance_validation.png",
                "behavior_pattern_counts.png",
                "outcome_summary.csv",
                "failure_reason_summary.csv",
                "release_range_summary.csv",
                "behavior_pattern_summary.csv",
                "braking_validation.csv",
                "exclusion_summary.csv",
                "success_failure_comparison.csv",
            ):
                self.assertTrue((figures / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
