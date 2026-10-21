import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.config import AppConfig
from tracking.core.control.capture_force import SI2026_CAPTURE_FORCE_MODEL
from manual_release_tracking.core.mediapipe_release_trigger import (
    MediaPipeReleaseTrigger,
)
from manual_release_tracking.manual_release_app import run_manual_release_app


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Automatic release capture with optional SI2026 trial logging."
        )
    )
    parser.add_argument(
        "--condition-id",
        default="baseline",
        help="Condition label written to every trial (for example center_freehand_50).",
    )
    parser.add_argument(
        "--release-x-mm",
        type=float,
        default=None,
        help="Planned release X coordinate in the AUTD frame.",
    )
    parser.add_argument(
        "--release-y-mm",
        type=float,
        default=None,
        help="Planned release Y coordinate in the AUTD frame.",
    )
    parser.add_argument(
        "--release-z-mm",
        "--release-height-mm",
        dest="release_z_mm",
        type=float,
        default=None,
        help="Planned release Z coordinate (release-height-mm is an alias).",
    )
    parser.add_argument(
        "--extra-delay-ms",
        type=float,
        default=0.0,
        help=(
            "Additional field-off time after the post-release motion fit is ready. "
            "Default 0 preserves the existing controller behavior."
        ),
    )
    parser.add_argument("--operator", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--autd-interface",
        default="",
        help=(
            "AUTD-connected Windows adapter Name from Get-NetAdapter. "
            "Empty uses SDK automatic discovery."
        ),
    )
    parser.add_argument(
        "--success-hold-sec",
        type=float,
        default=5.0,
        help="A trial succeeds when non-contact hold continues this long after release.",
    )
    parser.add_argument(
        "--observation-sec",
        type=float,
        default=10.0,
        help=(
            "Continue logging a successful trial until this time after release. "
            "Must be at least --success-hold-sec."
        ),
    )
    parser.add_argument(
        "--target-valid-trials",
        type=int,
        default=100,
        help="Planned number of non-invalid physical release attempts.",
    )
    parser.add_argument(
        "--paper-data-dir",
        default="./manual_release_tracking/si2026_data",
    )
    parser.add_argument(
        "--no-paper-log",
        action="store_true",
        help="Disable the new trial-oriented logger.",
    )
    args = parser.parse_args()
    if args.extra_delay_ms < 0.0:
        parser.error("--extra-delay-ms must be non-negative")
    if args.success_hold_sec <= 0.0:
        parser.error("--success-hold-sec must be positive")
    if args.observation_sec < args.success_hold_sec:
        parser.error("--observation-sec must be at least --success-hold-sec")
    if args.target_valid_trials <= 0:
        parser.error("--target-valid-trials must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = AppConfig(
        radius=19.0,
        default_z=400.0,
        static_intensity_ratio=0.6,
        # Actual mass of the sphere used both for the 300-trial force
        # calibration and for the free-fall capture experiment.
        ball_mass_kg=0.00053,

        # Use calibrated stereo 3D position as the feedback measurement.
        enable_stereo_triangulation=True,
        use_stereo_position_for_control=True,
        stereo_camera_to_autd_npz="./tracking/calibration/stereo_camera_to_autd.npz",

        # First automatic-release experiment:
        # keep exactly the successful manual LOCAL_HOLD controller.
        enable_base_move=False,
        enable_radius_change=False,
        enable_auto_demo=False,
        use_gravity_prediction_z=False,

        log_csv_path="./manual_release_tracking/auto_release_log.csv",
    )
    cfg.autd_interface_name = str(args.autd_interface).strip()

    # MediaPipe Tasks Hand Landmarker settings used by automatic release detection.
    cfg.mediapipe_hand_model_path = (
        "./manual_release_tracking/models/hand_landmarker.task"
    )
    cfg.mediapipe_input_width_px = 640
    cfg.mediapipe_tip_gap_normalized_max = 0.45
    cfg.mediapipe_min_opposition_angle_deg = 60.0
    cfg.mediapipe_grasp_confirm_ms = 50.0
    cfg.mediapipe_release_confirm_ms = 15.0
    cfg.mediapipe_released_latch_ms = 500.0
    cfg.mediapipe_result_pair_tolerance_ms = 40.0
    cfg.mediapipe_submit_max_fps = 90.0
    # Z camera remains part of stereo 3D reconstruction, but its hand landmarks
    # are not reliable enough for the grasp/release decision on the current rig.
    cfg.mediapipe_auto_release_camera = "xy"

    # Reject a stale asynchronous release result. This is only a trigger-age
    # guard; the hold target itself uses the newest stereo measurement.
    cfg.mediapipe_auto_release_max_result_age_ms = 80.0

    # Release-specific capture. Position/velocity are estimated continuously,
    # including while the sphere is still held by the user.
    cfg.auto_release_position_current_weight = 0.55
    # Velocity is fitted from the recent multi-frame history. The regression
    # itself provides smoothing, so use its newest result directly rather than
    # adding another lagging IIR blend.
    cfg.auto_release_velocity_current_weight = 1.00
    cfg.auto_release_motion_max_gap_sec = 0.10
    cfg.auto_release_velocity_window_sec = 0.050
    cfg.auto_release_min_velocity_samples = 5
    cfg.auto_release_min_velocity_span_sec = 0.020
    cfg.auto_release_max_motion_age_sec = 0.050
    # Used only until this process has measured its first AUTD command latency.
    # Subsequent prediction horizons use measurement age plus the runtime
    # enqueue-to-send-completion latency estimate.
    cfg.auto_release_actuation_prediction_sec = 0.0075

    # The initial capture command no longer uses motion while the sphere is
    # held. The first separation resets a dedicated estimator, the field stays
    # off, and capture starts only after release confirmation plus at least three
    # post-release stereo frames spanning 10 ms. Z is fitted with known gravity
    # during this no-field observation window.
    cfg.auto_release_post_release_window_sec = 0.060
    cfg.auto_release_post_release_min_samples = 3
    cfg.auto_release_post_release_min_span_sec = 0.010
    cfg.auto_release_post_release_timeout_sec = 0.100
    # Keep recent timestamped stereo measurements so frames captured after the
    # release image but before the asynchronous MediaPipe result are not lost.
    cfg.auto_release_stereo_history_sec = 0.200

    # Physical-release gate.  MediaPipe separation alone is not allowed to
    # emit ultrasound: three consecutive stereo estimates must show both a
    # downward velocity and a measurable downward displacement.
    cfg.auto_release_gate_max_vz_mm_s = -100.0
    cfg.auto_release_gate_min_drop_mm = 2.5
    cfg.auto_release_gate_consecutive_estimates = 3
    cfg.auto_release_gate_timeout_sec = 0.120

    # Build the constant-deceleration Z reference after the exact initial field
    # command is confirmed sent. Braking transitions directly to LOCAL_HOLD.
    cfg.auto_release_nominal_stop_time_sec = 0.100
    cfg.auto_release_min_stop_time_sec = 0.050
    cfg.auto_release_minimum_alignment_time_sec = 0.100

    # The first multi-frame predicted XY position is the reference. After the
    # initial command is sent, braking XY uses the same timestamp-aware PID as
    # LOCAL_HOLD; only Z is overridden by the braking trajectory.
    # The measured acoustic influence range is about 20 mm, so the final
    # command is conservatively limited to 15 mm from the measured sphere.
    cfg.auto_release_xy_capture_radius_mm = 15.0
    cfg.auto_release_z_workspace_margin_mm = 10.0
    cfg.auto_release_trajectory_kp_z = 0.30
    cfg.auto_release_trajectory_kd_z = 0.01
    cfg.auto_release_trajectory_max_correction_mm = 5.0
    cfg.auto_release_brake_exit_vz_mm_s = -30.0
    cfg.auto_release_brake_exit_stable_sec = 0.015

    # Convert the initial predicted Z velocity to the force needed to stop in
    # 100 ms, then invert the 300-trial camera step-response calibration. Once
    # the initial command is sent, the same force calculation uses the time
    # remaining until the braking trajectory's fixed end time.
    # A weighted model with intensity and categorical z offset, corrected from
    # the nominal 0.50 g analysis to the measured 0.53 g mass, gives at the
    # intended zero offset F[A] = 9.94437622 A - 1.02485060 mN (239 valid fits
    # in session 20261020_123846_25876). The table samples that line at the
    # available commands and the controller interpolates between the points.
    cfg.capture_time_sec = 0.100
    cfg.capture_force_safety_factor = 1.0
    cfg.capture_max_intensity_ratio = 1.0
    cfg.capture_intensity_levels = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    cfg.capture_loadcell_model = SI2026_CAPTURE_FORCE_MODEL
    cfg.auto_release_interpolate_intensity = True
    cfg.auto_release_intensity_deadband_ratio = 0.005
    cfg.auto_release_upward_vz_mm_s = 20.0
    cfg.auto_release_upward_intensity_ratio = 0.5
    cfg.auto_release_max_intensity_ratio = 1.0
    # Hold at the released position first, then move the PID reference back to
    # AUTD center / z=400 using the existing speed-limited return controller.
    cfg.auto_release_local_hold_before_return_sec = 3.0
    # For the SI2026 trajectory experiment, never move the hold reference back
    # to home before 10 s from the detected release.  This runner-only setting
    # can be returned to 0/3 s after the experiment without changing the core.
    cfg.auto_release_min_return_from_release_sec = 10.0
    cfg.return_home_speed_xy_mm_s = 20.0
    cfg.return_home_speed_z_mm_s = 15.0
    # Loss of stereo tracking still silences the field. The 1 s braking limit
    # transitions directly to LOCAL_HOLD after release confirmation.
    cfg.auto_release_capture_measurement_timeout_sec = 0.100
    cfg.auto_release_capture_timeout_sec = 1.0
    cfg.auto_release_capture_log_path = (
        "./manual_release_tracking/auto_release_capture_events.csv"
    )
    cfg.auto_release_delay_log_path = (
        "./manual_release_tracking/auto_release_delay_measurements.csv"
    )
    cfg.auto_release_trajectory_log_path = (
        "./manual_release_tracking/auto_release_trajectory_log.csv"
    )

    # SI2026 paper data. These options only observe the existing controller.
    # The optional extra field-off delay is zero by default, so invoking this
    # script without arguments retains the previous capture timing.
    cfg.si2026_logging_enabled = not args.no_paper_log
    cfg.si2026_data_directory = args.paper_data_dir
    cfg.autd_update_log_path = str(
        Path(args.paper_data_dir) / "autd_update_telemetry.csv"
    )
    cfg.si2026_condition_id = args.condition_id
    cfg.si2026_planned_release_x_mm = args.release_x_mm
    cfg.si2026_planned_release_y_mm = args.release_y_mm
    cfg.si2026_planned_release_z_mm = args.release_z_mm
    cfg.si2026_extra_field_delay_sec = args.extra_delay_ms / 1000.0
    cfg.si2026_success_hold_sec = args.success_hold_sec
    cfg.si2026_observation_sec = args.observation_sec
    cfg.si2026_target_valid_trials = args.target_valid_trials
    cfg.si2026_operator = args.operator
    cfg.si2026_notes = args.notes

    # R remains a fallback. Automatic mode starts from the first explicit
    # XY-camera separation after GRASPED; contact return cancels it.
    cfg.manual_release_key = "r"
    cfg.manual_release_pre_hold_sec = 0.3
    cfg.false_release_reject_key = "x"

    trigger = None
    try:
        trigger = MediaPipeReleaseTrigger(cfg)
        run_manual_release_app(cfg, auto_release_trigger=trigger)
    finally:
        if trigger is not None:
            trigger.close()
