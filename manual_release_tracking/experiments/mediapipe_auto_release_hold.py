import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.config import AppConfig
from manual_release_tracking.core.mediapipe_release_trigger import (
    MediaPipeReleaseTrigger,
)
from manual_release_tracking.manual_release_app import run_manual_release_app


if __name__ == "__main__":
    cfg = AppConfig(
        radius=19.0,
        default_z=400.0,
        static_intensity_ratio=0.6,

        # Use calibrated stereo 3D position as the feedback measurement.
        enable_stereo_triangulation=True,
        use_stereo_position_for_control=True,
        stereo_camera_to_autd_npz="./tracking/calibration/stereo_camera_to_autd.npz",

        # First automatic-release experiment:
        # keep exactly the successful manual LOCAL_HOLD controller.
        enable_base_move=False,
        enable_radius_change=False,
        enable_fall_recovery=False,
        enable_auto_demo=False,
        use_gravity_prediction_z=False,

        log_csv_path="./manual_release_tracking/auto_release_log.csv",
    )

    # MediaPipe Tasks Hand Landmarker settings used by the preview.
    cfg.mediapipe_hand_model_path = (
        "./manual_release_tracking/models/hand_landmarker.task"
    )
    cfg.mediapipe_input_width_px = 640
    cfg.mediapipe_tip_gap_normalized_max = 0.45
    cfg.mediapipe_min_opposition_angle_deg = 60.0
    cfg.mediapipe_grasp_confirm_ms = 50.0
    cfg.mediapipe_release_confirm_ms = 35.0
    cfg.mediapipe_released_latch_ms = 500.0
    cfg.mediapipe_result_pair_tolerance_ms = 40.0
    cfg.mediapipe_submit_max_fps = 60.0
    # Z camera remains part of stereo 3D reconstruction, but its hand landmarks
    # are not reliable enough for the grasp/release decision on the current rig.
    cfg.mediapipe_auto_release_camera = "xy"

    # Reject a stale asynchronous release result. This is only a trigger-age
    # guard; the hold target itself uses the newest stereo measurement.
    cfg.mediapipe_auto_release_max_result_age_ms = 80.0

    # Release-specific capture. Position/velocity are estimated continuously,
    # including while the sphere is still held by the user.
    cfg.auto_release_position_current_weight = 0.55
    cfg.auto_release_velocity_current_weight = 0.50
    cfg.auto_release_motion_max_gap_sec = 0.10
    cfg.auto_release_max_motion_age_sec = 0.050
    cfg.auto_release_actuation_prediction_sec = 0.013

    # Build a constant-deceleration Z reference after the first field is sent.
    # Intensity remains the existing velocity-based staged schedule for this
    # experiment so that trajectory and force-control effects stay separable.
    cfg.auto_release_nominal_stop_time_sec = 0.120
    cfg.auto_release_min_stop_time_sec = 0.050
    cfg.auto_release_z_workspace_margin_mm = 10.0
    cfg.auto_release_trajectory_kp_z = 0.30
    cfg.auto_release_trajectory_kd_z = 0.01
    cfg.auto_release_trajectory_max_correction_mm = 5.0
    cfg.auto_release_brake_exit_vz_mm_s = -30.0
    cfg.auto_release_brake_exit_stable_sec = 0.015

    # Convert the measured/predicted Z velocity to the force needed to stop in
    # 120 ms, then select the smallest load-cell level that can provide it.
    # This experiment shifts the measured model by +1 mN so intensity 0.6
    # corresponds to the approximately 5 mN needed for static levitation.
    cfg.fall_capture_time_sec = 0.120
    cfg.fall_capture_force_safety_factor = 1.0
    cfg.fall_capture_max_intensity_ratio = 0.8
    cfg.fall_capture_intensity_levels = (0.5, 0.6, 0.7, 0.8)
    cfg.fall_capture_loadcell_model = (
        (0.5, 4.0),
        (0.6, 5.0),
        (0.7, 6.0),
        (0.8, 7.0),
    )
    cfg.auto_release_interpolate_intensity = True
    cfg.auto_release_intensity_deadband_ratio = 0.005
    cfg.auto_release_upward_vz_mm_s = 20.0
    cfg.auto_release_upward_intensity_ratio = 0.5
    cfg.auto_release_max_intensity_ratio = 0.8
    # CAPTURE_SETTLE uses the same force command. The upward safety latch is
    # retained so an upward-moving sphere immediately receives intensity 0.5.
    cfg.auto_release_settle_normal_intensity_ratio = 0.6
    cfg.auto_release_settle_upward_intensity_ratio = 0.5
    cfg.auto_release_settle_upward_enter_vz_mm_s = 60.0
    cfg.auto_release_settle_upward_exit_vz_mm_s = 20.0

    cfg.auto_release_local_hold_vz_abs_mm_s = 60.0
    cfg.auto_release_local_hold_vxy_max_mm_s = 60.0
    cfg.auto_release_local_hold_xy_distance_max_mm = 15.0
    cfg.auto_release_local_hold_z_error_max_mm = 10.0
    cfg.auto_release_local_hold_stable_sec = 0.025
    # Hold at the released position first, then move the PID reference back to
    # AUTD center / z=400 using the existing speed-limited return controller.
    cfg.auto_release_local_hold_before_return_sec = 3.0
    cfg.return_home_speed_xy_mm_s = 20.0
    cfg.return_home_speed_z_mm_s = 15.0
    # Loss of stereo tracking still silences the field. The 1 s capture limit
    # only forces CAPTURE_ALIGN -> CAPTURE_SETTLE; it no longer stops output.
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

    # R remains a fallback. Automatic mode starts from the first explicit
    # XY-camera separation after GRASPED; contact return cancels it.
    cfg.manual_release_key = "r"
    cfg.manual_release_pre_hold_sec = 0.3

    trigger = None
    try:
        trigger = MediaPipeReleaseTrigger(cfg)
        run_manual_release_app(cfg, auto_release_trigger=trigger)
    finally:
        if trigger is not None:
            trigger.close()
