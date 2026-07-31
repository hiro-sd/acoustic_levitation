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

    # CAPTURE_ALIGN follows the predicted sphere position with a modest,
    # velocity-dependent braking intensity. Normal LOCAL_HOLD remains 0.6.
    cfg.auto_release_slow_down_vz_mm_s = -30.0
    cfg.auto_release_fast_down_vz_mm_s = -100.0
    cfg.auto_release_upward_vz_mm_s = 20.0
    cfg.auto_release_upward_intensity_ratio = 0.5
    cfg.auto_release_slow_intensity_ratio = 0.7
    cfg.auto_release_max_intensity_ratio = 0.8
    cfg.auto_release_local_hold_vz_abs_mm_s = 30.0
    cfg.auto_release_local_hold_stable_sec = 0.050
    cfg.auto_release_capture_measurement_timeout_sec = 0.100
    cfg.auto_release_capture_timeout_sec = 1.0
    cfg.auto_release_capture_log_path = (
        "./manual_release_tracking/auto_release_capture_events.csv"
    )
    cfg.auto_release_delay_log_path = (
        "./manual_release_tracking/auto_release_delay_measurements.csv"
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
