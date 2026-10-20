import time
import threading

import cv2
import numpy as np

from autd3.commands import SetSilencer

from tracking.core.config import AppConfig
from tracking.core.autd_client import AutdController
from tracking.core.vision import (
    load_ximea_api,
    init_ximea_camera,
    load_intrinsic,
    build_undistort_maps,
    undistort_frame,
    rotate_frame_if_needed,
    SharedFrameBuffer,
    camera_capture_loop,
    safe_close_camera,
)
from tracking.core.ball_tracker import RoiBallTracker, draw_ball_detection
from tracking.core.autd_sender import (
    make_autd_arrangement,
    AutdSender,
    set_tracking_target,
)
from tracking.core.controller import PredictionPIDController
from tracking.core.models import HomePosition, Measurement3D, Target3D
from tracking.core.prediction_timing import prediction_timing_from_timestamps
from tracking.core.control.intensity import compute_z_intensity_boost_ratio
from tracking.core.control.capture_force import (
    apply_capture_intensity_slew,
    capture_buoyancy_mN,
    capture_force_command_for_velocity,
    measured_force_for_intensity_mN,
)
from tracking.core.control.hold import (
    LOCAL_HOLD,
    NORMAL_HOLD,
    RETURN_TO_HOME,
    HoldTelemetry,
    is_return_home_done,
    move_towards,
)
from tracking.core.runtime.display import (
    DisplayControlState,
    DisplayMetrics,
    render_tracking_window,
)
from tracking.core.runtime.logging import StabilityLogger
from tracking.core.runtime.input import is_key_pressed, update_base_position
from tracking.core.runtime.demo import SquareZDemo
from tracking.core.stereo import RigidTransform, StereoTriangulator
from manual_release_tracking.core.mediapipe_release_trigger import (
    should_stop_automatic_hold,
)
from manual_release_tracking.core.auto_release_capture import (
    CAPTURE_ALIGN,
    CAPTURE_SETTLE,
    SETTLE_INTENSITY_DOWNWARD_RESCUE,
    SETTLE_INTENSITY_NORMAL,
    SETTLE_INTENSITY_UPWARD_DAMPING,
    AutoReleaseCaptureLogger,
    AutoReleaseDelayLogger,
    AutoReleaseTrajectoryLogger,
    PredictedCapture,
    RecentStereoHistory,
    StereoMotionEstimator,
    anchored_pd_target_xy,
    braking_reference_at,
    capture_activation_ready,
    clamp_target_xy_to_ball,
    limit_upward_capture_target_z,
    make_braking_plan,
    predict_capture_position,
    should_start_return_home,
    should_start_settle_timeout_return_home,
    trajectory_target_z,
    update_brake_exit_stability,
    update_capture_stability,
    update_release_motion_gate,
)
from manual_release_tracking.core.si2026_experiment_logging import (
    SI2026Condition,
    SI2026ExperimentLogger,
    release_hold_success_reached,
)


def run_manual_release_app(cfg: AppConfig, auto_release_trigger=None):
    """
    共通実行本体。

    手から離した球体をステレオ3D計測で捕捉・保持する共通実行本体。
    """

    # 1. Calibration files
    try:
        mtx_cam_xy, dist_cam_xy = load_intrinsic(cfg.intrinsic_xy_npz, "xy")
        mtx_cam_z, dist_cam_z = load_intrinsic(cfg.intrinsic_z_npz, "z")
        use_undistort = True
        print("[INFO] Vision pipeline: remap undistort each camera frame, then detect.")
    except Exception as e:
        print(f"[WARN] Intrinsic parameters load failed: {e}")
        mtx_cam_xy = None
        dist_cam_xy = None
        mtx_cam_z = None
        dist_cam_z = None
        use_undistort = False

    stereo_triangulator = None
    camera_to_autd = None
    if cfg.enable_stereo_triangulation:
        try:
            stereo_triangulator = StereoTriangulator.from_npz(
                cfg.stereo_npz,
                input_is_undistorted=use_undistort,
            )
            print(f"[INFO] Loaded stereo calibration from {cfg.stereo_npz}")
        except Exception as e:
            print(f"[ERROR] Stereo calibration load failed: {e}")
            return

        if cfg.stereo_camera_to_autd_npz:
            try:
                camera_to_autd = RigidTransform.from_npz(cfg.stereo_camera_to_autd_npz)
                print(f"[INFO] Loaded stereo camera-to-AUTD transform from {cfg.stereo_camera_to_autd_npz}")
            except Exception as e:
                print(f"[ERROR] camera-to-AUTD transform load failed: {e}")
                return

        if cfg.use_stereo_position_for_control and camera_to_autd is None:
            print(
                "[ERROR] use_stereo_position_for_control=True requires "
                "cfg.stereo_camera_to_autd_npz. Stereo 3D is initially in cam1 coordinates, "
                "not AUTD coordinates."
            )
            return

    # 2. Camera initialization
    xiapi = load_ximea_api(cfg)

    if xiapi is None:
        print("[ERROR] XIMEA API not available.")
        return

    try:
        cam_xy, img_xy = init_ximea_camera(xiapi, cfg.camera_xy_sn, "xy", cfg)
        cam_z, img_z = init_ximea_camera(xiapi, cfg.camera_z_sn, "z", cfg)
    except Exception as e:
        print(f"[ERROR] Camera Init Failed: {e}")
        return

    # 最初の1フレームで画像サイズとremap mapを作る
    try:
        cam_xy.get_image(img_xy)
        frame0_xy_time = time.perf_counter()
        frame0_raw_xy = img_xy.get_image_data_numpy()

        cam_z.get_image(img_z)
        frame0_z_time = time.perf_counter()
        frame0_raw_z = img_z.get_image_data_numpy()
    except Exception as e:
        print(f"[ERROR] Failed to get first camera frame: {e}")
        safe_close_camera(cam_xy)
        safe_close_camera(cam_z)
        return

    if use_undistort:
        map1_xy, map2_xy = build_undistort_maps(frame0_raw_xy.shape, mtx_cam_xy, dist_cam_xy)

        frame0_xy = undistort_frame(frame0_raw_xy, map1_xy, map2_xy)

        if cfg.rotate_z_frame and cfg.z_intrinsic_is_rotated:
            frame0_z_for_intrinsic = rotate_frame_if_needed(
                frame0_raw_z,
                True,
                cfg.rotate_z_code,
            )
            map1_z, map2_z = build_undistort_maps(
                frame0_z_for_intrinsic.shape,
                mtx_cam_z,
                dist_cam_z,
            )
            frame0_z = undistort_frame(frame0_z_for_intrinsic, map1_z, map2_z)
        else:
            map1_z, map2_z = build_undistort_maps(frame0_raw_z.shape, mtx_cam_z, dist_cam_z)
            frame0_z = undistort_frame(frame0_raw_z, map1_z, map2_z)
    else:
        map1_xy = map2_xy = None
        map1_z = map2_z = None
        frame0_xy = frame0_raw_xy
        frame0_z = frame0_raw_z

    if not (use_undistort and cfg.z_intrinsic_is_rotated):
        frame0_z = rotate_frame_if_needed(frame0_z, cfg.rotate_z_frame, cfg.rotate_z_code)

    H_xy, W_xy = frame0_xy.shape[:2]
    H_z, W_z = frame0_z.shape[:2]

    frame_buffer = SharedFrameBuffer(maxlen=cfg.camera_sync_buffer_size)
    frame_buffer.set_xy(frame0_xy.copy(), frame0_xy_time)
    frame_buffer.set_z(frame0_z.copy(), frame0_z_time)

    running_event = threading.Event()
    running_event.set()

    cam_thread_xy = threading.Thread(
        target=camera_capture_loop,
        args=(
            cam_xy,
            img_xy,
            "xy",
            cfg,
            frame_buffer,
            running_event,
            use_undistort,
            map1_xy,
            map2_xy,
        ),
        daemon=True,
    )

    cam_thread_z = threading.Thread(
        target=camera_capture_loop,
        args=(
            cam_z,
            img_z,
            "z",
            cfg,
            frame_buffer,
            running_event,
            use_undistort,
            map1_z,
            map2_z,
        ),
        daemon=True,
    )

    cam_thread_xy.start()
    cam_thread_z.start()

    # 3. AUTD initialization
    autd_arrangement = make_autd_arrangement()

    print("[INFO] Opening AUTD Controller...")

    sender = None
    logger = StabilityLogger(cfg)
    capture_logger = AutoReleaseCaptureLogger(
        str(
            getattr(
                cfg,
                "auto_release_capture_log_path",
                "./manual_release_tracking/auto_release_capture_events.csv",
            )
        )
    )
    delay_logger = AutoReleaseDelayLogger(
        str(
            getattr(
                cfg,
                "auto_release_delay_log_path",
                "./manual_release_tracking/auto_release_delay_measurements.csv",
            )
        )
    )
    trajectory_logger = AutoReleaseTrajectoryLogger(
        str(
            getattr(
                cfg,
                "auto_release_trajectory_log_path",
                "./manual_release_tracking/auto_release_trajectory_log.csv",
            )
        )
    )
    paper_logger = None

    try:
        paper_logger = SI2026ExperimentLogger(
            getattr(
                cfg,
                "si2026_data_directory",
                "./manual_release_tracking/si2026_data",
            ),
            config=cfg,
            condition=SI2026Condition(
                condition_id=str(
                    getattr(cfg, "si2026_condition_id", "baseline")
                ),
                operator=str(getattr(cfg, "si2026_operator", "")),
                planned_release_x_mm=getattr(
                    cfg,
                    "si2026_planned_release_x_mm",
                    None,
                ),
                planned_release_y_mm=getattr(
                    cfg,
                    "si2026_planned_release_y_mm",
                    None,
                ),
                planned_release_z_mm=getattr(
                    cfg,
                    "si2026_planned_release_z_mm",
                    None,
                ),
                extra_field_delay_ms=(
                    float(
                        getattr(
                            cfg,
                            "si2026_extra_field_delay_sec",
                            0.0,
                        )
                    )
                    * 1000.0
                ),
                notes=str(getattr(cfg, "si2026_notes", "")),
            ),
            enabled=bool(getattr(cfg, "si2026_logging_enabled", False)),
        )
        with AutdController.open(
            autd_arrangement,
            interface_name=cfg.autd_interface_name,
            open_timeout_sec=cfg.autd_open_timeout_sec,
            command_timeout_sec=cfg.autd_command_timeout_sec,
        ) as autd:

            autd.send(SetSilencer())

            base_center = autd.center()

            home = HomePosition(
                x=float(base_center[0]),
                y=float(base_center[1]),
                z=float(cfg.default_z),
            )
            paper_logger.set_reference_center(home.x, home.y, home.z)

            display_origin = HomePosition(
                x=float(home.x),
                y=float(home.y),
                z=float(home.z),
            )

            sender = AutdSender(autd, cfg)
            current_radius = float(cfg.radius)
            current_intensity_ratio = float(cfg.static_intensity_ratio)
            target_intensity_ratio = float(cfg.static_intensity_ratio)
            sender.start()

            def estimated_actuation_delay_sec() -> float:
                return sender.estimated_command_latency_sec(
                    float(
                        getattr(
                            cfg,
                            "auto_release_actuation_prediction_sec",
                            0.0075,
                        )
                    )
                )

            def timestamp_prediction_horizon_sec(
                measurement_time_sec: float,
                now_sec: float,
            ) -> float:
                return prediction_timing_from_timestamps(
                    measurement_time_sec=measurement_time_sec,
                    now_sec=now_sec,
                    actuation_delay_sec=estimated_actuation_delay_sec(),
                ).horizon_sec

            controller = PredictionPIDController(cfg)
            controller.reset(Target3D(home.x, home.y, home.z))
            motion_estimator = StereoMotionEstimator(
                position_current_weight=float(
                    getattr(cfg, "auto_release_position_current_weight", 0.55)
                ),
                velocity_current_weight=float(
                    getattr(cfg, "auto_release_velocity_current_weight", 0.50)
                ),
                max_update_gap_sec=float(
                    getattr(cfg, "auto_release_motion_max_gap_sec", 0.10)
                ),
                velocity_window_sec=float(
                    getattr(cfg, "auto_release_velocity_window_sec", 0.050)
                ),
                minimum_velocity_samples=int(
                    getattr(cfg, "auto_release_min_velocity_samples", 5)
                ),
                minimum_velocity_span_sec=float(
                    getattr(
                        cfg,
                        "auto_release_min_velocity_span_sec",
                        0.020,
                    )
                ),
            )
            post_release_motion_estimator = StereoMotionEstimator(
                # The dedicated estimator is reset at the first explicit
                # separation, so every sample belongs to post-release motion.
                # The multi-frame fit itself filters random position noise.
                position_current_weight=1.0,
                velocity_current_weight=1.0,
                max_update_gap_sec=float(
                    getattr(cfg, "auto_release_motion_max_gap_sec", 0.10)
                ),
                velocity_window_sec=float(
                    getattr(cfg, "auto_release_post_release_window_sec", 0.060)
                ),
                minimum_velocity_samples=int(
                    getattr(
                        cfg,
                        "auto_release_post_release_min_samples",
                        3,
                    )
                ),
                minimum_velocity_span_sec=float(
                    getattr(
                        cfg,
                        "auto_release_post_release_min_span_sec",
                        0.010,
                    )
                ),
                known_z_acceleration_mm_s2=-float(cfg.gravity_mm_s2),
            )
            recent_stereo_history = RecentStereoHistory(
                retention_sec=float(
                    getattr(
                        cfg,
                        "auto_release_stereo_history_sec",
                        0.200,
                    )
                )
            )

            # 4. Runtime state
            tracking_active = False
            auto_hold_active = False
            auto_release_confirmed = False
            auto_release_timestamp_ms = None
            capture_started_at = None
            capture_stable_since = None
            capture_brake_exit_since = None
            capture_brake_exit_last_measurement_time = None
            capture_settle_started_at = None
            capture_settle_last_measurement_time = None
            capture_settle_intensity_state = SETTLE_INTENSITY_NORMAL
            capture_start_estimate = None
            capture_initial_prediction = None
            capture_onset_prediction = None
            capture_braking_plan = None
            capture_xy_anchor = None
            capture_xy_pd_saturation_logged = False
            capture_last_trajectory_log_measurement_time = None
            pending_capture_command_seq = None
            pending_capture_sent_logged = False
            capture_upward_brake_active = False
            release_observation_event = None
            release_observation_started_at = None
            release_observation_confirmed = False
            release_motion_ready_at = None
            release_gate_origin_z_mm = None
            release_gate_consecutive_count = 0
            release_gate_last_measurement_time = None
            release_gate_passed = False
            release_gate_downward_displacement_mm = None
            prev_enter_pressed = False
            prev_demo_toggle_pressed = False
            prev_log_trigger_pressed = False
            prev_delay_toggle_pressed = False
            prev_manual_release_pressed = False
            prev_false_release_reject_pressed = False

            control_mode = NORMAL_HOLD
            demo_active = False
            demo = SquareZDemo(cfg, display_origin)

            local_hold_start_time = None
            mode_transition_reason = ""

            return_setpoint = HomePosition(
                x=home.x,
                y=home.y,
                z=home.z,
            )

            tracker_xy = RoiBallTracker(W_xy, H_xy, cfg)
            tracker_z = RoiBallTracker(W_z, H_z, cfg)

            last_target = Target3D(home.x, home.y, home.z)

            fps_start_time = time.time()
            loop_fps_count = 0
            loop_display_fps = 0.0
            new_xy_fps_count = 0
            new_z_fps_count = 0
            new_xy_display_fps = 0.0
            new_z_display_fps = 0.0
            frame_count = 0

            prev_loop_time = time.time()
            last_synced_pair_time = time.perf_counter()

            window_title = "Tracking App"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            manual_release_key = str(getattr(cfg, "manual_release_key", "r")).lower()
            manual_release_pre_hold_sec = float(
                getattr(cfg, "manual_release_pre_hold_sec", 0.3)
            )
            false_release_reject_key = str(
                getattr(cfg, "false_release_reject_key", "x")
            ).lower()

            def force_command_for_vz(vz_mm_s, current_intensity=None):
                return capture_force_command_for_velocity(
                    cfg,
                    float(vz_mm_s),
                    current_intensity=current_intensity,
                    downward_hysteresis_mN=float(
                        getattr(
                            cfg,
                            "auto_release_force_down_hysteresis_mN",
                            0.05,
                        )
                    ),
                    interpolate_intensity=bool(
                        getattr(
                            cfg,
                            "auto_release_interpolate_intensity",
                            False,
                        )
                    ),
                    minimum_intensity=float(cfg.static_intensity_ratio),
                    intensity_deadband=float(
                        getattr(
                            cfg,
                            "auto_release_intensity_deadband_ratio",
                            cfg.intensity_update_eps,
                        )
                    ),
                )

            def settle_force_intensity(
                vz_mm_s,
                current_state,
                current_intensity,
            ):
                """Force-based settling with the existing upward safety latch."""
                vz = float(vz_mm_s)
                upward_enter = float(
                    getattr(
                        cfg,
                        "auto_release_settle_upward_enter_vz_mm_s",
                        60.0,
                    )
                )
                upward_exit = float(
                    getattr(
                        cfg,
                        "auto_release_settle_upward_exit_vz_mm_s",
                        20.0,
                    )
                )
                force_command = force_command_for_vz(
                    vz,
                    current_intensity=current_intensity,
                )

                upward_latched = (
                    current_state == SETTLE_INTENSITY_UPWARD_DAMPING
                    and vz >= upward_exit
                )
                if vz > upward_enter or upward_latched:
                    return (
                        SETTLE_INTENSITY_UPWARD_DAMPING,
                        float(
                            getattr(
                                cfg,
                                "auto_release_settle_upward_intensity_ratio",
                                0.5,
                            )
                        ),
                        force_command,
                    )

                normal_ratio = float(
                    getattr(
                        cfg,
                        "auto_release_settle_normal_intensity_ratio",
                        cfg.static_intensity_ratio,
                    )
                )
                commanded = float(force_command.commanded_intensity)
                state = (
                    SETTLE_INTENSITY_DOWNWARD_RESCUE
                    if commanded > normal_ratio + 1e-9
                    else SETTLE_INTENSITY_NORMAL
                )
                return state, commanded, force_command

            def arm_local_hold(
                x_mm,
                y_mm,
                z_mm,
                *,
                reason: str,
                label: str,
                prefer_filtered_z: bool,
            ) -> bool:
                nonlocal tracking_active
                nonlocal control_mode
                nonlocal local_hold_start_time
                nonlocal mode_transition_reason
                nonlocal current_intensity_ratio
                nonlocal return_setpoint
                nonlocal last_target

                if x_mm is None or y_mm is None or z_mm is None:
                    print(
                        f"[{label}] ignored: valid stereo 3D position is not available."
                    )
                    return False

                z_setpoint = float(z_mm)
                if (
                    prefer_filtered_z
                    and controller.prev_particle_z_filt is not None
                ):
                    z_setpoint = float(controller.prev_particle_z_filt)

                return_setpoint = HomePosition(
                    x=float(x_mm),
                    y=float(y_mm),
                    z=z_setpoint,
                )
                last_target = Target3D(
                    return_setpoint.x,
                    return_setpoint.y,
                    return_setpoint.z,
                )

                controller.reset(last_target)
                tracking_active = True
                control_mode = LOCAL_HOLD
                local_hold_start_time = time.time()
                mode_transition_reason = reason
                current_intensity_ratio = float(cfg.static_intensity_ratio)

                set_tracking_target(
                    sender,
                    cfg,
                    last_target.x,
                    last_target.y,
                    last_target.z,
                    home,
                    current_radius,
                    current_intensity_ratio,
                )

                print(
                    f"[{label}] LOCAL_HOLD started at "
                    f"({return_setpoint.x:.1f}, "
                    f"{return_setpoint.y:.1f}, "
                    f"{return_setpoint.z:.1f}) mm."
                )
                return True

            def start_release_observation(event):
                """Begin a no-field observation using only post-release frames."""
                nonlocal release_observation_event
                nonlocal release_observation_started_at
                nonlocal release_observation_confirmed
                nonlocal release_motion_ready_at
                nonlocal release_gate_origin_z_mm
                nonlocal release_gate_consecutive_count
                nonlocal release_gate_last_measurement_time
                nonlocal release_gate_passed
                nonlocal release_gate_downward_displacement_mm

                post_release_motion_estimator.reset()
                release_time_sec = float(event.timestamp_ms) / 1000.0
                release_samples = recent_stereo_history.samples_since(
                    release_time_sec
                )
                for sample in release_samples:
                    post_release_motion_estimator.update(
                        sample.measurement_time_sec,
                        sample.x_mm,
                        sample.y_mm,
                        sample.z_mm,
                    )
                seeded_sample_count = len(release_samples)
                seeded_estimate = post_release_motion_estimator.latest
                release_state_estimate = (
                    release_samples[0] if release_samples else seeded_estimate
                )
                seeded_span_ms = (
                    0.0
                    if seeded_estimate is None
                    else seeded_estimate.history_span_sec * 1000.0
                )
                release_observation_event = event
                release_observation_started_at = time.perf_counter()
                release_observation_confirmed = False
                release_motion_ready_at = None
                release_gate_origin_z_mm = (
                    float(release_samples[0].z_mm)
                    if release_samples
                    else None
                    if seeded_estimate is None
                    else float(seeded_estimate.z_mm)
                )
                release_gate_consecutive_count = 0
                release_gate_last_measurement_time = None
                release_gate_passed = False
                release_gate_downward_displacement_mm = None
                paper_logger.start_trial(
                    release_timestamp_ms=int(event.timestamp_ms),
                    release_estimate=release_state_estimate,
                )
                capture_logger.write(
                    event="post_release_observation_started",
                    release_timestamp_ms=int(event.timestamp_ms),
                    event_time_sec=release_observation_started_at,
                    estimate=seeded_estimate,
                    target=None,
                    intensity=0.0,
                    reason=(
                        "field_off_collecting_post_release_frames;"
                        f"buffered_samples={seeded_sample_count};"
                        f"buffered_span_ms={seeded_span_ms:.1f}"
                    ),
                )
                print(
                    "[AUTO_RELEASE] RELEASE_CANDIDATE -> "
                    "POST_RELEASE_OBSERVE (field remains off), "
                    f"replayed={seeded_sample_count} frames"
                )

            def clear_release_observation(
                reason: str,
                *,
                log_event=True,
                finish_paper_trial=True,
                paper_status="invalid",
                reset_gate_state=True,
            ):
                nonlocal release_observation_event
                nonlocal release_observation_started_at
                nonlocal release_observation_confirmed
                nonlocal release_motion_ready_at
                nonlocal release_gate_origin_z_mm
                nonlocal release_gate_consecutive_count
                nonlocal release_gate_last_measurement_time
                nonlocal release_gate_passed
                nonlocal release_gate_downward_displacement_mm

                estimate = post_release_motion_estimator.latest
                if log_event and release_observation_event is not None:
                    capture_logger.write(
                        event="post_release_observation_ended",
                        release_timestamp_ms=int(
                            release_observation_event.timestamp_ms
                        ),
                        estimate=estimate,
                        target=None,
                        intensity=0.0,
                        reason=reason,
                    )
                post_release_motion_estimator.reset()
                if finish_paper_trial:
                    paper_logger.finish_trial(
                        status=paper_status,
                        reason=reason,
                    )
                release_observation_event = None
                release_observation_started_at = None
                release_observation_confirmed = False
                release_motion_ready_at = None
                if reset_gate_state:
                    release_gate_origin_z_mm = None
                    release_gate_consecutive_count = 0
                    release_gate_last_measurement_time = None
                    release_gate_passed = False
                    release_gate_downward_displacement_mm = None

            def arm_capture_align(event, estimate) -> bool:
                nonlocal tracking_active
                nonlocal auto_hold_active
                nonlocal auto_release_confirmed
                nonlocal auto_release_timestamp_ms
                nonlocal control_mode
                nonlocal capture_started_at
                nonlocal capture_stable_since
                nonlocal capture_brake_exit_since
                nonlocal capture_brake_exit_last_measurement_time
                nonlocal capture_settle_started_at
                nonlocal capture_settle_last_measurement_time
                nonlocal capture_settle_intensity_state
                nonlocal capture_start_estimate
                nonlocal capture_initial_prediction
                nonlocal capture_onset_prediction
                nonlocal capture_braking_plan
                nonlocal capture_xy_anchor
                nonlocal capture_xy_pd_saturation_logged
                nonlocal capture_last_trajectory_log_measurement_time
                nonlocal release_gate_origin_z_mm
                nonlocal release_gate_consecutive_count
                nonlocal release_gate_last_measurement_time
                nonlocal release_gate_passed
                nonlocal release_gate_downward_displacement_mm
                nonlocal pending_capture_command_seq
                nonlocal pending_capture_sent_logged
                nonlocal capture_upward_brake_active
                nonlocal current_intensity_ratio
                nonlocal return_setpoint
                nonlocal last_target
                nonlocal mode_transition_reason

                now_sec = time.perf_counter()
                maximum_age_sec = float(
                    getattr(
                        cfg,
                        "auto_release_max_motion_age_sec",
                        0.050,
                    )
                )
                if estimate is None:
                    print(
                        "[AUTO_RELEASE] RELEASE_CANDIDATE ignored: "
                        "no stereo motion estimate."
                    )
                    return False
                if not bool(estimate.velocity_ready):
                    print(
                        "[AUTO_RELEASE] RELEASE_CANDIDATE ignored: "
                        "multi-frame motion estimate is not ready "
                        f"(samples={estimate.sample_count}, "
                        f"span={estimate.history_span_sec * 1000.0:.1f} ms)."
                    )
                    return False
                measurement_age = now_sec - estimate.measurement_time_sec
                if measurement_age < 0.0 or measurement_age > maximum_age_sec:
                    print(
                        "[AUTO_RELEASE] RELEASE_CANDIDATE ignored: "
                        f"stereo estimate age={measurement_age * 1000.0:.1f} ms."
                    )
                    return False

                prediction = predict_capture_position(
                    estimate,
                    now_sec=now_sec,
                    actuation_delay_sec=estimated_actuation_delay_sec(),
                    gravity_mm_s2=float(cfg.gravity_mm_s2),
                    apply_gravity=True,
                )
                last_target = Target3D(
                    x=float(prediction.x_mm),
                    y=float(prediction.y_mm),
                    z=float(np.clip(prediction.z_mm, cfg.z_min, cfg.z_max)),
                )
                return_setpoint = HomePosition(
                    x=float(estimate.x_mm),
                    y=float(estimate.y_mm),
                    z=float(estimate.z_mm),
                )
                initial_force_command = force_command_for_vz(
                    prediction.vz_mm_s,
                )
                current_intensity_ratio = float(
                    initial_force_command.commanded_intensity
                )
                if prediction.vz_mm_s > float(
                    getattr(cfg, "auto_release_upward_vz_mm_s", 20.0)
                ):
                    current_intensity_ratio = float(
                        getattr(
                            cfg,
                            "auto_release_upward_intensity_ratio",
                            0.5,
                        )
                    )

                controller.reset(last_target)
                tracking_active = True
                auto_hold_active = True
                auto_release_confirmed = False
                auto_release_timestamp_ms = int(event.timestamp_ms)
                control_mode = CAPTURE_ALIGN
                capture_started_at = now_sec
                capture_stable_since = None
                capture_brake_exit_since = None
                capture_brake_exit_last_measurement_time = None
                capture_settle_started_at = None
                capture_settle_last_measurement_time = None
                capture_settle_intensity_state = SETTLE_INTENSITY_NORMAL
                capture_start_estimate = estimate
                capture_initial_prediction = prediction
                capture_onset_prediction = None
                capture_braking_plan = None
                capture_xy_anchor = (
                    float(prediction.x_mm),
                    float(prediction.y_mm),
                )
                capture_xy_pd_saturation_logged = False
                capture_last_trajectory_log_measurement_time = None
                pending_capture_sent_logged = False
                capture_upward_brake_active = False
                mode_transition_reason = "release_candidate_capture_align"

                pending_capture_command_seq = set_tracking_target(
                    sender,
                    cfg,
                    last_target.x,
                    last_target.y,
                    last_target.z,
                    home,
                    current_radius,
                    current_intensity_ratio,
                )
                capture_logger.write(
                    event="release_candidate_command_enqueued",
                    release_timestamp_ms=auto_release_timestamp_ms,
                    event_time_sec=now_sec,
                    estimate=estimate,
                    prediction=prediction,
                    target=last_target,
                    intensity=current_intensity_ratio,
                    command_seq=pending_capture_command_seq,
                    reason=(
                        "first_explicit_fingertip_separation;"
                        f"required_force_mN="
                        f"{initial_force_command.required_force_mN:.3f};"
                        f"force_saturated="
                        f"{int(initial_force_command.saturated)};"
                        f"motion_samples={estimate.sample_count};"
                        f"motion_span_ms="
                        f"{estimate.history_span_sec * 1000.0:.1f}"
                    ),
                )
                print(
                    "[AUTO_RELEASE] RELEASE_CANDIDATE -> CAPTURE_ALIGN "
                    f"p=({last_target.x:.1f}, {last_target.y:.1f}, "
                    f"{last_target.z:.1f}) mm, "
                    f"v=({estimate.vx_mm_s:.1f}, "
                    f"{estimate.vy_mm_s:.1f}, {estimate.vz_mm_s:.1f}) mm/s, "
                    f"fit={estimate.sample_count} frames/"
                    f"{estimate.history_span_sec * 1000.0:.1f} ms, "
                    f"prediction={prediction.horizon_sec * 1000.0:.1f} ms, "
                    f"intensity={current_intensity_ratio:.2f}"
                )
                return True

            def enter_capture_settle(
                reason: str,
                estimate,
                *,
                event_time_sec: float | None = None,
            ) -> bool:
                """End Z braking while preserving the fixed XY capture anchor."""
                nonlocal control_mode
                nonlocal capture_started_at
                nonlocal capture_stable_since
                nonlocal capture_brake_exit_since
                nonlocal capture_brake_exit_last_measurement_time
                nonlocal capture_settle_started_at
                nonlocal capture_settle_last_measurement_time
                nonlocal capture_settle_intensity_state
                nonlocal capture_upward_brake_active
                nonlocal capture_onset_prediction
                nonlocal capture_braking_plan
                nonlocal capture_xy_anchor
                nonlocal capture_xy_pd_saturation_logged
                nonlocal capture_last_trajectory_log_measurement_time
                nonlocal current_intensity_ratio
                nonlocal return_setpoint
                nonlocal last_target
                nonlocal mode_transition_reason

                if estimate is None:
                    return False

                now_sec = (
                    time.perf_counter()
                    if event_time_sec is None
                    else float(event_time_sec)
                )
                paper_logger.mark_braking_end(now_sec, estimate, reason)
                settle_z = float(np.clip(estimate.z_mm, cfg.z_min, cfg.z_max))
                settle_anchor_x = float(
                    estimate.x_mm
                    if capture_xy_anchor is None
                    else capture_xy_anchor[0]
                )
                settle_anchor_y = float(
                    estimate.y_mm
                    if capture_xy_anchor is None
                    else capture_xy_anchor[1]
                )
                settle_xy_target = anchored_pd_target_xy(
                    anchor_x_mm=settle_anchor_x,
                    anchor_y_mm=settle_anchor_y,
                    ball_x_mm=estimate.x_mm,
                    ball_y_mm=estimate.y_mm,
                    vx_mm_s=estimate.vx_mm_s,
                    vy_mm_s=estimate.vy_mm_s,
                    position_gain=float(
                        getattr(cfg, "auto_release_capture_xy_kp", cfg.kp_xy)
                    ),
                    velocity_gain_sec=float(
                        getattr(cfg, "auto_release_capture_xy_kd", cfg.kd_xy)
                    ),
                    prediction_horizon_sec=timestamp_prediction_horizon_sec(
                        estimate.measurement_time_sec,
                        now_sec,
                    ),
                    maximum_distance_mm=float(
                        getattr(
                            cfg,
                            "auto_release_xy_capture_radius_mm",
                            15.0,
                        )
                    ),
                )
                return_setpoint = HomePosition(
                    x=settle_anchor_x,
                    y=settle_anchor_y,
                    z=settle_z,
                )
                last_target = Target3D(
                    settle_xy_target.x_mm,
                    settle_xy_target.y_mm,
                    return_setpoint.z,
                )
                controller.reset(last_target)
                control_mode = CAPTURE_SETTLE
                capture_started_at = None
                capture_stable_since = None
                capture_brake_exit_since = None
                capture_brake_exit_last_measurement_time = None
                capture_settle_started_at = now_sec
                capture_settle_last_measurement_time = None
                capture_settle_intensity_state = SETTLE_INTENSITY_NORMAL
                capture_upward_brake_active = False
                capture_onset_prediction = None
                capture_braking_plan = None
                capture_last_trajectory_log_measurement_time = None
                mode_transition_reason = reason
                (
                    capture_settle_intensity_state,
                    current_intensity_ratio,
                    settle_force_command,
                ) = settle_force_intensity(
                    estimate.vz_mm_s,
                    capture_settle_intensity_state,
                    current_intensity_ratio,
                )
                set_tracking_target(
                    sender,
                    cfg,
                    last_target.x,
                    last_target.y,
                    last_target.z,
                    home,
                    current_radius,
                    current_intensity_ratio,
                )
                capture_logger.write(
                    event="capture_align_to_settle",
                    release_timestamp_ms=auto_release_timestamp_ms,
                    event_time_sec=now_sec,
                    estimate=estimate,
                    prediction=capture_initial_prediction,
                    target=last_target,
                    intensity=current_intensity_ratio,
                    reason=(
                        f"{reason};vz={float(estimate.vz_mm_s):.1f};"
                        f"intensity_state={capture_settle_intensity_state};"
                        f"required_force_mN="
                        f"{settle_force_command.required_force_mN:.3f};"
                        f"force_saturated="
                        f"{int(settle_force_command.saturated)};"
                        f"xy_anchor=({return_setpoint.x:.2f},"
                        f"{return_setpoint.y:.2f});"
                        f"target=({last_target.x:.2f},"
                        f"{last_target.y:.2f},{last_target.z:.2f})"
                    ),
                )
                print(
                    "[AUTO_RELEASE] CAPTURE_ALIGN -> CAPTURE_SETTLE "
                    f"reason={reason}, "
                    f"setpoint=({last_target.x:.1f}, {last_target.y:.1f}, "
                    f"{last_target.z:.1f}) mm, "
                    f"vz={float(estimate.vz_mm_s):.1f} mm/s"
                )
                return True

            def stop_automatic_hold(
                reason: str,
                *,
                paper_status: str | None = None,
            ):
                nonlocal tracking_active
                nonlocal auto_hold_active
                nonlocal auto_release_confirmed
                nonlocal control_mode
                nonlocal local_hold_start_time
                nonlocal mode_transition_reason
                nonlocal current_intensity_ratio
                nonlocal capture_started_at
                nonlocal capture_stable_since
                nonlocal capture_brake_exit_since
                nonlocal capture_brake_exit_last_measurement_time
                nonlocal capture_settle_started_at
                nonlocal capture_settle_last_measurement_time
                nonlocal capture_settle_intensity_state
                nonlocal capture_upward_brake_active
                nonlocal capture_onset_prediction
                nonlocal capture_braking_plan
                nonlocal capture_xy_anchor
                nonlocal capture_xy_pd_saturation_logged
                nonlocal capture_last_trajectory_log_measurement_time
                nonlocal release_gate_origin_z_mm
                nonlocal release_gate_consecutive_count
                nonlocal release_gate_last_measurement_time
                nonlocal release_gate_passed
                nonlocal release_gate_downward_displacement_mm

                stop_time_sec = time.perf_counter()
                if (
                    not paper_logger.success_reached
                    and release_hold_success_reached(
                        elapsed_from_release_sec=(
                            paper_logger.elapsed_from_release_sec(stop_time_sec)
                        ),
                        required_hold_sec=float(
                            getattr(cfg, "si2026_success_hold_sec", 5.0)
                        ),
                        field_started=paper_logger.field_started,
                        automatic_hold_active=auto_hold_active,
                    )
                ):
                    paper_logger.mark_success_reached(stop_time_sec)
                if paper_status is None:
                    paper_status = (
                        "success"
                        if paper_logger.success_reached
                        else "failure"
                        if paper_logger.field_started
                        else "invalid"
                    )
                paper_logger.finish_trial(
                    status=paper_status,
                    reason=reason,
                    event_time_sec=stop_time_sec,
                )

                tracking_active = False
                auto_hold_active = False
                auto_release_confirmed = False
                control_mode = NORMAL_HOLD
                local_hold_start_time = None
                mode_transition_reason = reason
                current_intensity_ratio = 0.0
                capture_started_at = None
                capture_stable_since = None
                capture_brake_exit_since = None
                capture_brake_exit_last_measurement_time = None
                capture_settle_started_at = None
                capture_settle_last_measurement_time = None
                capture_settle_intensity_state = SETTLE_INTENSITY_NORMAL
                capture_upward_brake_active = False
                capture_onset_prediction = None
                capture_braking_plan = None
                capture_xy_anchor = None
                capture_xy_pd_saturation_logged = False
                capture_last_trajectory_log_measurement_time = None
                release_gate_origin_z_mm = None
                release_gate_consecutive_count = 0
                release_gate_last_measurement_time = None
                release_gate_passed = False
                release_gate_downward_displacement_mm = None
                controller.reset(last_target)

                # Keep the sender thread alive for the next release, but make the
                # current field silent immediately.
                set_tracking_target(
                    sender,
                    cfg,
                    last_target.x,
                    last_target.y,
                    last_target.z,
                    home,
                    current_radius,
                    current_intensity_ratio,
                )
                capture_logger.write(
                    event="automatic_field_stopped",
                    release_timestamp_ms=auto_release_timestamp_ms,
                    estimate=motion_estimator.latest,
                    target=last_target,
                    intensity=current_intensity_ratio,
                    reason=reason,
                )
                print(
                    "[AUTO_RELEASE] automatic LOCAL_HOLD stopped: "
                    f"{reason}"
                )

            def current_log_mode() -> str:
                if not tracking_active:
                    return "FIXED"
                if cfg.enable_delay_compensation:
                    return "PID"
                return "PID_NO_DELAY"

            print("=================================================")
            if auto_release_trigger is None:
                print("  READY FOR MANUAL RELEASE HANDOFF.")
                print("  No ultrasound is emitted until the manual trigger.")
                print(
                    f"  Pinch/hold the sphere in view, press "
                    f"[{manual_release_key.upper()}], wait about "
                    f"{manual_release_pre_hold_sec:.1f}s, then release."
                )
            else:
                print("  READY FOR AUTOMATIC MEDIAPIPE RELEASE HANDOFF.")
                print("  No ultrasound is emitted before GRASPED -> RELEASED.")
                print(
                    f"  [{manual_release_key.upper()}] remains available as "
                    "a manual fallback trigger."
                )
            print(
                f"  Press [{false_release_reject_key.upper()}] to reject a "
                "false release and silence the field."
            )
            print("  Press [ENTER] only to PAUSE after the trigger.")
            print(
                f"  Press [{cfg.delay_compensation_toggle_key.upper()}] to "
                "toggle delay compensation."
            )
            if cfg.enable_base_move:
                print("  Use [Arrow keys] to move base XY.")
                print("  Use [PageUp/PageDown] to move base Z.")
            if cfg.enable_auto_demo:
                print(f"  Press [{cfg.demo_toggle_key.upper()}] to START/STOP square-Z demo.")
            if cfg.log_enabled:
                print(f"  Press [{cfg.log_trigger_key.upper()}] to START {cfg.log_duration_sec:.0f}s logging.")
            print("  Press [ESC] to EXIT and STOP ultrasound.")
            print("=================================================")

            # 5. Main loop
            while True:
                now_loop = time.time()
                dt_loop = max(1e-3, now_loop - prev_loop_time)
                prev_loop_time = now_loop

                # Key handling
                if is_key_pressed("esc"):
                    print("[INFO] ESC pressed. Exit.")
                    break

                false_release_reject_pressed = is_key_pressed(
                    false_release_reject_key
                )
                if (
                    false_release_reject_pressed
                    and not prev_false_release_reject_pressed
                ):
                    if release_observation_event is not None:
                        clear_release_observation(
                            "operator_rejected_false_release",
                            paper_status="invalid",
                        )
                        if auto_release_trigger is not None:
                            auto_release_trigger.reset()
                        print(
                            "[AUTO_RELEASE] false release rejected before "
                            "field activation."
                        )
                    elif auto_hold_active:
                        paper_trial_was_active = paper_logger.active
                        stop_automatic_hold(
                            "operator_rejected_false_release",
                            paper_status="invalid",
                        )
                        excluded_trial_id = None
                        if not paper_trial_was_active:
                            excluded_trial_id = (
                                paper_logger.exclude_recent_failed_trial(
                                    reason=(
                                        "operator_rejected_false_release_"
                                        "after_automatic_stop"
                                    )
                                )
                            )
                        if auto_release_trigger is not None:
                            auto_release_trigger.reset()
                        print(
                            "[AUTO_RELEASE] false release rejected; "
                            "ultrasound silenced."
                            + (
                                f" Excluded {excluded_trial_id}."
                                if excluded_trial_id is not None
                                else ""
                            )
                        )
                    else:
                        excluded_trial_id = (
                            paper_logger.exclude_recent_failed_trial(
                                reason=(
                                    "operator_rejected_false_release_"
                                    "after_automatic_stop"
                                )
                            )
                        )
                        if excluded_trial_id is None:
                            print(
                                "[AUTO_RELEASE] reject key ignored: no active "
                                "or recent failed automatic-release trial."
                            )
                        else:
                            print(
                                "[AUTO_RELEASE] recent false release excluded: "
                                f"{excluded_trial_id}"
                            )
                    time.sleep(0.1)
                prev_false_release_reject_pressed = (
                    false_release_reject_pressed
                )

                enter_pressed = is_key_pressed("enter")
                if enter_pressed and not prev_enter_pressed:
                    if tracking_active:
                        print("[INFO] >>> TRACKING PAUSED <<<")
                        if paper_logger.active:
                            paper_logger.finish_trial(
                                status=(
                                    "success"
                                    if paper_logger.success_reached
                                    else "failure"
                                    if paper_logger.field_started
                                    else "invalid"
                                ),
                                reason="tracking_paused_by_operator",
                                event_time_sec=time.perf_counter(),
                            )
                        tracking_active = False
                        auto_hold_active = False
                        auto_release_confirmed = False
                        capture_started_at = None
                        capture_stable_since = None
                        capture_brake_exit_since = None
                        capture_brake_exit_last_measurement_time = None
                        capture_settle_started_at = None
                        capture_settle_last_measurement_time = None
                        capture_settle_intensity_state = (
                            SETTLE_INTENSITY_NORMAL
                        )
                        control_mode = NORMAL_HOLD
                        local_hold_start_time = None
                        mode_transition_reason = "tracking_paused"
                        current_intensity_ratio = 0.0
                        set_tracking_target(
                            sender,
                            cfg,
                            last_target.x,
                            last_target.y,
                            last_target.z,
                            home,
                            current_radius,
                            current_intensity_ratio,
                        )
                    else:
                        print(
                            f"[INFO] ENTER ignored before trigger. "
                            f"Use [{manual_release_key.upper()}] to start local hold."
                        )

                    time.sleep(0.2)

                prev_enter_pressed = enter_pressed

                delay_toggle_pressed = is_key_pressed(cfg.delay_compensation_toggle_key)
                if delay_toggle_pressed and not prev_delay_toggle_pressed:
                    cfg.enable_delay_compensation = not cfg.enable_delay_compensation
                    state = "ON" if cfg.enable_delay_compensation else "OFF"
                    print(f"[INFO] Delay compensation: {state}")
                    if tracking_active:
                        controller.reset(last_target)
                    time.sleep(0.2)

                prev_delay_toggle_pressed = delay_toggle_pressed

                if cfg.enable_auto_demo:
                    demo_toggle_pressed = is_key_pressed(cfg.demo_toggle_key)

                    if demo_toggle_pressed and not prev_demo_toggle_pressed:
                        demo_active = not demo_active
                        demo.reset()

                        if demo_active:
                            print("[DEMO] Square-Z demo started.")
                        else:
                            print("[DEMO] Square-Z demo stopped.")

                    prev_demo_toggle_pressed = demo_toggle_pressed

                if cfg.enable_auto_demo and demo_active and tracking_active:
                    home = demo.update(dt_loop)
                    control_mode = NORMAL_HOLD
                    return_setpoint = HomePosition(
                        x=home.x,
                        y=home.y,
                        z=home.z,
                    )

                elif cfg.enable_base_move:
                    home = update_base_position(home, cfg, dt_loop)

                    if not tracking_active:
                        last_target = Target3D(home.x, home.y, home.z)

                if cfg.enable_radius_change:
                    radius_step = cfg.radius_change_speed_mm_s * dt_loop

                    # 右 or 上でradiusを大きくする
                    if is_key_pressed("right") or is_key_pressed("up"):
                        current_radius += radius_step

                    # 左 or 下でradiusを小さくする
                    if is_key_pressed("left") or is_key_pressed("down"):
                        current_radius -= radius_step

                    current_radius = float(
                        np.clip(
                            current_radius,
                            cfg.radius_min,
                            cfg.radius_max,
                        )
                    )

                if cfg.log_enabled:
                    log_trigger_pressed = is_key_pressed(cfg.log_trigger_key)

                    if log_trigger_pressed and not prev_log_trigger_pressed and not logger.active:
                        logger.start(current_log_mode())

                    prev_log_trigger_pressed = log_trigger_pressed

                logger.stop_if_finished()

                # Camera watchdog and software-synchronized frame pair
                now_camera = time.perf_counter()
                age_xy, age_z = frame_buffer.get_frame_ages(now_camera)

                if (
                    age_xy > cfg.camera_frame_timeout_sec
                    or age_z > cfg.camera_frame_timeout_sec
                ):
                    sender.stop()
                    raise RuntimeError(
                        "Camera watchdog timeout: "
                        f"XY age={age_xy:.3f}s, Z age={age_z:.3f}s"
                    )

                synced_pair = frame_buffer.get_synced_pair(
                    cfg.camera_sync_tolerance_sec
                )

                if synced_pair is None:
                    if now_camera - last_synced_pair_time > cfg.camera_frame_timeout_sec:
                        sender.stop()
                        raise RuntimeError(
                            "Camera synchronization timeout: no frame pair within "
                            f"{cfg.camera_sync_tolerance_sec * 1000.0:.1f} ms"
                        )
                    time.sleep(0.001)
                    continue

                (
                    frame_xy,
                    frame_xy_time,
                    frame_z,
                    frame_z_time,
                    frame_sync_skew,
                ) = synced_pair
                last_synced_pair_time = now_camera

                # get_synced_pair() は消費済みの新規フレームだけを返す。
                new_xy_fps_count += 1
                new_z_fps_count += 1

                frame_count += 1
                loop_fps_count += 1
                do_display = (frame_count % cfg.display_every_n_frames == 0)

                if do_display:
                    if frame_xy.ndim == 2:
                        frame_xy_bgr = cv2.cvtColor(frame_xy, cv2.COLOR_GRAY2BGR)
                    else:
                        frame_xy_bgr = frame_xy.copy()

                    if frame_z.ndim == 2:
                        frame_z_bgr = cv2.cvtColor(frame_z, cv2.COLOR_GRAY2BGR)
                    else:
                        frame_z_bgr = frame_z.copy()
                else:
                    frame_xy_bgr = None
                    frame_z_bgr = None

                now_fps = time.time()
                if now_fps - fps_start_time >= 1.0:
                    elapsed = now_fps - fps_start_time
                    loop_display_fps = loop_fps_count / elapsed
                    new_xy_display_fps = new_xy_fps_count / elapsed
                    new_z_display_fps = new_z_fps_count / elapsed

                    loop_fps_count = 0
                    new_xy_fps_count = 0
                    new_z_fps_count = 0
                    fps_start_time = now_fps

                # Detection XY
                det_xy = tracker_xy.detect(frame_xy)
                detected_xy = det_xy.detected
                u_xy = np.nan
                v_xy = np.nan
                x_mm = None
                y_mm = None
                stereo_cam_point = None
                stereo_autd_point = None

                if detected_xy:
                    u_xy, v_xy = det_xy.center

                if do_display:
                    draw_ball_detection(frame_xy_bgr, det_xy, tracking_active)

                # Detection Z
                det_z = tracker_z.detect(frame_z)
                detected_z = det_z.detected
                u_z = np.nan
                v_z = np.nan
                z_mm = None

                if detected_z:
                    u_z, v_z = det_z.center

                if (
                    stereo_triangulator is not None
                    and detected_xy
                    and detected_z
                    and np.isfinite(u_xy)
                    and np.isfinite(v_xy)
                    and np.isfinite(u_z)
                    and np.isfinite(v_z)
                ):
                    try:
                        stereo_cam_point = stereo_triangulator.triangulate_pixels(
                            (float(u_xy), float(v_xy)),
                            (float(u_z), float(v_z)),
                        )

                        if camera_to_autd is not None:
                            stereo_autd = camera_to_autd.apply(stereo_cam_point)
                            stereo_autd_point = stereo_autd
                            if cfg.use_stereo_position_for_control:
                                x_mm = float(stereo_autd[0])
                                y_mm = float(stereo_autd[1])
                                z_mm = float(stereo_autd[2])

                    except Exception as e:
                        print(f"[WARN] Stereo triangulation failed: {e}")

                motion_estimate = motion_estimator.latest
                if x_mm is not None and y_mm is not None and z_mm is not None:
                    measurement_time_sec = max(
                        float(frame_xy_time),
                        float(frame_z_time),
                    )
                    motion_estimate = motion_estimator.update(
                        measurement_time_sec,
                        x_mm,
                        y_mm,
                        z_mm,
                    )
                    history_added = recent_stereo_history.add(
                        measurement_time_sec,
                        x_mm,
                        y_mm,
                        z_mm,
                    )
                    if (
                        release_observation_event is not None
                        and history_added
                    ):
                        post_release_motion_estimator.update(
                            measurement_time_sec,
                            x_mm,
                            y_mm,
                            z_mm,
                        )

                if do_display:
                    draw_ball_detection(frame_z_bgr, det_z, tracking_active)
                    if stereo_cam_point is not None:
                        if stereo_autd_point is not None:
                            stereo_autd_relative = stereo_autd_point - np.array(
                                [
                                    float(base_center[0]),
                                    float(base_center[1]),
                                    float(base_center[2]),
                                ],
                                dtype=float,
                            )
                            stereo_label = (
                                "Stereo point: "
                                f"{stereo_autd_relative[0]:.1f}, "
                                f"{stereo_autd_relative[1]:.1f}, "
                                f"{stereo_autd_relative[2]:.1f}"
                            )
                        else:
                            stereo_label = (
                                "ST cam1: "
                                f"{stereo_cam_point.x:.1f}, "
                                f"{stereo_cam_point.y:.1f}, "
                                f"{stereo_cam_point.z:.1f}"
                            )
                        cv2.putText(
                            frame_xy_bgr,
                            stereo_label,
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame_z_bgr,
                            stereo_label,
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2,
                        )

                auto_release_status = None
                if auto_release_trigger is not None:
                    auto_release_status = auto_release_trigger.update(
                        frame_xy=frame_xy,
                        frame_xy_time=frame_xy_time,
                        ball_xy=det_xy,
                        frame_z=frame_z,
                        frame_z_time=frame_z_time,
                        ball_z=det_z,
                    )
                    if do_display:
                        auto_release_trigger.draw(
                            frame_xy_bgr,
                            frame_z_bgr,
                        )

                if cfg.enable_z_intensity_boost and tracking_active:
                    target_intensity_ratio = compute_z_intensity_boost_ratio(
                        cfg,
                        home.z,
                        z_mm,
                    )

                else:
                    target_intensity_ratio = float(cfg.static_intensity_ratio)
                
                method = "XY+Z" if (detected_xy and detected_z) else "PARTIAL"
                hold_telemetry = HoldTelemetry(
                    z_drop_mm=None if z_mm is None else float(home.z - z_mm),
                    temporary_home=return_setpoint,
                    mode_transition_reason=mode_transition_reason,
                )

                manual_release_pressed = is_key_pressed(manual_release_key)
                if manual_release_pressed and not prev_manual_release_pressed:
                    if arm_local_hold(
                        x_mm,
                        y_mm,
                        z_mm,
                        reason="manual_release_trigger",
                        label="MANUAL_RELEASE",
                        prefer_filtered_z=True,
                    ):
                        auto_hold_active = False
                        auto_release_confirmed = False
                        if release_observation_event is not None:
                            clear_release_observation(
                                "manual_release_trigger",
                                log_event=False,
                            )
                        print(
                            f"Release after ~{manual_release_pre_hold_sec:.1f}s."
                        )

                prev_manual_release_pressed = manual_release_pressed

                if (
                    auto_release_status is not None
                    and auto_release_status.candidate_event is not None
                    and not tracking_active
                    and release_observation_event is None
                ):
                    start_release_observation(
                        auto_release_status.candidate_event
                    )

                if (
                    auto_release_status is not None
                    and auto_release_status.candidate_cancelled
                ):
                    if release_observation_event is not None:
                        clear_release_observation(
                            "release_candidate_contact_returned"
                        )
                    elif auto_hold_active and not auto_release_confirmed:
                        stop_automatic_hold(
                            "release_candidate_contact_returned"
                        )

                if (
                    auto_release_status is not None
                    and auto_release_status.release_confirmed
                ):
                    if release_observation_event is not None:
                        release_observation_confirmed = True
                        post_release_estimate = (
                            post_release_motion_estimator.latest
                        )
                        capture_logger.write(
                            event="release_confirmed",
                            release_timestamp_ms=int(
                                release_observation_event.timestamp_ms
                            ),
                            estimate=post_release_estimate,
                            target=None,
                            intensity=0.0,
                            reason=(
                                f"camera={auto_release_status.decision_camera};"
                                "field_off_waiting_for_post_release_fit"
                            ),
                        )
                    elif (
                        auto_release_status.event is not None
                        and not tracking_active
                    ):
                        # Fallback if the candidate edge was missed. Start the
                        # post-release window now instead of using pre-release
                        # motion from the continuously running estimator.
                        start_release_observation(
                            auto_release_status.event
                        )
                        release_observation_confirmed = True

                # The post-release fit and physical-motion gate start at the
                # first explicit separation candidate.  They therefore run in
                # parallel with MediaPipe's release-confirmation timer, while
                # acoustic output remains gated by confirmation below.
                if release_observation_event is not None:
                    now_observation = time.perf_counter()
                    observation_timeout_sec = float(
                        getattr(
                            cfg,
                            "auto_release_post_release_timeout_sec",
                            0.100,
                        )
                    )
                    gate_timeout_sec = float(
                        getattr(
                            cfg,
                            "auto_release_gate_timeout_sec",
                            0.120,
                        )
                    )
                    observation_estimate = (
                        post_release_motion_estimator.latest
                    )
                    if (
                        release_observation_started_at is not None
                        and release_motion_ready_at is None
                        and now_observation - release_observation_started_at
                        > observation_timeout_sec
                    ):
                        clear_release_observation(
                            "post_release_motion_fit_timeout",
                            paper_status="invalid",
                        )
                    elif (
                        observation_estimate is not None
                        and observation_estimate.velocity_ready
                    ):
                        observation_event = release_observation_event
                        if release_motion_ready_at is None:
                            release_motion_ready_at = now_observation
                            paper_logger.mark_motion_ready(now_observation)
                            capture_logger.write(
                                event="post_release_motion_ready",
                                release_timestamp_ms=int(
                                    observation_event.timestamp_ms
                                ),
                                event_time_sec=now_observation,
                                estimate=observation_estimate,
                                target=None,
                                intensity=0.0,
                                reason=(
                                    f"samples={observation_estimate.sample_count};"
                                    f"span_ms="
                                    f"{observation_estimate.history_span_sec * 1000.0:.1f};"
                                    "free_flight_z_model=1;"
                                    "extra_field_delay_ms="
                                    f"{float(getattr(cfg, 'si2026_extra_field_delay_sec', 0.0)) * 1000.0:.1f}"
                                ),
                            )

                        if (
                            release_gate_origin_z_mm is None
                            and observation_estimate is not None
                        ):
                            release_gate_origin_z_mm = float(
                                observation_estimate.z_mm
                            )

                        if (
                            release_gate_last_measurement_time
                            != observation_estimate.measurement_time_sec
                        ):
                            gate_result = update_release_motion_gate(
                                origin_z_mm=release_gate_origin_z_mm,
                                current_z_mm=observation_estimate.z_mm,
                                vz_mm_s=observation_estimate.vz_mm_s,
                                maximum_vz_mm_s=float(
                                    getattr(
                                        cfg,
                                        "auto_release_gate_max_vz_mm_s",
                                        -100.0,
                                    )
                                ),
                                minimum_downward_displacement_mm=float(
                                    getattr(
                                        cfg,
                                        "auto_release_gate_min_drop_mm",
                                        2.5,
                                    )
                                ),
                                required_consecutive_estimates=int(
                                    getattr(
                                        cfg,
                                        "auto_release_gate_consecutive_estimates",
                                        3,
                                    )
                                ),
                                previous_consecutive_count=(
                                    release_gate_consecutive_count
                                ),
                            )
                            release_gate_last_measurement_time = (
                                observation_estimate.measurement_time_sec
                            )
                            release_gate_consecutive_count = (
                                gate_result.consecutive_count
                            )
                            release_gate_downward_displacement_mm = (
                                gate_result.downward_displacement_mm
                            )
                            paper_logger.update_release_gate_observation(
                                origin_z_mm=release_gate_origin_z_mm,
                                estimate=observation_estimate,
                                downward_displacement_mm=(
                                    gate_result.downward_displacement_mm
                                ),
                                evidence_seen=(
                                    gate_result.velocity_condition_met
                                    and gate_result.displacement_condition_met
                                ),
                            )
                            if gate_result.passed and not release_gate_passed:
                                release_gate_passed = True
                                paper_logger.mark_release_gate_passed(
                                    event_time_sec=now_observation,
                                    estimate=observation_estimate,
                                    downward_displacement_mm=(
                                        gate_result.downward_displacement_mm
                                    ),
                                    consecutive_count=(
                                        gate_result.consecutive_count
                                    ),
                                )
                                capture_logger.write(
                                    event="release_motion_gate_passed",
                                    release_timestamp_ms=int(
                                        observation_event.timestamp_ms
                                    ),
                                    event_time_sec=now_observation,
                                    estimate=observation_estimate,
                                    target=None,
                                    intensity=0.0,
                                    reason=(
                                        f"vz_threshold_mm_s="
                                        f"{float(getattr(cfg, 'auto_release_gate_max_vz_mm_s', -100.0)):.1f};"
                                        f"drop_threshold_mm="
                                        f"{float(getattr(cfg, 'auto_release_gate_min_drop_mm', 2.5)):.2f};"
                                        f"drop_mm="
                                        f"{gate_result.downward_displacement_mm:.2f};"
                                        f"consecutive="
                                        f"{gate_result.consecutive_count}"
                                    ),
                                )

                        extra_field_delay_sec = max(
                            0.0,
                            float(
                                getattr(
                                    cfg,
                                    "si2026_extra_field_delay_sec",
                                    0.0,
                                )
                            ),
                        )
                        extra_delay_complete = (
                            now_observation - release_motion_ready_at
                            >= extra_field_delay_sec
                        )
                        activation_lower_z_mm = float(cfg.z_min) + float(
                            getattr(
                                cfg,
                                "auto_release_z_workspace_margin_mm",
                                10.0,
                            )
                        )
                        if (
                            extra_field_delay_sec > 0.0
                            and observation_estimate.z_mm
                            <= activation_lower_z_mm
                        ):
                            clear_release_observation(
                                "no_field_workspace_guard_during_observation",
                                paper_status="failure",
                            )
                        elif (
                            release_observation_started_at is not None
                            and not release_gate_passed
                            and now_observation
                            - release_observation_started_at
                            > gate_timeout_sec
                        ):
                            clear_release_observation(
                                "release_motion_gate_timeout",
                                paper_status="invalid",
                            )
                            if auto_release_trigger is not None:
                                auto_release_trigger.reset()
                        elif (
                            capture_activation_ready(
                                release_confirmed=(
                                    release_observation_confirmed
                                ),
                                motion_gate_passed=release_gate_passed,
                                extra_delay_complete=extra_delay_complete,
                            )
                            and arm_capture_align(
                                observation_event,
                                observation_estimate,
                            )
                        ):
                            auto_release_confirmed = True
                            clear_release_observation(
                                "post_release_fit_handed_to_capture",
                                log_event=False,
                                finish_paper_trial=False,
                                reset_gate_state=False,
                            )
                            # Re-arm only after the field starts. A later
                            # confirmed grasp then stops every capture state.
                            auto_release_trigger.reset()

                if (
                    auto_release_status is not None
                    and release_observation_event is not None
                    and should_stop_automatic_hold(
                        True,
                        auto_release_status.decision_state,
                    )
                ):
                    clear_release_observation(
                        "regrasp_detected_during_post_release_observation",
                        paper_status="invalid",
                    )

                if (
                    auto_release_status is not None
                    and auto_release_confirmed
                    and should_stop_automatic_hold(
                        auto_hold_active,
                        auto_release_status.decision_state,
                    )
                ):
                    stop_automatic_hold(
                        "regrasp_detected_during_auto_hold"
                    )

                if auto_hold_active:
                    now_capture_guard = time.perf_counter()
                    maximum_capture_age_sec = float(
                        getattr(
                            cfg,
                            "auto_release_capture_measurement_timeout_sec",
                            0.100,
                        )
                    )
                    if (
                        motion_estimate is None
                        or now_capture_guard
                        - motion_estimate.measurement_time_sec
                        > maximum_capture_age_sec
                    ):
                        stop_automatic_hold(
                            "capture_stereo_measurement_timeout"
                        )
                    elif control_mode == CAPTURE_ALIGN:
                        brake_exit_ready = False
                        if (
                            capture_brake_exit_last_measurement_time
                            != motion_estimate.measurement_time_sec
                        ):
                            (
                                capture_brake_exit_since,
                                brake_exit_ready,
                            ) = update_brake_exit_stability(
                                now_sec=now_capture_guard,
                                vz_mm_s=motion_estimate.vz_mm_s,
                                release_confirmed=auto_release_confirmed,
                                stable_since_sec=capture_brake_exit_since,
                                minimum_vz_mm_s=float(
                                    getattr(
                                        cfg,
                                        "auto_release_brake_exit_vz_mm_s",
                                        -30.0,
                                    )
                                ),
                                required_duration_sec=float(
                                    getattr(
                                        cfg,
                                        "auto_release_brake_exit_stable_sec",
                                        0.015,
                                    )
                                ),
                            )
                            capture_brake_exit_last_measurement_time = (
                                motion_estimate.measurement_time_sec
                            )
                        capture_timed_out = (
                            capture_started_at is not None
                            and now_capture_guard - capture_started_at
                            > float(
                                getattr(
                                    cfg,
                                    "auto_release_capture_timeout_sec",
                                    1.0,
                                )
                            )
                        )
                        minimum_alignment_duration_sec = float(
                            getattr(
                                cfg,
                                "auto_release_minimum_alignment_time_sec",
                                0.120,
                            )
                        )
                        minimum_alignment_complete = (
                            capture_braking_plan is not None
                            and now_capture_guard
                            - capture_braking_plan.start_time_sec
                            >= minimum_alignment_duration_sec
                        )
                        if (
                            brake_exit_ready
                            and minimum_alignment_complete
                        ):
                            enter_capture_settle(
                                "measured_vz_near_zero_after_minimum_alignment",
                                motion_estimate,
                                event_time_sec=now_capture_guard,
                            )
                        elif capture_timed_out:
                            if auto_release_confirmed:
                                enter_capture_settle(
                                    "capture_align_time_limit",
                                    motion_estimate,
                                    event_time_sec=now_capture_guard,
                                )
                            else:
                                stop_automatic_hold(
                                    "release_confirmation_timeout"
                                )

                if (
                    pending_capture_command_seq is not None
                    and not pending_capture_sent_logged
                ):
                    sent = sender.get_first_sent_at_or_after(
                        pending_capture_command_seq
                    )
                    if sent is not None:
                        pending_capture_sent_logged = True
                        capture_logger.write(
                            event="capture_field_sent",
                            release_timestamp_ms=auto_release_timestamp_ms,
                            event_time_sec=sent.sent_time_sec,
                            estimate=capture_start_estimate,
                            prediction=capture_initial_prediction,
                            target=Target3D(
                                sent.target.x,
                                sent.target.y,
                                sent.target.z,
                            ),
                            intensity=sent.target.intensity_ratio,
                            command_seq=sent.sequence,
                            reason=(
                                f"candidate_seq={pending_capture_command_seq}"
                                if sent.sequence == pending_capture_command_seq
                                else (
                                    f"candidate_seq={pending_capture_command_seq};"
                                    f"sent_seq={sent.sequence};"
                                    "candidate_command_superseded=1"
                                )
                            ),
                        )
                        if (
                            capture_start_estimate is not None
                            and capture_initial_prediction is not None
                            and auto_release_timestamp_ms is not None
                        ):
                            delay_logger.write(
                                release_timestamp_ms=auto_release_timestamp_ms,
                                estimate=capture_start_estimate,
                                prediction=capture_initial_prediction,
                                sent=sent,
                                candidate_command_seq=(
                                    pending_capture_command_seq
                                ),
                                configured_prediction_delay_sec=max(
                                    0.0,
                                    capture_initial_prediction.horizon_sec
                                    - max(
                                        0.0,
                                        float(capture_started_at)
                                        - capture_start_estimate.measurement_time_sec,
                                    ),
                                ),
                            )
                            capture_onset_prediction = predict_capture_position(
                                capture_start_estimate,
                                now_sec=sent.sent_time_sec,
                                actuation_delay_sec=0.0,
                                gravity_mm_s2=float(cfg.gravity_mm_s2),
                                apply_gravity=True,
                            )
                            maximum_capture_intensity = float(
                                getattr(
                                    cfg,
                                    "auto_release_max_intensity_ratio",
                                    0.8,
                                )
                            )
                            maximum_capture_force_mn = (
                                measured_force_for_intensity_mN(
                                    cfg,
                                    maximum_capture_intensity,
                                )
                            )
                            capture_braking_plan = make_braking_plan(
                                start_time_sec=sent.sent_time_sec,
                                initial_z_mm=capture_onset_prediction.z_mm,
                                initial_vz_mm_s=(
                                    capture_onset_prediction.vz_mm_s
                                ),
                                nominal_duration_sec=float(
                                    getattr(
                                        cfg,
                                        "auto_release_nominal_stop_time_sec",
                                        0.120,
                                    )
                                ),
                                maximum_duration_sec=float(
                                    getattr(
                                        cfg,
                                        "auto_release_capture_timeout_sec",
                                        1.0,
                                    )
                                ),
                                minimum_duration_sec=float(
                                    getattr(
                                        cfg,
                                        "auto_release_min_stop_time_sec",
                                        0.050,
                                    )
                                ),
                                ball_mass_kg=float(cfg.ball_mass_kg),
                                gravity_mm_s2=float(cfg.gravity_mm_s2),
                                maximum_force_mN=maximum_capture_force_mn,
                                z_min_mm=float(cfg.z_min),
                                workspace_margin_mm=float(
                                    getattr(
                                        cfg,
                                        "auto_release_z_workspace_margin_mm",
                                        10.0,
                                    )
                                ),
                                buoyancy_mN=capture_buoyancy_mN(cfg),
                            )
                            capture_logger.write(
                                event="capture_braking_plan_started",
                                release_timestamp_ms=(
                                    auto_release_timestamp_ms
                                ),
                                event_time_sec=sent.sent_time_sec,
                                estimate=capture_start_estimate,
                                prediction=capture_onset_prediction,
                                target=Target3D(
                                    sent.target.x,
                                    sent.target.y,
                                    sent.target.z,
                                ),
                                intensity=sent.target.intensity_ratio,
                                command_seq=sent.sequence,
                                reason=(
                                    f"duration_ms="
                                    f"{capture_braking_plan.duration_sec * 1000.0:.1f};"
                                    "xy_control=fixed_anchor_pd;"
                                    f"xy_anchor=("
                                    f"{capture_xy_anchor[0]:.2f},"
                                    f"{capture_xy_anchor[1]:.2f});"
                                    f"stop_z_mm="
                                    f"{capture_braking_plan.stop_z_mm:.2f};"
                                    f"distance_mm="
                                    f"{capture_braking_plan.stopping_distance_mm:.2f};"
                                    f"required_force_mN="
                                    f"{capture_braking_plan.required_force_mN:.3f};"
                                    f"max_force_mN="
                                    f"{capture_braking_plan.maximum_force_mN:.3f};"
                                    f"feasible="
                                    f"{int(capture_braking_plan.feasible)};"
                                    f"force_saturated="
                                    f"{int(capture_braking_plan.force_saturated)};"
                                    f"workspace_limited="
                                    f"{int(capture_braking_plan.workspace_limited)};"
                                    f"timeout_limited="
                                    f"{int(capture_braking_plan.timeout_limited)}"
                                ),
                            )
                            paper_logger.mark_field_sent(
                                field_time_sec=sent.sent_time_sec,
                                prediction=capture_onset_prediction,
                                anchor=capture_xy_anchor,
                                plan=capture_braking_plan,
                            )

                paper_hold_elapsed_sec = (
                    paper_logger.elapsed_from_release_sec()
                )
                paper_now_sec = time.perf_counter()
                paper_outside_z_workspace = bool(
                    motion_estimate is not None
                    and (
                        motion_estimate.z_mm < float(cfg.z_min)
                        or motion_estimate.z_mm > float(cfg.z_max)
                    )
                )
                if (
                    paper_logger.active
                    and not paper_logger.success_reached
                    and release_hold_success_reached(
                        elapsed_from_release_sec=paper_hold_elapsed_sec,
                        required_hold_sec=float(
                            getattr(cfg, "si2026_success_hold_sec", 5.0)
                        ),
                        field_started=paper_logger.field_started,
                        automatic_hold_active=auto_hold_active,
                    )
                ):
                    paper_logger.mark_success_reached(paper_now_sec)
                    capture_logger.write(
                        event="success_criterion_reached",
                        release_timestamp_ms=auto_release_timestamp_ms,
                        event_time_sec=paper_now_sec,
                        estimate=motion_estimate,
                        target=last_target,
                        intensity=current_intensity_ratio,
                        reason=(
                            f"required_hold_sec="
                            f"{float(getattr(cfg, 'si2026_success_hold_sec', 5.0)):.3f};"
                            "logging_continues_until_observation_end"
                        ),
                    )

                if (
                    paper_logger.active
                    and paper_logger.field_started
                    and paper_outside_z_workspace
                ):
                    paper_logger.finish_trial(
                        status=(
                            "success"
                            if paper_logger.success_reached
                            else "failure"
                        ),
                        reason=(
                            "post_success_sphere_left_z_workspace"
                            if paper_logger.success_reached
                            else "sphere_left_z_workspace_before_success"
                        ),
                        event_time_sec=paper_now_sec,
                    )
                elif (
                    paper_logger.active
                    and paper_logger.success_reached
                    and paper_hold_elapsed_sec is not None
                    and paper_hold_elapsed_sec
                    >= float(getattr(cfg, "si2026_observation_sec", 10.0))
                ):
                    paper_logger.finish_trial(
                        status="success",
                        reason="post_success_observation_completed",
                        event_time_sec=paper_now_sec,
                    )

                # Control
                if tracking_active:
                    meas = Measurement3D(
                        detected_xy=detected_xy and x_mm is not None and y_mm is not None,
                        detected_z=detected_z and z_mm is not None,
                        x=x_mm,
                        y=y_mm,
                        z=z_mm,
                        t_xy=frame_xy_time,
                        t_z=frame_z_time,
                    )

                    if meas.detected_xy or meas.detected_z:
                        # An automatic release is held at its temporary position
                        # first. Once the dwell time expires, reuse the existing
                        # speed-limited RETURN_TO_HOME controller.
                        return_delay_sec = float(
                            getattr(
                                cfg,
                                "auto_release_local_hold_before_return_sec",
                                3.0,
                            )
                        )
                        minimum_return_from_release_sec = float(
                            getattr(
                                cfg,
                                "auto_release_min_return_from_release_sec",
                                0.0,
                            )
                        )
                        release_elapsed_for_return_sec = (
                            None
                            if auto_release_timestamp_ms is None
                            else max(
                                0.0,
                                time.perf_counter()
                                - float(auto_release_timestamp_ms) / 1000.0,
                            )
                        )
                        if (
                            control_mode == LOCAL_HOLD
                            and release_elapsed_for_return_sec is not None
                            and release_elapsed_for_return_sec
                            >= minimum_return_from_release_sec
                            and should_start_return_home(
                                now_sec=time.time(),
                                local_hold_started_sec=local_hold_start_time,
                                delay_sec=return_delay_sec,
                                automatic_hold=auto_hold_active,
                            )
                        ):
                            control_mode = RETURN_TO_HOME
                            local_hold_start_time = None
                            mode_transition_reason = (
                                "auto_release_local_hold_elapsed"
                            )
                            capture_logger.write(
                                event="local_hold_to_return_home",
                                release_timestamp_ms=auto_release_timestamp_ms,
                                estimate=motion_estimate,
                                target=last_target,
                                intensity=current_intensity_ratio,
                                reason=(
                                    f"local_hold_elapsed_sec="
                                    f"{return_delay_sec:.3f};"
                                    f"release_elapsed_sec="
                                    f"{release_elapsed_for_return_sec:.3f};"
                                    f"minimum_release_elapsed_sec="
                                    f"{minimum_return_from_release_sec:.3f};"
                                    f"home=({home.x:.2f},{home.y:.2f},"
                                    f"{home.z:.2f})"
                                ),
                            )
                            print(
                                "[AUTO_RELEASE] LOCAL_HOLD -> "
                                "RETURN_TO_HOME "
                                f"after {return_delay_sec:.1f}s; "
                                f"home=({home.x:.1f}, {home.y:.1f}, "
                                f"{home.z:.1f}) mm"
                            )

                        # 1. RETURN_TO_HOME中は一時基準位置をゆっくりhomeへ戻す。
                        #    LOCAL_HOLD中はreturn_setpointを一時基準としてその場保持する。
                        if control_mode == RETURN_TO_HOME:
                            return_setpoint = HomePosition(
                                x=move_towards(
                                    return_setpoint.x,
                                    home.x,
                                    cfg.return_home_speed_xy_mm_s * dt_loop,
                                ),
                                y=move_towards(
                                    return_setpoint.y,
                                    home.y,
                                    cfg.return_home_speed_xy_mm_s * dt_loop,
                                ),
                                z=move_towards(
                                    return_setpoint.z,
                                    home.z,
                                    cfg.return_home_speed_z_mm_s * dt_loop,
                                ),
                            )

                            control_home = return_setpoint
                        elif control_mode in [
                            LOCAL_HOLD,
                            CAPTURE_ALIGN,
                            CAPTURE_SETTLE,
                        ]:
                            control_home = return_setpoint
                        else:
                            control_home = home

                        # 2. まず通常PIDを計算する。
                        controller_now_sec = time.perf_counter()
                        target, debug = controller.update(
                            meas,
                            control_home,
                            now_sec=controller_now_sec,
                            actuation_delay_sec=estimated_actuation_delay_sec(),
                        )

                        vx_now = float(debug.vx)
                        vy_now = float(debug.vy)
                        vz_now = float(debug.vz)

                        hold_telemetry.vz_mm_s = vz_now

                        # 3. 自動解放捕捉中は段階別に焦点群中心と強度を決める。
                        capture_intensity_ratio = None
                        if (
                            control_mode == CAPTURE_ALIGN
                            and motion_estimate is not None
                        ):
                            now_capture = time.perf_counter()
                            trajectory_reference = None
                            trajectory_correction_z = 0.0
                            if capture_braking_plan is not None:
                                trajectory_reference = braking_reference_at(
                                    capture_braking_plan,
                                    now_sec=now_capture,
                                )
                                desired_target_z, trajectory_correction_z = (
                                    trajectory_target_z(
                                        reference_z_mm=(
                                            trajectory_reference.z_mm
                                        ),
                                        reference_vz_mm_s=(
                                            trajectory_reference.vz_mm_s
                                        ),
                                        measured_z_mm=motion_estimate.z_mm,
                                        measured_vz_mm_s=(
                                            motion_estimate.vz_mm_s
                                        ),
                                        position_gain=float(
                                            getattr(
                                                cfg,
                                                "auto_release_trajectory_kp_z",
                                                0.30,
                                            )
                                        ),
                                        velocity_gain=float(
                                            getattr(
                                                cfg,
                                                "auto_release_trajectory_kd_z",
                                                0.01,
                                            )
                                        ),
                                        maximum_correction_mm=float(
                                            getattr(
                                                cfg,
                                                "auto_release_trajectory_max_correction_mm",
                                                5.0,
                                            )
                                        ),
                                    )
                                )
                                prediction = PredictedCapture(
                                    horizon_sec=(
                                        trajectory_reference.elapsed_sec
                                    ),
                                    x_mm=float(
                                        capture_xy_anchor[0]
                                        if capture_xy_anchor is not None
                                        else capture_initial_prediction.x_mm
                                    ),
                                    y_mm=float(
                                        capture_xy_anchor[1]
                                        if capture_xy_anchor is not None
                                        else capture_initial_prediction.y_mm
                                    ),
                                    z_mm=float(trajectory_reference.z_mm),
                                    vx_mm_s=0.0,
                                    vy_mm_s=0.0,
                                    vz_mm_s=float(
                                        trajectory_reference.vz_mm_s
                                    ),
                                )
                            else:
                                # The first field has not been confirmed as sent
                                # yet. Keep the one-shot free-fall intercept target.
                                prediction = capture_initial_prediction
                                desired_target_z = last_target.z

                            upward_vz_threshold = float(
                                getattr(
                                    cfg,
                                    "auto_release_upward_vz_mm_s",
                                    20.0,
                                )
                            )
                            upward_brake_now = (
                                float(motion_estimate.vz_mm_s)
                                > upward_vz_threshold
                            )
                            limited_target_z = limit_upward_capture_target_z(
                                desired_target_z,
                                last_target.z,
                                upward_brake_active=upward_brake_now,
                            )
                            xy_pd_target = None
                            if (
                                capture_braking_plan is not None
                                and capture_xy_anchor is not None
                            ):
                                xy_pd_target = anchored_pd_target_xy(
                                    anchor_x_mm=capture_xy_anchor[0],
                                    anchor_y_mm=capture_xy_anchor[1],
                                    ball_x_mm=motion_estimate.x_mm,
                                    ball_y_mm=motion_estimate.y_mm,
                                    vx_mm_s=motion_estimate.vx_mm_s,
                                    vy_mm_s=motion_estimate.vy_mm_s,
                                    position_gain=float(
                                        getattr(
                                            cfg,
                                            "auto_release_capture_xy_kp",
                                            cfg.kp_xy,
                                        )
                                    ),
                                    velocity_gain_sec=float(
                                        getattr(
                                            cfg,
                                            "auto_release_capture_xy_kd",
                                            cfg.kd_xy,
                                        )
                                    ),
                                    prediction_horizon_sec=(
                                        timestamp_prediction_horizon_sec(
                                            motion_estimate.measurement_time_sec,
                                            now_capture,
                                        )
                                    ),
                                    maximum_distance_mm=float(
                                        getattr(
                                            cfg,
                                            "auto_release_xy_capture_radius_mm",
                                            15.0,
                                        )
                                    ),
                                )
                                capture_target_x = xy_pd_target.x_mm
                                capture_target_y = xy_pd_target.y_mm
                                xy_pd_vxy = float(
                                    np.hypot(
                                        motion_estimate.vx_mm_s,
                                        motion_estimate.vy_mm_s,
                                    )
                                )
                            else:
                                # Preserve the one-shot multi-frame prediction
                                # until the sender confirms that the first field
                                # has actually been emitted.
                                capture_target_x = float(prediction.x_mm)
                                capture_target_y = float(prediction.y_mm)
                            target = Target3D(
                                x=capture_target_x,
                                y=capture_target_y,
                                z=float(
                                    np.clip(
                                        limited_target_z,
                                        cfg.z_min,
                                        cfg.z_max,
                                    )
                                ),
                            )
                            if (
                                xy_pd_target is not None
                                and xy_pd_target.limited
                                and not capture_xy_pd_saturation_logged
                            ):
                                capture_xy_pd_saturation_logged = True
                                capture_logger.write(
                                    event="capture_xy_pd_saturated",
                                    release_timestamp_ms=(
                                        auto_release_timestamp_ms
                                    ),
                                    event_time_sec=now_capture,
                                    estimate=motion_estimate,
                                    target=target,
                                    intensity=current_intensity_ratio,
                                    reason=(
                                        f"anchor=("
                                        f"{xy_pd_target.anchor_x_mm:.2f},"
                                        f"{xy_pd_target.anchor_y_mm:.2f});"
                                        f"requested=("
                                        f"{xy_pd_target.requested_x_mm:.2f},"
                                        f"{xy_pd_target.requested_y_mm:.2f});"
                                        f"limited_distance_mm="
                                        f"{xy_pd_target.distance_mm:.2f};"
                                        f"vxy_mm_s="
                                        f"{xy_pd_vxy:.2f}"
                                    ),
                                )
                            force_command = force_command_for_vz(
                                motion_estimate.vz_mm_s,
                                current_intensity=current_intensity_ratio,
                            )
                            capture_intensity_ratio = float(
                                force_command.commanded_intensity
                            )
                            if upward_brake_now:
                                capture_intensity_ratio = float(
                                    getattr(
                                        cfg,
                                        "auto_release_upward_intensity_ratio",
                                        0.5,
                                    )
                                )
                            vx_now = float(motion_estimate.vx_mm_s)
                            vy_now = float(motion_estimate.vy_mm_s)
                            vz_now = float(motion_estimate.vz_mm_s)
                            hold_telemetry.vz_mm_s = vz_now
                            hold_telemetry.predicted_x_mm = prediction.x_mm
                            hold_telemetry.predicted_y_mm = prediction.y_mm
                            hold_telemetry.predicted_z_mm = prediction.z_mm
                            hold_telemetry.predicted_vz_mm_s = prediction.vz_mm_s
                            hold_telemetry.commanded_intensity = (
                                capture_intensity_ratio
                            )
                            hold_telemetry.required_force_mN = (
                                force_command.required_force_mN
                            )
                            hold_telemetry.capture_force_saturated = (
                                force_command.saturated
                            )
                            hold_telemetry.xy_distance_to_target_mm = float(
                                np.hypot(
                                    motion_estimate.x_mm - target.x,
                                    motion_estimate.y_mm - target.y,
                                )
                            )

                            if (
                                capture_braking_plan is not None
                                and trajectory_reference is not None
                                and auto_release_timestamp_ms is not None
                                and (
                                    capture_last_trajectory_log_measurement_time
                                    != motion_estimate.measurement_time_sec
                                )
                            ):
                                trajectory_logger.write(
                                    release_timestamp_ms=(
                                        auto_release_timestamp_ms
                                    ),
                                    estimate=motion_estimate,
                                    plan=capture_braking_plan,
                                    reference=trajectory_reference,
                                    target=target,
                                    target_z_correction_mm=(
                                        trajectory_correction_z
                                    ),
                                    xy_distance_mm=(
                                        hold_telemetry.xy_distance_to_target_mm
                                    ),
                                    commanded_intensity=(
                                        capture_intensity_ratio
                                    ),
                                    upward_brake_active=upward_brake_now,
                                )
                                capture_last_trajectory_log_measurement_time = (
                                    motion_estimate.measurement_time_sec
                                )

                            if upward_brake_now != capture_upward_brake_active:
                                capture_upward_brake_active = upward_brake_now
                                capture_logger.write(
                                    event=(
                                        "capture_upward_brake_started"
                                        if upward_brake_now
                                        else "capture_upward_brake_ended"
                                    ),
                                    release_timestamp_ms=(
                                        auto_release_timestamp_ms
                                    ),
                                    event_time_sec=now_capture,
                                    estimate=motion_estimate,
                                    prediction=prediction,
                                    target=target,
                                    intensity=capture_intensity_ratio,
                                    reason=(
                                        f"vz>{upward_vz_threshold:.1f}"
                                        if upward_brake_now
                                        else f"vz<={upward_vz_threshold:.1f}"
                                    ),
                                )

                        elif (
                            control_mode == CAPTURE_SETTLE
                            and motion_estimate is not None
                        ):
                            now_capture = time.perf_counter()
                            vx_now = float(motion_estimate.vx_mm_s)
                            vy_now = float(motion_estimate.vy_mm_s)
                            vz_now = float(motion_estimate.vz_mm_s)
                            vxy_now = float(np.hypot(vx_now, vy_now))
                            if capture_xy_anchor is not None:
                                settle_xy_pd_target = anchored_pd_target_xy(
                                    anchor_x_mm=capture_xy_anchor[0],
                                    anchor_y_mm=capture_xy_anchor[1],
                                    ball_x_mm=motion_estimate.x_mm,
                                    ball_y_mm=motion_estimate.y_mm,
                                    vx_mm_s=vx_now,
                                    vy_mm_s=vy_now,
                                    position_gain=float(
                                        getattr(
                                            cfg,
                                            "auto_release_capture_xy_kp",
                                            cfg.kp_xy,
                                        )
                                    ),
                                    velocity_gain_sec=float(
                                        getattr(
                                            cfg,
                                            "auto_release_capture_xy_kd",
                                            cfg.kd_xy,
                                        )
                                    ),
                                    prediction_horizon_sec=(
                                        timestamp_prediction_horizon_sec(
                                            motion_estimate.measurement_time_sec,
                                            now_capture,
                                        )
                                    ),
                                    maximum_distance_mm=float(
                                        getattr(
                                            cfg,
                                            "auto_release_xy_capture_radius_mm",
                                            15.0,
                                        )
                                    ),
                                )
                                target = Target3D(
                                    x=settle_xy_pd_target.x_mm,
                                    y=settle_xy_pd_target.y_mm,
                                    z=target.z,
                                )
                                if (
                                    settle_xy_pd_target.limited
                                    and not capture_xy_pd_saturation_logged
                                ):
                                    capture_xy_pd_saturation_logged = True
                                    capture_logger.write(
                                        event="capture_xy_pd_saturated",
                                        release_timestamp_ms=(
                                            auto_release_timestamp_ms
                                        ),
                                        event_time_sec=now_capture,
                                        estimate=motion_estimate,
                                        target=target,
                                        intensity=current_intensity_ratio,
                                        reason=(
                                            "phase=CAPTURE_SETTLE;"
                                            f"anchor=("
                                            f"{settle_xy_pd_target.anchor_x_mm:.2f},"
                                            f"{settle_xy_pd_target.anchor_y_mm:.2f});"
                                            f"requested=("
                                            f"{settle_xy_pd_target.requested_x_mm:.2f},"
                                            f"{settle_xy_pd_target.requested_y_mm:.2f});"
                                            f"limited_distance_mm="
                                            f"{settle_xy_pd_target.distance_mm:.2f};"
                                            f"vxy_mm_s={vxy_now:.2f}"
                                        ),
                                    )
                            xy_distance_now = float(
                                np.hypot(
                                    motion_estimate.x_mm - return_setpoint.x,
                                    motion_estimate.y_mm - return_setpoint.y,
                                )
                            )
                            z_tracking_error = float(
                                abs(motion_estimate.z_mm - return_setpoint.z)
                            )
                            new_settle_measurement = (
                                capture_settle_last_measurement_time
                                != motion_estimate.measurement_time_sec
                            )
                            previous_settle_intensity_state = (
                                capture_settle_intensity_state
                            )
                            previous_settle_intensity_ratio = float(
                                current_intensity_ratio
                            )
                            (
                                capture_settle_intensity_state,
                                capture_intensity_ratio,
                                settle_force_command,
                            ) = settle_force_intensity(
                                vz_now,
                                capture_settle_intensity_state,
                                current_intensity_ratio,
                            )
                            if (
                                new_settle_measurement
                                and (
                                    capture_settle_intensity_state
                                    != previous_settle_intensity_state
                                    or abs(
                                        float(capture_intensity_ratio)
                                        - previous_settle_intensity_ratio
                                    )
                                    > 1e-9
                                )
                            ):
                                capture_logger.write(
                                    event="capture_settle_intensity_changed",
                                    release_timestamp_ms=(
                                        auto_release_timestamp_ms
                                    ),
                                    event_time_sec=now_capture,
                                    estimate=motion_estimate,
                                    target=target,
                                    intensity=capture_intensity_ratio,
                                    reason=(
                                        f"{previous_settle_intensity_state}->"
                                        f"{capture_settle_intensity_state};"
                                        f"intensity="
                                        f"{previous_settle_intensity_ratio:.2f}->"
                                        f"{float(capture_intensity_ratio):.2f};"
                                        f"vz={vz_now:.1f};"
                                        f"required_force_mN="
                                        f"{settle_force_command.required_force_mN:.3f};"
                                        f"force_saturated="
                                        f"{int(settle_force_command.saturated)}"
                                    ),
                                )
                            hold_telemetry.vz_mm_s = vz_now
                            sphere_target_distance_now = float(
                                np.hypot(
                                    motion_estimate.x_mm - target.x,
                                    motion_estimate.y_mm - target.y,
                                )
                            )
                            hold_telemetry.xy_distance_to_target_mm = (
                                sphere_target_distance_now
                            )
                            hold_telemetry.commanded_intensity = (
                                capture_intensity_ratio
                            )
                            hold_telemetry.required_force_mN = (
                                settle_force_command.required_force_mN
                            )
                            hold_telemetry.capture_force_saturated = (
                                settle_force_command.saturated
                            )

                            stable_vz_limit = float(
                                getattr(
                                    cfg,
                                    "auto_release_local_hold_vz_abs_mm_s",
                                    30.0,
                                )
                            )
                            stable_vxy_limit = float(
                                getattr(
                                    cfg,
                                    "auto_release_local_hold_vxy_max_mm_s",
                                    30.0,
                                )
                            )
                            stable_xy_distance_limit = float(
                                getattr(
                                    cfg,
                                    "auto_release_local_hold_xy_distance_max_mm",
                                    15.0,
                                )
                            )
                            stable_z_error_limit = float(
                                getattr(
                                    cfg,
                                    "auto_release_local_hold_z_error_max_mm",
                                    10.0,
                                )
                            )
                            stable_duration_sec = float(
                                getattr(
                                    cfg,
                                    "auto_release_local_hold_stable_sec",
                                    0.050,
                                )
                            )
                            capture_ready = False
                            if new_settle_measurement:
                                (
                                    capture_stable_since,
                                    capture_ready,
                                ) = update_capture_stability(
                                    now_sec=now_capture,
                                    vz_mm_s=vz_now,
                                    release_confirmed=auto_release_confirmed,
                                    stable_since_sec=capture_stable_since,
                                    maximum_abs_vz_mm_s=stable_vz_limit,
                                    required_duration_sec=stable_duration_sec,
                                    trajectory_complete=True,
                                    vxy_mm_s=vxy_now,
                                    maximum_vxy_mm_s=stable_vxy_limit,
                                    xy_distance_mm=xy_distance_now,
                                    maximum_xy_distance_mm=(
                                        stable_xy_distance_limit
                                    ),
                                    z_tracking_error_mm=z_tracking_error,
                                    maximum_z_tracking_error_mm=(
                                        stable_z_error_limit
                                    ),
                                    intensity_stable=(
                                        capture_settle_intensity_state
                                        == SETTLE_INTENSITY_NORMAL
                                    ),
                                )
                                capture_settle_last_measurement_time = (
                                    motion_estimate.measurement_time_sec
                                )
                            if capture_ready:
                                control_mode = LOCAL_HOLD
                                local_hold_start_time = time.time()
                                capture_stable_since = None
                                capture_settle_started_at = None
                                capture_settle_last_measurement_time = None
                                capture_settle_intensity_state = (
                                    SETTLE_INTENSITY_NORMAL
                                )
                                current_intensity_ratio = float(
                                    cfg.static_intensity_ratio
                                )
                                capture_intensity_ratio = (
                                    current_intensity_ratio
                                )
                                mode_transition_reason = (
                                    "auto_release_settle_stable"
                                )
                                # The normal controller was evaluated during
                                # capture but its XY output/integral were not
                                # used. Reset it before LOCAL_HOLD so no hidden
                                # integral windup is carried into normal PID.
                                controller.reset(target)
                                capture_logger.write(
                                    event="capture_settle_to_local_hold",
                                    release_timestamp_ms=(
                                        auto_release_timestamp_ms
                                    ),
                                    event_time_sec=now_capture,
                                    estimate=motion_estimate,
                                    target=target,
                                    intensity=current_intensity_ratio,
                                    reason=(
                                        "intensity_state=NORMAL;"
                                        f"abs_vz<={stable_vz_limit:.1f};"
                                        f"vxy<={stable_vxy_limit:.1f};"
                                        f"xy_distance<="
                                        f"{stable_xy_distance_limit:.1f};"
                                        f"z_error<={stable_z_error_limit:.1f};"
                                        f"stable_for="
                                        f"{stable_duration_sec:.3f}s"
                                    ),
                                )
                                paper_logger.mark_local_hold_reached(
                                    now_capture
                                )
                                print(
                                    "[AUTO_RELEASE] CAPTURE_SETTLE -> "
                                    "LOCAL_HOLD "
                                    f"setpoint=({return_setpoint.x:.1f}, "
                                    f"{return_setpoint.y:.1f}, "
                                    f"{return_setpoint.z:.1f}) mm, "
                                    f"vxy={vxy_now:.1f} mm/s, "
                                    f"vz={vz_now:.1f} mm/s"
                                )
                            elif (
                                release_elapsed_for_return_sec is not None
                                and release_elapsed_for_return_sec
                                >= minimum_return_from_release_sec
                                and should_start_settle_timeout_return_home(
                                    now_sec=now_capture,
                                    settle_started_sec=capture_settle_started_at,
                                    timeout_sec=float(
                                        getattr(
                                            cfg,
                                            "auto_release_settle_return_timeout_sec",
                                            10.0,
                                        )
                                    ),
                                    automatic_hold=auto_hold_active,
                                    release_confirmed=auto_release_confirmed,
                                    measurement_updated=new_settle_measurement,
                                    vz_mm_s=vz_now,
                                    maximum_abs_vz_mm_s=float(
                                        getattr(
                                            cfg,
                                            "auto_release_settle_return_max_abs_vz_mm_s",
                                            200.0,
                                        )
                                    ),
                                    vxy_mm_s=vxy_now,
                                    maximum_vxy_mm_s=float(
                                        getattr(
                                            cfg,
                                            "auto_release_settle_return_max_vxy_mm_s",
                                            200.0,
                                        )
                                    ),
                                    target_distance_mm=(
                                        sphere_target_distance_now
                                    ),
                                    maximum_target_distance_mm=float(
                                        getattr(
                                            cfg,
                                            "auto_release_settle_return_max_target_distance_mm",
                                            15.0,
                                        )
                                    ),
                                    z_mm=motion_estimate.z_mm,
                                    z_min_mm=cfg.z_min,
                                    z_max_mm=cfg.z_max,
                                    workspace_margin_mm=float(
                                        getattr(
                                            cfg,
                                            "auto_release_settle_return_z_workspace_margin_mm",
                                            10.0,
                                        )
                                    ),
                                )
                            ):
                                settle_elapsed_sec = max(
                                    0.0,
                                    now_capture
                                    - float(capture_settle_started_at),
                                )
                                # Begin the return at the current filtered
                                # sphere position so the timeout transition
                                # itself does not introduce a reference jump.
                                return_setpoint = HomePosition(
                                    x=float(motion_estimate.x_mm),
                                    y=float(motion_estimate.y_mm),
                                    z=float(
                                        np.clip(
                                            motion_estimate.z_mm,
                                            cfg.z_min,
                                            cfg.z_max,
                                        )
                                    ),
                                )
                                target = Target3D(
                                    return_setpoint.x,
                                    return_setpoint.y,
                                    return_setpoint.z,
                                )
                                control_mode = RETURN_TO_HOME
                                local_hold_start_time = None
                                capture_stable_since = None
                                capture_settle_started_at = None
                                capture_settle_last_measurement_time = None
                                capture_settle_intensity_state = (
                                    SETTLE_INTENSITY_NORMAL
                                )
                                mode_transition_reason = (
                                    "auto_release_settle_timeout_return_home"
                                )
                                controller.reset(target)
                                capture_logger.write(
                                    event=(
                                        "capture_settle_timeout_return_home"
                                    ),
                                    release_timestamp_ms=(
                                        auto_release_timestamp_ms
                                    ),
                                    event_time_sec=now_capture,
                                    estimate=motion_estimate,
                                    target=target,
                                    intensity=current_intensity_ratio,
                                    reason=(
                                        f"settle_elapsed_sec="
                                        f"{settle_elapsed_sec:.3f};"
                                        f"vxy_mm_s={vxy_now:.2f};"
                                        f"vz_mm_s={vz_now:.2f};"
                                        f"sphere_target_distance_mm="
                                        f"{sphere_target_distance_now:.2f};"
                                        f"z_mm={motion_estimate.z_mm:.2f};"
                                        f"return_start=("
                                        f"{return_setpoint.x:.2f},"
                                        f"{return_setpoint.y:.2f},"
                                        f"{return_setpoint.z:.2f});"
                                        f"home=({home.x:.2f},{home.y:.2f},"
                                        f"{home.z:.2f})"
                                    ),
                                )
                                paper_logger.finish_trial(
                                    status=(
                                        "success"
                                        if paper_logger.success_reached
                                        else "failure"
                                    ),
                                    reason=(
                                        "post_success_settle_timeout_return_home"
                                        if paper_logger.success_reached
                                        else "capture_settle_timeout_return_home"
                                    ),
                                    event_time_sec=now_capture,
                                )
                                print(
                                    "[AUTO_RELEASE] CAPTURE_SETTLE -> "
                                    "RETURN_TO_HOME after "
                                    f"{settle_elapsed_sec:.1f}s; "
                                    f"vxy={vxy_now:.1f} mm/s, "
                                    f"vz={vz_now:.1f} mm/s, "
                                    f"start=({return_setpoint.x:.1f}, "
                                    f"{return_setpoint.y:.1f}, "
                                    f"{return_setpoint.z:.1f}) mm"
                                )

                        # Final safety guard for both capture phases. The
                        # anchored PD helper already applies this vector limit;
                        # keep the guard here so future mode-specific changes
                        # cannot command outside the measured influence range.
                        if (
                            control_mode in [CAPTURE_ALIGN, CAPTURE_SETTLE]
                            and motion_estimate is not None
                        ):
                            xy_target_limit = clamp_target_xy_to_ball(
                                target_x_mm=target.x,
                                target_y_mm=target.y,
                                ball_x_mm=motion_estimate.x_mm,
                                ball_y_mm=motion_estimate.y_mm,
                                maximum_distance_mm=float(
                                    getattr(
                                        cfg,
                                        "auto_release_xy_capture_radius_mm",
                                        15.0,
                                    )
                                ),
                            )
                            target = Target3D(
                                x=xy_target_limit.x_mm,
                                y=xy_target_limit.y_mm,
                                z=target.z,
                            )
                            hold_telemetry.xy_distance_to_target_mm = (
                                xy_target_limit.distance_mm
                            )

                        # 5. LOCAL_HOLD中は一時基準でその場保持する。
                        #    自動releaseでは設定時間後にRETURN_TO_HOMEへ移る。

                        # 6. RETURN_TO_HOME完了判定
                        if control_mode == RETURN_TO_HOME:
                            if is_return_home_done(
                                cfg,
                                home,
                                z_mm,
                                x_mm,
                                y_mm,
                                vx_now,
                                vy_now,
                                vz_now,
                            ):
                                control_mode = NORMAL_HOLD
                                mode_transition_reason = "return_home_done"

                                return_setpoint = HomePosition(
                                    x=home.x,
                                    y=home.y,
                                    z=home.z,
                                )

                                controller.reset(
                                    Target3D(
                                        home.x,
                                        home.y,
                                        home.z,
                                    )
                                )

                                if auto_hold_active:
                                    capture_logger.write(
                                        event="return_home_completed",
                                        release_timestamp_ms=(
                                            auto_release_timestamp_ms
                                        ),
                                        estimate=motion_estimate,
                                        target=Target3D(
                                            home.x,
                                            home.y,
                                            home.z,
                                        ),
                                        intensity=current_intensity_ratio,
                                        reason="return_home_done",
                                    )

                                print("[RECOVERY] RETURN_TO_HOME -> NORMAL_HOLD")

                        hold_telemetry.temporary_home = return_setpoint
                        hold_telemetry.mode_transition_reason = mode_transition_reason

                        # 7. intensity更新。
                        #    通常保持時は既存LPFを維持し、捕捉関連状態だけ上昇即時/下降slewを使う。
                        if (
                            control_mode in [CAPTURE_ALIGN, CAPTURE_SETTLE]
                            and capture_intensity_ratio is not None
                        ):
                            # Release capture/settling is velocity-scheduled.
                            # Apply both increases and decreases immediately so
                            # braking strength disappears as soon as vz settles.
                            current_intensity_ratio = float(
                                np.clip(
                                    capture_intensity_ratio,
                                    0.0,
                                    min(
                                        cfg.intensity_max_ratio,
                                        float(
                                            getattr(
                                                cfg,
                                                "auto_release_max_intensity_ratio",
                                                0.8,
                                            )
                                        ),
                                    ),
                                )
                            )
                            hold_telemetry.commanded_intensity = (
                                current_intensity_ratio
                            )
                        elif control_mode in [LOCAL_HOLD, RETURN_TO_HOME]:
                            current_intensity_ratio = apply_capture_intensity_slew(
                                current_intensity_ratio,
                                cfg.static_intensity_ratio,
                                cfg,
                                dt_loop,
                            )
                            hold_telemetry.commanded_intensity = current_intensity_ratio
                        else:
                            current_intensity_ratio = (
                                cfg.intensity_lpf_alpha * current_intensity_ratio
                                + (1.0 - cfg.intensity_lpf_alpha) * target_intensity_ratio
                            )

                            current_intensity_ratio = float(
                                np.clip(
                                    current_intensity_ratio,
                                    cfg.static_intensity_ratio,
                                    cfg.intensity_max_ratio,
                                )
                            )

                        hold_telemetry.actual_intensity = current_intensity_ratio
                        last_target = target

                        set_tracking_target(
                            sender,
                            cfg,
                            target.x,
                            target.y,
                            target.z,
                            home,
                            current_radius,
                            current_intensity_ratio,
                        )

                    if do_display:
                        target_relative = np.array(
                            [last_target.x, last_target.y, last_target.z],
                            dtype=float,
                        ) - np.array(
                            [
                                float(base_center[0]),
                                float(base_center[1]),
                                float(base_center[2]),
                            ],
                            dtype=float,
                        )
                        cv2.putText(
                            frame_xy_bgr,
                            f"Target XY: {target_relative[0]:.1f}, {target_relative[1]:.1f}",
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )
                        cv2.putText(
                            frame_z_bgr,
                            f"Target Z: {target_relative[2]:.1f}",
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )
                        if auto_hold_active:
                            capture_vz = (
                                None
                                if motion_estimate is None
                                else motion_estimate.vz_mm_s
                            )
                            capture_label = (
                                f"AUTO CONTROL: {control_mode} | "
                                f"vz={'--' if capture_vz is None else f'{capture_vz:.1f}'} "
                                f"| intensity={current_intensity_ratio:.2f}"
                            )
                            cv2.putText(
                                frame_xy_bgr,
                                capture_label,
                                (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2,
                            )
                            cv2.putText(
                                frame_z_bgr,
                                capture_label,
                                (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2,
                            )

                paper_sample_time = time.perf_counter()
                paper_reference = (
                    None
                    if capture_braking_plan is None
                    else braking_reference_at(
                        capture_braking_plan,
                        now_sec=paper_sample_time,
                    )
                )
                paper_logger.record_sample(
                    sample_time_sec=paper_sample_time,
                    frame_xy_time_sec=frame_xy_time,
                    frame_z_time_sec=frame_z_time,
                    frame_sync_skew_sec=frame_sync_skew,
                    detected_xy=detected_xy,
                    detected_z=detected_z,
                    raw_x_mm=x_mm,
                    raw_y_mm=y_mm,
                    raw_z_mm=z_mm,
                    estimate=motion_estimate,
                    control_mode=control_mode,
                    reference=paper_reference,
                    anchor=capture_xy_anchor,
                    setpoint=return_setpoint,
                    target=last_target,
                    commanded_intensity=(
                        current_intensity_ratio
                        if paper_logger.field_started
                        else 0.0
                    ),
                    xy_distance_to_target_mm=(
                        hold_telemetry.xy_distance_to_target_mm
                    ),
                    required_force_mN=(
                        hold_telemetry.required_force_mN
                    ),
                    force_saturated=(
                        hold_telemetry.capture_force_saturated
                    ),
                    plan=capture_braking_plan,
                    release_gate_origin_z_mm=release_gate_origin_z_mm,
                    release_gate_downward_displacement_mm=(
                        release_gate_downward_displacement_mm
                    ),
                    release_gate_consecutive_count=(
                        release_gate_consecutive_count
                    ),
                    release_gate_passed=release_gate_passed,
                    mode_transition_reason=mode_transition_reason,
                )

                logger.write(
                    frame_xy_time=frame_xy_time,
                    frame_z_time=frame_z_time,
                    u_xy=u_xy,
                    v_xy=v_xy,
                    v_z=v_z,
                    x_mm=x_mm,
                    y_mm=y_mm,
                    z_mm=z_mm,
                    home=home,
                    target=last_target,
                    control_mode=control_mode,
                    hold=hold_telemetry,
                )
                mode_transition_reason = ""
                
                if do_display:
                    exit_requested = render_tracking_window(
                        window_title,
                        frame_xy_bgr,
                        frame_z_bgr,
                        DisplayMetrics(
                            loop_fps=loop_display_fps,
                            pair_fps=new_xy_display_fps,
                            z_fps=new_z_display_fps,
                            sync_skew_sec=frame_sync_skew,
                            autd_fps=sender.display_fps,
                            method=method,
                            v_z_px=v_z,
                        ),
                        DisplayControlState(
                            tracking_active=tracking_active,
                            control_mode=control_mode,
                            demo_active=demo_active,
                            home=home,
                            origin=display_origin,
                            return_setpoint=return_setpoint,
                            radius=current_radius,
                            intensity_ratio=current_intensity_ratio,
                        ),
                    )
                    if exit_requested:
                        break

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt.")

    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")

    finally:
        print("[INFO] stopping...")

        logger.close()
        trajectory_logger.close()
        if paper_logger is not None:
            paper_logger.close()

        running_event.clear()

        try:
            cam_thread_xy.join(timeout=1.0)
            cam_thread_z.join(timeout=1.0)
        except Exception:
            pass

        if sender is not None:
            try:
                sender.stop()
            except Exception:
                pass

        if auto_release_trigger is not None:
            try:
                auto_release_trigger.close()
            except Exception:
                pass

        safe_close_camera(cam_xy)
        safe_close_camera(cam_z)

        cv2.destroyAllWindows()

        print("[INFO] stopped.")
