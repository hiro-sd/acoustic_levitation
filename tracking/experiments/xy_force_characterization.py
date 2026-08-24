import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pyautd3 import Controller, Silencer
from pyautd3.link.twincat import TwinCAT

from tracking.core.autd_sender import AutdSender, make_autd_arrangement
from tracking.core.config import AppConfig
from tracking.core.controller import PredictionPIDController
from tracking.core.models import HomePosition, Measurement3D, Target3D
from tracking.core.runtime.cameras import start_camera_runtime
from tracking.core.runtime.input import is_key_pressed
from tracking.core.runtime.stereo_pipeline import StereoBallPipeline
from tracking.core.weighted_stm import make_weighted_dwell_pattern
from tracking.core.xy_force_measurement import (
    PulseSample,
    WindowedMotionEstimator3D,
    XYForceLogger,
    pulse_safety_reason,
)


DIRECTIONS = {
    "0": ("UNIFORM", 0.0),
    "1": ("BIAS_+X", 0.0),
    "2": ("BIAS_-X", math.pi),
    "3": ("BIAS_+Y", math.pi / 2.0),
    "4": ("BIAS_-Y", -math.pi / 2.0),
}


def _put(frame, text, y, color=(255, 255, 255), scale=0.58):
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
    )


def _fmt(value, digits=3):
    return "" if value is None else f"{float(value):.{digits}f}"


def run_xy_force_characterization(cfg: AppConfig):
    """Measure lateral acceleration from a short weighted-dwell STM pulse."""

    if cfg.autd_field_mode != "stm_circle":
        raise ValueError("XY force characterization requires stm_circle")

    camera_runtime = start_camera_runtime(cfg)
    if camera_runtime is None:
        return
    pipeline = StereoBallPipeline(
        width_xy=camera_runtime.width_xy,
        height_xy=camera_runtime.height_xy,
        width_z=camera_runtime.width_z,
        height_z=camera_runtime.height_z,
        cfg=cfg,
        triangulator=camera_runtime.stereo_triangulator,
        camera_to_autd=camera_runtime.camera_to_autd,
    )
    estimator = WindowedMotionEstimator3D(
        window_sec=float(cfg.xy_force_motion_window_sec),
        minimum_samples=int(cfg.xy_force_motion_min_samples),
        minimum_span_sec=float(cfg.xy_force_motion_min_span_sec),
    )
    logger = XYForceLogger(
        cfg.xy_force_frame_log_path,
        cfg.xy_force_summary_log_path,
        cfg.ball_mass_kg,
    )
    logger.open()

    sender = None
    try:
        with Controller.open(make_autd_arrangement(), TwinCAT()) as autd:
            autd.send(Silencer())
            center = autd.center()
            home = HomePosition(
                x=float(center[0]),
                y=float(center[1]),
                z=float(cfg.default_z),
            )
            last_target = Target3D(home.x, home.y, home.z)
            controller = PredictionPIDController(cfg)
            controller.reset(last_target)
            sender = AutdSender(autd, cfg)
            initial_sequence = sender.set_target(
                home.x,
                home.y,
                home.z,
                radius=cfg.radius,
                intensity_ratio=cfg.static_intensity_ratio,
            )
            sender.start()

            tracking_active = False
            phase = "BASELINE"
            trial_id = 0
            active_side = "UNIFORM"
            active_angle = 0.0
            active_level = 0.0
            active_counts = None
            pulse_target_xy = None
            pulse_initial_xyz = None
            pulse_command_sequence = None
            pulse_stop_command_sequence = None
            pulse_stop_reason = ""
            pulse_field_started_at = None
            pulse_field_stopped_at = None
            pulse_samples = []
            pulse_max_xy_displacement = 0.0
            pulse_max_z_drop = 0.0
            recovery_until = None
            arming_started_at = None
            last_command_sequence = initial_sequence
            current_bias_index = 0
            bias_levels = tuple(float(v) for v in cfg.xy_force_bias_levels)
            previous_keys = {key: False for key in ["enter", "b", *DIRECTIONS]}
            window_title = "XY Force Characterization"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            last_pair_time = time.perf_counter()
            frame_count = 0
            fps_count = 0
            fps_started_at = time.perf_counter()
            display_fps = 0.0
            last_estimate = None
            last_event = "startup"
            last_reason = ""

            print("=================================================")
            print(" XY FORCE CHARACTERIZATION")
            print(" ENTER: start/pause normal PID")
            print(" B: cycle dwell bias level")
            print(" 0: uniform pulse, 1:+X, 2:-X, 3:+Y, 4:-Y side bias")
            print(" ESC: exit")
            print(" A direction names the strongly weighted ring side;")
            print(" the measured acceleration determines the actual force sign.")
            print("=================================================")

            while True:
                if is_key_pressed("esc"):
                    break

                now = time.perf_counter()
                age_xy, age_z = camera_runtime.frame_buffer.get_frame_ages(now)
                if (
                    age_xy > cfg.camera_frame_timeout_sec
                    or age_z > cfg.camera_frame_timeout_sec
                ):
                    raise RuntimeError(
                        "Camera watchdog timeout: "
                        f"XY={age_xy:.3f}s, Z={age_z:.3f}s"
                    )
                pair = camera_runtime.frame_buffer.get_synced_pair(
                    cfg.camera_sync_tolerance_sec
                )
                if pair is None:
                    if now - last_pair_time > cfg.camera_frame_timeout_sec:
                        raise RuntimeError("Camera synchronization timeout")
                    time.sleep(0.001)
                    continue

                frame_xy, time_xy, frame_z, time_z, sync_skew = pair
                last_pair_time = now
                frame_count += 1
                fps_count += 1
                draw = frame_count % cfg.display_every_n_frames == 0
                result, frame_xy_bgr, frame_z_bgr = pipeline.process(
                    frame_xy,
                    frame_z,
                    active=tracking_active,
                    draw=draw,
                )
                measurement_time = max(float(time_xy), float(time_z))
                if result.valid_3d:
                    last_estimate = estimator.update(
                        measurement_time,
                        result.x_mm,
                        result.y_mm,
                        result.z_mm,
                    )

                event = "frame"
                reason = ""
                enter_pressed = is_key_pressed("enter")
                if enter_pressed and not previous_keys["enter"]:
                    tracking_active = not tracking_active
                    if tracking_active:
                        controller.reset(last_target)
                        estimator.reset()
                        print("[XY_FORCE] normal PID started")
                    else:
                        if phase in {"BIAS_ARMING", "BIAS_PULSE"}:
                            reason = "tracking_paused"
                        phase = "BASELINE"
                        active_counts = None
                        pulse_field_started_at = None
                        pulse_target_xy = None
                        pulse_initial_xyz = None
                        pulse_samples = []
                        print("[XY_FORCE] normal PID paused")
                previous_keys["enter"] = enter_pressed

                bias_pressed = is_key_pressed("b")
                if bias_pressed and not previous_keys["b"] and phase == "BASELINE":
                    current_bias_index = (current_bias_index + 1) % len(bias_levels)
                    print(
                        "[XY_FORCE] bias level="
                        f"{bias_levels[current_bias_index]:.2f}"
                    )
                previous_keys["b"] = bias_pressed

                current_xyz = (
                    None
                    if not result.valid_3d
                    else (result.x_mm, result.y_mm, result.z_mm)
                )
                if tracking_active and result.valid_3d:
                    normal_measurement = Measurement3D(
                        detected_xy=True,
                        detected_z=True,
                        x=result.x_mm,
                        y=result.y_mm,
                        z=result.z_mm,
                        t_xy=time_xy,
                        t_z=time_z,
                    )
                    if phase in {"BIAS_ARMING", "BIAS_PULSE", "BIAS_STOPPING"}:
                        z_only = Measurement3D(
                            detected_xy=False,
                            detected_z=True,
                            z=result.z_mm,
                            t_z=time_z,
                        )
                        z_target, _ = controller.update(z_only, home)
                        last_target = Target3D(
                            float(pulse_target_xy[0]),
                            float(pulse_target_xy[1]),
                            float(z_target.z),
                        )
                    else:
                        last_target, _ = controller.update(
                            normal_measurement,
                            home,
                        )

                for key, (side, angle) in DIRECTIONS.items():
                    pressed = is_key_pressed(key)
                    if pressed and not previous_keys[key]:
                        if not tracking_active or phase != "BASELINE":
                            print("[XY_FORCE] pulse ignored: PID or phase not ready")
                        elif not result.valid_3d or last_estimate is None:
                            print("[XY_FORCE] pulse ignored: no stereo 3D")
                        elif not last_estimate.velocity_ready:
                            print("[XY_FORCE] pulse ignored: velocity not ready")
                        else:
                            vxy = float(
                                np.hypot(
                                    last_estimate.vx_mm_s,
                                    last_estimate.vy_mm_s,
                                )
                            )
                            xy_error = float(
                                np.hypot(
                                    result.x_mm - home.x,
                                    result.y_mm - home.y,
                                )
                            )
                            z_error = abs(float(result.z_mm - home.z))
                            stable = (
                                vxy <= cfg.xy_force_start_max_vxy_mm_s
                                and abs(last_estimate.vz_mm_s)
                                <= cfg.xy_force_start_max_abs_vz_mm_s
                                and xy_error <= cfg.xy_force_start_max_xy_error_mm
                                and z_error <= cfg.xy_force_start_max_z_error_mm
                            )
                            if not stable:
                                print(
                                    "[XY_FORCE] pulse ignored: sphere not stable "
                                    f"vxy={vxy:.1f}, vz={last_estimate.vz_mm_s:.1f}, "
                                    f"xy_error={xy_error:.1f}, z_error={z_error:.1f}"
                                )
                            else:
                                trial_id += 1
                                active_side = side
                                active_angle = float(angle)
                                active_level = (
                                    0.0
                                    if key == "0"
                                    else bias_levels[current_bias_index]
                                )
                                pattern = make_weighted_dwell_pattern(
                                    point_num=cfg.point_num,
                                    total_slots=cfg.xy_force_total_slots,
                                    bias_angle_rad=active_angle,
                                    bias_level=active_level,
                                )
                                active_counts = pattern.slot_counts
                                pulse_target_xy = (
                                    float(last_target.x),
                                    float(last_target.y),
                                )
                                pulse_initial_xyz = current_xyz
                                pulse_samples = []
                                pulse_max_xy_displacement = 0.0
                                pulse_max_z_drop = 0.0
                                pulse_field_started_at = None
                                pulse_field_stopped_at = None
                                pulse_command_sequence = None
                                pulse_stop_command_sequence = None
                                pulse_stop_reason = ""
                                arming_started_at = now
                                phase = "BIAS_ARMING"
                                event = "pulse_armed"
                                last_event = event
                                print(
                                    f"[XY_FORCE] trial={trial_id} {side} "
                                    f"bias={active_level:.2f} counts={active_counts}"
                                )
                    previous_keys[key] = pressed

                if phase == "BIAS_ARMING":
                    sent = (
                        None
                        if pulse_command_sequence is None
                        else sender.get_first_sent_at_or_after(
                            pulse_command_sequence
                        )
                    )
                    if sent is not None and sent.target.focus_dwell_counts is not None:
                        pulse_field_started_at = float(sent.sent_time_sec)
                        if current_xyz is not None:
                            pulse_initial_xyz = current_xyz
                        pulse_samples = []
                        phase = "BIAS_PULSE"
                        event = "bias_field_started"
                        last_event = event
                    elif now - arming_started_at > cfg.xy_force_arming_timeout_sec:
                        phase = "RECOVERY"
                        recovery_until = now + cfg.xy_force_recovery_sec
                        active_counts = None
                        reason = "weighted_command_timeout"
                        event = "pulse_aborted"

                if (
                    phase in {"BIAS_PULSE", "BIAS_STOPPING"}
                    and current_xyz is not None
                    and pulse_initial_xyz is not None
                ):
                    pulse_max_xy_displacement = max(
                        pulse_max_xy_displacement,
                        float(
                            np.hypot(
                                current_xyz[0] - pulse_initial_xyz[0],
                                current_xyz[1] - pulse_initial_xyz[1],
                            )
                        ),
                    )
                    pulse_max_z_drop = max(
                        pulse_max_z_drop,
                        float(pulse_initial_xyz[2] - current_xyz[2]),
                    )

                if phase == "BIAS_PULSE":
                    if (
                        result.valid_3d
                        and measurement_time >= pulse_field_started_at
                    ):
                        pulse_samples.append(
                            PulseSample(
                                measurement_time_sec=measurement_time,
                                x_mm=result.x_mm,
                                y_mm=result.y_mm,
                                z_mm=result.z_mm,
                            )
                        )
                    safety_reason = pulse_safety_reason(
                        measurement_valid=result.valid_3d,
                        initial_xyz=pulse_initial_xyz,
                        current_xyz=current_xyz,
                        vz_mm_s=(
                            None if last_estimate is None else last_estimate.vz_mm_s
                        ),
                        maximum_xy_displacement_mm=cfg.xy_force_max_xy_displacement_mm,
                        maximum_z_drop_mm=cfg.xy_force_max_z_drop_mm,
                        maximum_abs_vz_mm_s=cfg.xy_force_max_abs_vz_mm_s,
                    )
                    elapsed = now - pulse_field_started_at
                    if safety_reason is not None or elapsed >= cfg.xy_force_pulse_sec:
                        pulse_stop_reason = (
                            safety_reason or "pulse_duration_complete"
                        )
                        reason = pulse_stop_reason
                        event = "bias_stop_enqueued"
                        phase = "BIAS_STOPPING"
                        active_counts = None
                        pulse_stop_command_sequence = None

                if phase == "BIAS_STOPPING":
                    if (
                        result.valid_3d
                        and measurement_time >= pulse_field_started_at
                        and (
                            not pulse_samples
                            or measurement_time
                            > pulse_samples[-1].measurement_time_sec
                        )
                    ):
                        pulse_samples.append(
                            PulseSample(
                                measurement_time_sec=measurement_time,
                                x_mm=result.x_mm,
                                y_mm=result.y_mm,
                                z_mm=result.z_mm,
                            )
                        )
                    stopped = (
                        None
                        if pulse_stop_command_sequence is None
                        else sender.get_first_sent_at_or_after(
                            pulse_stop_command_sequence
                        )
                    )
                    if (
                        stopped is not None
                        and stopped.target.focus_dwell_counts is None
                    ):
                        stop_time = float(stopped.sent_time_sec)
                        pulse_field_stopped_at = stop_time
                        pulse_samples = [
                            sample
                            for sample in pulse_samples
                            if sample.measurement_time_sec <= stop_time
                        ]
                        phase = "RECOVERY"
                        recovery_until = now + cfg.xy_force_recovery_sec
                        event = "bias_field_stopped"
                        reason = pulse_stop_reason
                        estimate = logger.write_summary(
                            trial_id=trial_id,
                            bias_side=active_side,
                            bias_angle_deg=np.degrees(active_angle),
                            bias_level=active_level,
                            slot_counts=pattern.slot_counts,
                            samples=pulse_samples,
                            field_duration_sec=(
                                stop_time - pulse_field_started_at
                            ),
                            maximum_xy_displacement_mm=pulse_max_xy_displacement,
                            maximum_z_drop_mm=pulse_max_z_drop,
                            stop_reason=pulse_stop_reason,
                        )
                        if estimate is None:
                            print(
                                f"[XY_FORCE] trial={trial_id} ended: "
                                f"{pulse_stop_reason}; "
                                "not enough samples for acceleration"
                            )
                        else:
                            force_x = cfg.ball_mass_kg * estimate.ax_mm_s2
                            force_y = cfg.ball_mass_kg * estimate.ay_mm_s2
                            print(
                                f"[XY_FORCE] trial={trial_id} ended: "
                                f"{pulse_stop_reason}; "
                                f"a=({estimate.ax_mm_s2:.1f}, "
                                f"{estimate.ay_mm_s2:.1f}) mm/s^2, "
                                f"F=({force_x:.3f}, {force_y:.3f}) mN"
                            )
                        controller.reset(last_target)
                        last_event = event
                        last_reason = pulse_stop_reason

                if phase == "RECOVERY" and now >= recovery_until:
                    phase = "BASELINE"
                    active_side = "UNIFORM"
                    active_level = 0.0
                    pulse_target_xy = None
                    pulse_initial_xyz = None
                    pulse_field_started_at = None
                    pulse_field_stopped_at = None
                    pulse_samples = []
                    event = "recovery_complete"
                    last_event = event

                last_command_sequence = sender.set_target(
                    last_target.x,
                    last_target.y,
                    last_target.z,
                    radius=cfg.radius,
                    intensity_ratio=cfg.static_intensity_ratio,
                    focus_dwell_counts=active_counts,
                )
                if phase == "BIAS_ARMING" and pulse_command_sequence is None:
                    pulse_command_sequence = last_command_sequence
                if (
                    phase == "BIAS_STOPPING"
                    and pulse_stop_command_sequence is None
                ):
                    pulse_stop_command_sequence = last_command_sequence

                dx = dy = z_drop = xy_displacement = None
                if current_xyz is not None and pulse_initial_xyz is not None:
                    dx = current_xyz[0] - pulse_initial_xyz[0]
                    dy = current_xyz[1] - pulse_initial_xyz[1]
                    xy_displacement = float(np.hypot(dx, dy))
                    z_drop = pulse_initial_xyz[2] - current_xyz[2]
                pulse_elapsed_ms = (
                    None
                    if pulse_field_started_at is None
                    else max(
                        0.0,
                        (
                            now
                            if pulse_field_stopped_at is None
                            else pulse_field_stopped_at
                        )
                        - pulse_field_started_at,
                    )
                    * 1000.0
                )
                field_counts = (
                    pattern.slot_counts
                    if phase == "BIAS_STOPPING"
                    else active_counts
                )
                counts_for_log = (
                    ""
                    if field_counts is None
                    else ";".join(str(v) for v in field_counts)
                )
                logger.write_frame(
                    [
                        f"{time.time():.6f}",
                        event,
                        trial_id,
                        phase,
                        active_side,
                        f"{np.degrees(active_angle):.3f}",
                        f"{active_level:.3f}",
                        counts_for_log,
                        f"{measurement_time:.6f}",
                        _fmt(result.x_mm),
                        _fmt(result.y_mm),
                        _fmt(result.z_mm),
                        _fmt(None if last_estimate is None else last_estimate.vx_mm_s),
                        _fmt(None if last_estimate is None else last_estimate.vy_mm_s),
                        _fmt(None if last_estimate is None else last_estimate.vz_mm_s),
                        _fmt(last_target.x),
                        _fmt(last_target.y),
                        _fmt(last_target.z),
                        _fmt(xy_displacement),
                        _fmt(z_drop),
                        _fmt(pulse_elapsed_ms),
                        last_command_sequence,
                        reason,
                    ]
                )

                elapsed_fps = now - fps_started_at
                if elapsed_fps >= 1.0:
                    display_fps = fps_count / elapsed_fps
                    fps_count = 0
                    fps_started_at = now
                if draw:
                    status_color = (0, 255, 0) if tracking_active else (0, 165, 255)
                    _put(
                        frame_xy_bgr,
                        f"Phase: {phase} | Trial: {trial_id}",
                        30,
                        status_color,
                    )
                    _put(
                        frame_xy_bgr,
                        f"Side: {active_side} | bias={active_level:.2f}",
                        60,
                        (0, 255, 255),
                    )
                    _put(
                        frame_xy_bgr,
                        f"counts={counts_for_log or 'uniform 8-point'}",
                        90,
                        (0, 255, 255),
                        0.48,
                    )
                    velocity_text = (
                        "velocity unavailable"
                        if last_estimate is None
                        else (
                            f"v=({last_estimate.vx_mm_s:.1f}, "
                            f"{last_estimate.vy_mm_s:.1f}, "
                            f"{last_estimate.vz_mm_s:.1f}) mm/s"
                        )
                    )
                    _put(frame_xy_bgr, velocity_text, 120)
                    _put(
                        frame_xy_bgr,
                        f"FPS={display_fps:.1f} AUTD={sender.display_fps:.1f} "
                        f"sync={sync_skew * 1000.0:.2f}ms",
                        150,
                    )
                    _put(
                        frame_xy_bgr,
                        f"Last: {last_event} {last_reason}",
                        180,
                        (255, 255, 0),
                        0.48,
                    )
                    _put(frame_z_bgr, f"Phase: {phase}", 30, status_color)
                    _put(
                        frame_z_bgr,
                        f"Z={_fmt(result.z_mm, 1)} target={last_target.z:.1f}",
                        60,
                        (0, 255, 255),
                    )
                    _put(
                        frame_z_bgr,
                        "ENTER PID | B bias | 0 uniform | 1:+X 2:-X 3:+Y 4:-Y",
                        frame_z_bgr.shape[0] - 20,
                        (255, 255, 255),
                        0.42,
                    )
                    height = max(frame_xy_bgr.shape[0], frame_z_bgr.shape[0])
                    width = max(frame_xy_bgr.shape[1], frame_z_bgr.shape[1])
                    xy_display = cv2.resize(frame_xy_bgr, (width, height))
                    z_display = cv2.resize(frame_z_bgr, (width, height))
                    cv2.imshow(window_title, np.hstack([xy_display, z_display]))
                    cv2.waitKey(1)

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt")
    except Exception as error:
        print(f"[ERROR] XY force experiment failed: {error}")
    finally:
        logger.close()
        if sender is not None:
            try:
                sender.stop()
            except Exception:
                pass
        camera_runtime.close()
        cv2.destroyAllWindows()
        print("[INFO] XY force experiment stopped")


if __name__ == "__main__":
    cfg = AppConfig(
        radius=19.0,
        point_num=8,
        default_z=400.0,
        static_intensity_ratio=0.6,
        stm_freq_hz=100.0,
        autd_field_mode="stm_circle",
        enable_stereo_triangulation=True,
        use_stereo_position_for_control=True,
        stereo_camera_to_autd_npz=(
            "./tracking/calibration/stereo_camera_to_autd.npz"
        ),
        enable_base_move=False,
        enable_radius_change=False,
        enable_fall_recovery=False,
        enable_auto_demo=False,
        use_gravity_prediction_z=False,
        display_every_n_frames=2,
    )
    cfg.xy_force_total_slots = 64
    cfg.xy_force_bias_levels = (0.10, 0.20, 0.30)
    cfg.xy_force_pulse_sec = 0.080
    cfg.xy_force_recovery_sec = 2.0
    cfg.xy_force_arming_timeout_sec = 0.100
    cfg.xy_force_motion_window_sec = 0.060
    cfg.xy_force_motion_min_samples = 3
    cfg.xy_force_motion_min_span_sec = 0.010
    cfg.xy_force_start_max_vxy_mm_s = 60.0
    cfg.xy_force_start_max_abs_vz_mm_s = 60.0
    cfg.xy_force_start_max_xy_error_mm = 15.0
    cfg.xy_force_start_max_z_error_mm = 10.0
    cfg.xy_force_max_xy_displacement_mm = 10.0
    cfg.xy_force_max_z_drop_mm = 10.0
    cfg.xy_force_max_abs_vz_mm_s = 150.0
    cfg.xy_force_frame_log_path = "./tracking/xy_force_frame_log.csv"
    cfg.xy_force_summary_log_path = "./tracking/xy_force_trial_summary.csv"
    run_xy_force_characterization(cfg)
