"""Measure the force of the circular STM from a short optical step response.

The ordinary stereo PID first stabilizes the sphere.  During each trial the PID
is frozen, a circular STM with a prescribed center/intensity is presented for a
short interval, and the observed 3-D trajectory is written to disk.  Force is
estimated from a timestamped quadratic trajectory fit rather than frame-to-
frame second differences.
"""

from __future__ import annotations

import argparse
import math
import random
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from autd3.commands import SetSilencer

from tracking.core.acoustic_force_measurement import (
    AcousticForceExperimentLogger,
    ForceStepSample,
    acoustic_force_mN,
    buoyancy_force_mN,
    estimate_position_resolution,
    estimate_step_acceleration,
    make_session_id,
    step_safety_reason,
)
from tracking.core.autd_client import AutdController
from tracking.core.autd_sender import AutdSender, make_autd_arrangement
from tracking.core.config import AppConfig
from tracking.core.controller import PredictionPIDController
from tracking.core.models import HomePosition, Measurement3D, Target3D
from tracking.core.runtime.cameras import start_camera_runtime
from tracking.core.runtime.input import is_key_pressed
from tracking.core.runtime.stereo_pipeline import StereoBallPipeline
from tracking.core.xy_force_measurement import WindowedMotionEstimator3D


@dataclass(frozen=True)
class StepCondition:
    intensity_ratio: float
    offset_x_mm: float
    offset_y_mm: float
    offset_z_mm: float


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one comma-separated value is required")
    if not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("all values must be finite")
    return values


def _format(value, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def _modulation_byte(intensity_ratio: float) -> int:
    return int(np.clip(float(intensity_ratio), 0.0, 1.0) * 255)


def _put(frame, text, y, color=(255, 255, 255), scale=0.52):
    cv2.putText(
        frame,
        str(text),
        (10, int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        2,
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _make_sample(result, measurement_time: float, sync_skew: float):
    if not result.valid_3d or not result.det_xy.detected or not result.det_z.detected:
        return None
    return ForceStepSample(
        measurement_time_sec=float(measurement_time),
        x_mm=float(result.x_mm),
        y_mm=float(result.y_mm),
        z_mm=float(result.z_mm),
        u_xy_px=float(result.det_xy.center[0]),
        v_xy_px=float(result.det_xy.center[1]),
        radius_xy_px=float(result.det_xy.radius_px),
        u_z_px=float(result.det_z.center[0]),
        v_z_px=float(result.det_z.center[1]),
        radius_z_px=float(result.det_z.radius_px),
        sync_skew_sec=float(sync_skew),
    )


def _sample_log_row(
    *,
    session_id: str,
    trial_id: int,
    condition_index: int,
    phase: str,
    event: str,
    sample: ForceStepSample,
    field_started_time_sec: float | None,
    estimate,
    field_center: tuple[float, float, float] | None,
    intensity: float,
    command_sequence: int | None,
    reason: str,
):
    elapsed_ms = (
        None
        if field_started_time_sec is None
        else (sample.measurement_time_sec - field_started_time_sec) * 1000.0
    )
    return [
        session_id,
        trial_id,
        condition_index,
        phase,
        event,
        f"{sample.measurement_time_sec:.9f}",
        _format(elapsed_ms),
        _format(sample.sync_skew_sec * 1000.0),
        _format(sample.u_xy_px),
        _format(sample.v_xy_px),
        _format(sample.radius_xy_px),
        _format(sample.u_z_px),
        _format(sample.v_z_px),
        _format(sample.radius_z_px),
        _format(sample.x_mm, 6),
        _format(sample.y_mm, 6),
        _format(sample.z_mm, 6),
        _format(None if estimate is None else estimate.vx_mm_s),
        _format(None if estimate is None else estimate.vy_mm_s),
        _format(None if estimate is None else estimate.vz_mm_s),
        _format(None if field_center is None else field_center[0]),
        _format(None if field_center is None else field_center[1]),
        _format(None if field_center is None else field_center[2]),
        _format(intensity, 4),
        "" if command_sequence is None else int(command_sequence),
        reason,
    ]


def _make_schedule(args) -> list[StepCondition]:
    conditions = [
        StepCondition(intensity, dx, dy, dz)
        for _ in range(int(args.repeats))
        for intensity in args.intensities
        for dx in args.x_offsets_mm
        for dy in args.y_offsets_mm
        for dz in args.z_offsets_mm
    ]
    if not args.no_randomize:
        random.Random(args.seed).shuffle(conditions)
    return conditions


def _summary_row(
    *,
    session_id: str,
    trial_id: int,
    condition_index: int,
    condition: StepCondition,
    stop_reason: str,
    field_started_time_sec: float,
    field_stopped_time_sec: float,
    analysis_delay_sec: float,
    estimate,
    resolution,
    mass_kg: float,
    gravity_mm_s2: float,
    buoyancy_mn: float,
    initial_xyz: tuple[float, float, float],
    onset_xyz: tuple[float, float, float],
    field_center: tuple[float, float, float],
    max_xy_displacement_mm: float,
    max_abs_z_displacement_mm: float,
    sent,
):
    valid = estimate is not None and stop_reason == "pulse_duration_complete"
    modulation_byte = _modulation_byte(condition.intensity_ratio)
    row = {
        "session_id": session_id,
        "trial_id": trial_id,
        "condition_index": condition_index,
        "intensity_ratio": f"{condition.intensity_ratio:.4f}",
        "modulation_byte": modulation_byte,
        "quantized_intensity_ratio": f"{modulation_byte / 255.0:.8f}",
        "offset_x_mm": f"{condition.offset_x_mm:.3f}",
        "offset_y_mm": f"{condition.offset_y_mm:.3f}",
        "offset_z_mm": f"{condition.offset_z_mm:.3f}",
        "status": "valid" if valid else "invalid",
        "stop_reason": stop_reason,
        "pulse_duration_ms": f"{(field_stopped_time_sec - field_started_time_sec) * 1000.0:.3f}",
        "analysis_delay_ms": f"{analysis_delay_sec * 1000.0:.3f}",
        "buoyancy_mN": f"{buoyancy_mn:.6f}",
        "initial_x_mm": f"{initial_xyz[0]:.6f}",
        "initial_y_mm": f"{initial_xyz[1]:.6f}",
        "initial_z_mm": f"{initial_xyz[2]:.6f}",
        "onset_x_mm": f"{onset_xyz[0]:.6f}",
        "onset_y_mm": f"{onset_xyz[1]:.6f}",
        "onset_z_mm": f"{onset_xyz[2]:.6f}",
        "field_center_x_mm": f"{field_center[0]:.6f}",
        "field_center_y_mm": f"{field_center[1]:.6f}",
        "field_center_z_mm": f"{field_center[2]:.6f}",
        # Positive means sphere position minus circular-field center.
        "actual_relative_x_mm": f"{onset_xyz[0] - field_center[0]:.6f}",
        "actual_relative_y_mm": f"{onset_xyz[1] - field_center[1]:.6f}",
        "actual_relative_z_mm": f"{onset_xyz[2] - field_center[2]:.6f}",
        "maximum_xy_displacement_mm": f"{max_xy_displacement_mm:.6f}",
        "maximum_abs_z_displacement_mm": f"{max_abs_z_displacement_mm:.6f}",
        "command_enqueued_time_sec": f"{sent.target.enqueued_time_sec:.9f}",
        "send_started_time_sec": f"{sent.send_started_time_sec:.9f}",
        "field_started_time_sec": f"{field_started_time_sec:.9f}",
        "field_stopped_time_sec": f"{field_stopped_time_sec:.9f}",
    }
    if estimate is not None:
        row.update(
            {
                "fit_sample_count": estimate.sample_count,
                "fit_span_ms": f"{estimate.span_sec * 1000.0:.3f}",
                "ax_mm_s2": f"{estimate.x.acceleration_mm_s2:.6f}",
                "ay_mm_s2": f"{estimate.y.acceleration_mm_s2:.6f}",
                "az_mm_s2": f"{estimate.z.acceleration_mm_s2:.6f}",
                "ax_se_mm_s2": f"{estimate.x.acceleration_standard_error_mm_s2:.6f}",
                "ay_se_mm_s2": f"{estimate.y.acceleration_standard_error_mm_s2:.6f}",
                "az_se_mm_s2": f"{estimate.z.acceleration_standard_error_mm_s2:.6f}",
                "fit_rms_x_mm": f"{estimate.x.residual_rms_mm:.6f}",
                "fit_rms_y_mm": f"{estimate.y.residual_rms_mm:.6f}",
                "fit_rms_z_mm": f"{estimate.z.residual_rms_mm:.6f}",
                "fit_r2_x": f"{estimate.x.r_squared:.6f}",
                "fit_r2_y": f"{estimate.y.r_squared:.6f}",
                "fit_r2_z": f"{estimate.z.r_squared:.6f}",
                "fit_mean_x_mm": f"{estimate.mean_x_mm:.6f}",
                "fit_mean_y_mm": f"{estimate.mean_y_mm:.6f}",
                "fit_mean_z_mm": f"{estimate.mean_z_mm:.6f}",
                "fit_mean_relative_x_mm": f"{estimate.mean_x_mm - field_center[0]:.6f}",
                "fit_mean_relative_y_mm": f"{estimate.mean_y_mm - field_center[1]:.6f}",
                "fit_mean_relative_z_mm": f"{estimate.mean_z_mm - field_center[2]:.6f}",
                # kg*mm/s^2 is numerically mN.
                "force_x_mN": f"{mass_kg * estimate.x.acceleration_mm_s2:.6f}",
                "force_y_mN": f"{mass_kg * estimate.y.acceleration_mm_s2:.6f}",
                "acoustic_force_z_mN": f"{acoustic_force_mN(mass_kg=mass_kg, acceleration_mm_s2=estimate.z.acceleration_mm_s2, gravity_mm_s2=gravity_mm_s2, buoyancy_mN=buoyancy_mn):.6f}",
                "force_x_se_mN": f"{mass_kg * estimate.x.acceleration_standard_error_mm_s2:.6f}",
                "force_y_se_mN": f"{mass_kg * estimate.y.acceleration_standard_error_mm_s2:.6f}",
                "acoustic_force_z_se_mN": f"{mass_kg * estimate.z.acceleration_standard_error_mm_s2:.6f}",
            }
        )
    if resolution is not None:
        row.update(
            {
                "baseline_sample_count": resolution.sample_count,
                "baseline_span_ms": f"{resolution.span_sec * 1000.0:.3f}",
                "baseline_std_x_mm": f"{resolution.std_x_mm:.6f}",
                "baseline_std_y_mm": f"{resolution.std_y_mm:.6f}",
                "baseline_std_z_mm": f"{resolution.std_z_mm:.6f}",
                "baseline_robust_sigma_x_mm": f"{resolution.robust_sigma_x_mm:.6f}",
                "baseline_robust_sigma_y_mm": f"{resolution.robust_sigma_y_mm:.6f}",
                "baseline_robust_sigma_z_mm": f"{resolution.robust_sigma_z_mm:.6f}",
                "single_difference_95_x_mm": f"{resolution.single_difference_95_x_mm:.6f}",
                "single_difference_95_y_mm": f"{resolution.single_difference_95_y_mm:.6f}",
                "single_difference_95_z_mm": f"{resolution.single_difference_95_z_mm:.6f}",
                "baseline_std_u_xy_px": f"{resolution.std_u_xy_px:.6f}",
                "baseline_std_v_xy_px": f"{resolution.std_v_xy_px:.6f}",
                "baseline_std_u_z_px": f"{resolution.std_u_z_px:.6f}",
                "baseline_std_v_z_px": f"{resolution.std_v_z_px:.6f}",
                "median_radius_xy_px": f"{resolution.median_radius_xy_px:.3f}",
                "median_radius_z_px": f"{resolution.median_radius_z_px:.3f}",
                "approximate_scale_xy_mm_per_px": f"{resolution.approximate_scale_xy_mm_per_px:.6f}",
                "approximate_scale_z_mm_per_px": f"{resolution.approximate_scale_z_mm_per_px:.6f}",
            }
        )
    return row


def run_experiment(cfg: AppConfig, args) -> Path | None:
    schedule = _make_schedule(args)
    session_id = make_session_id()
    output_directory = Path(args.output_root) / f"session_{session_id}"
    logger = AcousticForceExperimentLogger(output_directory, session_id)

    camera_runtime = start_camera_runtime(cfg)
    if camera_runtime is None:
        return None
    print(
        "[FORCE] processed camera resolution: "
        f"XY={camera_runtime.width_xy}x{camera_runtime.height_xy}, "
        f"Z={camera_runtime.width_z}x{camera_runtime.height_z} "
        "(the Z image may be rotated)"
    )
    if (camera_runtime.width_xy, camera_runtime.height_xy) != (648, 488):
        print(
            "[WARN] XY camera resolution differs from the expected 648x488; "
            "the actual size is stored in session.json."
        )
    pipeline = StereoBallPipeline(
        width_xy=camera_runtime.width_xy,
        height_xy=camera_runtime.height_xy,
        width_z=camera_runtime.width_z,
        height_z=camera_runtime.height_z,
        cfg=cfg,
        triangulator=camera_runtime.stereo_triangulator,
        camera_to_autd=camera_runtime.camera_to_autd,
    )
    motion_estimator = WindowedMotionEstimator3D(
        window_sec=0.060,
        minimum_samples=5,
        minimum_span_sec=0.020,
    )
    sender = None
    try:
        with AutdController.open(
            make_autd_arrangement(),
            interface_name=cfg.autd_interface_name,
            open_timeout_sec=cfg.autd_open_timeout_sec,
            command_timeout_sec=cfg.autd_command_timeout_sec,
        ) as autd:
            autd.send(SetSilencer())
            center = autd.center()
            home = HomePosition(float(center[0]), float(center[1]), float(cfg.default_z))
            last_hold_target = Target3D(home.x, home.y, home.z)
            controller = PredictionPIDController(cfg)
            controller.reset(last_hold_target)
            sender = AutdSender(autd, cfg)
            sender.set_target(
                home.x,
                home.y,
                home.z,
                radius=cfg.radius,
                intensity_ratio=args.baseline_intensity,
            )
            sender.start()

            metadata = {
                "schema_version": 1,
                "session_id": session_id,
                "created_at": datetime.now().astimezone().isoformat(),
                "operator": args.operator,
                "git_commit": _git_commit(),
                "camera": {
                    "sensor_resolution_px": [648, 488],
                    "processed_xy_resolution_px": [
                        camera_runtime.width_xy,
                        camera_runtime.height_xy,
                    ],
                    "processed_z_resolution_px": [
                        camera_runtime.width_z,
                        camera_runtime.height_z,
                    ],
                    "note": (
                        "Pixel count alone does not determine millimetres per pixel. "
                        "The logs include detected sphere radius and stable-hold XYZ "
                        "spread so displacement resolution can be estimated in situ."
                    ),
                },
                "sphere": {
                    "mass_kg": args.sphere_mass_g * 1e-3,
                    "diameter_mm": args.sphere_diameter_mm,
                    "air_density_kg_m3": args.air_density_kg_m3,
                },
                "field": {
                    "radius_mm": cfg.radius,
                    "point_num": cfg.point_num,
                    "stm_cycle_frequency_hz": cfg.stm_freq_hz,
                    "baseline_intensity": args.baseline_intensity,
                    "intensity_note": (
                        "The requested modulation amplitude ratio is converted "
                        "to int(255 * ratio); per-trial quantized values are "
                        "stored in trial_summary.csv."
                    ),
                },
                "experiment": {
                    "pulse_ms": args.pulse_ms,
                    "analysis_delay_ms": args.analysis_delay_ms,
                    "recovery_sec": args.recovery_sec,
                    "stable_sec": args.stable_sec,
                    "baseline_window_sec": args.baseline_window_sec,
                    "minimum_fit_samples": args.minimum_fit_samples,
                    "minimum_fit_span_ms": args.minimum_fit_span_ms,
                    "random_seed": args.seed,
                    "randomized": not args.no_randomize,
                    "conditions": [condition.__dict__ for condition in schedule],
                },
                "calibration": {
                    "intrinsic_xy_npz": cfg.intrinsic_xy_npz,
                    "intrinsic_z_npz": cfg.intrinsic_z_npz,
                    "stereo_npz": cfg.stereo_npz,
                    "stereo_camera_to_autd_npz": cfg.stereo_camera_to_autd_npz,
                },
            }
            logger.open(metadata)

            phase = "HOLD"
            tracking_active = False
            sequence_running = False
            schedule_index = 0
            trial_id = 0
            current_condition = None
            condition_index = 0
            stable_since = None
            recovery_until = None
            baseline_buffer: deque[ForceStepSample] = deque()
            trial_baseline_samples: list[ForceStepSample] = []
            pulse_samples: list[ForceStepSample] = []
            pulse_sequence = None
            arming_started_at = None
            stop_sequence = None
            pulse_sent = None
            field_started_at = None
            field_stopped_at = None
            initial_xyz = None
            field_center = None
            stop_reason = ""
            max_xy_displacement = 0.0
            max_abs_z_displacement = 0.0
            previous_enter = False
            previous_s = False
            last_pair_time = time.perf_counter()
            last_event = "startup"
            last_motion = None
            display_fps = 0.0
            fps_count = 0
            fps_started = time.perf_counter()
            window_title = "Acoustic Force Step Response"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

            print("============================================================")
            print(" ACOUSTIC FORCE STEP RESPONSE")
            print(" ENTER: start/pause normal PID hold")
            print(" S: start/pause the randomized force-measurement schedule")
            print(" ESC: stop safely and exit")
            print(f" trials={len(schedule)} output={output_directory}")
            print(" offset means: field center minus sphere position at step start")
            print("============================================================")

            while True:
                if is_key_pressed("esc"):
                    break
                now = time.perf_counter()
                age_xy, age_z = camera_runtime.frame_buffer.get_frame_ages(now)
                if age_xy > cfg.camera_frame_timeout_sec or age_z > cfg.camera_frame_timeout_sec:
                    raise RuntimeError(
                        f"Camera watchdog timeout: XY={age_xy:.3f}s, Z={age_z:.3f}s"
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
                fps_count += 1
                draw = fps_count % max(1, cfg.display_every_n_frames) == 0
                result, frame_xy_bgr, frame_z_bgr = pipeline.process(
                    frame_xy,
                    frame_z,
                    active=tracking_active,
                    draw=draw,
                )
                measurement_time = max(float(time_xy), float(time_z))
                sample = _make_sample(result, measurement_time, sync_skew)
                if sample is not None:
                    last_motion = motion_estimator.update(
                        measurement_time,
                        sample.x_mm,
                        sample.y_mm,
                        sample.z_mm,
                    )

                enter_pressed = is_key_pressed("enter")
                if enter_pressed and not previous_enter:
                    tracking_active = not tracking_active
                    stable_since = None
                    if tracking_active:
                        controller.reset(last_hold_target)
                        print("[FORCE] PID hold started")
                    else:
                        sequence_running = False
                        print("[FORCE] PID hold paused")
                        if phase in {"ARMING", "PULSE"}:
                            stop_reason = "operator_paused"
                            stop_sequence = sender.set_target(
                                last_hold_target.x,
                                last_hold_target.y,
                                last_hold_target.z,
                                radius=cfg.radius,
                                intensity_ratio=args.baseline_intensity,
                            )
                            phase = "STOPPING"
                previous_enter = enter_pressed

                s_pressed = is_key_pressed("s")
                if s_pressed and not previous_s:
                    if not tracking_active:
                        print("[FORCE] Start PID hold with ENTER first")
                    elif schedule_index >= len(schedule):
                        print("[FORCE] Schedule already complete")
                    else:
                        sequence_running = not sequence_running
                        print(
                            "[FORCE] schedule "
                            + ("running" if sequence_running else "paused after current trial")
                        )
                previous_s = s_pressed

                # Ordinary PID is active only outside a frozen step/stop command.
                if tracking_active and sample is not None and phase in {"HOLD", "RECOVERY"}:
                    measurement = Measurement3D(
                        detected_xy=True,
                        detected_z=True,
                        x=sample.x_mm,
                        y=sample.y_mm,
                        z=sample.z_mm,
                        t_xy=float(time_xy),
                        t_z=float(time_z),
                    )
                    last_hold_target, _ = controller.update(measurement, home)
                    sender.set_target(
                        last_hold_target.x,
                        last_hold_target.y,
                        last_hold_target.z,
                        radius=cfg.radius,
                        intensity_ratio=args.baseline_intensity,
                    )

                if phase == "HOLD" and sample is not None:
                    baseline_buffer.append(sample)
                    cutoff = sample.measurement_time_sec - args.baseline_window_sec
                    while baseline_buffer and baseline_buffer[0].measurement_time_sec < cutoff:
                        baseline_buffer.popleft()

                stable = False
                if (
                    phase == "HOLD"
                    and tracking_active
                    and sample is not None
                    and last_motion is not None
                    and last_motion.velocity_ready
                ):
                    vxy = math.hypot(last_motion.vx_mm_s, last_motion.vy_mm_s)
                    xy_error = math.hypot(sample.x_mm - home.x, sample.y_mm - home.y)
                    z_error = abs(sample.z_mm - home.z)
                    stable = (
                        vxy <= args.stable_max_vxy_mm_s
                        and abs(last_motion.vz_mm_s) <= args.stable_max_abs_vz_mm_s
                        and xy_error <= args.stable_max_xy_error_mm
                        and z_error <= args.stable_max_z_error_mm
                    )
                    if stable:
                        if stable_since is None:
                            stable_since = now
                    else:
                        stable_since = None

                if (
                    phase == "HOLD"
                    and tracking_active
                    and sequence_running
                    and stable_since is not None
                    and now - stable_since >= args.stable_sec
                    and schedule_index < len(schedule)
                    and sample is not None
                ):
                    current_condition = schedule[schedule_index]
                    schedule_index += 1
                    condition_index = schedule_index
                    trial_id += 1
                    initial_xyz = (sample.x_mm, sample.y_mm, sample.z_mm)
                    field_center = (
                        sample.x_mm + current_condition.offset_x_mm,
                        sample.y_mm + current_condition.offset_y_mm,
                        sample.z_mm + current_condition.offset_z_mm,
                    )
                    trial_baseline_samples = list(baseline_buffer)
                    pulse_samples = []
                    max_xy_displacement = 0.0
                    max_abs_z_displacement = 0.0
                    stop_reason = ""
                    stop_sequence = None
                    pulse_sent = None
                    field_started_at = None
                    field_stopped_at = None
                    pulse_sequence = sender.set_target(
                        *field_center,
                        radius=cfg.radius,
                        intensity_ratio=current_condition.intensity_ratio,
                    )
                    arming_started_at = now
                    phase = "ARMING"
                    stable_since = None
                    last_event = "step_command_enqueued"
                    for baseline_sample in trial_baseline_samples:
                        logger.write_sample(
                            _sample_log_row(
                                session_id=session_id,
                                trial_id=trial_id,
                                condition_index=condition_index,
                                phase="BASELINE_WINDOW",
                                event="baseline",
                                sample=baseline_sample,
                                field_started_time_sec=None,
                                estimate=None,
                                field_center=field_center,
                                intensity=args.baseline_intensity,
                                command_sequence=pulse_sequence,
                                reason="",
                            )
                        )
                    print(
                        f"[FORCE] trial={trial_id}/{len(schedule)} "
                        f"A={current_condition.intensity_ratio:.2f} "
                        f"offset=({current_condition.offset_x_mm:.1f}, "
                        f"{current_condition.offset_y_mm:.1f}, "
                        f"{current_condition.offset_z_mm:.1f}) mm"
                    )

                if phase == "ARMING":
                    sent = sender.get_first_sent_at_or_after(pulse_sequence)
                    if sent is not None:
                        pulse_sent = sent
                        field_started_at = float(sent.sent_time_sec)
                        phase = "PULSE"
                        last_event = "step_field_started"
                    elif now - arming_started_at > args.arming_timeout_sec:
                        stop_reason = "step_command_timeout"
                        stop_sequence = sender.set_target(
                            last_hold_target.x,
                            last_hold_target.y,
                            last_hold_target.z,
                            radius=cfg.radius,
                            intensity_ratio=args.baseline_intensity,
                        )
                        phase = "STOPPING"

                if phase in {"PULSE", "STOPPING"} and sample is not None:
                    if field_started_at is not None and sample.measurement_time_sec >= field_started_at:
                        if (
                            not pulse_samples
                            or sample.measurement_time_sec
                            > pulse_samples[-1].measurement_time_sec
                        ):
                            pulse_samples.append(sample)
                        logger.write_sample(
                            _sample_log_row(
                                session_id=session_id,
                                trial_id=trial_id,
                                condition_index=condition_index,
                                phase=phase,
                                event=last_event,
                                sample=sample,
                                field_started_time_sec=field_started_at,
                                estimate=last_motion,
                                field_center=field_center,
                                intensity=current_condition.intensity_ratio,
                                command_sequence=pulse_sequence,
                                reason=stop_reason,
                            )
                        )
                        last_event = "frame"
                    if initial_xyz is not None:
                        dx = sample.x_mm - initial_xyz[0]
                        dy = sample.y_mm - initial_xyz[1]
                        dz = sample.z_mm - initial_xyz[2]
                        max_xy_displacement = max(max_xy_displacement, math.hypot(dx, dy))
                        max_abs_z_displacement = max(max_abs_z_displacement, abs(dz))

                if phase == "PULSE":
                    speed = (
                        None
                        if last_motion is None
                        else math.sqrt(
                            last_motion.vx_mm_s**2
                            + last_motion.vy_mm_s**2
                            + last_motion.vz_mm_s**2
                        )
                    )
                    safety_reason = step_safety_reason(
                        measurement_valid=sample is not None,
                        initial_xyz=initial_xyz,
                        current_xyz=(
                            None
                            if sample is None
                            else (sample.x_mm, sample.y_mm, sample.z_mm)
                        ),
                        speed_mm_s=speed,
                        maximum_xy_displacement_mm=args.maximum_xy_displacement_mm,
                        maximum_abs_z_displacement_mm=args.maximum_abs_z_displacement_mm,
                        maximum_speed_mm_s=args.maximum_speed_mm_s,
                        z_min_mm=cfg.z_min,
                        z_max_mm=cfg.z_max,
                    )
                    if (
                        safety_reason is not None
                        or now - field_started_at >= args.pulse_ms * 1e-3
                    ):
                        stop_reason = safety_reason or "pulse_duration_complete"
                        stop_sequence = sender.set_target(
                            last_hold_target.x,
                            last_hold_target.y,
                            last_hold_target.z,
                            radius=cfg.radius,
                            intensity_ratio=args.baseline_intensity,
                        )
                        phase = "STOPPING"
                        last_event = "recovery_command_enqueued"

                if phase == "STOPPING" and stop_sequence is not None:
                    stopped = sender.get_first_sent_at_or_after(stop_sequence)
                    if stopped is not None:
                        if field_started_at is None or pulse_sent is None:
                            logger.write_summary(
                                {
                                    "session_id": session_id,
                                    "trial_id": trial_id,
                                    "condition_index": condition_index,
                                    "intensity_ratio": f"{current_condition.intensity_ratio:.4f}",
                                    "modulation_byte": _modulation_byte(
                                        current_condition.intensity_ratio
                                    ),
                                    "quantized_intensity_ratio": f"{_modulation_byte(current_condition.intensity_ratio) / 255.0:.8f}",
                                    "offset_x_mm": f"{current_condition.offset_x_mm:.3f}",
                                    "offset_y_mm": f"{current_condition.offset_y_mm:.3f}",
                                    "offset_z_mm": f"{current_condition.offset_z_mm:.3f}",
                                    "status": "invalid",
                                    "stop_reason": stop_reason,
                                    "initial_x_mm": f"{initial_xyz[0]:.6f}",
                                    "initial_y_mm": f"{initial_xyz[1]:.6f}",
                                    "initial_z_mm": f"{initial_xyz[2]:.6f}",
                                    "field_center_x_mm": f"{field_center[0]:.6f}",
                                    "field_center_y_mm": f"{field_center[1]:.6f}",
                                    "field_center_z_mm": f"{field_center[2]:.6f}",
                                }
                            )
                            print(
                                f"[FORCE] trial={trial_id} invalid: {stop_reason}"
                            )
                            controller.reset(last_hold_target)
                            phase = "RECOVERY"
                            recovery_until = now + args.recovery_sec
                            baseline_buffer.clear()
                            last_event = "step_aborted"
                            continue
                        field_stopped_at = float(stopped.sent_time_sec)
                        pulse_samples = [
                            item
                            for item in pulse_samples
                            if item.measurement_time_sec <= field_stopped_at
                        ]
                        estimate = estimate_step_acceleration(
                            pulse_samples,
                            field_started_time_sec=field_started_at,
                            analysis_delay_sec=args.analysis_delay_ms * 1e-3,
                            field_stopped_time_sec=field_stopped_at,
                            minimum_samples=args.minimum_fit_samples,
                            minimum_span_sec=args.minimum_fit_span_ms * 1e-3,
                        )
                        if pulse_samples:
                            onset_sample = min(
                                pulse_samples,
                                key=lambda item: abs(
                                    item.measurement_time_sec - field_started_at
                                ),
                            )
                            onset_xyz = (
                                onset_sample.x_mm,
                                onset_sample.y_mm,
                                onset_sample.z_mm,
                            )
                        else:
                            onset_xyz = initial_xyz
                        resolution = estimate_position_resolution(
                            trial_baseline_samples,
                            sphere_diameter_mm=args.sphere_diameter_mm,
                        )
                        buoyancy_mn = buoyancy_force_mN(
                            diameter_mm=args.sphere_diameter_mm,
                            air_density_kg_m3=args.air_density_kg_m3,
                        )
                        logger.write_summary(
                            _summary_row(
                                session_id=session_id,
                                trial_id=trial_id,
                                condition_index=condition_index,
                                condition=current_condition,
                                stop_reason=stop_reason,
                                field_started_time_sec=field_started_at,
                                field_stopped_time_sec=field_stopped_at,
                                analysis_delay_sec=args.analysis_delay_ms * 1e-3,
                                estimate=estimate,
                                resolution=resolution,
                                mass_kg=args.sphere_mass_g * 1e-3,
                                gravity_mm_s2=cfg.gravity_mm_s2,
                                buoyancy_mn=buoyancy_mn,
                                initial_xyz=initial_xyz,
                                onset_xyz=onset_xyz,
                                field_center=field_center,
                                max_xy_displacement_mm=max_xy_displacement,
                                max_abs_z_displacement_mm=max_abs_z_displacement,
                                sent=pulse_sent,
                            )
                        )
                        if estimate is None:
                            print(
                                f"[FORCE] trial={trial_id} invalid: "
                                f"{stop_reason}; insufficient fit samples"
                            )
                        else:
                            force_z = acoustic_force_mN(
                                mass_kg=args.sphere_mass_g * 1e-3,
                                acceleration_mm_s2=estimate.z.acceleration_mm_s2,
                                gravity_mm_s2=cfg.gravity_mm_s2,
                                buoyancy_mN=buoyancy_mn,
                            )
                            force_se = (
                                args.sphere_mass_g
                                * 1e-3
                                * estimate.z.acceleration_standard_error_mm_s2
                            )
                            print(
                                f"[FORCE] trial={trial_id} {stop_reason}: "
                                f"az={estimate.z.acceleration_mm_s2:.1f} mm/s^2, "
                                f"Fz={force_z:.3f} +/- {1.96 * force_se:.3f} mN "
                                f"(fit 95%, n={estimate.sample_count})"
                            )
                        controller.reset(last_hold_target)
                        phase = "RECOVERY"
                        recovery_until = now + args.recovery_sec
                        baseline_buffer.clear()
                        last_event = "step_complete"

                if phase == "RECOVERY" and now >= recovery_until:
                    phase = "HOLD"
                    stable_since = None
                    last_event = "recovery_complete"
                    if schedule_index >= len(schedule):
                        sequence_running = False
                        print(f"[FORCE] all {len(schedule)} trials completed")

                fps_elapsed = now - fps_started
                if fps_elapsed >= 1.0:
                    display_fps = fps_count / fps_elapsed
                    fps_count = 0
                    fps_started = now
                if draw:
                    status_color = (0, 255, 0) if tracking_active else (0, 165, 255)
                    _put(frame_xy_bgr, f"Phase: {phase}  Trial: {trial_id}/{len(schedule)}", 28, status_color)
                    _put(frame_xy_bgr, f"PID={'ON' if tracking_active else 'OFF'} Schedule={'RUN' if sequence_running else 'PAUSE'}", 55, status_color)
                    current_text = "condition: waiting"
                    if current_condition is not None:
                        current_text = (
                            f"A={current_condition.intensity_ratio:.2f} "
                            f"offset=({current_condition.offset_x_mm:.1f},"
                            f"{current_condition.offset_y_mm:.1f},"
                            f"{current_condition.offset_z_mm:.1f}) mm"
                        )
                    _put(frame_xy_bgr, current_text, 82, (0, 255, 255), 0.47)
                    velocity_text = "velocity unavailable"
                    if last_motion is not None:
                        velocity_text = (
                            f"v=({last_motion.vx_mm_s:.1f}, {last_motion.vy_mm_s:.1f}, "
                            f"{last_motion.vz_mm_s:.1f}) mm/s"
                        )
                    _put(frame_xy_bgr, velocity_text, 109, scale=0.47)
                    _put(
                        frame_xy_bgr,
                        f"Camera={camera_runtime.width_xy}x{camera_runtime.height_xy} "
                        f"FPS={display_fps:.1f} sync={sync_skew * 1000.0:.2f}ms",
                        136,
                        scale=0.45,
                    )
                    _put(frame_z_bgr, f"Phase: {phase}", 28, status_color)
                    z_text = "Z unavailable"
                    if sample is not None:
                        z_text = f"Z={sample.z_mm:.2f} mm  radius={sample.radius_z_px:.1f}px"
                    _put(frame_z_bgr, z_text, 55, (0, 255, 255))
                    _put(
                        frame_z_bgr,
                        "ENTER PID | S schedule | ESC exit",
                        frame_z_bgr.shape[0] - 18,
                        scale=0.46,
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
        print(f"[ERROR] Acoustic-force experiment failed: {error}")
        raise
    finally:
        logger.close()
        if sender is not None:
            try:
                sender.stop()
            except Exception:
                pass
        camera_runtime.close()
        cv2.destroyAllWindows()
        print(f"[INFO] Acoustic-force data: {output_directory}")
    return output_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure circular-STM force from stereo step responses."
    )
    parser.add_argument("--intensities", type=_parse_float_list, default=(0.5, 0.6, 0.7, 0.8))
    parser.add_argument("--x-offsets-mm", type=_parse_float_list, default=(0.0,))
    parser.add_argument("--y-offsets-mm", type=_parse_float_list, default=(0.0,))
    parser.add_argument("--z-offsets-mm", type=_parse_float_list, default=(0.0,))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--pulse-ms", type=float, default=60.0)
    parser.add_argument("--analysis-delay-ms", type=float, default=10.0)
    parser.add_argument("--minimum-fit-samples", type=int, default=6)
    parser.add_argument("--minimum-fit-span-ms", type=float, default=25.0)
    parser.add_argument("--baseline-intensity", type=float, default=0.6)
    parser.add_argument("--baseline-window-sec", type=float, default=0.30)
    parser.add_argument("--stable-sec", type=float, default=0.25)
    parser.add_argument("--recovery-sec", type=float, default=2.0)
    parser.add_argument("--stable-max-vxy-mm-s", type=float, default=40.0)
    parser.add_argument("--stable-max-abs-vz-mm-s", type=float, default=40.0)
    parser.add_argument("--stable-max-xy-error-mm", type=float, default=12.0)
    parser.add_argument("--stable-max-z-error-mm", type=float, default=8.0)
    parser.add_argument("--maximum-xy-displacement-mm", type=float, default=15.0)
    parser.add_argument("--maximum-abs-z-displacement-mm", type=float, default=15.0)
    parser.add_argument("--maximum-speed-mm-s", type=float, default=500.0)
    parser.add_argument("--arming-timeout-sec", type=float, default=0.20)
    parser.add_argument("--sphere-mass-g", type=float, default=0.5)
    parser.add_argument("--sphere-diameter-mm", type=float, default=38.0)
    parser.add_argument("--air-density-kg-m3", type=float, default=1.204)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--operator", default="")
    parser.add_argument("--interface-name", default="")
    parser.add_argument("--output-root", default="./tracking/acoustic_force_data")
    return parser


def validate_args(parser: argparse.ArgumentParser, args):
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.pulse_ms <= args.analysis_delay_ms:
        parser.error("--pulse-ms must be greater than --analysis-delay-ms")
    if any(value < 0.0 or value > 1.0 for value in args.intensities):
        parser.error("all --intensities values must be between 0 and 1")
    if not 0.0 <= args.baseline_intensity <= 1.0:
        parser.error("--baseline-intensity must be between 0 and 1")
    if args.sphere_mass_g <= 0.0 or args.sphere_diameter_mm <= 0.0:
        parser.error("sphere mass and diameter must be positive")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    cfg = AppConfig(
        radius=19.0,
        point_num=8,
        default_z=400.0,
        static_intensity_ratio=args.baseline_intensity,
        stm_freq_hz=100.0,
        autd_field_mode="stm_circle",
        enable_stereo_triangulation=True,
        use_stereo_position_for_control=True,
        stereo_camera_to_autd_npz="./tracking/calibration/stereo_camera_to_autd.npz",
        enable_base_move=False,
        enable_radius_change=False,
        enable_auto_demo=False,
        use_gravity_prediction_z=False,
        display_every_n_frames=2,
        autd_interface_name=args.interface_name,
        ball_mass_kg=args.sphere_mass_g * 1e-3,
        ball_diameter_mm=args.sphere_diameter_mm,
    )
    run_experiment(cfg, args)


if __name__ == "__main__":
    main()
