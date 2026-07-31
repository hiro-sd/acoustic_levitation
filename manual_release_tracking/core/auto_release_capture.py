from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass

import numpy as np


CAPTURE_ALIGN = "CAPTURE_ALIGN"


@dataclass(frozen=True)
class MotionEstimate3D:
    measurement_time_sec: float
    x_mm: float
    y_mm: float
    z_mm: float
    vx_mm_s: float
    vy_mm_s: float
    vz_mm_s: float


@dataclass(frozen=True)
class PredictedCapture:
    horizon_sec: float
    x_mm: float
    y_mm: float
    z_mm: float
    vx_mm_s: float
    vy_mm_s: float
    vz_mm_s: float


@dataclass(frozen=True)
class BrakingPlan:
    start_time_sec: float
    initial_z_mm: float
    initial_vz_mm_s: float
    duration_sec: float
    acceleration_mm_s2: float
    stop_z_mm: float
    stopping_distance_mm: float
    available_drop_mm: float
    required_force_mN: float
    maximum_force_mN: float
    force_saturated: bool
    workspace_limited: bool
    timeout_limited: bool
    feasible: bool


@dataclass(frozen=True)
class BrakingReference:
    elapsed_sec: float
    z_mm: float
    vz_mm_s: float
    acceleration_mm_s2: float
    complete: bool


class StereoMotionEstimator:
    """Lightweight filtered position/velocity estimator that runs before control."""

    def __init__(
        self,
        *,
        position_current_weight: float = 0.55,
        velocity_current_weight: float = 0.50,
        max_update_gap_sec: float = 0.10,
    ):
        self.position_current_weight = float(
            np.clip(position_current_weight, 0.0, 1.0)
        )
        self.velocity_current_weight = float(
            np.clip(velocity_current_weight, 0.0, 1.0)
        )
        self.max_update_gap_sec = float(max_update_gap_sec)
        self._estimate: MotionEstimate3D | None = None

    @property
    def latest(self) -> MotionEstimate3D | None:
        return self._estimate

    def reset(self):
        self._estimate = None

    def update(
        self,
        measurement_time_sec: float,
        x_mm: float | None,
        y_mm: float | None,
        z_mm: float | None,
    ) -> MotionEstimate3D | None:
        if x_mm is None or y_mm is None or z_mm is None:
            return self._estimate

        values = np.asarray([x_mm, y_mm, z_mm], dtype=float)
        if not np.all(np.isfinite(values)):
            return self._estimate

        timestamp = float(measurement_time_sec)
        previous = self._estimate
        if previous is None:
            self._estimate = MotionEstimate3D(
                timestamp,
                float(values[0]),
                float(values[1]),
                float(values[2]),
                0.0,
                0.0,
                0.0,
            )
            return self._estimate

        dt = timestamp - previous.measurement_time_sec
        if dt <= 0.0:
            return previous

        if dt > self.max_update_gap_sec:
            self._estimate = MotionEstimate3D(
                timestamp,
                float(values[0]),
                float(values[1]),
                float(values[2]),
                0.0,
                0.0,
                0.0,
            )
            return self._estimate

        previous_position = np.asarray(
            [previous.x_mm, previous.y_mm, previous.z_mm],
            dtype=float,
        )
        position_weight = self.position_current_weight
        filtered_position = (
            position_weight * values
            + (1.0 - position_weight) * previous_position
        )
        raw_velocity = (filtered_position - previous_position) / max(1e-3, dt)
        previous_velocity = np.asarray(
            [previous.vx_mm_s, previous.vy_mm_s, previous.vz_mm_s],
            dtype=float,
        )
        velocity_weight = self.velocity_current_weight
        filtered_velocity = (
            velocity_weight * raw_velocity
            + (1.0 - velocity_weight) * previous_velocity
        )

        self._estimate = MotionEstimate3D(
            timestamp,
            float(filtered_position[0]),
            float(filtered_position[1]),
            float(filtered_position[2]),
            float(filtered_velocity[0]),
            float(filtered_velocity[1]),
            float(filtered_velocity[2]),
        )
        return self._estimate


def predict_capture_position(
    estimate: MotionEstimate3D,
    *,
    now_sec: float,
    actuation_delay_sec: float,
    gravity_mm_s2: float,
    apply_gravity: bool = True,
) -> PredictedCapture:
    measurement_age_sec = max(
        0.0,
        float(now_sec) - estimate.measurement_time_sec,
    )
    horizon = measurement_age_sec + max(0.0, float(actuation_delay_sec))
    gravity = float(gravity_mm_s2) if apply_gravity else 0.0
    predicted_z = (
        estimate.z_mm
        + estimate.vz_mm_s * horizon
        - 0.5 * gravity * horizon * horizon
    )
    predicted_vz = estimate.vz_mm_s - gravity * horizon
    return PredictedCapture(
        horizon_sec=float(horizon),
        x_mm=float(estimate.x_mm + estimate.vx_mm_s * horizon),
        y_mm=float(estimate.y_mm + estimate.vy_mm_s * horizon),
        z_mm=float(predicted_z),
        vx_mm_s=float(estimate.vx_mm_s),
        vy_mm_s=float(estimate.vy_mm_s),
        vz_mm_s=float(predicted_vz),
    )


def make_braking_plan(
    *,
    start_time_sec: float,
    initial_z_mm: float,
    initial_vz_mm_s: float,
    nominal_duration_sec: float,
    maximum_duration_sec: float,
    minimum_duration_sec: float,
    ball_mass_kg: float,
    gravity_mm_s2: float,
    maximum_force_mN: float,
    z_min_mm: float,
    workspace_margin_mm: float,
) -> BrakingPlan:
    """Create a constant-deceleration Z trajectory within force/workspace limits."""
    mass_kg = max(1e-9, float(ball_mass_kg))
    gravity = max(0.0, float(gravity_mm_s2))
    initial_vz = float(initial_vz_mm_s)
    downward_speed = max(0.0, -initial_vz)
    minimum_duration = max(1e-3, float(minimum_duration_sec))
    nominal_duration = max(minimum_duration, float(nominal_duration_sec))
    maximum_duration = max(minimum_duration, float(maximum_duration_sec))
    maximum_force = max(0.0, float(maximum_force_mN))
    lower_boundary_z = float(z_min_mm) + max(0.0, float(workspace_margin_mm))
    available_drop = max(0.0, float(initial_z_mm) - lower_boundary_z)

    if downward_speed <= 1e-9:
        weight_mn = mass_kg * (gravity / 1000.0) * 1000.0
        return BrakingPlan(
            start_time_sec=float(start_time_sec),
            initial_z_mm=float(initial_z_mm),
            initial_vz_mm_s=initial_vz,
            duration_sec=0.0,
            acceleration_mm_s2=0.0,
            stop_z_mm=float(initial_z_mm),
            stopping_distance_mm=0.0,
            available_drop_mm=available_drop,
            required_force_mN=float(weight_mn),
            maximum_force_mN=maximum_force,
            force_saturated=weight_mn > maximum_force,
            workspace_limited=False,
            timeout_limited=False,
            feasible=weight_mn <= maximum_force,
        )

    maximum_gross_acceleration = (
        maximum_force / 1000.0 / mass_kg * 1000.0
    )
    maximum_net_acceleration = maximum_gross_acceleration - gravity
    if maximum_net_acceleration > 0.0:
        force_limited_duration = downward_speed / maximum_net_acceleration
    else:
        force_limited_duration = float("inf")

    ideal_duration = max(nominal_duration, force_limited_duration)
    workspace_max_duration = (
        2.0 * available_drop / downward_speed
        if available_drop > 0.0
        else 0.0
    )
    workspace_limited = ideal_duration > workspace_max_duration
    timeout_limited = ideal_duration > maximum_duration
    feasible = (
        np.isfinite(ideal_duration)
        and not workspace_limited
        and not timeout_limited
    )

    if feasible:
        duration = ideal_duration
    else:
        candidates = [maximum_duration]
        if workspace_max_duration > 0.0:
            candidates.append(workspace_max_duration)
        if np.isfinite(ideal_duration):
            candidates.append(ideal_duration)
        duration = max(minimum_duration, min(candidates))

    acceleration = downward_speed / duration
    stopping_distance = 0.5 * downward_speed * duration
    stop_z = max(lower_boundary_z, float(initial_z_mm) - stopping_distance)
    required_force = (
        mass_kg
        * ((gravity + acceleration) / 1000.0)
        * 1000.0
    )
    force_saturated = required_force > maximum_force + 1e-9

    return BrakingPlan(
        start_time_sec=float(start_time_sec),
        initial_z_mm=float(initial_z_mm),
        initial_vz_mm_s=initial_vz,
        duration_sec=float(duration),
        acceleration_mm_s2=float(acceleration),
        stop_z_mm=float(stop_z),
        stopping_distance_mm=float(stopping_distance),
        available_drop_mm=float(available_drop),
        required_force_mN=float(required_force),
        maximum_force_mN=maximum_force,
        force_saturated=bool(force_saturated),
        workspace_limited=bool(workspace_limited),
        timeout_limited=bool(timeout_limited),
        feasible=bool(feasible and not force_saturated),
    )


def braking_reference_at(
    plan: BrakingPlan,
    *,
    now_sec: float,
) -> BrakingReference:
    if plan.duration_sec <= 0.0:
        return BrakingReference(
            elapsed_sec=max(0.0, float(now_sec) - plan.start_time_sec),
            z_mm=plan.stop_z_mm,
            vz_mm_s=0.0,
            acceleration_mm_s2=0.0,
            complete=True,
        )

    elapsed = max(0.0, float(now_sec) - plan.start_time_sec)
    t = min(elapsed, plan.duration_sec)
    z_ref = (
        plan.initial_z_mm
        + plan.initial_vz_mm_s * t
        + 0.5 * plan.acceleration_mm_s2 * t * t
    )
    vz_ref = plan.initial_vz_mm_s + plan.acceleration_mm_s2 * t
    complete = elapsed >= plan.duration_sec
    if complete:
        z_ref = plan.stop_z_mm
        vz_ref = 0.0

    return BrakingReference(
        elapsed_sec=float(elapsed),
        z_mm=float(z_ref),
        vz_mm_s=float(vz_ref),
        acceleration_mm_s2=float(plan.acceleration_mm_s2),
        complete=bool(complete),
    )


def trajectory_target_z(
    *,
    reference_z_mm: float,
    reference_vz_mm_s: float,
    measured_z_mm: float,
    measured_vz_mm_s: float,
    position_gain: float,
    velocity_gain: float,
    maximum_correction_mm: float,
) -> tuple[float, float]:
    correction = (
        float(position_gain)
        * (float(reference_z_mm) - float(measured_z_mm))
        + float(velocity_gain)
        * (float(reference_vz_mm_s) - float(measured_vz_mm_s))
    )
    correction = float(
        np.clip(
            correction,
            -abs(float(maximum_correction_mm)),
            abs(float(maximum_correction_mm)),
        )
    )
    return float(reference_z_mm) + correction, correction


def capture_intensity_for_vz(
    vz_mm_s: float,
    *,
    normal_ratio: float = 0.6,
    slow_ratio: float = 0.7,
    maximum_ratio: float = 0.8,
    upward_ratio: float = 0.5,
    fast_down_threshold_mm_s: float = -100.0,
    slow_down_threshold_mm_s: float = -30.0,
    upward_threshold_mm_s: float = 20.0,
) -> float:
    """Small release-specific braking schedule; z is positive upward."""
    vz = float(vz_mm_s)
    if vz > float(upward_threshold_mm_s):
        return float(upward_ratio)
    if vz < float(fast_down_threshold_mm_s):
        return float(maximum_ratio)
    if vz < float(slow_down_threshold_mm_s):
        return float(slow_ratio)
    return float(normal_ratio)


def limit_upward_capture_target_z(
    predicted_z_mm: float,
    previous_target_z_mm: float,
    *,
    upward_brake_active: bool,
) -> float:
    """Do not let the capture field chase a sphere that is moving upward."""
    if not upward_brake_active:
        return float(predicted_z_mm)
    return float(min(predicted_z_mm, previous_target_z_mm))


def capture_align_target_xy(
    initial_prediction: PredictedCapture | None,
    current_prediction: PredictedCapture,
) -> tuple[float, float]:
    """Keep XY at the first capture command instead of chasing lateral motion."""
    anchor = (
        current_prediction
        if initial_prediction is None
        else initial_prediction
    )
    return float(anchor.x_mm), float(anchor.y_mm)


def update_capture_stability(
    *,
    now_sec: float,
    vz_mm_s: float,
    release_confirmed: bool,
    stable_since_sec: float | None,
    maximum_abs_vz_mm_s: float,
    required_duration_sec: float,
    trajectory_complete: bool = True,
    vxy_mm_s: float = 0.0,
    maximum_vxy_mm_s: float = float("inf"),
    xy_distance_mm: float = 0.0,
    maximum_xy_distance_mm: float = float("inf"),
    z_tracking_error_mm: float = 0.0,
    maximum_z_tracking_error_mm: float = float("inf"),
) -> tuple[float | None, bool]:
    stable = (
        bool(release_confirmed)
        and bool(trajectory_complete)
        and abs(float(vz_mm_s)) <= float(maximum_abs_vz_mm_s)
        and abs(float(vxy_mm_s)) <= float(maximum_vxy_mm_s)
        and abs(float(xy_distance_mm)) <= float(maximum_xy_distance_mm)
        and abs(float(z_tracking_error_mm))
        <= float(maximum_z_tracking_error_mm)
    )
    if not stable:
        return None, False
    if stable_since_sec is None:
        return float(now_sec), False
    ready = (
        float(now_sec) - float(stable_since_sec)
        >= float(required_duration_sec)
    )
    return float(stable_since_sec), bool(ready)


CAPTURE_LOG_HEADER = [
    "wall_timestamp",
    "event",
    "release_timestamp_ms",
    "elapsed_from_release_ms",
    "measurement_time_sec",
    "measurement_age_ms",
    "x_mm",
    "y_mm",
    "z_mm",
    "vx_mm_s",
    "vy_mm_s",
    "vz_mm_s",
    "prediction_horizon_ms",
    "predicted_x_mm",
    "predicted_y_mm",
    "predicted_z_mm",
    "predicted_vz_mm_s",
    "target_x_mm",
    "target_y_mm",
    "target_z_mm",
    "intensity",
    "command_seq",
    "reason",
]


class AutoReleaseCaptureLogger:
    """Append-only event log for release-to-AUTD timing and capture handoff."""

    def __init__(self, path: str):
        self.path = str(path)

    @staticmethod
    def _number(value, digits: int = 6):
        if value is None:
            return ""
        return f"{float(value):.{digits}f}"

    def write(
        self,
        *,
        event: str,
        release_timestamp_ms: int | None = None,
        event_time_sec: float | None = None,
        estimate: MotionEstimate3D | None = None,
        prediction: PredictedCapture | None = None,
        target=None,
        intensity: float | None = None,
        command_seq: int | None = None,
        reason: str = "",
    ):
        now_sec = time.perf_counter() if event_time_sec is None else float(event_time_sec)
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0

        elapsed_ms = (
            None
            if release_timestamp_ms is None
            else now_sec * 1000.0 - float(release_timestamp_ms)
        )
        measurement_age_ms = (
            None
            if estimate is None
            else max(0.0, now_sec - estimate.measurement_time_sec) * 1000.0
        )

        with open(self.path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not exists:
                writer.writerow(CAPTURE_LOG_HEADER)
            writer.writerow(
                [
                    f"{time.time():.6f}",
                    str(event),
                    "" if release_timestamp_ms is None else int(release_timestamp_ms),
                    self._number(elapsed_ms, 3),
                    self._number(
                        None if estimate is None else estimate.measurement_time_sec
                    ),
                    self._number(measurement_age_ms, 3),
                    self._number(None if estimate is None else estimate.x_mm, 3),
                    self._number(None if estimate is None else estimate.y_mm, 3),
                    self._number(None if estimate is None else estimate.z_mm, 3),
                    self._number(None if estimate is None else estimate.vx_mm_s, 3),
                    self._number(None if estimate is None else estimate.vy_mm_s, 3),
                    self._number(None if estimate is None else estimate.vz_mm_s, 3),
                    self._number(
                        None if prediction is None else prediction.horizon_sec * 1000.0,
                        3,
                    ),
                    self._number(None if prediction is None else prediction.x_mm, 3),
                    self._number(None if prediction is None else prediction.y_mm, 3),
                    self._number(None if prediction is None else prediction.z_mm, 3),
                    self._number(None if prediction is None else prediction.vz_mm_s, 3),
                    self._number(None if target is None else target.x, 3),
                    self._number(None if target is None else target.y, 3),
                    self._number(None if target is None else target.z, 3),
                    self._number(intensity, 3),
                    "" if command_seq is None else int(command_seq),
                    str(reason),
                ]
            )


DELAY_LOG_HEADER = [
    "wall_timestamp",
    "release_timestamp_ms",
    "measurement_time_sec",
    "command_enqueued_time_sec",
    "sender_dequeued_time_sec",
    "autd_send_started_time_sec",
    "autd_send_completed_time_sec",
    "release_to_enqueue_ms",
    "measurement_to_enqueue_ms",
    "enqueue_to_send_complete_ms",
    "enqueue_to_dequeue_ms",
    "dequeue_to_send_start_ms",
    "autd_send_duration_ms",
    "measurement_to_send_complete_ms",
    "release_to_send_complete_ms",
    "configured_prediction_delay_ms",
    "prediction_horizon_ms",
    "prediction_shortfall_ms",
    "candidate_command_seq",
    "sent_command_seq",
    "command_superseded",
    "x_mm",
    "y_mm",
    "z_mm",
    "vx_mm_s",
    "vy_mm_s",
    "vz_mm_s",
    "predicted_x_mm",
    "predicted_y_mm",
    "predicted_z_mm",
    "target_x_mm",
    "target_y_mm",
    "target_z_mm",
    "intensity",
]


class AutoReleaseDelayLogger:
    """One row per initial release command for effective-delay estimation."""

    def __init__(self, path: str):
        self.path = str(path)

    @staticmethod
    def _number(value, digits: int = 6):
        if value is None:
            return ""
        return f"{float(value):.{digits}f}"

    def write(
        self,
        *,
        release_timestamp_ms: int,
        estimate: MotionEstimate3D,
        prediction: PredictedCapture,
        sent,
        candidate_command_seq: int,
        configured_prediction_delay_sec: float,
    ):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0

        release_time_sec = float(release_timestamp_ms) / 1000.0
        enqueued_time_sec = float(sent.target.enqueued_time_sec)
        dequeued_time_sec = float(sent.dequeued_time_sec)
        send_started_time_sec = float(sent.send_started_time_sec)
        send_completed_time_sec = float(sent.sent_time_sec)
        measurement_to_send_sec = (
            send_completed_time_sec - float(estimate.measurement_time_sec)
        )
        prediction_shortfall_sec = (
            measurement_to_send_sec - float(prediction.horizon_sec)
        )

        with open(self.path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not exists:
                writer.writerow(DELAY_LOG_HEADER)
            writer.writerow(
                [
                    f"{time.time():.6f}",
                    int(release_timestamp_ms),
                    self._number(estimate.measurement_time_sec),
                    self._number(enqueued_time_sec),
                    self._number(dequeued_time_sec),
                    self._number(send_started_time_sec),
                    self._number(send_completed_time_sec),
                    self._number(
                        (enqueued_time_sec - release_time_sec) * 1000.0,
                        3,
                    ),
                    self._number(
                        (
                            enqueued_time_sec
                            - float(estimate.measurement_time_sec)
                        )
                        * 1000.0,
                        3,
                    ),
                    self._number(
                        (send_completed_time_sec - enqueued_time_sec)
                        * 1000.0,
                        3,
                    ),
                    self._number(
                        (dequeued_time_sec - enqueued_time_sec) * 1000.0,
                        3,
                    ),
                    self._number(
                        (send_started_time_sec - dequeued_time_sec) * 1000.0,
                        3,
                    ),
                    self._number(
                        (send_completed_time_sec - send_started_time_sec)
                        * 1000.0,
                        3,
                    ),
                    self._number(measurement_to_send_sec * 1000.0, 3),
                    self._number(
                        (send_completed_time_sec - release_time_sec) * 1000.0,
                        3,
                    ),
                    self._number(
                        float(configured_prediction_delay_sec) * 1000.0,
                        3,
                    ),
                    self._number(prediction.horizon_sec * 1000.0, 3),
                    self._number(prediction_shortfall_sec * 1000.0, 3),
                    int(candidate_command_seq),
                    int(sent.sequence),
                    int(sent.sequence != int(candidate_command_seq)),
                    self._number(estimate.x_mm, 3),
                    self._number(estimate.y_mm, 3),
                    self._number(estimate.z_mm, 3),
                    self._number(estimate.vx_mm_s, 3),
                    self._number(estimate.vy_mm_s, 3),
                    self._number(estimate.vz_mm_s, 3),
                    self._number(prediction.x_mm, 3),
                    self._number(prediction.y_mm, 3),
                    self._number(prediction.z_mm, 3),
                    self._number(sent.target.x, 3),
                    self._number(sent.target.y, 3),
                    self._number(sent.target.z, 3),
                    self._number(sent.target.intensity_ratio, 3),
                ]
            )


TRAJECTORY_LOG_HEADER = [
    "wall_timestamp",
    "release_timestamp_ms",
    "measurement_time_sec",
    "elapsed_from_field_ms",
    "plan_duration_ms",
    "plan_stop_z_mm",
    "plan_stopping_distance_mm",
    "plan_required_force_mN",
    "plan_maximum_force_mN",
    "plan_feasible",
    "plan_force_saturated",
    "plan_workspace_limited",
    "x_mm",
    "y_mm",
    "z_mm",
    "vx_mm_s",
    "vy_mm_s",
    "vz_mm_s",
    "reference_z_mm",
    "reference_vz_mm_s",
    "reference_acceleration_mm_s2",
    "z_error_mm",
    "vz_error_mm_s",
    "target_z_correction_mm",
    "target_x_mm",
    "target_y_mm",
    "target_z_mm",
    "xy_distance_mm",
    "commanded_intensity",
    "upward_brake_active",
    "trajectory_complete",
]


class AutoReleaseTrajectoryLogger:
    """Automatic per-measurement log for braking-trajectory validation."""

    def __init__(self, path: str):
        self.path = str(path)
        self._file = None
        self._writer = None

    @staticmethod
    def _number(value, digits: int = 6):
        if value is None:
            return ""
        return f"{float(value):.{digits}f}"

    def write(
        self,
        *,
        release_timestamp_ms: int,
        estimate: MotionEstimate3D,
        plan: BrakingPlan,
        reference: BrakingReference,
        target,
        target_z_correction_mm: float,
        xy_distance_mm: float,
        commanded_intensity: float,
        upward_brake_active: bool,
    ):
        if self._writer is None:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
            self._file = open(self.path, "a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            if not exists:
                self._writer.writerow(TRAJECTORY_LOG_HEADER)

        self._writer.writerow(
            [
                    f"{time.time():.6f}",
                    int(release_timestamp_ms),
                    self._number(estimate.measurement_time_sec),
                    self._number(reference.elapsed_sec * 1000.0, 3),
                    self._number(plan.duration_sec * 1000.0, 3),
                    self._number(plan.stop_z_mm, 3),
                    self._number(plan.stopping_distance_mm, 3),
                    self._number(plan.required_force_mN, 3),
                    self._number(plan.maximum_force_mN, 3),
                    int(plan.feasible),
                    int(plan.force_saturated),
                    int(plan.workspace_limited),
                    self._number(estimate.x_mm, 3),
                    self._number(estimate.y_mm, 3),
                    self._number(estimate.z_mm, 3),
                    self._number(estimate.vx_mm_s, 3),
                    self._number(estimate.vy_mm_s, 3),
                    self._number(estimate.vz_mm_s, 3),
                    self._number(reference.z_mm, 3),
                    self._number(reference.vz_mm_s, 3),
                    self._number(reference.acceleration_mm_s2, 3),
                    self._number(reference.z_mm - estimate.z_mm, 3),
                    self._number(reference.vz_mm_s - estimate.vz_mm_s, 3),
                    self._number(target_z_correction_mm, 3),
                    self._number(target.x, 3),
                    self._number(target.y, 3),
                    self._number(target.z, 3),
                    self._number(xy_distance_mm, 3),
                    self._number(commanded_intensity, 3),
                    int(upward_brake_active),
                    int(reference.complete),
            ]
        )

    def close(self):
        try:
            if self._file is not None:
                self._file.close()
        finally:
            self._file = None
            self._writer = None
