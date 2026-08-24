from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionEstimate3D:
    measurement_time_sec: float
    x_mm: float
    y_mm: float
    z_mm: float
    vx_mm_s: float
    vy_mm_s: float
    vz_mm_s: float
    sample_count: int
    history_span_sec: float
    velocity_ready: bool


class WindowedMotionEstimator3D:
    """Small least-squares estimator used only by the force experiment."""

    def __init__(
        self,
        *,
        window_sec: float = 0.060,
        minimum_samples: int = 3,
        minimum_span_sec: float = 0.010,
    ):
        self.window_sec = max(1e-3, float(window_sec))
        self.minimum_samples = max(2, int(minimum_samples))
        self.minimum_span_sec = max(0.0, float(minimum_span_sec))
        self._samples: deque[tuple[float, np.ndarray]] = deque()
        self.latest: MotionEstimate3D | None = None

    def reset(self):
        self._samples.clear()
        self.latest = None

    def update(self, timestamp: float, x_mm: float, y_mm: float, z_mm: float):
        values = np.asarray([x_mm, y_mm, z_mm], dtype=float)
        if not np.all(np.isfinite(values)):
            return self.latest
        t = float(timestamp)
        if self._samples and t <= self._samples[-1][0]:
            return self.latest

        self._samples.append((t, values))
        oldest = t - self.window_sec
        while len(self._samples) > 1 and self._samples[0][0] < oldest:
            self._samples.popleft()

        times = np.asarray([sample[0] for sample in self._samples])
        positions = np.asarray([sample[1] for sample in self._samples])
        span = float(times[-1] - times[0])
        ready = (
            len(self._samples) >= self.minimum_samples
            and span + 1e-9 >= self.minimum_span_sec
        )
        velocity = np.zeros(3, dtype=float)
        fitted_position = positions[-1]
        if ready:
            relative = times - times[-1]
            design = np.column_stack([np.ones_like(relative), relative])
            coefficients, *_ = np.linalg.lstsq(
                design,
                positions,
                rcond=None,
            )
            fitted_position = coefficients[0]
            velocity = coefficients[1]

        self.latest = MotionEstimate3D(
            measurement_time_sec=t,
            x_mm=float(fitted_position[0]),
            y_mm=float(fitted_position[1]),
            z_mm=float(fitted_position[2]),
            vx_mm_s=float(velocity[0]),
            vy_mm_s=float(velocity[1]),
            vz_mm_s=float(velocity[2]),
            sample_count=len(self._samples),
            history_span_sec=span,
            velocity_ready=ready,
        )
        return self.latest


@dataclass(frozen=True)
class PulseSample:
    measurement_time_sec: float
    x_mm: float
    y_mm: float
    z_mm: float


@dataclass(frozen=True)
class AccelerationEstimate3D:
    ax_mm_s2: float
    ay_mm_s2: float
    az_mm_s2: float
    sample_count: int
    span_sec: float


def estimate_pulse_acceleration(
    samples: list[PulseSample],
    *,
    minimum_samples: int = 5,
    minimum_span_sec: float = 0.020,
) -> AccelerationEstimate3D | None:
    """Fit p(t)=p0+v0*t+0.5*a*t^2 over one short bias pulse."""

    if len(samples) < int(minimum_samples):
        return None
    times = np.asarray([sample.measurement_time_sec for sample in samples])
    positions = np.asarray(
        [[sample.x_mm, sample.y_mm, sample.z_mm] for sample in samples]
    )
    relative = times - times[0]
    span = float(relative[-1])
    if span + 1e-9 < float(minimum_span_sec):
        return None
    design = np.column_stack(
        [
            np.ones_like(relative),
            relative,
            0.5 * relative**2,
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, positions, rcond=None)
    acceleration = coefficients[2]
    return AccelerationEstimate3D(
        ax_mm_s2=float(acceleration[0]),
        ay_mm_s2=float(acceleration[1]),
        az_mm_s2=float(acceleration[2]),
        sample_count=len(samples),
        span_sec=span,
    )


def pulse_safety_reason(
    *,
    measurement_valid: bool,
    initial_xyz: tuple[float, float, float],
    current_xyz: tuple[float, float, float] | None,
    vz_mm_s: float | None,
    maximum_xy_displacement_mm: float,
    maximum_z_drop_mm: float,
    maximum_abs_vz_mm_s: float,
) -> str | None:
    if not measurement_valid or current_xyz is None:
        return "stereo_measurement_lost"
    dx = float(current_xyz[0]) - float(initial_xyz[0])
    dy = float(current_xyz[1]) - float(initial_xyz[1])
    if np.hypot(dx, dy) > float(maximum_xy_displacement_mm):
        return "xy_displacement_limit"
    if float(initial_xyz[2]) - float(current_xyz[2]) > float(maximum_z_drop_mm):
        return "z_drop_limit"
    if vz_mm_s is not None and abs(float(vz_mm_s)) > float(maximum_abs_vz_mm_s):
        return "vz_limit"
    return None


FRAME_LOG_HEADER = [
    "wall_timestamp",
    "event",
    "trial_id",
    "phase",
    "bias_side",
    "bias_angle_deg",
    "bias_level",
    "slot_counts",
    "measurement_time_sec",
    "x_mm",
    "y_mm",
    "z_mm",
    "vx_mm_s",
    "vy_mm_s",
    "vz_mm_s",
    "target_x_mm",
    "target_y_mm",
    "target_z_mm",
    "xy_displacement_mm",
    "z_drop_mm",
    "pulse_elapsed_ms",
    "command_seq",
    "reason",
]


SUMMARY_LOG_HEADER = [
    "wall_timestamp",
    "trial_id",
    "bias_side",
    "bias_angle_deg",
    "bias_level",
    "slot_counts",
    "sample_count",
    "field_duration_ms",
    "sample_span_ms",
    "ax_mm_s2",
    "ay_mm_s2",
    "az_mm_s2",
    "force_x_mN",
    "force_y_mN",
    "force_xy_mN",
    "force_along_bias_mN",
    "force_perpendicular_mN",
    "maximum_xy_displacement_mm",
    "maximum_z_drop_mm",
    "stop_reason",
]


class XYForceLogger:
    def __init__(self, frame_path: str, summary_path: str, ball_mass_kg: float):
        self.frame_path = str(frame_path)
        self.summary_path = str(summary_path)
        self.ball_mass_kg = float(ball_mass_kg)
        self._frame_file = None
        self._summary_file = None
        self._frame_writer = None
        self._summary_writer = None
        self._frame_rows_since_flush = 0

    @staticmethod
    def _open(path: str, header):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        file = open(path, "a", newline="", encoding="utf-8")
        writer = csv.writer(file)
        if not exists:
            writer.writerow(header)
            file.flush()
        return file, writer

    def open(self):
        self._frame_file, self._frame_writer = self._open(
            self.frame_path,
            FRAME_LOG_HEADER,
        )
        self._summary_file, self._summary_writer = self._open(
            self.summary_path,
            SUMMARY_LOG_HEADER,
        )

    def write_frame(self, row):
        if self._frame_writer is None:
            return
        self._frame_writer.writerow(row)
        self._frame_rows_since_flush += 1
        if self._frame_rows_since_flush >= 32 or row[1] != "frame":
            self._frame_file.flush()
            self._frame_rows_since_flush = 0

    def write_summary(
        self,
        *,
        trial_id: int,
        bias_side: str,
        bias_angle_deg: float,
        bias_level: float,
        slot_counts: tuple[int, ...],
        samples: list[PulseSample],
        field_duration_sec: float,
        maximum_xy_displacement_mm: float,
        maximum_z_drop_mm: float,
        stop_reason: str,
    ):
        if self._summary_writer is None:
            return None
        estimate = estimate_pulse_acceleration(samples)
        if estimate is None:
            ax = ay = az = None
            sample_count = len(samples)
            sample_span_ms = (
                0.0
                if len(samples) < 2
                else (
                    samples[-1].measurement_time_sec
                    - samples[0].measurement_time_sec
                )
                * 1000.0
            )
        else:
            ax, ay, az = (
                estimate.ax_mm_s2,
                estimate.ay_mm_s2,
                estimate.az_mm_s2,
            )
            sample_count = estimate.sample_count
            sample_span_ms = estimate.span_sec * 1000.0

        force_x = None if ax is None else self.ball_mass_kg * ax
        force_y = None if ay is None else self.ball_mass_kg * ay
        force_xy = (
            None
            if force_x is None or force_y is None
            else float(np.hypot(force_x, force_y))
        )
        angle = np.deg2rad(float(bias_angle_deg))
        force_along = (
            None
            if force_x is None or force_y is None
            else force_x * np.cos(angle) + force_y * np.sin(angle)
        )
        force_perpendicular = (
            None
            if force_x is None or force_y is None
            else -force_x * np.sin(angle) + force_y * np.cos(angle)
        )
        number = lambda value: "" if value is None else f"{float(value):.6f}"
        self._summary_writer.writerow(
            [
                f"{time.time():.6f}",
                int(trial_id),
                str(bias_side),
                f"{float(bias_angle_deg):.3f}",
                f"{float(bias_level):.3f}",
                ";".join(str(value) for value in slot_counts),
                int(sample_count),
                f"{float(field_duration_sec) * 1000.0:.3f}",
                f"{float(sample_span_ms):.3f}",
                number(ax),
                number(ay),
                number(az),
                number(force_x),
                number(force_y),
                number(force_xy),
                number(force_along),
                number(force_perpendicular),
                f"{float(maximum_xy_displacement_mm):.3f}",
                f"{float(maximum_z_drop_mm):.3f}",
                str(stop_reason),
            ]
        )
        self._summary_file.flush()
        return estimate

    def close(self):
        for file in (self._frame_file, self._summary_file):
            try:
                if file is not None:
                    file.close()
            except Exception:
                pass
        self._frame_file = None
        self._summary_file = None
        self._frame_writer = None
        self._summary_writer = None
        self._frame_rows_since_flush = 0
