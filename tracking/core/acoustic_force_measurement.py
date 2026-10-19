from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ForceStepSample:
    measurement_time_sec: float
    x_mm: float
    y_mm: float
    z_mm: float
    u_xy_px: float
    v_xy_px: float
    radius_xy_px: float
    u_z_px: float
    v_z_px: float
    radius_z_px: float
    sync_skew_sec: float


@dataclass(frozen=True)
class AxisQuadraticFit:
    position_at_first_sample_mm: float
    velocity_at_first_sample_mm_s: float
    acceleration_mm_s2: float
    acceleration_standard_error_mm_s2: float
    residual_rms_mm: float
    r_squared: float


@dataclass(frozen=True)
class StepAccelerationEstimate:
    x: AxisQuadraticFit
    y: AxisQuadraticFit
    z: AxisQuadraticFit
    sample_count: int
    span_sec: float
    first_sample_time_sec: float
    last_sample_time_sec: float
    mean_x_mm: float
    mean_y_mm: float
    mean_z_mm: float


@dataclass(frozen=True)
class PositionResolutionEstimate:
    sample_count: int
    span_sec: float
    std_x_mm: float
    std_y_mm: float
    std_z_mm: float
    robust_sigma_x_mm: float
    robust_sigma_y_mm: float
    robust_sigma_z_mm: float
    single_difference_95_x_mm: float
    single_difference_95_y_mm: float
    single_difference_95_z_mm: float
    std_u_xy_px: float
    std_v_xy_px: float
    std_u_z_px: float
    std_v_z_px: float
    median_radius_xy_px: float
    median_radius_z_px: float
    approximate_scale_xy_mm_per_px: float
    approximate_scale_z_mm_per_px: float


def sphere_volume_m3(diameter_mm: float) -> float:
    diameter_m = float(diameter_mm) * 1e-3
    return math.pi * diameter_m**3 / 6.0


def buoyancy_force_mN(
    *,
    diameter_mm: float,
    air_density_kg_m3: float,
    gravity_m_s2: float = 9.80665,
) -> float:
    return (
        float(air_density_kg_m3)
        * sphere_volume_m3(diameter_mm)
        * float(gravity_m_s2)
        * 1000.0
    )


def acoustic_force_mN(
    *,
    mass_kg: float,
    acceleration_mm_s2: float,
    gravity_mm_s2: float,
    buoyancy_mN: float,
) -> float:
    """Return upward acoustic force; kg*mm/s^2 is numerically mN."""

    return (
        float(mass_kg)
        * (float(acceleration_mm_s2) + float(gravity_mm_s2))
        - float(buoyancy_mN)
    )


def _fit_axis(relative_time_sec: np.ndarray, values_mm: np.ndarray):
    design = np.column_stack(
        [
            np.ones_like(relative_time_sec),
            relative_time_sec,
            0.5 * relative_time_sec**2,
        ]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, values_mm, rcond=None)
    fitted = design @ coefficients
    residual = values_mm - fitted
    residual_sum_squares = float(residual @ residual)
    centered = values_mm - float(np.mean(values_mm))
    total_sum_squares = float(centered @ centered)
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 1e-15
        else 1.0
    )
    dof = int(values_mm.size - rank)
    acceleration_standard_error = float("nan")
    if rank == 3 and dof > 0:
        residual_variance = residual_sum_squares / dof
        covariance = residual_variance * np.linalg.pinv(design.T @ design)
        acceleration_standard_error = math.sqrt(max(0.0, float(covariance[2, 2])))
    return AxisQuadraticFit(
        position_at_first_sample_mm=float(coefficients[0]),
        velocity_at_first_sample_mm_s=float(coefficients[1]),
        acceleration_mm_s2=float(coefficients[2]),
        acceleration_standard_error_mm_s2=acceleration_standard_error,
        residual_rms_mm=float(math.sqrt(residual_sum_squares / values_mm.size)),
        r_squared=float(r_squared),
    )


def estimate_step_acceleration(
    samples: Iterable[ForceStepSample],
    *,
    field_started_time_sec: float,
    analysis_delay_sec: float = 0.010,
    field_stopped_time_sec: float | None = None,
    minimum_samples: int = 6,
    minimum_span_sec: float = 0.025,
) -> StepAccelerationEstimate | None:
    """Fit p(t)=p0+v0*t+0.5*a*t^2 to the steady part of a step."""

    start = float(field_started_time_sec) + max(0.0, float(analysis_delay_sec))
    stop = float("inf") if field_stopped_time_sec is None else float(field_stopped_time_sec)
    selected = sorted(
        (
            sample
            for sample in samples
            if start <= float(sample.measurement_time_sec) <= stop
        ),
        key=lambda sample: sample.measurement_time_sec,
    )
    if len(selected) < int(minimum_samples):
        return None
    times = np.asarray([sample.measurement_time_sec for sample in selected], dtype=float)
    if np.any(np.diff(times) <= 0.0):
        unique_indices = np.concatenate(([True], np.diff(times) > 0.0))
        times = times[unique_indices]
        selected = [sample for sample, keep in zip(selected, unique_indices) if keep]
    if len(selected) < int(minimum_samples):
        return None
    relative = times - times[0]
    span = float(relative[-1])
    if span + 1e-12 < float(minimum_span_sec):
        return None
    positions = np.asarray(
        [[sample.x_mm, sample.y_mm, sample.z_mm] for sample in selected],
        dtype=float,
    )
    if not np.all(np.isfinite(positions)):
        return None
    return StepAccelerationEstimate(
        x=_fit_axis(relative, positions[:, 0]),
        y=_fit_axis(relative, positions[:, 1]),
        z=_fit_axis(relative, positions[:, 2]),
        sample_count=len(selected),
        span_sec=span,
        first_sample_time_sec=float(times[0]),
        last_sample_time_sec=float(times[-1]),
        mean_x_mm=float(np.mean(positions[:, 0])),
        mean_y_mm=float(np.mean(positions[:, 1])),
        mean_z_mm=float(np.mean(positions[:, 2])),
    )


def _robust_sigma(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))


def estimate_position_resolution(
    samples: Iterable[ForceStepSample],
    *,
    sphere_diameter_mm: float,
) -> PositionResolutionEstimate | None:
    """Estimate a conservative position resolution from a stable hold window.

    The observed spread includes real sphere motion as well as camera noise, so
    the result is an upper bound on the measurement noise.  The 95% values are
    for the difference of two individual position samples.
    """

    selected = sorted(samples, key=lambda sample: sample.measurement_time_sec)
    if len(selected) < 3:
        return None
    positions = np.asarray(
        [[sample.x_mm, sample.y_mm, sample.z_mm] for sample in selected],
        dtype=float,
    )
    std = np.std(positions, axis=0, ddof=1)
    robust = np.asarray(
        [_robust_sigma(positions[:, axis]) for axis in range(3)],
        dtype=float,
    )
    # Difference of two independent samples has sqrt(2) times the single-frame
    # standard deviation.  Use the larger conventional/robust estimate.
    effective_sigma = np.maximum(std, robust)
    detectable = 1.96 * math.sqrt(2.0) * effective_sigma
    radius_xy = np.asarray([sample.radius_xy_px for sample in selected], dtype=float)
    radius_z = np.asarray([sample.radius_z_px for sample in selected], dtype=float)
    pixels = np.asarray(
        [
            [sample.u_xy_px, sample.v_xy_px, sample.u_z_px, sample.v_z_px]
            for sample in selected
        ],
        dtype=float,
    )
    pixel_std = np.std(pixels, axis=0, ddof=1)
    median_radius_xy = float(np.median(radius_xy))
    median_radius_z = float(np.median(radius_z))

    def scale(radius_px: float) -> float:
        return (
            float(sphere_diameter_mm) / (2.0 * radius_px)
            if np.isfinite(radius_px) and radius_px > 0.0
            else float("nan")
        )

    return PositionResolutionEstimate(
        sample_count=len(selected),
        span_sec=float(
            selected[-1].measurement_time_sec
            - selected[0].measurement_time_sec
        ),
        std_x_mm=float(std[0]),
        std_y_mm=float(std[1]),
        std_z_mm=float(std[2]),
        robust_sigma_x_mm=float(robust[0]),
        robust_sigma_y_mm=float(robust[1]),
        robust_sigma_z_mm=float(robust[2]),
        single_difference_95_x_mm=float(detectable[0]),
        single_difference_95_y_mm=float(detectable[1]),
        single_difference_95_z_mm=float(detectable[2]),
        std_u_xy_px=float(pixel_std[0]),
        std_v_xy_px=float(pixel_std[1]),
        std_u_z_px=float(pixel_std[2]),
        std_v_z_px=float(pixel_std[3]),
        median_radius_xy_px=median_radius_xy,
        median_radius_z_px=median_radius_z,
        approximate_scale_xy_mm_per_px=scale(median_radius_xy),
        approximate_scale_z_mm_per_px=scale(median_radius_z),
    )


def step_safety_reason(
    *,
    measurement_valid: bool,
    initial_xyz: tuple[float, float, float],
    current_xyz: tuple[float, float, float] | None,
    speed_mm_s: float | None,
    maximum_xy_displacement_mm: float,
    maximum_abs_z_displacement_mm: float,
    maximum_speed_mm_s: float,
    z_min_mm: float,
    z_max_mm: float,
) -> str | None:
    if not measurement_valid or current_xyz is None:
        return "stereo_measurement_lost"
    x, y, z = (float(value) for value in current_xyz)
    dx = x - float(initial_xyz[0])
    dy = y - float(initial_xyz[1])
    dz = z - float(initial_xyz[2])
    if math.hypot(dx, dy) > float(maximum_xy_displacement_mm):
        return "xy_displacement_limit"
    if abs(dz) > float(maximum_abs_z_displacement_mm):
        return "z_displacement_limit"
    if z < float(z_min_mm) or z > float(z_max_mm):
        return "z_workspace_limit"
    if speed_mm_s is not None and float(speed_mm_s) > float(maximum_speed_mm_s):
        return "speed_limit"
    return None


SAMPLE_HEADER = [
    "session_id",
    "trial_id",
    "condition_index",
    "phase",
    "event",
    "measurement_time_sec",
    "elapsed_from_field_ms",
    "sync_skew_ms",
    "u_xy_px",
    "v_xy_px",
    "radius_xy_px",
    "u_z_px",
    "v_z_px",
    "radius_z_px",
    "x_mm",
    "y_mm",
    "z_mm",
    "vx_mm_s",
    "vy_mm_s",
    "vz_mm_s",
    "field_center_x_mm",
    "field_center_y_mm",
    "field_center_z_mm",
    "commanded_intensity",
    "command_sequence",
    "reason",
]


SUMMARY_HEADER = [
    "session_id",
    "trial_id",
    "condition_index",
    "intensity_ratio",
    "modulation_byte",
    "quantized_intensity_ratio",
    "offset_x_mm",
    "offset_y_mm",
    "offset_z_mm",
    "status",
    "stop_reason",
    "pulse_duration_ms",
    "analysis_delay_ms",
    "fit_sample_count",
    "fit_span_ms",
    "ax_mm_s2",
    "ay_mm_s2",
    "az_mm_s2",
    "ax_se_mm_s2",
    "ay_se_mm_s2",
    "az_se_mm_s2",
    "fit_rms_x_mm",
    "fit_rms_y_mm",
    "fit_rms_z_mm",
    "fit_r2_x",
    "fit_r2_y",
    "fit_r2_z",
    "force_x_mN",
    "force_y_mN",
    "acoustic_force_z_mN",
    "force_x_se_mN",
    "force_y_se_mN",
    "acoustic_force_z_se_mN",
    "buoyancy_mN",
    "initial_x_mm",
    "initial_y_mm",
    "initial_z_mm",
    "onset_x_mm",
    "onset_y_mm",
    "onset_z_mm",
    "field_center_x_mm",
    "field_center_y_mm",
    "field_center_z_mm",
    "actual_relative_x_mm",
    "actual_relative_y_mm",
    "actual_relative_z_mm",
    "fit_mean_x_mm",
    "fit_mean_y_mm",
    "fit_mean_z_mm",
    "fit_mean_relative_x_mm",
    "fit_mean_relative_y_mm",
    "fit_mean_relative_z_mm",
    "baseline_sample_count",
    "baseline_span_ms",
    "baseline_std_x_mm",
    "baseline_std_y_mm",
    "baseline_std_z_mm",
    "baseline_robust_sigma_x_mm",
    "baseline_robust_sigma_y_mm",
    "baseline_robust_sigma_z_mm",
    "single_difference_95_x_mm",
    "single_difference_95_y_mm",
    "single_difference_95_z_mm",
    "baseline_std_u_xy_px",
    "baseline_std_v_xy_px",
    "baseline_std_u_z_px",
    "baseline_std_v_z_px",
    "median_radius_xy_px",
    "median_radius_z_px",
    "approximate_scale_xy_mm_per_px",
    "approximate_scale_z_mm_per_px",
    "maximum_xy_displacement_mm",
    "maximum_abs_z_displacement_mm",
    "command_enqueued_time_sec",
    "send_started_time_sec",
    "field_started_time_sec",
    "field_stopped_time_sec",
]


class AcousticForceExperimentLogger:
    def __init__(self, output_directory: str | Path, session_id: str):
        self.output_directory = Path(output_directory)
        self.session_id = str(session_id)
        self.sample_path = self.output_directory / "step_samples.csv"
        self.summary_path = self.output_directory / "trial_summary.csv"
        self.session_path = self.output_directory / "session.json"
        self._sample_file = None
        self._summary_file = None
        self._sample_writer = None
        self._summary_writer = None
        self._rows_since_flush = 0

    def open(self, session_metadata: dict):
        self.output_directory.mkdir(parents=True, exist_ok=False)
        with self.session_path.open("w", encoding="utf-8") as file:
            json.dump(session_metadata, file, ensure_ascii=False, indent=2)
        self._sample_file = self.sample_path.open("w", newline="", encoding="utf-8")
        self._summary_file = self.summary_path.open("w", newline="", encoding="utf-8")
        self._sample_writer = csv.writer(self._sample_file)
        self._summary_writer = csv.DictWriter(
            self._summary_file,
            fieldnames=SUMMARY_HEADER,
        )
        self._sample_writer.writerow(SAMPLE_HEADER)
        self._summary_writer.writeheader()
        self._sample_file.flush()
        self._summary_file.flush()

    def write_sample(self, row: list):
        if self._sample_writer is None:
            return
        self._sample_writer.writerow(row)
        self._rows_since_flush += 1
        if self._rows_since_flush >= 32 or row[4] != "frame":
            self._sample_file.flush()
            self._rows_since_flush = 0

    def write_summary(self, row: dict):
        if self._summary_writer is None:
            return
        normalized = {key: row.get(key, "") for key in SUMMARY_HEADER}
        self._summary_writer.writerow(normalized)
        self._summary_file.flush()

    def close(self):
        for file in (self._sample_file, self._summary_file):
            try:
                if file is not None:
                    file.close()
            except Exception:
                pass
        self._sample_file = None
        self._summary_file = None
        self._sample_writer = None
        self._summary_writer = None


def dataclass_columns(value) -> dict:
    if value is None:
        return {}
    return asdict(value)


def make_session_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
