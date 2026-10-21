from dataclasses import dataclass
import math

import numpy as np

from ..config import AppConfig


# Weighted zero-offset fit from camera step-response session
# 20261020_123846_25876 (300 attempts, 239 valid acceleration fits).
#
# The session was initially analyzed with a nominal mass of 0.50 g, giving
#   F_0.50(A) = 9.381487 A - 0.986042 mN.
# The sphere used in that same session was subsequently measured as 0.53 g.
# With the recorded buoyancy F_b=0.339232 mN, every fitted force is corrected
# without refitting the trajectories by
#   F_0.53 = (0.53 / 0.50) * (F_0.50 + F_b) - F_b.
# Therefore the mass-corrected model used by the capture controller is
#   F_acoustic(A) = 9.94437622 A - 1.02485060 mN.
SI2026_CAPTURE_FORCE_MODEL: tuple[tuple[float, float], ...] = (
    (0.5, 3.947338),
    (0.6, 4.941775),
    (0.7, 5.936213),
    (0.8, 6.930650),
    (0.9, 7.925088),
    (1.0, 8.919526),
)


@dataclass(frozen=True)
class CaptureForceCommand:
    required_force_mN: float
    commanded_intensity: float
    saturated: bool


def capture_buoyancy_mN(cfg: AppConfig) -> float:
    diameter_m = float(cfg.ball_diameter_mm) * 1e-3
    volume_m3 = math.pi * diameter_m**3 / 6.0
    return (
        float(cfg.air_density_kg_m3)
        * volume_m3
        * 9.80665
        * 1000.0
    )


def required_capture_force_mN(
    cfg: AppConfig,
    predicted_vz_mm_s: float,
) -> float:
    """Return the upward force required to stop the current downward speed."""
    u_m_s = max(0.0, -float(predicted_vz_mm_s) / 1000.0)
    t_capture = max(1e-6, float(cfg.capture_time_sec))
    force_n = float(cfg.ball_mass_kg) * (9.80665 + u_m_s / t_capture)
    acoustic_force_mn = 1000.0 * force_n - capture_buoyancy_mN(cfg)
    return max(
        0.0,
        acoustic_force_mn * float(cfg.capture_force_safety_factor),
    )


def measured_force_for_intensity_mN(cfg: AppConfig, intensity: float) -> float:
    model = sorted(
        (float(i), float(f)) for i, f in cfg.capture_loadcell_model
    )
    xs = np.array([point[0] for point in model], dtype=float)
    ys = np.array([point[1] for point in model], dtype=float)
    return float(np.interp(float(intensity), xs, ys))


def capture_intensity_from_force(
    cfg: AppConfig,
    required_force_mN: float,
) -> CaptureForceCommand:
    """Select the smallest configured intensity that supplies the force."""
    maximum_intensity = float(cfg.capture_max_intensity_ratio)
    levels = tuple(
        float(value)
        for value in cfg.capture_intensity_levels
        if float(value) <= maximum_intensity
    )
    if not levels:
        levels = (maximum_intensity,)

    maximum_force = measured_force_for_intensity_mN(cfg, max(levels))
    saturated = float(required_force_mN) > maximum_force
    for level in levels:
        if measured_force_for_intensity_mN(cfg, level) >= required_force_mN:
            return CaptureForceCommand(
                required_force_mN=float(required_force_mN),
                commanded_intensity=level,
                saturated=saturated,
            )

    return CaptureForceCommand(
        required_force_mN=float(required_force_mN),
        commanded_intensity=max(levels),
        saturated=True,
    )


def capture_intensity_interpolated_from_force(
    cfg: AppConfig,
    required_force_mN: float,
) -> CaptureForceCommand:
    """Invert the calibrated force model with piecewise-linear interpolation."""
    maximum_intensity = float(cfg.capture_max_intensity_ratio)
    model = sorted(
        (
            (float(intensity), float(force_mn))
            for intensity, force_mn in cfg.capture_loadcell_model
            if float(intensity) <= maximum_intensity + 1e-9
        ),
        key=lambda point: point[1],
    )
    if not model:
        model = [(maximum_intensity, 0.0)]

    forces = np.asarray([point[1] for point in model], dtype=float)
    intensities = np.asarray([point[0] for point in model], dtype=float)
    required = float(required_force_mN)
    commanded = float(np.interp(required, forces, intensities))
    saturated = required > float(forces[-1]) + 1e-9
    return CaptureForceCommand(
        required_force_mN=required,
        commanded_intensity=float(
            np.clip(commanded, intensities[0], maximum_intensity)
        ),
        saturated=bool(saturated),
    )


def capture_force_command_for_velocity(
    cfg: AppConfig,
    predicted_vz_mm_s: float,
    *,
    current_intensity: float | None = None,
    downward_hysteresis_mN: float = 0.0,
    interpolate_intensity: bool = False,
    minimum_intensity: float | None = None,
    intensity_deadband: float = 0.0,
) -> CaptureForceCommand:
    """Convert downward velocity to a calibrated acoustic intensity."""
    required_force = required_capture_force_mN(cfg, predicted_vz_mm_s)
    requested = (
        capture_intensity_interpolated_from_force(cfg, required_force)
        if interpolate_intensity
        else capture_intensity_from_force(cfg, required_force)
    )
    commanded = float(requested.commanded_intensity)

    if minimum_intensity is not None:
        commanded = max(commanded, float(minimum_intensity))

    if interpolate_intensity:
        if (
            current_intensity is not None
            and abs(commanded - float(current_intensity))
            < max(0.0, float(intensity_deadband))
        ):
            commanded = float(current_intensity)
    elif current_intensity is not None and commanded < float(current_intensity):
        lower_level_force = measured_force_for_intensity_mN(cfg, commanded)
        decrease_boundary = lower_level_force - max(
            0.0,
            float(downward_hysteresis_mN),
        )
        if required_force > decrease_boundary:
            commanded = float(current_intensity)

    return CaptureForceCommand(
        required_force_mN=float(required_force),
        commanded_intensity=float(commanded),
        saturated=bool(requested.saturated),
    )


def apply_capture_intensity_slew(
    current: float,
    target: float,
    cfg: AppConfig,
    dt_sec: float,
) -> float:
    """Apply increases immediately and limit only intensity decreases."""
    current = float(current)
    target = float(np.clip(target, 0.0, cfg.intensity_max_ratio))
    if target >= current:
        return target
    maximum_drop = max(
        0.0,
        float(cfg.capture_intensity_down_slew_per_sec) * float(dt_sec),
    )
    return float(max(target, current - maximum_drop))
