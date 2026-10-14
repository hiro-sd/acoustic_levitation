from dataclasses import dataclass

import numpy as np

from ..config import AppConfig
from ..models import HomePosition


NORMAL_HOLD = "NORMAL_HOLD"
LOCAL_HOLD = "LOCAL_HOLD"
RETURN_TO_HOME = "RETURN_TO_HOME"


@dataclass
class HoldTelemetry:
    z_drop_mm: float | None = None
    vz_mm_s: float | None = None
    predicted_x_mm: float | None = None
    predicted_y_mm: float | None = None
    predicted_z_mm: float | None = None
    predicted_vz_mm_s: float | None = None
    xy_distance_to_target_mm: float | None = None
    required_force_mN: float | None = None
    commanded_intensity: float | None = None
    actual_intensity: float | None = None
    capture_force_saturated: bool = False
    temporary_home: HomePosition | None = None
    mode_transition_reason: str = ""


def move_towards(current: float, target: float, maximum_step: float) -> float:
    difference = float(target) - float(current)
    if abs(difference) <= float(maximum_step):
        return float(target)
    return float(current + np.sign(difference) * float(maximum_step))


def is_return_home_done(
    cfg: AppConfig,
    home: HomePosition,
    z_mm: float | None,
    x_mm: float | None,
    y_mm: float | None,
    vx: float,
    vy: float,
    vz: float,
) -> bool:
    if x_mm is None or y_mm is None or z_mm is None:
        return False

    error_xy = float(np.hypot(x_mm - home.x, y_mm - home.y))
    error_z = abs(z_mm - home.z)
    speed_xy = float(np.hypot(vx, vy))
    return (
        error_xy <= cfg.return_home_done_error_xy_mm
        and error_z <= cfg.return_home_done_error_z_mm
        and speed_xy <= cfg.return_home_done_speed_xy_mm_s
        and abs(vz) <= cfg.return_home_done_vz_mm_s
    )
