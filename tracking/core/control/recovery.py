import numpy as np

from ..config import AppConfig
from ..models import HomePosition


def move_towards(current: float, target: float, max_step: float) -> float:
    diff = target - current
    if abs(diff) <= max_step:
        return float(target)
    return float(current + np.sign(diff) * max_step)


def should_enter_fall_recovery(
    cfg: AppConfig,
    home_z: float,
    z_mm: float | None,
    vz: float,
) -> bool:
    if z_mm is None:
        return False

    return (
        home_z - z_mm >= cfg.fall_drop_threshold_mm
        and vz <= cfg.fall_vz_threshold_mm_s
    )


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

    err_xy = float(np.hypot(x_mm - home.x, y_mm - home.y))
    err_z = abs(z_mm - home.z)
    speed_xy = float(np.hypot(vx, vy))

    return (
        err_xy <= cfg.return_home_done_error_xy_mm
        and err_z <= cfg.return_home_done_error_z_mm
        and speed_xy <= cfg.return_home_done_speed_xy_mm_s
        and abs(vz) <= cfg.return_home_done_vz_mm_s
    )
