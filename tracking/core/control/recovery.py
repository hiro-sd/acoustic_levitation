import numpy as np
from dataclasses import dataclass

from ..config import AppConfig
from ..models import HomePosition


NORMAL_HOLD = "NORMAL_HOLD"
FOLLOW_AND_BRAKE = "FOLLOW_AND_BRAKE"
LOCAL_HOLD = "LOCAL_HOLD"
RETURN_TO_HOME = "RETURN_TO_HOME"


@dataclass(frozen=True)
class CaptureForceCommand:
    required_force_mN: float
    commanded_intensity: float
    saturated: bool


@dataclass
class RecoveryTelemetry:
    z_drop_mm: float | None = None
    vz_mm_s: float | None = None
    predicted_z_mm: float | None = None
    predicted_vz_mm_s: float | None = None
    required_force_mN: float | None = None
    commanded_intensity: float | None = None
    capture_force_saturated: bool = False
    temporary_home: HomePosition | None = None
    mode_transition_reason: str = ""


def move_towards(current: float, target: float, max_step: float) -> float:
    diff = target - current
    if abs(diff) <= max_step:
        return float(target)
    return float(current + np.sign(diff) * max_step)


def predict_falling_position(
    cfg: AppConfig,
    x_mm: float | None,
    y_mm: float | None,
    z_mm: float | None,
    vx_mm_s: float,
    vy_mm_s: float,
    vz_mm_s: float,
) -> tuple[float | None, float | None, float | None, float]:
    """
    システム遅延後の球位置・速度を予測する。

    z軸は上向き正なので、重力は -g。
    """
    dt = float(cfg.fall_system_delay_sec)
    predicted_x = None if x_mm is None else float(x_mm + vx_mm_s * dt)
    predicted_y = None if y_mm is None else float(y_mm + vy_mm_s * dt)
    predicted_z = None if z_mm is None else float(
        z_mm
        + vz_mm_s * dt
        - 0.5 * cfg.gravity_mm_s2 * (dt ** 2)
        + cfg.fall_recovery_z_offset_mm
    )
    predicted_vz = float(vz_mm_s - cfg.gravity_mm_s2 * dt)
    return predicted_x, predicted_y, predicted_z, predicted_vz


def required_capture_force_mN(
    cfg: AppConfig,
    predicted_vz_mm_s: float,
) -> float:
    """
    予測下向き速度 u と捕捉時間 T_capture から必要上向き力を計算する。

    F_req = m * (g + u / T_capture)

    u は下向き速度の大きさ [m/s]。vzは上向き正なので u=max(0,-vz)。
    戻り値は安全率込み [mN]。
    """
    u_m_s = max(0.0, -float(predicted_vz_mm_s) / 1000.0)
    t_capture = max(1e-6, float(cfg.fall_capture_time_sec))
    force_n = float(cfg.ball_mass_kg) * (9.80665 + u_m_s / t_capture)
    return 1000.0 * force_n * float(cfg.fall_capture_force_safety_factor)


def measured_force_for_intensity_mN(cfg: AppConfig, intensity: float) -> float:
    model = sorted((float(i), float(f)) for i, f in cfg.fall_capture_loadcell_model)
    xs = np.array([p[0] for p in model], dtype=float)
    ys = np.array([p[1] for p in model], dtype=float)
    return float(np.interp(float(intensity), xs, ys))


def capture_intensity_from_force(
    cfg: AppConfig,
    required_force_mN: float,
) -> CaptureForceCommand:
    """
    ロードセルモデルに基づき、まずは 0.8/0.9/1.0 の段階制御で強度を選ぶ。
    """
    levels = tuple(float(v) for v in cfg.fall_capture_intensity_levels)
    max_model_force = max(float(force) for _, force in cfg.fall_capture_loadcell_model)
    saturated = float(required_force_mN) > max_model_force

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


def apply_capture_intensity_slew(
    current: float,
    target: float,
    cfg: AppConfig,
    dt_sec: float,
) -> float:
    """
    捕捉用intensityは上昇即時、下降のみslew-rate制限で緩やかにする。
    """
    current = float(current)
    target = float(np.clip(target, 0.0, cfg.intensity_max_ratio))
    if target >= current:
        return target
    max_drop = max(0.0, float(cfg.fall_intensity_down_slew_per_sec) * float(dt_sec))
    return float(max(target, current - max_drop))


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


def fall_detection_reason(
    cfg: AppConfig,
    home: HomePosition,
    x_mm: float | None,
    y_mm: float | None,
    z_mm: float | None,
    vz_mm_s: float,
    descending_frames: int,
) -> str | None:
    """
    FOLLOW_AND_BRAKEへ入る条件。

    - 下降速度
    - 下降の連続フレーム数
    - 通常保持領域からの逸脱
    """
    if z_mm is None:
        return None

    z_drop = float(home.z - z_mm)
    xy_err = None
    if x_mm is not None and y_mm is not None:
        xy_err = float(np.hypot(x_mm - home.x, y_mm - home.y))

    fast_down = float(vz_mm_s) <= float(cfg.fall_vz_threshold_mm_s)
    enough_frames = int(descending_frames) >= int(cfg.fall_descending_frames)
    outside_z = z_drop >= float(cfg.fall_hold_region_z_mm)
    outside_xy = xy_err is not None and xy_err >= float(cfg.fall_hold_region_xy_mm)
    outside_hold_region = outside_z or outside_xy

    if fast_down and enough_frames and outside_hold_region:
        parts = [
            f"vz={vz_mm_s:.1f}",
            f"descending_frames={descending_frames}",
            f"z_drop={z_drop:.1f}",
        ]
        if xy_err is not None:
            parts.append(f"xy_err={xy_err:.1f}")
        return "fall_detected:" + ",".join(parts)
    return None


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
