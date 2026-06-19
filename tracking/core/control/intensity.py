from ..config import AppConfig


def compute_z_intensity_boost_ratio(
    cfg: AppConfig,
    home_z: float,
    z_mm: float | None,
) -> float:
    """Z方向の落下量から目標強度を線形補間する。"""
    base = float(cfg.static_intensity_ratio)
    max_ratio = float(cfg.intensity_max_ratio)

    if z_mm is None:
        return base

    drop_mm = float(home_z - z_mm)
    if drop_mm <= cfg.z_boost_release_mm:
        return base
    if drop_mm >= cfg.z_boost_full_drop_mm:
        return max_ratio

    denom = cfg.z_boost_full_drop_mm - cfg.z_boost_release_mm
    if denom <= 1e-6:
        return max_ratio

    blend = (drop_mm - cfg.z_boost_release_mm) / denom
    return base + blend * (max_ratio - base)
