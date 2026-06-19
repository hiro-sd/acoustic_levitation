from ..config import AppConfig
from ..models import HomePosition


def is_key_pressed(key: str) -> bool:
    # 実機専用依存を遅延importし、制御・解析モジュールを単体で読めるようにする。
    import keyboard

    return bool(keyboard.is_pressed(key))


def update_base_position(home: HomePosition, cfg: AppConfig, dt_sec: float) -> HomePosition:
    """Read movement keys and return the updated control setpoint."""
    if not cfg.enable_base_move:
        return home

    step_xy = cfg.base_move_speed_mm_s * max(0.0, dt_sec)
    step_z = cfg.base_z_move_speed_mm_s * max(0.0, dt_sec)
    x, y, z = float(home.x), float(home.y), float(home.z)

    if is_key_pressed("left"):
        x -= step_xy
    if is_key_pressed("right"):
        x += step_xy
    if is_key_pressed("up"):
        y += step_xy
    if is_key_pressed("down"):
        y -= step_xy
    if is_key_pressed("page up"):
        z += step_z
    if is_key_pressed("page down"):
        z -= step_z

    return HomePosition(x=x, y=y, z=z)
