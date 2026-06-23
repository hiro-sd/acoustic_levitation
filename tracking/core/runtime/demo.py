import math
from dataclasses import dataclass

from ..config import AppConfig
from ..models import HomePosition


@dataclass
class SquareZDemo:
    """Generate an automatic demo trajectory for the STM center."""

    cfg: AppConfig
    origin: HomePosition
    elapsed_sec: float = 0.0

    def reset(self) -> None:
        self.elapsed_sec = 0.0

    def update(self, dt_sec: float) -> HomePosition:
        self.elapsed_sec += max(0.0, float(dt_sec))

        dx, dy = _square_offset_from_center_start(
            t_sec=self.elapsed_sec,
            side_mm=float(self.cfg.demo_square_size_mm),
            speed_mm_s=float(self.cfg.demo_xy_speed_mm_s),
        )
        dz = _sinusoidal_z_offset(
            t_sec=self.elapsed_sec,
            amplitude_mm=float(self.cfg.demo_z_amplitude_mm),
            period_sec=float(self.cfg.demo_z_period_sec),
        )

        z = min(
            float(self.cfg.z_max),
            max(float(self.cfg.z_min), float(self.origin.z + dz)),
        )

        return HomePosition(
            x=float(self.origin.x + dx),
            y=float(self.origin.y + dy),
            z=z,
        )


def _square_offset_from_center_start(
    t_sec: float,
    side_mm: float,
    speed_mm_s: float,
) -> tuple[float, float]:
    """Return XY offset that starts at center, then loops around a square."""
    side = max(0.0, float(side_mm))
    speed = max(1e-6, float(speed_mm_s))
    half = 0.5 * side

    if side <= 0.0:
        return 0.0, 0.0

    # 中心から最初の角まで滑らかに移動してから、四角形の辺上を周回する。
    start_corner = (half, -half)
    center_to_corner = math.hypot(start_corner[0], start_corner[1])
    travel = speed * max(0.0, float(t_sec))

    if travel < center_to_corner:
        ratio = travel / center_to_corner
        return start_corner[0] * ratio, start_corner[1] * ratio

    perimeter_pos = (travel - center_to_corner) % (4.0 * side)

    if perimeter_pos < side:
        return half, -half + perimeter_pos

    perimeter_pos -= side
    if perimeter_pos < side:
        return half - perimeter_pos, half

    perimeter_pos -= side
    if perimeter_pos < side:
        return -half, half - perimeter_pos

    perimeter_pos -= side
    return -half + perimeter_pos, -half


def _sinusoidal_z_offset(
    t_sec: float,
    amplitude_mm: float,
    period_sec: float,
) -> float:
    amplitude = max(0.0, float(amplitude_mm))
    period = max(1e-6, float(period_sec))
    return amplitude * math.sin(2.0 * math.pi * max(0.0, float(t_sec)) / period)
