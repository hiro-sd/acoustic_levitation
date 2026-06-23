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

        dx, dy, edge_index, edge_progress = _rectangle_offset_from_center_start(
            t_sec=self.elapsed_sec,
            x_min_mm=float(self.cfg.demo_x_min_mm),
            x_max_mm=float(self.cfg.demo_x_max_mm),
            y_min_mm=float(self.cfg.demo_y_min_mm),
            y_max_mm=float(self.cfg.demo_y_max_mm),
            speed_mm_s=float(self.cfg.demo_xy_speed_mm_s),
        )
        z = _corner_safe_z_position(
            edge_index=edge_index,
            edge_progress=edge_progress,
            corner_z_mm=float(self.cfg.demo_z_corner_mm),
            z_min_mm=float(self.cfg.demo_z_min_mm),
            z_max_mm=float(self.cfg.demo_z_max_mm),
        )

        z = min(
            float(self.cfg.z_max),
            max(float(self.cfg.z_min), z),
        )

        return HomePosition(
            x=float(self.origin.x + dx),
            y=float(self.origin.y + dy),
            z=z,
        )


def _rectangle_offset_from_center_start(
    t_sec: float,
    x_min_mm: float,
    x_max_mm: float,
    y_min_mm: float,
    y_max_mm: float,
    speed_mm_s: float,
) -> tuple[float, float, int | None, float]:
    """
    Return XY offset that starts at center, then loops around a rectangle.

    edge_index is None while moving from center to the first corner.
    edge_progress is 0.0 at a corner, 0.5 at the middle of an edge,
    and 1.0 at the next corner.
    """
    x_min = min(float(x_min_mm), float(x_max_mm))
    x_max = max(float(x_min_mm), float(x_max_mm))
    y_min = min(float(y_min_mm), float(y_max_mm))
    y_max = max(float(y_min_mm), float(y_max_mm))
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    speed = max(1e-6, float(speed_mm_s))

    if width <= 0.0 or height <= 0.0:
        return 0.0, 0.0, None, 0.0

    # 中心から最初の角まで滑らかに移動してから、指定矩形の辺上を周回する。
    start_corner = (x_min, y_min)
    center_to_corner = math.hypot(start_corner[0], start_corner[1])
    travel = speed * max(0.0, float(t_sec))

    if travel < center_to_corner:
        ratio = travel / center_to_corner
        return start_corner[0] * ratio, start_corner[1] * ratio, None, 0.0

    perimeter_pos = (travel - center_to_corner) % (2.0 * (width + height))

    if perimeter_pos < height:
        return x_min, y_min + perimeter_pos, 0, perimeter_pos / height

    perimeter_pos -= height
    if perimeter_pos < width:
        return x_min + perimeter_pos, y_max, 1, perimeter_pos / width

    perimeter_pos -= width
    if perimeter_pos < height:
        return x_max, y_max - perimeter_pos, 2, perimeter_pos / height

    perimeter_pos -= height
    return x_max - perimeter_pos, y_min, 3, perimeter_pos / width


def _corner_safe_z_position(
    edge_index: int | None,
    edge_progress: float,
    corner_z_mm: float,
    z_min_mm: float,
    z_max_mm: float,
) -> float:
    z_min = min(float(z_min_mm), float(z_max_mm))
    z_max = max(float(z_min_mm), float(z_max_mm))
    corner_z = min(z_max, max(z_min, float(corner_z_mm)))

    # 中心から最初の角へ向かう間と、四隅では安全側の基準高さを保つ。
    if edge_index is None:
        return corner_z

    progress = min(1.0, max(0.0, float(edge_progress)))
    edge_peak = math.sin(math.pi * progress)

    # 各辺の中央でだけ上下端へ近づく。四隅では edge_peak=0 なので corner_z に戻る。
    if edge_index % 2 == 0:
        return corner_z + edge_peak * (z_max - corner_z)

    return corner_z - edge_peak * (corner_z - z_min)
