from dataclasses import dataclass

import cv2
import numpy as np

from .config import AppConfig
from .vision import clamp_roi, track_ball_cv


@dataclass(frozen=True)
class BallDetection:
    roi: tuple[int, int, int, int]
    center: tuple[float, float] | None = None
    radius_px: float | None = None

    @property
    def detected(self) -> bool:
        return self.center is not None and self.radius_px is not None


class RoiBallTracker:
    """Owns the adaptive ROI state for one camera."""

    def __init__(self, width: int, height: int, cfg: AppConfig):
        self.width = int(width)
        self.height = int(height)
        self.cfg = cfg
        self.roi_cx = self.width // 2
        self.roi_cy = self.height // 2
        self.roi_size = int(cfg.roi_init_size)

    def detect(self, frame: np.ndarray) -> BallDetection:
        roi = clamp_roi(
            self.roi_cx,
            self.roi_cy,
            self.roi_size,
            self.width,
            self.height,
            self.cfg,
        )
        detection, _ = track_ball_cv(frame, roi, self.cfg)

        if detection is None:
            self.roi_size = int(
                min(self.cfg.roi_max_size, self.roi_size * self.cfg.roi_expand_on_lost)
            )
            return BallDetection(roi=roi)

        u, v, radius = detection
        self.roi_cx, self.roi_cy = int(u), int(v)
        desired_size = int(2 * (2.5 * radius + self.cfg.roi_margin))
        self.roi_size = int(0.7 * self.roi_size + 0.3 * desired_size)
        self.roi_size = int(
            np.clip(self.roi_size, self.cfg.roi_min_size, self.cfg.roi_max_size)
        )
        return BallDetection(roi=roi, center=(u, v), radius_px=radius)


def draw_ball_detection(
    frame: np.ndarray,
    detection: BallDetection,
    tracking_active: bool,
):
    x1, y1, x2, y2 = detection.roi
    if detection.detected:
        u, v = detection.center
        color = (0, 255, 0) if tracking_active else (200, 200, 200)
        cv2.circle(frame, (int(u), int(v)), int(max(2, detection.radius_px)), color, 2)
        roi_color = (255, 255, 0)
    else:
        roi_color = (0, 255, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)
