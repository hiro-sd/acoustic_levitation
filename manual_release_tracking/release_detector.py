from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FingerDistanceResult:
    """2D image-space relation between the detected ball and nearby finger candidates."""

    distance_px: float | None
    nearest_ball_point: tuple[int, int] | None
    nearest_finger_point: tuple[int, int] | None
    contours: list[np.ndarray]
    skin_mask_roi: np.ndarray
    roi: tuple[int, int, int, int]

    @property
    def has_finger_candidate(self) -> bool:
        return self.distance_px is not None


class ReleaseState:
    WAITING_FOR_GRASP = "WAITING_FOR_GRASP"
    GRASPED = "GRASPED"
    RELEASED = "RELEASED"


@dataclass
class ReleaseDetectorConfig:
    roi_scale: float = 4.0
    roi_margin_px: int = 40
    min_finger_area_px: int = 80
    grasp_distance_px: float = 12.0
    release_distance_px: float = 28.0
    grasp_confirm_frames: int = 3
    release_confirm_frames: int = 2


class ReleaseStateMachine:
    """
    Conservative preview-only state machine.

    It does not control AUTD. It only answers:
    - is a finger-like contour close enough to the ball to call it grasped?
    - did that contour separate far enough to call it released?
    """

    def __init__(self, cfg: ReleaseDetectorConfig | None = None):
        self.cfg = cfg or ReleaseDetectorConfig()
        self.state = ReleaseState.WAITING_FOR_GRASP
        self.grasp_count = 0
        self.release_count = 0
        self.transition_reason = ""

    def reset(self):
        self.state = ReleaseState.WAITING_FOR_GRASP
        self.grasp_count = 0
        self.release_count = 0
        self.transition_reason = "manual_reset"

    def update(self, distance_px: float | None, ball_detected: bool) -> str:
        self.transition_reason = ""

        if not ball_detected:
            self.grasp_count = 0
            self.release_count = 0
            if self.state != ReleaseState.WAITING_FOR_GRASP:
                self.state = ReleaseState.WAITING_FOR_GRASP
                self.transition_reason = "ball_lost"
            return self.state

        if self.state == ReleaseState.WAITING_FOR_GRASP:
            if distance_px is not None and distance_px <= self.cfg.grasp_distance_px:
                self.grasp_count += 1
            else:
                self.grasp_count = 0

            if self.grasp_count >= self.cfg.grasp_confirm_frames:
                self.state = ReleaseState.GRASPED
                self.release_count = 0
                self.transition_reason = "finger_close_to_ball"

        elif self.state == ReleaseState.GRASPED:
            if distance_px is None or distance_px >= self.cfg.release_distance_px:
                self.release_count += 1
            else:
                self.release_count = 0

            if self.release_count >= self.cfg.release_confirm_frames:
                self.state = ReleaseState.RELEASED
                self.transition_reason = "finger_separated_from_ball"

        return self.state


def _clamp_rect(x1: int, y1: int, x2: int, y2: int, width: int, height: int):
    return (
        max(0, min(width - 1, int(x1))),
        max(0, min(height - 1, int(y1))),
        max(1, min(width, int(x2))),
        max(1, min(height, int(y2))),
    )


def _skin_mask_rgb(roi_rgb: np.ndarray) -> np.ndarray:
    """
    Lightweight skin-color candidate mask for preview.

    This is intentionally simple and tunable. It is not meant to be a perfect hand detector;
    it just finds finger-like regions near the ball for release timing experiments.
    """

    ycrcb = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2YCrCb)
    lower_ycrcb = np.array([30, 125, 70], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 140], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    lower_hsv = np.array([0, 20, 40], dtype=np.uint8)
    upper_hsv = np.array([25, 220, 255], dtype=np.uint8)
    mask_hsv_1 = cv2.inRange(hsv, lower_hsv, upper_hsv)
    lower_hsv_red = np.array([160, 20, 40], dtype=np.uint8)
    upper_hsv_red = np.array([179, 220, 255], dtype=np.uint8)
    mask_hsv_2 = cv2.inRange(hsv, lower_hsv_red, upper_hsv_red)

    mask = cv2.bitwise_and(mask_ycrcb, cv2.bitwise_or(mask_hsv_1, mask_hsv_2))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def detect_finger_distance(
    frame_rgb: np.ndarray,
    ball_center: tuple[float, float] | None,
    ball_radius_px: float | None,
    cfg: ReleaseDetectorConfig | None = None,
) -> FingerDistanceResult:
    cfg = cfg or ReleaseDetectorConfig()
    height, width = frame_rgb.shape[:2]

    if ball_center is None or ball_radius_px is None:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return FingerDistanceResult(None, None, None, [], empty, (0, 0, 1, 1))

    cx, cy = float(ball_center[0]), float(ball_center[1])
    radius = max(1.0, float(ball_radius_px))
    half = int(radius * cfg.roi_scale + cfg.roi_margin_px)
    x1, y1, x2, y2 = _clamp_rect(
        int(cx - half),
        int(cy - half),
        int(cx + half),
        int(cy + half),
        width,
        height,
    )

    roi_rgb = frame_rgb[y1:y2, x1:x2]
    if roi_rgb.size == 0:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return FingerDistanceResult(None, None, None, [], empty, (x1, y1, x2, y2))

    mask = _skin_mask_rgb(roi_rgb)

    # Remove most of the ball interior. If fingers overlap the ball boundary, the outer parts remain.
    cx_roi = int(round(cx - x1))
    cy_roi = int(round(cy - y1))
    ball_inner_radius = int(round(radius * 0.85))
    cv2.circle(mask, (cx_roi, cy_roi), ball_inner_radius, 0, thickness=-1)

    contours_roi, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_full: list[np.ndarray] = []
    best_distance = None
    best_ball_point = None
    best_finger_point = None

    for contour_roi in contours_roi:
        area = cv2.contourArea(contour_roi)
        if area < cfg.min_finger_area_px:
            continue

        contour_full = contour_roi + np.array([[[x1, y1]]], dtype=contour_roi.dtype)
        contours_full.append(contour_full)

        pts = contour_full.reshape(-1, 2).astype(np.float32)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        dist_to_center = np.sqrt(dx * dx + dy * dy)
        outside_distance = np.maximum(0.0, dist_to_center - radius)
        idx = int(np.argmin(outside_distance))
        d = float(outside_distance[idx])

        if best_distance is None or d < best_distance:
            best_distance = d
            fp = pts[idx]
            norm = max(1e-6, float(dist_to_center[idx]))
            bp = np.array([cx, cy], dtype=np.float32) + (fp - np.array([cx, cy])) * (
                radius / norm
            )
            best_finger_point = (int(round(float(fp[0]))), int(round(float(fp[1]))))
            best_ball_point = (int(round(float(bp[0]))), int(round(float(bp[1]))))

    return FingerDistanceResult(
        distance_px=best_distance,
        nearest_ball_point=best_ball_point,
        nearest_finger_point=best_finger_point,
        contours=contours_full,
        skin_mask_roi=mask,
        roi=(x1, y1, x2, y2),
    )


def draw_release_debug(
    frame_bgr: np.ndarray,
    result: FingerDistanceResult,
    state: str,
):
    x1, y1, x2, y2 = result.roi
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 0, 255), 1)

    for contour in result.contours:
        cv2.drawContours(frame_bgr, [contour], -1, (255, 0, 255), 2)

    if result.nearest_ball_point is not None and result.nearest_finger_point is not None:
        cv2.line(
            frame_bgr,
            result.nearest_ball_point,
            result.nearest_finger_point,
            (0, 165, 255),
            2,
        )
        cv2.circle(frame_bgr, result.nearest_ball_point, 4, (0, 255, 255), -1)
        cv2.circle(frame_bgr, result.nearest_finger_point, 4, (0, 165, 255), -1)

    distance_label = (
        "finger dist: --"
        if result.distance_px is None
        else f"finger dist: {result.distance_px:.1f}px"
    )
    cv2.putText(
        frame_bgr,
        f"{state} | {distance_label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
