from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


_LOWER_YCRCB = np.array([30, 125, 70], dtype=np.uint8)
_UPPER_YCRCB = np.array([255, 180, 140], dtype=np.uint8)
_LOWER_HSV_1 = np.array([0, 20, 40], dtype=np.uint8)
_UPPER_HSV_1 = np.array([25, 220, 255], dtype=np.uint8)
_LOWER_HSV_2 = np.array([160, 20, 40], dtype=np.uint8)
_UPPER_HSV_2 = np.array([179, 220, 255], dtype=np.uint8)
_MORPH_KERNEL = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class FingerDistanceResult:
    """2D image-space relation between the detected ball and nearby finger candidates."""

    distance_px: float | None
    finger_area_px: float
    contact_area_px: float
    contact_ratio: float
    near_area_normalized: float
    normalized_gap: float | None
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
    # Only the neighborhood that can physically touch or just leave the sphere is used.
    # Geometric features are normalized by the detected sphere radius.
    roi_scale: float = 2.05
    roi_margin_px: int = 6
    min_finger_area_px: int = 20
    ball_remove_radius_scale: float = 1.02
    contact_outer_radius_scale: float = 1.45
    near_outer_radius_scale: float = 2.00

    # Initial grasp evidence. Strong contact can arm while the held sphere is moving;
    # otherwise a stable 3D position is also required.
    grasp_contact_ratio_min: float = 0.025
    strong_grasp_contact_ratio: float = 0.060
    grasp_near_area_normalized_min: float = 0.050
    grasp_gap_normalized_max: float = 0.30
    grasp_window_frames: int = 7
    grasp_required_votes: int = 5
    position_stability_frames: int = 8
    position_stable_std_mm: float = 2.5

    # Release is judged relative to the automatically learned GRASPED baseline.
    release_contact_baseline_ratio: float = 0.35
    release_gap_delta_normalized: float = 0.12
    release_near_area_normalized_max: float = 0.020
    release_window_frames: int = 5
    release_required_votes: int = 4
    baseline_update_alpha: float = 0.02
    ball_lost_grace_frames: int = 5


@dataclass(frozen=True)
class ReleaseDecisionDebug:
    grasp_candidate: bool = False
    release_candidate: bool = False
    baseline_contact_ratio: float | None = None
    baseline_gap_normalized: float | None = None
    contact_to_baseline: float | None = None
    gap_delta_normalized: float | None = None
    grasp_votes: int = 0
    release_votes: int = 0


class ReleaseStateMachine:
    """
    Adaptive preview-only state machine.

    It does not control AUTD. It only answers:
    - is there stable skin contact in a narrow ring around the sphere?
    - did that contact decrease and move away relative to the held baseline?
    """

    def __init__(self, cfg: ReleaseDetectorConfig | None = None):
        self.cfg = cfg or ReleaseDetectorConfig()
        self.state = ReleaseState.WAITING_FOR_GRASP
        self._grasp_votes = deque(maxlen=self.cfg.grasp_window_frames)
        self._release_votes = deque(maxlen=self.cfg.release_window_frames)
        self._grasp_contact_samples: deque[float | None] = deque(
            maxlen=self.cfg.grasp_window_frames
        )
        self._grasp_gap_samples: deque[float | None] = deque(
            maxlen=self.cfg.grasp_window_frames
        )
        self._ball_lost_count = 0
        self.baseline_contact_ratio: float | None = None
        self.baseline_gap_normalized: float | None = None
        self.debug = ReleaseDecisionDebug()
        self.transition_reason = ""

    def reset(self):
        self.state = ReleaseState.WAITING_FOR_GRASP
        self._grasp_votes.clear()
        self._release_votes.clear()
        self._grasp_contact_samples.clear()
        self._grasp_gap_samples.clear()
        self._ball_lost_count = 0
        self.baseline_contact_ratio = None
        self.baseline_gap_normalized = None
        self.debug = ReleaseDecisionDebug()
        self.transition_reason = "manual_reset"

    def update(
        self,
        result: FingerDistanceResult | None,
        ball_detected: bool,
        position_stable: bool = False,
    ) -> str:
        self.transition_reason = ""

        if not ball_detected or result is None:
            self._ball_lost_count += 1
            # Pinching can briefly hide the sphere. Freeze the decision during a short
            # dropout instead of interpreting missing data as an immediate release.
            if self._ball_lost_count <= self.cfg.ball_lost_grace_frames:
                self.debug = ReleaseDecisionDebug(
                    baseline_contact_ratio=self.baseline_contact_ratio,
                    baseline_gap_normalized=self.baseline_gap_normalized,
                    grasp_votes=sum(self._grasp_votes),
                    release_votes=sum(self._release_votes),
                )
                return self.state

            previous_state = self.state
            self.reset()
            if previous_state != ReleaseState.WAITING_FOR_GRASP:
                self.transition_reason = "ball_lost_timeout"
            return self.state

        self._ball_lost_count = 0
        gap = result.normalized_gap
        has_near_skin = (
            result.near_area_normalized
            >= self.cfg.grasp_near_area_normalized_min
        )
        grasp_contact = result.contact_ratio >= self.cfg.grasp_contact_ratio_min
        gap_is_close = gap is not None and gap <= self.cfg.grasp_gap_normalized_max
        strong_contact = result.contact_ratio >= self.cfg.strong_grasp_contact_ratio
        grasp_candidate = (
            grasp_contact
            and has_near_skin
            and gap_is_close
            and (position_stable or strong_contact)
        )

        release_candidate = False
        contact_to_baseline = None
        gap_delta = None

        if self.state == ReleaseState.WAITING_FOR_GRASP:
            self._grasp_votes.append(bool(grasp_candidate))
            self._grasp_contact_samples.append(
                float(result.contact_ratio) if grasp_candidate else None
            )
            self._grasp_gap_samples.append(
                float(gap) if grasp_candidate and gap is not None else None
            )
            contact_samples = [
                value
                for value in self._grasp_contact_samples
                if value is not None
            ]
            gap_samples = [
                value
                for value in self._grasp_gap_samples
                if value is not None
            ]

            if (
                len(self._grasp_votes) >= self.cfg.grasp_required_votes
                and sum(self._grasp_votes) >= self.cfg.grasp_required_votes
                and contact_samples
            ):
                self.state = ReleaseState.GRASPED
                self._release_votes.clear()
                self.baseline_contact_ratio = float(np.median(contact_samples))
                self.baseline_gap_normalized = (
                    float(np.median(gap_samples)) if gap_samples else 0.0
                )
                self.transition_reason = "stable_contact_ring"

        elif self.state == ReleaseState.GRASPED:
            baseline_contact = max(
                1e-6,
                float(self.baseline_contact_ratio or result.contact_ratio or 1e-6),
            )
            baseline_gap = float(self.baseline_gap_normalized or 0.0)
            contact_to_baseline = float(result.contact_ratio / baseline_contact)
            gap_delta = None if gap is None else float(gap - baseline_gap)

            contact_dropped = (
                contact_to_baseline <= self.cfg.release_contact_baseline_ratio
            )
            gap_increased = (
                gap_delta is not None
                and gap_delta >= self.cfg.release_gap_delta_normalized
            )
            near_skin_gone = (
                result.near_area_normalized
                <= self.cfg.release_near_area_normalized_max
            )
            release_candidate = contact_dropped and (gap_increased or near_skin_gone)
            self._release_votes.append(bool(release_candidate))

            if (
                len(self._release_votes) >= self.cfg.release_required_votes
                and sum(self._release_votes) >= self.cfg.release_required_votes
            ):
                self.state = ReleaseState.RELEASED
                self.transition_reason = "contact_ring_separated"
            elif (
                not release_candidate
                and contact_to_baseline >= 0.70
                and gap is not None
                and abs(gap_delta or 0.0)
                < self.cfg.release_gap_delta_normalized * 0.5
            ):
                # Follow slow lighting/grip changes, but not rapid release motion.
                alpha = self.cfg.baseline_update_alpha
                self.baseline_contact_ratio = (
                    (1.0 - alpha) * baseline_contact
                    + alpha * float(result.contact_ratio)
                )
                self.baseline_gap_normalized = (
                    (1.0 - alpha) * baseline_gap + alpha * float(gap)
                )

        self.debug = ReleaseDecisionDebug(
            grasp_candidate=grasp_candidate,
            release_candidate=release_candidate,
            baseline_contact_ratio=self.baseline_contact_ratio,
            baseline_gap_normalized=self.baseline_gap_normalized,
            contact_to_baseline=contact_to_baseline,
            gap_delta_normalized=gap_delta,
            grasp_votes=sum(self._grasp_votes),
            release_votes=sum(self._release_votes),
        )
        return self.state


def _clamp_rect(x1: int, y1: int, x2: int, y2: int, width: int, height: int):
    return (
        max(0, min(width - 1, int(x1))),
        max(0, min(height - 1, int(y1))),
        max(1, min(width, int(x2))),
        max(1, min(height, int(y2))),
    )


def _skin_mask_bgr(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Lightweight skin-color candidate mask for preview.

    This is intentionally simple and tunable. It is not meant to be a perfect hand detector;
    it just finds finger-like regions near the ball for release timing experiments.
    """

    # XIMEA frames follow the same BGR display convention used by tracking/core/app.py.
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    mask_ycrcb = cv2.inRange(ycrcb, _LOWER_YCRCB, _UPPER_YCRCB)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask_hsv_1 = cv2.inRange(hsv, _LOWER_HSV_1, _UPPER_HSV_1)
    mask_hsv_2 = cv2.inRange(hsv, _LOWER_HSV_2, _UPPER_HSV_2)

    mask = cv2.bitwise_and(mask_ycrcb, cv2.bitwise_or(mask_hsv_1, mask_hsv_2))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=1)
    return mask


def detect_finger_distance(
    frame_bgr: np.ndarray,
    ball_center: tuple[float, float] | None,
    ball_radius_px: float | None,
    cfg: ReleaseDetectorConfig | None = None,
) -> FingerDistanceResult:
    cfg = cfg or ReleaseDetectorConfig()
    height, width = frame_bgr.shape[:2]

    if ball_center is None or ball_radius_px is None:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return FingerDistanceResult(
            distance_px=None,
            finger_area_px=0.0,
            contact_area_px=0.0,
            contact_ratio=0.0,
            near_area_normalized=0.0,
            normalized_gap=None,
            nearest_ball_point=None,
            nearest_finger_point=None,
            contours=[],
            skin_mask_roi=empty,
            roi=(0, 0, 1, 1),
        )

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

    roi_bgr = frame_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return FingerDistanceResult(
            distance_px=None,
            finger_area_px=0.0,
            contact_area_px=0.0,
            contact_ratio=0.0,
            near_area_normalized=0.0,
            normalized_gap=None,
            nearest_ball_point=None,
            nearest_finger_point=None,
            contours=[],
            skin_mask_roi=empty,
            roi=(x1, y1, x2, y2),
        )

    mask = _skin_mask_bgr(roi_bgr)

    # Restrict the skin mask to two narrow, radius-normalized rings around the sphere.
    # This rejects palms/arms elsewhere in the ROI and makes the thresholds independent
    # of camera magnification.
    cx_roi = int(round(cx - x1))
    cy_roi = int(round(cy - y1))
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    radial_sq = (xx - cx_roi) ** 2 + (yy - cy_roi) ** 2
    inner_radius = radius * cfg.ball_remove_radius_scale
    contact_outer_radius = radius * cfg.contact_outer_radius_scale
    near_outer_radius = radius * cfg.near_outer_radius_scale
    analysis_ring = (radial_sq > inner_radius**2) & (
        radial_sq <= near_outer_radius**2
    )
    contact_ring = (radial_sq > inner_radius**2) & (
        radial_sq <= contact_outer_radius**2
    )
    mask = cv2.bitwise_and(mask, mask, mask=analysis_ring.astype(np.uint8) * 255)

    # Remove isolated color noise before calculating contact features.
    contours_roi, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    filtered_mask = np.zeros_like(mask)
    contours_full: list[np.ndarray] = []

    for contour_roi in contours_roi:
        area = cv2.contourArea(contour_roi)
        if area < cfg.min_finger_area_px:
            continue

        cv2.drawContours(filtered_mask, [contour_roi], -1, 255, thickness=-1)
        contour_full = contour_roi + np.array([[[x1, y1]]], dtype=contour_roi.dtype)
        contours_full.append(contour_full)

    skin_pixels = filtered_mask > 0
    contact_area = float(np.count_nonzero(skin_pixels & contact_ring))
    near_area = float(np.count_nonzero(skin_pixels))
    contact_ring_area = max(1, int(np.count_nonzero(contact_ring)))
    contact_ratio = contact_area / contact_ring_area
    near_area_normalized = near_area / max(1.0, radius * radius)

    best_distance = None
    normalized_gap = None
    best_ball_point = None
    best_finger_point = None

    skin_y, skin_x = np.nonzero(skin_pixels)
    if skin_x.size:
        dx = skin_x.astype(np.float32) - float(cx_roi)
        dy = skin_y.astype(np.float32) - float(cy_roi)
        distances_to_center = np.sqrt(dx * dx + dy * dy)
        outside_distances = np.maximum(0.0, distances_to_center - radius)

        # A low percentile is more robust than a single closest noise pixel.
        best_distance = float(np.percentile(outside_distances, 10.0))
        normalized_gap = best_distance / radius

        nearest_idx = int(np.argmin(outside_distances))
        fp_roi = np.array(
            [skin_x[nearest_idx], skin_y[nearest_idx]],
            dtype=np.float32,
        )
        fp = fp_roi + np.array([x1, y1], dtype=np.float32)
        norm = max(1e-6, float(distances_to_center[nearest_idx]))
        bp = np.array([cx, cy], dtype=np.float32) + (
            fp - np.array([cx, cy], dtype=np.float32)
        ) * (radius / norm)
        best_finger_point = (
            int(round(float(fp[0]))),
            int(round(float(fp[1]))),
        )
        best_ball_point = (
            int(round(float(bp[0]))),
            int(round(float(bp[1]))),
        )

    return FingerDistanceResult(
        distance_px=best_distance,
        finger_area_px=near_area,
        contact_area_px=contact_area,
        contact_ratio=contact_ratio,
        near_area_normalized=near_area_normalized,
        normalized_gap=normalized_gap,
        nearest_ball_point=best_ball_point,
        nearest_finger_point=best_finger_point,
        contours=contours_full,
        skin_mask_roi=filtered_mask,
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
        "gap: --"
        if result.normalized_gap is None
        else f"gap/r: {result.normalized_gap:.2f}"
    )
    cv2.putText(
        frame_bgr,
        (
            f"{state} | {distance_label} | "
            f"contact: {result.contact_ratio:.3f}"
        ),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
