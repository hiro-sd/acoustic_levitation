from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..ball_tracker import RoiBallTracker, draw_ball_detection


@dataclass(frozen=True)
class StereoFrameResult:
    det_xy: object
    det_z: object
    x_mm: float | None
    y_mm: float | None
    z_mm: float | None
    camera_point: object | None
    autd_point: object | None

    @property
    def valid_3d(self) -> bool:
        return self.x_mm is not None and self.y_mm is not None and self.z_mm is not None


def _display_frame(frame):
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame.copy()


class StereoBallPipeline:
    def __init__(
        self,
        *,
        width_xy: int,
        height_xy: int,
        width_z: int,
        height_z: int,
        cfg,
        triangulator,
        camera_to_autd,
    ):
        self.cfg = cfg
        self.tracker_xy = RoiBallTracker(width_xy, height_xy, cfg)
        self.tracker_z = RoiBallTracker(width_z, height_z, cfg)
        self.triangulator = triangulator
        self.camera_to_autd = camera_to_autd

    def process(self, frame_xy, frame_z, *, active: bool, draw: bool):
        frame_xy_bgr = _display_frame(frame_xy) if draw else None
        frame_z_bgr = _display_frame(frame_z) if draw else None
        det_xy = self.tracker_xy.detect(frame_xy)
        det_z = self.tracker_z.detect(frame_z)
        if draw:
            draw_ball_detection(frame_xy_bgr, det_xy, active)
            draw_ball_detection(frame_z_bgr, det_z, active)

        x_mm = y_mm = z_mm = None
        camera_point = None
        autd_point = None
        if det_xy.detected and det_z.detected and self.triangulator is not None:
            try:
                u_xy, v_xy = det_xy.center
                u_z, v_z = det_z.center
                if np.all(np.isfinite([u_xy, v_xy, u_z, v_z])):
                    camera_point = self.triangulator.triangulate_pixels(
                        (float(u_xy), float(v_xy)),
                        (float(u_z), float(v_z)),
                    )
                    if self.camera_to_autd is not None:
                        autd_point = self.camera_to_autd.apply(camera_point)
                        if self.cfg.use_stereo_position_for_control:
                            x_mm = float(autd_point[0])
                            y_mm = float(autd_point[1])
                            z_mm = float(autd_point[2])
            except Exception as error:
                print(f"[WARN] Stereo triangulation failed: {error}")
        return (
            StereoFrameResult(
                det_xy=det_xy,
                det_z=det_z,
                x_mm=x_mm,
                y_mm=y_mm,
                z_mm=z_mm,
                camera_point=camera_point,
                autd_point=autd_point,
            ),
            frame_xy_bgr,
            frame_z_bgr,
        )
