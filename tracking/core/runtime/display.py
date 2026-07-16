from dataclasses import dataclass

import cv2
import numpy as np

from ..models import HomePosition


@dataclass(frozen=True)
class DisplayMetrics:
    loop_fps: float
    pair_fps: float
    z_fps: float
    sync_skew_sec: float
    autd_fps: float
    method: str
    v_z_px: float


@dataclass(frozen=True)
class DisplayControlState:
    tracking_active: bool
    control_mode: str
    demo_active: bool
    home: HomePosition
    origin: HomePosition
    return_setpoint: HomePosition
    radius: float
    intensity_ratio: float


def _put(frame, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
    )


def render_tracking_window(
    window_title: str,
    frame_xy: np.ndarray,
    frame_z: np.ndarray,
    metrics: DisplayMetrics,
    state: DisplayControlState,
) -> bool:
    """Render one paired observation and return True when ESC is requested."""
    status_text = "ACTIVE" if state.tracking_active else "WAIT (Press ENTER)"
    status_color = (0, 255, 0) if state.tracking_active else (0, 165, 255)
    mode_color = (
        (0, 255, 255)
        if state.control_mode in {"FOLLOW_AND_BRAKE", "LOCAL_HOLD", "RETURN_TO_HOME"}
        else (255, 255, 255)
    )
    ref_text = (
        f"Ref Point: ({state.home.x - state.origin.x:.1f}, "
        f"{state.home.y - state.origin.y:.1f}, "
        f"{state.home.z - state.origin.z + 400.0:.1f})"
    )

    _put(
        frame_xy,
        f"Loop FPS: {metrics.loop_fps:.1f} | " #| NewXY FPS: {metrics.pair_fps:.1f} | "
        # f"Sync: {metrics.sync_skew_sec * 1000.0:.2f} ms | "
        f"AUTD FPS: {metrics.autd_fps:.1f}",# | {metrics.method}",
        30,
    )
    _put(frame_xy, ref_text, 60, (0, 255, 255))
    # _put(
    #     frame_xy,
    #     f"return setpoint: ({state.return_setpoint.x:.1f}, "
    #     f"{state.return_setpoint.y:.1f}, {state.return_setpoint.z:.1f})",
    #     90,
    #     (0, 255, 255),
    # )
    # _put(frame_xy, f"INTENSITY: {state.intensity_ratio:.3f}", 120, (0, 255, 255))
    # demo_text = "ON" if state.demo_active else "OFF"
    # _put(frame_xy, f"MODE: {state.control_mode} | DEMO: {demo_text}", 150, mode_color)
    _put(frame_xy, f"STATUS: {status_text}", frame_xy.shape[0] - 20, status_color, 0.8)

    _put(
        frame_z,
        f"Loop FPS: {metrics.loop_fps:.1f}",# | NewZ FPS: {metrics.z_fps:.1f} | "
        #f"v_z={metrics.v_z_px:.1f}",
        30,
    )
    _put(frame_z, ref_text, 60, (0, 255, 255))
    # _put(frame_z, f"RADIUS: {state.radius:.1f} mm", 90, (0, 255, 255))
    # _put(frame_z, f"INTENSITY: {state.intensity_ratio:.3f}", 120, (0, 255, 255))
    # _put(frame_z, f"MODE: {state.control_mode} | DEMO: {demo_text}", 150, mode_color)
    _put(frame_z, f"STATUS: {status_text}", frame_z.shape[0] - 20, status_color, 0.8)

    display_h = max(frame_xy.shape[0], frame_z.shape[0])
    display_w = max(frame_xy.shape[1], frame_z.shape[1])
    frame_xy_disp = cv2.resize(frame_xy, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
    frame_z_disp = cv2.resize(frame_z, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
    tiled = np.hstack([frame_xy_disp, frame_z_disp])
    cv2.line(tiled, (display_w, 0), (display_w, tiled.shape[0] - 1), (255, 255, 255), 1)
    cv2.imshow(window_title, tiled)
    return bool(cv2.waitKey(1) & 0xFF == 27)
