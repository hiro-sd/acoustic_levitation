import sys
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.ball_tracker import RoiBallTracker, draw_ball_detection
from tracking.core.config import AppConfig
from tracking.core.stereo import RigidTransform, StereoTriangulator
from tracking.core.vision import (
    SharedFrameBuffer,
    build_undistort_maps,
    camera_capture_loop,
    init_ximea_camera,
    load_affine_matrix,
    load_intrinsic,
    load_ximea_api,
    load_z_model,
    rotate_frame_if_needed,
    safe_close_camera,
    undistort_frame,
)
from manual_release_tracking.release_detector import (
    ReleaseDetectorConfig,
    ReleaseState,
    ReleaseStateMachine,
    detect_finger_distance,
    draw_release_debug,
)


class VelocityEstimator:
    def __init__(self, maxlen: int = 8):
        self.samples = deque(maxlen=maxlen)

    def reset(self):
        self.samples.clear()

    def update(self, t: float, x: float | None, y: float | None, z: float | None):
        if x is None or y is None or z is None:
            return None

        self.samples.append((float(t), np.array([x, y, z], dtype=float)))
        if len(self.samples) < 2:
            return np.zeros(3, dtype=float)

        t0, p0 = self.samples[0]
        t1, p1 = self.samples[-1]
        dt = max(1e-3, t1 - t0)
        return (p1 - p0) / dt


def _to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    if frame_rgb.ndim == 2:
        return cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def _put_lines(frame: np.ndarray, lines: list[str], x: int, y: int, color):
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + 24 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
        )


def _resize_to_height(frame: np.ndarray, height: int):
    h, w = frame.shape[:2]
    if h == height:
        return frame
    scale = height / max(1, h)
    return cv2.resize(frame, (int(w * scale), height), interpolation=cv2.INTER_AREA)


def _release_signal(result, release_cfg: ReleaseDetectorConfig):
    close = (
        result.distance_px is not None
        and result.distance_px <= release_cfg.grasp_distance_px
        and result.finger_area_px >= release_cfg.grasp_finger_area_px
    )
    separated = (
        result.distance_px is None
        or result.distance_px >= release_cfg.release_distance_px
        or result.finger_area_px <= release_cfg.release_finger_area_px
    )
    return close, separated


def _combine_release_signals(
    mode: str,
    xy_signal: tuple[bool, bool],
    z_signal: tuple[bool, bool],
):
    xy_close, xy_separated = xy_signal
    z_close, z_separated = z_signal

    if mode == "xy":
        return xy_close, xy_separated
    if mode == "z":
        return z_close, z_separated
    if mode == "either":
        return xy_close or z_close, xy_separated or z_separated
    if mode == "both":
        return xy_close and z_close, xy_separated and z_separated

    raise ValueError(
        f"Unknown release_preview_camera={mode!r}. "
        "Use 'xy', 'z', 'either', or 'both'."
    )


def run_release_detection_preview(cfg: AppConfig):
    # Calibration files
    try:
        A_affine, affine_uv_type = load_affine_matrix(cfg.affine_xy_json)
        use_affine = True
        print(f"[INFO] Loaded affine matrix from {cfg.affine_xy_json}")
        print(f"[INFO] affine input_uv_type = {affine_uv_type}")
    except Exception as e:
        print(f"[WARN] Affine matrix load failed: {e}")
        A_affine = None
        use_affine = False

    try:
        z_a, z_b = load_z_model(cfg.affine_z_json)
        use_z_model = True
    except Exception as e:
        print(f"[WARN] z model load failed: {e}")
        z_a = z_b = 0.0
        use_z_model = False

    try:
        mtx_cam_xy, dist_cam_xy = load_intrinsic(cfg.intrinsic_xy_npz, "xy")
        mtx_cam_z, dist_cam_z = load_intrinsic(cfg.intrinsic_z_npz, "z")
        use_undistort = True
        print("[INFO] Vision pipeline: remap undistort each camera frame, then detect.")
    except Exception as e:
        print(f"[WARN] Intrinsic parameters load failed: {e}")
        mtx_cam_xy = dist_cam_xy = None
        mtx_cam_z = dist_cam_z = None
        use_undistort = False

    stereo_triangulator = None
    camera_to_autd = None
    if cfg.enable_stereo_triangulation:
        stereo_triangulator = StereoTriangulator.from_npz(
            cfg.stereo_npz,
            input_is_undistorted=use_undistort,
        )
        print(f"[INFO] Loaded stereo calibration from {cfg.stereo_npz}")

        if cfg.stereo_camera_to_autd_npz:
            camera_to_autd = RigidTransform.from_npz(cfg.stereo_camera_to_autd_npz)
            print(
                f"[INFO] Loaded stereo camera-to-AUTD transform from "
                f"{cfg.stereo_camera_to_autd_npz}"
            )

    # Camera initialization
    xiapi = load_ximea_api(cfg)
    if xiapi is None:
        print("[ERROR] XIMEA API not available.")
        return

    cam_xy = cam_z = None
    running_event = threading.Event()
    cam_thread_xy = cam_thread_z = None

    try:
        cam_xy, img_xy = init_ximea_camera(xiapi, cfg.camera_xy_sn, "xy", cfg)
        cam_z, img_z = init_ximea_camera(xiapi, cfg.camera_z_sn, "z", cfg)

        cam_xy.get_image(img_xy)
        frame0_xy_time = time.perf_counter()
        frame0_raw_xy = img_xy.get_image_data_numpy()

        cam_z.get_image(img_z)
        frame0_z_time = time.perf_counter()
        frame0_raw_z = img_z.get_image_data_numpy()

        if use_undistort:
            map1_xy, map2_xy = build_undistort_maps(
                frame0_raw_xy.shape,
                mtx_cam_xy,
                dist_cam_xy,
            )
            frame0_xy = undistort_frame(frame0_raw_xy, map1_xy, map2_xy)

            if cfg.rotate_z_frame and cfg.z_intrinsic_is_rotated:
                frame0_z_for_intrinsic = rotate_frame_if_needed(
                    frame0_raw_z,
                    True,
                    cfg.rotate_z_code,
                )
                map1_z, map2_z = build_undistort_maps(
                    frame0_z_for_intrinsic.shape,
                    mtx_cam_z,
                    dist_cam_z,
                )
                frame0_z = undistort_frame(frame0_z_for_intrinsic, map1_z, map2_z)
            else:
                map1_z, map2_z = build_undistort_maps(
                    frame0_raw_z.shape,
                    mtx_cam_z,
                    dist_cam_z,
                )
                frame0_z = undistort_frame(frame0_raw_z, map1_z, map2_z)
        else:
            map1_xy = map2_xy = None
            map1_z = map2_z = None
            frame0_xy = frame0_raw_xy
            frame0_z = frame0_raw_z

        if not (use_undistort and cfg.z_intrinsic_is_rotated):
            frame0_z = rotate_frame_if_needed(
                frame0_z,
                cfg.rotate_z_frame,
                cfg.rotate_z_code,
            )

        H_xy, W_xy = frame0_xy.shape[:2]
        H_z, W_z = frame0_z.shape[:2]

        frame_buffer = SharedFrameBuffer(maxlen=cfg.camera_sync_buffer_size)
        frame_buffer.set_xy(frame0_xy.copy(), frame0_xy_time)
        frame_buffer.set_z(frame0_z.copy(), frame0_z_time)

        running_event.set()
        cam_thread_xy = threading.Thread(
            target=camera_capture_loop,
            args=(
                cam_xy,
                img_xy,
                "xy",
                cfg,
                frame_buffer,
                running_event,
                use_undistort,
                map1_xy,
                map2_xy,
            ),
            daemon=True,
        )
        cam_thread_z = threading.Thread(
            target=camera_capture_loop,
            args=(
                cam_z,
                img_z,
                "z",
                cfg,
                frame_buffer,
                running_event,
                use_undistort,
                map1_z,
                map2_z,
            ),
            daemon=True,
        )
        cam_thread_xy.start()
        cam_thread_z.start()

        tracker_xy = RoiBallTracker(W_xy, H_xy, cfg)
        tracker_z = RoiBallTracker(W_z, H_z, cfg)
        release_cfg = ReleaseDetectorConfig()
        release_sm = ReleaseStateMachine(release_cfg)
        velocity = VelocityEstimator()
        release_camera = str(getattr(cfg, "release_preview_camera", "z")).lower()

        fps_start_time = time.time()
        frame_count = 0
        display_fps = 0.0
        last_synced_pair_time = time.perf_counter()
        window_title = "Release Detection Preview"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

        print("=================================================")
        print("  RELEASE DETECTION PREVIEW")
        print("  This preview does not open AUTD and emits no ultrasound.")
        print("  Hold/pinch the sphere, then release it.")
        print(f"  Release decision camera: {release_camera}")
        print("  Watch finger distance and WAITING/GRASPED/RELEASED state.")
        print("  Press [SPACE] to reset the state machine.")
        print("  Press [ESC] to exit.")
        print("=================================================")

        while True:
            now_camera = time.perf_counter()
            age_xy, age_z = frame_buffer.get_frame_ages(now_camera)
            if age_xy > cfg.camera_frame_timeout_sec or age_z > cfg.camera_frame_timeout_sec:
                raise RuntimeError(
                    "Camera watchdog timeout: "
                    f"XY age={age_xy:.3f}s, Z age={age_z:.3f}s"
                )

            synced_pair = frame_buffer.get_synced_pair(cfg.camera_sync_tolerance_sec)
            if synced_pair is None:
                if now_camera - last_synced_pair_time > cfg.camera_frame_timeout_sec:
                    raise RuntimeError(
                        "Camera synchronization timeout: no frame pair within "
                        f"{cfg.camera_sync_tolerance_sec * 1000.0:.1f} ms"
                    )
                time.sleep(0.001)
                continue

            frame_xy, frame_xy_time, frame_z, frame_z_time, frame_sync_skew = synced_pair
            last_synced_pair_time = now_camera
            frame_count += 1

            if time.time() - fps_start_time >= 1.0:
                elapsed = max(1e-3, time.time() - fps_start_time)
                display_fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            det_xy = tracker_xy.detect(frame_xy)
            det_z = tracker_z.detect(frame_z)
            detected_xy = det_xy.detected
            detected_z = det_z.detected

            u_xy = v_xy = u_z = v_z = np.nan
            x_mm = y_mm = z_mm = None
            stereo_cam_point = None
            stereo_autd_point = None

            if detected_xy:
                u_xy, v_xy = det_xy.center
                if use_affine:
                    uv_homo = np.array([[u_xy, v_xy, 1.0]], dtype=np.float32).T
                    xy_local = (A_affine @ uv_homo).flatten()
                    x_mm = float(xy_local[0])
                    y_mm = float(xy_local[1])

            if detected_z:
                u_z, v_z = det_z.center
                if use_z_model:
                    z_mm = float(z_a * v_z + z_b)

            if (
                stereo_triangulator is not None
                and camera_to_autd is not None
                and detected_xy
                and detected_z
                and np.isfinite(u_xy)
                and np.isfinite(v_xy)
                and np.isfinite(u_z)
                and np.isfinite(v_z)
            ):
                try:
                    stereo_cam_point = stereo_triangulator.triangulate_pixels(
                        (float(u_xy), float(v_xy)),
                        (float(u_z), float(v_z)),
                    )
                    stereo_autd = camera_to_autd.apply(stereo_cam_point)
                    stereo_autd_point = stereo_autd
                    x_mm = float(stereo_autd[0])
                    y_mm = float(stereo_autd[1])
                    z_mm = float(stereo_autd[2])
                except Exception as e:
                    print(f"[WARN] Stereo triangulation failed: {e}")

            velocity_xyz = velocity.update(time.perf_counter(), x_mm, y_mm, z_mm)
            if velocity_xyz is None:
                vx = vy = vz = np.nan
                pred = None
            else:
                vx, vy, vz = [float(v) for v in velocity_xyz]
                dt_pred = float(getattr(cfg, "release_preview_pred_sec", 0.010))
                pred = np.array(
                    [
                        x_mm + vx * dt_pred,
                        y_mm + vy * dt_pred,
                        z_mm + vz * dt_pred - 0.5 * cfg.gravity_mm_s2 * dt_pred * dt_pred,
                    ],
                    dtype=float,
                )

            result_xy = detect_finger_distance(
                frame_xy,
                det_xy.center if detected_xy else None,
                det_xy.radius_px if detected_xy else None,
                release_cfg,
            )
            result_z = detect_finger_distance(
                frame_z,
                det_z.center if detected_z else None,
                det_z.radius_px if detected_z else None,
                release_cfg,
            )

            xy_signal = _release_signal(result_xy, release_cfg)
            z_signal = _release_signal(result_z, release_cfg)
            is_close, is_separated = _combine_release_signals(
                release_camera,
                xy_signal,
                z_signal,
            )

            selected_result = result_z if release_camera == "z" else result_xy
            if release_camera == "either":
                selected_distance = min(
                    [
                        d
                        for d in [result_xy.distance_px, result_z.distance_px]
                        if d is not None
                    ],
                    default=None,
                )
                selected_area = max(result_xy.finger_area_px, result_z.finger_area_px)
            elif release_camera == "both":
                selected_distance = max(
                    [
                        d
                        for d in [result_xy.distance_px, result_z.distance_px]
                        if d is not None
                    ],
                    default=None,
                )
                selected_area = min(result_xy.finger_area_px, result_z.finger_area_px)
            else:
                selected_distance = selected_result.distance_px
                selected_area = selected_result.finger_area_px

            state = release_sm.update(
                is_close,
                is_separated,
                ball_detected=bool(detected_xy and detected_z),
            )

            frame_xy_bgr = _to_bgr(frame_xy)
            frame_z_bgr = _to_bgr(frame_z)
            draw_ball_detection(frame_xy_bgr, det_xy, tracking_active=False)
            draw_ball_detection(frame_z_bgr, det_z, tracking_active=False)
            draw_release_debug(frame_xy_bgr, result_xy, f"XY {state}")
            draw_release_debug(frame_z_bgr, result_z, f"Z {state}")

            xyz_label = (
                "AUTD xyz: --"
                if x_mm is None or y_mm is None or z_mm is None
                else f"AUTD xyz: {x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f} mm"
            )
            v_label = (
                "v xyz: --"
                if not np.isfinite(vz)
                else f"v xyz: {vx:.1f}, {vy:.1f}, {vz:.1f} mm/s"
            )
            pred_label = (
                "pred 10ms: --"
                if pred is None
                else f"pred 10ms: {pred[0]:.1f}, {pred[1]:.1f}, {pred[2]:.1f}"
            )
            xy_dist_label = (
                "XY dist=--"
                if result_xy.distance_px is None
                else f"XY dist={result_xy.distance_px:.1f}px"
            )
            z_dist_label = (
                "Z dist=--"
                if result_z.distance_px is None
                else f"Z dist={result_z.distance_px:.1f}px"
            )
            selected_label = (
                f"selected({release_camera}) dist=--"
                if selected_distance is None
                else f"selected({release_camera}) dist={selected_distance:.1f}px"
            )
            area_label = (
                f"XY area={result_xy.finger_area_px:.0f}px, "
                f"Z area={result_z.finger_area_px:.0f}px, "
                f"selected area={selected_area:.0f}px"
            )
            threshold_label = (
                f"grasp: dist<={release_cfg.grasp_distance_px:.0f}px "
                f"& area>={release_cfg.grasp_finger_area_px:.0f}px | "
                f"release: dist>={release_cfg.release_distance_px:.0f}px "
                f"or area<={release_cfg.release_finger_area_px:.0f}px"
            )

            color = {
                ReleaseState.WAITING_FOR_GRASP: (200, 200, 200),
                ReleaseState.GRASPED: (0, 255, 255),
                ReleaseState.RELEASED: (0, 255, 0),
            }.get(state, (255, 255, 255))

            _put_lines(
                frame_xy_bgr,
                [
                    f"State: {state}",
                    f"Reason: {release_sm.transition_reason or '-'}",
                    xyz_label,
                    v_label,
                    pred_label,
                    selected_label,
                    f"{xy_dist_label}, {z_dist_label}",
                    area_label,
                    threshold_label,
                    f"close={is_close}, separated={is_separated}",
                    f"fps={display_fps:.1f}, sync={frame_sync_skew * 1000.0:.1f}ms",
                    "SPACE: reset, ESC: exit",
                ],
                10,
                60,
                color,
            )

            target_h = min(frame_xy_bgr.shape[0], frame_z_bgr.shape[0], 720)
            show_xy = _resize_to_height(frame_xy_bgr, target_h)
            show_z = _resize_to_height(frame_z_bgr, target_h)
            combined = np.hstack([show_xy, show_z])
            cv2.imshow(window_title, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord(" "):
                release_sm.reset()
                velocity.reset()
                print("[INFO] release state reset.")

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt.")
    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")
    finally:
        print("[INFO] stopping...")
        running_event.clear()
        try:
            if cam_thread_xy is not None:
                cam_thread_xy.join(timeout=1.0)
            if cam_thread_z is not None:
                cam_thread_z.join(timeout=1.0)
        except Exception:
            pass
        safe_close_camera(cam_xy)
        safe_close_camera(cam_z)
        cv2.destroyAllWindows()
        print("[INFO] stopped.")


if __name__ == "__main__":
    cfg = AppConfig(
        enable_stereo_triangulation=True,
        use_stereo_position_for_control=True,
        stereo_camera_to_autd_npz="./tracking/calibration/stereo_camera_to_autd.npz",
        log_enabled=False,
    )
    cfg.release_preview_pred_sec = 0.010
    cfg.release_preview_camera = "z"
    run_release_detection_preview(cfg)
