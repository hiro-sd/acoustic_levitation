from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from ..stereo import RigidTransform, StereoTriangulator
from ..vision import (
    SharedFrameBuffer,
    build_undistort_maps,
    camera_capture_loop,
    init_ximea_camera,
    load_intrinsic,
    load_ximea_api,
    rotate_frame_if_needed,
    safe_close_camera,
    undistort_frame,
)


@dataclass
class CameraRuntime:
    cam_xy: object
    cam_z: object
    frame_buffer: SharedFrameBuffer
    running_event: threading.Event
    thread_xy: threading.Thread
    thread_z: threading.Thread
    width_xy: int
    height_xy: int
    width_z: int
    height_z: int
    stereo_triangulator: StereoTriangulator | None
    camera_to_autd: RigidTransform | None

    def close(self):
        self.running_event.clear()
        try:
            self.thread_xy.join(timeout=1.0)
            self.thread_z.join(timeout=1.0)
        except Exception:
            pass
        safe_close_camera(self.cam_xy)
        safe_close_camera(self.cam_z)


def _load_stereo_runtime(cfg, *, input_is_undistorted: bool):
    triangulator = None
    camera_to_autd = None
    if not cfg.enable_stereo_triangulation:
        return triangulator, camera_to_autd

    triangulator = StereoTriangulator.from_npz(
        cfg.stereo_npz,
        input_is_undistorted=input_is_undistorted,
    )
    print(f"[INFO] Loaded stereo calibration from {cfg.stereo_npz}")
    if cfg.stereo_camera_to_autd_npz:
        camera_to_autd = RigidTransform.from_npz(
            cfg.stereo_camera_to_autd_npz
        )
        print(
            "[INFO] Loaded stereo camera-to-AUTD transform from "
            f"{cfg.stereo_camera_to_autd_npz}"
        )
    if cfg.use_stereo_position_for_control and camera_to_autd is None:
        raise ValueError(
            "use_stereo_position_for_control=True requires "
            "cfg.stereo_camera_to_autd_npz"
        )
    return triangulator, camera_to_autd


def start_camera_runtime(cfg) -> CameraRuntime | None:
    """Open and synchronize the two cameras for an isolated experiment."""

    try:
        mtx_xy, dist_xy = load_intrinsic(cfg.intrinsic_xy_npz, "xy")
        mtx_z, dist_z = load_intrinsic(cfg.intrinsic_z_npz, "z")
        use_undistort = True
    except Exception as error:
        print(f"[WARN] Intrinsic parameters load failed: {error}")
        mtx_xy = dist_xy = mtx_z = dist_z = None
        use_undistort = False

    try:
        stereo_triangulator, camera_to_autd = _load_stereo_runtime(
            cfg,
            input_is_undistorted=use_undistort,
        )
    except Exception as error:
        print(f"[ERROR] Stereo runtime load failed: {error}")
        return None

    xiapi = load_ximea_api(cfg)
    if xiapi is None:
        print("[ERROR] XIMEA API not available.")
        return None
    cam_xy = None
    cam_z = None
    try:
        cam_xy, img_xy = init_ximea_camera(
            xiapi, cfg.camera_xy_sn, "xy", cfg
        )
        cam_z, img_z = init_ximea_camera(
            xiapi, cfg.camera_z_sn, "z", cfg
        )
    except Exception as error:
        print(f"[ERROR] Camera Init Failed: {error}")
        if cam_xy is not None:
            safe_close_camera(cam_xy)
        if cam_z is not None:
            safe_close_camera(cam_z)
        return None

    try:
        cam_xy.get_image(img_xy)
        time_xy = time.perf_counter()
        raw_xy = img_xy.get_image_data_numpy()
        cam_z.get_image(img_z)
        time_z = time.perf_counter()
        raw_z = img_z.get_image_data_numpy()
    except Exception as error:
        print(f"[ERROR] Failed to get first camera frame: {error}")
        safe_close_camera(cam_xy)
        safe_close_camera(cam_z)
        return None

    if use_undistort:
        map1_xy, map2_xy = build_undistort_maps(
            raw_xy.shape, mtx_xy, dist_xy
        )
        frame_xy = undistort_frame(raw_xy, map1_xy, map2_xy)
        if cfg.rotate_z_frame and cfg.z_intrinsic_is_rotated:
            z_for_intrinsic = rotate_frame_if_needed(
                raw_z, True, cfg.rotate_z_code
            )
            map1_z, map2_z = build_undistort_maps(
                z_for_intrinsic.shape, mtx_z, dist_z
            )
            frame_z = undistort_frame(z_for_intrinsic, map1_z, map2_z)
        else:
            map1_z, map2_z = build_undistort_maps(
                raw_z.shape, mtx_z, dist_z
            )
            frame_z = undistort_frame(raw_z, map1_z, map2_z)
    else:
        map1_xy = map2_xy = map1_z = map2_z = None
        frame_xy = raw_xy
        frame_z = raw_z

    if not (use_undistort and cfg.z_intrinsic_is_rotated):
        frame_z = rotate_frame_if_needed(
            frame_z, cfg.rotate_z_frame, cfg.rotate_z_code
        )

    height_xy, width_xy = frame_xy.shape[:2]
    height_z, width_z = frame_z.shape[:2]
    frame_buffer = SharedFrameBuffer(maxlen=cfg.camera_sync_buffer_size)
    frame_buffer.set_xy(frame_xy.copy(), time_xy)
    frame_buffer.set_z(frame_z.copy(), time_z)
    running_event = threading.Event()
    running_event.set()
    thread_xy = threading.Thread(
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
    thread_z = threading.Thread(
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
    thread_xy.start()
    thread_z.start()
    return CameraRuntime(
        cam_xy=cam_xy,
        cam_z=cam_z,
        frame_buffer=frame_buffer,
        running_event=running_event,
        thread_xy=thread_xy,
        thread_z=thread_z,
        width_xy=width_xy,
        height_xy=height_xy,
        width_z=width_z,
        height_z=height_z,
        stereo_triangulator=stereo_triangulator,
        camera_to_autd=camera_to_autd,
    )
