import csv
import os
import time
import threading

import cv2
import keyboard
import numpy as np

from pyautd3 import Controller, Silencer, Static
from pyautd3.link.twincat import TwinCAT
from ultralytics import cfg

from .config import AppConfig
from .vision import (
    load_ximea_api,
    init_ximea_camera,
    load_intrinsic,
    load_affine_matrix,
    load_z_model,
    build_undistort_maps,
    undistort_frame,
    rotate_frame_if_needed,
    clamp_roi,
    track_ball_cv,
    SharedFrameBuffer,
    camera_capture_loop,
)
from .autd_sender import (
    make_autd_arrangement,
    AutdSender,
)
from .controller import (
    PredictionPIDController,
    HomePosition,
    Measurement3D,
    Target3D,
)


def update_base_position(home: HomePosition, cfg: AppConfig, dt_sec: float) -> HomePosition:
    """
    矢印キーによる基準位置移動。
    cfg.enable_base_move=True のときだけ使う。
    """
    if not cfg.enable_base_move:
        return home

    step_xy = cfg.base_move_speed_mm_s * max(0.0, dt_sec)
    step_z = cfg.base_z_move_speed_mm_s * max(0.0, dt_sec)

    x = float(home.x)
    y = float(home.y)
    z = float(home.z)

    if keyboard.is_pressed("left"):
        x -= step_xy
    if keyboard.is_pressed("right"):
        x += step_xy
    if keyboard.is_pressed("up"):
        y += step_xy
    if keyboard.is_pressed("down"):
        y -= step_xy
    if keyboard.is_pressed("page up"):
        z += step_z
    if keyboard.is_pressed("page down"):
        z -= step_z

    return HomePosition(x=x, y=y, z=z)


def _sender_set_target(
    sender: AutdSender,
    cfg: AppConfig,
    target_x: float,
    target_y: float,
    target_z: float,
    home: HomePosition,
    radius: float | None = None,
    intensity_ratio: float | None = None,
):
    """
    OutputMaskあり/なしの差、動的radius、動的intensityをここに閉じ込める。
    """
    if cfg.use_output_mask:
        sender.set_target(
            target_x,
            target_y,
            target_z,
            home.x,
            home.y,
            radius=radius,
            intensity_ratio=intensity_ratio,
        )
    else:
        sender.set_target(
            target_x,
            target_y,
            target_z,
            radius=radius,
            intensity_ratio=intensity_ratio,
        )


def limit_step(new_val: float, old_val: float, max_step: float) -> float:
    return old_val + float(np.clip(new_val - old_val, -max_step, max_step))


def move_towards(current: float, target: float, max_step: float) -> float:
    diff = target - current
    if abs(diff) <= max_step:
        return float(target)
    return float(current + np.sign(diff) * max_step)


def should_enter_fall_recovery(
    cfg: AppConfig,
    home_z: float,
    z_mm: float | None,
    vz: float,
) -> bool:
    if z_mm is None:
        return False

    z_drop = home_z - z_mm

    return (
        z_drop >= cfg.fall_drop_threshold_mm
        and vz <= cfg.fall_vz_threshold_mm_s
    )


# def is_capture_stable(
#     cfg: AppConfig,
#     vx: float,
#     vy: float,
#     vz: float,
# ) -> bool:
#     # speed_xy = float(np.hypot(vx, vy))

#     return (
#         # speed_xy <= cfg.fall_captured_speed_xy_mm_s
#         abs(vz) <= cfg.fall_captured_vz_mm_s
#     )


def is_return_home_done(
    cfg: AppConfig,
    home: HomePosition,
    z_mm: float | None,
    x_mm: float | None,
    y_mm: float | None,
    vx: float,
    vy: float,
    vz: float,
) -> bool:
    if x_mm is None or y_mm is None or z_mm is None:
        return False

    err_xy = float(np.hypot(x_mm - home.x, y_mm - home.y))
    err_z = abs(z_mm - home.z)
    speed_xy = float(np.hypot(vx, vy))

    return (
        err_xy <= cfg.return_home_done_error_xy_mm
        and err_z <= cfg.return_home_done_error_z_mm
        and speed_xy <= cfg.return_home_done_speed_xy_mm_s
        and abs(vz) <= cfg.return_home_done_vz_mm_s
    )


def compute_z_intensity_boost_ratio(
    cfg: AppConfig,
    home_z: float,
    z_mm: float | None,
) -> float:
    """
    z方向の落下量に応じて intensity_ratio の目標値を返す。

    - zが home_z から 1.5mm以内なら base
    - zが home_z から 5mm以上下がったら max
    - その間は線形補間
    """
    base = float(cfg.intensity_base_ratio)
    max_ratio = float(cfg.intensity_max_ratio)

    if z_mm is None:
        return base

    drop_mm = float(home_z - z_mm)

    # 下がっていない、または1.5mm以内なら通常強度
    if drop_mm <= cfg.z_boost_release_mm:
        return base

    # 5mm以上下がったら最大強度
    if drop_mm >= cfg.z_boost_full_drop_mm:
        return max_ratio

    # 1.5mm〜5mmの間は線形補間
    denom = cfg.z_boost_full_drop_mm - cfg.z_boost_release_mm
    if denom <= 1e-6:
        return max_ratio

    s = (drop_mm - cfg.z_boost_release_mm) / denom
    return base + s * (max_ratio - base)


def _start_log_session(cfg: AppConfig, mode: str):
    """
    30秒ログ開始。
    analyze_stability.py で読める列構成に合わせる。
    """
    os.makedirs(os.path.dirname(cfg.log_csv_path), exist_ok=True)

    file_exists = os.path.exists(cfg.log_csv_path)
    log_file = open(cfg.log_csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)

    if not file_exists:
        writer.writerow(
            [
                "timestamp",
                "mode",
                "u_xy_px",
                "v_xy_px",
                "v_z_px",
                "x_mm",
                "y_mm",
                "z_mm",
                "center_x_mm",
                "center_y_mm",
                "center_z_mm",
                "autd_target_x_mm",
                "autd_target_y_mm",
                "autd_target_z_mm",
            ]
        )

    end_time = time.time() + cfg.log_duration_sec
    print(f"[LOG] Start {cfg.log_duration_sec:.1f}s logging: mode={mode}")

    return log_file, writer, end_time


def _safe_close_camera(cam):
    try:
        cam.stop_acquisition()
    except Exception:
        pass

    try:
        cam.close_device()
    except Exception:
        pass


def run_tracking_app(cfg: AppConfig):
    """
    共通実行本体。

    tracking_fast_feedback.py / tracking_demo.py / tracking_demo_outputmask.py の
    共通部分をここに集約する。

    実験ごとの差分は cfg で切り替える。
    """

    # 1. Calibration files
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
        print(f"[ERROR] z model load failed: {e}")
        print("[ERROR] 先に coordinate_transformation_z.py と fit_affine_from_csv_z.py を実行してください。")
        return

    try:
        mtx_cam_xy, dist_cam_xy = load_intrinsic(cfg.intrinsic_xy_npz, "xy")
        mtx_cam_z, dist_cam_z = load_intrinsic(cfg.intrinsic_z_npz, "z")
        use_undistort = True
        print("[INFO] Vision pipeline: remap undistort each camera frame, then detect.")
    except Exception as e:
        print(f"[WARN] Intrinsic parameters load failed: {e}")
        mtx_cam_xy = None
        dist_cam_xy = None
        mtx_cam_z = None
        dist_cam_z = None
        use_undistort = False

    # 2. Camera initialization
    xiapi = load_ximea_api(cfg)

    if xiapi is None:
        print("[ERROR] XIMEA API not available.")
        return

    try:
        cam_xy, img_xy = init_ximea_camera(xiapi, cfg.camera_xy_sn, "xy", cfg)
        cam_z, img_z = init_ximea_camera(xiapi, cfg.camera_z_sn, "z", cfg)
    except Exception as e:
        print(f"[ERROR] Camera Init Failed: {e}")
        return

    # 最初の1フレームで画像サイズとremap mapを作る
    try:
        cam_xy.get_image(img_xy)
        frame0_raw_xy = img_xy.get_image_data_numpy()

        cam_z.get_image(img_z)
        frame0_raw_z = img_z.get_image_data_numpy()
    except Exception as e:
        print(f"[ERROR] Failed to get first camera frame: {e}")
        _safe_close_camera(cam_xy)
        _safe_close_camera(cam_z)
        return

    if use_undistort:
        map1_xy, map2_xy = build_undistort_maps(frame0_raw_xy.shape, mtx_cam_xy, dist_cam_xy)
        map1_z, map2_z = build_undistort_maps(frame0_raw_z.shape, mtx_cam_z, dist_cam_z)

        frame0_xy = undistort_frame(frame0_raw_xy, map1_xy, map2_xy)
        frame0_z = undistort_frame(frame0_raw_z, map1_z, map2_z)
    else:
        map1_xy = map2_xy = None
        map1_z = map2_z = None
        frame0_xy = frame0_raw_xy
        frame0_z = frame0_raw_z

    frame0_z = rotate_frame_if_needed(frame0_z, cfg.rotate_z_frame, cfg.rotate_z_code)

    H_xy, W_xy = frame0_xy.shape[:2]
    H_z, W_z = frame0_z.shape[:2]

    frame_buffer = SharedFrameBuffer()
    frame_buffer.set_xy(frame0_xy.copy(), time.time())
    frame_buffer.set_z(frame0_z.copy(), time.time())

    running_event = threading.Event()
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

    # 3. AUTD initialization
    autd_arrangement = make_autd_arrangement()

    print("[INFO] Opening AUTD Controller...")

    sender = None
    log_file = None
    log_writer = None

    try:
        with Controller.open(
            autd_arrangement,
            TwinCAT(),
        ) as autd:

            autd.send(Silencer())
            autd.send(Static(intensity=int(0xFF * cfg.static_intensity_ratio)))

            base_center = autd.center()

            home = HomePosition(
                x=float(base_center[0]),
                y=float(base_center[1]),
                z=float(cfg.default_z),
            )

            display_origin = HomePosition(
                x=float(home.x),
                y=float(home.y),
                z=float(home.z),
            )

            sender = AutdSender(autd, cfg)
            current_radius = float(cfg.radius)
            current_intensity_ratio = float(cfg.intensity_base_ratio)
            target_intensity_ratio = float(cfg.intensity_base_ratio)
            _sender_set_target(sender, cfg, home.x, home.y, home.z, home, current_radius, current_intensity_ratio)
            sender.start()

            controller = PredictionPIDController(cfg)
            controller.reset(Target3D(home.x, home.y, home.z))

            # 4. Runtime state
            tracking_active = False
            prev_enter_pressed = False
            prev_log_trigger_pressed = False

            control_mode = "NORMAL_HOLD"

            fall_recovery_start_time = None

            return_setpoint = HomePosition(
                x=home.x,
                y=home.y,
                z=home.z,
            )

            roi_cx_xy, roi_cy_xy = W_xy // 2, H_xy // 2
            roi_size_xy = cfg.roi_init_size

            roi_cx_z, roi_cy_z = W_z // 2, H_z // 2
            roi_size_z = cfg.roi_init_size

            last_u_xy = W_xy / 2
            last_v_xy = H_xy / 2
            last_v_z = H_z / 2

            last_target = Target3D(home.x, home.y, home.z)

            fps_start_time = time.time()
            loop_fps_count = 0
            loop_display_fps = 0.0
            new_xy_fps_count = 0
            new_z_fps_count = 0
            new_xy_display_fps = 0.0
            new_z_display_fps = 0.0
            frame_count = 0

            prev_frame_xy_time = 0.0
            prev_frame_z_time = 0.0
            prev_loop_time = time.time()

            log_session_active = False
            log_session_mode = ""
            log_session_end_time = 0.0
            last_logged_xy_time = None
            last_logged_z_time = None

            window_title = "Tracking App"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

            print("=================================================")
            print("  READY TO LEVITATE.")
            print("  Press [ENTER] to START/PAUSE feedback.")
            if cfg.enable_base_move:
                print("  Use [Arrow keys] to move base XY.")
                print("  Use [PageUp/PageDown] to move base Z.")
            if cfg.log_enabled:
                print(f"  Press [{cfg.log_trigger_key.upper()}] to START {cfg.log_duration_sec:.0f}s logging.")
            print("  Press [ESC] to EXIT and STOP ultrasound.")
            print("=================================================")

            # 5. Main loop
            while True:
                now_loop = time.time()
                dt_loop = max(1e-3, now_loop - prev_loop_time)
                prev_loop_time = now_loop

                # Key handling
                if keyboard.is_pressed("esc"):
                    print("[INFO] ESC pressed. Exit.")
                    break

                enter_pressed = keyboard.is_pressed("enter")
                if enter_pressed and not prev_enter_pressed:
                    tracking_active = not tracking_active

                    if tracking_active:
                        print("[INFO] >>> TRACKING ACTIVATED <<<")
                        controller.reset(last_target)
                    else:
                        print("[INFO] >>> TRACKING PAUSED: Return to initial base position <<<")
                        control_mode = "NORMAL_HOLD"
                        fall_recovery_start_time = None

                        home = HomePosition(
                            x=float(display_origin.x),
                            y=float(display_origin.y),
                            z=float(display_origin.z),
                        )
                        return_setpoint = HomePosition(
                            x=home.x,
                            y=home.y,
                            z=home.z,
                        )
                        last_target = Target3D(home.x, home.y, home.z)
                        controller.reset(last_target)
                        _sender_set_target(sender, cfg, home.x, home.y, home.z, home, current_radius, current_intensity_ratio)

                    time.sleep(0.2)

                prev_enter_pressed = enter_pressed

                if cfg.enable_base_move:
                    home = update_base_position(home, cfg, dt_loop)

                    if not tracking_active:
                        last_target = Target3D(home.x, home.y, home.z)
                        _sender_set_target(sender, cfg, home.x, home.y, home.z, home, current_radius, current_intensity_ratio)

                if cfg.enable_radius_change:
                    radius_step = cfg.radius_change_speed_mm_s * dt_loop

                    # 右 or 上でradiusを大きくする
                    if keyboard.is_pressed("right") or keyboard.is_pressed("up"):
                        current_radius += radius_step

                    # 左 or 下でradiusを小さくする
                    if keyboard.is_pressed("left") or keyboard.is_pressed("down"):
                        current_radius -= radius_step

                    current_radius = float(
                        np.clip(
                            current_radius,
                            cfg.radius_min,
                            cfg.radius_max,
                        )
                    )

                if cfg.log_enabled:
                    log_trigger_pressed = keyboard.is_pressed(cfg.log_trigger_key)

                    if log_trigger_pressed and not prev_log_trigger_pressed and not log_session_active:
                        log_session_mode = "PID" if tracking_active else "FIXED"
                        log_file, log_writer, log_session_end_time = _start_log_session(
                            cfg,
                            log_session_mode,
                        )
                        log_session_active = True
                        last_logged_xy_time = None
                        last_logged_z_time = None

                    prev_log_trigger_pressed = log_trigger_pressed

                if log_session_active and time.time() >= log_session_end_time:
                    log_session_active = False
                    if log_file is not None:
                        log_file.close()
                        log_file = None
                        log_writer = None
                    print("[LOG] Finished logging.")

                # Latest frames
                frame_xy, frame_xy_time = frame_buffer.get_xy()
                frame_z, frame_z_time = frame_buffer.get_z()

                if frame_xy is None or frame_z is None:
                    time.sleep(0.001)
                    continue

                new_xy = frame_xy_time != prev_frame_xy_time
                new_z = frame_z_time != prev_frame_z_time

                if new_xy:
                    new_xy_fps_count += 1
                    prev_frame_xy_time = frame_xy_time

                if new_z:
                    new_z_fps_count += 1
                    prev_frame_z_time = frame_z_time

                frame_count += 1
                loop_fps_count += 1
                do_display = (frame_count % cfg.display_every_n_frames == 0)

                if do_display:
                    if frame_xy.ndim == 2:
                        frame_xy_bgr = cv2.cvtColor(frame_xy, cv2.COLOR_GRAY2BGR)
                    else:
                        frame_xy_bgr = frame_xy.copy()

                    if frame_z.ndim == 2:
                        frame_z_bgr = cv2.cvtColor(frame_z, cv2.COLOR_GRAY2BGR)
                    else:
                        frame_z_bgr = frame_z.copy()
                else:
                    frame_xy_bgr = None
                    frame_z_bgr = None

                now_fps = time.time()
                if now_fps - fps_start_time >= 1.0:
                    elapsed = now_fps - fps_start_time
                    loop_display_fps = loop_fps_count / elapsed
                    new_xy_display_fps = new_xy_fps_count / elapsed
                    new_z_display_fps = new_z_fps_count / elapsed

                    loop_fps_count = 0
                    new_xy_fps_count = 0
                    new_z_fps_count = 0
                    fps_start_time = now_fps

                # Detection XY
                x1_xy, y1_xy, x2_xy, y2_xy = clamp_roi(
                    roi_cx_xy,
                    roi_cy_xy,
                    roi_size_xy,
                    W_xy,
                    H_xy,
                    cfg,
                )

                det_xy, _ = track_ball_cv(
                    frame_xy,
                    (x1_xy, y1_xy, x2_xy, y2_xy),
                    cfg,
                )

                detected_xy = det_xy is not None

                u_xy = np.nan
                v_xy = np.nan
                x_mm = None
                y_mm = None

                if detected_xy:
                    u_xy, v_xy, r_xy_px = det_xy
                    
                    # ROI中心を検出された球の中心へ移動
                    roi_cx_xy, roi_cy_xy = int(u_xy), int(v_xy)

                    # 元コードと同じROIサイズ更新
                    desired_xy = int(2 * (2.5 * r_xy_px + cfg.roi_margin))
                    roi_size_xy = int(0.7 * roi_size_xy + 0.3 * desired_xy)
                    roi_size_xy = int(
                        max(
                            cfg.roi_min_size,
                            min(cfg.roi_max_size, roi_size_xy),
                        )
                    )

                    if do_display:
                        color = (0, 255, 0) if tracking_active else (200, 200, 200)
                        cv2.circle(
                            frame_xy_bgr,
                            (int(u_xy), int(v_xy)),
                            int(max(2, r_xy_px)),
                            color,
                            2,
                        )
                        cv2.rectangle(
                            frame_xy_bgr,
                            (x1_xy, y1_xy),
                            (x2_xy, y2_xy),
                            (255, 255, 0),
                            2,
                        )

                    if use_affine:
                        uv_homo = np.array([[u_xy, v_xy, 1.0]], dtype=np.float32).T
                        xy_local = (A_affine @ uv_homo).flatten()
                        x_mm = float(home.x + xy_local[0])
                        y_mm = float(home.y + xy_local[1])

                else:
                    roi_size_xy = int(
                        min(
                            cfg.roi_max_size,
                            roi_size_xy * cfg.roi_expand_on_lost,
                        )
                    )

                    if do_display:
                        cv2.rectangle(
                            frame_xy_bgr,
                            (x1_xy, y1_xy),
                            (x2_xy, y2_xy),
                            (0, 255, 255),
                            2,
                        )

                # Detection Z
                x1_z, y1_z, x2_z, y2_z = clamp_roi(
                    roi_cx_z,
                    roi_cy_z,
                    roi_size_z,
                    W_z,
                    H_z,
                    cfg,
                )

                det_z, _ = track_ball_cv(
                    frame_z,
                    (x1_z, y1_z, x2_z, y2_z),
                    cfg,
                )

                detected_z = det_z is not None

                v_z = np.nan
                z_mm = None

                if detected_z:
                    u_z, v_z, r_z_px = det_z

                    roi_cx_z, roi_cy_z = int(u_z), int(v_z)

                    desired_z = int(2 * (2.5 * r_z_px + cfg.roi_margin))
                    roi_size_z = int(0.7 * roi_size_z + 0.3 * desired_z)
                    roi_size_z = int(
                        max(
                            cfg.roi_min_size,
                            min(cfg.roi_max_size, roi_size_z),
                        )
                    )

                    if do_display:
                        color = (0, 255, 0) if tracking_active else (200, 200, 200)
                        cv2.circle(
                            frame_z_bgr,
                            (int(u_z), int(v_z)),
                            int(max(2, r_z_px)),
                            color,
                            2,
                        )
                        cv2.rectangle(
                            frame_z_bgr,
                            (x1_z, y1_z),
                            (x2_z, y2_z),
                            (255, 255, 0),
                            2,
                        )

                    if use_z_model:
                        z_mm = float(z_a * v_z + z_b)

                else:
                    roi_size_z = int(
                        min(
                            cfg.roi_max_size,
                            roi_size_z * cfg.roi_expand_on_lost,
                        )
                    )

                    if do_display:
                        cv2.rectangle(
                            frame_z_bgr,
                            (x1_z, y1_z),
                            (x2_z, y2_z),
                            (0, 255, 255),
                            2,
                        )
                
                if cfg.enable_fall_recovery and control_mode == "FALL_RECOVERY":
                    target_intensity_ratio = float(cfg.fall_recovery_intensity_ratio)

                elif cfg.enable_z_intensity_boost and tracking_active:
                    target_intensity_ratio = compute_z_intensity_boost_ratio(
                        cfg,
                        home.z,
                        z_mm,
                    )

                else:
                    target_intensity_ratio = float(cfg.intensity_base_ratio)

                current_intensity_ratio = (
                    cfg.intensity_lpf_alpha * current_intensity_ratio
                    + (1.0 - cfg.intensity_lpf_alpha) * target_intensity_ratio
                )

                current_intensity_ratio = float(
                    np.clip(
                        current_intensity_ratio,
                        cfg.intensity_base_ratio,
                        cfg.intensity_max_ratio,
                    )
                )
                
                method = "XY+Z" if (detected_xy and detected_z) else "PARTIAL"

                # Control
                if tracking_active:
                    meas = Measurement3D(
                        detected_xy=detected_xy and x_mm is not None and y_mm is not None and new_xy,
                        detected_z=detected_z and z_mm is not None and new_z,
                        x=x_mm,
                        y=y_mm,
                        z=z_mm,
                        t_xy=frame_xy_time,
                        t_z=frame_z_time,
                    )

                    if meas.detected_xy or meas.detected_z:
                        # ====================================================
                        # 1. RETURN_TO_HOME中は一時基準位置をゆっくりhomeへ戻す
                        # ====================================================
                        if control_mode == "RETURN_TO_HOME":
                            return_setpoint = HomePosition(
                                x=move_towards(
                                    return_setpoint.x,
                                    home.x,
                                    cfg.return_home_speed_xy_mm_s * dt_loop,
                                ),
                                y=move_towards(
                                    return_setpoint.y,
                                    home.y,
                                    cfg.return_home_speed_xy_mm_s * dt_loop,
                                ),
                                z=move_towards(
                                    return_setpoint.z,
                                    home.z,
                                    cfg.return_home_speed_z_mm_s * dt_loop,
                                ),
                            )

                            control_home = return_setpoint
                        else:
                            control_home = home

                        # ====================================================
                        # 2. まず通常PIDを計算
                        #    FALL_RECOVERY中は後でtargetを上書きする
                        # ====================================================
                        target, debug = controller.update(meas, control_home)

                        vx_now = float(debug.vx)
                        vy_now = float(debug.vy)
                        vz_now = float(debug.vz)

                        # ====================================================
                        # 3. NORMAL_HOLD / RETURN_TO_HOME中に落下検知したら
                        #    FALL_RECOVERYへ入る
                        # ====================================================
                        if cfg.enable_fall_recovery:
                            if control_mode in ["NORMAL_HOLD", "RETURN_TO_HOME"]:
                                if should_enter_fall_recovery(cfg, home.z, z_mm, vz_now):
                                    control_mode = "FALL_RECOVERY"
                                    fall_recovery_start_time = time.time()

                                    print("[RECOVERY] -> FALL_RECOVERY")

                        # ====================================================
                        # 4. FALL_RECOVERY中は0.5秒だけ物体を追従
                        # ====================================================
                        if control_mode == "FALL_RECOVERY":
                            elapsed_recovery = 0.0
                            if fall_recovery_start_time is not None:
                                elapsed_recovery = time.time() - fall_recovery_start_time

                            # 0.5秒未満なら、物体の現在位置/予測位置へ追従
                            if elapsed_recovery < cfg.fall_recovery_hold_time_s:
                                if x_mm is not None and y_mm is not None:
                                    catch_x = x_mm + vx_now * cfg.fall_recovery_dt_pred_xy
                                    catch_y = y_mm + vy_now * cfg.fall_recovery_dt_pred_xy

                                    catch_x = last_target.x + float(
                                        np.clip(
                                            catch_x - last_target.x,
                                            -cfg.fall_recovery_target_xy_step_mm,
                                            cfg.fall_recovery_target_xy_step_mm,
                                        )
                                    )
                                    catch_y = last_target.y + float(
                                        np.clip(
                                            catch_y - last_target.y,
                                            -cfg.fall_recovery_target_xy_step_mm,
                                            cfg.fall_recovery_target_xy_step_mm,
                                        )
                                    )
                                else:
                                    catch_x = last_target.x
                                    catch_y = last_target.y

                                if z_mm is not None:
                                    catch_z = (
                                        z_mm
                                        + vz_now * cfg.fall_recovery_dt_pred_z
                                        + cfg.fall_recovery_z_offset_mm
                                    )

                                    catch_z = float(np.clip(catch_z, cfg.z_min, cfg.z_max))

                                    catch_z = last_target.z + float(
                                        np.clip(
                                            catch_z - last_target.z,
                                            -cfg.fall_recovery_target_z_step_mm,
                                            cfg.fall_recovery_target_z_step_mm,
                                        )
                                    )
                                else:
                                    catch_z = last_target.z

                                target = Target3D(
                                    x=float(catch_x),
                                    y=float(catch_y),
                                    z=float(catch_z),
                                )

                            # 0.5秒経過したら、その時点の物体位置でPIDへ移行
                            else:
                                sx = x_mm if x_mm is not None else last_target.x
                                sy = y_mm if y_mm is not None else last_target.y
                                sz = z_mm if z_mm is not None else last_target.z

                                return_setpoint = HomePosition(
                                    x=float(sx),
                                    y=float(sy),
                                    z=float(sz),
                                )

                                control_mode = "RETURN_TO_HOME"
                                fall_recovery_start_time = None

                                # 捕捉位置を基準にPIDを再開するためリセット
                                controller.reset(
                                    Target3D(
                                        return_setpoint.x,
                                        return_setpoint.y,
                                        return_setpoint.z,
                                    )
                                )

                                target = Target3D(
                                    x=return_setpoint.x,
                                    y=return_setpoint.y,
                                    z=return_setpoint.z,
                                )

                                print(
                                    f"[RECOVERY] FALL_RECOVERY -> RETURN_TO_HOME "
                                    f"setpoint=({return_setpoint.x:.1f}, "
                                    f"{return_setpoint.y:.1f}, {return_setpoint.z:.1f})"
                                )

                        # ====================================================
                        # 5. RETURN_TO_HOME完了判定
                        # ====================================================
                        if control_mode == "RETURN_TO_HOME":
                            if is_return_home_done(
                                cfg,
                                home,
                                z_mm,
                                x_mm,
                                y_mm,
                                vx_now,
                                vy_now,
                                vz_now,
                            ):
                                control_mode = "NORMAL_HOLD"

                                return_setpoint = HomePosition(
                                    x=home.x,
                                    y=home.y,
                                    z=home.z,
                                )

                                controller.reset(
                                    Target3D(
                                        home.x,
                                        home.y,
                                        home.z,
                                    )
                                )

                                print("[RECOVERY] RETURN_TO_HOME -> NORMAL_HOLD")

                        last_target = target

                        _sender_set_target(
                            sender,
                            cfg,
                            target.x,
                            target.y,
                            target.z,
                            home,
                            current_radius,
                            current_intensity_ratio,
                        )

                    if do_display:
                        cv2.putText(
                            frame_xy_bgr,
                            f"TGT XY: {last_target.x:.1f}, {last_target.y:.1f}",
                            (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )
                        cv2.putText(
                            frame_z_bgr,
                            f"TGT Z: {last_target.z:.1f}",
                            (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

                # Logging
                if log_session_active and log_writer is not None:
                    # 同じフレームの重複ログを避ける
                    should_log = (
                        last_logged_xy_time != frame_xy_time
                        or last_logged_z_time != frame_z_time
                    )

                    if should_log:
                        last_logged_xy_time = frame_xy_time
                        last_logged_z_time = frame_z_time

                        log_writer.writerow(
                            [
                                f"{time.time():.4f}",
                                log_session_mode,
                                f"{u_xy:.2f}" if np.isfinite(u_xy) else "",
                                f"{v_xy:.2f}" if np.isfinite(v_xy) else "",
                                f"{v_z:.2f}" if np.isfinite(v_z) else "",
                                f"{x_mm:.3f}" if x_mm is not None else "",
                                f"{y_mm:.3f}" if y_mm is not None else "",
                                f"{z_mm:.3f}" if z_mm is not None else "",
                                f"{home.x:.3f}",
                                f"{home.y:.3f}",
                                f"{home.z:.3f}",
                                f"{last_target.x:.3f}",
                                f"{last_target.y:.3f}",
                                f"{last_target.z:.3f}",
                            ]
                        )
                
                now = time.time()
                if now - fps_start_time >= 1.0:
                    elapsed = now - fps_start_time
                    loop_display_fps = loop_fps_count / elapsed
                    new_xy_display_fps = new_xy_fps_count / elapsed
                    new_z_display_fps = new_z_fps_count / elapsed

                    fps_start_time = now
                    loop_fps_count = 0
                    new_xy_fps_count = 0
                    new_z_fps_count = 0

                # Display
                status_text = "ACTIVE" if tracking_active else "WAIT (Press ENTER)"
                status_color = (0, 255, 0) if tracking_active else (0, 165, 255)

                if do_display:
                    cv2.putText(
                        frame_xy_bgr,
                        f"Loop FPS: {loop_display_fps:.1f} | NewXY FPS: {new_xy_display_fps:.1f} | AUTD FPS: {sender.display_fps:.1f} | {method}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame_xy_bgr,
                        f"Ref Point: ({home.x - display_origin.x:.1f}, {home.y - display_origin.y:.1f}, {home.z - display_origin.z + 400.0:.1f})",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame_xy_bgr,
                        # f"RADIUS: {current_radius:.1f} mm",
                        f"return setpoint: ({return_setpoint.x:.1f}, {return_setpoint.y:.1f}, {return_setpoint.z:.1f})",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame_xy_bgr,
                        f"INTENSITY: {current_intensity_ratio:.3f}",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame_xy_bgr,
                        f"MODE: {control_mode}",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255) if control_mode == "FALL_RECOVERY" else (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame_xy_bgr,
                        f"STATUS: {status_text}",
                        (10, H_xy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        status_color,
                        2,
                    )

                    cv2.putText(
                        frame_z_bgr,
                        f"Loop FPS: {loop_display_fps:.1f} | NewZ FPS: {new_z_display_fps:.1f} | v_z={v_z:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame_z_bgr,
                        f"Ref Point: ({home.x - display_origin.x:.1f}, {home.y - display_origin.y:.1f}, {home.z - display_origin.z + 400.0:.1f})",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame_z_bgr,
                        f"RADIUS: {current_radius:.1f} mm",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame_z_bgr,
                        f"INTENSITY: {current_intensity_ratio:.3f}",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame_z_bgr,
                        f"MODE: {control_mode}",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255) if control_mode == "FALL_RECOVERY" else (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame_z_bgr,
                        f"STATUS: {status_text}",
                        (10, H_z - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        status_color,
                        2,
                    )

                    display_h = max(frame_xy_bgr.shape[0], frame_z_bgr.shape[0])
                    display_w = max(frame_xy_bgr.shape[1], frame_z_bgr.shape[1])

                    frame_xy_disp = cv2.resize(
                        frame_xy_bgr,
                        (display_w, display_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    frame_z_disp = cv2.resize(
                        frame_z_bgr,
                        (display_w, display_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    tiled = np.hstack([frame_xy_disp, frame_z_disp])
                    split_x = frame_xy_disp.shape[1]

                    cv2.line(
                        tiled,
                        (split_x, 0),
                        (split_x, tiled.shape[0] - 1),
                        (255, 255, 255),
                        1,
                    )

                    cv2.imshow(window_title, tiled)

                    if cv2.waitKey(1) & 0xFF == 27:
                        break

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt.")

    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")

    finally:
        print("[INFO] stopping...")

        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass

        running_event.clear()

        try:
            cam_thread_xy.join(timeout=1.0)
            cam_thread_z.join(timeout=1.0)
        except Exception:
            pass

        if sender is not None:
            try:
                sender.stop()
            except Exception:
                pass

        _safe_close_camera(cam_xy)
        _safe_close_camera(cam_z)

        cv2.destroyAllWindows()

        print("[INFO] stopped.")