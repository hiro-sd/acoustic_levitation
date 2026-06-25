import time
import threading

import cv2
import numpy as np

from pyautd3 import Controller, Silencer, Static
from pyautd3.link.twincat import TwinCAT

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
    SharedFrameBuffer,
    camera_capture_loop,
    safe_close_camera,
)
from .ball_tracker import RoiBallTracker, draw_ball_detection
from .autd_sender import (
    make_autd_arrangement,
    AutdSender,
    set_tracking_target,
)
from .controller import PredictionPIDController
from .models import HomePosition, Measurement3D, Target3D
from .control.intensity import compute_z_intensity_boost_ratio
from .control.recovery import (
    is_return_home_done,
    move_towards,
    should_enter_fall_recovery,
)
from .runtime.display import (
    DisplayControlState,
    DisplayMetrics,
    render_tracking_window,
)
from .runtime.logging import StabilityLogger
from .runtime.input import is_key_pressed, update_base_position
from .runtime.demo import SquareZDemo
from .stereo import RigidTransform, StereoTriangulator


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

    stereo_triangulator = None
    camera_to_autd = None
    if cfg.enable_stereo_triangulation:
        try:
            stereo_triangulator = StereoTriangulator.from_npz(
                cfg.stereo_npz,
                input_is_undistorted=use_undistort,
            )
            print(f"[INFO] Loaded stereo calibration from {cfg.stereo_npz}")
        except Exception as e:
            print(f"[ERROR] Stereo calibration load failed: {e}")
            return

        if cfg.stereo_camera_to_autd_npz:
            try:
                camera_to_autd = RigidTransform.from_npz(cfg.stereo_camera_to_autd_npz)
                print(f"[INFO] Loaded stereo camera-to-AUTD transform from {cfg.stereo_camera_to_autd_npz}")
            except Exception as e:
                print(f"[ERROR] camera-to-AUTD transform load failed: {e}")
                return

        if cfg.use_stereo_position_for_control and camera_to_autd is None:
            print(
                "[ERROR] use_stereo_position_for_control=True requires "
                "cfg.stereo_camera_to_autd_npz. Stereo 3D is initially in cam1 coordinates, "
                "not AUTD coordinates."
            )
            return

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
        frame0_xy_time = time.perf_counter()
        frame0_raw_xy = img_xy.get_image_data_numpy()

        cam_z.get_image(img_z)
        frame0_z_time = time.perf_counter()
        frame0_raw_z = img_z.get_image_data_numpy()
    except Exception as e:
        print(f"[ERROR] Failed to get first camera frame: {e}")
        safe_close_camera(cam_xy)
        safe_close_camera(cam_z)
        return

    if use_undistort:
        map1_xy, map2_xy = build_undistort_maps(frame0_raw_xy.shape, mtx_cam_xy, dist_cam_xy)

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
            map1_z, map2_z = build_undistort_maps(frame0_raw_z.shape, mtx_cam_z, dist_cam_z)
            frame0_z = undistort_frame(frame0_raw_z, map1_z, map2_z)
    else:
        map1_xy = map2_xy = None
        map1_z = map2_z = None
        frame0_xy = frame0_raw_xy
        frame0_z = frame0_raw_z

    if not (use_undistort and cfg.z_intrinsic_is_rotated):
        frame0_z = rotate_frame_if_needed(frame0_z, cfg.rotate_z_frame, cfg.rotate_z_code)

    H_xy, W_xy = frame0_xy.shape[:2]
    H_z, W_z = frame0_z.shape[:2]

    frame_buffer = SharedFrameBuffer(maxlen=cfg.camera_sync_buffer_size)
    frame_buffer.set_xy(frame0_xy.copy(), frame0_xy_time)
    frame_buffer.set_z(frame0_z.copy(), frame0_z_time)

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
    logger = StabilityLogger(cfg)

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
            current_intensity_ratio = float(cfg.static_intensity_ratio)
            target_intensity_ratio = float(cfg.static_intensity_ratio)
            set_tracking_target(sender, cfg, home.x, home.y, home.z, home, current_radius, current_intensity_ratio)
            sender.start()

            controller = PredictionPIDController(cfg)
            controller.reset(Target3D(home.x, home.y, home.z))

            # 4. Runtime state
            tracking_active = False
            prev_enter_pressed = False
            prev_demo_toggle_pressed = False
            prev_log_trigger_pressed = False

            control_mode = "NORMAL_HOLD"
            demo_active = False
            demo = SquareZDemo(cfg, display_origin)

            fall_recovery_start_time = None

            return_setpoint = HomePosition(
                x=home.x,
                y=home.y,
                z=home.z,
            )

            tracker_xy = RoiBallTracker(W_xy, H_xy, cfg)
            tracker_z = RoiBallTracker(W_z, H_z, cfg)

            last_target = Target3D(home.x, home.y, home.z)

            fps_start_time = time.time()
            loop_fps_count = 0
            loop_display_fps = 0.0
            new_xy_fps_count = 0
            new_z_fps_count = 0
            new_xy_display_fps = 0.0
            new_z_display_fps = 0.0
            frame_count = 0

            prev_loop_time = time.time()
            last_synced_pair_time = time.perf_counter()

            window_title = "Tracking App"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

            print("=================================================")
            print("  READY TO LEVITATE.")
            print("  Press [ENTER] to START/PAUSE feedback.")
            if cfg.enable_base_move:
                print("  Use [Arrow keys] to move base XY.")
                print("  Use [PageUp/PageDown] to move base Z.")
            if cfg.enable_auto_demo:
                print(f"  Press [{cfg.demo_toggle_key.upper()}] to START/STOP square-Z demo.")
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
                if is_key_pressed("esc"):
                    print("[INFO] ESC pressed. Exit.")
                    break

                enter_pressed = is_key_pressed("enter")
                if enter_pressed and not prev_enter_pressed:
                    tracking_active = not tracking_active

                    if tracking_active:
                        print("[INFO] >>> TRACKING ACTIVATED <<<")
                        controller.reset(last_target)
                    else:
                        print("[INFO] >>> TRACKING PAUSED: Return to initial base position <<<")
                        control_mode = "NORMAL_HOLD"
                        fall_recovery_start_time = None
                        demo_active = False
                        demo.reset()

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
                        set_tracking_target(sender, cfg, home.x, home.y, home.z, home, current_radius, current_intensity_ratio)

                    time.sleep(0.2)

                prev_enter_pressed = enter_pressed

                if cfg.enable_auto_demo:
                    demo_toggle_pressed = is_key_pressed(cfg.demo_toggle_key)

                    if demo_toggle_pressed and not prev_demo_toggle_pressed:
                        demo_active = not demo_active
                        demo.reset()

                        if demo_active:
                            print("[DEMO] Square-Z demo started.")
                        else:
                            print("[DEMO] Square-Z demo stopped.")

                    prev_demo_toggle_pressed = demo_toggle_pressed

                if cfg.enable_auto_demo and demo_active and tracking_active:
                    home = demo.update(dt_loop)
                    control_mode = "NORMAL_HOLD"
                    return_setpoint = HomePosition(
                        x=home.x,
                        y=home.y,
                        z=home.z,
                    )

                elif cfg.enable_base_move:
                    home = update_base_position(home, cfg, dt_loop)

                    if not tracking_active:
                        last_target = Target3D(home.x, home.y, home.z)
                        set_tracking_target(sender, cfg, home.x, home.y, home.z, home, current_radius, current_intensity_ratio)

                if cfg.enable_radius_change:
                    radius_step = cfg.radius_change_speed_mm_s * dt_loop

                    # 右 or 上でradiusを大きくする
                    if is_key_pressed("right") or is_key_pressed("up"):
                        current_radius += radius_step

                    # 左 or 下でradiusを小さくする
                    if is_key_pressed("left") or is_key_pressed("down"):
                        current_radius -= radius_step

                    current_radius = float(
                        np.clip(
                            current_radius,
                            cfg.radius_min,
                            cfg.radius_max,
                        )
                    )

                if cfg.log_enabled:
                    log_trigger_pressed = is_key_pressed(cfg.log_trigger_key)

                    if log_trigger_pressed and not prev_log_trigger_pressed and not logger.active:
                        logger.start("PID" if tracking_active else "FIXED")

                    prev_log_trigger_pressed = log_trigger_pressed

                logger.stop_if_finished()

                # Camera watchdog and software-synchronized frame pair
                now_camera = time.perf_counter()
                age_xy, age_z = frame_buffer.get_frame_ages(now_camera)

                if (
                    age_xy > cfg.camera_frame_timeout_sec
                    or age_z > cfg.camera_frame_timeout_sec
                ):
                    sender.stop()
                    raise RuntimeError(
                        "Camera watchdog timeout: "
                        f"XY age={age_xy:.3f}s, Z age={age_z:.3f}s"
                    )

                synced_pair = frame_buffer.get_synced_pair(
                    cfg.camera_sync_tolerance_sec
                )

                if synced_pair is None:
                    if now_camera - last_synced_pair_time > cfg.camera_frame_timeout_sec:
                        sender.stop()
                        raise RuntimeError(
                            "Camera synchronization timeout: no frame pair within "
                            f"{cfg.camera_sync_tolerance_sec * 1000.0:.1f} ms"
                        )
                    time.sleep(0.001)
                    continue

                (
                    frame_xy,
                    frame_xy_time,
                    frame_z,
                    frame_z_time,
                    frame_sync_skew,
                ) = synced_pair
                last_synced_pair_time = now_camera

                # get_synced_pair() は消費済みの新規フレームだけを返す。
                new_xy_fps_count += 1
                new_z_fps_count += 1

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
                det_xy = tracker_xy.detect(frame_xy)
                detected_xy = det_xy.detected
                u_xy = np.nan
                v_xy = np.nan
                x_mm = None
                y_mm = None
                stereo_cam_point = None
                stereo_autd_point = None

                if detected_xy:
                    u_xy, v_xy = det_xy.center

                    if use_affine:
                        uv_homo = np.array([[u_xy, v_xy, 1.0]], dtype=np.float32).T
                        xy_local = (A_affine @ uv_homo).flatten()
                        # カメラ座標変換は移動する目標位置 home ではなく、
                        # キャリブレーション時の固定AUTD原点を基準にする。
                        x_mm = float(display_origin.x + xy_local[0])
                        y_mm = float(display_origin.y + xy_local[1])

                if do_display:
                    draw_ball_detection(frame_xy_bgr, det_xy, tracking_active)

                # Detection Z
                det_z = tracker_z.detect(frame_z)
                detected_z = det_z.detected
                u_z = np.nan
                v_z = np.nan
                z_mm = None

                if detected_z:
                    u_z, v_z = det_z.center

                    if use_z_model:
                        z_mm = float(z_a * v_z + z_b)

                if (
                    stereo_triangulator is not None
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

                        if camera_to_autd is not None:
                            stereo_autd = camera_to_autd.apply(stereo_cam_point)
                            stereo_autd_point = stereo_autd
                            if cfg.use_stereo_position_for_control:
                                x_mm = float(stereo_autd[0])
                                y_mm = float(stereo_autd[1])
                                z_mm = float(stereo_autd[2])

                    except Exception as e:
                        print(f"[WARN] Stereo triangulation failed: {e}")

                if do_display:
                    draw_ball_detection(frame_z_bgr, det_z, tracking_active)
                    if stereo_cam_point is not None:
                        if stereo_autd_point is not None:
                            stereo_autd_relative = stereo_autd_point - np.array(
                                [
                                    float(base_center[0]),
                                    float(base_center[1]),
                                    float(base_center[2]),
                                ],
                                dtype=float,
                            )
                            stereo_label = (
                                "ST AUTD rel: "
                                f"{stereo_autd_relative[0]:.1f}, "
                                f"{stereo_autd_relative[1]:.1f}, "
                                f"{stereo_autd_relative[2]:.1f}"
                            )
                        else:
                            stereo_label = (
                                "ST cam1: "
                                f"{stereo_cam_point.x:.1f}, "
                                f"{stereo_cam_point.y:.1f}, "
                                f"{stereo_cam_point.z:.1f}"
                            )
                        cv2.putText(
                            frame_xy_bgr,
                            stereo_label,
                            (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
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
                    target_intensity_ratio = float(cfg.static_intensity_ratio)

                current_intensity_ratio = (
                    cfg.intensity_lpf_alpha * current_intensity_ratio
                    + (1.0 - cfg.intensity_lpf_alpha) * target_intensity_ratio
                )

                current_intensity_ratio = float(
                    np.clip(
                        current_intensity_ratio,
                        cfg.static_intensity_ratio,
                        cfg.intensity_max_ratio,
                    )
                )
                
                method = "XY+Z" if (detected_xy and detected_z) else "PARTIAL"

                # Control
                if tracking_active:
                    meas = Measurement3D(
                        detected_xy=detected_xy and x_mm is not None and y_mm is not None,
                        detected_z=detected_z and z_mm is not None,
                        x=x_mm,
                        y=y_mm,
                        z=z_mm,
                        t_xy=frame_xy_time,
                        t_z=frame_z_time,
                    )

                    if meas.detected_xy or meas.detected_z:
                        # 1. RETURN_TO_HOME中は一時基準位置をゆっくりhomeへ戻す
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

                        # 2. まず通常PIDを計算
                        #    FALL_RECOVERY中は後でtargetを上書きする
                        target, debug = controller.update(meas, control_home)

                        vx_now = float(debug.vx)
                        vy_now = float(debug.vy)
                        vz_now = float(debug.vz)

                        # 3. NORMAL_HOLD / RETURN_TO_HOME中に落下検知したら
                        #    FALL_RECOVERYへ入る
                        if cfg.enable_fall_recovery:
                            if control_mode in ["NORMAL_HOLD", "RETURN_TO_HOME"]:
                                if should_enter_fall_recovery(cfg, home.z, z_mm, vz_now):
                                    control_mode = "FALL_RECOVERY"
                                    fall_recovery_start_time = time.time()

                                    print("[RECOVERY] -> FALL_RECOVERY")

                        # 4. FALL_RECOVERY中は0.5秒だけ物体を追従
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

                        # 5. RETURN_TO_HOME完了判定
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

                        set_tracking_target(
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

                logger.write(
                    frame_xy_time=frame_xy_time,
                    frame_z_time=frame_z_time,
                    u_xy=u_xy,
                    v_xy=v_xy,
                    v_z=v_z,
                    x_mm=x_mm,
                    y_mm=y_mm,
                    z_mm=z_mm,
                    home=home,
                    target=last_target,
                )
                
                if do_display:
                    exit_requested = render_tracking_window(
                        window_title,
                        frame_xy_bgr,
                        frame_z_bgr,
                        DisplayMetrics(
                            loop_fps=loop_display_fps,
                            pair_fps=new_xy_display_fps,
                            z_fps=new_z_display_fps,
                            sync_skew_sec=frame_sync_skew,
                            autd_fps=sender.display_fps,
                            method=method,
                            v_z_px=v_z,
                        ),
                        DisplayControlState(
                            tracking_active=tracking_active,
                            control_mode=control_mode,
                            demo_active=demo_active,
                            home=home,
                            origin=display_origin,
                            return_setpoint=return_setpoint,
                            radius=current_radius,
                            intensity_ratio=current_intensity_ratio,
                        ),
                    )
                    if exit_requested:
                        break

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt.")

    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")

    finally:
        print("[INFO] stopping...")

        logger.close()

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

        safe_close_camera(cam_xy)
        safe_close_camera(cam_z)

        cv2.destroyAllWindows()

        print("[INFO] stopped.")
