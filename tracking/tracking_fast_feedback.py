from math import dist
import os
import time
import sys
import threading
import csv
import numpy as np
import cv2
import keyboard
import json

# 高速カメラで物体の位置をトラッキングしつつ、別スレッドでAUTDの焦点を更新するコード
#   - affine_uv_to_xy.json は「補正済みピクセル座標 (u, v)」前提
#   - よって、このコードでは毎フレーム画像を先に undistort し、
#     その補正済み画像で球検出を行う
#   - 検出した (u, v) に対して undistortPoints はもう掛けない

# XIMEA設定
sys.path.append(r"C:\Users\Hiroto Yoshida\Desktop\XIMEA\API\Python\v3")
try:
    from ximea import xiapi
except ImportError:
    print("[WARN] ximea モジュールなし。")
    xiapi = None

# AUTD関連
from pyautd3 import AUTD3, Controller, FociSTM, Hz, Silencer, Static, Focus, FocusOption, Intensity, Phase
from pyautd3.link.twincat import TwinCAT

# 設定
AFFINE_XY_JSON = "./tracking/affine_uv_to_xy.json"
AFFINE_Z_JSON = "./tracking/affine_v_to_z.json"

INTRINSIC_XY_NPZ = "./tracking/calibration/intrinsic_charuco_cam1.npz"
INTRINSIC_Z_NPZ = "./tracking/calibration/intrinsic_charuco_cam2.npz"

CAMERA_XY_SN = "43430551"
CAMERA_Z_SN = "43435351"
ROTATE_Z_FRAME = True
ROTATE_Z_CODE = cv2.ROTATE_90_CLOCKWISE

# CV
USE_OTSU = False
FIXED_THRESH = 160
BLUR_KSIZE = 5
MIN_AREA_PX = 200
MAX_AREA_PX = 200000

# ROI
ROI_INIT_SIZE = 240
ROI_MIN_SIZE = 120
ROI_MAX_SIZE = 640
ROI_MARGIN = 40
ROI_EXPAND_ON_LOST = 1.15

# 描画間引き
DISPLAY_EVERY_N_FRAMES = 3

# カメラ
EXPOSURE_US = 5000

# AUTD物理設定
POINT_NUM = 8
RADIUS = 23.5
DEFAULT_Z = 400.0
AUTD_LOOP_SLEEP_SEC = 0.001

# XY制御（予測PD）
K_P_XY = 0.15 # [P] 中心に引き戻す強さ (0.0 なら自然な復元力のみ)
K_D_XY = 0.015 # [D] 揺れを抑えるブレーキの強さ (速度に対する抵抗)
DT_PRED_XY = 0.015 # 15 ms 先を予測

# Z制御（予測P）
K_P_Z = 0.1
DT_PRED_Z = 0.015 # 15 ms 先を予測
Z_MIN = 330.0
Z_MAX = 470.0

# 円周点オフセットを事前計算
CIRCLE_OFFSETS = np.array(
    [
        [
            RADIUS * np.cos(np.pi / 8 + 2.0 * np.pi * i / POINT_NUM),
            RADIUS * np.sin(np.pi / 8 + 2.0 * np.pi * i / POINT_NUM),
            0.0,
        ]
        for i in range(POINT_NUM)
    ],
    dtype=np.float32,
)

# ログ設定
LOG_ENABLED = True
LOG_CSV_PATH = "./tracking/stability_log.csv"
LOG_DURATION_SEC = 30.0
LOG_TRIGGER_KEY = "l"

# CPUスレッド
# os.environ.setdefault("OMP_NUM_THREADS", "4")

# AUTD配置（3行×3列）
autd_arrangement = [
    AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[0.0, (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[0.0, 2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, 2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[2 * AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[2 * AUTD3.DEVICE_WIDTH, (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[2 * AUTD3.DEVICE_WIDTH, 2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
]

# スレッド共有変数
shared_target_pos = None  # (x, y, z) [mm]
program_running = True
autd_display_fps = 0.0
pos_lock = threading.Lock()
shared_target_seq = 0

# 最新フレーム共有用
latest_frame_xy = None
latest_frame_z = None
latest_frame_xy_time = 0.0
latest_frame_z_time = 0.0
frame_xy_lock = threading.Lock()
frame_z_lock = threading.Lock()

def init_ximea_camera(camera_sn: str, role: str):
    if xiapi is None:
        raise RuntimeError("XIMEA API not loaded")

    cam = xiapi.Camera()
    cam.open_device_by_SN(camera_sn)
    print(f"[INFO] XIMEA camera opened ({role}) SN={camera_sn}.")

    cam.set_imgdataformat("XI_RGB24")

    try:
        cam.enable_auto_wb()
    except AttributeError:
        pass

    if hasattr(cam, "disable_aeag"):
        try:
            cam.disable_aeag()
        except Exception:
            pass

    cam.set_exposure(EXPOSURE_US)
    print(f"[INFO] Exposure set to {EXPOSURE_US} us")

    try:
        framerate_min = cam.get_framerate_minimum()
        framerate_max = cam.get_framerate_maximum()
        framerate_inc = cam.get_framerate_increment()

        target_framerate = framerate_max
        if framerate_inc > 0:
            target_framerate = np.floor(framerate_max / framerate_inc) * framerate_inc
            if target_framerate < framerate_min:
                target_framerate = framerate_max

        cam.set_framerate(target_framerate)
        applied_framerate = cam.get_framerate()
        print(
            f"[INFO] Frame rate set ({role}): "
            f"min={framerate_min:.2f}, max={framerate_max:.2f}, "
            f"inc={framerate_inc:.2f}, applied={applied_framerate:.2f} fps"
        )
    except AttributeError:
        print(f"[WARN] Framerate API unavailable ({role}); skipped frame rate configuration.")
    except Exception as e:
        print(f"[WARN] Failed to configure framerate ({role}): {e}")

    img = xiapi.Image()
    cam.start_acquisition()
    print(f"[INFO] XIMEA acquisition started ({role}).")
    return cam, img


def load_intrinsic(npz_path: str, role: str):
    data = np.load(npz_path, allow_pickle=True)
    if "camera_matrix" in data and "dist_coeffs" in data:
        mtx = data["camera_matrix"].astype(np.float32)
        dist = data["dist_coeffs"].astype(np.float32)
    elif "mtx" in data and "dist" in data:
        mtx = data["mtx"].astype(np.float32)
        dist = data["dist"].astype(np.float32)
    else:
        keys = ", ".join(data.files)
        raise KeyError(
            f"Unsupported intrinsic keys for {role}: {keys}. "
            "Expected (camera_matrix, dist_coeffs) or (mtx, dist)."
        )
    print(f"[INFO] Loaded intrinsic parameters ({role}) from {npz_path}")
    return mtx, dist


def rotate_frame_if_needed(frame: np.ndarray, do_rotate: bool, rotate_code: int) -> np.ndarray:
    if not do_rotate:
        return frame
    return cv2.rotate(frame, rotate_code)


def load_z_model(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Z model JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    a = float(data["a"])
    b = float(data["b"])
    print(f"[INFO] Loaded z model from {json_path}: z = {a:.6f} * v + {b:.6f}")
    return a, b


def clamp_roi(cx, cy, size, w, h):
    size = int(max(ROI_MIN_SIZE, min(ROI_MAX_SIZE, size)))
    half = size // 2
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))

    if (x2 - x1) < size:
        if x1 == 0:
            x2 = min(w, x1 + size)
        elif x2 == w:
            x1 = max(0, x2 - size)

    if (y2 - y1) < size:
        if y1 == 0:
            y2 = min(h, y1 + size)
        elif y2 == h:
            y1 = max(0, y2 - size)

    return x1, y1, x2, y2


def build_undistort_maps(frame_shape, mtx: np.ndarray, dist: np.ndarray):
    """
    毎フレーム cv2.undistort() する代わりに、
    最初に1回だけ remap 用テーブルを作る。
    """
    h, w = frame_shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(
        mtx,
        dist,
        None,
        mtx,
        (w, h),
        cv2.CV_16SC2,
    )
    return map1, map2


def undistort_frame(frame: np.ndarray, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
    """
    事前計算済み remap テーブルで高速に歪み補正する。
    """
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


def track_ball_cv(frame_rgb: np.ndarray, roi_rect):
    x1, y1, x2, y2 = roi_rect
    roi = frame_rgb[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi
    if BLUR_KSIZE > 1:
        gray = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)

    if USE_OTSU:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray, FIXED_THRESH, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, bw

    roi_cx, roi_cy = (x2 - x1) / 2.0, (y2 - y1) / 2.0
    best = None
    best_score = -1e18

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX or area > MAX_AREA_PX:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.6:
            continue

        M = cv2.moments(cnt)
        if M["m00"] <= 1e-6:
            continue

        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        dist2 = (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2
        score = area - 0.8 * dist2

        if score > best_score:
            best_score = score
            best = cnt

    if best is None:
        return None, bw

    (xc, yc), r = cv2.minEnclosingCircle(best)
    return (float(x1 + xc), float(y1 + yc), float(r)), bw


def camera_capture_loop(cam, img, role, use_undistort, map1, map2, do_rotate=False, rotate_code=None):
    global program_running
    global latest_frame_xy, latest_frame_z
    global latest_frame_xy_time, latest_frame_z_time

    while program_running:
        try:
            cam.get_image(img)
            frame = img.get_image_data_numpy()

            if use_undistort:
                frame = undistort_frame(frame, map1, map2)

            if do_rotate:
                frame = rotate_frame_if_needed(frame, True, rotate_code)

            now_t = time.time()

            if role == "xy":
                with frame_xy_lock:
                    latest_frame_xy = frame
                    latest_frame_xy_time = now_t
            else:
                with frame_z_lock:
                    latest_frame_z = frame
                    latest_frame_z_time = now_t

        except Exception as e:
            print(f"[CAM {role} Thread Error] {e}")
            time.sleep(0.01)


def autd_control_loop(autd):
    global shared_target_pos, shared_target_seq, program_running, autd_display_fps
    print("[THREAD] AUTD Control Thread Started.")

    fps_start_time = time.perf_counter()
    fps_frame_count = 0
    last_seq = -1

    # 計測用
    send_time_ema_ms = 0.0
    build_time_ema_ms = 0.0

    while program_running:
        tgt = None
        seq = last_seq

        # 最新ターゲットだけ取得
        with pos_lock:
            if shared_target_pos is not None:
                tgt = shared_target_pos
                seq = shared_target_seq

        # 新しいターゲットがまだ無ければ少し待つ
        if tgt is None or seq == last_seq:
            time.sleep(AUTD_LOOP_SLEEP_SEC)
            continue

        tx, ty, tz = tgt

        try:
            t0 = time.perf_counter()

            # 中心だけ更新
            center_vec = np.array([tx, ty, tz], dtype=np.float32)

            # 事前計算済みオフセットを加える
            foci = center_vec[None, :] + CIRCLE_OFFSETS

            t1 = time.perf_counter()

            # generator ではなく list で渡す
            stm = FociSTM(
                foci=[foci[i] for i in range(foci.shape[0])],
                config=100 * Hz,
            ).into_nearest()

            t2 = time.perf_counter()

            autd.send(stm)

            t3 = time.perf_counter()

            build_ms = (t2 - t0) * 1000.0
            send_ms = (t3 - t2) * 1000.0
            build_time_ema_ms = 0.9 * build_time_ema_ms + 0.1 * build_ms
            send_time_ema_ms = 0.9 * send_time_ema_ms + 0.1 * send_ms

            last_seq = seq
            fps_frame_count += 1

        except Exception as e:
            print(f"[AUTD Thread Error] {e}")
            time.sleep(0.01)

        now = time.perf_counter()
        if now - fps_start_time >= 1.0:
            autd_display_fps = fps_frame_count / (now - fps_start_time)
            print(
                f"[AUTD] fps={autd_display_fps:.1f}, "
                f"build_ema={build_time_ema_ms:.2f} ms, "
                f"send_ema={send_time_ema_ms:.2f} ms"
            )
            fps_start_time = now
            fps_frame_count = 0

    print("[THREAD] AUTD Control Thread Stopped.")


def load_affine_matrix(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    A = np.array(data["A_2x3"], dtype=np.float32)
    uv_type = data.get("input_uv_type", "unknown")
    return A, uv_type

def set_shared_target_pos(x: float, y: float, z: float):
    global shared_target_pos, shared_target_seq
    with pos_lock:
        shared_target_pos = (float(x), float(y), float(z))
        shared_target_seq += 1

# メイン
def main():
    global shared_target_pos, program_running
    global latest_frame_xy, latest_frame_z
    # global latest_frame_xy_time, latest_frame_z_time

    # affine 読み込み
    try:
        A_affine, affine_uv_type = load_affine_matrix(AFFINE_XY_JSON)
        use_affine = True
        print(f"[INFO] Loaded affine matrix from {AFFINE_XY_JSON}")
        print(f"[INFO] affine input_uv_type = {affine_uv_type}")
    except Exception as e:
        print(f"[WARN] Affine matrix load failed: {e}")
        A_affine = None
        affine_uv_type = "unknown"
        use_affine = False

    # intrinsic 読み込み
    try:
        mtx_cam_xy, dist_cam_xy = load_intrinsic(INTRINSIC_XY_NPZ, "xy")
        mtx_cam_z, dist_cam_z = load_intrinsic(INTRINSIC_Z_NPZ, "z")
        use_undistort = True
        print("[INFO] Vision pipeline: undistort each camera frame, then detect.")
    except Exception as e:
        print(f"[WARN] Intrinsic parameters load failed: {e}")
        mtx_cam_xy = None
        dist_cam_xy = None
        mtx_cam_z = None
        dist_cam_z = None
        use_undistort = False

    # z変換モデル読み込み
    try:
        z_a, z_b = load_z_model(AFFINE_Z_JSON)
        use_z_model = True
    except Exception as e:
        print(f"[ERROR] z model load failed: {e}")
        print("[ERROR] 先に coordinate_transformation_z.py と fit_affine_from_csv_z.py を実行して、affine_v_to_z.json を生成してください。")
        return

    # カメラ初期化（2台）
    try:
        cam_xy, img_xy = init_ximea_camera(CAMERA_XY_SN, "xy")
        cam_z, img_z = init_ximea_camera(CAMERA_Z_SN, "z")
    except Exception as e:
        print(f"[ERROR] Camera Init Failed: {e}")
        return

    print("[INFO] Opening AUTD Controller...")
    try:
        with Controller.open(
            autd_arrangement,
            TwinCAT(),
        ) as autd:

            autd.send(Silencer())
            autd.send(Static(intensity=int(0xFF * 0.9)))

            base_center = autd.center()
            set_shared_target_pos(base_center[0], base_center[1], DEFAULT_Z)

            t = threading.Thread(target=autd_control_loop, args=(autd,))
            t.start()

            print("[INFO] Vision loop started.")
            print("=================================================")
            print("  READY TO LEVITATE.")
            print("  Press [ENTER] to START dynamic tracking.")
            print("  Press [ENTER] again to PAUSE (Return to center).")
            print(f"  Press [{LOG_TRIGGER_KEY.upper()}] to START {LOG_DURATION_SEC:.0f}s logging in current mode.")
            print("  Press [ESC] to EXIT and STOP ultrasound.")
            print("=================================================")

            window_tile = "Tracking"
            cv2.namedWindow(window_tile, cv2.WINDOW_NORMAL)

            cam_fps_start_time = time.time()
            cam_fps_frame_count = 0
            cam_display_fps = 0.0
            frame_count = 0

            # 最初の1フレーム取得して画像サイズ確認
            cam_xy.get_image(img_xy)
            frame0_raw_xy = img_xy.get_image_data_numpy()

            cam_z.get_image(img_z)
            frame0_raw_z = img_z.get_image_data_numpy()

            if use_undistort:
                map1_xy, map2_xy = build_undistort_maps(frame0_raw_xy.shape, mtx_cam_xy, dist_cam_xy)
                map1_z, map2_z = build_undistort_maps(frame0_raw_z.shape, mtx_cam_z, dist_cam_z)

                frame0 = undistort_frame(frame0_raw_xy, map1_xy, map2_xy)
                frame0_z = undistort_frame(frame0_raw_z, map1_z, map2_z)
            else:
                map1_xy = map2_xy = None
                map1_z = map2_z = None
                frame0 = frame0_raw_xy
                frame0_z = frame0_raw_z

            frame0_z = rotate_frame_if_needed(frame0_z, ROTATE_Z_FRAME, ROTATE_Z_CODE)

            H_xy, W_xy = frame0.shape[:2]
            H_z, W_z = frame0_z.shape[:2]

            # 最新フレームを初期投入
            with frame_xy_lock:
                latest_frame_xy = frame0.copy()
                # latest_frame_xy_time = time.time()

            with frame_z_lock:
                latest_frame_z = frame0_z.copy()
                # latest_frame_z_time = time.time()

            # カメラ取得スレッド開始
            t_cam_xy = threading.Thread(
                target=camera_capture_loop,
                args=(cam_xy, img_xy, "xy", use_undistort, map1_xy, map2_xy, False, None),
                daemon=True,
            )
            t_cam_z = threading.Thread(
                target=camera_capture_loop,
                args=(cam_z, img_z, "z", use_undistort, map1_z, map2_z, ROTATE_Z_FRAME, ROTATE_Z_CODE),
                daemon=True,
            )
            t_cam_xy.start()
            t_cam_z.start()

            roi_cx_xy, roi_cy_xy = W_xy // 2, H_xy // 2
            roi_size_xy = ROI_INIT_SIZE
            roi_cx_z, roi_cy_z = W_z // 2, H_z // 2
            roi_size_z = ROI_INIT_SIZE
            method = "CV"

            tracking_active = False
            prev_enter_state = False
            prev_log_trigger_state = False

            # XY 予測PD用
            prev_particle_x = None
            prev_particle_y = None
            prev_vx = 0.0
            prev_vy = 0.0
            prev_time_ctrl = time.time()

            # Z 速度予測用
            prev_particle_z = None
            prev_vz = 0.0

            # Z見失い時保持用
            last_valid_target_z = DEFAULT_Z

            # ログ制御
            log_file_initialized = False
            log_session_active = False
            log_session_end_time = 0.0
            log_session_mode = ""
            log_file = None
            log_writer = None

            while True:
                if keyboard.is_pressed("esc"):
                    break

                # Enterで追従ON/OFF
                current_enter_state = keyboard.is_pressed("enter")
                if current_enter_state and not prev_enter_state:
                    tracking_active = not tracking_active

                    if tracking_active:
                        print("[INFO] >>> TRACKING ACTIVATED <<<")
                        prev_particle_x = None
                        prev_particle_y = None
                        prev_particle_z = None
                        prev_vx = 0.0
                        prev_vy = 0.0
                        prev_vz = 0.0
                        prev_time_ctrl = time.time()
                    else:
                        print("[INFO] >>> TRACKING PAUSED (Center Fixed) <<<")
                        set_shared_target_pos(base_center[0], base_center[1], DEFAULT_Z)
                prev_enter_state = current_enter_state

                # ログ開始
                current_log_trigger_state = keyboard.is_pressed(LOG_TRIGGER_KEY)
                if LOG_ENABLED and current_log_trigger_state and not prev_log_trigger_state:
                    if log_session_active:
                        remain = max(0.0, log_session_end_time - time.time())
                        print(f"[LOG] Recording in progress ({log_session_mode}), remaining {remain:.1f}s")
                    else:
                        if not log_file_initialized:
                            with open(LOG_CSV_PATH, "w", newline="", encoding="utf-8") as f_init:
                                w_init = csv.writer(f_init)
                                w_init.writerow([
                                    "timestamp", "mode",
                                    "u_xy_px", "v_xy_px", "v_z_px",
                                    "x_mm", "y_mm", "z_mm",
                                    "center_x_mm", "center_y_mm", "center_z_mm",
                                    "autd_target_x_mm", "autd_target_y_mm", "autd_target_z_mm"
                                ])
                            log_file_initialized = True
                            print(f"[LOG] Log file initialized: {LOG_CSV_PATH}")

                        log_file = open(LOG_CSV_PATH, "a", newline="", encoding="utf-8")
                        log_writer = csv.writer(log_file)
                        log_session_active = True
                        log_session_mode = "PD" if tracking_active else "FIXED"
                        now_log = time.time()
                        log_session_end_time = now_log + LOG_DURATION_SEC
                        print(f"[LOG] START {log_session_mode} logging for {LOG_DURATION_SEC:.0f}s")
                prev_log_trigger_state = current_log_trigger_state

                # ログ自動停止
                if log_session_active and time.time() >= log_session_end_time:
                    log_session_active = False
                    if log_file is not None:
                        log_file.close()
                        log_file = None
                        log_writer = None
                    print(f"[LOG] DONE {log_session_mode} logging ({LOG_DURATION_SEC:.0f}s)")

                # 1. 最新フレーム取得（2カメラは別スレッドで取得済み）
                with frame_xy_lock:
                    frame_xy = None if latest_frame_xy is None else latest_frame_xy.copy()

                with frame_z_lock:
                    frame_z = None if latest_frame_z is None else latest_frame_z.copy()

                if frame_xy is None or frame_z is None:
                    time.sleep(0.001)
                    continue

                frame_count += 1
                cam_fps_frame_count += 1
                do_display = (frame_count % DISPLAY_EVERY_N_FRAMES == 0)

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

                # 2. ROI処理 & トラッキング（XYカメラ）
                x1_xy, y1_xy, x2_xy, y2_xy = clamp_roi(roi_cx_xy, roi_cy_xy, roi_size_xy, W_xy, H_xy)
                track_xy, _ = track_ball_cv(frame_xy, (x1_xy, y1_xy, x2_xy, y2_xy))

                # 2b. ROI処理 & トラッキング（Zカメラ）
                x1_z, y1_z, x2_z, y2_z = clamp_roi(roi_cx_z, roi_cy_z, roi_size_z, W_z, H_z)
                track_z, _ = track_ball_cv(frame_z, (x1_z, y1_z, x2_z, y2_z))

                detected_xy = False
                detected_z = False
                u_xy, v_xy, r_xy = 0.0, 0.0, 0.0
                u_z, v_z, r_z = 0.0, 0.0, 0.0

                if track_xy is not None:
                    u_xy, v_xy, r_xy = track_xy
                    roi_cx_xy, roi_cy_xy = int(u_xy), int(v_xy)
                    desired_xy = int(2 * (2.5 * r_xy + ROI_MARGIN))
                    roi_size_xy = int(0.7 * roi_size_xy + 0.3 * desired_xy)
                    roi_size_xy = int(max(ROI_MIN_SIZE, min(ROI_MAX_SIZE, roi_size_xy)))
                    detected_xy = True

                    if do_display:
                        color = (0, 255, 0) if tracking_active else (200, 200, 200)
                        cv2.circle(frame_xy_bgr, (int(u_xy), int(v_xy)), int(max(2, r_xy)), color, 2)
                        cv2.rectangle(frame_xy_bgr, (x1_xy, y1_xy), (x2_xy, y2_xy), (255, 255, 0), 2)
                else:
                    roi_size_xy = int(min(ROI_MAX_SIZE, roi_size_xy * ROI_EXPAND_ON_LOST))

                    if do_display:
                        cv2.rectangle(frame_xy_bgr, (x1_xy, y1_xy), (x2_xy, y2_xy), (0, 255, 255), 2)

                if track_z is not None:
                    u_z, v_z, r_z = track_z
                    roi_cx_z, roi_cy_z = int(u_z), int(v_z)
                    desired_z = int(2 * (2.5 * r_z + ROI_MARGIN))
                    roi_size_z = int(0.7 * roi_size_z + 0.3 * desired_z)
                    roi_size_z = int(max(ROI_MIN_SIZE, min(ROI_MAX_SIZE, roi_size_z)))
                    detected_z = True

                    if do_display:
                        color = (0, 255, 0) if tracking_active else (200, 200, 200)
                        cv2.circle(frame_z_bgr, (int(u_z), int(v_z)), int(max(2, r_z)), color, 2)
                        cv2.rectangle(frame_z_bgr, (x1_z, y1_z), (x2_z, y2_z), (255, 255, 0), 2)
                else:
                    roi_size_z = int(min(ROI_MAX_SIZE, roi_size_z * ROI_EXPAND_ON_LOST))

                    if do_display:
                        cv2.rectangle(frame_z_bgr, (x1_z, y1_z), (x2_z, y2_z), (0, 255, 255), 2)

                method = "XY+Z" if (detected_xy and detected_z) else "PARTIAL"

                # 3. 制御（XY: 予測PD / Z: 独立P + 見失い時保持）
                loop_target_x = float(base_center[0])
                loop_target_y = float(base_center[1])
                loop_target_z = float(last_valid_target_z)

                if tracking_active:
                    target_x = base_center[0]
                    target_y = base_center[1]
                    target_z = last_valid_target_z

                    now_ctrl = time.time()
                    dt_ctrl = max(1e-3, now_ctrl - prev_time_ctrl)
                    prev_time_ctrl = now_ctrl

                    # XY: 予測PD
                    if use_affine and detected_xy:
                        uv_homo = np.array([[u_xy, v_xy, 1.0]], dtype=np.float32).T
                        xy_affine = (A_affine @ uv_homo).flatten()
                        current_x = base_center[0] + xy_affine[0]
                        current_y = base_center[1] + xy_affine[1]

                        if prev_particle_x is not None and dt_ctrl > 0:
                            raw_vx = (current_x - prev_particle_x) / dt_ctrl
                            raw_vy = (current_y - prev_particle_y) / dt_ctrl
                            vx = 0.5 * prev_vx + 0.5 * raw_vx
                            vy = 0.5 * prev_vy + 0.5 * raw_vy
                        else:
                            vx, vy = 0.0, 0.0

                        prev_particle_x = current_x
                        prev_particle_y = current_y
                        prev_vx = vx
                        prev_vy = vy

                        x_pred = current_x + vx * DT_PRED_XY
                        y_pred = current_y + vy * DT_PRED_XY

                        # 最終的に物体を留めておきたい目標位置 (AUTDの中心)
                        setpoint_x = base_center[0]
                        setpoint_y = base_center[1]

                        # 目標位置 + (P制御: ズレの逆へ) + (D制御: 速度の逆へ)    
                        target_x = setpoint_x + K_P_XY * (setpoint_x - x_pred) - K_D_XY * vx
                        target_y = setpoint_y + K_P_XY * (setpoint_y - y_pred) - K_D_XY * vy

                    # Z: 速度予測P（速度はxyと同様に平滑化）
                    if use_z_model and detected_z:
                        current_z = z_a * v_z + z_b
                        setpoint_z = DEFAULT_Z

                        if prev_particle_z is not None and dt_ctrl > 0:
                            raw_vz = (current_z - prev_particle_z) / dt_ctrl
                            vz = 0.5 * prev_vz + 0.5 * raw_vz
                        else:
                            vz = 0.0

                        prev_particle_z = current_z
                        prev_vz = vz

                        z_pred = current_z + vz * DT_PRED_Z

                        # 焦点を高くすると物体も高くなる系
                        target_z = setpoint_z + K_P_Z * (setpoint_z - z_pred)
                        target_z = float(np.clip(target_z, Z_MIN, Z_MAX))

                        # 検出できたときだけ更新して保持
                        last_valid_target_z = target_z

                    set_shared_target_pos(target_x, target_y, target_z)
                    loop_target_x = float(target_x)
                    loop_target_y = float(target_y)
                    loop_target_z = float(target_z)

                    if do_display:
                        cv2.putText(
                            frame_xy_bgr,
                            f"TGT XY: {target_x:.1f}, {target_y:.1f}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )
                        cv2.putText(
                            frame_z_bgr,
                            f"TGT Z: {target_z:.1f}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

                # 4. ログ（制御計算後のAUTD目標中心を毎ループ記録）
                if log_session_active and log_writer is not None:
                    x_log = np.nan
                    y_log = np.nan
                    z_log = np.nan

                    if use_affine and detected_xy:
                        uv_homo_log = np.array([[u_xy, v_xy, 1.0]], dtype=np.float32).T
                        xy_log = (A_affine @ uv_homo_log).flatten()
                        x_log = float(base_center[0] + xy_log[0])
                        y_log = float(base_center[1] + xy_log[1])

                    if use_z_model and detected_z:
                        z_log = float(z_a * v_z + z_b)

                    log_writer.writerow([
                        f"{time.time():.4f}",
                        log_session_mode,
                        f"{u_xy:.2f}", f"{v_xy:.2f}", f"{v_z:.2f}",
                        f"{x_log:.3f}" if np.isfinite(x_log) else "",
                        f"{y_log:.3f}" if np.isfinite(y_log) else "",
                        f"{z_log:.3f}" if np.isfinite(z_log) else "",
                        f"{base_center[0]:.3f}", f"{base_center[1]:.3f}", f"{DEFAULT_Z:.3f}",
                        f"{loop_target_x:.3f}", f"{loop_target_y:.3f}", f"{loop_target_z:.3f}",
                    ])

                # 5. FPS更新 & 表示
                now = time.time()
                if now - cam_fps_start_time >= 1.0:
                    cam_display_fps = cam_fps_frame_count / (now - cam_fps_start_time)
                    cam_fps_start_time = now
                    cam_fps_frame_count = 0

                status_text = "ACTIVE" if tracking_active else "WAIT (Press ENTER)"
                status_color = (0, 255, 0) if tracking_active else (0, 165, 255)

                if do_display:
                    cv2.putText(
                        frame_xy_bgr,
                        f"CAM FPS: {cam_display_fps:.1f} | AUTD FPS: {autd_display_fps:.1f} | {method}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
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
                        f"CAM FPS: {cam_display_fps:.1f} | v_z={v_z:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
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
                    if frame_xy_bgr.shape[0] != frame_z_bgr.shape[0]:
                        target_h = frame_xy_bgr.shape[0]
                        scale = target_h / frame_z_bgr.shape[0]
                        resized_w = max(1, int(frame_z_bgr.shape[1] * scale))
                        frame_z_disp = cv2.resize(frame_z_bgr, (resized_w, target_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        frame_z_disp = frame_z_bgr

                    tiled = np.hstack([frame_xy_bgr, frame_z_disp])
                    split_x = frame_xy_bgr.shape[1]
                    cv2.line(tiled, (split_x, 0), (split_x, tiled.shape[0] - 1), (255, 255, 255), 1)
                    cv2.putText(tiled, "XY", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(tiled, "Z", (split_x + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    cv2.imshow(window_tile, tiled)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            # 終了処理
            program_running = False
            if log_file is not None:
                log_file.close()
                print(f"[LOG] Saved (closed session): {LOG_CSV_PATH}")

            print("[INFO] Waiting for AUTD thread to close...")
            t.join(timeout=2.0)

    except Exception as e:
        print(f"[ERROR] Runtime Error: {e}")

    finally:
        program_running = False

        if "t" in locals() and t.is_alive():
            t.join(timeout=1.0)

        if "t_cam_xy" in locals() and t_cam_xy.is_alive():
            t_cam_xy.join(timeout=1.0)

        if "t_cam_z" in locals() and t_cam_z.is_alive():
            t_cam_z.join(timeout=1.0)

        if "log_file" in locals() and log_file is not None:
            try:
                log_file.close()
                print(f"[LOG] Saved (on exit): {LOG_CSV_PATH}")
            except Exception:
                pass

        for cam_obj in (locals().get("cam_xy"), locals().get("cam_z")):
            if cam_obj is None:
                continue
            try:
                cam_obj.stop_acquisition()
            except Exception:
                pass
            try:
                cam_obj.close_device()
            except Exception:
                pass

        cv2.destroyAllWindows()
        print("[INFO] Finished.")


if __name__ == "__main__":
    main()