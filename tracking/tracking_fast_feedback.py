import os
import time
import sys
import threading
import math
import numpy as np
import cv2
import keyboard
import json

from ultralytics import YOLO

# 高速カメラで物体の位置を最速でトラッキングしつつ、別スレッドでAUTDの焦点を更新するコード

# XIMEA設定
sys.path.append(r"C:\Users\Hiroto Yoshida\Desktop\XIMEA\API\Python\v3")
try:
    from ximea import xiapi
except ImportError:
    print("[WARN] ximea モジュールなし。")
    xiapi = None

# AUTD関連
from pyautd3 import AUTD3, Controller, FociSTM, Hz, Silencer, Static
from pyautd3_link_soem import SOEM, SOEMOption, Status

# 設定
MODEL_PATH = "tracking/train/weights/best.pt"
TARGET_CLASS_NAME = "levitatedball"
CONF_THRES = 0.5
AFFINE_JSON = "affine_uv_to_xy.json"

# YOLO再捕捉の軽量化
YOLO_IMGSZ = 320
YOLO_REFIND_COOLDOWN_S = 0.3

# 古典CVトラッキング
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

# 描画間引き（1なら毎フレーム描画）
DISPLAY_EVERY_N_FRAMES = 3

# ロスト判定
LOST_MAX_FRAMES = 10

# カメラ
EXPOSURE_US = 5000

# AUTD 物理設定
POINT_NUM = 8
RADIUS = 23.5
DEFAULT_Z = 400.0  # 基準高さ

# 中心引き戻し（フィードバック）の設定 
PULL_RATIO = 0.03   # 距離に対して何％中心に寄せるか（0.03 = 3%）
MAX_PULL_MM = 1.0   # 1フレームあたりの最大移動量(mm)

# CPUスレッド
os.environ.setdefault("OMP_NUM_THREADS", "4")

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
program_running = True    # スレッド終了フラグ
autd_display_fps = 0.0    # AUTDのFPS表示用
pos_lock = threading.Lock() # 排他制御


def err_handler(slave: int, status: Status) -> None:
    print(f"slave [{slave}]: {status}")
    if status == Status.Lost():
        os._exit(-1)

def init_ximea_camera():
    if xiapi is None:
        raise RuntimeError("XIMEA API not loaded")
    cam = xiapi.Camera()
    cam.open_device()
    cam.set_imgdataformat('XI_RGB24')
    try:
        cam.enable_auto_wb()
    except AttributeError:
        pass
    cam.set_exposure(EXPOSURE_US)
    img = xiapi.Image()
    cam.start_acquisition()
    return cam, img

def find_target_class_id(model: YOLO, target_name: str) -> int | None:
    names = model.model.names
    for cid, name in names.items():
        if str(name).lower() == target_name.lower():
            return int(cid)
    return None

def yolo_refind_center(model: YOLO, frame_rgb_or_bgr: np.ndarray, target_cls_id: int | None):
    results = model.predict(source=frame_rgb_or_bgr, imgsz=YOLO_IMGSZ, conf=CONF_THRES, device="cpu", verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = None
    best_score = -1.0
    for b in boxes:
        score = float(b.conf[0].item())
        if score < CONF_THRES: continue
        cls_id = int(b.cls[0].item())
        if target_cls_id is not None and cls_id != target_cls_id: continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if score > best_score:
            best_score = score
            best = (u, v, score, (x1, y1, x2, y2))
    return best

def clamp_roi(cx, cy, size, w, h):
    size = int(max(ROI_MIN_SIZE, min(ROI_MAX_SIZE, size)))
    half = size // 2
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))
    if (x2 - x1) < size:
        if x1 == 0: x2 = min(w, x1 + size)
        elif x2 == w: x1 = max(0, x2 - size)
    if (y2 - y1) < size:
        if y1 == 0: y2 = min(h, y1 + size)
        elif y2 == h: y1 = max(0, y2 - size)
    return x1, y1, x2, y2

def track_ball_cv(frame_rgb: np.ndarray, roi_rect):
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = roi_rect
    roi = frame_rgb[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi
    if BLUR_KSIZE > 1: gray = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    
    if USE_OTSU: _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else: _, bw = cv2.threshold(gray, FIXED_THRESH, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, bw

    roi_cx, roi_cy = (x2 - x1) / 2.0, (y2 - y1) / 2.0
    best, best_score = None, -1e18
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX or area > MAX_AREA_PX: continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.6: 
            continue

        M = cv2.moments(cnt)
        if M["m00"] <= 1e-6: continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        dist2 = (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2
        score = area - 0.8 * dist2
        if score > best_score:
            best_score = score
            best = (cnt, cx, cy, area)

    if best is None: return None, bw
    cnt, cx, cy, area = best
    (xc, yc), r = cv2.minEnclosingCircle(cnt)
    return (float(x1 + xc), float(y1 + yc), float(r)), bw


# AUTD制御用スレッド関数
def autd_control_loop(autd):
    global shared_target_pos, program_running, autd_display_fps
    print("[THREAD] AUTD Control Thread Started.")
    
    # AUTD用のFPS計算タイマー
    fps_start_time = time.time()
    fps_frame_count = 0
    
    while program_running:
        tgt = None
        with pos_lock:
            if shared_target_pos is not None:
                tgt = shared_target_pos
        
        if tgt is None:
            time.sleep(0.002)
            continue

        tx, ty, tz = tgt
        center_vec = np.array([tx, ty, tz])
        
        try:
            stm = FociSTM(
                foci=(
                    center_vec + RADIUS * np.array([np.cos(theta), np.sin(theta), 0.0])
                    for theta in (
                        np.pi / 8 + 2.0 * np.pi * i / POINT_NUM
                        for i in range(POINT_NUM)
                    )
                ),
                config=100 * Hz,
            ).into_nearest()

            autd.send(stm)
            fps_frame_count += 1
            
        except Exception as e:
            pass # スレッド終了時のエラーは無視

        # FPS計算 (1秒ごとに更新)
        now = time.time()
        if now - fps_start_time >= 1.0:
            autd_display_fps = fps_frame_count / (now - fps_start_time)
            fps_start_time = now
            fps_frame_count = 0

        time.sleep(0.004)

    print("[THREAD] AUTD Control Thread Stopped.")

def load_affine_matrix(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    A = np.array(data["A_2x3"], dtype=np.float32)
    return A


def main():
    global shared_target_pos, program_running

    print(f"[INFO] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    target_cls_id = find_target_class_id(model, TARGET_CLASS_NAME)
    
    try:
        A_affine = load_affine_matrix(AFFINE_JSON)
        use_affine = True
        print(f"[INFO] Loaded affine matrix from {AFFINE_JSON}")
    except Exception as e:
        print(f"[WARN] Affine matrix load failed: {e}. Using MM_PER_PIXEL fallback.")
        A_affine = None
        use_affine = False
    
    try:
        cam, img = init_ximea_camera()
    except Exception as e:
        print(f"[ERROR] Camera Init Failed: {e}")
        return

    print("[INFO] Opening AUTD Controller...")
    try:
        with Controller.open(
            autd_arrangement,
            SOEM(err_handler=err_handler, option=SOEMOption()),
        ) as autd:
            
            autd.send(Silencer())
            autd.send(Static(intensity=int(0xFF * 0.9)))

            base_center = autd.center()
            with pos_lock:
                shared_target_pos = (base_center[0], base_center[1], DEFAULT_Z)

            t = threading.Thread(target=autd_control_loop, args=(autd,))
            t.start()

            print("[INFO] Vision loop started.")
            print("=================================================")
            print("  READY TO LEVITATE.")
            print("  Place the particle at the center.")
            print("  Press [ENTER] to START dynamic tracking.")
            print("  Press [ESC] to EXIT.")
            print("=================================================")
            
            window_name = "Tracking & Control"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            # カメラFPS計算用
            cam_fps_start_time = time.time()
            cam_fps_frame_count = 0
            cam_display_fps = 0.0

            frame_count = 0

            cam.get_image(img)
            frame0 = img.get_image_data_numpy()
            H, W = frame0.shape[:2]
            
            roi_cx, roi_cy = W // 2, H // 2
            roi_size = ROI_INIT_SIZE
            lost_count = 0
            last_yolo_t = 0.0
            method = "CV"

            tracking_active = False

            while True:
                if keyboard.is_pressed("esc"):
                    break
                
                if not tracking_active and keyboard.is_pressed("enter"):
                    tracking_active = True
                    print("[INFO] >>> TRACKING ACTIVATED <<<")
                    time.sleep(0.2)

                cam.get_image(img)
                frame = img.get_image_data_numpy()
                frame_count += 1
                cam_fps_frame_count += 1
                do_display = (frame_count % DISPLAY_EVERY_N_FRAMES == 0)
                
                frame_bgr = frame.copy() if do_display else None

                x1, y1, x2, y2 = clamp_roi(roi_cx, roi_cy, roi_size, W, H)
                roi_rect = (x1, y1, x2, y2)
                track, bw = track_ball_cv(frame, roi_rect)

                detected = False
                u, v, r = 0, 0, 0

                if track is not None:
                    u, v, r = track
                    method = "CV"
                    lost_count = 0
                    roi_cx, roi_cy = int(u), int(v)
                    desired = int(2 * (2.5 * r + ROI_MARGIN))
                    roi_size = int(0.7 * roi_size + 0.3 * desired)
                    roi_size = int(max(ROI_MIN_SIZE, min(ROI_MAX_SIZE, roi_size)))
                    detected = True
                    
                    if do_display:
                        color = (0, 255, 0) if tracking_active else (200, 200, 200)
                        cv2.circle(frame_bgr, (int(u), int(v)), int(max(2, r)), color, 2)
                        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)
                else:
                    lost_count += 1
                    roi_size = int(min(ROI_MAX_SIZE, roi_size * ROI_EXPAND_ON_LOST))
                    
                    now_mono = time.monotonic()
                    if lost_count >= LOST_MAX_FRAMES and (now_mono - last_yolo_t) >= YOLO_REFIND_COOLDOWN_S:
                        det = yolo_refind_center(model, frame, target_cls_id)
                        last_yolo_t = now_mono
                        if det:
                            u, v, score, (bx1, by1, bx2, by2) = det
                            method = "YOLO"
                            lost_count = 0
                            roi_cx, roi_cy = int(u), int(v)
                            roi_size = ROI_INIT_SIZE
                            detected = True
                            if do_display:
                                cv2.rectangle(frame_bgr, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 0, 255), 2)
                        else:
                            method = "LOST"
                    else:
                        method = "LOST"
                    
                    if do_display:
                        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # 座標計算 と 中心引き戻し制御
                if detected and tracking_active:
                    if use_affine:
                        # 1. カメラで見た物体の絶対座標を計算
                        uv_homo = np.array([[u, v, 1]], dtype=np.float32).T 
                        xy_affine = (A_affine @ uv_homo).flatten() 
                        current_x = base_center[0] + xy_affine[0]
                        current_y = base_center[1] + xy_affine[1]
                        
                        # 2. 中心 (base_center) への距離とベクトルを計算
                        dx = base_center[0] - current_x
                        dy = base_center[1] - current_y
                        dist = math.hypot(dx, dy)
                        
                        # 3. 中心に向けて少しだけ引っ張る（フィードバック制御）
                        pull_dist = min(MAX_PULL_MM, dist * PULL_RATIO)
                        
                        if dist > 0.1: # 誤差範囲(0.1mm)より外側にいる場合のみ引っ張る
                            target_x = current_x + (dx / dist) * pull_dist
                            target_y = current_y + (dy / dist) * pull_dist
                        else:
                            target_x = current_x
                            target_y = current_y
                    
                    target_z = DEFAULT_Z

                    with pos_lock:
                        shared_target_pos = (target_x, target_y, target_z)
                        
                    if do_display:
                        cv2.putText(frame_bgr, f"TGT: {target_x:.1f}, {target_y:.1f}", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # カメラFPSの更新
                now = time.time()
                if now - cam_fps_start_time >= 1.0:
                    cam_display_fps = cam_fps_frame_count / (now - cam_fps_start_time)
                    cam_fps_start_time = now
                    cam_fps_frame_count = 0

                status_text = "ACTIVE" if tracking_active else "WAIT (Press ENTER)"
                status_color = (0, 255, 0) if tracking_active else (0, 165, 255)

                if do_display:
                    # カメラとAUTD両方のFPSを表示
                    cv2.putText(frame_bgr, f"CAM FPS: {cam_display_fps:.1f} | AUTD FPS: {autd_display_fps:.1f} | {method}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.putText(frame_bgr, f"STATUS: {status_text}", (10, H - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                    cv2.imshow(window_name, frame_bgr)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            
            program_running = False
            t.join() 
            autd.send(Static(intensity=0)) 
            print("[INFO] AUTD emission stopped.")

    except Exception as e:
        print(f"[ERROR] Runtime Error: {e}")
    finally:
        program_running = False
        if 't' in locals() and t.is_alive():
            t.join()
        
        try:
            cam.stop_acquisition()
            cam.close_device()
        except: pass
        cv2.destroyAllWindows()
        print("[INFO] Finished.")

if __name__ == "__main__":
    main()