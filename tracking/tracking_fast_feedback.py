import os
import time
import sys
import threading
import numpy as np
import cv2
import keyboard

from ultralytics import YOLO

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

# YOLO再捕捉の軽量化
YOLO_IMGSZ = 320
YOLO_REFIND_COOLDOWN_S = 0.3

# CVトラッキング
USE_OTSU = True
FIXED_THRESH = 180
BLUR_KSIZE = 5
MIN_AREA_PX = 200
MAX_AREA_PX = 200000

# ROI
ROI_INIT_SIZE = 320
ROI_MIN_SIZE = 160
ROI_MAX_SIZE = 640
ROI_MARGIN = 40
ROI_EXPAND_ON_LOST = 1.25

# ロスト判定
LOST_MAX_FRAMES = 10

# カメラ
EXPOSURE_US = 5000

# AUTD物理設定
POINT_NUM = 8
RADIUS = 23.0
DEFAULT_Z = 400.0  # 基準高さ (mm)

# 座標変換設定 (要調整)
# カメラ画像の1ピクセルが現実の何mmに相当するか
# (例: 画面幅640pxが 現実の200mmなら 200/640 = 0.3125)
MM_PER_PIXEL = 0.5 

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

# AUTD制御用スレッド関数 (カメラのFPSに依存せず、最新のターゲット位置へAUTDの焦点を更新し続けるスレッド)
def autd_control_loop(autd):
    global shared_target_pos, program_running
    print("[THREAD] AUTD Control Thread Started.")
    
    last_sent_time = 0
    
    while program_running:
        # 1. 座標取得（排他制御）
        tgt = None
        with pos_lock:
            if shared_target_pos is not None:
                tgt = shared_target_pos
        
        # ターゲットがない場合は何もしない（あるいは初期位置に戻す）
        if tgt is None:
            time.sleep(0.002)
            continue

        tx, ty, tz = tgt
        
        # 2. STM生成（指定された中心座標の周りに8点を配置）
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

            # 3. 送信
            autd.send(stm)
            
        except Exception as e:
            print(f"[THREAD ERROR] {e}")

        # 送信頻度の調整（早すぎても通信が詰まるため、適度なSleepを入れる）
        # 0.004s = 4ms (約250Hz)
        time.sleep(0.004)

    print("[THREAD] AUTD Control Thread Stopped.")


def main():
    global shared_target_pos, program_running

    # 1. YOLOロード
    print(f"[INFO] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    target_cls_id = find_target_class_id(model, TARGET_CLASS_NAME)
    
    # 2. カメラ初期化
    try:
        cam, img = init_ximea_camera()
    except Exception as e:
        print(f"[ERROR] Camera Init Failed: {e}")
        return

    # 3. AUTD起動
    print("[INFO] Opening AUTD Controller...")
    try:
        with Controller.open(
            autd_arrangement,
            SOEM(err_handler=err_handler, option=SOEMOption()),
        ) as autd:
            
            # 初期化送信
            autd.send(Silencer())
            autd.send(Static(intensity=int(0xFF * 0.8)))

            # AUTD全体の中心座標を取得（基準点）
            base_center = autd.center()
            # 初期ターゲット位置をセット（スレッドが動き出すための初期値）
            with pos_lock:
                shared_target_pos = (base_center[0], base_center[1], DEFAULT_Z)

            # スレッド起動
            t = threading.Thread(target=autd_control_loop, args=(autd,))
            t.start()

            print("[INFO] Vision loop started. Press ESC to exit.")
            
            # メインループ
            window_name = "Tracking & Control"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            prev_time = time.time()
            fps = 0.0

            # 最初の1フレーム取得して画像サイズ確認
            cam.get_image(img)
            frame0 = img.get_image_data_numpy()
            H, W = frame0.shape[:2]
            
            roi_cx, roi_cy = W // 2, H // 2
            roi_size = ROI_INIT_SIZE
            lost_count = 0
            last_yolo_t = 0.0
            method = "CV"

            while True:
                if keyboard.is_pressed("esc"):
                    break

                # 1. 画像取得
                cam.get_image(img)
                frame = img.get_image_data_numpy()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # 表示用

                # 2. ROI処理 & トラッキング
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
                    
                    # 描画
                    cv2.circle(frame_bgr, (int(u), int(v)), int(max(2, r)), (0, 255, 0), 2)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)

                else:
                    # ロスト時
                    lost_count += 1
                    roi_size = int(min(ROI_MAX_SIZE, roi_size * ROI_EXPAND_ON_LOST))
                    
                    now_mono = time.monotonic()
                    if lost_count >= LOST_MAX_FRAMES and (now_mono - last_yolo_t) >= YOLO_REFIND_COOLDOWN_S:
                        det = yolo_refind_center(model, frame_bgr, target_cls_id)
                        last_yolo_t = now_mono
                        if det:
                            u, v, score, (bx1, by1, bx2, by2) = det
                            method = "YOLO"
                            lost_count = 0
                            roi_cx, roi_cy = int(u), int(v)
                            roi_size = ROI_INIT_SIZE
                            detected = True
                            cv2.rectangle(frame_bgr, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 0, 255), 2)
                        else:
                            method = "LOST"
                    else:
                        method = "LOST"
                    
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # 3. 座標変換 & スレッドへの指示更新
                if detected:
                    # [座標変換] 画像中心(W/2, H/2) を AUTD中心(base_center) と仮定
                    # カメラの取り付け向きによって符号 (+/-) を逆にする必要がある
                    
                    # ピクセル変位 (画像中心からのズレ)
                    dx_px = u - (W / 2)
                    dy_px = v - (H / 2)
                    
                    # ミリメートル変位
                    dx_mm = dx_px * MM_PER_PIXEL
                    dy_mm = dy_px * MM_PER_PIXEL

                    # AUTD座標系へ適用 (X軸, Y軸の向きに注意する)
                    # ここでは「画像右＝AUTD X正」「画像下＝AUTD Y正」と仮定
                    target_x = base_center[0] + dx_mm
                    target_y = base_center[1] + dy_mm
                    target_z = DEFAULT_Z

                    # スレッドへ渡す
                    with pos_lock:
                        shared_target_pos = (target_x, target_y, target_z)
                        
                    cv2.putText(frame_bgr, f"TGT: {target_x:.1f}, {target_y:.1f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 4. FPS計測・表示
                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0: fps = 0.9 * fps + 0.1 * (1.0 / dt)

                cv2.putText(frame_bgr, f"FPS: {fps:.1f} | {method}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow(window_name, frame_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except Exception as e:
        print(f"[ERROR] Runtime Error: {e}")
    finally:
        # 終了処理：スレッドを止める
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