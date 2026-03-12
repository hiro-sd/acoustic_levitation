import os
import time
import sys
import threading
import math
import numpy as np
import cv2
import keyboard
import json

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
# from pyautd3_link_soem import SOEM, SOEMOption, Status
from pyautd3.link.twincat import TwinCAT

# 設定
AFFINE_JSON = "affine_uv_to_xy.json"

# CVトラッキング
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

# カメラ
EXPOSURE_US = 5000

# AUTD 物理設定
POINT_NUM = 8
RADIUS = 23.5
DEFAULT_Z = 400.0  

# CPUスレッド
os.environ.setdefault("OMP_NUM_THREADS", "4")

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
program_running = True    # スレッド終了フラグ
autd_display_fps = 0.0    # AUTDのFPS表示用
pos_lock = threading.Lock() # 排他制御

# soemのエラーハンドラ
# def err_handler(slave: int, status: Status) -> None:
#     print(f"[AUTD SOEM] slave [{slave}]: {status}")
#     if status == Status.Lost():
#         os._exit(-1)

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
            best = cnt

    if best is None: return None, bw
    cnt = best
    (xc, yc), r = cv2.minEnclosingCircle(cnt)
    return (float(x1 + xc), float(y1 + yc), float(r)), bw

# AUTD制御用スレッド関数
def autd_control_loop(autd):
    global shared_target_pos, program_running, autd_display_fps
    print("[THREAD] AUTD Control Thread Started.")
    
    # AUTD用のFPS計算タイマー
    fps_start_time = time.time()
    fps_frame_count = 0
    last_sent_pos = None  
    
    while program_running:
        tgt = None
        with pos_lock:
            if shared_target_pos is not None:
                tgt = shared_target_pos
        
        if tgt is None:
            time.sleep(0.03) # 0.05
            continue

        # 重複データの連続送信を防ぐ（通信パンク対策）
        if last_sent_pos is not None:
            dx = tgt[0] - last_sent_pos[0]
            dy = tgt[1] - last_sent_pos[1]
            dz = tgt[2] - last_sent_pos[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < 0.1:
                time.sleep(0.01) 
                continue

        tx, ty, tz = tgt
        center_vec = np.array([tx, ty, tz])
        
        try:
            # 対称音場
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

            # 非対称音場
            # 正方向角リスト
            # angles_fwd = [np.pi/8 + 2.0 * np.pi * i / POINT_NUM for i in range(POINT_NUM)] # 穴の位置をずらすためにπ/8を加える
            # # 逆方向角リスト
            # angles_rev = angles_fwd[::-1][1:-1] # [::-1]で逆順にし、[1:-1]で最初と最後を除く
            # # forward + reverse の 2 周分を連結
            # angles = angles_fwd + angles_rev

            # # 円軌道上に焦点を配置するための時空間変調
            # foci = (
            #     center_vec + RADIUS * np.array([np.cos(a), np.sin(a), 0.0])
            #     for a in angles
            # )
            # stm = FociSTM(foci=foci, config=70 * Hz).into_nearest()

            autd.send(stm)
            last_sent_pos = tgt 
            fps_frame_count += 1
            
        except Exception as e:
            print(f"[AUTD Thread Error] {e}")
            # エラー時はデバイスが自動復帰するまで0.5秒待つ
            time.sleep(0.5)

        # FPS計算 (1秒ごとに更新)
        now = time.time()
        if now - fps_start_time >= 1.0:
            autd_display_fps = fps_frame_count / (now - fps_start_time)
            fps_start_time = now
            fps_frame_count = 0

        # 通信の安全マージン: 約200Hz更新
        time.sleep(0.005)

    print("[THREAD] AUTD Control Thread Stopped.")

def load_affine_matrix(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    A = np.array(data["A_2x3"], dtype=np.float32)
    return A

def main():
    global shared_target_pos, program_running

    # アフィン行列読み込み
    try:
        A_affine = load_affine_matrix(AFFINE_JSON)
        use_affine = True
        print(f"[INFO] Loaded affine matrix from {AFFINE_JSON}")
    except Exception as e:
        print(f"[WARN] Affine matrix load failed: {e}")
        A_affine = None
        use_affine = False
    
    # 1. カメラ初期化
    try:
        cam, img = init_ximea_camera()
    except Exception as e:
        print(f"[ERROR] Camera Init Failed: {e}")
        return

    # 2. AUTD起動
    print("[INFO] Opening AUTD Controller...")
    try:
        with Controller.open(
            autd_arrangement,
            # SOEM(err_handler=err_handler, option=SOEMOption()),
            TwinCAT(),
        ) as autd:
            
            # 初期化送信
            autd.send(Silencer())
            autd.send(Static(intensity=int(0xFF * 0.9)))

            # 基準座標（中心）を取得
            base_center = autd.center()
            with pos_lock:
                shared_target_pos = (base_center[0], base_center[1], DEFAULT_Z)

            # スレッド起動
            t = threading.Thread(target=autd_control_loop, args=(autd,))
            t.start()

            print("[INFO] Vision loop started.")
            print("=================================================")
            print("  READY TO LEVITATE.")
            print("  Press [ENTER] to START dynamic tracking.")
            print("  Press [ENTER] again to PAUSE (Return to center).")
            print("  Press [ESC] to EXIT and STOP ultrasound.")
            print("=================================================")
            
            window_name = "Tracking & Control" # カメラ表示をなくす場合はこれをコメントアウト
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # カメラ表示をなくす場合はこれをコメントアウト

            # カメラFPS計算用
            cam_fps_start_time = time.time()
            cam_fps_frame_count = 0
            cam_display_fps = 0.0
            frame_count = 0

            # 最初の1フレーム取得して画像サイズ確認
            cam.get_image(img)
            frame0 = img.get_image_data_numpy()
            H, W = frame0.shape[:2]
            
            roi_cx, roi_cy = W // 2, H // 2
            roi_size = ROI_INIT_SIZE
            method = "CV"

            # --- 制御用変数の初期化（whileループの直前に配置） ---
            prev_particle_x = None
            prev_particle_y = None
            prev_vx = 0.0
            prev_vy = 0.0
            prev_time_pd = time.time()

            tracking_active = False
            prev_enter_state = False

            while True:
                # キー入力確認
                if keyboard.is_pressed("esc"):
                    break
                
                # Enterキーのトグル処理
                current_enter_state = keyboard.is_pressed("enter")
                if current_enter_state and not prev_enter_state:
                    tracking_active = not tracking_active
                    
                    if tracking_active:
                        print("[INFO] >>> TRACKING ACTIVATED <<<")
                    else:
                        print("[INFO] >>> TRACKING PAUSED (Center Fixed) <<<")
                        # オフになった瞬間、ターゲットを初期位置（中心）に戻す
                        with pos_lock:
                            shared_target_pos = (base_center[0], base_center[1], DEFAULT_Z)
                prev_enter_state = current_enter_state

                # 1. 画像取得
                cam.get_image(img)
                frame = img.get_image_data_numpy()
                frame_count += 1
                cam_fps_frame_count += 1
                do_display = (frame_count % DISPLAY_EVERY_N_FRAMES == 0) # カメラ表示をなくす場合はこれをコメントアウト
                # do_display = False # カメラ表示をする場合はこれをコメントアウト

                frame_bgr = frame.copy() if do_display else None # カメラ表示をなくす場合はこれをコメントアウト
                # frame_bgr = None # カメラ表示をする場合はこれをコメントアウト

                # 2. ROI処理 & トラッキング
                x1, y1, x2, y2 = clamp_roi(roi_cx, roi_cy, roi_size, W, H)
                roi_rect = (x1, y1, x2, y2)
                track, _ = track_ball_cv(frame, roi_rect)

                detected = False
                u, v, r = 0, 0, 0

                if track is not None:
                    # 追跡成功時
                    u, v, r = track
                    method = "CV"
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
                    # ロスト時
                    # ROIを少しずつ広げて再発見を試みる
                    roi_size = int(min(ROI_MAX_SIZE, roi_size * ROI_EXPAND_ON_LOST))
                    method = "LOST"
                    
                    if do_display:
                        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # 3. 座標計算 と 中心引き戻し制御（PD制御）
                if detected and tracking_active:
                    if use_affine:
                        # 1. カメラで見た物体の絶対座標を計算
                        uv_homo = np.array([[u, v, 1]], dtype=np.float32).T 
                        xy_affine = (A_affine @ uv_homo).flatten() 
                        current_x = base_center[0] + xy_affine[0]
                        current_y = base_center[1] + xy_affine[1]

                        # 2. 現在の時刻と前回からの経過時間(dt)を計算
                        current_time_pd = time.time()
                        dt_pd = current_time_pd - prev_time_pd
                        
                        # 3. 速度の計算と平滑化（ノイズによる暴走を防ぐローパスフィルタ）
                        if prev_particle_x is not None and dt_pd > 0:
                            raw_vx = (current_x - prev_particle_x) / dt_pd
                            raw_vy = (current_y - prev_particle_y) / dt_pd
                            # 前回の速度と50%ずつ混ぜて滑らかにする
                            vx = 0.5 * prev_vx + 0.5 * raw_vx
                            vy = 0.5 * prev_vy + 0.5 * raw_vy
                        else:
                            vx, vy = 0.0, 0.0

                        # 値の更新
                        prev_particle_x = current_x
                        prev_particle_y = current_y
                        prev_vx = vx
                        prev_vy = vy
                        prev_time_pd = current_time_pd

                        # PD制御ゲイン (ここで安定性をチューニング)
                        K_p = 0.2   # [P] 中心に引き戻す強さ (0.0 なら自然な復元力のみ)
                        K_d = 0.008 # [D] 揺れを抑えるブレーキの強さ (速度に対する抵抗)

                        # 最終的に物体を留めておきたい目標位置 (AUTDの中心)
                        setpoint_x = base_center[0]
                        setpoint_y = base_center[1]

                        # 4. 新しい焦点位置の計算
                        # 目標位置 + (P制御: ズレの逆へ) + (D制御: 速度の逆へ)
                        target_x = setpoint_x + K_p * (setpoint_x - current_x) - K_d * vx
                        target_y = setpoint_y + K_p * (setpoint_y - current_y) - K_d * vy

                        # 5. 焦点を動かしすぎてボールが落ちるのを防ぐリミッター (中心から±8mm以内)
                        limit_mm = 8.0
                        target_x = np.clip(target_x, setpoint_x - limit_mm, setpoint_x + limit_mm)
                        target_y = np.clip(target_y, setpoint_y - limit_mm, setpoint_y + limit_mm)
                        
                    target_z = DEFAULT_Z

                    with pos_lock:
                        shared_target_pos = (target_x, target_y, target_z)
                        
                    if do_display:
                        cv2.putText(frame_bgr, f"TGT: {target_x:.1f}, {target_y:.1f}", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 4. カメラFPSの更新と画面表示
                now = time.time()
                if now - cam_fps_start_time >= 1.0:
                    cam_display_fps = cam_fps_frame_count / (now - cam_fps_start_time)
                    cam_fps_start_time = now
                    cam_fps_frame_count = 0

                status_text = "ACTIVE" if tracking_active else "WAIT (Press ENTER)"
                status_color = (0, 255, 0) if tracking_active else (0, 165, 255)

                if do_display:
                    cv2.putText(frame_bgr, f"CAM FPS: {cam_display_fps:.1f} | AUTD FPS: {autd_display_fps:.1f} | {method}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame_bgr, f"STATUS: {status_text}", (10, H - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    cv2.imshow(window_name, frame_bgr) # カメラ表示をなくす場合はこれをコメントアウト
                    if cv2.waitKey(1) & 0xFF == 27: # カメラ表示をなくす場合はこれをコメントアウト
                        break # カメラ表示をなくす場合はこれをコメントアウト
            
            # 終了処理
            program_running = False
            print("[INFO] Waiting for AUTD thread to close...")
            t.join(timeout=2.0)

    except Exception as e:
        print(f"[ERROR] Runtime Error: {e}")
    finally:
        program_running = False
        if 't' in locals() and t.is_alive():
            t.join(timeout=1.0)
        
        try:
            cam.stop_acquisition()
            cam.close_device()
        except: pass
        cv2.destroyAllWindows() # カメラ表示をなくす場合はこれをコメントアウト
        print("[INFO] Finished.")

if __name__ == "__main__":
    main()