import os
import time
import sys
import numpy as np
import cv2
import keyboard

from ultralytics import YOLO

# ===== XIMEA 設定 =====
sys.path.append(r"C:\Users\Hiroto Yoshida\Desktop\XIMEA\API\Python\v3")
try:
    from ximea import xiapi
except ImportError:
    print("[WARN] ximea モジュールなし。")
    xiapi = None

# ===== AUTD 関連 =====
from pyautd3 import AUTD3, Controller, FociSTM, Hz, Silencer, Static
from pyautd3_link_soem import SOEM, SOEMOption, Status


# ========== 設定 ==========
MODEL_PATH = "tracking/train/weights/best.pt"
TARGET_CLASS_NAME = "levitatedball"
CONF_THRES = 0.5

# YOLO再捕捉の軽量化（推奨）
YOLO_IMGSZ = 320                # 640→320で軽く（再捕捉なので十分なことが多い）
YOLO_REFIND_COOLDOWN_S = 0.3    # ロスト時にYOLO再実行する最小間隔

# 古典CVトラッキング（主役）
USE_OTSU = True                 # True: Otsu自動しきい値 / False: 固定しきい値
FIXED_THRESH = 180              # USE_OTSU=False のとき使用（170〜200あたりで調整）
BLUR_KSIZE = 5                  # 3 or 5
MIN_AREA_PX = 200               # 小さすぎる輪郭を除外（環境に合わせて）
MAX_AREA_PX = 200000            # 大きすぎる輪郭を除外（環境に合わせて）

# ROI（中央付近のみを見る：ここが効く）
ROI_INIT_SIZE = 320             # 初期ROIの一辺(px)
ROI_MIN_SIZE = 160
ROI_MAX_SIZE = 640
ROI_MARGIN = 40                 # 半径推定に足すマージン(px)
ROI_EXPAND_ON_LOST = 1.25       # ロスト時にROIを広げる倍率

# ロスト判定
LOST_MAX_FRAMES = 10            # 連続ロストがこれを超えたらYOLO再捕捉に頼る

# カメラ
EXPOSURE_US = 5000

# AUTD
POINT_NUM = 8
RADIUS = 23.0
X0, Y0, Z0 = 0.0, 0.0, 400.0

# CPUスレッド（お好み）
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


def err_handler(slave: int, status: Status) -> None:
    print(f"slave [{slave}]: {status}")
    if status == Status.Lost():
        os._exit(-1)


def init_ximea_camera():
    if xiapi is None:
        raise RuntimeError("XIMEA API not loaded")
    cam = xiapi.Camera()
    cam.open_device()
    cam.set_imgdataformat('XI_RGB24')  # RGB24
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
    """
    YOLOで再捕捉：中心(u,v)とbboxを返す
    """
    results = model.predict(
        source=frame_rgb_or_bgr,
        imgsz=YOLO_IMGSZ,
        conf=CONF_THRES,
        device="cpu",
        verbose=False,
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    best_score = -1.0
    for b in boxes:
        score = float(b.conf[0].item())
        if score < CONF_THRES:
            continue
        cls_id = int(b.cls[0].item())
        if target_cls_id is not None and cls_id != target_cls_id:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        if score > best_score:
            best_score = score
            best = (u, v, score, (x1, y1, x2, y2))
    return best


def clamp_roi(cx, cy, size, w, h):
    size = int(max(ROI_MIN_SIZE, min(ROI_MAX_SIZE, size)))
    half = size // 2
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(w, cx + half))
    y2 = int(min(h, cy + half))
    # 端に当たってサイズが縮むのを抑える（可能ならサイズを維持）
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


def track_ball_cv(frame_rgb: np.ndarray, roi_rect, last_center=None):
    """
    ROI内で古典CVにより白球を追跡し、(u,v,r) を返す。
    - 2値化→輪郭→外接円 or 楕円フィット
    """
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = roi_rect
    roi = frame_rgb[y1:y2, x1:x2]

    # グレースケール
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi

    # ブラー（輪郭を滑らかに）
    if BLUR_KSIZE and BLUR_KSIZE > 1:
        gray = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)

    # 白い球を抽出（高輝度を白として二値化）
    if USE_OTSU:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray, FIXED_THRESH, 255, cv2.THRESH_BINARY)

    # ノイズ除去＆穴埋め
    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, bw

    # 候補選別：面積と中心近さでスコアリング
    roi_cx = (x2 - x1) / 2.0
    roi_cy = (y2 - y1) / 2.0
    best = None
    best_score = -1e18

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX or area > MAX_AREA_PX:
            continue

        # 重心
        M = cv2.moments(cnt)
        if M["m00"] <= 1e-6:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # 中心に近いほど良い（中央付近にしかいない前提を最大活用）
        dist2 = (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2

        # 面積が大きいほど良いが、中心近さを強く優先
        score = (area) - 0.8 * dist2

        if score > best_score:
            best_score = score
            best = (cnt, cx, cy, area)

    if best is None:
        return None, bw

    cnt, cx, cy, area = best

    # 外接円（中心と半径）
    (xc, yc), r = cv2.minEnclosingCircle(cnt)

    # ROI座標→画像座標
    u = x1 + xc
    v = y1 + yc
    return (float(u), float(v), float(r)), bw


def main():
    # 1. YOLOロード（再捕捉用）
    print(f"[INFO] Loading YOLO model (fallback)...")
    model = YOLO(MODEL_PATH)
    target_cls_id = find_target_class_id(model, TARGET_CLASS_NAME)
    print(f"[INFO] target class id: {target_cls_id}")

    # 2. カメラ初期化
    try:
        cam, img = init_ximea_camera()
    except Exception as e:
        print(f"[ERROR] Camera Init Failed: {e}")
        return

    # 3. AUTD起動 & 初期送信（あなたの現行通り：一度だけ）
    print("[INFO] Opening AUTD Controller...")
    try:
        with Controller.open(
            autd_arrangement,
            SOEM(err_handler=err_handler, option=SOEMOption()),
        ) as autd:

            print("[INFO] AUTD Connected. Sending initial STM...")

            autd.send(Silencer())
            m = Static(intensity=int(0xFF * 0.8))

            center = autd.center() + np.array([X0, Y0, Z0])
            stm = FociSTM(
                foci=(
                    center + RADIUS * np.array([np.cos(theta), np.sin(theta), 0.0])
                    for theta in (
                        np.pi / 8 + 2.0 * np.pi * i / POINT_NUM
                        for i in range(POINT_NUM)
                    )
                ),
                config=100 * Hz,
            ).into_nearest()

            autd.send((m, stm))
            print("[INFO] STM sent. Starting vision loop (FAST CV tracking)...")

            # --- メインループ（トラッキング） ---
            window_name = "FAST CV Tracking (ROI) + YOLO fallback | ESC"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            prev_time = time.time()
            fps = 0.0

            # 初期ROI中心は画面中心（初フレーム取得して決める）
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
                frame = img.get_image_data_numpy()  # RGB24 (H,W,3) の想定

                # 2. ROI切り出し（中央付近のみ）
                x1, y1, x2, y2 = clamp_roi(roi_cx, roi_cy, roi_size, W, H)
                roi_rect = (x1, y1, x2, y2)

                # 3. 古典CVトラッキング
                track, bw = track_ball_cv(frame, roi_rect)

                if track is not None:
                    u, v, r = track
                    method = "CV"
                    lost_count = 0

                    # ROIを追従（中心更新）
                    roi_cx = int(round(u))
                    roi_cy = int(round(v))

                    # 半径に応じてROIサイズも調整（z変化でサイズ変動しても追える）
                    # ROI = 2*(k*r + margin) くらいが目安
                    desired = int(2 * (2.5 * r + ROI_MARGIN))
                    roi_size = int(0.7 * roi_size + 0.3 * desired)
                    roi_size = int(max(ROI_MIN_SIZE, min(ROI_MAX_SIZE, roi_size)))

                    # 表示用にBGRへ（見た目だけ）
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # 描画
                    cv2.circle(frame_bgr, (int(u), int(v)), int(max(2, r)), (0, 255, 0), 2)
                    cv2.circle(frame_bgr, (int(u), int(v)), 4, (0, 255, 0), -1)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)

                else:
                    # ロスト時：ROIを広げつつ、一定間隔でYOLO再捕捉
                    lost_count += 1
                    roi_size = int(min(ROI_MAX_SIZE, roi_size * ROI_EXPAND_ON_LOST))
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    now_mono = time.monotonic()
                    can_run_yolo = (now_mono - last_yolo_t) >= YOLO_REFIND_COOLDOWN_S

                    if lost_count >= LOST_MAX_FRAMES and can_run_yolo:
                        det = yolo_refind_center(model, frame_bgr, target_cls_id)
                        last_yolo_t = now_mono

                        if det is not None:
                            u, v, score, (bx1, by1, bx2, by2) = det
                            method = "YOLO(refind)"
                            lost_count = 0

                            roi_cx = int(round(u))
                            roi_cy = int(round(v))
                            roi_size = ROI_INIT_SIZE  # 再捕捉したら一旦初期サイズへ

                            cv2.rectangle(frame_bgr, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 0, 255), 2)
                            cv2.circle(frame_bgr, (int(u), int(v)), 5, (0, 0, 255), -1)
                        else:
                            method = "LOST"
                    else:
                        method = "LOST"

                    # ROI描画
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)

                # 4. FPS表示
                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                cv2.putText(
                    frame_bgr,
                    f"FPS: {fps:.1f} | method: {method} | ROI: {roi_size}px | lost: {lost_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow(window_name, frame_bgr)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except Exception as e:
        print(f"[ERROR] Runtime Error: {e}")
    finally:
        try:
            cam.stop_acquisition()
            cam.close_device()
        except:
            pass
        cv2.destroyAllWindows()
        print("[INFO] Finished.")


if __name__ == "__main__":
    main()
