import os
import time
import sys
import csv
import numpy as np
import cv2
import keyboard

from ultralytics import YOLO

# yoloで物体検出し、(u,v)と(x,y)の対応点をキャリブレーション用に保存するスクリプト

# ===== XIMEA のパス設定（あなたの環境に合わせて） =====
sys.path.append(r"C:\Users\Hiroto Yoshida\Desktop\XIMEA\API\Python\v3")
from ximea import xiapi

# ===== AUTD 関連 =====
from pyautd3 import AUTD3, Controller, FociSTM, Hz, Silencer, Static
# from pyautd3_link_soem import SOEM, SOEMOption, Status
from pyautd3.link.twincat import TwinCAT

# ===================== 設定 =====================
MODEL_PATH = "tracking/train/weights/best.pt"   # ←独自モデルならここをあなたの.ptに
IMG_SIZE = 640
CONF_THRES = 0.5

EXPOSURE_US = 5000

# AUTD（あなたの元コードに合わせて）
POINT_NUM = 8
RADIUS = 23.5
X0, Y0, Z0 = -35.0, 0.0, 400.0

# キャリブレーション点保存
CSV_PATH = "calib_points_uv_xy.csv"
CAPTURE_SECONDS = 10.0
MIN_SAMPLES_TO_SAVE = 30  # 5秒間でこれ未満しか取れなかったら保存しない（目安）

os.environ.setdefault("OMP_NUM_THREADS", "4")


# AUTDの配置（あなたの元コードのまま）
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


# def err_handler(slave: int, status: Status) -> None:
#     print(f"slave [{slave}]: {status}")
#     if status == Status.Lost():
#         os._exit(-1)


def init_ximea_camera():
    cam = xiapi.Camera()
    cam.open_device()
    print("[INFO] XIMEA camera opened.")

    cam.set_imgdataformat('XI_RGB24')

    # 自動ホワイトバランス（使えるなら）
    try:
        cam.enable_auto_wb()
    except AttributeError:
        try:
            cam.set_param('auto_wb', 1)
        except Exception:
            print("[WARN] auto white balance could not be enabled.")

    cam.set_exposure(EXPOSURE_US)
    print(f"[INFO] Exposure set to {EXPOSURE_US} us")

    img = xiapi.Image()
    cam.start_acquisition()
    print("[INFO] XIMEA acquisition started.")
    return cam, img


def ensure_csv_header(path: str):
    """CSVがなければヘッダ行を作る"""
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp",
            "u_mean", "v_mean",
            "n_samples",
            "x_cmd_mm", "y_cmd_mm", "z_cmd_mm",
        ])
    print(f"[INFO] Created CSV: {path}")


def pick_best_detection(results, conf_thres: float):
    """
    YOLO結果から「一番信頼度が高いbbox」の中心(u,v)を返す。
    戻り値: (u, v, score) or None
    """
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    best_score = -1.0
    for box in boxes:
        score = float(box.conf[0].item())
        if score < conf_thres:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        if score > best_score:
            best_score = score
            best = (u, v, score)
    return best


def main():
    ensure_csv_header(CSV_PATH)

    print(f"[INFO] Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cam, img = init_ximea_camera()

    with Controller.open(
        autd_arrangement,
        # SOEM(err_handler=err_handler, option=SOEMOption()),
        TwinCAT(),
    ) as autd:

        autd.send(Silencer())
        m = Static(intensity=int(0xFF * 0.9))

        window_name = "XIMEA + YOLO + AUTD | 'c': capture 5s mean (u,v) | ESC: quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        prev_time = time.time()
        fps = 0.0

        # いま送っている中心オフセット（これをCSVに一緒に保存）
        x_cmd, y_cmd, z_cmd = X0, Y0, Z0

        # --- 5秒平均キャプチャ状態 ---
        capturing = False
        capture_end_t = 0.0
        uv_samples = []  # list[(u,v)]

        # AUTD送信（円軌道STM）
        center = autd.center() + np.array([x_cmd, y_cmd, z_cmd])
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

        try:
            while True:

                # XIMEAフレーム取得
                cam.get_image(img)
                frame = img.get_image_data_numpy()

                if frame.ndim == 2:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame

                # YOLO推論
                results = model.predict(
                    source=frame_bgr,
                    imgsz=IMG_SIZE,
                    conf=CONF_THRES,
                    device="cpu",
                    verbose=False
                )

                annotated = frame_bgr.copy()

                # 最良の検出中心 (u,v)
                best = pick_best_detection(results, CONF_THRES)
                if best is not None:
                    u, v, score = best

                    # 中心点を描画
                    cv2.circle(annotated, (int(u), int(v)), 6, (0, 255, 0), -1)
                    cv2.putText(
                        annotated,
                        f"u={u:.1f}, v={v:.1f}, conf={score:.2f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    # キャプチャ中ならサンプル追加
                    if capturing:
                        uv_samples.append((u, v))

                # FPS表示
                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                # キー判定（OpenCV側）
                key = cv2.waitKey(1) & 0xFF

                # 終了（ESC）
                if key == 27 or keyboard.is_pressed("esc"):
                    print("[INFO] Exit.")
                    break

                # 'c' を押したらキャプチャ開始（5秒平均）
                if not capturing and (key == ord('c') or keyboard.is_pressed('c')):
                    capturing = True
                    uv_samples = []
                    capture_end_t = time.monotonic() + CAPTURE_SECONDS
                    print(f"[INFO] Start capturing for {CAPTURE_SECONDS:.1f}s ... "
                          f"(x_cmd={x_cmd:.2f}, y_cmd={y_cmd:.2f}, z_cmd={z_cmd:.2f})")

                # キャプチャ終了判定
                if capturing:
                    remaining = capture_end_t - time.monotonic()
                    cv2.putText(
                        annotated,
                        f"CAPTURING... {max(0.0, remaining):.1f}s | samples={len(uv_samples)}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    if remaining <= 0.0:
                        capturing = False

                        if len(uv_samples) < MIN_SAMPLES_TO_SAVE:
                            print(f"[WARN] Not enough samples: {len(uv_samples)}. Not saved.")
                        else:
                            u_mean = float(np.mean([p[0] for p in uv_samples]))
                            v_mean = float(np.mean([p[1] for p in uv_samples]))
                            ts = time.strftime("%Y-%m-%d %H:%M:%S")

                            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                                w = csv.writer(f)
                                w.writerow([ts, u_mean, v_mean, len(uv_samples), x_cmd, y_cmd, z_cmd])

                            print(f"[INFO] Saved: u_mean={u_mean:.2f}, v_mean={v_mean:.2f}, "
                                  f"n={len(uv_samples)} -> {CSV_PATH}")

                info_text = f"FPS: {fps:.1f} | conf: {CONF_THRES} | press 'c' to save 5s mean"
                cv2.putText(
                    annotated,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow(window_name, annotated)

        finally:
            print("[INFO] Cleanup...")
            try:
                cam.stop_acquisition()
            except Exception:
                pass
            try:
                cam.close_device()
            except Exception:
                pass
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
