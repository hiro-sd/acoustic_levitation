import os
import time
import sys
import numpy as np
import cv2
import keyboard

from ultralytics import YOLO

# ===== XIMEA のパス設定 =====
# あなたの環境に合わせて必要なら書き換え
sys.path.append(r"C:\Users\Hiroto Yoshida\Desktop\XIMEA\API\Python\v3")
from ximea import xiapi

# ===== AUTD 関連 =====
from pyautd3 import (
    AUTD3,
    Controller,
    FociSTM,
    Focus,
    FocusOption,
    GainSTM,
    GainSTMMode,
    GainSTMOption,
    Group,
    Hz,
    Null,
    Silencer,
    Static,
)
from pyautd3_link_soem import SOEM, SOEMOption, Status

# ========== 設定 ==========
# --- YOLO ---
MODEL_PATH = "/Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation/tracking/train/weights/best.pt"
IMG_SIZE = 640
CONF_THRES = 0.5

# --- XIMEA ---
EXPOSURE_US = 5000  # 露光時間 [µs]（暗ければ増やす、明るすぎれば減らす）

# --- AUTD ---
POINT_NUM = 8       # 円周上の焦点数
RADIUS = 23.0       # 円軌道の半径 [mm]
X0, Y0, Z0 = 0.0, 0.0, 400.0  # AUTD中心からの初期オフセット [mm]

os.environ.setdefault("OMP_NUM_THREADS", "4")

# AUTDの配置
autd_arrangement = [
    AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[0.0, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[0.0, -2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, -2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),
]

# SOEMのエラーハンドラ
def err_handler(slave: int, status: Status) -> None:
    print(f"slave [{slave}]: {status}")
    if status == Status.Lost():
        os._exit(-1)


def init_ximea_camera():
    """XIMEA高速カメラの初期化"""
    cam = xiapi.Camera()
    cam.open_device()
    print("[INFO] XIMEA camera opened.")

    # カラー画像 (RGB24) で取得
    cam.set_imgdataformat('XI_RGB24')

    # 自動ホワイトバランス（使える場合）
    try:
        cam.enable_auto_wb()
    except AttributeError:
        try:
            cam.set_param('auto_wb', 1)
        except Exception:
            print("[WARN] auto white balance could not be enabled.")

    # 露光時間設定
    cam.set_exposure(EXPOSURE_US)
    print(f"[INFO] Exposure set to {EXPOSURE_US} us")

    img = xiapi.Image()
    cam.start_acquisition()
    print("[INFO] XIMEA acquisition started.")

    return cam, img


def find_sports_ball_class_id(model: YOLO) -> int | None:
    """
    YOLOモデルのクラス名一覧から 'sports ball' のクラスIDを探す。
    見つからなければ None を返す。
    """
    names = model.model.names  # dict: {class_id: class_name}
    for cid, name in names.items():
        if name.lower() == "sports ball":
            return int(cid)
    return None


def main():
    # ========= YOLO 初期化 =========
    print(f"[INFO] Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    sports_ball_cls_id = find_sports_ball_class_id(model)
    if sports_ball_cls_id is None:
        print("[WARN] 'sports ball' クラスが見つかりませんでした。全クラスを表示します。")
    else:
        print(f"[INFO] 'sports ball' class id = {sports_ball_cls_id}")

    # ========= XIMEA 初期化 =========
    cam, img = init_ximea_camera()

    # ========= AUTD + SOEM 起動 =========
    with Controller.open(
        autd_arrangement,
        SOEM(err_handler=err_handler, option=SOEMOption()),
    ) as autd:

        firmware_version = autd.firmware_version()
        print(
            "\n".join(
                [f"[{i}]: {firm}" for i, firm in enumerate(firmware_version)],
            ),
        )

        # サイレンサーと振幅設定
        autd.send(Silencer())
        m = Static(intensity=int(0xFF))  # 小球なら *0.65 等してもOK

        window_name = "YOLO11 + XIMEA + AUTD (CPU) - Press ESC to quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        prev_time = time.time()
        fps = 0.0

        # 円運動の中心位置
        x, y, z = X0, Y0, Z0

        try:
            while True:
                # ======= 終了キー判定 =======
                if keyboard.is_pressed("esc"):
                    print("[INFO] ESC pressed. Exiting...")
                    break

                # ======= AUTD: 円軌道STMを送信 =======
                center = autd.center() + np.array([x, y, z])

                stm = FociSTM(
                    foci=(
                        center + RADIUS * np.array([np.cos(theta), np.sin(theta), 0.0])
                        for theta in (
                            np.pi / 8 + 2.0 * np.pi * i / POINT_NUM
                            for i in range(POINT_NUM)
                        )
                    ),
                    config=100 * Hz,  # 更新周波数
                ).into_nearest()

                autd.send((m, stm))

                # ======= XIMEA: フレーム取得 =======
                cam.get_image(img)
                frame = img.get_image_data_numpy()

                if frame.ndim == 3:
                    frame_bgr = frame.copy()
                else:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

# TODO: sports ballではなく学習させたlevitatedballのみを検出するようなコードに変更する
                # ======= YOLO: sports ball のみ検出 =======
                results = model.predict(
                    source=frame_bgr,
                    imgsz=IMG_SIZE,
                    conf=CONF_THRES,
                    device="cpu",
                    verbose=False,
                )

                # 自前で sports ball だけ矩形描画
                annotated = frame_bgr.copy()
                boxes = results[0].boxes

                detected_count = 0
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        score = float(box.conf[0].item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        # sports ball のみ描画（クラスIDが分かっていれば）
                        if (sports_ball_cls_id is None) or (cls_id == sports_ball_cls_id):
                            detected_count += 1
                            # 枠線
                            cv2.rectangle(
                                annotated,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 255, 0),
                                2,
                            )
                            # ラベル
                            label = f"{model.model.names[cls_id]} {score:.2f}"
                            cv2.putText(
                                annotated,
                                label,
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )

                # ======= FPS 計算 =======
                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                info_text = (
                    f"FPS: {fps:.1f} | model: {MODEL_PATH} "
                    f"| imgsz: {IMG_SIZE} | conf: {CONF_THRES} "
                    f"| sports_ball: {detected_count}"
                )
                cv2.putText(
                    annotated,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # ======= 画面表示 =======
                cv2.imshow(window_name, annotated)

                # OpenCV側でも ESC を拾っておく（どちらでも終了できるように）
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("[INFO] ESC pressed (OpenCV). Exiting...")
                    break

        finally:
            # 後処理
            print("[INFO] Cleaning up...")
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
