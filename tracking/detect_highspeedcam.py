import os
import time
import sys
import cv2
from ultralytics import YOLO

# XIMEA をデフォルトパスに入れている場合（標準インストール）
sys.path.append(r"C:\Users\Hiroto Yoshida\Desktop\XIMEA\API\Python\v3")


# XIMEA 用
from ximea import xiapi

# 設定（必要に応じてここだけ書き換えればOK）
MODEL_PATH = "yolo11n.pt"  # yolo11n.pt / yolo11s.pt / 自分の学習済みモデルなど
IMG_SIZE = 640             # 推論時の入力サイズ
CONF_THRES = 0.5           # バウンディングボックスの信頼度しきい値

# XIMEA 側の設定（必要に応じて調整）
EXPOSURE_US = 5000         # 露光時間 [µs]（明るさ・モーションブラーに応じて調整）
# MQ003CG-CM はもともと 640x480 が標準解像度

# CPUでスレッド数を制限したい場合（お好みで）
os.environ.setdefault("OMP_NUM_THREADS", "4")


def main():

    # モデル読み込み（YOLO11）
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)  # 公式YOLO11 Detectモデル or 自分の学習済みモデル

    # XIMEA カメラ初期化
    print("[INFO] Initializing XIMEA camera...")
    cam = xiapi.Camera()

    # 複数台ある場合は open_device(device=0) のように指定可能
    cam.open_device()
    print("[INFO] XIMEA camera opened.")

    # 画像フォーマットをカラー (RGB24) に設定
    cam.set_imgdataformat('XI_RGB24')

    # 自動ホワイトバランスON
    try:
        cam.enable_auto_wb()
    except AttributeError:
        cam.set_param('auto_wb', 1)

    # 露光時間設定（単位は µs）
    cam.set_exposure(EXPOSURE_US)
    print(f"[INFO] Exposure set to {EXPOSURE_US} us")

    # 画像バッファ
    img = xiapi.Image()

    # 取得開始
    cam.start_acquisition()
    print("[INFO] XIMEA acquisition started.")

    window_name = "YOLO11 + XIMEA (CPU) - Press q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:

            # XIMEA から 1 フレーム取得
            cam.get_image(img)
            frame = img.get_image_data_numpy()

            # カラーの場合: frame は通常 RGB。OpenCV 互換の BGR に変換しておく
            # モノクロの場合: 2次元配列なので BGR に拡張
            if frame.ndim == 3:
                frame_bgr = frame.copy()
            else:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # YOLO11で推論（CPU指定）
            results = model.predict(
                source=frame_bgr,   # 画像(ndarray)をそのまま渡せる
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                device="cpu",       # CPUで実行
                verbose=False
            )

            # アノテーション済み画像を取得（バウンディングボックスとラベル付き）
            annotated_frame = results[0].plot()

            # FPS計算と表示
            now = time.time()          # 現在時刻
            dt = now - prev_time       # 前フレームからの経過時間
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)  # 簡易移動平均

            info_text = (
                f"FPS: {fps:.1f} | model: {MODEL_PATH} "
                f"| imgsz: {IMG_SIZE} | conf: {CONF_THRES}"
            )
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 画面表示
            cv2.imshow(window_name, annotated_frame)

            # 'q' キーで終了
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] 'q' キーが押されたので終了します。")
                break



    finally:
        # XIMEA 側の後処理
        print("[INFO] Stopping acquisition and closing XIMEA camera...")
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
