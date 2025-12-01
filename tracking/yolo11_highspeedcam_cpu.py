import os
import time
import cv2
from ultralytics import YOLO
import ximea

# trackingフォルダにxiapi.pyをコピーした状態
# TODO: CHATGPTの⑤動作確認から再開する

MODEL_PATH = "yolo11n.pt"  # yolo11n.pt / yolo11s.pt / 自分の学習済みモデルなど
IMG_SIZE = 640             # 推論時の入力サイズ
CONF_THRES = 0.25          # バウンディングボックスの信頼度閾値
CAP_WIDTH = 1280           # 使いたい解像度（XIMEA側で設定できれば使用）
CAP_HEIGHT = 720

# CPUでスレッド数を制限したい場合
os.environ.setdefault("OMP_NUM_THREADS", "4")


def main():

    # モデル読み込み（YOLO11）
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # XIMEAカメラ初期化
    print("[INFO] Initializing XIMEA camera...")
    cam = xi.Camera()

    # デバイスオープン（複数台ある場合は device=0,1,... を指定）
    cam.open_device()  # 例: cam.open_device(device=0)

    # 露光時間やゲイン、解像度などの設定（環境に合わせて調整）
    # 単位は µs（例: 5000 = 5 ms）※値は適宜変更してください
    cam.set_exposure(5000)

    # 解像度を設定したい場合（モデルやカメラによっては別パラメータ名の場合あり）
    # 利用環境のサンプルコードやマニュアルに合わせて調整してください。
    # try:
    #     cam.set_width(CAP_WIDTH)
    #     cam.set_height(CAP_HEIGHT)
    # except Exception as e:
    #     print("[WARN] 解像度設定に失敗しました:", e)

    # 画像バッファ
    img = xi.Image()

    # キャプチャ開始
    cam.start_acquisition()
    print("[INFO] XIMEA acquisition started.")

    window_name = "YOLO11 + XIMEA (CPU) - Press q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            # XIMEA からフレーム取得
            cam.get_image(img)                       # カメラから1フレーム取得
            frame = img.get_image_data_numpy()      # NumPy配列として取得

            # XIMEAのPythonサンプルでは RGB で返ってくることが多いので、
            # OpenCV と同じ BGR に変換しておく（必須ではないが無難）
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                # モノクロの場合などは3chに変換しておく
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
            now = time.time()
            dt = now - prev_time
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
                print("[INFO] 'q' が押されたので終了します。")
                break

    finally:
        # 後処理
        print("[INFO] Stopping acquisition and closing XIMEA device...")
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