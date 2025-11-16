import os
import time
import cv2
from ultralytics import YOLO

# --------------------------------------------------
# 設定（必要に応じてここだけ書き換えればOK）
# --------------------------------------------------
MODEL_PATH = "yolo11n.pt"  # yolo11n.pt / yolo11s.pt / 自分の学習済みモデルなど
CAMERA_INDEX = 0           # 使用するWebカメラ番号（通常0）
IMG_SIZE = 640             # 推論時の入力サイズ
CONF_THRES = 0.25          # バウンディングボックスの信頼度しきい値
CAP_WIDTH = 1280           # キャプチャ解像度（対応しないカメラもあります）
CAP_HEIGHT = 720

# CPUでスレッド数を制限したい場合（お好みで）
os.environ.setdefault("OMP_NUM_THREADS", "4")


def main():
    # -----------------------------
    # モデル読み込み（YOLO11）
    # -----------------------------
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)  # 公式YOLO11 Detectモデル or 自分の学習済みモデル

    # -----------------------------
    # カメラ初期化
    # -----------------------------
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"カメラ {CAMERA_INDEX} をオープンできませんでした。")

    # 解像度の指定（効かない場合もあります）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    window_name = "YOLO11 (CPU) - Press q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] フレームを取得できませんでした。")
                break

            # -------------------------
            # YOLO11で推論（CPU指定）
            # -------------------------
            results = model.predict(
                source=frame,      # 画像(ndarray)をそのまま渡せる
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                device="cpu",      # CPUで実行
                verbose=False
            )

            # アノテーション済み画像を取得（バウンディングボックスとラベル付き）
            annotated_frame = results[0].plot()

            # -------------------------
            # FPS計算と表示
            # -------------------------
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

            # -------------------------
            # 画面表示
            # -------------------------
            cv2.imshow(window_name, annotated_frame)

            # 'q' キーで終了
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] 'q' が押されたので終了します。")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
