import os
from time import perf_counter
import cv2
import sys
from ultralytics import YOLO

# XIMEA import (既存環境に合わせてパスを調整してください)
sys.path.append(r"C:\Users\Hiroto Yoshida\Desktop\XIMEA\API\Python\v3")
from ximea import xiapi

# 設定
MODEL_PATH = "yolo11n.pt"
IMG_SIZE = 640
CONF_THRES = 0.5
EXPOSURE_US = 5000
DRAW = True   # True: 軽量描画あり / False: 描画なし（推論のみ計測）
SHOW_WINDOW = True

os.environ.setdefault("OMP_NUM_THREADS", "4")

def draw_boxes_cv(frame, boxes, model_names):
    """軽量描画: boxes から座標・conf・cls を取り出して cv2 で描画する"""
    # frame は BGR (OpenCV) 想定
    annotated = frame.copy()
    if boxes is None:
        return annotated

    # まず座標群を安全に取り出す（ultralytics のバージョン差に対応）
    coords = []
    confs = []
    clss = []
    try:
        # 多くのバージョンでは boxes.xyxy / boxes.conf / boxes.cls を持つ
        xyxy = boxes.xyxy.tolist()   # [[x1,y1,x2,y2], ...]
        confs = boxes.conf.tolist()
        clss = boxes.cls.tolist()
        coords = xyxy
    except Exception:
        # フォールバック: boxes をイテレートして個々の box から取得
        try:
            for b in boxes:
                # b.xyxy および b.conf, b.cls が配列的に入っていることが多い
                c = getattr(b, "xyxy", None)
                if c is None:
                    continue
                # 形が [ [x1,y1,x2,y2] ] の場合を想定
                cc = c[0].tolist() if hasattr(c[0], "tolist") else list(c[0])
                coords.append(cc)
                confs.append(float(getattr(b, "conf", [0])[0]))
                clss.append(int(getattr(b, "cls", [0])[0]))
        except Exception:
            # それでも取れなければ空のまま
            pass

    # 描画ループ
    for (box, conf, cls_id) in zip(coords, confs, clss):
        try:
            x1, y1, x2, y2 = map(int, box)
        except Exception:
            continue
        label = model_names.get(int(cls_id), str(int(cls_id)))
        text = f"{label} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, text, (x1, max(15, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated

def main():
    print("[INFO] Loading model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    # XIMEA 初期化
    cam = xiapi.Camera()
    cam.open_device()
    cam.set_imgdataformat('XI_RGB24')
    try:
        cam.enable_auto_wb()
    except AttributeError:
        cam.set_param('auto_wb', 1)
    cam.set_exposure(EXPOSURE_US)
    img = xiapi.Image()
    cam.start_acquisition()
    print("[INFO] XIMEA started")

    if SHOW_WINDOW:
        window_name = "Manual Draw Test - q to quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # FPS 計測用
    start = perf_counter()
    proc_count = 0

    try:
        while True:
            # 取得
            cam.get_image(img)
            frame = img.get_image_data_numpy()  # XIMEA は通常 RGB 配列を返す

            # OpenCV で扱うため BGR に変換（Ultralytics に渡す画像の色空間が
            # 実行環境で期待するものと一致しているか要確認）
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # 推論（CPU）
            results = model.predict(
                source=frame_bgr,
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                device="cpu",
                verbose=False,
            )

            # 軽量描画: results[0].boxes から描画（plot() を使わない）
            r = results[0]
            boxes = getattr(r, "boxes", None)

            if DRAW:
                annotated = draw_boxes_cv(frame_bgr, boxes, model.names)
            else:
                annotated = frame_bgr  # 描画なしで性能計測

            proc_count += 1
            elapsed = perf_counter() - start
            if elapsed >= 1.0:
                fps = proc_count / elapsed
                print(f"[PROC FPS] {fps:.2f} (DRAW={DRAW})")
                proc_count = 0
                start = perf_counter()

            # ウィンドウ表示（描画あり/なしどちらでも）
            if SHOW_WINDOW:
                cv2.putText(annotated, f"DRAW={DRAW}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] q pressed - exiting")
                    break

    finally:
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