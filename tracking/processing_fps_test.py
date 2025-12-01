#  model.predict を含む処理のスループットを計測するスクリプト
from time import perf_counter
from ultralytics import YOLO
import cv2
import sys

# XIMEA をデフォルトパスに入れている場合（標準インストール）
sys.path.append(r"C:\Users\Hiroto Yoshida\Desktop\XIMEA\API\Python\v3")
from ximea import xiapi

model = YOLO("yolo11n.pt")
cam = xiapi.Camera(); cam.open_device()
cam.set_imgdataformat('XI_RGB24'); img = xiapi.Image(); cam.start_acquisition()

try:
    t0 = perf_counter()
    count = 0
    while True:
        cam.get_image(img)
        frame = img.get_image_data_numpy()
        # minimal conversion if needed:
        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _ = model.predict(source=frame, imgsz=640, conf=0.5, device="cpu", verbose=False)
        count += 1
        if perf_counter() - t0 >= 1.0:
            print("processing fps:", count / (perf_counter() - t0))
            count = 0
            t0 = perf_counter()
finally:
    cam.stop_acquisition(); cam.close_device()