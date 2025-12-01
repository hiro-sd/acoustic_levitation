# 高速カメラのfpsを計測するスクリプト
from time import perf_counter
import sys

# XIMEA をデフォルトパスに入れている場合（標準インストール）
sys.path.append(r"C:\Users\Hiroto Yoshida\Desktop\XIMEA\API\Python\v3")
from ximea import xiapi

img = xiapi.Image()
cam = xiapi.Camera()
cam.open_device()
cam.set_imgdataformat('XI_RGB24')
cam.set_exposure(5000)
cam.start_acquisition()
try:
    t0 = perf_counter()
    count = 0
    while True:
        cam.get_image(img)
        _ = img.get_image_data_numpy()
        count += 1
        if perf_counter() - t0 >= 1.0:
            print("capture fps:", count / (perf_counter() - t0))
            count = 0
            t0 = perf_counter()
finally:
    cam.stop_acquisition()
    cam.close_device()