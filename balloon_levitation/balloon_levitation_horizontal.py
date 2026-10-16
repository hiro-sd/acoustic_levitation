import sys
from pathlib import Path

import keyboard
import numpy as np
from autd3.commands import SetSilencer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.autd_client import AutdController
from tracking.core.autd_patterns import make_masked_focus_pattern_stm
from tracking.core.autd_sender import make_autd_arrangement, make_constant_modulation

# 半球型の紙風船を水平方向に移動させるプログラム

# AUTDの配置
autd_arrangement = make_autd_arrangement()

# SOEMのエラーハンドラ
# def err_handler(slave: int, status: Status) -> None:
#     print(f"slave [{slave}]: {status}")
#     if status == Status.Lost():
#         os._exit(-1)

if __name__ == "__main__":
    with AutdController.open(autd_arrangement) as autd:
        firmware_version = autd.read_firmware_version()
        print(
            "\n".join(
                [f"[{i}]: {firm}" for i, firm in enumerate(firmware_version)],
            ),
        )

        autd.send(SetSilencer())

        # 物体の質量に応じてintensityを調整する
        m = make_constant_modulation(1.0)

        point_num = 8 # 円周上の点の数
        radius = 45.0 # 円軌道の半径
        x, y = 0.0, 0.0 # x,y座標の初期値
        x_min, x_max = -100.0, 100.0 # x座標の最小値と最大値
        y_min, y_max = -150.0, 150.0 # y座標の最小値と最大値
        prev_x, prev_y = None, None # x,y座標を保存するための変数
        step = 8.0 # 1回の操作で移動する距離

        while True:
            if keyboard.is_pressed("esc"): # ESCキーで終了
                print("終了")
                break

            # X方向への移動
            if keyboard.is_pressed("right"):
                x = min(x + step, x_max)
            elif keyboard.is_pressed("left"):
                x = max(x - step, x_min)

            # Y方向への移動
            if keyboard.is_pressed("up"):
                y = min(y + step, y_max)
            elif keyboard.is_pressed("down"):
                y = max(y - step, y_min)

            # キーボード操作があった場合に処理を実行
            if (x != prev_x or y != prev_y):
                prev_x, prev_y = x, y

                center = autd.center() + np.array([x, y, 400.0]) # 円軌道の中心座標を更新

                # 円軌道上に焦点を配置するための時空間変調 (水平方向への移動)
                points = [
                    center + radius * np.array([np.cos(theta), np.sin(theta), 0])
                    for theta in (2.0 * np.pi * i / point_num for i in range(point_num))
                ]
                stm = make_masked_focus_pattern_stm(
                    autd.geometry,
                    points,
                    100,
                    mask_center_xy=center[:2],
                    mask_radius_mm=150.0,
                )

                autd.send((m, stm))
                print(f"x: {x:.2f}mm, y: {y:.2f}mm, z: 400.0 mm")
