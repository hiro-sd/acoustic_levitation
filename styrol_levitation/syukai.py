import sys
from pathlib import Path

import keyboard
import numpy as np
from autd3.commands import SetSilencer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.autd_client import AutdController
from tracking.core.autd_patterns import make_single_focus_stm
from tracking.core.autd_sender import make_autd_arrangement, make_constant_modulation

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

        m = make_constant_modulation(1.0)

        point_num = 8 # 円周上の点の数
        radius = 23.0 # 円軌道の半径 (小球は19 mm)
        x, y, z = 0.0, 0.0, 400.0 # x,y,z座標の初期値
        x_min, x_max = -100.0, 100.0 # x座標の最小値と最大値
        y_min, y_max = -150.0, 150.0 # y座標の最小値と最大値
        z_min, z_max = 200.0, 700.0 # z座標の最小値と最大値 (修論では、範囲は 244.0, 642.0)
        prev_x, prev_y, prev_z = None, None, None # 前回のx,y,z座標を保存するための変数
        step = 1.0 # 1回の操作で移動する距離

        while True:
            if keyboard.is_pressed("esc"):
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

            # Z方向への移動
            if keyboard.is_pressed("u"):
                z = min(z + step, z_max)
            elif keyboard.is_pressed("d"):
                z = max(z - step, z_min)

            # キーボード操作があった場合に処理を実行
            if (x != prev_x or y != prev_y or z != prev_z):
                prev_x, prev_y, prev_z = x, y, z

                center = autd.center() + np.array([x, y, z]) # 円軌道の中心座標を更新

                # 円軌道上に焦点を配置するための時空間変調 (鉛直方向への移動)
                points = [
                    center + radius * np.array([np.cos(theta), np.sin(theta), 0])
                    for theta in (
                        np.pi / 8 + 2.0 * np.pi * i / point_num
                        for i in range(point_num)
                    )
                ]
                stm = make_single_focus_stm(points, 100.0)

                autd.send((m, stm))
                print(f"x: {x:.2f}mm, y: {y:.2f}mm, z: {z:.2f}mm")
