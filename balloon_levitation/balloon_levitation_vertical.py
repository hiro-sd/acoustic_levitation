import sys
from pathlib import Path

import keyboard
import numpy as np
from autd3.commands import SetSilencer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.autd_client import AutdController
from tracking.core.autd_patterns import make_single_focus_stm
from tracking.core.autd_sender import make_autd_arrangement, make_constant_modulation

# 半球型の紙風船を鉛直方向に移動させるプログラム

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
        z = 400.0 # z座標の初期値
        z_min, z_max = 200.0, 700.0 # z座標の最小値と最大値
        prev_z = None # z座標を保存するための変数
        step = 1.0 # 1回の操作で移動する距離

        while True:
            if keyboard.is_pressed("esc"): # ESCキーで終了
                print("終了")
                break

            # Z方向への移動
            if keyboard.is_pressed("up"):
                z = min(z + step, z_max)
            elif keyboard.is_pressed("down"):
                z = max(z - step, z_min)

            # キーボード操作があった場合に処理を実行
            if (z != prev_z):
                prev_z = z

                center = autd.center() + np.array([0.0, 0.0, z]) # 円軌道の中心座標を更新

                # 円軌道上に焦点を配置するための時空間変調 (鉛直方向への移動)
                stm = make_single_focus_stm(
                    [
                        center + radius * np.array([np.cos(theta), np.sin(theta), 0])
                        for theta in (2.0 * np.pi * i / point_num for i in range(point_num))
                    ],
                    100,
                )

                autd.send((m, stm))
                print(f"x: 0.0 mm, y: 0.0 mm, z: {z:.2f}mm")
