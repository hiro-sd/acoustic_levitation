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

# 六角錐型の紙風船を回転させるプログラム

# AUTDの配置
autd_arrangement = make_autd_arrangement()

# 等間隔な6焦点の位置を連続的に回転させるSTM
def stm_rotating(center, radius, point_num, total_steps):
    # total_steps: 1周を何分割して連続回転させるか（大きいほど滑らか）
    angles_all = []
    for step in range(total_steps):
        # 1ステップごとに全体を少し回転
        offset = 2 * np.pi * step / total_steps
        for i in range(point_num):
            angle = offset + 2 * np.pi * i / point_num
            angles_all.append(angle)
    foci = [center + radius * np.array([np.cos(a), np.sin(a), 0.0]) for a in angles_all]
    return make_single_focus_stm(foci, 1)

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
        m = make_constant_modulation(0.5)
        
        center = autd.center() + np.array([0.0, 0.0, 400.0])
        stm = stm_rotating(center, radius=45.0, point_num=6, total_steps=100)
        autd.send((m, stm))

        while True:
            if keyboard.is_pressed("esc"): # ESCキーで終了
                print("終了")
                break
