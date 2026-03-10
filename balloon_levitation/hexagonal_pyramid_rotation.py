import os
import numpy as np, os, keyboard
from pyautd3 import (
    AUTD3, Controller, FociSTM, Hz, Silencer, Static
)
# from pyautd3_link_soem import SOEM, SOEMOption, Status
from pyautd3.link.twincat import TwinCAT

# 六角錐型の紙風船を回転させるプログラム

# AUTDの配置
autd_arrangement = [
    AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]), 
    AUTD3(pos=[0.0, (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[0.0, 2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, 2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[2 * AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[2 * AUTD3.DEVICE_WIDTH, (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[2 * AUTD3.DEVICE_WIDTH, 2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    ]

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
    foci = (center + radius * np.array([np.cos(a), np.sin(a), 0.0]) for a in angles_all)
    return FociSTM(foci=foci, config=1 * Hz).into_nearest()

# SOEMのエラーハンドラ
# def err_handler(slave: int, status: Status) -> None:
#     print(f"slave [{slave}]: {status}")
#     if status == Status.Lost():
#         os._exit(-1)

if __name__ == "__main__":
    with Controller.open(
        autd_arrangement,
        # SOEM(err_handler=err_handler, option=SOEMOption()),
        TwinCAT(),
    ) as autd:
        firmware_version = autd.firmware_version()
        print(
            "\n".join(
                [f"[{i}]: {firm}" for i, firm in enumerate(firmware_version)],
            ),
        )

        autd.send(Silencer())

        # 物体の質量に応じてintensityを調整する
        m = Static(intensity=int(0xFF*0.5)) # 振幅変調を行わず、常に同じ振幅で出力する
        
        center = autd.center() + np.array([0.0, 0.0, 400.0])
        stm = stm_rotating(center, radius=45.0, point_num=6, total_steps=100)
        autd.send((m, stm))

        while True:
            if keyboard.is_pressed("esc"): # ESCキーで終了
                print("終了")
                break