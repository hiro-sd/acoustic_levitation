import os
import time
import numpy as np, os, keyboard
from pyautd3 import (
    AUTD3, Controller, FociSTM, Hz, Silencer, Static,
)
from pyautd3_link_soem import SOEM, SOEMOption, Status # SOEMを使用するために追加した
from pyautd3.link.simulator import Simulator # シミュレータを使用するために追加した
from pyautd3_emulator import Emulator # エミュレータを使用するために追加した

# 色々試すためのファイル

# AUTDの配置
autd_arrangement = [
    AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]), 
    AUTD3(pos=[0.0, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[0.0, -2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, -2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),
    ]

def stm_dev(center: np.ndarray, radius: float, point_num: int) -> FociSTM:
    # 正方向角リスト
    angles_fwd = [np.pi/8 + 2.0 * np.pi * i / point_num for i in range(point_num)] # 穴の位置をずらすためにπ/8を加える
    # 逆方向角リスト
    angles_rev = angles_fwd[::-1][1:3] + angles_fwd[::-1][5:-1] # 対向する4点が弱くなる
    # angles_rev = angles_fwd[::-1][0:3] + angles_fwd[::-1][4:-1] # 対向する2点が弱くなる
    # forward + reverse の 2 周分を連結
    angles = angles_fwd + angles_rev

    # 円軌道上に焦点を配置するための時空間変調
    foci = (
        center + radius * np.array([np.cos(a), np.sin(a), 0.0])
        for a in angles
    )
    return FociSTM(foci=foci, config=80 * Hz).into_nearest()

# SOEMのエラーハンドラ
def err_handler(slave: int, status: Status) -> None:
    print(f"slave [{slave}]: {status}")
    if status == Status.Lost():
        os._exit(-1)

if __name__ == "__main__":
    with Controller.open(
        autd_arrangement,
        # Simulator("127.0.0.1:8080"), # シミュレータを使用する
        SOEM(err_handler=err_handler, option=SOEMOption()), # SOEMを使用する
    ) as autd:
        firmware_version = autd.firmware_version()
        print(
            "\n".join(
                [f"[{i}]: {firm}" for i, firm in enumerate(firmware_version)],
            ),
        )

        autd.send(Silencer())

        m = Static(intensity=int(0xFF * 0.65)) # 振幅変調を行わず、常に同じ振幅を出力する

        point_num = 8 # 円周上の点の数
        radius = 19.0 # 円の半径
        x, y, z = 0.0, 0.0, 400.0 # x,y,z座標の初期値
        x_min, x_max = -100.0, 100.0 # x座標の最小値と最大値
        y_min, y_max = -150.0, 150.0 # y座標の最小値と最大値
        z_min, z_max = 200.0, 700.0 # 244.0, 642.0 # z座標の最小値と最大値
        prev_x, prev_y, prev_z = None, None, None # 前回のx,y,z座標を保存するための変数
        step = 0.01 # 1回の操作で移動する距離

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
            
            # キーボード操作またはパターン変化があった場合にSTMを更新
            if (x != prev_x or y != prev_y or z != prev_z):

                prev_x, prev_y, prev_z = x, y, z # 前回の座標を更新

                center = autd.center() + np.array([x, y, z]) # 円軌道の中心座標を更新

                stm = stm_dev(center=center, radius=radius, point_num=point_num)

                autd.send((m, stm))
                print(f"x: {x:.2f}mm, y: {y:.2f}mm, z: {z:.2f}mm")