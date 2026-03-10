import os
import numpy as np, os, keyboard
from pyautd3 import (
    AUTD3, Controller, Focus, FocusOption, GainSTM, GainSTMMode, GainSTMOption, Group, Hz, Null, Silencer, Static,
)
# from pyautd3_link_soem import SOEM, SOEMOption, Status
from pyautd3.link.twincat import TwinCAT

# 半球型の紙風船を水平方向に移動させるプログラム

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
        m = Static(intensity=0xFF) # 振幅変調を行わず、常に同じ振幅で出力する

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
                gains = [] # gainsにGroupのリストを格納する
                for theta in (2.0 * np.pi * i / point_num for i in range(point_num)):
                    focus = Focus( # 単焦点を形成するGain
                        pos = center + radius * np.array([np.cos(theta), np.sin(theta), 0]),
                        option = FocusOption(),
                    )

                    gain = Group(
                        # 円軌道中心から150mm以内のトランスデューサにのみSTMを適用する
                        key_map = lambda _: lambda tr: "in" if np.linalg.norm(tr.position()[:2] - center[:2]) <= 150.0 else "out",
                        gain_map={"in": focus, "out": Null()},
                    )

                    gains.append(gain)

                    stm = GainSTM(
                        gains,
                        config = 100 * Hz, # 100Hzで更新
                        option = GainSTMOption(
                            mode = GainSTMMode.PhaseIntensityFull,
                        ),
                    ).into_nearest()

                autd.send((m, stm))
                print(f"x: {x:.2f}mm, y: {y:.2f}mm, z: 400.0 mm")