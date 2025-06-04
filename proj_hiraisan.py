import os
import numpy as np, os, keyboard
from pyautd3 import (
    AUTD3, Controller, FociSTM, Focus, FocusOption, GainSTM, GainSTMMode, GainSTMOption, Group, Hz, Null, Silencer, Static,
)
from pyautd3_link_soem import SOEM, SOEMOption, Status # SOEMを使用するために追加した
from pyautd3.link.simulator import Simulator # シミュレータを使用するために追加した
from pyautd3_emulator import Emulator # エミュレータを使用するために追加した

# AUTDの配置
autd_arrangement = [
    AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]), 
    AUTD3(pos=[0.0, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[0.0, -2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, -2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),
    ]

# SOEMのエラーハンドラ
def err_handler(slave: int, status: Status) -> None:
    print(f"slave [{slave}]: {status}")
    if status == Status.Lost():
        os._exit(-1)

# 平井さんの修論を参考にして作成したファイル
if __name__ == "__main__":
    with Controller.open(
        autd_arrangement,
        # Simulator("127.0.0.1:8080"), # シミュレータを使用する際
        SOEM(err_handler=err_handler, option=SOEMOption()), # SOEMを使用する際
    ) as autd:
        firmware_version = autd.firmware_version()
        print(
            "\n".join(
                [f"[{i}]: {firm}" for i, firm in enumerate(firmware_version)],
            ),
        )

        autd.send(Silencer())

        m = Static(intensity=0xFF) # 振幅変調を行わず、常に同じ振幅を出力する

        point_num = 7 # 円周上の点の数
        radius = 45.0 # 円軌道の半径
        x, y, z = 0.0, 0.0, 400.0 # x,y,z座標の初期値
        x_min, x_max = -100.0, 100.0 # x座標の最小値と最大値
        y_min, y_max = -150.0, 150.0 # y座標の最小値と最大値
        z_min, z_max = 200.0, 700.0 # 244.0, 642.0 # z座標の最小値と最大値
        prev_x, prev_y, prev_z = None, None, None # 前回のx,y,z座標を保存するための変数
        step = 1.0 # 1回の操作で移動する距離
        switch = False # STMの切り替えを行うかどうかのフラグ
        prev_switch = None # 前回のswitchの値を保存するための変数

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
            
            # STMの切り替え
            if keyboard.is_pressed("enter"):
                switch = not switch 
                if not switch:
                    print("FociSTMを使用します")
                else:
                    print("GainSTMを使用します")

            # キーボード操作があった場合に処理を実行
            if (x != prev_x or y != prev_y or z != prev_z or switch != prev_switch):
                prev_x, prev_y, prev_z, prev_switch = x, y, z, switch # 前回の座標とswitchの値を更新

                center = autd.center() + np.array([x, y, z]) # 円軌道の中心座標を更新

                if not switch: 
                    # 円軌道上に焦点を配置するための時空間変調 (鉛直方向への移動のみならこれでよい) (単焦点)
                    stm = FociSTM(
                        foci = (
                            center + radius * np.array([np.cos(theta), np.sin(theta), 0])
                            for theta in (2.0 * np.pi * i / point_num for i in range(point_num))
                            ),
                        config = 100 * Hz, # 100Hzで更新(1秒間に円周上を100周する)
                    ).into_nearest() # point_num = 40kHz/Nを満たすNが存在しない場合、エラーになる

                else:
                    # 円軌道上に焦点を配置するための時空間変調 (水平方向へも移動したい時)
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
                        config = 100 * Hz, # 100Hzで更新(1秒間に円周上を100周する)
                        option = GainSTMOption(
                            mode = GainSTMMode.PhaseIntensityFull,
                        ),
                    ).into_nearest()

                autd.send((m, stm))
                print(f"x: {x:.2f}mm, y: {y:.2f}mm, z: {z:.2f}mm")