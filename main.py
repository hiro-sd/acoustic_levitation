import os
import numpy as np
import time # 時間制御のために追加した
import keyboard # キーボード操作を検出するために追加した
from pyautd3 import (
    AUTD3,
    Controller,
    FociSTM,
    Focus,
    FocusOption,
    Group,
    Hz,
    Null,
    Phase,
    Silencer,
    Static,
)
from pyautd3.gain import Custom
from pyautd3_link_soem import SOEM, SOEMOption, Status # SOEMを使用するために追加した
from pyautd3.link.simulator import Simulator # シミュレータを使用するために追加した
from pyautd3_emulator import Emulator # エミュレータを使用するために追加した

def err_handler(slave: int, status: Status) -> None:
    print(f"slave [{slave}]: {status}")
    if status == Status.Lost():
        os._exit(-1)

# 6台のAUTD3を3行2列の配置で使用する例
# 05
# 14
# 23
autd_arrangement = [
    AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]), 
    AUTD3(pos=[0.0, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[0.0, -2*(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, -2*(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),
]

if __name__ == "__main__":
    with Controller.open(
        autd_arrangement,
        # Simulator("127.0.0.1:8080"), # シミュレータを使用するために追加した
        SOEM(err_handler=err_handler, option=SOEMOption()),
    ) as autd:
        firmware_version = autd.firmware_version()
        print(
            "\n".join(
                [f"[{i}]: {firm}" for i, firm in enumerate(firmware_version)],
            ),
        )

        autd.send(Silencer())

        m = Static(intensity=int(0xFF*0.7)) # 振幅変調を行わず、常に同じ振幅を出力する(最大出力の0.7倍)

        point_num = 7 # 円周上の点の数
        radius = 50.0 # 円の半径
        x = 0.0 # x座標の初期値
        x_min, x_max = -100.0, 100.0 # x座標の最小値と最大値
        y = 0.0 # y座標の初期値
        y_min, y_max = -150.0, 150.0 # y座標の最小値と最大値
        z = 400.0 # z座標の初期値
        z_min, z_max = 200.0, 650.0 # 244.0, 642.0 # z座標の最小値と最大値
        step = 1.0 # 1回の操作で移動する量
        prev_x, prev_y, prev_z = None, None, None # 前回のx,y,z座標を保存するための変数

        while True:
            if keyboard.is_pressed("esc"):
                print("終了")
                break

            # if keyboard.is_pressed("enter"):
            #     print("リセット")
            #     x, y, z = 0.0, 0.0, 400.0

            # X方向への移動
            if keyboard.is_pressed("right"):
                x = min(x + step, x_max)
            if keyboard.is_pressed("left"):
                x = max(x - step, x_min)

            # Y方向への移動
            if keyboard.is_pressed("up"):
                y = min(y + step, y_max)
            if keyboard.is_pressed("down"):
                y = max(y - step, y_min)

            # Z方向への移動
            if keyboard.is_pressed("u"):
                z = min(z + step, z_max)
            if keyboard.is_pressed("d"):
                z = max(z - step, z_min)

            if (x != prev_x or y != prev_y or z != prev_z):
                prev_x, prev_y, prev_z = x, y, z # 前回の座標を更新

                center = autd.center() + np.array([x, y, z]) # 円軌道の中心座標を更新

                # 円軌道上に焦点を配置するための時空間変調
                stm = FociSTM(
                    foci = (
                        center + radius * np.array([np.cos(theta), np.sin(theta), 0])
                        for theta in (2.0 * np.pi * i / point_num for i in range(point_num))),
                    config=100 * Hz, # 100Hzで更新(1秒間に円周上を100周する)
                ).into_nearest() # point_num = 40kHz/Nを満たすNが存在しない場合、エラーになる

                # 円軌道中心から150mm以内のトランスデューサにのみSTMを適用するためのグループ化
                grp = Group(
                    key_map = lambda _: lambda tr: "target" if np.linalg.norm(tr.position()[:2] - center[:2]) <= 150.0 else "null",
                    gain_map = {"in": Focus(pos = center, option = FocusOption()), "out": Null()},
                )

                autd.send((m, grp)) # これだとグルーピングできてはいるが、STMが適用されない
                # autd.send((m, stm)) # これだとSTMが適用されるが、グルーピングできていない (一度これで水平方向の移動をテストするべき?)
                print(f"x: {x:.2f}mm, y: {y:.2f}mm, z: {z:.2f}mm")

            time.sleep(0.01) # 10msの間隔でループを回す

        autd.close()

# TODO: グルーピングとSTMの適用、Staticを同時に行う方法を考える (水平方向への移動の実装のため)