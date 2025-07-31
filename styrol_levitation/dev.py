import os
import numpy as np, os, keyboard
from pyautd3 import (
    AUTD3, Controller, FociSTM, Hz, Silencer, Static, GainSTM, GainSTMMode, GainSTMOption, EmitIntensity
)
from pyautd3.gain.holo import GSPAT, EmissionConstraint, GSPATOption, NalgebraBackend, Pa
from pyautd3_link_soem import SOEM, SOEMOption, Status # SOEMを使用するために追加した
from pyautd3.link.simulator import Simulator # シミュレータを使用するために追加した
from pyautd3_emulator import Emulator # エミュレータを使用するために追加した

# いろいろ試すためのファイル

# AUTDの配置
autd_arrangement = [
    AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]), 
    AUTD3(pos=[0.0, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[0.0, -2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, -2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),
    ]

# FociSTMで往復軌道を形成するための関数
def stm_dev(center: np.ndarray, radius: float, point_num: int) -> FociSTM:
    # 正方向角リスト
    angles_fwd = [np.pi/8 + 2.0 * np.pi * i / point_num for i in range(point_num)] # 穴の位置をずらすためにπ/8を加える
    angles_basic = angles_fwd + angles_fwd[1:-1]
    # 逆方向角リスト
    angles_rev = angles_fwd[4:] + angles_fwd[:4]
    angles_opposite = angles_rev + angles_rev[1:-1]

    angles = []
    for _ in range(point_num):
        tmp = []
        tmp =(angles_basic) + (angles_opposite) # forward + reverseの2周分を連結
        angles += tmp
        angles_fwd = angles_fwd[1:] + angles_fwd[:1] # 1つずらす
        angles_basic = angles_fwd + angles_fwd[1:-1]
        angles_rev = angles_fwd[4:] + angles_fwd[:4]
        angles_opposite = angles_rev + angles_rev[1:-1]
    # 円軌道上に焦点を配置する
    foci = (
        center + radius * np.array([np.cos(a), np.sin(a), 0.0])
        for a in angles
    )
    return FociSTM(foci=foci, config=4 * Hz).into_nearest()

# 静的8焦点を生成する関数
def multi_focal_points(center, radius, point_num) -> GSPAT:
    angles = [np.pi/8 + 2.0 * np.pi * i / point_num for i in range(point_num)]
    foci = []
    for angle in angles:
        p = center + radius * np.array([np.cos(angle), np.sin(angle), 0.0])
        foci.append((p, 5e4 * Pa))
    return GSPAT(
        foci=foci,
        option=GSPATOption(
                repeat=100,
                constraint=EmissionConstraint.Clamp(EmitIntensity.MIN, EmitIntensity.MAX),
            ),
            backend=NalgebraBackend(),
        )

# ２つの往復軌道を作成する関数
# GSPATで点1と点5に焦点を配置し、それらをGainSTMで半周ずつ回す
def stm_dev2(center: np.ndarray, radius: float, point_num: int) -> GainSTM:
    gains = [],
    j = 1,
    k = 1
    for i in range(point_num-2):
        angles = [np.pi/point_num + 2.0 * np.pi * i / point_num for i in range(point_num)]
        if i < point_num // 2:
            p1 = center + radius * np.array([np.cos(angles[i]), np.sin(angles[i]), 0.0])
            p2 = center + radius * np.array([np.cos(angles[i+(point_num//2)]), np.sin(angles[i+(point_num//2)]), 0.0])
        else:
            p1 = center + radius * np.array([np.cos(angles[i-(j*2)]), np.sin(angles[i-(j*2)]), 0.0])
            p2 = center + radius * np.array([np.cos(angles[i+(k*2)]), np.sin(angles[i+(k*2)]), 0.0])
            j += 1
            k -= 1

        focal_points = GSPAT(
            foci=[
                (p1, 5e4 * Pa),
                (p2, 5e4 * Pa),
            ],
            option=GSPATOption(
                repeat=100,
                constraint=EmissionConstraint.Clamp(EmitIntensity.MIN, EmitIntensity.MAX),
            ),
            backend=NalgebraBackend(),
        )
        gains.append(focal_points)

    return GainSTM( # GainSTMでGSPATで作成した2焦点を回す
        gains,
        config=100 * Hz,
        option = GainSTMOption(
                        mode = GainSTMMode.PhaseIntensityFull,
                    ),
    ).into_nearest()

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

                g = multi_focal_points(center=center, radius=radius, point_num=point_num)
                stm = stm_dev2(center=center, radius=radius, point_num=point_num)

                autd.send((m, g))
                print(f"x: {x:.2f}mm, y: {y:.2f}mm, z: {z:.2f}mm")