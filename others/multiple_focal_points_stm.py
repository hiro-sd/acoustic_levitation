import numpy as np, os, keyboard
from pyautd3 import (
    AUTD3, Controller, ControlPoint, ControlPoints, EmitIntensity, FociSTM, Focus, FocusOption, GainSTM, GainSTMMode, GainSTMOption, Group, Hz, Null, Phase, Silencer, Static,
)
from pyautd3.gain.holo import GSPAT, EmissionConstraint, GSPATOption, NalgebraBackend, Pa
# from pyautd3_link_soem import SOEM, SOEMOption, Status # SOEMを使用するために追加した
from pyautd3.link.twincat import TwinCAT # TwinCATを使用するために追加した
from pyautd3.link.simulator import Simulator # シミュレータを使用するために追加した
from pyautd3_emulator import Emulator # エミュレータを使用するために追加した

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

# FociSTMでn焦点を作成する関数
def multiple_foci_stm(n):
    FociSTM(
        foci=(
            ControlPoints(
                points=[
                    ControlPoint(
                        point=center + radius * np.array([np.cos(theta + 2.0 * np.pi * k / n), np.sin(theta + 2.0 * np.pi * k / n), 0.0]), 
                        phase_offset=Phase.ZERO,
                        )
                             for k in range(n) 
                        ],
                
                        intensity=EmitIntensity.MAX, 
                        )
                        for theta in (2.0 * np.pi * i / point_num for i in range(point_num))
                    ),
                    config=100 * Hz,
                    ).into_nearest()

# GSPATアルゴリズムで2つの焦点を生成する関数 (対角線上に2焦点を生成する)
def make_gspat_gain(center, theta, radius):
    p1 = center + radius * np.array([np.cos(theta), np.sin(theta), 0.0])
    p2 = center + radius * np.array([np.cos(theta + np.pi), np.sin(theta + np.pi), 0.0])
    return GSPAT(
        foci=
            [(p1, 5e5 * Pa), (p2, 5e5 * Pa)],
            option = GSPATOption(
                repeat = 100,
                constraint = EmissionConstraint.Clamp(EmitIntensity.MIN, EmitIntensity.MAX),
            ),
            backend = NalgebraBackend(),
        )

if __name__ == "__main__":
    with Controller.open(
        autd_arrangement,
        # Simulator("127.0.0.1:8080"), # シミュレータを使用するために追加した
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
        m = Static(intensity=int(0xFF*0.8)) # 振幅変調を行わず、常に同じ振幅を出力する

        point_num = 8 # 円周上の点の数
        radius = 19.0 # 円の半径
        x, y, z = 0.0, 0.0, 400.0 # x,y,z座標の初期値
        x_min, x_max = -100.0, 100.0 # x座標の最小値と最大値
        y_min, y_max = -150.0, 150.0 # y座標の最小値と最大値
        z_min, z_max = 200.0, 700.0 # 244.0, 642.0 # z座標の最小値と最大値
        prev_x, prev_y, prev_z = None, None, None # 前回のx,y,z座標を保存するための変数
        step = 5.0 # 1回の操作で移動する距離

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
                prev_x, prev_y, prev_z = x, y, z # 前回の座標を更新

                center = autd.center() + np.array([x, y, z]) # 円軌道の中心座標を更新

                # 円軌道上に焦点を配置するための時空間変調 (FociSTMを使用する場合) 
                # stm = multiple_foci_stm(2)

                # GSPATで多焦点を作成する場合
                gains = [
                    make_gspat_gain(center, np.pi/8 + 2.0 * np.pi * i / point_num, radius)
                    for i in range(point_num)
                ]

                stm = GainSTM(
                    gains,
                    config = 120 * Hz,
                    option = GainSTMOption(
                        mode = GainSTMMode.PhaseIntensityFull,
                    ),
                ).into_nearest()

                autd.send((m, stm))
                print(f"x: {x:.2f}mm, y: {y:.2f}mm, z: {z:.2f}mm")