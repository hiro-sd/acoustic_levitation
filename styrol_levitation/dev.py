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

# ２つの半周往復軌道を作成する関数
# GSPATで点1と点5に焦点を配置し、それらをGainSTMで半周ずつ回す
def stm_dev2(center: np.ndarray, radius: float, point_num: int) -> GainSTM: # 0.75が最適
    gains = []
    
    # 基本の角度リストを生成
    base_angles = [np.pi/8 + 2.0 * np.pi * i / point_num for i in range(point_num)]
    
    # 各周回での開始点オフセット
    for cycle in range(point_num):
        # 現在の周回での角度リスト（開始点をずらす）
        angles = base_angles[cycle:] + base_angles[:cycle]
        
        # 半周分の点を生成（point_num//2個）
        for step in range(point_num // 2):
            # 2つの焦点の位置を計算（対角の位置）
            idx1 = step
            idx2 = (step + point_num // 2) % point_num
            
            p1 = center + radius * np.array([np.cos(angles[idx1]), np.sin(angles[idx1]), 0.0])
            p2 = center + radius * np.array([np.cos(angles[idx2]), np.sin(angles[idx2]), 0.0])
            
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
        
        # 逆方向の半周分の点を生成
        for step in range(point_num // 2 - 2, 0, -1):
            idx1 = step
            idx2 = (step + point_num // 2) % point_num
            
            p1 = center + radius * np.array([np.cos(angles[idx1]), np.sin(angles[idx1]), 0.0])
            p2 = center + radius * np.array([np.cos(angles[idx2]), np.sin(angles[idx2]), 0.0])
            
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
    return GainSTM(
        gains,
        config=21 * Hz,
        option = GainSTMOption(
                        mode = GainSTMMode.PhaseIntensityFull,
                    ),
    ).into_nearest()

# 円周上のランダムな100点を周期的に切り替えるSTM
def stm_random(center: np.ndarray, radius: float) -> FociSTM:
    num_points = 100
    # 0～2πの範囲でランダムな角度を100個生成
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    # 角度をソートしても良いが、ランダムなままでもOK
    foci = (
        center + radius * np.array([np.cos(a), np.sin(a), 0.0])
        for a in angles
    )
    return FociSTM(foci=foci, config=9 * Hz).into_nearest()

# 逆向きの往復軌道を交互に繰り返すSTM
def stm_repeat(center: np.ndarray, radius: float, point_num: int) -> FociSTM:
    # 正方向角リスト
    angles_fwd = [np.pi/8 + 2.0 * np.pi * i / point_num for i in range(point_num)] # 穴の位置をずらすためにπ/8を加える
    angles_basic = angles_fwd + angles_fwd[1:-1]
    # 逆方向角リスト
    angles_rev = angles_fwd[4:] + angles_fwd[:4]
    angles_opposite = angles_rev + angles_rev[1:-1]

    angles = (angles_basic * 10) + (angles_opposite * 10)
    # 円軌道上に焦点を配置する
    foci = (
        center + radius * np.array([np.cos(a), np.sin(a), 0.0])
        for a in angles
    )
    return FociSTM(foci=foci, config=1 * Hz).into_nearest()

def stm_double(center: np.ndarray, radius: float, point_num: int) -> GainSTM:
    # 円周上に等間隔8点
    angles = [np.pi/8 + 2.0 * np.pi * i / point_num for i in range(point_num)]
    # 各点の座標
    points = [center + radius * np.array([np.cos(a), np.sin(a), 0.0]) for a in angles]

    # ペアのインデックス列（0始まり）
    pair_indices = [
        (0, 4), (1, 5), (2, 6), (3, 7),
        (4, 0), (5, 1), (6, 2), (7, 3),
        (6, 2), (5, 1), (4, 0), (3, 7), (2, 6), (1, 5)
    ]

    gains = []
    for idx1, idx2 in pair_indices:
        focal_points = GSPAT(
            foci=[
                (points[idx1], 5e4 * Pa),
                (points[idx2], 5e4 * Pa),
            ],
            option=GSPATOption(
                repeat=100,
                constraint=EmissionConstraint.Clamp(EmitIntensity.MIN, EmitIntensity.MAX),
            ),
            backend=NalgebraBackend(),
        )
        gains.append(focal_points)

    return GainSTM(
        gains,
        config=72 * Hz,
        option=GainSTMOption(
            mode=GainSTMMode.PhaseIntensityFull,
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

        m = Static(intensity=int(0xFF * 0.78)) # 振幅変調を行わず、常に同じ振幅を出力する

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

                # g = multi_focal_points(center=center, radius=radius, point_num=point_num)
                stm = stm_double(center=center, radius=radius, point_num=point_num)

                autd.send((m, stm))
                print(f"x: {x:.2f}mm, y: {y:.2f}mm, z: {z:.2f}mm")