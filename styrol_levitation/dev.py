import os
import time
import numpy as np, os, keyboard
from pyautd3 import (
    AUTD3, Controller, FociSTM, Focus, FocusOption, GainSTM, GainSTMMode, GainSTMOption, Group, Hz, Null, Silencer, Static,
)
from pyautd3_link_soem import SOEM, SOEMOption, Status # SOEMを使用するために追加した
from pyautd3.link.simulator import Simulator # シミュレータを使用するために追加した
from pyautd3_emulator import Emulator # エミュレータを使用するために追加した

# 色々検証してみるためのファイル

# AUTDの配置
autd_arrangement = [
    AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]), 
    AUTD3(pos=[0.0, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[0.0, -2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, -2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),
    ]

# FociSTMで円軌道を交互に回す関数
def stm_alternately(center: np.ndarray, radius: float, point_num: int) -> FociSTM:
    # 正方向角リスト
    angles_fwd = [2.0 * np.pi * i / point_num for i in range(point_num)]
    # 逆方向角リスト
    angles_rev = angles_fwd[::-1][1:-1] # [::-1]で逆順にし、[1:-1]で最初と最後を除く
    # forward + reverse の 2 周分を連結
    angles = angles_fwd + angles_rev 

    # 円軌道上に焦点を配置するための時空間変調
    foci = (
        center + radius * np.array([np.cos(a), np.sin(a), 0.0])
        for a in angles
    )
    return FociSTM(foci=foci, config=70 * Hz).into_nearest()

# 時間経過に応じてC型パターンを切り替える関数
def stm_graduate(center: np.ndarray, radius: float) -> FociSTM:

    if not hasattr(stm_graduate, "_start_time"):
        stm_graduate._start_time = time.time()
    if not hasattr(stm_graduate, "_last_index"):
        stm_graduate._last_index = -1

    # 経過秒
    elapsed = time.time() - stm_graduate._start_time
    pattern_duration = 7.0                              # [s]
    pattern_index = int(elapsed // pattern_duration) % 8 # 0–7

    # 切り替わりを検出してログ
    if pattern_index != stm_graduate._last_index:
        print(f"パターンが切り替わりました: パターン {pattern_index + 1} "
              f"({int(elapsed)} 秒経過)")
        stm_graduate._last_index = pattern_index

    # angles_patterns = [
    #     [0, 1, 2, 3, 4, 5, 4, 3, 2, 1],
    #     [1, 2, 3, 4, 5, 0, 5, 4, 3, 2],
    #     [2, 3, 4, 5, 0, 1, 0, 5, 4, 3],
    #     [3, 4, 5, 0, 1, 2, 1, 0, 5, 4],
    #     [4, 5, 0, 1, 2, 3, 2, 1, 0, 5],
    #     [5, 0, 1, 2, 3, 4, 3, 2, 1, 0],
    # ]

    angles_patterns = [
        [0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 0, 7, 6, 5, 4, 3, 2],
        [2, 3, 4, 5, 6, 7, 0, 1, 0, 7, 6, 5, 4, 3],
        [3, 4, 5, 6, 7, 0, 1, 2, 1, 0, 7, 6, 5, 4],
        [4, 5, 6, 7, 0, 1, 2, 3, 2, 1, 0, 7, 6, 5],
        [5, 6, 7, 0, 1, 2, 3, 4, 3, 2 ,1 ,0, 7, 6],
        [6, 7, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 7],
        [7, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0],
    ]

    current_angles = [k * np.pi / 4 for k in angles_patterns[pattern_index]]

    foci = [
        center + radius * np.array([np.cos(a), np.sin(a), 0.0])
        for a in current_angles
    ]

    return FociSTM(foci=foci, config=70 * Hz).into_nearest()

# SOEMのエラーハンドラ
def err_handler(slave: int, status: Status) -> None:
    print(f"slave [{slave}]: {status}")
    if status == Status.Lost():
        os._exit(-1)

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

        # 直径4cm半球のパラメータ(いまのところ)
        # m = Static(intensity=int(0xFF * 0.65)) # 振幅変調を行わず、常に同じ振幅を出力する
        # radius = 23.0 # 円の半径

        # 直径4.5cm球?のパラメータ(いまのところ)
        m = Static(intensity=int(0xFF)) # 振幅変調を行わず、常に同じ振幅を出力する

        point_num = 8 # 円周上の点の数
        radius = 24.0 # 円の半径
        x, y, z = 0.0, 0.0, 400.0 # x,y,z座標の初期値
        x_min, x_max = -100.0, 100.0 # x座標の最小値と最大値
        y_min, y_max = -150.0, 150.0 # y座標の最小値と最大値
        z_min, z_max = 200.0, 700.0 # 244.0, 642.0 # z座標の最小値と最大値
        prev_x, prev_y, prev_z = None, None, None # 前回のx,y,z座標を保存するための変数
        step = 0.01 # 1回の操作で移動する距離

        # 傾き角度の初期化
        current_tilt = 0.0  # 現在の傾き（ラジアン）
        max_tilt = np.pi / 2  # 最大傾き（90度）
        tilt_step = 0.001  # 1回あたりの傾き変化量（ラジアン）

        # パターン変化の検出用変数
        last_pattern_index = -1

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

            # 傾き方向の設定
            if keyboard.is_pressed("t"):
                # 最大傾きを超えない範囲で傾きを増加
                current_tilt = min(current_tilt + tilt_step, max_tilt)
            elif keyboard.is_pressed("r"):
                # 最小傾きを超えない範囲で傾きを減少
                current_tilt = max(current_tilt - tilt_step, -max_tilt)
            
            # 現在のパターンインデックスを取得（パターン変化検出のため）
            if hasattr(stm_graduate, "_start_time"):
                elapsed = time.time() - stm_graduate._start_time
                current_pattern_index = int(elapsed // 7.0) % 8
            else:
                current_pattern_index = 0
            
            # キーボード操作、傾き変化、またはパターン変化があった場合にSTMを更新
            if (x != prev_x or y != prev_y or z != prev_z or 
                keyboard.is_pressed("t") or keyboard.is_pressed("r") or 
                current_pattern_index != last_pattern_index):
                
                prev_x, prev_y, prev_z = x, y, z # 前回の座標を更新
                last_pattern_index = current_pattern_index # 前回のパターンインデックスを更新

                center = autd.center() + np.array([x, y, z]) # 円軌道の中心座標を更新

                # 回転行列を作成（x軸周りの回転） (current_tiltだけ傾ける)
                # rotation_matrix = np.array([
                #     [1, 0, 0],
                #     [0, np.cos(current_tilt), -np.sin(current_tilt)],
                #     [0, np.sin(current_tilt), np.cos(current_tilt)]
                # ])

                # stm = FociSTM(
                #     foci = (
                #         # 水平な円を回転行列で変換し、現在の傾きに応じた円にする (@は行列の積を表す)
                #         center + rotation_matrix @ (radius * np.array([np.cos(theta), np.sin(theta), 0]))
                #         for theta in (2.0 * np.pi * i / point_num for i in range(point_num))
                #         ),
                #     config = 100 * Hz,
                # ).into_nearest()

                stm = stm_graduate(center=center, radius=radius)

                autd.send((m, stm))
                print(f"x: {x:.2f}mm, y: {y:.2f}mm, z: {z:.2f}mm, 傾斜: {np.degrees(current_tilt):.1f}度")

            # # 毎ループでランダムなSTMパターンを生成
            # center = autd.center() + np.array([x, y, z])
            # stm = stm_random(center=center, radius=radius, point_num=point_num)
            # autd.send((m, stm))

            # # 位置が変更された場合のみ座標を表示
            # if (x != prev_x or y != prev_y or z != prev_z or keyboard.is_pressed("t") or keyboard.is_pressed("r")):
            #     prev_x, prev_y, prev_z = x, y, z
            #     print(f"x: {x:.2f}mm, y: {y:.2f}mm, z: {z:.2f}mm, 傾斜: {np.degrees(current_tilt):.1f}度")