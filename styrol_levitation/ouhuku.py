import os
import time
import numpy as np, os, keyboard
from pyautd3 import (
    AUTD3, Controller, FociSTM, Hz, Silencer, Static,
)
from pyautd3_link_soem import SOEM, SOEMOption, Status # SOEMを使用するために追加した
from pyautd3.link.simulator import Simulator # シミュレータを使用するために追加した
from pyautd3_emulator import Emulator # エミュレータを使用するために追加した

# 往復軌道音場を形成するためのファイル

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
def stm_ouhuku(center: np.ndarray, radius: float, point_num: int) -> FociSTM:
    # 正方向角リスト
    angles_fwd = [np.pi/8 + 2.0 * np.pi * i / point_num for i in range(point_num)] # 穴の位置をずらすためにπ/8を加える
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

# 往復軌道を高速で回すための関数
def stm_ouhuku_modified(center: np.ndarray, radius: float, point_num: int) -> FociSTM:
    # 正方向角リスト
    angles_fwd = [np.pi/8 + 2.0 * np.pi * i / point_num for i in range(point_num)] # 穴の位置をずらすためにπ/8を加える
    # 逆方向角リスト
    angles_rev = angles_fwd[::-1][1:-1] # [::-1]で逆順にし、[1:-1]で最初と最後を除く
    angles = []

    for _ in range(point_num):
        tmp = []
        tmp = angles_fwd + angles_rev # forward + reverseの2周分を連結
        angles += tmp
        angles_fwd = angles_fwd[1:] + angles_fwd[:1] # 1つずらす
        angles_rev = angles_fwd[::-1][1:-1] # [::-1]で逆順にし、[1:-1]で最初と最後を除く

    # 円軌道上に焦点を配置するための時空間変調
    foci = (
        center + radius * np.array([np.cos(a), np.sin(a), 0.0])
        for a in angles
    )
    return FociSTM(foci=foci, config=9 * Hz).into_nearest()

# 対向する4点が弱くなる往復軌道を形成するための関数
def stm_balance(center: np.ndarray, radius: float, point_num: int) -> FociSTM:
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
    return FociSTM(foci=foci, config=84 * Hz).into_nearest()

# === 追加: パターン時間スケジュール ===
HOLD = 5.0               # パターン0/4の保持時間[秒]
TRANSIENT = 1.0  # 一度だけ経由する時間（約0.2s）

# 0→(1→2→3)→4→(5→6→7)→0…
SCHEDULE = [
    (0, HOLD),
    (1, TRANSIENT),
    (2, TRANSIENT),
    (3, TRANSIENT),
    (4, HOLD),
    (5, TRANSIENT),
    (6, TRANSIENT),
    (7, TRANSIENT),
]

TOTAL_PERIOD = sum(d for _, d in SCHEDULE)

def get_scheduled_pattern_index(elapsed: float) -> int:
    """経過時間からスケジュール上の現在パターン(0..7)を返す"""
    t = elapsed % TOTAL_PERIOD
    acc = 0.0
    for pat, dur in SCHEDULE:
        if acc + dur > t:
            return pat
        acc += dur
    return SCHEDULE[-1][0]  # 念のため


# 時間経過に応じてC型パターンを切り替える関数
def stm_gradually(center: np.ndarray, radius: float) -> FociSTM:
    if not hasattr(stm_gradually, "_start_time"):
        stm_gradually._start_time = time.time()
    if not hasattr(stm_gradually, "_last_index"):
        stm_gradually._last_index = -1

    # 経過秒
    elapsed = time.time() - stm_gradually._start_time
    pattern_index = get_scheduled_pattern_index(elapsed)

    # 切替ログ
    if pattern_index != stm_gradually._last_index:
        print(f"パターンが切り替わりました: パターン {pattern_index} "
              f"({int(elapsed)} 秒経過)")
        stm_gradually._last_index = pattern_index

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

    # angles_patterns = [
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    #     [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    #     [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0],
    #     [4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1],
    #     [5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2],
    #     [6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3],
    #     [7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4],
    #     [8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5],
    #     [9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6],
    #     [10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7],
    #     [11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    # ]

    current_angles = [k * np.pi / 4 for k in angles_patterns[pattern_index]]
    foci = [center + radius * np.array([np.cos(a), np.sin(a), 0.0])
            for a in current_angles]

    return FociSTM(foci=foci, config=70 * Hz).into_nearest()

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

            # 現在のパターンインデックスを取得（パターン変化検出のため）
            if hasattr(stm_gradually, "_start_time"):
                elapsed = time.time() - stm_gradually._start_time
                current_pattern_index = get_scheduled_pattern_index(elapsed)
            else:
                current_pattern_index = 0

            
            # キーボード操作またはパターン変化があった場合にSTMを更新
            if (x != prev_x or y != prev_y or z != prev_z or current_pattern_index != last_pattern_index):

                prev_x, prev_y, prev_z = x, y, z # 前回の座標を更新
                last_pattern_index = current_pattern_index # 前回のパターンインデックスを更新

                center = autd.center() + np.array([x, y, z]) # 円軌道の中心座標を更新

                stm = stm_ouhuku(center=center, radius=radius, point_num=point_num)

                autd.send((m, stm))
                print(f"x: {x:.2f}mm, y: {y:.2f}mm, z: {z:.2f}mm")