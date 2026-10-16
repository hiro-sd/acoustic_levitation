import sys
from pathlib import Path

import keyboard
import numpy as np
from autd3.commands import SetSilencer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.autd_client import AutdController
from tracking.core.autd_patterns import (
    make_gspat_pattern,
    make_gspat_pattern_stm,
    make_single_focus_stm,
)
from tracking.core.autd_sender import make_autd_arrangement, make_constant_modulation

# いろいろ試すためのファイル

# AUTDの配置
autd_arrangement = make_autd_arrangement()

# FociSTMで往復軌道を形成するための関数
def stm_dev(center: np.ndarray, radius: float, point_num: int):
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
    return make_single_focus_stm(foci, 4.0)

# 静的8焦点を生成する関数
def multi_focal_points(geometry, center, radius, point_num):
    angles = [np.pi/8 + 2.0 * np.pi * i / point_num for i in range(point_num)]
    points = [
        center + radius * np.array([np.cos(angle), np.sin(angle), 0.0])
        for angle in angles
    ]
    return make_gspat_pattern(geometry, points, pressure_pa=5e4, repeat=100)

# ２つの半周往復軌道を作成する関数
# GSPATで点1と点5に焦点を配置し、それらをGainSTMで半周ずつ回す
def stm_dev2(geometry, center: np.ndarray, radius: float, point_num: int):
    frames = []
    
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
            
            frames.append([p1, p2])
        
        # 逆方向の半周分の点を生成
        for step in range(point_num // 2 - 2, 0, -1):
            idx1 = step
            idx2 = (step + point_num // 2) % point_num
            
            p1 = center + radius * np.array([np.cos(angles[idx1]), np.sin(angles[idx1]), 0.0])
            p2 = center + radius * np.array([np.cos(angles[idx2]), np.sin(angles[idx2]), 0.0])
            
            frames.append([p1, p2])
    return make_gspat_pattern_stm(
        geometry,
        frames,
        21.0,
        pressure_pa=5e4,
        repeat=100,
    )

# 円周上のランダムな100点を周期的に切り替えるSTM
def stm_random(center: np.ndarray, radius: float):
    num_points = 100
    # 0～2πの範囲でランダムな角度を100個生成
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    # 角度をソートしても良いが、ランダムなままでもOK
    foci = (
        center + radius * np.array([np.cos(a), np.sin(a), 0.0])
        for a in angles
    )
    return make_single_focus_stm(foci, 9.0)

# 逆向きの往復軌道を交互に繰り返すSTM
def stm_repeat(center: np.ndarray, radius: float, point_num: int):
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
    return make_single_focus_stm(foci, 1.0)

def stm_double(geometry, center: np.ndarray, radius: float, point_num: int):
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

    frames = [[points[idx1], points[idx2]] for idx1, idx2 in pair_indices]
    return make_gspat_pattern_stm(
        geometry,
        frames,
        70.0,
        pressure_pa=5e4,
        repeat=100,
    )

# 対向する位置に交互に焦点を作るSTM
def stm_opposite(center: np.ndarray, radius: float, point_num: int):
    # 正方向角リスト
    angles = [np.pi/8 + 2.0 * np.pi * i / point_num for i in range(point_num)] # 穴の位置をずらすためにπ/8を加える
    angles_new= [val for pair in zip(angles[:point_num//2], angles[point_num//2:]) for val in pair] # 対向する位置に交互に焦点を作るように順番を入れ替え
    
    # 円軌道上に焦点を配置する
    foci = (
        center + radius * np.array([np.cos(a), np.sin(a), 0.0])
        for a in angles_new
    )
    return make_single_focus_stm(foci, 100.0)

# SOEMのエラーハンドラ
# def err_handler(slave: int, status: Status) -> None:
#     print(f"slave [{slave}]: {status}")
#     if status == Status.Lost():
#         os._exit(-1)

if __name__ == "__main__":
    with AutdController.open(autd_arrangement) as autd:
        firmware_version = autd.read_firmware_version()
        print(
            "\n".join(
                [f"[{i}]: {firm}" for i, firm in enumerate(firmware_version)],
            ),
        )

        autd.send(SetSilencer())

        m = make_constant_modulation(1.0)

        point_num = 8 # 円周上の点の数
        radius = 23.0 # 円の半径 (小球は19 mm)
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
                stm = stm_random(center=center, radius=radius)

                autd.send((m, stm))
                print(f"x: {x:.2f}mm, y: {y:.2f}mm, z: {z:.2f}mm")
