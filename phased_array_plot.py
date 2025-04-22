import matplotlib.pyplot as plt
import numpy as np

# 各AUTD3ユニットの定数（単位: mm）
TRANSDUCER_PITCH = 10.16  # トランスデューサ間のピッチ
NUM_COLS = 18 # 列数
NUM_ROWS = 14 # 行数
DEVICE_WIDTH = 192.0 # AUTDの幅
DEVICE_HEIGHT = 151.4 # AUTDの高さ

# トランスデューサ座標の生成（1台分）
def generate_transducer_positions():
    positions = []
    for i in range(NUM_COLS):
        for j in range(NUM_ROWS):
            # ネジ穴用に3つ抜けている　[(1,1), (2,1), (16,1)]
            if not (i in [1, 2, 16] and j == 1):
                x = i * TRANSDUCER_PITCH + 5.08  # 左マージン5.08mm
                y = j * TRANSDUCER_PITCH + 5.08  # 下マージン5.08mm
                positions.append([x, y])
    return np.array(positions)

# AUTD3配置（3行2列）に基づいて全体配置を生成
autd_arrangement = [
    [0, 0],
    [0, -1],
    [0, -2],
    [1, -2],
    [1, -1],
    [1, 0]
]

unit_positions = []
for col, row in autd_arrangement:
    offset_x = col * DEVICE_WIDTH
    offset_y = row * DEVICE_HEIGHT
    unit_pos = generate_transducer_positions() + np.array([offset_x, offset_y])
    unit_positions.append(unit_pos)

# 全トランスデューサ座標を結合
all_positions = np.vstack(unit_positions)

# 描画
plt.figure(figsize=(8, 5))
plt.scatter(all_positions[:, 0], all_positions[:, 1], s=10)
plt.title(f"Transducer Layout (Total: {len(all_positions)})")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.axis("equal")
plt.grid(True, alpha=0.5)
plt.show()
