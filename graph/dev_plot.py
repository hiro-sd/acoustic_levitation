import numpy as np
import matplotlib.pyplot as plt

# AUTDの配置関連の定数とフォーマット
TRANSDUCER_PITCH = 10.16
NUM_COLS = 18
NUM_ROWS = 14
DEVICE_WIDTH = 192.0
DEVICE_HEIGHT = 151.4

# トランスデューサ座標の生成（1台分）
def generate_transducer_positions():
    positions = []
    for i in range(NUM_COLS):
        for j in range(NUM_ROWS):
            if not (i in [1, 2, 16] and j == 1):
                x = i * TRANSDUCER_PITCH + 5.08
                y = j * TRANSDUCER_PITCH + 5.08
                positions.append([x, y])
    return np.array(positions)

# AUTD3配置（3行2列）
autd_arrangement = [
    [0, 0],
    [0, -1],
    [0, -2],
    [1, -2],
    [1, -1],
    [1, 0]
]

# 既存のSTMパスのコード
dev_radius = 24.0
point_num = 7

# ...既存のSTMパス生成コード...
angles_fwd = [2.0 * np.pi * i / point_num for i in range(point_num)]
angles_rev = angles_fwd[::-1][1:-1]
dev_angles = angles_fwd + angles_rev

offset_delta = 1.5
seen = {}
dev_xy = []
for a in dev_angles:
    k = seen.get(a, 0)
    r = dev_radius + k * offset_delta
    dev_xy.append((r * np.cos(a), r * np.sin(a)))
    seen[a] = k + 1

# プロット（STMパスとAUTD配置を重ねて表示）
plt.figure(figsize=(12, 8))

# AUTDの配置をプロット
unit_positions = []
for col, row in autd_arrangement:
    offset_x = col * DEVICE_WIDTH - DEVICE_WIDTH  # 中心を原点に調整
    offset_y = row * DEVICE_HEIGHT + DEVICE_HEIGHT / 2  # 中心を原点に調整
    unit_pos = generate_transducer_positions() + np.array([offset_x, offset_y])
    unit_positions.append(unit_pos)
    
all_positions = np.vstack(unit_positions)
plt.scatter(all_positions[:, 0], all_positions[:, 1], s=10, color='grey', label='Transducers')

# STMパスをプロット
for i, (x, y) in enumerate(dev_xy):
    plt.plot(x, y, marker='o', color='blue')
    xn, yn = dev_xy[(i + 1) % len(dev_xy)]
    plt.annotate('', xy=(xn, yn), xytext=(x, y),
                 arrowprops=dict(arrowstyle='->'))
    
    if i < 7:
        label_pos = (0.7 * x, 0.7 * y)
    else:
        label_pos = (1.3 * x, 1.3 * y)
    plt.text(label_pos[0], label_pos[1], str(i + 1),
             ha='center', va='center', fontsize=7, color='black', fontweight='bold')

plt.gca().set_aspect('equal', adjustable='box')
plt.title('STM path with AUTD array layout')
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.xlim(-215, 215)
plt.ylim(-245, 245)
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()