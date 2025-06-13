import numpy as np
import matplotlib.pyplot as plt

# STMの軌道を可視化するためのコード
dev_radius = 24.0
point_num = 6

# 正方向 → 逆方向で 12 点
angles_fwd = [2.0 * np.pi * i / point_num for i in range(point_num)]
angles_rev = angles_fwd[::-1][1:-1]
dev_angles = angles_fwd + angles_rev  # 12 個

# 重複角度は半径を 1 mm ずつ外側へずらす
offset_delta = 1.5
seen = {}
dev_xy = []
for a in dev_angles:
    k = seen.get(a, 0)
    r = dev_radius + k * offset_delta
    dev_xy.append((r * np.cos(a), r * np.sin(a)))
    seen[a] = k + 1

plt.figure()
for i, (x, y) in enumerate(dev_xy):
    plt.plot(x, y, marker='o', color='grey')
    xn, yn = dev_xy[(i + 1) % len(dev_xy)]  # 最後→最初 も含めて矢印
    plt.annotate('', xy=(xn, yn), xytext=(x, y),
                 arrowprops=dict(arrowstyle='->'))
    
    # 番号ラベル
    if i < 7:  # 1-7: 内側へ
        label_pos = (0.85 * x, 0.85 * y)
    else:      # 8-12: 外側へ
        label_pos = (1.15 * x, 1.15 * y)
    plt.text(label_pos[0], label_pos[1], str(i + 1),
             ha='center', va='center', fontsize=10, fontweight='bold')

plt.gca().set_aspect('equal', adjustable='box')
plt.title('STM path (12 points)') 
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.xlim(-30, 30)
plt.ylim(-30, 30)
plt.grid(True)

balloon_radius = 24.0
balloon_point_num = 6
balloon_angles = [2.0 * np.pi * i / balloon_point_num for i in range(balloon_point_num)]
balloon_xy = [(balloon_radius * np.cos(a), balloon_radius * np.sin(a)) for a in balloon_angles]

plt.figure()
for i, (x, y) in enumerate(balloon_xy):
    plt.plot(x, y, marker='o', color='grey')
    xn, yn = balloon_xy[(i + 1) % len(balloon_xy)]  # 全周を矢印で接続
    plt.annotate('', xy=(xn, yn), xytext=(x, y),
                 arrowprops=dict(arrowstyle='->'))
    # 番号ラベル
    if i < 7:  # 1-7: 内側へ
        label_pos = (0.85 * x, 0.85 * y)
    plt.text(label_pos[0], label_pos[1], str(i + 1),
             ha='center', va='center', fontsize=10, fontweight='bold')
    
plt.gca().set_aspect('equal', adjustable='box')
plt.title('STM path (7 points)')
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.xlim(-30, 30)
plt.ylim(-30, 30)
plt.grid(True)

plt.show()