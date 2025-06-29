import numpy as np
import matplotlib.pyplot as plt

# STMの軌道を可視化するためのコード
dev_radius = 24.0
point_num = 8

# 正方向 → 逆方向で 14点
angles_fwd = [2.0 * np.pi * i / point_num for i in range(point_num)]
angles_rev = angles_fwd[::-1][1:-1]
dev_angles = angles_fwd + angles_rev  # 14個

# 重複角度は半径を 1 mm ずつ外側へずらす
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
    if i < 8:
        label_pos = (0.85 * x, 0.85 * y)
    else:
        label_pos = (1.15 * x, 1.15 * y)
    plt.text(label_pos[0], label_pos[1], str(i + 1),
             ha='center', va='center', fontsize=10, fontweight='bold')

plt.gca().set_aspect('equal', adjustable='box')
plt.title('STM path (14 points)') 
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.xlim(-30, 30)
plt.ylim(-30, 30)
plt.grid(True)

# 点1-2間に穴が空いた場合の図を追加する
angles = angles_fwd[7:] + angles_fwd[0:7]
angles_rev = angles[::-1][1:-1]
dev_angles = angles + angles_rev  # 14個

# 重複角度は半径を 1 mm ずつ外側へずらす
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
    if i < 8:
        label_pos = (0.85 * x, 0.85 * y)
    else:
        label_pos = (1.15 * x, 1.15 * y)
    plt.text(label_pos[0], label_pos[1], str(i + 1),
             ha='center', va='center', fontsize=10, fontweight='bold')

plt.gca().set_aspect('equal', adjustable='box')
plt.title('STM path (14 points)') 
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.xlim(-30, 30)
plt.ylim(-30, 30)
plt.grid(True)


balloon_radius = 24.0
balloon_point_num = 8
balloon_angles = [2.0 * np.pi * i / balloon_point_num for i in range(balloon_point_num)]
balloon_xy = [(balloon_radius * np.cos(a), balloon_radius * np.sin(a)) for a in balloon_angles]

plt.figure()
for i, (x, y) in enumerate(balloon_xy):
    plt.plot(x, y, marker='o', color='grey')
    xn, yn = balloon_xy[(i + 1) % len(balloon_xy)]  # 全周を矢印で接続
    plt.annotate('', xy=(xn, yn), xytext=(x, y),
                 arrowprops=dict(arrowstyle='->'))
    # 番号ラベル
    if i < 8:  # 1-8: 内側へ
        label_pos = (0.85 * x, 0.85 * y)
    plt.text(label_pos[0], label_pos[1], str(i + 1),
             ha='center', va='center', fontsize=10, fontweight='bold')
    
plt.gca().set_aspect('equal', adjustable='box')
plt.title('STM path (8 points)')
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.xlim(-30, 30)
plt.ylim(-30, 30)
plt.grid(True)

plt.figure()
# 円形の線を描画（緑色）
circle = plt.Circle((0, 0), balloon_radius, fill=False, color='grey', linewidth=3)
plt.gca().add_patch(circle)

# 点とラベルを描画
for i, (x, y) in enumerate(balloon_xy):
    plt.plot(x, y, marker='o', color='red', markersize=15)
    # 番号ラベル
    if i < 8:  # 1-8: 内側へ
        label_pos = (0.85 * x, 0.85 * y)
    plt.text(label_pos[0], label_pos[1], str(i + 1),
             ha='center', va='center', fontsize=15, fontweight='bold')
    
plt.gca().set_aspect('equal', adjustable='box')
# plt.title('STM path (8 points)')
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.xlim(-30, 30)
plt.ylim(-30, 30)
plt.grid(True)

plt.figure()
# 1-7点目のみプロット（8点目を除外してC型に）
for i in range(7):
    x, y = balloon_xy[i]
    plt.plot(x, y, marker='o', color='grey')
    if i < 6:
        xn, yn = balloon_xy[i + 1]
        plt.annotate('', xy=(xn, yn), xytext=(x, y),
                     arrowprops=dict(arrowstyle='->'))
    # 番号ラベル
    label_pos = (0.85 * x, 0.85 * y)
    plt.text(label_pos[0], label_pos[1], str(i + 1),
             ha='center', va='center', fontsize=10, fontweight='bold')

# 7点目から1点目への矢印を追加
plt.annotate('', xy=(balloon_xy[0]), xytext=(balloon_xy[6]),
             arrowprops=dict(arrowstyle='->'))

plt.gca().set_aspect('equal', adjustable='box')
plt.title('STM path (7 points)')
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.xlim(-30, 30)
plt.ylim(-30, 30)
plt.grid(True)

plt.show()