import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# STMの軌道を可視化するためのコード
N = 8
radius_fwd = 18.5  # 往路の円弧半径
radius_point = 19.0 # 点の半径
radius_rev = 19.5  # 復路の円弧半径
angles = [2 * np.pi * i / N for i in range(N)]

fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.grid(True)

# 点とラベル（1-8: 内側, radius_point上に配置）
for i, a in enumerate(angles):
    x = radius_point * np.cos(a)
    y = radius_point * np.sin(a)
    ax.plot(x, y, 'o', color='red', markersize=15)
    ax.text(0.85*x, 0.85*y, str(i+1), ha='center', va='center', fontsize=15, fontweight='bold')

# 1→2→…→8（1-8間は結ばない, radius_fwd上）
for i in range(N-1):
    theta1 = np.rad2deg(angles[i])
    theta2 = np.rad2deg(angles[i+1])
    arc = Arc((0,0), 2*radius_fwd, 2*radius_fwd, theta1=theta1, theta2=theta2, color='grey', linewidth=3)
    ax.add_patch(arc)
    # 矢印
    mid_angle = np.deg2rad((theta1 + theta2) / 2)
    x = radius_fwd * np.cos(mid_angle)
    y = radius_fwd * np.sin(mid_angle)
    dx = -0.5 * radius_fwd * np.sin(mid_angle)
    dy = 0.5 * radius_fwd * np.cos(mid_angle)
    ax.arrow(x, y, dx*0.1, dy*0.1, head_width=1.2, head_length=2, fc='grey', ec='grey')

# 8→7→…→1（復路、9-14の番号を点のすぐ外側に表示、矢印は逆向き, radius_rev上）
for idx, i in enumerate(range(N-1, 0, -1)):
    theta1 = np.rad2deg(angles[i])
    theta2 = np.rad2deg(angles[i-1])
    arc = Arc((0,0), 2*radius_rev, 2*radius_rev, theta1=theta2, theta2=theta1, color='grey', linewidth=3)
    ax.add_patch(arc)
    # 矢印（逆向き）
    mid_angle = np.deg2rad((theta1 + theta2) / 2)
    x = radius_rev * np.cos(mid_angle)
    y = radius_rev * np.sin(mid_angle)
    dx = 0.5 * radius_rev * np.sin(mid_angle)
    dy = -0.5 * radius_rev * np.cos(mid_angle)
    ax.arrow(x, y, dx*0.1, dy*0.1, head_width=1.2, head_length=2, fc='grey', ec='grey')
    # ラベル（9-14を点のすぐ外側に）
    label = 9 + idx
    if label > 14:
        continue
    px = radius_point * np.cos(angles[i-1])
    py = radius_point * np.sin(angles[i-1])
    lx = 1.15 * px
    ly = 1.15 * py
    ax.text(lx, ly, str(label), ha='center', va='center', fontsize=15, fontweight='bold')

# 2→1も追加（ラベルは不要, radius_rev上）
theta1 = np.rad2deg(angles[1])
theta2 = np.rad2deg(angles[0])
arc = Arc((0,0), 2*radius_rev, 2*radius_rev, theta1=theta2, theta2=theta1, color='grey', linewidth=3)
ax.add_patch(arc)
mid_angle = np.deg2rad((theta1 + theta2) / 2)
x = radius_rev * np.cos(mid_angle)
y = radius_rev * np.sin(mid_angle)
dx = 0.5 * radius_rev * np.sin(mid_angle)
dy = -0.5 * radius_rev * np.cos(mid_angle)
ax.arrow(x, y, dx*0.1, dy*0.1, head_width=1.2, head_length=2, fc='grey', ec='grey')

balloon_radius = 19.0
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
plt.xlim(-25, 25)
plt.ylim(-25, 25)
plt.grid(True)

fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.grid(True)

# 点とラベル（1-8: 内側, radius_point上に配置）
for i, a in enumerate(angles):
    x = radius_point * np.cos(a)
    y = radius_point * np.sin(a)
    ax.plot(x, y, 'o', color='red', markersize=15)
    ax.text(0.85*x, 0.85*y, str(i+1), ha='center', va='center', fontsize=15, fontweight='bold')

# 1→2→…→8（1-8間は結ばない, radius_fwd上）
for i in range(N-1):
    theta1 = np.rad2deg(angles[i])
    theta2 = np.rad2deg(angles[i+1])
    arc = Arc((0,0), 2*balloon_radius, 2*balloon_radius, theta1=theta1, theta2=theta2, color='grey', linewidth=3)
    ax.add_patch(arc)
    # 矢印
    mid_angle = np.deg2rad((theta1 + theta2) / 2)
    x = balloon_radius * np.cos(mid_angle)
    y = balloon_radius * np.sin(mid_angle)
    dx = -0.5 * balloon_radius * np.sin(mid_angle)
    dy = 0.5 * balloon_radius * np.cos(mid_angle)
    ax.arrow(x, y, dx*0.1, dy*0.1, head_width=1.2, head_length=2, fc='grey', ec='grey')

# 8→1の円弧と矢印を追加
theta1 = np.rad2deg(angles[N-1])  # 8番の角度
theta2 = np.rad2deg(angles[0])    # 1番の角度
# 8番から1番への角度差が大きい場合の処理
if theta1 - theta2 > 180:
    theta2 += 360
arc = Arc((0,0), 2*balloon_radius, 2*balloon_radius, theta1=theta1, theta2=theta2, color='grey', linewidth=3)
ax.add_patch(arc)
# 矢印の位置を修正
mid_angle = angles[N-1] + (2*np.pi - (angles[N-1] - angles[0])) / 2
if mid_angle > 2*np.pi:
    mid_angle -= 2*np.pi
x = balloon_radius * np.cos(mid_angle)
y = balloon_radius * np.sin(mid_angle)
dx = -0.5 * balloon_radius * np.sin(mid_angle)
dy = 0.5 * balloon_radius * np.cos(mid_angle)
ax.arrow(x, y, dx*0.1, dy*0.1, head_width=1.2, head_length=2, fc='grey', ec='grey')

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.xlim(-25, 25)
plt.ylim(-25, 25)
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
plt.xlim(-25, 25)
plt.ylim(-25, 25)
plt.grid(True)

plt.show()