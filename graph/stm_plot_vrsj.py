import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# STMの軌道を可視化するためのコード
N = 8
radius_fwd = 18.5  # 往路の円弧半径
radius_point = 19.0 # 点の半径
radius_rev = 19.5  # 復路の円弧半径
angles = [2 * np.pi * i / N for i in range(N)]
balloon_xy = [(radius_point * np.cos(a), radius_point * np.sin(a)) for a in angles]

fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
ax.set_xlim(-27.5, 27.5)
ax.set_ylim(-27.5, 27.5)
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.grid(True)

# 点とラベル（1-8: 内側, radius_point上に配置）
for i, a in enumerate(angles):
    x = radius_point * np.cos(a)
    y = radius_point * np.sin(a)
    ax.plot(x, y, 'o', color='red', markersize=15)
    ax.text(1.2 * x, 1.2 * y, str(i+1), ha='center', va='center', fontsize=12.5, fontweight='bold')

# Figure1: 8点の外側に2本の曲線矢印（点1→点2, 点5→点6の向き）
arrow_radius = radius_point + 7.5  # 点より外側

# 点1→点2の曲線矢印（外側）
arc1_center = (0, 0)
arc_range_reduction = np.pi/16  # 弧の範囲を短くする角度
arc1_start_angle = np.degrees(angles[0] + arc_range_reduction)
arc1_end_angle = np.degrees(angles[1] - arc_range_reduction)
arc1 = Arc(arc1_center, 2*arrow_radius, 2*arrow_radius, 
           theta1=arc1_start_angle, theta2=arc1_end_angle, 
           linewidth=3, color='grey')
ax.add_patch(arc1)
# 矢印の先端（曲線の終点より少し先に配置）
arrow_extension = np.pi/64  # 曲線の終点から少し先に延ばす角度
x2 = arrow_radius * np.cos(angles[1] - arc_range_reduction + arrow_extension)
y2 = arrow_radius * np.sin(angles[1] - arc_range_reduction + arrow_extension)
# 矢印の向きを計算（接線方向）
arrow_direction_angle = angles[1] - arc_range_reduction + arrow_extension + np.pi/2
dx = 0.3 * np.cos(arrow_direction_angle)
dy = 0.3 * np.sin(arrow_direction_angle)
ax.annotate('', xy=(x2, y2), xytext=(x2-dx, y2-dy), 
            arrowprops=dict(arrowstyle='-|>', lw=3, color='grey', 
                          mutation_scale=25, shrinkA=0, shrinkB=0))

# 点5→点6の曲線矢印（外側）
arc2_start_angle = np.degrees(angles[4] + arc_range_reduction)
arc2_end_angle = np.degrees(angles[5] - arc_range_reduction)
arc2 = Arc(arc1_center, 2*arrow_radius, 2*arrow_radius, 
           theta1=arc2_start_angle, theta2=arc2_end_angle, 
           linewidth=3, color='grey')
ax.add_patch(arc2)
# 矢印の先端（曲線の終点より少し先に配置）
x6 = arrow_radius * np.cos(angles[5] - arc_range_reduction + arrow_extension)
y6 = arrow_radius * np.sin(angles[5] - arc_range_reduction + arrow_extension)
# 矢印の向きを計算（接線方向）
arrow_direction_angle = angles[5] - arc_range_reduction + arrow_extension + np.pi/2
dx = 0.3 * np.cos(arrow_direction_angle)
dy = 0.3 * np.sin(arrow_direction_angle)
ax.annotate('', xy=(x6, y6), xytext=(x6-dx, y6-dy), 
            arrowprops=dict(arrowstyle='-|>', lw=3, color='grey', 
                          mutation_scale=25, shrinkA=0, shrinkB=0))

plt.tight_layout()

# ------------- パラメータ -------------
N               = 8
radius_point    = 19.0               # 点（1〜8）の半径
outer_radius    = radius_point + 7.5 # 外側矢印
inner_radius    = radius_point - 7.5 # 内側矢印（曲線）
inner_label_r   = radius_point * 0.65  # 追加番号 (9〜14) の半径
arc_trim        = np.pi / 64
arrow_span      = np.pi /32
arrow_kw = dict(arrowstyle='-|>', lw=3, color='grey',
                mutation_scale=25, shrinkA=0, shrinkB=0)

angles = [2*np.pi*i/N for i in range(N)]   # 各点の角度 (rad)

# ------------- Figure -------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(-27.5, 27.5)
ax.set_ylim(-27.5, 27.5)
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.grid(True)

# ---------- 点(1〜8)とラベル ----------
for i, a in enumerate(angles):
    # 点そのもの
    x, y = radius_point * np.cos(a), radius_point * np.sin(a)
    ax.plot(x, y, 'o', color='red', markersize=15)

    # ── 外側ラベル（1〜8） ───────────────────────────
    ax.text(1.2 * x, 1.2 * y, str(i + 1),
            ha='center', va='center',
            fontsize=12.5, fontweight='bold')

# ── 内側ラベル（9〜14：点7→2に対応） ─────────────────
inner_map = {6: 9, 5: 10, 4: 11, 3: 12, 2: 13, 1: 14}
for idx, label in inner_map.items():
    a      = angles[idx]
    xi, yi = radius_point * np.cos(a), radius_point * np.sin(a)
    ax.text(0.8 * xi, 0.8 * yi, str(label),
            ha='center', va='center',
            fontsize=12.5, fontweight='bold', color='black')

# ---------- 曲線矢印を描く関数 ----------
def curved_arrow(ax, r, start, end, direction='ccw'):
    """direction='ccw' または 'cw'"""
    if direction == 'ccw':                             # 反時計回り
        theta1 = start + arc_trim
        theta2 = end   - arc_trim
        if theta2 < theta1:
            theta2 += 2*np.pi
        phi_base = theta2
        phi_tip  = phi_base + arrow_span
    else:                                              # 時計回り
        theta1 = end   + arc_trim                      # CCW 弧に変換
        theta2 = start - arc_trim
        if theta2 < theta1:
            theta2 += 2*np.pi
        phi_base = theta1
        phi_tip  = phi_base - arrow_span

    # 弧
    arc = Arc((0, 0), 2*r, 2*r,
              theta1=np.degrees(theta1),
              theta2=np.degrees(theta2),
              lw=3, color='grey')
    ax.add_patch(arc)

    # 矢印
    base = (r*np.cos(phi_base), r*np.sin(phi_base))
    tip  = (r*np.cos(phi_tip ), r*np.sin(phi_tip ))
    ax.annotate('', xy=tip, xytext=base, arrowprops=arrow_kw)

# ---------- 4 本の矢印 ----------
# 外側（1→2, 5→6）
curved_arrow(ax, outer_radius, angles[0], angles[1], 'ccw')
curved_arrow(ax, outer_radius, angles[4], angles[5], 'ccw')
# 内側（2→1, 6→5）
curved_arrow(ax, inner_radius, angles[1], angles[0], 'cw')
curved_arrow(ax, inner_radius, angles[5], angles[4], 'cw')

plt.tight_layout()
plt.show()
