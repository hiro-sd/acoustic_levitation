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
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_xlabel('X [mm]', fontsize=25)
ax.set_ylabel('Y [mm]', fontsize=25)
ax.grid(True)

# 点とラベル（1-8: 内側, radius_point上に配置）
for i, a in enumerate(angles):
    x = radius_point * np.cos(a)
    y = radius_point * np.sin(a)
    ax.plot(x, y, 'o', color='red', markersize=15)
    ax.text(1.15 * x, 1.15 * y, str(i+1), ha='center', va='center', fontsize=20, fontweight='bold')

# Figure1: 8点の外側に2本の曲線矢印（点1→点2, 点5→点6の向き）
arrow_radius = radius_point + 5  # 点より外側

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

plt.tick_params(labelsize=20)
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
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_xlabel('X [mm]', fontsize=25)
ax.set_ylabel('Y [mm]', fontsize=25)
ax.grid(True)

# ---------- 点(1〜8)とラベル ----------
for i, a in enumerate(angles):
    # 点そのもの
    x, y = radius_point * np.cos(a), radius_point * np.sin(a)
    ax.plot(x, y, 'o', color='red', markersize=15)

    # ── 外側ラベル（1〜8） ───────────────────────────
    ax.text(1.15 * x, 1.15 * y, str(i + 1),
            ha='center', va='center',
            fontsize=20, fontweight='bold')

# ── 内側ラベル（9〜14：点7→2に対応） ─────────────────
inner_map = {6: 9, 5: 10, 4: 11, 3: 12, 2: 13, 1: 14}
for idx, label in inner_map.items():
    a      = angles[idx]
    xi, yi = radius_point * np.cos(a), radius_point * np.sin(a)
    ax.text(0.85 * xi, 0.85 * yi, str(label),
            ha='center', va='center',
            fontsize=20, fontweight='bold')

# --- ここから追加：点1・点8を囲う2本の曲線矢印 ---
from matplotlib.patches import FancyArrowPatch
xy = np.array([
    [radius_point * np.cos(angles[0]), radius_point * np.sin(angles[0])],
    [radius_point * np.cos(angles[7]), radius_point * np.sin(angles[7])],
])
# 点1（xy[0]）を囲う矢印（反転）
arrow1 = FancyArrowPatch(
    (21.55, 7.65), (11.65, 1.99),
    arrowstyle='<|-',
    connectionstyle="Arc3,rad=-2.0",
    linewidth=3,
    color='grey',
    mutation_scale=25,
    zorder=2
)
ax.add_patch(arrow1)
# 点8（xy[1]）を囲う矢印（反転）
arrow2 = FancyArrowPatch(
    (11.8, -21.5), (5.9, -10.5),
    arrowstyle='-|>',
    connectionstyle="Arc3,rad=2.0",
    linewidth=3,
    color='grey',
    mutation_scale=25,
    zorder=2
)
ax.add_patch(arrow2)

plt.tick_params(labelsize=20)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

# ==== パラメータ ====
r = 1.0                  # 球（円）の半径
theta_deg = 150           # 点の極角（中心角度）[deg]
theta = np.deg2rad(theta_deg)

# 円周上の点
px = r * np.cos(theta)
pz = r * np.sin(theta)

# ====== 描画開始 ======
fig, ax = plt.subplots(figsize=(6.5, 6.0))

# --- 円本体 ---
circle = Circle((0, 0), r, fill=False, linewidth=2.5, color='black')
ax.add_patch(circle)

# --- 軸（x, z）を矢印で描く ---
# x 軸（左向き矢印）
ax.add_patch(FancyArrowPatch((1.2*r, 0), (-1.3*r, 0),
                             arrowstyle='-|>',
                             mutation_scale=18, linewidth=2, color='black'))
ax.text(-1.35*r, -0.05*r, 'x', fontsize=16)

# z 軸（上向き矢印）
ax.add_patch(FancyArrowPatch((0, -1.25*r), (0, 1.3*r),
                             arrowstyle='-|>',
                             mutation_scale=18, linewidth=2, color='black'))
ax.text(0.05*r, 1.32*r, 'z', fontsize=16)

# --- 点, 中心, 線分 ---
# 点を描画
ax.scatter([px], [pz], s=100, c='black', zorder=5)

# 中心→点 の線分 (r)
ax.plot([0, px], [0, pz], color='black', linewidth=2)

# 点→x軸への垂線 (d)
ax.plot([px, px], [pz, 0], color='black', linewidth=2)

# x 軸上の投影点と中心を結ぶ線（横線）
ax.plot([0, px], [0, 0], color='black', linewidth=2)

# --- 直角マーク（小さな四角形） ---
# 直角部分を少しだけ内側にずらして描く
offset = 0.08 * r
rect = Polygon([[px, 0],
                [px, offset],
                [px - offset, offset],
                [px - offset, 0]],
               closed=True, fill=False, linewidth=2, color='black')
ax.add_patch(rect)

# --- 角度 θ の弧を描く ---
angle_arc = np.linspace(0, theta, 100)
arc_radius = 0.25 * r
ax.plot(arc_radius * np.cos(angle_arc),
        arc_radius * np.sin(angle_arc),
        color='black', linewidth=2)
ax.text(arc_radius * 1.1 * np.cos(theta/2),
        arc_radius * 1.1 * np.sin(theta/2),
        r'$\theta$', fontsize=16)

# --- ラベル ---
ax.text((px+0)/2 + 0.02*r, (pz+0)/2, 'r', fontsize=16)                  # 斜辺 r
ax.text(px + 0.02*r, (pz)/2, 'd', fontsize=16)                          # 垂線 d
# right angle “⌟” の下に小さくラベル入れるならここで
# ax.text(px - 0.09*r, 0.02*r, 'h', fontsize=14)

# 軸範囲・スタイル調整
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-1.5*r, 1.5*r)
ax.set_ylim(-1.5*r, 1.5*r)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.tight_layout()
plt.savefig('figure_theta_d_r.png', dpi=300)
plt.show()

