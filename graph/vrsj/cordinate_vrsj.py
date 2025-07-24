import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# 3D空間に矢印を描画するためのヘルパークラス
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs3d)


def draw_3d_coordinate_system():
    """
    指定された定義に基づいて三次元座標系の説明図を描画します。
    - 極角θ: X軸とのなす角
    - 方位角φ: YZ平面への射影とY軸とのなす角
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- 1. 点とベクトルの定義 ---
    p = np.array([3, 4, 5])
    origin = np.array([0, 0, 0])
    x_axis_vec = np.array([p[0], 0, 0])

    # --- 2. 座標軸の描画 (エラー箇所を修正) ---
    arrow_prop = dict(mutation_scale=20, arrowstyle='-|>', color='black', lw=3)
    
    # ax.quiverの代わりにArrow3Dクラスを使用
    ax.add_artist(Arrow3D([0, 7], [0, 0], [0, 0], **arrow_prop)) # X軸
    ax.add_artist(Arrow3D([0, 0], [0, 7], [0, 0], **arrow_prop)) # Y軸
    ax.add_artist(Arrow3D([0, 0], [0, 0], [0, 7], **arrow_prop)) # Z軸

    ax.text(7.5, 0, 0, 'X', fontsize=25)
    ax.text(0, 7.5, 0, 'Y', fontsize=25)
    ax.text(0, 0, 7.5, 'Z', fontsize=25)


    # --- 3. 点とベクトルの描画 ---
    ax.scatter(p[0], p[1], p[2], color='red', s=100, label='P(x, y, z)')
    ax.text(p[0], p[1], p[2] + 0.5, 'P', fontsize=25, color='red')
    ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color='red', lw=3)
    
    # --- 4. 射影と補助線の描画 ---
    p_yz_projection = np.array([0, p[1], p[2]])
    ax.plot([p[0], 0], [p[1], p[1]], [p[2], p[2]], 'k--')
    ax.plot([0, p_yz_projection[0]], [0, p_yz_projection[1]], [0, p_yz_projection[2]], color='blue', lw=3)
    ax.scatter(0, p[1], p[2], color='blue', s=50)

    # --- 5. 角度を示す円弧の描画 ---
    # 極角θ
    vec1 = p - origin
    vec2 = x_axis_vec - origin
    # p[0]が0の場合のゼロ除算を避ける
    if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        # θの円弧を描画
        arc_radius_theta = 2.5
        # 3D空間内の円弧を正しく描画するための回転計算
        norm_p_yz = np.sqrt(p[1]**2 + p[2]**2)
        if norm_p_yz > 0:
            # YZ平面内での単位ベクトル
            p_yz_unit = p_yz_projection / norm_p_yz
            # YZ平面とXY平面の間の回転軸（Y軸に平行）
            rot_axis = np.array([0, -p[2], p[1]])
            # 回転角度
            rot_angle = np.arccos(p[1]/norm_p_yz) * np.sign(p[2])
            
            # 回転行列の作成は複雑なので、より単純なアプローチで描画
            t = np.linspace(0, theta_rad, 100)
            x_arc = arc_radius_theta * np.cos(t)
            # ベクトルpとX軸を含む平面上に円弧を配置
            norm_v = np.linalg.norm(p - x_axis_vec)
            if norm_v > 0:
                v = (p-x_axis_vec)/norm_v
                y_arc = arc_radius_theta*np.sin(t)*v[1]
                z_arc = arc_radius_theta*np.sin(t)*v[2]
                ax.plot(x_arc, y_arc, z_arc, color='green', lw=3)

                # ラベル位置の計算
                label_pos_vec = arc_radius_theta*np.cos(theta_rad/2)*x_axis_vec/p[0] + arc_radius_theta*np.sin(theta_rad/2)*v
                ax.text(label_pos_vec[0] * 1.2, label_pos_vec[1] * 1.2, label_pos_vec[2] * 1.2,
                        'θ', fontsize=25, color='green')

    # 方位角φ
    phi_rad = np.arctan2(p[2], p[1])
    arc_radius_phi = 2
    t_phi = np.linspace(0, phi_rad, 50)
    x_phi = np.zeros_like(t_phi)
    y_phi = arc_radius_phi * np.cos(t_phi)
    z_phi = arc_radius_phi * np.sin(t_phi)
    ax.plot(x_phi, y_phi, z_phi, color='purple', lw=3)
    ax.text(0.3,
            arc_radius_phi * np.cos(phi_rad/2) * 1.2,
            arc_radius_phi * np.sin(phi_rad/2) * 1.2,
            'φ', fontsize=25, color='purple')

    # --- 6. グラフの見た目を調整 ---
    ax.set_xlabel('X axis', fontsize=25)
    ax.set_ylabel('Y axis', fontsize=25)
    ax.set_zlabel('Z axis', fontsize=25)
    # 軸の数字（目盛りラベル）を消去
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.set_title('Custom Spherical Coordinate System', fontsize=16)
    max_val = np.max(p)
    ax.set_xlim(-1, max_val + 3)
    ax.set_ylim(-1, max_val + 3)
    ax.set_zlim(-1, max_val + 3)
    ax.set_box_aspect([1,1,1])

    ax.view_init(elev=20, azim=30)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    draw_3d_coordinate_system()

