import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロットに必要
import seaborn as sns # 提案1の実装に必要

def get_marker_coords(row, color):
    """データフレームの行から指定された色のマーカー座標を取得"""
    if color == 'red':
        return row['red_x'], row['red_y']
    elif color == 'blue':
        return row['blue_x'], row['blue_y']
    else:
        return -1, -1

def compute_3d_on_sphere(cx, cy, r_pix, mx, my, r_mm):
    """2D画像座標を3D物理座標に変換"""
    dx = mx - cx
    dy = my - cy
    d_pix = np.sqrt(dx**2 + dy**2)
    if r_pix == 0 or d_pix > r_pix:
        return None
    
    # 球面座標系への変換
    theta = np.arcsin(d_pix / r_pix)
    phi = np.arctan2(dy, dx)
    
    # カメラ座標系の3D座標
    x_img = r_mm * np.sin(theta) * np.cos(phi)
    y_img = r_mm * np.sin(theta) * np.sin(phi)
    z_img = r_mm * np.cos(theta)
    
    # 物理座標系への変換
    x_phys = z_img
    y_phys = x_img
    z_phys = -y_img
    return np.array([x_phys, y_phys, z_phys])

def estimate_rotation_axis(csv_path, fps, r_mm):
    """CSVから回転軸を推定し、複数の方法で可視化する"""
    df = pd.read_csv(csv_path)
    colors = ['red', 'blue']
    color_map = {'red': 'red', 'blue': 'blue'}
    
    t_dict = {c: [] for c in colors}
    axis_dict = {c: [] for c in colors}
    
    # データ処理と回転軸の計算
    for color in colors:
        mask = (df[f'{color}_x'] != -1) & (df[f'{color}_y'] != -1)
        indices = np.where(mask)[0]
        if len(indices) < 2:
            continue
        
        splits = np.split(indices, np.where(np.diff(indices) != 1)[0]+1)
        
        for seg in splits:
            if len(seg) < 2:
                continue
            t = seg / fps
            coords3d = []
            for i in seg:
                row = df.iloc[i]
                cx, cy, r_pix = row['cx'], row['cy'], row['radius']
                mx, my = get_marker_coords(row, color)
                p = compute_3d_on_sphere(cx, cy, r_pix, mx, my, r_mm)
                coords3d.append(p if p is not None else np.nan)
            
            coords3d = np.array(coords3d)
            if np.any(np.isnan(coords3d)):
                continue
                
            n_vecs = np.cross(coords3d[:-1], coords3d[1:])
            n_norm = np.linalg.norm(n_vecs, axis=1, keepdims=True)
            n_vecs_normalized = np.divide(n_vecs, n_norm, out=np.zeros_like(n_vecs), where=n_norm!=0)
            
            t_mid = (t[1:] + t[:-1]) / 2
            t_dict[color].append(t_mid)
            axis_dict[color].append(n_vecs_normalized)

    # --- 可視化セクション ---

    # 時系列プロット（x/y/z成分ごとに色分け）
    plt.figure(figsize=(10, 6))
    # 色設定
    time_colors = {
        'red':   ['#d62728', '#ff7f0e', '#e377c2'],  # x=赤, y=オレンジ, z=ピンク
        'blue':  ['#1f77b4', '#17becf', '#9467bd']   # x=青, y=水色, z=紫
    }
    for color in colors:
        if t_dict[color]:
            t_all_color = np.concatenate(t_dict[color])
            axis_all_color = np.concatenate(axis_dict[color])
            plt.plot(t_all_color, axis_all_color[:,0], '.', markersize=5, color=time_colors[color][0], label=f'{color} x')
            plt.plot(t_all_color, axis_all_color[:,1], '.', markersize=5, color=time_colors[color][1], label=f'{color} y')
            plt.plot(t_all_color, axis_all_color[:,2], '.', markersize=5, color=time_colors[color][2], label=f'{color} z')
    plt.xlabel('Time [s]', fontsize=20)
    plt.ylabel('Rotation axis component', fontsize=20)
    plt.title('Time evolution of rotation axis (normalized)', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-1, 1)  # y軸を-1～1に固定
    plt.tight_layout()
    plt.show()

    # 2. 3D分布プロット（red/blue markerで色分け）
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    marker_colors = {'red': 'red', 'blue': 'blue'}
    for color in colors:
        if axis_dict[color]:
            axis_all_color = np.concatenate(axis_dict[color])
            ax.scatter(axis_all_color[:,0], axis_all_color[:,1], axis_all_color[:,2], s=10, color=marker_colors[color], label=f'{color} marker')
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_zlabel('Z', fontsize=20)
    ax.set_title('3D distribution of rotation axis (normalized)', fontsize=20)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=20)
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    xs = np.cos(u)*np.sin(v)
    ys = np.sin(u)*np.sin(v)
    zs = np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.2)
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()

    # --- 新しいプロットのためのデータ準備 ---
    all_axes_list = [np.concatenate(axis_dict[c]) for c in colors if axis_dict[c]]
    
    if not all_axes_list:
        print("No valid data to plot.")
        return

    combined_axis_data = np.concatenate(all_axes_list)

    # 各成分のヒストグラム
    x_coords, y_coords, z_coords = combined_axis_data[:, 0], combined_axis_data[:, 1], combined_axis_data[:, 2]
    # --- 全データの最大密度値を外部から受け取る ---
    global GLOBAL_HIST_YLIM
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('Distribution of Rotation Axis Components', fontsize=25)
    bins = 50
    xlim = (-1, 1)
    ylim = GLOBAL_HIST_YLIM if 'GLOBAL_HIST_YLIM' in globals() else None
    # X Component
    axes[0].hist(x_coords, bins=bins, color='red', alpha=0.7, density=True, range=xlim)
    axes[0].set_title('X Component', fontsize=25)
    axes[0].set_xlabel('Value', fontsize=25)
    axes[0].set_ylabel('Probability Density', fontsize=25)
    axes[0].set_xlim(xlim)
    if ylim: axes[0].set_ylim(ylim)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].tick_params(labelsize=25)
    # Y Component
    axes[1].hist(y_coords, bins=bins, color='green', alpha=0.7, density=True, range=xlim)
    axes[1].set_title('Y Component', fontsize=25)
    axes[1].set_xlabel('Value', fontsize=25)
    axes[1].set_xlim(xlim)
    if ylim: axes[1].set_ylim(ylim)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].tick_params(labelsize=20)
    # Z Component
    axes[2].hist(z_coords, bins=bins, color='blue', alpha=0.7, density=True, range=xlim)
    axes[2].set_title('Z Component', fontsize=25)
    axes[2].set_xlabel('Value', fontsize=25)
    axes[2].set_xlim(xlim)
    if ylim: axes[2].set_ylim(ylim)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].tick_params(labelsize=20)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='markers.csvから回転軸の時間変化を推定・可視化')
    parser.add_argument('--csv', type=str, default='/Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation/rotation_measurement/syukai/markers_syukai.csv, /Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation/rotation_measurement/ouhuku/markers_ouhuku.csv', help='markers.csvファイルパス')
    parser.add_argument('--fps', type=float, default=500.0, help='フレームレート[Hz]')
    parser.add_argument('--radius_mm', type=float, default=18.0, help='球の半径[mm]')
    args = parser.parse_args()
    
    # --- 2ファイル分のヒストグラムy軸最大値を事前計算 ---
    # ファイルパスを2つ指定する場合はここで両方処理
    csv_paths = [args.csv]
    if ',' in args.csv:
        csv_paths = [p.strip() for p in args.csv.split(',')]
    all_hist_vals = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        colors = ['red', 'blue']
        axis_dict = {c: [] for c in colors}
        for color in colors:
            mask = (df[f'{color}_x'] != -1) & (df[f'{color}_y'] != -1)
            indices = np.where(mask)[0]
            if len(indices) < 2:
                continue
            splits = np.split(indices, np.where(np.diff(indices) != 1)[0]+1)
            for seg in splits:
                if len(seg) < 2:
                    continue
                coords3d = []
                for i in seg:
                    row = df.iloc[i]
                    cx, cy, r_pix = row['cx'], row['cy'], row['radius']
                    mx, my = get_marker_coords(row, color)
                    p = compute_3d_on_sphere(cx, cy, r_pix, mx, my, args.radius_mm)
                    coords3d.append(p if p is not None else np.nan)
                coords3d = np.array(coords3d)
                if np.any(np.isnan(coords3d)):
                    continue
                n_vecs = np.cross(coords3d[:-1], coords3d[1:])
                n_norm = np.linalg.norm(n_vecs, axis=1, keepdims=True)
                n_vecs_normalized = np.divide(n_vecs, n_norm, out=np.zeros_like(n_vecs), where=n_norm!=0)
                axis_dict[color].append(n_vecs_normalized)
        all_axes_list = [np.concatenate(axis_dict[c]) for c in colors if axis_dict[c]]
        if not all_axes_list:
            continue
        combined_axis_data = np.concatenate(all_axes_list)
        x_coords, y_coords, z_coords = combined_axis_data[:, 0], combined_axis_data[:, 1], combined_axis_data[:, 2]
        bins = 50
        xlim = (-1, 1)
        all_hist = [np.histogram(arr, bins=bins, range=xlim, density=True)[0] for arr in [x_coords, y_coords, z_coords]]
        all_hist_vals.extend([h.max() for h in all_hist])
    global GLOBAL_HIST_YLIM
    GLOBAL_HIST_YLIM = (0, max(all_hist_vals) * 1.05) if all_hist_vals else None
    # --- 通常通り1ファイル処理 ---
    for csv_path in csv_paths:
        estimate_rotation_axis(csv_path, args.fps, args.radius_mm)

if __name__ == '__main__':
    main()