import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_marker_coords(row, color):
    if color == 'red':
        return row['red_x'], row['red_y']
    elif color == 'blue':
        return row['blue_x'], row['blue_y']
    else:
        return -1, -1

def compute_3d_on_sphere(cx, cy, r_pix, mx, my, r_mm):
    dx = mx - cx
    dy = my - cy
    d_pix = np.sqrt(dx**2 + dy**2)
    if r_pix == 0 or d_pix > r_pix:
        return None
    theta = np.arcsin(d_pix / r_pix)
    phi = np.arctan2(dy, dx)
    x = r_mm * np.sin(theta) * np.cos(phi)
    y = r_mm * np.sin(theta) * np.sin(phi)
    z = r_mm * np.cos(theta)
    return np.array([x, y, z])

def process_file(csv_path, fps, r_mm):
    df = pd.read_csv(csv_path)
    colors = ['red', 'blue']
    color_map = {'red': 'red', 'blue': 'blue'}
    time_dict = {c: [] for c in colors}
    omega_dict = {c: [] for c in colors}
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
                if p is None:
                    coords3d.append(np.nan)
                else:
                    coords3d.append(p)
            coords3d = np.array(coords3d)
            if np.isnan(coords3d).any():
                continue
            dot = np.sum(coords3d[1:] * coords3d[:-1], axis=1) / (r_mm**2)
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)
            omega = theta * fps
            t_mid = (t[1:] + t[:-1]) / 2
            time_dict[color].append(t_mid)
            omega_dict[color].append(omega)
    all_time = []
    all_omega = []
    for color in colors:
        if not time_dict[color]:
            continue
        time_all = np.concatenate(time_dict[color])
        omega_all = np.concatenate(omega_dict[color])
        all_time.append(time_all)
        all_omega.append(omega_all)
    if all_omega:
        omega_concat = np.concatenate(all_omega)
        mean_omega = np.mean(omega_concat)
        mean_rpm = mean_omega * 60 / (2 * np.pi)
        textstr = f"Mean ω = {mean_omega:.2f} rad/s\nMean RPM = {mean_rpm:.2f} rpm"
    else:
        textstr = "No data"
    return time_dict, omega_dict, textstr

def main():
    parser = argparse.ArgumentParser(description='2つのmarkers.csvから球面上の角速度(rad/s)を同時プロット')
    parser.add_argument('--csv1', type=str, default='/Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation/rotation_measurement/syukai/markers_syukai.csv', help='1つ目のmarkers.csvファイルパス')
    parser.add_argument('--csv2', type=str, default='/Users/yoshidahiroto/Downloads/修士関連/acoustic_levitation/rotation_measurement/ouhuku/markers_ouhuku.csv', help='2つ目のmarkers.csvファイルパス')
    parser.add_argument('--fps', type=float, default=500.0, help='フレームレート[Hz]')
    parser.add_argument('--radius_mm', type=float, default=18.0, help='球の半径[mm]')
    args = parser.parse_args()

    # 2ファイル分処理
    results = []
    for csv_path in [args.csv1, args.csv2]:
        time_dict, omega_dict, textstr = process_file(csv_path, args.fps, args.radius_mm)
        results.append((time_dict, omega_dict, textstr))

    # y軸範囲を揃える
    all_omega_vals = []
    for time_dict, omega_dict, _ in results:
        for color in ['red', 'blue']:
            if omega_dict[color]:
                all_omega_vals.append(np.concatenate(omega_dict[color]))
    if all_omega_vals:
        y_min = min([arr.min() for arr in all_omega_vals])
        y_max = max([arr.max() for arr in all_omega_vals])
    else:
        y_min, y_max = 0, 1

    color_map = {'red': 'red', 'blue': 'blue'}
    for i, (time_dict, omega_dict, textstr) in enumerate(results):
        plt.figure(figsize=(18, 6))
        for color in ['red', 'blue']:
            if not time_dict[color]:
                continue
            time_all = np.concatenate(time_dict[color])
            omega_all = np.concatenate(omega_dict[color])
            plt.plot(time_all, omega_all, '.', color=color_map[color], label=f'{color} marker', markersize=10)
        plt.ylim(y_min, y_max)
        plt.ylabel('Angular velocity [rad/s]', fontsize=25)
        plt.title('angular velocity transition', fontsize=25)
        plt.tick_params(labelsize=20)
        plt.legend(fontsize=25)
        plt.gca().text(0.98, 0.98, textstr.strip(), fontsize=25, color='black',
                      ha='right', va='top', transform=plt.gca().transAxes,
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        plt.xlabel('Time [s]', fontsize=25)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()

