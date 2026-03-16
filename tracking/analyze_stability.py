import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

# stability_log.csv を読み込み、固定音場(FIXED) vs PD制御(PD) の安定性を比較する。
# 指標: 各条件での x, y の標準偏差(σ) と 最大偏差

LOG_PATH = os.path.join(os.path.dirname(__file__), "stability_log.csv")

def main():
    # データ読み込み
    try:
        df = pd.read_csv(LOG_PATH)
    except FileNotFoundError:
        print(f"[ERROR] {LOG_PATH} が見つかりません。先に tracking_fast_feedback.py を実行してください。")
        sys.exit(1)

    print(f"[INFO] Total frames: {len(df)}")
    print(df["mode"].value_counts())
    print()

    # 原点は計測時のAUTD中心を使う（旧ログは中央値へフォールバック）
    if "center_x_mm" in df.columns and "center_y_mm" in df.columns:
        origin_x = float(df["center_x_mm"].iloc[0])
        origin_y = float(df["center_y_mm"].iloc[0])
        print(f"[INFO] 原点: AUTD中心を使用 x={origin_x:.3f}, y={origin_y:.3f} [mm]")
    else:
        origin_x = float(df["x_mm"].median())
        origin_y = float(df["y_mm"].median())
        print("[WARN] center_x_mm / center_y_mm が無いため、中央値を原点として使用")

    df["dx"] = df["x_mm"] - origin_x
    df["dy"] = df["y_mm"] - origin_y
    df["dist"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

    results = {}
    for mode in ["FIXED", "PD"]:
        sub = df[df["mode"] == mode]
        if len(sub) == 0:
            print(f"[WARN] mode={mode} のデータがありません。")
            continue
        sx = sub["dx"].std()
        sy = sub["dy"].std()
        sr = sub["dist"].std()
        max_r = sub["dist"].max()
        mean_r = sub["dist"].mean()
        results[mode] = {"n": len(sub), "sigma_x": sx, "sigma_y": sy,
                         "sigma_r": sr, "max_dist": max_r, "mean_dist": mean_r}
        label = "固定音場" if mode == "FIXED" else "PD制御"
        print(f"=== {label} ({mode}) ===")
        print(f"  フレーム数    : {len(sub)}")
        print(f"  σx [mm]      : {sx:.3f}")
        print(f"  σy [mm]      : {sy:.3f}")
        print(f"  σr [mm]      : {sr:.3f}  ← 総合安定性指標")
        print(f"  max dist [mm]: {max_r:.3f}")
        print(f"  mean dist[mm]: {mean_r:.3f}")
        print()

    if len(results) == 2:
        ratio = results["FIXED"]["sigma_r"] / results["PD"]["sigma_r"]
        print(f"[RESULT] PD制御は固定音場に比べ σr が {ratio:.2f} 倍 {'改善' if ratio > 1 else '悪化'}")

    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"FIXED": "#e07b54", "PD": "#4c9be8"}
    labels = {"FIXED": "固定音場 (FIXED)", "PD": "PD制御 (PD)"}

    # 1. XY散布図
    ax = axes[0]
    for mode, c in colors.items():
        sub = df[df["mode"] == mode]
        if len(sub) == 0: continue
        ax.scatter(sub["dx"], sub["dy"], s=3, alpha=0.3, color=c, label=labels[mode])
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel("dy [mm]")
    ax.set_title("位置分布 (XY散布図)")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.legend()
    ax.set_aspect("equal")

    # 2. 時系列: 中心からの距離
    ax = axes[1]
    for mode, c in colors.items():
        sub = df[df["mode"] == mode].copy()
        if len(sub) == 0: continue
        t0 = sub["timestamp"].iloc[0]
        ax.plot(sub["timestamp"] - t0, sub["dist"], lw=0.6, alpha=0.7, color=c, label=labels[mode])
    ax.set_xlabel("経過時間 [s]")
    ax.set_ylabel("中心からの距離 [mm]")
    ax.set_title("時系列: 中心偏差")
    ax.legend()

    # 3. 棒グラフ: σr 比較
    ax = axes[2]
    modes_avail = [m for m in ["FIXED", "PD"] if m in results]
    sigma_vals = [results[m]["sigma_r"] for m in modes_avail]
    bar_colors = [colors[m] for m in modes_avail]
    bar_labels = [labels[m] for m in modes_avail]
    bars = ax.bar(bar_labels, sigma_vals, color=bar_colors, width=0.4)
    for bar, val in zip(bars, sigma_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", va="bottom")
    ax.set_ylabel("σr [mm] (小さいほど安定)")
    ax.set_title("安定性比較 (中心偏差の標準偏差)")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "stability_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] 図を保存しました: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
