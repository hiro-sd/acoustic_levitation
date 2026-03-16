import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# stability_log.csv を読み込み、固定音場(FIXED) vs PD制御(PD) の安定性を比較する。
# 指標: 各条件での x, y の標準偏差(σ) と 最大偏差

LOG_DIR = os.path.dirname(__file__)
LOG_PATH = os.path.join(LOG_DIR, "stability_log.csv")
EXPECTED_COLUMNS = [
    "timestamp",
    "mode",
    "u_px",
    "v_px",
    "x_mm",
    "y_mm",
    "center_x_mm",
    "center_y_mm",
]


def load_log_dataframe(log_path: str) -> pd.DataFrame:
    """Load both headered and headerless stability logs."""
    # まず通常のヘッダ付きCSVとして読む
    df = pd.read_csv(log_path)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"timestamp", "mode", "x_mm", "y_mm"}
    if required.issubset(df.columns):
        return df

    # 旧形式(ヘッダ無し)なら再読込して列名を割り当てる
    df = pd.read_csv(log_path, header=None)
    if df.shape[1] < 6:
        raise ValueError(
            f"Unsupported log format: expected >= 6 columns, got {df.shape[1]}"
        )

    if df.shape[1] <= len(EXPECTED_COLUMNS):
        df.columns = EXPECTED_COLUMNS[: df.shape[1]]
    else:
        extra = [f"extra_{i}" for i in range(df.shape[1] - len(EXPECTED_COLUMNS))]
        df.columns = EXPECTED_COLUMNS + extra

    return df


def normalize_log_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and drop invalid rows required for analysis."""
    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(str).str.strip().str.upper()

    for col in ["timestamp", "x_mm", "y_mm", "center_x_mm", "center_y_mm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = ["timestamp", "mode", "x_mm", "y_mm"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"必須列が不足しています: {missing}")

    df = df.dropna(subset=["timestamp", "x_mm", "y_mm"])
    return df


def compute_metrics(df: pd.DataFrame):
    """Compute mode-wise metrics and return processed dataframe."""
    # 原点は計測時のAUTD中心を使う（旧ログは中央値へフォールバック）
    if "center_x_mm" in df.columns and "center_y_mm" in df.columns:
        origin_x = float(df["center_x_mm"].iloc[0])
        origin_y = float(df["center_y_mm"].iloc[0])
        origin_msg = f"[INFO] 原点: AUTD中心を使用 x={origin_x:.3f}, y={origin_y:.3f} [mm]"
    else:
        origin_x = float(df["x_mm"].median())
        origin_y = float(df["y_mm"].median())
        origin_msg = "[WARN] center_x_mm / center_y_mm が無いため、中央値を原点として使用"

    out = df.copy()
    out["dx"] = out["x_mm"] - origin_x
    out["dy"] = out["y_mm"] - origin_y
    out["dist"] = np.sqrt(out["dx"] ** 2 + out["dy"] ** 2)

    results = {}
    for mode in ["FIXED", "PD"]:
        sub = out[out["mode"] == mode]
        if len(sub) == 0:
            continue
        results[mode] = {
            "n": int(len(sub)),
            "sigma_x": float(sub["dx"].std()),
            "sigma_y": float(sub["dy"].std()),
            "sigma_r": float(sub["dist"].std()),
            "max_dist": float(sub["dist"].max()),
            "mean_dist": float(sub["dist"].mean()),
            "p95_dist": float(sub["dist"].quantile(0.95)),
        }
    return out, results, origin_msg


def resolve_input_paths(inputs):
    """Resolve paths/globs into existing CSV file paths."""
    if not inputs:
        return [LOG_PATH] if os.path.exists(LOG_PATH) else []

    resolved = []
    for token in inputs:
        # ワイルドカード対応
        if any(ch in token for ch in ["*", "?", "["]):
            resolved.extend(glob.glob(token))
            continue

        # ディレクトリ指定の場合は stability_log*.csv を対象にする
        if os.path.isdir(token):
            resolved.extend(glob.glob(os.path.join(token, "stability_log*.csv")))
            continue

        # 単一ファイル指定
        if os.path.exists(token):
            resolved.append(token)

    # 重複除去しつつ順序維持
    uniq = []
    seen = set()
    for p in resolved:
        ab = os.path.abspath(p)
        if ab in seen:
            continue
        seen.add(ab)
        uniq.append(ab)
    return uniq


def print_mode_results(results):
    for mode in ["FIXED", "PD"]:
        if mode not in results:
            print(f"[WARN] mode={mode} のデータがありません。")
            continue
        m = results[mode]
        label = "固定音場" if mode == "FIXED" else "PD制御"
        print(f"=== {label} ({mode}) ===")
        print(f"  フレーム数    : {m['n']}")
        print(f"  σx [mm]      : {m['sigma_x']:.3f}")
        print(f"  σy [mm]      : {m['sigma_y']:.3f}")
        print(f"  σr [mm]      : {m['sigma_r']:.3f}  ← 総合安定性指標")
        print(f"  max dist [mm]: {m['max_dist']:.3f}")
        print(f"  mean dist[mm]: {m['mean_dist']:.3f}")
        print(f"  p95 dist [mm]: {m['p95_dist']:.3f}")
        print()


def plot_single_file(df, results, out_path):
    """Original 3-panel visualization for one file."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"FIXED": "#e07b54", "PD": "#4c9be8"}
    labels = {"FIXED": "固定音場 (FIXED)", "PD": "PD制御 (PD)"}

    # 1. XY散布図
    ax = axes[0]
    for mode, c in colors.items():
        sub = df[df["mode"] == mode]
        if len(sub) == 0:
            continue
        ax.scatter(sub["dx"], sub["dy"], s=3, alpha=0.3, color=c, label=labels[mode])
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel("dy [mm]")
    ax.set_title("位置分布 (XY散布図)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.legend()
    ax.set_aspect("equal")

    # 2. 時系列: 中心からの距離
    ax = axes[1]
    for mode, c in colors.items():
        sub = df[df["mode"] == mode].copy()
        if len(sub) == 0:
            continue
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
    plt.savefig(out_path, dpi=150)


def plot_multi_file(summary_df, out_path):
    """Visualization for multiple files comparison."""
    colors = {"FIXED": "#e07b54", "PD": "#4c9be8"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # 1) 各ファイルのσr（モード別）
    ax = axes[0]
    for mode in ["FIXED", "PD"]:
        sub = summary_df[summary_df["mode"] == mode]
        if len(sub) == 0:
            continue
        ax.scatter(sub["file_idx"], sub["sigma_r"], s=45, alpha=0.8, color=colors[mode], label=mode)
    ax.set_xlabel("file index")
    ax.set_ylabel("sigma_r [mm]")
    ax.set_title("ファイル別 sigma_r")
    ax.legend()

    # 2) 各ファイルの mean_dist（モード別）
    ax = axes[1]
    for mode in ["FIXED", "PD"]:
        sub = summary_df[summary_df["mode"] == mode]
        if len(sub) == 0:
            continue
        ax.scatter(sub["file_idx"], sub["mean_dist"], s=45, alpha=0.8, color=colors[mode], label=mode)
    ax.set_xlabel("file index")
    ax.set_ylabel("mean_dist [mm]")
    ax.set_title("ファイル別 mean_dist")
    ax.legend()

    # 3) モード別平均σr
    ax = axes[2]
    mode_means = summary_df.groupby("mode")["sigma_r"].mean()
    labels = [m for m in ["FIXED", "PD"] if m in mode_means.index]
    vals = [float(mode_means[m]) for m in labels]
    bars = ax.bar(labels, vals, color=[colors[m] for m in labels], width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom")
    ax.set_ylabel("avg sigma_r [mm]")
    ax.set_title("モード別平均 sigma_r")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

def main():
    parser = argparse.ArgumentParser(
        description="stability log(s) を解析して FIXED / PD の安定性を比較します"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="CSVファイル, ディレクトリ, またはグロブ (例: tracking/stability_log*.csv)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="プロット表示を行わず、画像保存のみ実施",
    )
    args = parser.parse_args()

    paths = resolve_input_paths(args.inputs)
    if not paths:
        print("[ERROR] 解析対象のCSVが見つかりません。")
        print("        例: python tracking/analyze_stability.py tracking/stability_log*.csv")
        sys.exit(1)

    print(f"[INFO] Input files: {len(paths)}")

    all_rows = []
    first_df = None
    first_results = None

    for idx, path in enumerate(paths, start=1):
        print("=" * 72)
        print(f"[FILE {idx}] {path}")
        try:
            df = load_log_dataframe(path)
            df = normalize_log_dataframe(df)
            df, results, origin_msg = compute_metrics(df)
        except Exception as e:
            print(f"[ERROR] 解析失敗: {e}")
            continue

        print(f"[INFO] Total frames: {len(df)}")
        print(df["mode"].value_counts())
        print(origin_msg)
        print()
        print_mode_results(results)

        if "FIXED" in results and "PD" in results:
            ratio = results["FIXED"]["sigma_r"] / results["PD"]["sigma_r"]
            print(f"[RESULT] PD制御は固定音場に比べ σr が {ratio:.2f} 倍 {'改善' if ratio > 1 else '悪化'}")
        print()

        for mode, metrics in results.items():
            row = {
                "file_idx": idx,
                "file_name": os.path.basename(path),
                "file_path": path,
                "mode": mode,
            }
            row.update(metrics)
            all_rows.append(row)

        if first_df is None:
            first_df = df
            first_results = results

    if not all_rows:
        print("[ERROR] 有効なデータを1件も解析できませんでした。")
        sys.exit(1)

    summary_df = pd.DataFrame(all_rows)
    out_csv = os.path.join(LOG_DIR, "stability_comparison_summary.csv")
    summary_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("=" * 72)
    print(f"[INFO] サマリーCSVを保存しました: {out_csv}")

    print("[INFO] 全ファイル集計（mode別平均）")
    agg = summary_df.groupby("mode")[["sigma_r", "mean_dist", "max_dist", "p95_dist"]].agg(["mean", "std", "count"])
    print(agg)

    if len(paths) == 1 and first_df is not None and first_results is not None:
        out_img = os.path.join(LOG_DIR, "stability_comparison.png")
        plot_single_file(first_df, first_results, out_img)
    else:
        out_img = os.path.join(LOG_DIR, "stability_comparison_multi.png")
        plot_multi_file(summary_df, out_img)

    print(f"[INFO] 図を保存しました: {out_img}")

    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()
