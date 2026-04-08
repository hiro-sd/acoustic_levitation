import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# stability_log.csv を読み込み、
# FIXED vs PD の安定性を x, y, z の各方向および総合距離で比較する

LOG_DIR = os.path.dirname(__file__)
LOG_PATH = os.path.join(LOG_DIR, "stability_log.csv")

# 新形式ログ
EXPECTED_COLUMNS_NEW = [
    "timestamp",
    "mode",
    "u_xy_px",
    "v_xy_px",
    "v_z_px",
    "x_mm",
    "y_mm",
    "z_mm",
    "center_x_mm",
    "center_y_mm",
    "center_z_mm",
]

# 旧形式ログ
EXPECTED_COLUMNS_OLD = [
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
    """Load both current and legacy stability logs."""
    # まず通常のヘッダ付きCSVとして読む
    df = pd.read_csv(log_path)
    df.columns = [str(c).strip() for c in df.columns]

    required_new = {"timestamp", "mode", "x_mm", "y_mm", "z_mm"}
    required_old = {"timestamp", "mode", "x_mm", "y_mm"}

    if required_new.issubset(df.columns):
        return df
    if required_old.issubset(df.columns):
        return df

    # ヘッダ無しログへのフォールバック
    df = pd.read_csv(log_path, header=None)

    if df.shape[1] == len(EXPECTED_COLUMNS_NEW):
        df.columns = EXPECTED_COLUMNS_NEW
        return df

    if df.shape[1] == len(EXPECTED_COLUMNS_OLD):
        df.columns = EXPECTED_COLUMNS_OLD
        return df

    if df.shape[1] < 6:
        raise ValueError(
            f"Unsupported log format: expected >= 6 columns, got {df.shape[1]}"
        )

    # どちらにも合わない場合は old 形式ベースで拡張
    if df.shape[1] <= len(EXPECTED_COLUMNS_OLD):
        df.columns = EXPECTED_COLUMNS_OLD[: df.shape[1]]
    else:
        extra = [f"extra_{i}" for i in range(df.shape[1] - len(EXPECTED_COLUMNS_OLD))]
        df.columns = EXPECTED_COLUMNS_OLD + extra

    return df


def normalize_log_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and drop invalid rows required for analysis."""
    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(str).str.strip().str.upper()

    numeric_cols = [
        "timestamp",
        "u_xy_px",
        "v_xy_px",
        "v_z_px",
        "u_px",
        "v_px",
        "x_mm",
        "y_mm",
        "z_mm",
        "center_x_mm",
        "center_y_mm",
        "center_z_mm",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = ["timestamp", "mode", "x_mm", "y_mm"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"必須列が不足しています: {missing}")

    # z列が無い旧ログにも対応
    if "z_mm" not in df.columns:
        df["z_mm"] = np.nan

    if "center_z_mm" not in df.columns:
        df["center_z_mm"] = np.nan

    df = df.dropna(subset=["timestamp", "x_mm", "y_mm"])
    return df


def compute_metrics(df: pd.DataFrame):
    """Compute mode-wise metrics and return processed dataframe."""
    # 原点は中心座標を優先使用。無い列は中央値へフォールバック。
    if "center_x_mm" in df.columns and df["center_x_mm"].notna().any():
        origin_x = float(df["center_x_mm"].dropna().iloc[0])
    else:
        origin_x = float(df["x_mm"].median())

    if "center_y_mm" in df.columns and df["center_y_mm"].notna().any():
        origin_y = float(df["center_y_mm"].dropna().iloc[0])
    else:
        origin_y = float(df["y_mm"].median())

    if "center_z_mm" in df.columns and df["center_z_mm"].notna().any():
        origin_z = float(df["center_z_mm"].dropna().iloc[0])
        origin_z_msg = f"z={origin_z:.3f}"
    elif df["z_mm"].notna().any():
        origin_z = float(df["z_mm"].median())
        origin_z_msg = f"z median fallback={origin_z:.3f}"
    else:
        origin_z = np.nan
        origin_z_msg = "z unavailable"

    origin_msg = (
        f"[INFO] 原点: x={origin_x:.3f}, y={origin_y:.3f}, {origin_z_msg} [mm]"
    )

    out = df.copy()
    out["dx"] = out["x_mm"] - origin_x
    out["dy"] = out["y_mm"] - origin_y
    out["r_xy"] = np.sqrt(out["dx"] ** 2 + out["dy"] ** 2)

    if out["z_mm"].notna().any() and np.isfinite(origin_z):
        out["dz"] = out["z_mm"] - origin_z
        out["r_xyz"] = np.sqrt(out["dx"] ** 2 + out["dy"] ** 2 + out["dz"] ** 2)
    else:
        out["dz"] = np.nan
        out["r_xyz"] = np.nan

    results = {}
    for mode in ["FIXED", "PD"]:
        sub = out[out["mode"] == mode]
        if len(sub) == 0:
            continue

        z_available = sub["dz"].notna().any()

        result = {
            "n": int(len(sub)),
            "sigma_x": float(sub["dx"].std()),
            "sigma_y": float(sub["dy"].std()),
            "sigma_xy": float(sub["r_xy"].std()),
            "max_r_xy": float(sub["r_xy"].max()),
            "mean_r_xy": float(sub["r_xy"].mean()),
            "p95_r_xy": float(sub["r_xy"].quantile(0.95)),
        }

        if z_available:
            dz_valid = sub["dz"].dropna()
            rxyz_valid = sub["r_xyz"].dropna()
            result.update(
                {
                    "sigma_z": float(dz_valid.std()),
                    "max_abs_z": float(dz_valid.abs().max()),
                    "mean_abs_z": float(dz_valid.abs().mean()),
                    "sigma_xyz": float(rxyz_valid.std()),
                    "max_r_xyz": float(rxyz_valid.max()),
                    "mean_r_xyz": float(rxyz_valid.mean()),
                    "p95_r_xyz": float(rxyz_valid.quantile(0.95)),
                }
            )
        else:
            result.update(
                {
                    "sigma_z": np.nan,
                    "max_abs_z": np.nan,
                    "mean_abs_z": np.nan,
                    "sigma_xyz": np.nan,
                    "max_r_xyz": np.nan,
                    "mean_r_xyz": np.nan,
                    "p95_r_xyz": np.nan,
                }
            )

        results[mode] = result

    return out, results, origin_msg


def resolve_input_paths(inputs):
    """Resolve paths/globs into existing CSV file paths."""
    if not inputs:
        return [LOG_PATH] if os.path.exists(LOG_PATH) else []

    resolved = []
    for token in inputs:
        if any(ch in token for ch in ["*", "?", "["]):
            resolved.extend(glob.glob(token))
            continue

        if os.path.isdir(token):
            resolved.extend(glob.glob(os.path.join(token, "stability_log*.csv")))
            continue

        if os.path.exists(token):
            resolved.append(token)

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
        label = "固定音場" if mode == "FIXED" else "制御あり"

        print(f"=== {label} ({mode}) ===")
        print(f"  フレーム数         : {m['n']}")
        print(f"  sigma_x [mm]       : {m['sigma_x']:.3f}")
        print(f"  sigma_y [mm]       : {m['sigma_y']:.3f}")
        print(f"  sigma_xy [mm]      : {m['sigma_xy']:.3f}  ← XY総合安定性")
        print(f"  max_r_xy [mm]      : {m['max_r_xy']:.3f}")
        print(f"  mean_r_xy [mm]     : {m['mean_r_xy']:.3f}")
        print(f"  p95_r_xy [mm]      : {m['p95_r_xy']:.3f}")

        if np.isfinite(m["sigma_z"]):
            print(f"  sigma_z [mm]       : {m['sigma_z']:.3f}  ← Z安定性")
            print(f"  max_abs_z [mm]     : {m['max_abs_z']:.3f}")
            print(f"  mean_abs_z [mm]    : {m['mean_abs_z']:.3f}")
            print(f"  sigma_xyz [mm]     : {m['sigma_xyz']:.3f}  ← XYZ総合安定性")
            print(f"  max_r_xyz [mm]     : {m['max_r_xyz']:.3f}")
            print(f"  mean_r_xyz [mm]    : {m['mean_r_xyz']:.3f}")
            print(f"  p95_r_xyz [mm]     : {m['p95_r_xyz']:.3f}")
        else:
            print("  z方向データ        : 利用不可")

        print()


def print_improvement(results):
    if "FIXED" not in results or "PD" not in results:
        return

    fixed = results["FIXED"]
    pdm = results["PD"]

    def ratio_msg(key, label):
        a = fixed.get(key, np.nan)
        b = pdm.get(key, np.nan)
        if not (np.isfinite(a) and np.isfinite(b) and b > 0):
            return
        ratio = a / b
        print(
            f"[RESULT] {label}: FIXED / PD = {ratio:.2f} "
            f"({'PD改善' if ratio > 1 else 'PD悪化'})"
        )

    ratio_msg("sigma_x", "sigma_x")
    ratio_msg("sigma_y", "sigma_y")
    ratio_msg("sigma_xy", "sigma_xy")
    ratio_msg("sigma_z", "sigma_z")
    ratio_msg("sigma_xyz", "sigma_xyz")
    print()


def plot_single_file(df, results, out_path):
    """Visualization for one file with x,y,z stability."""
    colors = {"FIXED": "#e07b54", "PD": "#4c9be8"}
    labels = {"FIXED": "固定音場 (FIXED)", "PD": "制御あり (PD)"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. XY散布図
    ax = axes[0, 0]
    for mode, c in colors.items():
        sub = df[df["mode"] == mode]
        if len(sub) == 0:
            continue
        ax.scatter(sub["dx"], sub["dy"], s=5, alpha=0.35, color=c, label=labels[mode])
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel("dy [mm]")
    ax.set_title("位置分布 (XY散布図)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.legend()
    ax.set_aspect("equal")

    # 2. z時系列
    ax = axes[0, 1]
    z_plotted = False
    for mode, c in colors.items():
        sub = df[(df["mode"] == mode) & (df["dz"].notna())].copy()
        if len(sub) == 0:
            continue
        t0 = sub["timestamp"].iloc[0]
        ax.plot(sub["timestamp"] - t0, sub["dz"], lw=0.7, alpha=0.8, color=c, label=labels[mode])
        z_plotted = True
    ax.set_xlabel("経過時間 [s]")
    ax.set_ylabel("dz [mm]")
    ax.set_title("時系列: z偏差")
    if z_plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "zデータなし", ha="center", va="center", transform=ax.transAxes)

    # 3. XY距離時系列
    ax = axes[1, 0]
    for mode, c in colors.items():
        sub = df[df["mode"] == mode].copy()
        if len(sub) == 0:
            continue
        t0 = sub["timestamp"].iloc[0]
        ax.plot(sub["timestamp"] - t0, sub["r_xy"], lw=0.7, alpha=0.8, color=c, label=labels[mode])
    ax.set_xlabel("経過時間 [s]")
    ax.set_ylabel("r_xy [mm]")
    ax.set_title("時系列: XY中心偏差")
    ax.legend()

    # 4. sigma比較棒グラフ
    ax = axes[1, 1]
    metrics = ["sigma_x", "sigma_y", "sigma_z"]
    metric_labels = ["σx", "σy", "σz"]
    x = np.arange(len(metrics))
    width = 0.35

    fixed_vals = [results["FIXED"].get(m, np.nan) if "FIXED" in results else np.nan for m in metrics]
    pd_vals = [results["PD"].get(m, np.nan) if "PD" in results else np.nan for m in metrics]

    ax.bar(x - width / 2, fixed_vals, width, label="FIXED", color=colors["FIXED"])
    ax.bar(x + width / 2, pd_vals, width, label="PD", color=colors["PD"])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("標準偏差 [mm]")
    ax.set_title("軸別安定性比較")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def plot_multi_file(summary_df, out_path):
    """Visualization for multiple files comparison."""
    colors = {"FIXED": "#e07b54", "PD": "#4c9be8"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # 1) 各ファイルの sigma_xy
    ax = axes[0]
    for mode in ["FIXED", "PD"]:
        sub = summary_df[summary_df["mode"] == mode]
        if len(sub) == 0:
            continue
        ax.scatter(sub["file_idx"], sub["sigma_xy"], s=45, alpha=0.8, color=colors[mode], label=mode)
    ax.set_xlabel("file index")
    ax.set_ylabel("sigma_xy [mm]")
    ax.set_title("ファイル別 sigma_xy")
    ax.legend()

    # 2) 各ファイルの sigma_z
    ax = axes[1]
    for mode in ["FIXED", "PD"]:
        sub = summary_df[(summary_df["mode"] == mode) & (summary_df["sigma_z"].notna())]
        if len(sub) == 0:
            continue
        ax.scatter(sub["file_idx"], sub["sigma_z"], s=45, alpha=0.8, color=colors[mode], label=mode)
    ax.set_xlabel("file index")
    ax.set_ylabel("sigma_z [mm]")
    ax.set_title("ファイル別 sigma_z")
    ax.legend()

    # 3) モード別平均 sigma_xyz
    ax = axes[2]
    mode_means = summary_df.groupby("mode")["sigma_xyz"].mean()
    labels = [m for m in ["FIXED", "PD"] if m in mode_means.index and np.isfinite(mode_means[m])]
    vals = [float(mode_means[m]) for m in labels]
    if labels:
        bars = ax.bar(labels, vals, color=[colors[m] for m in labels], width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom")
    ax.set_ylabel("avg sigma_xyz [mm]")
    ax.set_title("モード別平均 sigma_xyz")

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
        print_improvement(results)

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

    agg_cols = ["sigma_x", "sigma_y", "sigma_xy", "sigma_z", "sigma_xyz", "mean_r_xy", "mean_r_xyz"]
    agg_cols = [c for c in agg_cols if c in summary_df.columns]
    agg = summary_df.groupby("mode")[agg_cols].agg(["mean", "std", "count"])
    print("[INFO] 全ファイル集計（mode別平均）")
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