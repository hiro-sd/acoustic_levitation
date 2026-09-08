import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
    }
)

# stability_log.csv を読み込み、
# FIXED vs PID の安定性を x, y, z の各方向および総合距離で比較する

LOG_DIR = os.path.dirname(__file__)
LOG_PATH = os.path.join(LOG_DIR, "stability_log.csv")
PLOT_OUT_DIR = os.path.join(LOG_DIR, "analyze_stability")

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
    "autd_target_x_mm",
    "autd_target_y_mm",
    "autd_target_z_mm",
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


def drop_consecutive_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    連続する重複行だけを除去する。
    timestamp は毎回違うので比較対象から外す。
    mode が変わる行は別セッションの可能性があるため残す。
    """
    compare_cols = [
        "mode",
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
        "autd_target_x_mm",
        "autd_target_y_mm",
        "autd_target_z_mm",
    ]
    compare_cols = [c for c in compare_cols if c in df.columns]

    if not compare_cols:
        return df

    prev = df[compare_cols].shift(1)
    same_as_prev = df[compare_cols].eq(prev).all(axis=1)

    # 先頭行は必ず残す
    same_as_prev.iloc[0] = False

    return df.loc[~same_as_prev].copy()


def normalize_log_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and drop invalid rows required for analysis."""
    if "session_mode" in df.columns:
        # 現行ログでは session_mode が FIXED/PID/PID_NO_DELAY、
        # mode が NORMAL_HOLD 等の制御状態を表す。
        df["control_mode"] = df["mode"] if "mode" in df.columns else ""
        df["mode"] = df["session_mode"]

    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(str).str.strip().str.upper()
        df["mode"] = df["mode"].replace({"PD": "PID"})

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
        "autd_target_x_mm",
        "autd_target_y_mm",
        "autd_target_z_mm",
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

    # 同一フレーム由来と思われる連続重複行を除去
    df = drop_consecutive_duplicate_rows(df)

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
    for mode in ["FIXED", "PID_NO_DELAY", "PID"]:
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
    mode_labels = {
        "FIXED": "固定音場",
        "PID_NO_DELAY": "制御あり・遅延補正なし",
        "PID": "制御あり・遅延補正あり",
    }

    for mode in ["FIXED", "PID_NO_DELAY", "PID"]:
        if mode not in results:
            print(f"[WARN] mode={mode} のデータがありません。")
            continue

        m = results[mode]
        label = mode_labels.get(mode, mode)

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
    labels = {
        "FIXED": "FIXED",
        "PID_NO_DELAY": "PID_NO_DELAY",
        "PID": "PID",
    }

    def ratio_msg(base_mode, compare_mode, key, label):
        a = results[base_mode].get(key, np.nan)
        b = results[compare_mode].get(key, np.nan)
        if not (np.isfinite(a) and np.isfinite(b) and b > 0):
            return
        ratio = a / b
        print(
            f"[RESULT] {label}: {labels[base_mode]} / {labels[compare_mode]} = {ratio:.2f} "
            f"({'改善' if ratio > 1 else '悪化'})"
        )

    pairs = []
    if "FIXED" in results and "PID" in results:
        pairs.append(("FIXED", "PID", "制御あり・遅延補正ありの効果"))
    if "FIXED" in results and "PID_NO_DELAY" in results:
        pairs.append(("FIXED", "PID_NO_DELAY", "制御あり・遅延補正なしの効果"))
    if "PID_NO_DELAY" in results and "PID" in results:
        pairs.append(("PID_NO_DELAY", "PID", "遅延補正の効果"))

    for base_mode, compare_mode, title in pairs:
        print(f"--- {title} ---")
        ratio_msg(base_mode, compare_mode, "sigma_x", "sigma_x")
        ratio_msg(base_mode, compare_mode, "sigma_y", "sigma_y")
        ratio_msg(base_mode, compare_mode, "sigma_xy", "sigma_xy")
        ratio_msg(base_mode, compare_mode, "sigma_z", "sigma_z")
        ratio_msg(base_mode, compare_mode, "sigma_xyz", "sigma_xyz")
    if pairs:
        print()


def plot_single_file(df, results, out_dir):
    """Save requested single-file charts as separate PNG images."""
    colors = {"FIXED": "#e07b54", "PID": "#4c9be8"}
    labels = {"FIXED": "Fixed", "PID": "Controlled"}
    label_fs = 20
    tick_fs = 20
    legend_fs = 18
    out_paths = []

    # 1) Position Scatter
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    for mode, c in colors.items():
        sub = df[df["mode"] == mode]
        if len(sub) == 0:
            continue
        ax.scatter(sub["dx"], sub["dy"], s=6, alpha=0.35, color=c, label=labels[mode])
    
    # Make plot square by setting equal axis limits
    dx_range = df["dx"].max() - df["dx"].min()
    dy_range = df["dy"].max() - df["dy"].min()
    shared_range = max(dx_range, dy_range) * 0.55
    center_x = (df["dx"].max() + df["dx"].min()) / 2
    center_y = (df["dy"].max() + df["dy"].min()) / 2
    ax.set_xlim(center_x - shared_range, center_x + shared_range)
    ax.set_ylim(center_y - shared_range, center_y + shared_range)
    
    ax.set_xlabel(r"$d_x$ [mm]", fontsize=label_fs)
    ax.set_ylabel(r"$d_y$ [mm]", fontsize=label_fs)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.legend(fontsize=16)
    ax.set_aspect("equal")
    plt.tight_layout()
    out_scatter = os.path.join(out_dir, "stability_position_scatter.png")
    plt.savefig(out_scatter, dpi=150)
    plt.close(fig)
    out_paths.append(out_scatter)

    # 1b) Position Scatter (dx, dz)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    plotted_dz = False
    for mode, c in colors.items():
        sub = df[(df["mode"] == mode) & (df["dz"].notna())]
        if len(sub) == 0:
            continue
        plotted_dz = True
        ax.scatter(sub["dx"], sub["dz"], s=6, alpha=0.35, color=c, label=labels[mode])
    ax.set_xlabel(r"$d_x$ [mm]", fontsize=label_fs)
    ax.set_ylabel(r"$d_z$ [mm]", fontsize=label_fs)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    if plotted_dz:
        ax.legend(fontsize=legend_fs)
    else:
        ax.text(0.5, 0.5, "zデータなし", ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout()
    out_scatter_dx_dz = os.path.join(out_dir, "stability_position_scatter_dx_dz.png")
    plt.savefig(out_scatter_dx_dz, dpi=150)
    plt.close(fig)
    out_paths.append(out_scatter_dx_dz)

    # 1c) Position Scatter (dy, dz)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    plotted_dz = False
    for mode, c in colors.items():
        sub = df[(df["mode"] == mode) & (df["dz"].notna())]
        if len(sub) == 0:
            continue
        plotted_dz = True
        ax.scatter(sub["dy"], sub["dz"], s=6, alpha=0.35, color=c, label=labels[mode])
    ax.set_xlabel(r"$d_y$ [mm]", fontsize=label_fs)
    ax.set_ylabel(r"$d_z$ [mm]", fontsize=label_fs)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    if plotted_dz:
        ax.legend(fontsize=legend_fs)
    else:
        ax.text(0.5, 0.5, "zデータなし", ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout()
    out_scatter_dy_dz = os.path.join(out_dir, "stability_position_scatter_dy_dz.png")
    plt.savefig(out_scatter_dy_dz, dpi=150)
    plt.close(fig)
    out_paths.append(out_scatter_dy_dz)

    # 2) Time Series: dx
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    work = df.copy()
    work["segment_id"] = (work["mode"] != work["mode"].shift(1)).cumsum()
    t0_all = float(work["timestamp"].iloc[0])

    # dx, dy, dz の3図で共通の縦軸レンジを使う
    axis_cols = ["dx", "dy", "dz"]
    stacked_vals = []
    for col in axis_cols:
        if col in work.columns:
            vals = pd.to_numeric(work[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if len(vals) > 0:
                stacked_vals.append(vals)

    if len(stacked_vals) > 0:
        shared_y_max = float(np.max(np.abs(np.concatenate(stacked_vals))))
        if shared_y_max < 1e-6:
            shared_y_max = 1.0
    else:
        shared_y_max = 1.0
    shared_y_lim = (-shared_y_max, shared_y_max)

    def save_axis_timeseries(col, ylabel, out_name):
        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        plotted_label = {"FIXED": False, "PID": False}
        plotted_any = False

        for _, sub_seg in work.groupby("segment_id", sort=True):
            mode = str(sub_seg["mode"].iloc[0]).upper()
            if mode not in colors:
                continue

            sub_seg_col = sub_seg[["timestamp", col]].copy()
            sub_seg_col[col] = pd.to_numeric(sub_seg_col[col], errors="coerce")
            sub_seg_col = sub_seg_col.dropna(subset=[col])
            if len(sub_seg_col) == 0:
                continue

            label = labels[mode] if not plotted_label[mode] else None
            ax.plot(
                sub_seg_col["timestamp"] - t0_all,
                sub_seg_col[col],
                lw=0.8,
                alpha=0.85,
                color=colors[mode],
                label=label,
            )
            plotted_label[mode] = True
            plotted_any = True

        ax.set_xlabel("Elapsed time [s]", fontsize=label_fs)
        ax.set_ylabel(ylabel, fontsize=label_fs)
        ax.axhline(0, color="k", lw=0.8, alpha=0.8)
        ax.set_ylim(shared_y_lim)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)
        if plotted_any:
            ax.legend(fontsize=legend_fs)
        else:
            ax.text(0.5, 0.5, "データなし", ha="center", va="center", transform=ax.transAxes)

        plt.tight_layout()
        out_path = os.path.join(out_dir, out_name)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        out_paths.append(out_path)

    save_axis_timeseries("dx", r"$d_x$ [mm]", "stability_time_series_dx.png")
    save_axis_timeseries("dy", r"$d_y$ [mm]", "stability_time_series_dy.png")
    save_axis_timeseries("dz", r"$d_z$ [mm]", "stability_time_series_dz.png")

    # 2b) Combined Time Series: dx, dy, dz in one figure
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10.0), sharex=True)
    axis_specs = [
        ("dx", r"$d_x$ [mm]", "データなし"),
        ("dy", r"$d_y$ [mm]", "データなし"),
        ("dz", r"$d_z$ [mm]", "zデータなし"),
    ]

    plotted_label = {"FIXED": False, "PID": False}

    for i, (col, ylabel, empty_msg) in enumerate(axis_specs):
        ax = axes[i]
        plotted_any = False

        for _, sub_seg in work.groupby("segment_id", sort=True):
            mode = str(sub_seg["mode"].iloc[0]).upper()
            if mode not in colors:
                continue

            sub_seg_col = sub_seg[["timestamp", col]].copy()
            sub_seg_col[col] = pd.to_numeric(sub_seg_col[col], errors="coerce")
            sub_seg_col = sub_seg_col.dropna(subset=[col])
            if len(sub_seg_col) == 0:
                continue

            label = labels[mode] if not plotted_label[mode] else None
            ax.plot(
                sub_seg_col["timestamp"] - t0_all,
                sub_seg_col[col],
                lw=0.8,
                alpha=0.85,
                color=colors[mode],
                label=label,
            )
            plotted_label[mode] = True
            plotted_any = True

        ax.set_ylabel(ylabel, fontsize=label_fs)
        ax.axhline(0, color="k", lw=0.8, alpha=0.8)
        ax.set_ylim(shared_y_lim)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)
        if not plotted_any:
            ax.text(0.5, 0.5, empty_msg, ha="center", va="center", transform=ax.transAxes)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, legend_labels, fontsize=legend_fs, loc="upper right")
    axes[-1].set_xlabel("Elapsed time [s]", fontsize=label_fs)

    plt.tight_layout()
    out_combined = os.path.join(out_dir, "stability_time_series.png")
    plt.savefig(out_combined, dpi=150)
    plt.close(fig)
    out_paths.append(out_combined)

    # 3) Axis-wise Stability Comparison
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    metrics = ["sigma_x", "sigma_y", "sigma_z", "sigma_xyz"]
    metric_labels = [
        r"$\mathit{std}_x$",
        r"$\mathit{std}_y$",
        r"$\mathit{std}_z$",
        r"$\mathit{std}_{xyz}$",
    ]
    x = np.arange(len(metrics))
    bar_modes = [m for m in ["FIXED", "PID_NO_DELAY", "PID"] if m in results]
    bar_colors = {
        "FIXED": colors["FIXED"],
        "PID_NO_DELAY": "#72b36a",
        "PID": colors["PID"],
    }
    bar_labels = {
        "FIXED": "Fixed",
        "PID_NO_DELAY": "Controlled (no compensation)",
        "PID": "Controlled",
    }
    width = min(0.8 / max(1, len(bar_modes)), 0.25)
    offsets = (np.arange(len(bar_modes)) - (len(bar_modes) - 1) / 2.0) * width
    for offset, mode in zip(offsets, bar_modes):
        vals = [results[mode].get(m, np.nan) for m in metrics]
        ax.bar(x + offset, vals, width, label=bar_labels[mode], color=bar_colors[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=tick_fs)
    ax.set_ylabel("Standard deviation [mm]", fontsize=label_fs)
    ax.tick_params(axis="y", which="major", labelsize=tick_fs)
    ax.legend(fontsize=legend_fs)
    plt.tight_layout()
    out_axis = os.path.join(out_dir, "stability_axis_wise_stability_comparison.png")
    plt.savefig(out_axis, dpi=150)
    plt.close(fig)
    out_paths.append(out_axis)

    return out_paths


def plot_multi_file(summary_df, out_path):
    """Visualization for multiple files comparison."""
    colors = {"FIXED": "#e07b54", "PID": "#4c9be8"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # 1) 各ファイルの sigma_xy
    ax = axes[0]
    for mode in ["FIXED", "PID"]:
        sub = summary_df[summary_df["mode"] == mode]
        if len(sub) == 0:
            continue
        ax.scatter(sub["file_idx"], sub["sigma_xy"], s=45, alpha=0.8, color=colors[mode], label=mode)
    ax.set_xlabel("file index")
    ax.set_ylabel("sigma_xy [mm]")
    ax.set_title("File-wise sigma_xy")
    ax.legend()

    # 2) 各ファイルの sigma_z
    ax = axes[1]
    for mode in ["FIXED", "PID"]:
        sub = summary_df[(summary_df["mode"] == mode) & (summary_df["sigma_z"].notna())]
        if len(sub) == 0:
            continue
        ax.scatter(sub["file_idx"], sub["sigma_z"], s=45, alpha=0.8, color=colors[mode], label=mode)
    ax.set_xlabel("file index")
    ax.set_ylabel("sigma_z [mm]")
    ax.set_title("File-wise sigma_z")
    ax.legend()

    # 3) モード別平均 sigma_xyz
    ax = axes[2]
    mode_means = summary_df.groupby("mode")["sigma_xyz"].mean()
    labels = [m for m in ["FIXED", "PID"] if m in mode_means.index and np.isfinite(mode_means[m])]
    vals = [float(mode_means[m]) for m in labels]
    if labels:
        bars = ax.bar(labels, vals, color=[colors[m] for m in labels], width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom")
    ax.set_ylabel("avg sigma_xyz [mm]")
    ax.set_title("Mode-wise Average sigma_xyz")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def plot_target_timeseries_single(df, out_path):
    """Plot measured object position and AUTD target center trajectories for single file."""
    target_cols = ["autd_target_x_mm", "autd_target_y_mm", "autd_target_z_mm"]
    if not all(c in df.columns for c in target_cols):
        return False

    if not df[target_cols].notna().any().any():
        return False

    work = df.copy()
    t0 = float(work["timestamp"].iloc[0])
    t = work["timestamp"] - t0

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # X
    ax = axes[0]
    if "x_mm" in work.columns:
        ax.plot(t, work["x_mm"], lw=0.8, alpha=0.75, color="#4c9be8", label="object x_mm")
    ax.plot(t, work["autd_target_x_mm"], lw=1.0, alpha=0.9, color="#e07b54", label="target x_mm")
    ax.set_ylabel("x [mm]")
    ax.set_title("Time Series: Object Position and AUTD Target Center")
    ax.legend(loc="upper right")

    # Y
    ax = axes[1]
    if "y_mm" in work.columns:
        ax.plot(t, work["y_mm"], lw=0.8, alpha=0.75, color="#4c9be8", label="object y_mm")
    ax.plot(t, work["autd_target_y_mm"], lw=1.0, alpha=0.9, color="#e07b54", label="target y_mm")
    ax.set_ylabel("y [mm]")
    ax.legend(loc="upper right")

    # Z
    ax = axes[2]
    if "z_mm" in work.columns:
        ax.plot(t, work["z_mm"], lw=0.8, alpha=0.75, color="#4c9be8", label="object z_mm")
    ax.plot(t, work["autd_target_z_mm"], lw=1.0, alpha=0.9, color="#e07b54", label="target z_mm")
    ax.set_ylabel("z [mm]")
    ax.set_xlabel("経過時間 [s]")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="stability log(s) を解析して FIXED / PID の安定性を比較します"
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

    os.makedirs(PLOT_OUT_DIR, exist_ok=True)

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

    # 各サンプルの統計を表示
    print()
    print("=" * 72)
    print("[INFO] 各サンプルの統計")
    print("=" * 72)
    cols_display = ["file_idx", "file_name", "mode", "n", "sigma_x", "sigma_y", "sigma_z", "sigma_xy", "sigma_xyz"]
    cols_display = [c for c in cols_display if c in summary_df.columns]
    summary_labels = {
        "FIXED": "固定音場 (FIXED)",
        "PID_NO_DELAY": "制御あり・遅延補正なし (PID_NO_DELAY)",
        "PID": "制御あり・遅延補正あり (PID)",
    }
    for mode in ["FIXED", "PID_NO_DELAY", "PID"]:
        sub = summary_df[summary_df["mode"] == mode][cols_display]
        if len(sub) > 0:
            label = summary_labels.get(mode, mode)
            print(f"\n--- {label} ---")
            print(sub.to_string(index=False))
    print()
    print("=" * 72)
    print("[INFO] 全ファイル集計（mode別統計）")
    print("=" * 72)
    agg_cols = ["sigma_x", "sigma_y", "sigma_xy", "sigma_z", "sigma_xyz", "mean_r_xy", "mean_r_xyz"]
    agg_cols = [c for c in agg_cols if c in summary_df.columns]
    agg = summary_df.groupby("mode")[agg_cols].agg(["mean", "std", "count"])
    print(agg)

    if len(paths) == 1 and first_df is not None and first_results is not None:
        out_imgs = plot_single_file(first_df, first_results, PLOT_OUT_DIR)

        out_target_img = os.path.join(PLOT_OUT_DIR, "stability_target_timeseries.png")
        if plot_target_timeseries_single(first_df, out_target_img):
            print(f"[INFO] AUTD目標中心の時系列図を保存しました: {out_target_img}")
        else:
            print("[INFO] AUTD目標中心列が無いため、時系列図の追加出力はスキップしました。")

        for out_img in out_imgs:
            print(f"[INFO] 図を保存しました: {out_img}")
    else:
        out_img = os.path.join(PLOT_OUT_DIR, "stability_comparison_multi.png")
        plot_multi_file(summary_df, out_img)
        print(f"[INFO] 図を保存しました: {out_img}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
