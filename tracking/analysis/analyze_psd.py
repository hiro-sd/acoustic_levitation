import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.signal import welch, detrend
except ImportError:
    raise ImportError(
        "scipy が必要です。以下でインストールしてください:\n"
        "pip install scipy"
    )


LOG_DIR = os.path.dirname(__file__)
DEFAULT_LOG_PATH = os.path.join(LOG_DIR, "stability_log.csv")


def load_log(log_path: str) -> pd.DataFrame:
    df = pd.read_csv(log_path)
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "timestamp",
        "mode",
        "x_mm",
        "y_mm",
        "z_mm",
        "center_x_mm",
        "center_y_mm",
        "center_z_mm",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["mode"] = df["mode"].astype(str).str.strip().str.upper()

    numeric_cols = [
        "timestamp",
        "x_mm",
        "y_mm",
        "z_mm",
        "center_x_mm",
        "center_y_mm",
        "center_z_mm",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "x_mm", "y_mm", "z_mm"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def drop_consecutive_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    同一フレーム由来と思われる連続重複を除去する。
    timestamp は毎回違うので比較対象に含めない。
    """
    compare_cols = [
        "mode",
        "x_mm",
        "y_mm",
        "z_mm",
        "center_x_mm",
        "center_y_mm",
        "center_z_mm",
    ]
    compare_cols = [c for c in compare_cols if c in df.columns]

    if len(compare_cols) == 0:
        return df

    same_as_prev = df[compare_cols].eq(df[compare_cols].shift(1)).all(axis=1)
    same_as_prev.iloc[0] = False

    before = len(df)
    df = df.loc[~same_as_prev].copy().reset_index(drop=True)
    after = len(df)

    print(f"[INFO] Drop consecutive duplicates: {before} -> {after}")

    return df


def add_deviation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    dx, dy, dz, r_xy を追加する。
    """
    out = df.copy()

    origin_x = float(out["center_x_mm"].dropna().iloc[0])
    origin_y = float(out["center_y_mm"].dropna().iloc[0])
    origin_z = float(out["center_z_mm"].dropna().iloc[0])

    out["dx"] = out["x_mm"] - origin_x
    out["dy"] = out["y_mm"] - origin_y
    out["dz"] = out["z_mm"] - origin_z
    out["r_xy"] = np.sqrt(out["dx"] ** 2 + out["dy"] ** 2)

    print(
        f"[INFO] Origin: "
        f"x={origin_x:.3f}, y={origin_y:.3f}, z={origin_z:.3f} mm"
    )

    return out


def extract_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    sub = df[df["mode"] == mode.upper()].copy()
    sub = sub.sort_values("timestamp").reset_index(drop=True)

    if len(sub) < 10:
        raise ValueError(f"Not enough data for mode={mode}: n={len(sub)}")

    return sub


def resample_uniform_time(
    sub: pd.DataFrame,
    value_col: str,
    fs_override: float | None = None,
):
    """
    PSD 用に不等間隔ログを等間隔に補間する。
    """
    t = sub["timestamp"].to_numpy(dtype=np.float64)
    y = sub[value_col].to_numpy(dtype=np.float64)

    # 先頭時刻を0にする
    t = t - t[0]

    # 同じ時刻がある場合を除去
    unique_mask = np.r_[True, np.diff(t) > 0]
    t = t[unique_mask]
    y = y[unique_mask]

    if len(t) < 10:
        raise ValueError(f"Not enough unique timestamp samples for {value_col}")

    dt = np.diff(t)
    dt_med = float(np.median(dt))

    if fs_override is None:
        fs = 1.0 / dt_med
    else:
        fs = float(fs_override)

    t_uniform = np.arange(t[0], t[-1], 1.0 / fs)

    if len(t_uniform) < 10:
        raise ValueError(f"Uniform samples too few for {value_col}")

    y_uniform = np.interp(t_uniform, t, y)

    # 平均値・線形ドリフトを除去
    y_uniform = detrend(y_uniform, type="linear")

    return t_uniform, y_uniform, fs


def compute_psd(y: np.ndarray, fs: float):
    """
    Welch法でPSDを計算する。
    """
    n = len(y)

    # nperseg は長すぎると平均回数が減るので、最大でも4096程度にする
    nperseg = min(4096, max(256, n // 4))

    f, pxx = welch(
        y,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False,
        scaling="density",
    )

    return f, pxx


def find_peak_frequency(f, pxx, f_min=0.1, f_max=50.0):
    """
    指定周波数範囲内でPSDピーク周波数を探す。
    """
    mask = (f >= f_min) & (f <= f_max)

    if not np.any(mask):
        return np.nan, np.nan

    f_sel = f[mask]
    p_sel = pxx[mask]

    idx = int(np.argmax(p_sel))
    return float(f_sel[idx]), float(p_sel[idx])


def plot_psd(
    f_dz,
    pxx_dz,
    f_rxy,
    pxx_rxy,
    mode: str,
    out_path: str,
    f_max_plot: float = 50.0,
):
    # Convert mode display: PD -> PID
    mode_display = "PID" if mode == "PD" else mode
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.semilogy(f_dz, pxx_dz)
    ax.set_ylabel("PSD of dz [mm^2/Hz]")
    ax.set_title(f"PSD of vertical displacement dz ({mode_display})")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.semilogy(f_rxy, pxx_rxy)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD of r_xy [mm^2/Hz]")
    ax.set_title(f"PSD of horizontal radial displacement r_xy ({mode_display})")
    ax.grid(True, which="both", alpha=0.3)

    ax.set_xlim(0, f_max_plot)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Saved PSD figure: {out_path}")


def plot_psd_comparison(
    f_fixed,
    pxx_fixed,
    f_pid,
    pxx_pid,
    value_name: str,
    out_path: str,
    f_max_plot: float = 50.0,
    ylim=None,
):
    """
    FIXED と PID を縦に並べて比較した PSD 描画
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # FIXED
    ax = axes[0]
    ax.semilogy(f_fixed, pxx_fixed)
    ax.set_ylabel(f"PSD of {value_name} [mm^2/Hz]")
    ax.set_title(f"PSD of {value_name} (FIXED)")
    ax.grid(True, which="both", alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)

    # PID
    ax = axes[1]
    ax.semilogy(f_pid, pxx_pid)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(f"PSD of {value_name} [mm^2/Hz]")
    ax.set_title(f"PSD of {value_name} (PID)")
    ax.grid(True, which="both", alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlim(0, f_max_plot)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Saved PSD comparison figure: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_path",
        nargs="?",
        default=DEFAULT_LOG_PATH,
        help="Path to stability_log.csv",
    )
    parser.add_argument(
        "--mode",
        default="FIXED",
        choices=["FIXED", "PD"],
        help="Mode to analyze",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=None,
        help="Optional resampling frequency [Hz]. If omitted, estimated from timestamps.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=50.0,
        help="Max frequency to show in plot [Hz]",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable consecutive duplicate removal",
    )
    args = parser.parse_args()

    df = load_log(args.log_path)

    if not args.no_dedup:
        df = drop_consecutive_duplicates(df)

    df = add_deviation_columns(df)

    sub = extract_mode(df, args.mode)

    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Samples: {len(sub)}")
    print(
        f"[INFO] Time span: "
        f"{sub['timestamp'].iloc[-1] - sub['timestamp'].iloc[0]:.3f} s"
    )

    # dz
    t_dz, y_dz, fs_dz = resample_uniform_time(sub, "dz", args.fs)
    f_dz, pxx_dz = compute_psd(y_dz, fs_dz)

    # r_xy
    t_rxy, y_rxy, fs_rxy = resample_uniform_time(sub, "r_xy", args.fs)
    f_rxy, pxx_rxy = compute_psd(y_rxy, fs_rxy)

    print(f"[INFO] Estimated fs for dz   : {fs_dz:.2f} Hz")
    print(f"[INFO] Estimated fs for r_xy : {fs_rxy:.2f} Hz")

    peak_f_dz, peak_p_dz = find_peak_frequency(f_dz, pxx_dz, f_min=0.1, f_max=args.fmax)
    peak_f_rxy, peak_p_rxy = find_peak_frequency(f_rxy, pxx_rxy, f_min=0.1, f_max=args.fmax)

    print()
    print("===== Dominant frequency peaks =====")
    print(f"dz   peak: {peak_f_dz:.3f} Hz, PSD={peak_p_dz:.6e}")
    print(f"r_xy peak: {peak_f_rxy:.3f} Hz, PSD={peak_p_rxy:.6e}")

    out_path = os.path.join(
        os.path.dirname(args.log_path),
        f"psd_{args.mode.lower()}_dz_rxy.png",
    )

    plot_psd(
        f_dz,
        pxx_dz,
        f_rxy,
        pxx_rxy,
        args.mode,
        out_path,
        f_max_plot=args.fmax,
    )

    plt.show()


if __name__ == "__main__":
    main()