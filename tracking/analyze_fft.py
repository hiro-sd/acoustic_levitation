import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.signal import detrend, windows
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

    if not compare_cols:
        return df

    same_as_prev = df[compare_cols].eq(df[compare_cols].shift(1)).all(axis=1)
    same_as_prev.iloc[0] = False

    before = len(df)
    df = df.loc[~same_as_prev].copy().reset_index(drop=True)
    after = len(df)

    print(f"[INFO] Drop consecutive duplicates: {before} -> {after}")

    return df


def add_deviation_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    FFT用に不等間隔ログを等間隔に補間する。
    """
    t = sub["timestamp"].to_numpy(dtype=np.float64)
    y = sub[value_col].to_numpy(dtype=np.float64)

    # 先頭時刻を0にする
    t = t - t[0]

    # 同じtimestampがある場合を除去
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

    return t_uniform, y_uniform, fs


def compute_fft_amplitude(y: np.ndarray, fs: float, use_window: bool = True):
    """
    片側FFT振幅スペクトルを計算する。
    単位はおおよそ [mm]。
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)

    # 平均・線形ドリフトを除去
    y = detrend(y, type="linear")

    if use_window:
        win = windows.hann(n)
        y_win = y * win

        # Hann窓による振幅低下を補正
        coherent_gain = np.sum(win) / n
    else:
        y_win = y
        coherent_gain = 1.0

    fft_vals = np.fft.rfft(y_win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # 片側振幅スペクトル
    amp = np.abs(fft_vals) / n / coherent_gain
    amp[1:-1] *= 2.0

    return freqs, amp


def find_peak_frequency(freqs, amp, f_min=0.1, f_max=50.0):
    mask = (freqs >= f_min) & (freqs <= f_max)

    if not np.any(mask):
        return np.nan, np.nan

    f_sel = freqs[mask]
    a_sel = amp[mask]

    idx = int(np.argmax(a_sel))
    return float(f_sel[idx]), float(a_sel[idx])


def compute_shared_amplitude_limits(
    df: pd.DataFrame,
    fs_override: float | None = None,
    use_window: bool = True,
):
    """
    FIXED / PD をまとめて見て、dz と r_xy それぞれの共通y軸上限を決める。
    """
    limits = {}

    for value_col in ["dz", "dx", "dy"]:
        amps_max = []

        for mode in ["FIXED", "PD"]:
            sub = extract_mode(df, mode)
            _, y_uniform, fs = resample_uniform_time(sub, value_col, fs_override)
            _, amp = compute_fft_amplitude(y_uniform, fs, use_window=use_window)

            if len(amp) > 0:
                amps_max.append(float(np.max(amp)))

        if amps_max:
            y_max = max(amps_max)
            if not np.isfinite(y_max) or y_max <= 0:
                y_max = 1.0
        else:
            y_max = 1.0

        limits[value_col] = (0.0, y_max * 1.05)

    return limits


def plot_fft(
    f_dz,
    amp_dz,
    f_rxy,
    amp_rxy,
    mode: str,
    out_path: str,
    f_max_plot: float = 50.0,
    log_y: bool = False,
    ylim_dz=None,
    ylim_rxy=None,
):
    # Convert mode display: PD -> PID
    mode_display = "PID" if mode == "PD" else mode
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    if log_y:
        ax.semilogy(f_dz, amp_dz)
    else:
        ax.plot(f_dz, amp_dz)
    ax.set_ylabel("Amplitude of dz [mm]")
    ax.set_title(f"FFT amplitude spectrum of dz ({mode_display})")
    ax.grid(True, which="both", alpha=0.3)
    if ylim_dz is not None:
        ax.set_ylim(ylim_dz)

    ax = axes[1]
    if log_y:
        ax.semilogy(f_rxy, amp_rxy)
    else:
        ax.plot(f_rxy, amp_rxy)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude of r_xy [mm]")
    ax.set_title(f"FFT amplitude spectrum of r_xy ({mode_display})")
    ax.grid(True, which="both", alpha=0.3)
    if ylim_rxy is not None:
        ax.set_ylim(ylim_rxy)

    ax.set_xlim(0, f_max_plot)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Saved FFT figure: {out_path}")


def plot_fft_comparison(
    f_fixed,
    amp_fixed,
    f_pid,
    amp_pid,
    value_name: str,
    out_path: str,
    f_max_plot: float = 50.0,
    log_y: bool = False,
    ylim=None,
):
    """
    FIXED と PID を縦に並べて比較した FFT 描画
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # FIXED
    ax = axes[0]
    if log_y:
        ax.semilogy(f_fixed, amp_fixed)
    else:
        ax.plot(f_fixed, amp_fixed)
    ax.set_ylabel(f"Amplitude of {value_name} [mm]")
    ax.set_title(f"FFT amplitude spectrum of {value_name} (FIXED)")
    ax.grid(True, which="both", alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)

    # PID
    ax = axes[1]
    if log_y:
        ax.semilogy(f_pid, amp_pid)
    else:
        ax.plot(f_pid, amp_pid)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(f"Amplitude of {value_name} [mm]")
    ax.set_title(f"FFT amplitude spectrum of {value_name} (PID)")
    ax.grid(True, which="both", alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlim(0, f_max_plot)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Saved FFT comparison figure: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_path",
        nargs="?",
        default=DEFAULT_LOG_PATH,
        help="Path to stability_log.csv",
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
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable Hann window",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use log scale for amplitude axis",
    )
    args = parser.parse_args()

    df = load_log(args.log_path)

    if not args.no_dedup:
        df = drop_consecutive_duplicates(df)

    df = add_deviation_columns(df)

    shared_limits = compute_shared_amplitude_limits(
        df,
        fs_override=args.fs,
        use_window=not args.no_window,
    )

    # Process both FIXED and PID
    data = {}
    for mode in ["FIXED", "PD"]:
        sub = extract_mode(df, mode)

        print(f"[INFO] Mode: {mode}")
        print(f"[INFO] Samples: {len(sub)}")
        print(
            f"[INFO] Time span: "
            f"{sub['timestamp'].iloc[-1] - sub['timestamp'].iloc[0]:.3f} s"
        )

        # dz
        t_dz, y_dz, fs_dz = resample_uniform_time(sub, "dz", args.fs)
        f_dz, amp_dz = compute_fft_amplitude(
            y_dz,
            fs_dz,
            use_window=not args.no_window,
        )

        # dx
        t_dx, y_dx, fs_dx = resample_uniform_time(sub, "dx", args.fs)
        f_dx, amp_dx = compute_fft_amplitude(
            y_dx,
            fs_dx,
            use_window=not args.no_window,
        )

        # dy
        t_dy, y_dy, fs_dy = resample_uniform_time(sub, "dy", args.fs)
        f_dy, amp_dy = compute_fft_amplitude(
            y_dy,
            fs_dy,
            use_window=not args.no_window,
        )

        print(f"[INFO] Estimated fs for dz   : {fs_dz:.2f} Hz")
        print(f"[INFO] Estimated fs for dx   : {fs_dx:.2f} Hz")
        print(f"[INFO] Estimated fs for dy   : {fs_dy:.2f} Hz")

        peak_f_dz, peak_amp_dz = find_peak_frequency(
            f_dz,
            amp_dz,
            f_min=0.1,
            f_max=args.fmax,
        )
        peak_f_dx, peak_amp_dx = find_peak_frequency(
            f_dx,
            amp_dx,
            f_min=0.1,
            f_max=args.fmax,
        )
        peak_f_dy, peak_amp_dy = find_peak_frequency(
            f_dy,
            amp_dy,
            f_min=0.1,
            f_max=args.fmax,
        )

        print()
        print("===== Dominant FFT amplitude peaks =====")
        print(f"dz   peak: {peak_f_dz:.3f} Hz, amplitude={peak_amp_dz:.6f} mm")
        print(f"dx   peak: {peak_f_dx:.3f} Hz, amplitude={peak_amp_dx:.6f} mm")
        print(f"dy   peak: {peak_f_dy:.3f} Hz, amplitude={peak_amp_dy:.6f} mm")
        print()

        data[mode] = {
            "f_dz": f_dz,
            "amp_dz": amp_dz,
            "f_dx": f_dx,
            "amp_dx": amp_dx,
            "f_dy": f_dy,
            "amp_dy": amp_dy,
        }

    # Output comparison figures for dz, dx and dy
    out_dir = os.path.dirname(args.log_path)

    # dz comparison
    out_path_dz = os.path.join(out_dir, "fft_comparison_dz.png")
    plot_fft_comparison(
        data["FIXED"]["f_dz"],
        data["FIXED"]["amp_dz"],
        data["PD"]["f_dz"],
        data["PD"]["amp_dz"],
        value_name="dz",
        out_path=out_path_dz,
        f_max_plot=args.fmax,
        log_y=args.log_y,
        ylim=shared_limits["dz"],
    )

    # dx comparison
    out_path_dx = os.path.join(out_dir, "fft_comparison_dx.png")
    plot_fft_comparison(
        data["FIXED"]["f_dx"],
        data["FIXED"]["amp_dx"],
        data["PD"]["f_dx"],
        data["PD"]["amp_dx"],
        value_name="dx",
        out_path=out_path_dx,
        f_max_plot=args.fmax,
        log_y=args.log_y,
        ylim=shared_limits["dx"],
    )

    # dy comparison
    out_path_dy = os.path.join(out_dir, "fft_comparison_dy.png")
    plot_fft_comparison(
        data["FIXED"]["f_dy"],
        data["FIXED"]["amp_dy"],
        data["PD"]["f_dy"],
        data["PD"]["amp_dy"],
        value_name="dy",
        out_path=out_path_dy,
        f_max_plot=args.fmax,
        log_y=args.log_y,
        ylim=shared_limits["dy"],
    )

    plt.show()


if __name__ == "__main__":
    main()