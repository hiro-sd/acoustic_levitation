from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


CONDITION_COLUMNS = [
    "offset_x_mm",
    "offset_y_mm",
    "offset_z_mm",
    "intensity_ratio",
]


def _ci_half_width(values: pd.Series, confidence: float = 0.95) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    count = len(values)
    if count < 2:
        return float("nan")
    standard_error = float(values.std(ddof=1) / math.sqrt(count))
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, count - 1))
    return critical * standard_error


def summarize_force(trials: pd.DataFrame) -> pd.DataFrame:
    valid = trials.loc[
        (trials["status"] == "valid")
        & trials["acoustic_force_z_mN"].notna()
    ].copy()
    rows = []
    for keys, group in valid.groupby(CONDITION_COLUMNS, dropna=False, sort=True):
        force = group["acoustic_force_z_mN"]
        rows.append(
            {
                **dict(zip(CONDITION_COLUMNS, keys)),
                "valid_trials": len(group),
                "mean_force_z_mN": float(force.mean()),
                "std_force_z_mN": float(force.std(ddof=1)) if len(group) > 1 else np.nan,
                "sem_force_z_mN": float(force.sem(ddof=1)) if len(group) > 1 else np.nan,
                "ci95_half_width_mN": _ci_half_width(force),
                "median_single_trial_fit_se_mN": float(
                    group["acoustic_force_z_se_mN"].median()
                ),
                "median_fit_rms_z_mm": float(group["fit_rms_z_mm"].median()),
                "median_fit_r2_z": float(group["fit_r2_z"].median()),
                "median_baseline_std_z_mm": float(group["baseline_std_z_mm"].median()),
                "median_single_difference_95_z_mm": float(
                    group["single_difference_95_z_mm"].median()
                ),
                "median_approximate_z_scale_mm_per_px": float(
                    group["approximate_scale_z_mm_per_px"].median()
                ),
            }
        )
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        columns=[
            *CONDITION_COLUMNS,
            "valid_trials",
            "mean_force_z_mN",
            "std_force_z_mN",
            "sem_force_z_mN",
            "ci95_half_width_mN",
            "median_single_trial_fit_se_mN",
            "median_fit_rms_z_mm",
            "median_fit_r2_z",
            "median_baseline_std_z_mm",
            "median_single_difference_95_z_mm",
            "median_approximate_z_scale_mm_per_px",
        ]
    )


def adjacent_intensity_differences(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    offset_columns = ["offset_x_mm", "offset_y_mm", "offset_z_mm"]
    if summary.empty:
        return pd.DataFrame(
            columns=[
                *offset_columns,
                "lower_intensity",
                "upper_intensity",
                "mean_force_difference_mN",
                "approximate_ci95_half_width_mN",
                "difference_resolved_95",
                "force_increases",
            ]
        )
    for offsets, group in summary.groupby(offset_columns, dropna=False, sort=True):
        ordered = group.sort_values("intensity_ratio")
        records = list(ordered.to_dict("records"))
        for lower, upper in zip(records, records[1:]):
            difference = upper["mean_force_z_mN"] - lower["mean_force_z_mN"]
            lower_ci = lower["ci95_half_width_mN"]
            upper_ci = upper["ci95_half_width_mN"]
            combined_ci = (
                math.sqrt(lower_ci**2 + upper_ci**2)
                if np.isfinite(lower_ci) and np.isfinite(upper_ci)
                else np.nan
            )
            rows.append(
                {
                    **dict(zip(offset_columns, offsets)),
                    "lower_intensity": lower["intensity_ratio"],
                    "upper_intensity": upper["intensity_ratio"],
                    "mean_force_difference_mN": difference,
                    "approximate_ci95_half_width_mN": combined_ci,
                    "difference_resolved_95": bool(
                        np.isfinite(combined_ci) and abs(difference) > combined_ci
                    ),
                    "force_increases": bool(difference > 0.0),
                }
            )
    return pd.DataFrame(rows)


def _plot_force(summary: pd.DataFrame, output_path: Path):
    if summary.empty:
        return
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    grouping = ["offset_x_mm", "offset_y_mm", "offset_z_mm"]
    for offsets, group in summary.groupby(grouping, dropna=False, sort=True):
        group = group.sort_values("intensity_ratio")
        label = f"offset=({offsets[0]:g}, {offsets[1]:g}, {offsets[2]:g}) mm"
        axis.errorbar(
            group["intensity_ratio"],
            group["mean_force_z_mN"],
            yerr=group["ci95_half_width_mN"],
            marker="o",
            capsize=4,
            label=label,
        )
    axis.set_xlabel("Commanded intensity ratio")
    axis.set_ylabel("Upward acoustic force [mN]")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def analyze(session_directory: Path, output_directory: Path):
    summary_path = session_directory / "trial_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"trial summary not found: {summary_path}")
    trials = pd.read_csv(summary_path)
    numeric_columns = [
        *CONDITION_COLUMNS,
        "acoustic_force_z_mN",
        "acoustic_force_z_se_mN",
        "fit_rms_z_mm",
        "fit_r2_z",
        "baseline_std_z_mm",
        "single_difference_95_z_mm",
        "approximate_scale_z_mm_per_px",
    ]
    for column in numeric_columns:
        trials[column] = pd.to_numeric(trials[column], errors="coerce")
    trials["status"] = trials["status"].fillna("").astype(str)

    output_directory.mkdir(parents=True, exist_ok=True)
    force_summary = summarize_force(trials)
    differences = adjacent_intensity_differences(force_summary)
    force_summary.to_csv(output_directory / "force_by_condition.csv", index=False)
    differences.to_csv(output_directory / "adjacent_intensity_differences.csv", index=False)
    _plot_force(force_summary, output_directory / "force_vs_intensity.png")

    valid_count = int((trials["status"] == "valid").sum())
    print("============================================================")
    print(" ACOUSTIC FORCE STEP-RESPONSE ANALYSIS")
    print(f" session: {session_directory}")
    print(f" valid trials: {valid_count}/{len(trials)}")
    if force_summary.empty:
        print(" No valid force estimates are available.")
    else:
        print("\nCondition summary (95% CI across repeated trials):")
        print(
            force_summary[
                [
                    "intensity_ratio",
                    "offset_z_mm",
                    "valid_trials",
                    "mean_force_z_mN",
                    "ci95_half_width_mN",
                    "median_single_trial_fit_se_mN",
                    "median_single_difference_95_z_mm",
                ]
            ].to_string(index=False)
        )
        if not differences.empty:
            resolved = int(differences["difference_resolved_95"].sum())
            increasing = int(differences["force_increases"].sum())
            print(
                f"\nAdjacent amplitude differences resolved at approximate 95%: "
                f"{resolved}/{len(differences)}"
            )
            print(
                f"Adjacent pairs with increasing mean force: "
                f"{increasing}/{len(differences)}"
            )
    print(f"\nOutputs: {output_directory}")
    print("============================================================")
    return force_summary, differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    session_directory = Path(args.session_dir)
    output_directory = (
        Path(args.output_dir)
        if args.output_dir
        else session_directory / "analysis"
    )
    analyze(session_directory, output_directory)


if __name__ == "__main__":
    main()
