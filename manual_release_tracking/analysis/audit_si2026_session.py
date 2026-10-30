"""Audit one SI2026 session and summarize exploratory pre-outcome factors.

Run after analyze_si2026_trials.py, using the same raw data directory and the
corresponding output directory. This never changes the eligibility rules.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from analyze_si2026_trials import _fit_logistic_regression, wilson_interval


FACTORS = (
    ("release_dx_mm", 10.0, "Release X offset", "mm"),
    ("release_dy_mm", 10.0, "Release Y offset", "mm"),
    ("release_dz_mm", 20.0, "Release Z offset", "mm"),
    ("release_to_field_ms", 10.0, "Release-to-field delay", "ms"),
    ("capture_state_abs_vz_mm_s", 100.0, "Onset |Vz|", "mm/s"),
    ("prediction_error_3d_mm", 5.0, "Initial prediction error", "mm"),
)


def audit(raw_dir: Path, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(raw_dir / "trial_summary.csv")
    samples = pd.read_csv(raw_dir / "trial_timeseries.csv")
    exclusions = pd.read_csv(raw_dir / "trial_exclusions.csv")
    delay = pd.read_csv(raw_dir / "auto_release_delay_measurements.csv")
    telemetry = pd.read_csv(raw_dir / "autd_update_telemetry.csv")
    trajectory = pd.read_csv(raw_dir / "auto_release_trajectory_log.csv")
    events = pd.read_csv(raw_dir / "auto_release_capture_events.csv")
    enriched = pd.read_csv(results_dir / "trial_metrics_enriched.csv")
    with next(raw_dir.glob("session_*.json")).open(encoding="utf-8") as stream:
        session = json.load(stream)

    checks = []

    def check(name: str, observed: object, expected: object, passed: bool):
        checks.append(
            {"check": name, "passed": bool(passed), "observed": observed,
             "expected": expected}
        )

    check("raw_vs_enriched_trial_ids", len(enriched), len(raw),
          raw.trial_id.equals(enriched.trial_id))
    check("unique_trial_ids", raw.trial_id.nunique(), len(raw),
          raw.trial_id.is_unique)
    check("contiguous_trial_index", raw.trial_index.nunique(), len(raw),
          sorted(raw.trial_index) == list(range(1, len(raw) + 1)))
    check("single_session", raw.session_id.nunique(), 1,
          raw.session_id.nunique() == 1
          and raw.session_id.iloc[0] == session["session_id"])
    check("condition_matches_session", raw.condition_id.nunique(), 1,
          raw.condition_id.nunique() == 1
          and raw.condition_id.iloc[0] == session["condition"]["condition_id"])
    check("status_matches_success_flag",
          int((raw.status.eq("success") != raw.success.astype(bool)).sum()),
          0, bool((raw.status.eq("success") == raw.success.astype(bool)).all()))
    field_rows = raw[raw.field_time_sec.notna()]
    timing_error_ms = (
        (field_rows.field_time_sec - field_rows.release_timestamp_ms / 1000.0)
        * 1000.0 - field_rows.release_to_field_ms
    ).abs()
    check("release_to_field_matches_timestamps_ms",
          float(timing_error_ms.max()), "<=0.01",
          bool(timing_error_ms.le(0.01).all()))
    check("manual_exclusion_ids_exist",
          len(set(exclusions.trial_id) - set(raw.trial_id)), 0,
          set(exclusions.trial_id).issubset(set(raw.trial_id)))
    check("timeseries_ids_known",
          len(set(samples.trial_id) - set(raw.trial_id)), 0,
          set(samples.trial_id).issubset(set(raw.trial_id)))
    sample_counts = samples.groupby("trial_id").size().reindex(raw.trial_id).fillna(0)
    mismatch = int(np.count_nonzero(
        raw.sample_count.fillna(0).to_numpy() != sample_counts.to_numpy()
    ))
    check("summary_sample_counts_match_timeseries", mismatch, 0, mismatch == 0)
    eligible = enriched[enriched.analysis_eligible.astype(bool)].copy()
    missing_series = int((~eligible.trial_id.isin(samples.trial_id)).sum())
    check("all_eligible_trials_have_timeseries", missing_series, 0,
          missing_series == 0)
    eligible_release_ids = set(eligible.release_timestamp_ms)
    for name, log in (("delay", delay), ("telemetry", telemetry),
                      ("trajectory", trajectory), ("capture_events", events)):
        missing = len(eligible_release_ids - set(log.release_timestamp_ms))
        check(f"all_eligible_trials_have_{name}_log", missing, 0,
              missing == 0)
    event_successes = events[events.event.eq("success_criterion_reached")]
    check("success_event_count_matches_summary", len(event_successes),
          int(eligible.status.eq("success").sum()),
          len(event_successes) == int(eligible.status.eq("success").sum()))
    missing_covariates = int(eligible[
        ["capture_state_vxy_mm_s", "release_xy_offset_mm", "release_to_field_ms"]
    ].isna().any(axis=1).sum())
    check("all_eligible_trials_have_primary_covariates", missing_covariates,
          0, missing_covariates == 0)
    target = int(session["config"]["si2026_target_valid_trials"])
    check("eligible_target_reached", len(eligible), target,
          len(eligible) == target)
    successes = eligible.status.eq("success")
    reached = eligible.success_reached_time_sec.notna()
    check("success_reached_timestamp_matches_status",
          int((successes != reached).sum()), 0, bool((successes == reached).all()))
    hold = float(session["config"]["si2026_success_hold_sec"])
    check("hold_requirement_matches_session",
          eligible.success_hold_requirement_sec.nunique(), 1,
          eligible.success_hold_requirement_sec.eq(hold).all())
    return pd.DataFrame(checks), eligible


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 1.0
    for rank_index in range(len(order) - 1, -1, -1):
        index = order[rank_index]
        running = min(running, p_values[index] * len(order) / (rank_index + 1))
        adjusted[index] = running
    return adjusted


def summarize_factors(eligible: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, increment, label, unit in FACTORS:
        selected = eligible.dropna(subset=[column, "capture_state_vxy_mm_s",
                                           "release_xy_offset_mm"])
        success = selected[selected.status.eq("success")][column]
        failure = selected[selected.status.eq("failure")][column]
        outcome = selected.status.eq("success").to_numpy(dtype=float)
        design = np.column_stack([
            np.ones(len(selected)),
            selected.capture_state_vxy_mm_s.to_numpy(dtype=float) / 50.0,
            selected.release_xy_offset_mm.to_numpy(dtype=float) / 10.0,
            selected[column].to_numpy(dtype=float) / increment,
        ])
        fit = _fit_logistic_regression(design, outcome)
        beta = float(fit["coefficients"][-1])
        se = float(np.sqrt(fit["covariance"][-1, -1]))
        z = 1.959963984540054
        rows.append({
            "factor": column, "label": label, "unit": unit,
            "increment": increment, "n": len(selected),
            "success_median": float(success.median()),
            "failure_median": float(failure.median()),
            "mann_whitney_p": float(mannwhitneyu(success, failure).pvalue),
            "adjusted_success_odds_ratio": math.exp(beta),
            "adjusted_or_ci95_low": math.exp(beta - z * se),
            "adjusted_or_ci95_high": math.exp(beta + z * se),
            "adjusted_wald_p": math.erfc(abs(beta / se) / math.sqrt(2.0)),
            "adjustment": "onset vxy + release XY offset",
            "model_converged": bool(fit["converged"]),
        })
    table = pd.DataFrame(rows)
    table["bh_q_six_exploratory_factors"] = _bh_adjust(
        table.adjusted_wald_p.to_numpy()
    )
    return table


def make_delay_bins(eligible: pd.DataFrame) -> pd.DataFrame:
    selected = eligible.dropna(subset=["release_to_field_ms"]).copy()
    selected["bin"] = pd.qcut(selected.release_to_field_ms, 4, labels=False) + 1
    rows = []
    for index, group in selected.groupby("bin"):
        n = len(group)
        successes = int(group.status.eq("success").sum())
        low, high = wilson_interval(successes, n)
        rows.append({
            "bin": int(index), "delay_min_ms": group.release_to_field_ms.min(),
            "delay_max_ms": group.release_to_field_ms.max(), "trials": n,
            "successes": successes, "success_rate": successes / n,
            "success_ci95_low": low, "success_ci95_high": high,
            "interval_method": "Wilson score, 95%",
        })
    return pd.DataFrame(rows)


def plot_delay_bins(bins: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    x = bins.bin.to_numpy(dtype=int)
    rate = bins.success_rate.to_numpy(dtype=float)
    axis.errorbar(x, rate, yerr=[rate - bins.success_ci95_low,
                                 bins.success_ci95_high - rate],
                  fmt="o", capsize=4, color="#1d5f8a", markersize=7)
    axis.set_xticks(x, [f"{lo:.1f}–{hi:.1f}\n(n={n})" for lo, hi, n in zip(
        bins.delay_min_ms, bins.delay_max_ms, bins.trials)])
    axis.set_xlabel("Release-to-field delay (ms), quartiles")
    axis.set_ylabel("5 s capture success rate")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    axis.set_title("Exploratory association; bins are not randomized")
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    checks, eligible = audit(args.data_dir, args.results_dir)
    checks.to_csv(args.results_dir / "data_integrity_checks.csv", index=False)
    if not checks.passed.all():
        failed = ", ".join(checks.loc[~checks.passed, "check"])
        raise SystemExit(f"Data-integrity checks failed: {failed}")
    factors = summarize_factors(eligible)
    delay_bins = make_delay_bins(eligible)
    factors.to_csv(args.results_dir / "exploratory_factor_summary.csv", index=False)
    delay_bins.to_csv(args.results_dir / "capture_success_vs_delay_bins.csv", index=False)
    plot_delay_bins(delay_bins, args.results_dir / "capture_success_vs_delay.png")
    print(f"Checks: {int(checks.passed.sum())}/{len(checks)} passed")
    print(f"Eligible: {len(eligible)}; successes: {eligible.status.eq('success').sum()}")
    print(factors[["factor", "adjusted_success_odds_ratio", "adjusted_wald_p",
                   "bh_q_six_exploratory_factors"]].to_string(index=False))


if __name__ == "__main__":
    main()
