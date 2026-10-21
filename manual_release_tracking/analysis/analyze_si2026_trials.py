from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AUTOMATIC_EXCLUSION_REASON_PARTS = (
    "release_candidate_contact_returned",
    "manual_release_trigger",
    "new_release_candidate_before_previous_trial_finished",
    "operator_rejected_false_release",
    "release_motion_gate_timeout",
)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)

def _numeric(dataframe: pd.DataFrame, columns: list[str]):
    for column in columns:
        if column in dataframe:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")


def load_trial_data(data_directory: Path):
    summary_path = data_directory / "trial_summary.csv"
    timeseries_path = data_directory / "trial_timeseries.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Trial summary not found: {summary_path}")
    if not timeseries_path.is_file():
        raise FileNotFoundError(f"Trial time series not found: {timeseries_path}")

    summary = pd.read_csv(summary_path)
    timeseries = pd.read_csv(timeseries_path)
    if summary.empty:
        raise ValueError(
            f"No completed trials are present in: {summary_path}"
        )
    _numeric(
        summary,
        [
            "field_time_sec",
            "planned_release_x_mm",
            "planned_release_y_mm",
            "planned_release_z_mm",
            "autd_center_x_mm",
            "autd_center_y_mm",
            "autd_center_z_mm",
            "release_x_mm",
            "release_y_mm",
            "release_z_mm",
            "schema_version",
            "release_gate_passed_time_sec",
            "release_gate_elapsed_ms",
            "release_gate_pass_vz_mm_s",
            "release_gate_pass_drop_mm",
            "release_gate_pass_consecutive_count",
            "release_to_field_ms",
            "field_prediction_x_mm",
            "field_prediction_y_mm",
            "field_prediction_z_mm",
            "field_prediction_vz_mm_s",
            "onset_x_mm",
            "onset_y_mm",
            "onset_z_mm",
            "onset_vx_mm_s",
            "onset_vy_mm_s",
            "onset_vz_mm_s",
            "onset_rxy_from_anchor_mm",
            "prediction_error_3d_mm",
            "anchor_x_mm",
            "anchor_y_mm",
            "plan_duration_ms",
            "plan_stop_z_mm",
            "plan_stopping_distance_mm",
            "braking_end_time_sec",
            "braking_end_from_field_ms",
            "braking_end_z_mm",
            "braking_end_vz_mm_s",
            "actual_stopping_distance_mm",
            "stopping_distance_error_mm",
            "plan_stop_z_error_mm",
            "local_hold_reached_time_sec",
            "capture_time_ms",
            "hold_rms_xy_mm",
            "hold_rms_z_mm",
            "ball_mass_kg",
            "minimum_z_mm",
            "maximum_drop_from_onset_mm",
            "maximum_commanded_intensity",
            "success_hold_requirement_sec",
            "success_reached_time_sec",
            "success_reached_elapsed_ms",
            "observation_requirement_sec",
            "post_success_observation_ms",
            "retained_duration_ms",
        ],
    )
    _numeric(
        timeseries,
        [
            "sample_time_sec",
            "measurement_time_sec",
            "elapsed_from_release_ms",
            "elapsed_from_field_ms",
            "filtered_x_mm",
            "filtered_y_mm",
            "filtered_z_mm",
            "vx_mm_s",
            "vy_mm_s",
            "vz_mm_s",
            "release_gate_origin_z_mm",
            "release_gate_downward_displacement_mm",
            "release_gate_consecutive_count",
            "release_gate_passed",
            "success_criterion_reached",
            "braking_elapsed_ms",
            "reference_z_mm",
            "reference_vz_mm_s",
            "reference_acceleration_mm_s2",
            "z_reference_error_mm",
            "vz_reference_error_mm_s",
            "anchor_x_mm",
            "anchor_y_mm",
            "target_x_mm",
            "target_y_mm",
            "target_z_mm",
            "commanded_intensity",
            "plan_duration_ms",
            "plan_stop_z_mm",
            "plan_stopping_distance_mm",
        ],
    )
    return summary, timeseries


def load_trial_exclusions(path: Path):
    if not path.is_file():
        return pd.DataFrame(columns=["trial_id", "reason"])
    exclusions = pd.read_csv(path, dtype=str).fillna("")
    required = {"trial_id", "reason"}
    missing = required.difference(exclusions.columns)
    if missing:
        raise ValueError(
            f"Exclusions file is missing columns {sorted(missing)}: {path}"
        )
    exclusions = exclusions[["trial_id", "reason"]].copy()
    exclusions["trial_id"] = exclusions["trial_id"].str.strip()
    exclusions["reason"] = exclusions["reason"].str.strip()
    exclusions = exclusions[exclusions["trial_id"] != ""]
    if (exclusions["reason"] == "").any():
        raise ValueError(f"Every excluded trial must have a reason: {path}")
    return exclusions.drop_duplicates(
        "trial_id",
        keep="last",
    )


def apply_analysis_exclusions(
    trials: pd.DataFrame,
    exclusions: pd.DataFrame,
):
    marked = trials.copy()
    status_valid = marked["status"].isin(["success", "failure"])
    reason_text = marked["reason"].fillna("").astype(str).str.lower()
    automatic = pd.Series(False, index=marked.index)
    for part in AUTOMATIC_EXCLUSION_REASON_PARTS:
        automatic |= reason_text.str.contains(part, regex=False)

    # In schema 1--3 a regrasp could be caused by a false image-only release
    # decision, so those rows remain excluded for backward compatibility.  In
    # schema 4+, the physical motion gate has already established real downward
    # flight; a later regrasp is therefore a genuine failed capture unless the
    # operator explicitly rejected the trial with the X key.
    regrasp = reason_text.str.contains("regrasp_detected", regex=False)
    schema = pd.to_numeric(
        marked.get("schema_version", pd.Series(np.nan, index=marked.index)),
        errors="coerce",
    )
    gate_passed = marked.get(
        "release_gate_passed_time_sec",
        pd.Series(np.nan, index=marked.index),
    ).notna()
    automatic |= regrasp & ((schema < 4) | ~gate_passed)

    manual_reasons = {}
    if not exclusions.empty:
        unknown = sorted(set(exclusions["trial_id"]) - set(marked["trial_id"]))
        if unknown:
            raise ValueError(
                "Exclusions file contains unknown trial IDs: "
                + ", ".join(unknown)
            )
        manual_reasons = dict(
            zip(exclusions["trial_id"], exclusions["reason"])
        )
    manual_reason = marked["trial_id"].map(manual_reasons).fillna("")
    manual = manual_reason != ""

    marked["analysis_excluded"] = (~status_valid) | automatic | manual
    marked["analysis_eligible"] = status_valid & ~automatic & ~manual
    marked["analysis_exclusion_source"] = ""
    marked.loc[~status_valid, "analysis_exclusion_source"] = "trial_status"
    marked.loc[automatic, "analysis_exclusion_source"] = "automatic_reason"
    marked.loc[manual, "analysis_exclusion_source"] = "manual_file"
    marked["analysis_exclusion_reason"] = ""
    marked.loc[~status_valid, "analysis_exclusion_reason"] = marked.loc[
        ~status_valid, "reason"
    ].fillna("")
    marked.loc[automatic, "analysis_exclusion_reason"] = marked.loc[
        automatic, "reason"
    ].fillna("")
    marked.loc[manual, "analysis_exclusion_reason"] = manual_reason[manual]
    return marked


def eligible_trials(trials: pd.DataFrame):
    if "analysis_eligible" in trials:
        return trials[trials["analysis_eligible"].astype(bool)]
    return trials[trials["status"].isin(["success", "failure"])]


def _interpolate_trial_at_field(samples: pd.DataFrame, field_time_sec: float):
    columns = {
        "onset_x_mm": "filtered_x_mm",
        "onset_y_mm": "filtered_y_mm",
        "onset_z_mm": "filtered_z_mm",
        "onset_vx_mm_s": "vx_mm_s",
        "onset_vy_mm_s": "vy_mm_s",
        "onset_vz_mm_s": "vz_mm_s",
    }
    usable = samples.dropna(subset=["measurement_time_sec"]).copy()
    usable = usable.sort_values("measurement_time_sec")
    usable = usable.drop_duplicates("measurement_time_sec", keep="last")
    if usable.empty:
        return {}
    times = usable["measurement_time_sec"].to_numpy(dtype=float)
    if field_time_sec < times[0] or field_time_sec > times[-1]:
        return {}

    result = {}
    for output_name, source_name in columns.items():
        valid = usable[["measurement_time_sec", source_name]].dropna()
        if len(valid) < 2:
            continue
        result[output_name] = float(
            np.interp(
                field_time_sec,
                valid["measurement_time_sec"].to_numpy(dtype=float),
                valid[source_name].to_numpy(dtype=float),
            )
        )
    return result


def enrich_onset_metrics(summary: pd.DataFrame, timeseries: pd.DataFrame):
    enriched = summary.copy()
    for index, trial in enriched.iterrows():
        field_time = trial.get("field_time_sec")
        if not np.isfinite(field_time):
            continue
        samples = timeseries[timeseries["trial_id"] == trial["trial_id"]]
        interpolated = _interpolate_trial_at_field(samples, float(field_time))
        for column, value in interpolated.items():
            enriched.at[index, column] = value

        x = enriched.at[index, "onset_x_mm"]
        y = enriched.at[index, "onset_y_mm"]
        z_mm = enriched.at[index, "onset_z_mm"]
        anchor_x = enriched.at[index, "anchor_x_mm"]
        anchor_y = enriched.at[index, "anchor_y_mm"]
        if all(np.isfinite(value) for value in (x, y, anchor_x, anchor_y)):
            enriched.at[index, "onset_rxy_from_anchor_mm"] = float(
                np.hypot(x - anchor_x, y - anchor_y)
            )

        prediction_columns = (
            ("prediction_error_x_mm", "field_prediction_x_mm", x),
            ("prediction_error_y_mm", "field_prediction_y_mm", y),
            ("prediction_error_z_mm", "field_prediction_z_mm", z_mm),
        )
        errors = []
        for output, predicted, measured in prediction_columns:
            predicted_value = enriched.at[index, predicted]
            if np.isfinite(predicted_value) and np.isfinite(measured):
                error = float(measured - predicted_value)
                enriched.at[index, output] = error
                errors.append(error)
        if len(errors) == 3:
            enriched.at[index, "prediction_error_3d_mm"] = float(
                np.linalg.norm(errors)
            )
    # The field prediction uses the dedicated post-release estimator and the
    # measured command-completion time. It is therefore the primary velocity
    # estimate for the capture-condition plot. Position can be interpolated
    # directly from the surrounding stereo samples.
    enriched["capture_state_z_mm"] = enriched["onset_z_mm"].fillna(
        enriched["field_prediction_z_mm"]
    )
    enriched["capture_state_vz_mm_s"] = enriched[
        "field_prediction_vz_mm_s"
    ].fillna(enriched["onset_vz_mm_s"])
    enriched["maximum_drop_from_onset_mm"] = (
        enriched["capture_state_z_mm"] - enriched["minimum_z_mm"]
    )
    return enriched


def select_representative_trial(
    summary: pd.DataFrame,
    requested_trial_id: str | None,
):
    if requested_trial_id:
        match = summary[summary["trial_id"] == requested_trial_id]
        if match.empty:
            raise ValueError(f"Unknown trial_id: {requested_trial_id}")
        return requested_trial_id

    successful = summary[summary["status"] == "success"].dropna(
        subset=["capture_state_vz_mm_s"]
    )
    if not successful.empty:
        if "observation_requirement_sec" in successful:
            complete = successful[
                successful["retained_duration_ms"]
                >= successful["observation_requirement_sec"] * 1000.0 - 50.0
            ]
            if not complete.empty:
                successful = complete
        median_speed = successful["capture_state_vz_mm_s"].abs().median()
        index = (
            successful["capture_state_vz_mm_s"].abs() - median_speed
        ).abs().idxmin()
        return str(successful.loc[index, "trial_id"])
    if summary.empty:
        raise ValueError("No trials are available")
    return str(summary.iloc[0]["trial_id"])


def resolve_release_center(
    trials: pd.DataFrame,
    *,
    center_x_mm: float | None = None,
    center_y_mm: float | None = None,
    center_z_mm: float | None = None,
):
    """Resolve the nominal AUTD-center release position used for offsets.

    An explicit command-line value has priority. Otherwise the median planned
    coordinate is used, followed by the median measured release coordinate.
    """
    resolved = []
    for axis, requested in zip(
        ("x", "y", "z"),
        (center_x_mm, center_y_mm, center_z_mm),
    ):
        if requested is not None and np.isfinite(requested):
            resolved.append(float(requested))
            continue
        center_column = f"autd_center_{axis}_mm"
        autd_center = (
            trials[center_column].dropna()
            if center_column in trials
            else pd.Series(dtype=float)
        )
        planned_column = f"planned_release_{axis}_mm"
        planned = (
            trials[planned_column].dropna()
            if planned_column in trials
            else pd.Series(dtype=float)
        )
        measured = eligible_trials(trials)[f"release_{axis}_mm"].dropna()
        if not autd_center.empty:
            source = autd_center
        elif not planned.empty:
            source = planned
        else:
            source = measured
        resolved.append(float(source.median()) if not source.empty else np.nan)
    return tuple(resolved)


def add_centered_release_metrics(
    trials: pd.DataFrame,
    center: tuple[float, float, float],
):
    enriched = trials.copy()
    for axis, center_value in zip(("x", "y", "z"), center):
        enriched[f"release_d{axis}_mm"] = (
            enriched[f"release_{axis}_mm"] - center_value
        )
        enriched[f"release_center_{axis}_mm"] = center_value
    enriched["release_xy_offset_mm"] = np.hypot(
        enriched["release_dx_mm"],
        enriched["release_dy_mm"],
    )
    enriched["release_3d_offset_mm"] = np.sqrt(
        enriched["release_dx_mm"] ** 2
        + enriched["release_dy_mm"] ** 2
        + enriched["release_dz_mm"] ** 2
    )
    enriched["capture_state_abs_vz_mm_s"] = enriched[
        "capture_state_vz_mm_s"
    ].abs()
    enriched["capture_state_vxy_mm_s"] = np.hypot(
        enriched["onset_vx_mm_s"],
        enriched["onset_vy_mm_s"],
    )
    return enriched


def make_outcome_summary(
    trials: pd.DataFrame,
    center: tuple[float, float, float],
    target_valid_trials: int = 100,
):
    groups = [("all", "", trials)]
    if "condition_id" in trials:
        groups.extend(
            ("condition", str(condition_id), group)
            for condition_id, group in trials.groupby("condition_id")
        )
    rows = []
    for scope, condition_id, group in groups:
        valid = eligible_trials(group)
        successes = int((valid["status"] == "success").sum())
        total = int(len(valid))
        lower, upper = wilson_interval(successes, total)
        rows.append(
            {
                "scope": scope,
                "condition_id": condition_id,
                "release_center_x_mm": center[0],
                "release_center_y_mm": center[1],
                "release_center_z_mm": center[2],
                "attempts": int(len(group)),
                "eligible_trials": total,
                "successes": successes,
                "failures": total - successes,
                "excluded_trials": int(len(group) - total),
                "target_valid_trials": int(target_valid_trials),
                "remaining_valid_trials": max(0, int(target_valid_trials) - total),
                "target_reached": int(total >= int(target_valid_trials)),
                "success_rate": successes / total if total else np.nan,
                "success_ci95_low": lower,
                "success_ci95_high": upper,
            }
        )
    return pd.DataFrame(rows)


def failure_reason_category(reason: str):
    value = str(reason).lower()
    if "regrasp" in value or "contact_returned" in value:
        return "Regrasp/contact"
    if "measurement" in value or "motion_fit" in value:
        return "Tracking/estimation"
    if "workspace" in value or "left_z" in value:
        return "Workspace departure"
    if "settle_timeout" in value:
        return "Settling timeout"
    if "release_confirmation" in value:
        return "Release confirmation"
    if "paused" in value or "application_closed" in value:
        return "Operator interruption"
    return "Other"


def make_failure_reason_summary(trials: pd.DataFrame):
    failures = eligible_trials(trials)
    failures = failures[failures["status"] == "failure"].copy()
    if failures.empty:
        return pd.DataFrame(columns=["category", "reason", "count"])
    failures["category"] = failures["reason"].map(failure_reason_category)
    return (
        failures.groupby(["category", "reason"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["count", "category"], ascending=[False, True])
    )


def make_release_range_summary(trials: pd.DataFrame):
    selected = eligible_trials(trials)
    metrics = (
        "release_x_mm",
        "release_y_mm",
        "release_z_mm",
        "release_xy_offset_mm",
        "release_3d_offset_mm",
    )
    rows = []
    for status in ("success", "failure"):
        group = selected[selected["status"] == status]
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "status": status,
                    "metric": metric,
                    "n": len(values),
                    "minimum": np.min(values) if len(values) else np.nan,
                    "q1": np.quantile(values, 0.25) if len(values) else np.nan,
                    "median": np.median(values) if len(values) else np.nan,
                    "q3": np.quantile(values, 0.75) if len(values) else np.nan,
                    "maximum": np.max(values) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def behavior_pattern(trial: pd.Series):
    if str(trial.get("status", "")) == "success":
        observation_sec = trial.get("observation_requirement_sec", 10.0)
        retained_ms = trial.get("retained_duration_ms")
        if (
            pd.notna(observation_sec)
            and pd.notna(retained_ms)
            and float(retained_ms) < float(observation_sec) * 1000.0 - 50.0
        ):
            return "Held 5 s; observation ended early"
        return "Held at least 5 s"
    reason = str(trial.get("reason", "")).lower()
    if "workspace" in reason or "left_z" in reason:
        return "Left workspace"
    if "measurement" in reason or "motion_fit" in reason:
        return "Tracking/estimation loss"
    if "regrasp" in reason:
        return "Regrasp after release"
    if pd.notna(trial.get("local_hold_reached_time_sec")):
        return "Lost after LOCAL_HOLD (<5 s)"
    if pd.notna(trial.get("braking_end_time_sec")):
        return "Lost during settling"
    if pd.notna(trial.get("field_time_sec")):
        return "Lost during braking"
    return "Other failure"


def make_behavior_pattern_summary(trials: pd.DataFrame):
    selected = eligible_trials(trials).copy()
    if selected.empty:
        return pd.DataFrame(columns=["behavior_pattern", "count", "fraction"])
    selected["behavior_pattern"] = selected.apply(behavior_pattern, axis=1)
    result = (
        selected.groupby("behavior_pattern")
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    result["fraction"] = result["count"] / len(selected)
    return result


def make_braking_validation_summary(trials: pd.DataFrame):
    selected = eligible_trials(trials).copy()
    columns = [
        "trial_id",
        "status",
        "plan_duration_ms",
        "plan_stop_z_mm",
        "plan_stopping_distance_mm",
        "braking_end_from_field_ms",
        "braking_end_reason",
        "braking_end_z_mm",
        "braking_end_vz_mm_s",
        "actual_stopping_distance_mm",
        "stopping_distance_error_mm",
        "plan_stop_z_error_mm",
    ]
    for column in columns:
        if column not in selected:
            selected[column] = np.nan
    return selected[columns]


def _placeholder_figure(output_path: Path, message: str):
    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    _save_figure(figure, output_path)


def plot_successful_trajectory(
    summary: pd.DataFrame,
    timeseries: pd.DataFrame,
    output_directory: Path,
    trial_id: str | None,
):
    output_path = output_directory / "successful_release_trajectory"
    if trial_id is None:
        _placeholder_figure(output_path, "No successful 5 s hold trial is available")
        return
    samples = timeseries[timeseries["trial_id"] == trial_id].copy()
    samples = samples.sort_values("sample_time_sec")
    samples = samples.dropna(
        subset=["elapsed_from_release_ms", "filtered_x_mm", "filtered_y_mm", "filtered_z_mm"]
    )
    if samples.empty:
        _placeholder_figure(output_path, f"No trajectory samples for {trial_id}")
        return

    trial = summary[summary["trial_id"] == trial_id].iloc[0]
    time_sec = samples["elapsed_from_release_ms"].to_numpy(dtype=float) / 1000.0
    field_time_sec = float(trial["release_to_field_ms"]) / 1000.0
    success_time_sec = float(trial.get("success_hold_requirement_sec", 5.0))
    figure = plt.figure(figsize=(9.0, 7.0), constrained_layout=True)
    axes = [
        figure.add_subplot(2, 2, 1),
        figure.add_subplot(2, 2, 2),
        figure.add_subplot(2, 2, 3),
        figure.add_subplot(2, 2, 4, projection="3d"),
    ]
    coordinates = (
        ("filtered_x_mm", "X position [mm]", "#762a83"),
        ("filtered_y_mm", "Y position [mm]", "#e08214"),
        ("filtered_z_mm", "Z position [mm]", "#2166ac"),
    )
    for axis, (column, label, color) in zip(axes[:3], coordinates):
        axis.plot(time_sec, samples[column], color=color, linewidth=1.2)
        axis.axvline(0.0, color="black", linewidth=0.8, label="Release")
        axis.axvline(
            field_time_sec,
            color="#d73027",
            linestyle="--",
            linewidth=1.0,
            label="First command complete",
        )
        axis.axvline(
            success_time_sec,
            color="#1a9850",
            linestyle=":",
            linewidth=1.0,
            label=f"{success_time_sec:g} s success criterion",
        )
        axis.set_xlabel("Time from release [s]")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[3].plot(
        samples["filtered_x_mm"],
        samples["filtered_y_mm"],
        samples["filtered_z_mm"],
        color="#2166ac",
        linewidth=1.1,
    )
    axes[3].scatter(
        [samples["filtered_x_mm"].iloc[0]],
        [samples["filtered_y_mm"].iloc[0]],
        [samples["filtered_z_mm"].iloc[0]],
        color="#1a9850",
        s=40,
        label="Release",
    )
    axes[3].scatter(
        [samples["filtered_x_mm"].iloc[-1]],
        [samples["filtered_y_mm"].iloc[-1]],
        [samples["filtered_z_mm"].iloc[-1]],
        color="#d73027",
        s=40,
        label="Observation end",
    )
    axes[3].set_xlabel("X [mm]")
    axes[3].set_ylabel("Y [mm]")
    axes[3].set_zlabel("Z [mm]")
    axes[3].legend(loc="best", fontsize=8)
    figure.suptitle(f"Successful release trajectory: {trial_id}", fontsize=10)
    _save_figure(figure, output_path)


def _resample_success_series(
    samples: pd.DataFrame,
    time_grid_ms: np.ndarray,
    column: str,
    *,
    initial_value: float | None = None,
    previous_value: bool = False,
):
    usable = samples[["elapsed_from_release_ms", column]].dropna().copy()
    if initial_value is not None:
        usable = pd.concat(
            [
                pd.DataFrame(
                    {
                        "elapsed_from_release_ms": [0.0],
                        column: [float(initial_value)],
                    }
                ),
                usable,
            ],
            ignore_index=True,
        )
    usable = (
        usable.groupby("elapsed_from_release_ms", as_index=False)[column]
        .mean()
        .sort_values("elapsed_from_release_ms")
    )
    if usable.empty:
        return np.full_like(time_grid_ms, np.nan, dtype=float)

    time_ms = usable["elapsed_from_release_ms"].to_numpy(dtype=float)
    values = usable[column].to_numpy(dtype=float)
    result = np.full_like(time_grid_ms, np.nan, dtype=float)
    covered = (time_grid_ms >= time_ms[0]) & (time_grid_ms <= time_ms[-1])
    if previous_value:
        indices = np.searchsorted(time_ms, time_grid_ms[covered], side="right") - 1
        result[covered] = values[np.clip(indices, 0, len(values) - 1)]
    else:
        result[covered] = np.interp(time_grid_ms[covered], time_ms, values)
    return result


def plot_successful_trials_mean_response(
    trials: pd.DataFrame,
    timeseries: pd.DataFrame,
    output_directory: Path,
):
    """Plot all successful trials and their release-aligned aggregate response."""
    output_path = output_directory / "successful_trials_mean_response"
    successful = eligible_trials(trials)
    successful = successful[successful["status"] == "success"].copy()
    if successful.empty:
        _placeholder_figure(output_path, "No successful trials are available")
        return

    time_grid_ms = np.arange(0.0, 5000.0 + 5.0, 5.0)
    matrices = {
        "filtered_z_mm": [],
        "vz_mm_s": [],
        "commanded_intensity": [],
    }
    for _, trial in successful.iterrows():
        samples = timeseries[timeseries["trial_id"] == trial["trial_id"]].copy()
        samples = samples.sort_values("elapsed_from_release_ms")
        samples = samples[samples["elapsed_from_release_ms"] >= 0.0]
        matrices["filtered_z_mm"].append(
            _resample_success_series(
                samples,
                time_grid_ms,
                "filtered_z_mm",
                initial_value=trial.get("release_z_mm"),
            )
        )
        matrices["vz_mm_s"].append(
            _resample_success_series(
                samples,
                time_grid_ms,
                "vz_mm_s",
            )
        )
        matrices["commanded_intensity"].append(
            _resample_success_series(
                samples,
                time_grid_ms,
                "commanded_intensity",
                initial_value=0.0,
                previous_value=True,
            )
        )

    for column in matrices:
        matrices[column] = np.asarray(matrices[column], dtype=float)

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(8.0, 7.2),
        sharex=True,
        constrained_layout=True,
    )
    panels = (
        ("filtered_z_mm", "Measured $z$ [mm]", "#2166ac"),
        ("vz_mm_s", "Measured $v_z$ [mm/s]", "#d73027"),
        ("commanded_intensity", "AUTD output", "#762a83"),
    )
    aggregate = pd.DataFrame({"time_from_release_s": time_grid_ms / 1000.0})
    for axis, (column, ylabel, color) in zip(axes, panels):
        matrix = matrices[column]
        for values in matrix:
            axis.plot(
                time_grid_ms / 1000.0,
                values,
                color=color,
                alpha=0.10,
                linewidth=0.55,
            )
        count = np.sum(np.isfinite(matrix), axis=0)
        mean = np.full(matrix.shape[1], np.nan, dtype=float)
        lower = np.full(matrix.shape[1], np.nan, dtype=float)
        upper = np.full(matrix.shape[1], np.nan, dtype=float)
        available = count > 0
        mean[available] = (
            np.nansum(matrix[:, available], axis=0) / count[available]
        )
        for sample_index in np.flatnonzero(available):
            values = matrix[:, sample_index]
            values = values[np.isfinite(values)]
            lower[sample_index], upper[sample_index] = np.percentile(
                values,
                [25.0, 75.0],
            )
        axis.fill_between(
            time_grid_ms / 1000.0,
            lower,
            upper,
            color=color,
            alpha=0.18,
            linewidth=0.0,
            label="Interquartile range",
        )
        axis.plot(
            time_grid_ms / 1000.0,
            mean,
            color=color,
            linewidth=2.2,
            label="Mean",
        )
        axis.set_ylabel(ylabel, fontsize=16)
        axis.set_xlim(0.0, 5.0)
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(True, alpha=0.22)
        aggregate[f"{column}_mean"] = mean
        aggregate[f"{column}_q1"] = lower
        aggregate[f"{column}_q3"] = upper
        aggregate[f"{column}_n"] = count

    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[2].set_ylim(-0.02, 0.84)
    axes[2].set_xlabel("Time from release [s]", fontsize=16)
    axes[2].set_xticks(np.arange(0.0, 5.1, 1.0))
    axes[0].plot([], [], color="#2166ac", alpha=0.25, linewidth=0.8,
                 label=f"Individual trials ($n={len(successful)}$)")
    axes[0].legend(loc="best", fontsize=13, ncol=3)
    _save_figure(figure, output_path)
    aggregate.to_csv(
        output_directory / "successful_trials_mean_response.csv",
        index=False,
    )


def plot_successful_z_trajectories(
    trials: pd.DataFrame,
    timeseries: pd.DataFrame,
    output_directory: Path,
):
    """Create separate, paper-ready z-position and z-velocity figures."""
    successful = eligible_trials(trials)
    successful = successful[successful["status"] == "success"].copy()
    position_path = output_directory / "successful_z_position_trajectory"
    velocity_path = output_directory / "successful_z_velocity_trajectory"
    if successful.empty:
        _placeholder_figure(position_path, "No successful trials are available")
        _placeholder_figure(velocity_path, "No successful trials are available")
        return

    time_grid_ms = np.arange(0.0, 1000.0 + 2.0, 2.0)
    z_rows = []
    vz_rows = []
    for _, trial in successful.iterrows():
        samples = timeseries[timeseries["trial_id"] == trial["trial_id"]].copy()
        samples = samples.sort_values("elapsed_from_release_ms")
        samples = samples[samples["elapsed_from_release_ms"] >= 0.0]
        z_rows.append(
            _resample_success_series(
                samples,
                time_grid_ms,
                "filtered_z_mm",
                initial_value=trial.get("release_z_mm"),
            )
        )
        vz_rows.append(
            _resample_success_series(samples, time_grid_ms, "vz_mm_s")
        )

    z_matrix = np.asarray(z_rows, dtype=float)
    vz_matrix = np.asarray(vz_rows, dtype=float)
    field_onset_ms = float(successful["release_to_field_ms"].median())
    hold_onset_ms = float(
        (
            successful["release_to_field_ms"]
            + successful["braking_end_from_field_ms"]
        ).median()
    )

    def summarize(matrix: np.ndarray):
        count = np.sum(np.isfinite(matrix), axis=0)
        median = np.full(matrix.shape[1], np.nan, dtype=float)
        q1 = np.full(matrix.shape[1], np.nan, dtype=float)
        q3 = np.full(matrix.shape[1], np.nan, dtype=float)
        for sample_index in np.flatnonzero(count > 0):
            values = matrix[:, sample_index]
            values = values[np.isfinite(values)]
            median[sample_index] = np.median(values)
            q1[sample_index], q3[sample_index] = np.percentile(
                values,
                [25.0, 75.0],
            )
        return count, median, q1, q3

    z_count, z_median, z_q1, z_q3 = summarize(z_matrix)
    vz_count, vz_median, vz_q1, vz_q3 = summarize(vz_matrix)
    time_sec = time_grid_ms / 1000.0

    def draw(
        matrix: np.ndarray,
        median: np.ndarray,
        q1: np.ndarray,
        q3: np.ndarray,
        *,
        color: str,
        ylabel: str,
        output_path: Path,
        zero_line: bool = False,
    ):
        figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        for values in matrix:
            axis.plot(time_sec, values, color="#7f7f7f", alpha=0.12, linewidth=0.55)
        axis.fill_between(
            time_sec,
            q1,
            q3,
            color=color,
            alpha=0.22,
            linewidth=0.0,
            label="Interquartile range",
        )
        axis.plot(time_sec, median, color=color, linewidth=2.5, label="Median")
        axis.axvline(
            field_onset_ms / 1000.0,
            color="#e66101",
            linestyle="--",
            linewidth=1.5,
            label=f"Field onset (median {field_onset_ms:.1f} ms)",
        )
        axis.axvline(
            hold_onset_ms / 1000.0,
            color="#1b9e77",
            linestyle=":",
            linewidth=1.8,
            label=f"Hold onset (median {hold_onset_ms:.1f} ms)",
        )
        if zero_line:
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Time from release [s]", fontsize=15)
        axis.set_ylabel(ylabel, fontsize=15)
        axis.tick_params(axis="both", labelsize=13)
        axis.grid(True, alpha=0.22)
        axis.plot(
            [],
            [],
            color="#7f7f7f",
            alpha=0.45,
            linewidth=0.8,
            label=f"Individual successful trials ($n={len(successful)}$)",
        )
        axis.legend(loc="best", fontsize=10)
        _save_figure(figure, output_path)

    draw(
        z_matrix,
        z_median,
        z_q1,
        z_q3,
        color="#2166ac",
        ylabel="Measured $z$ position [mm]",
        output_path=position_path,
    )
    draw(
        vz_matrix,
        vz_median,
        vz_q1,
        vz_q3,
        color="#d73027",
        ylabel="Measured $z$ velocity [mm/s]",
        output_path=velocity_path,
        zero_line=True,
    )

    pd.DataFrame(
        {
            "time_from_release_s": time_sec,
            "z_median_mm": z_median,
            "z_q1_mm": z_q1,
            "z_q3_mm": z_q3,
            "z_n": z_count,
            "vz_median_mm_s": vz_median,
            "vz_q1_mm_s": vz_q1,
            "vz_q3_mm_s": vz_q3,
            "vz_n": vz_count,
        }
    ).to_csv(output_directory / "successful_z_trajectory_summary.csv", index=False)


def plot_failed_trials_mean_response(
    trials: pd.DataFrame,
    timeseries: pd.DataFrame,
    output_directory: Path,
):
    """Plot failure responses while showing the changing contributing count."""
    output_path = output_directory / "failed_trials_mean_response"
    failed = eligible_trials(trials)
    failed = failed[failed["status"] == "failure"].copy()
    if failed.empty:
        _placeholder_figure(output_path, "No failed trials are available")
        return

    time_grid_ms = np.arange(0.0, 5000.0 + 5.0, 5.0)
    matrices = {
        "filtered_z_mm": [],
        "vz_mm_s": [],
        "commanded_intensity": [],
    }
    for _, trial in failed.iterrows():
        samples = timeseries[timeseries["trial_id"] == trial["trial_id"]].copy()
        samples = samples.sort_values("elapsed_from_release_ms")
        samples = samples[samples["elapsed_from_release_ms"] >= 0.0]
        matrices["filtered_z_mm"].append(
            _resample_success_series(
                samples,
                time_grid_ms,
                "filtered_z_mm",
                initial_value=trial.get("release_z_mm"),
            )
        )
        matrices["vz_mm_s"].append(
            _resample_success_series(
                samples,
                time_grid_ms,
                "vz_mm_s",
            )
        )
        matrices["commanded_intensity"].append(
            _resample_success_series(
                samples,
                time_grid_ms,
                "commanded_intensity",
                initial_value=0.0,
                previous_value=True,
            )
        )

    for column in matrices:
        matrices[column] = np.asarray(matrices[column], dtype=float)

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(8.0, 8.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.45]},
    )
    panels = (
        ("filtered_z_mm", "Measured $z$ [mm]", "#2166ac"),
        ("vz_mm_s", "Measured $v_z$ [mm/s]", "#d73027"),
        ("commanded_intensity", "Normalized output intensity", "#762a83"),
    )
    aggregate = pd.DataFrame({"time_from_release_s": time_grid_ms / 1000.0})
    contribution_count = None
    for axis, (column, ylabel, color) in zip(axes[:3], panels):
        matrix = matrices[column]
        for values in matrix:
            axis.plot(
                time_grid_ms / 1000.0,
                values,
                color=color,
                alpha=0.08,
                linewidth=0.55,
            )
        count = np.sum(np.isfinite(matrix), axis=0)
        if column == "filtered_z_mm":
            contribution_count = count
        mean = np.full(matrix.shape[1], np.nan, dtype=float)
        lower = np.full(matrix.shape[1], np.nan, dtype=float)
        upper = np.full(matrix.shape[1], np.nan, dtype=float)
        available = count > 0
        mean[available] = (
            np.nansum(matrix[:, available], axis=0) / count[available]
        )
        for sample_index in np.flatnonzero(available):
            values = matrix[:, sample_index]
            values = values[np.isfinite(values)]
            lower[sample_index], upper[sample_index] = np.percentile(
                values,
                [25.0, 75.0],
            )
        displayable = count >= 5
        axis.fill_between(
            time_grid_ms / 1000.0,
            np.where(displayable, lower, np.nan),
            np.where(displayable, upper, np.nan),
            color=color,
            alpha=0.18,
            linewidth=0.0,
            label="Interquartile range",
        )
        axis.plot(
            time_grid_ms / 1000.0,
            np.where(displayable, mean, np.nan),
            color=color,
            linewidth=2.2,
            label=r"Mean of available trials ($n\geq5$)",
        )
        axis.set_ylabel(ylabel)
        axis.set_xlim(0.0, 5.0)
        axis.grid(True, alpha=0.22)
        aggregate[f"{column}_mean"] = mean
        aggregate[f"{column}_q1"] = lower
        aggregate[f"{column}_q3"] = upper
        aggregate[f"{column}_n"] = count

    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[2].set_ylim(-0.02, 0.84)
    axes[0].plot([], [], color="#2166ac", alpha=0.25, linewidth=0.8,
                 label=f"Individual trials ($n={len(failed)}$)")
    axes[0].legend(loc="best", fontsize=9, ncol=3)

    axes[3].step(
        time_grid_ms / 1000.0,
        contribution_count,
        where="post",
        color="#4d4d4d",
        linewidth=1.8,
    )
    axes[3].set_ylabel("Trials\nremaining")
    axes[3].set_ylim(0.0, len(failed) + 3.0)
    axes[3].set_yticks([0, 20, 40, 60])
    axes[3].set_xlabel("Time from release [s]")
    axes[3].set_xticks(np.arange(0.0, 5.1, 1.0))
    axes[3].grid(True, alpha=0.22)
    aggregate["trials_remaining"] = contribution_count

    figure.suptitle(
        f"Measured response of failed capture trials ($n={len(failed)}$)",
        fontsize=12,
    )
    _save_figure(figure, output_path)
    aggregate.to_csv(
        output_directory / "failed_trials_mean_response.csv",
        index=False,
    )


def make_failed_trial_metric_summary(trials: pd.DataFrame):
    failed = eligible_trials(trials)
    failed = failed[failed["status"] == "failure"].copy()
    if failed.empty:
        return pd.DataFrame()

    reason_labels = {
        "capture_stereo_measurement_timeout": "Tracking/estimation loss",
        "sphere_left_z_workspace_before_success": "Workspace departure",
        "regrasp_detected_during_auto_hold": "Regrasp/contact",
    }
    metrics = (
        "retained_duration_ms",
        "release_to_field_ms",
        "capture_state_vz_mm_s",
        "capture_state_vxy_mm_s",
        "prediction_error_3d_mm",
        "plan_duration_ms",
        "plan_stopping_distance_mm",
        "actual_stopping_distance_mm",
        "stopping_distance_error_mm",
        "braking_end_vz_mm_s",
        "maximum_drop_from_onset_mm",
        "maximum_commanded_intensity",
    )
    groups = [("All failures", failed)]
    groups.extend(
        (
            reason_labels.get(str(reason), str(reason)),
            group,
        )
        for reason, group in failed.groupby("reason", sort=False)
    )
    rows = []
    for label, group in groups:
        row = {
            "failure_category": label,
            "trials": len(group),
            "fraction_of_failures": len(group) / len(failed),
            "braking_end_trials": int(group["braking_end_time_sec"].notna().sum()),
            "local_hold_trials": int(group["local_hold_reached_time_sec"].notna().sum()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_n"] = len(values)
            row[f"{metric}_mean"] = values.mean() if not values.empty else np.nan
            row[f"{metric}_std"] = (
                values.std(ddof=1) if len(values) >= 2 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_release_outcomes_3d(
    trials: pd.DataFrame,
    output_directory: Path,
    center: tuple[float, float, float],
):
    output_path = output_directory / "release_outcomes_3d"
    selected = eligible_trials(trials).dropna(
        subset=["release_x_mm", "release_y_mm", "release_z_mm"]
    )
    if selected.empty:
        _placeholder_figure(output_path, "No valid release positions are available")
        return

    figure = plt.figure(figsize=(12.0, 4.3), constrained_layout=True)
    axis_xy = figure.add_subplot(1, 3, 1)
    axis_xz = figure.add_subplot(1, 3, 2)
    axis_3d = figure.add_subplot(1, 3, 3, projection="3d")
    styles = {
        "success": ("#1a9850", "o", "Success"),
        "failure": ("#d73027", "x", "Failure"),
    }
    for status, (color, marker, label) in styles.items():
        group = selected[selected["status"] == status]
        if group.empty:
            continue
        axis_xy.scatter(
            group["release_x_mm"],
            group["release_y_mm"],
            color=color,
            marker=marker,
            s=42,
            alpha=0.8,
        )
        axis_xz.scatter(
            group["release_x_mm"],
            group["release_z_mm"],
            color=color,
            marker=marker,
            s=42,
            alpha=0.8,
        )
        axis_3d.scatter(
            group["release_x_mm"],
            group["release_y_mm"],
            group["release_z_mm"],
            color=color,
            marker=marker,
            s=48,
            alpha=0.8,
            label=f"{label} (n={len(group)})",
        )
    if all(np.isfinite(value) for value in center):
        axis_xy.scatter(
            [center[0]],
            [center[1]],
            color="black",
            marker="*",
            s=120,
        )
        axis_xz.scatter(
            [center[0]],
            [center[2]],
            color="black",
            marker="*",
            s=120,
        )
        axis_3d.scatter(
            [center[0]],
            [center[1]],
            [center[2]],
            color="black",
            marker="*",
            s=140,
            label="Nominal center",
        )
    axis_xy.set_xlabel("Release X [mm]")
    axis_xy.set_ylabel("Release Y [mm]")
    axis_xy.set_title("XY projection")
    axis_xz.set_xlabel("Release X [mm]")
    axis_xz.set_ylabel("Release Z [mm]")
    axis_xz.set_title("XZ projection")
    axis_3d.set_xlabel("Release X [mm]")
    axis_3d.set_ylabel("Release Y [mm]")
    axis_3d.set_zlabel("Release Z [mm]")
    axis_3d.set_title("3D")
    for axis in (axis_xy, axis_xz):
        axis.grid(True, alpha=0.2)
        axis.set_aspect("equal", adjustable="datalim")
    axis_3d.legend(loc="best", fontsize=8)
    figure.suptitle("Measured release positions and capture outcomes", fontsize=10)
    _save_figure(figure, output_path)


def plot_outcome_trajectory_comparison(
    trials: pd.DataFrame,
    timeseries: pd.DataFrame,
    output_directory: Path,
):
    output_path = output_directory / "outcome_trajectory_comparison"
    valid = eligible_trials(trials)
    if valid.empty:
        _placeholder_figure(output_path, "No valid trajectories are available")
        return

    figure, axes = plt.subplots(3, 1, figsize=(8.0, 7.8), sharex=True, constrained_layout=True)
    coordinate_columns = (
        ("filtered_x_mm", "$\\Delta x$ [mm]"),
        ("filtered_y_mm", "$\\Delta y$ [mm]"),
        ("filtered_z_mm", "$\\Delta z$ [mm]"),
    )
    colors = {"success": "#1a9850", "failure": "#d73027"}
    for _, trial in valid.iterrows():
        samples = timeseries[timeseries["trial_id"] == trial["trial_id"]].copy()
        samples = samples.sort_values("elapsed_from_release_ms")
        samples = samples[
            samples["elapsed_from_release_ms"].between(0.0, 10000.0)
        ]
        if samples.empty:
            continue
        for axis, (column, label) in zip(axes, coordinate_columns):
            usable = samples[["elapsed_from_release_ms", column]].dropna()
            if usable.empty:
                continue
            axis.plot(
                usable["elapsed_from_release_ms"] / 1000.0,
                usable[column] - usable[column].iloc[0],
                color=colors[str(trial["status"])],
                alpha=0.22,
                linewidth=0.8,
            )
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Time from release [s]")
    axes[0].plot([], [], color=colors["success"], label="Success")
    axes[0].plot([], [], color=colors["failure"], label="Failure")
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle("Post-release trajectories by capture outcome", fontsize=10)
    _save_figure(figure, output_path)


def plot_control_mechanism_validation(
    summary: pd.DataFrame,
    timeseries: pd.DataFrame,
    output_directory: Path,
    trial_id: str | None,
):
    output_path = output_directory / "control_mechanism_validation"
    if trial_id is None:
        _placeholder_figure(output_path, "No successful trial is available")
        return
    samples = timeseries[timeseries["trial_id"] == trial_id].copy()
    samples = samples.sort_values("elapsed_from_release_ms")
    samples = samples[samples["elapsed_from_release_ms"].between(0.0, 10000.0)]
    if samples.empty:
        _placeholder_figure(output_path, f"No control samples for {trial_id}")
        return

    trial = summary[summary["trial_id"] == trial_id].iloc[0]
    time_sec = samples["elapsed_from_release_ms"] / 1000.0
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(8.4, 8.7),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(time_sec, samples["filtered_z_mm"], label="Measured Z", linewidth=1.2)
    if samples["reference_z_mm"].notna().any():
        axes[0].plot(
            time_sec,
            samples["reference_z_mm"],
            linestyle="--",
            label="Braking reference Z",
        )
    plan_stop_z = trial.get("plan_stop_z_mm")
    if pd.notna(plan_stop_z):
        axes[0].axhline(
            float(plan_stop_z),
            color="black",
            linestyle=":",
            linewidth=1.0,
            label="Planned stop Z",
        )
    axes[0].set_ylabel("Z [mm]")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(time_sec, samples["vz_mm_s"], label="Measured $v_z$", linewidth=1.2)
    if samples["reference_vz_mm_s"].notna().any():
        axes[1].plot(
            time_sec,
            samples["reference_vz_mm_s"],
            linestyle="--",
            label="Braking reference $v_z$",
        )
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_ylabel("$v_z$ [mm/s]")
    axes[1].legend(loc="best", fontsize=8)

    axes[2].plot(
        time_sec,
        samples["commanded_intensity"],
        color="#762a83",
        linewidth=1.2,
    )
    axes[2].set_ylabel("Intensity ratio")

    mode_order = [
        "CAPTURE_BRAKING",
        # Retained so the same analysis can still render historical sessions.
        "CAPTURE_ALIGN",
        "CAPTURE_SETTLE",
        "LOCAL_HOLD",
        "RETURN_TO_HOME",
        "NORMAL_HOLD",
    ]
    observed_modes = [
        str(value)
        for value in samples.get("control_mode", pd.Series(dtype=str)).dropna().unique()
    ]
    modes = mode_order + [value for value in observed_modes if value not in mode_order]
    mode_to_index = {mode: index for index, mode in enumerate(modes)}
    mode_values = samples["control_mode"].map(mode_to_index)
    axes[3].step(time_sec, mode_values, where="post", color="#4d4d4d")
    axes[3].set_yticks(range(len(modes)), modes, fontsize=7)
    axes[3].set_ylabel("Control mode")
    axes[3].set_xlabel("Time from release [s]")

    event_lines = [
        (trial.get("release_to_field_ms"), "Field", "#d73027"),
        (
            None
            if pd.isna(trial.get("braking_end_from_field_ms"))
            or pd.isna(trial.get("release_to_field_ms"))
            else float(trial["braking_end_from_field_ms"])
            + float(trial["release_to_field_ms"]),
            "Braking end",
            "#4575b4",
        ),
        (
            float(trial.get("success_hold_requirement_sec", 5.0)) * 1000.0,
            f"{float(trial.get('success_hold_requirement_sec', 5.0)):g} s success",
            "#1a9850",
        ),
    ]
    for event_ms, label, color in event_lines:
        if event_ms is None or pd.isna(event_ms):
            continue
        for axis in axes:
            axis.axvline(
                float(event_ms) / 1000.0,
                color=color,
                linestyle=":",
                linewidth=0.9,
            )
        axes[0].text(
            float(event_ms) / 1000.0,
            1.01,
            label,
            color=color,
            fontsize=7,
            rotation=90,
            va="bottom",
            ha="right",
            transform=axes[0].get_xaxis_transform(),
        )
    for axis in axes:
        axis.grid(True, alpha=0.22)
    figure.suptitle(f"Control mechanism and measured response: {trial_id}", fontsize=10)
    _save_figure(figure, output_path)


def plot_braking_validation(
    braking_summary: pd.DataFrame,
    output_directory: Path,
):
    output_path = output_directory / "braking_distance_validation"
    usable = braking_summary.dropna(
        subset=["plan_stopping_distance_mm", "actual_stopping_distance_mm"]
    )
    if usable.empty:
        _placeholder_figure(output_path, "No completed braking transitions are available")
        return
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    status_styles = (
        ("success", "Success", "#1a9850"),
        ("failure", "Failure", "#d73027"),
    )
    for status, label, color in status_styles:
        status_data = usable[usable["status"] == status]
        axes[0].scatter(
            status_data["plan_stopping_distance_mm"],
            status_data["actual_stopping_distance_mm"],
            color=color,
            alpha=0.8,
            label=label,
        )
    limits = [
        float(
            min(
                usable["plan_stopping_distance_mm"].min(),
                usable["actual_stopping_distance_mm"].min(),
            )
        ),
        float(
            max(
                usable["plan_stopping_distance_mm"].max(),
                usable["actual_stopping_distance_mm"].max(),
            )
        ),
    ]
    axes[0].plot(
        limits,
        limits,
        color="black",
        linestyle="--",
        linewidth=0.9,
    )
    axes[0].set_xlabel("Planned stopping distance [mm]", fontsize=16)
    axes[0].set_ylabel("Measured stopping\ndistance [mm]", fontsize=16)
    axes[0].tick_params(axis="both", labelsize=14)
    axes[0].legend(loc="upper left", fontsize=13)
    axes[0].grid(True, alpha=0.25)

    groups = [
        usable.loc[usable["status"] == status, "stopping_distance_error_mm"].dropna()
        for status in ("success", "failure")
    ]
    if any(len(group) for group in groups):
        axes[1].boxplot(groups, tick_labels=["Success", "Failure"], showfliers=False)
        for position, (values, color) in enumerate(
            zip(groups, ("#1a9850", "#d73027")), start=1
        ):
            axes[1].scatter(
                np.full(len(values), position),
                values,
                color=color,
                alpha=0.75,
            )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("Distance error [mm]\n(measured - planned)", fontsize=16)
    axes[1].tick_params(axis="both", labelsize=14)
    axes[1].grid(True, axis="y", alpha=0.25)
    _save_figure(figure, output_path)


def plot_behavior_patterns(
    pattern_summary: pd.DataFrame,
    output_directory: Path,
):
    output_path = output_directory / "behavior_pattern_counts"
    if pattern_summary.empty:
        _placeholder_figure(output_path, "No eligible trials are available")
        return
    ordered = pattern_summary.sort_values("count", ascending=True)
    figure, axis = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    colors = [
        "#1a9850" if label == "Held at least 5 s" else "#d73027"
        for label in ordered["behavior_pattern"]
    ]
    axis.barh(ordered["behavior_pattern"], ordered["count"], color=colors)
    axis.set_xlabel("Number of eligible trials")
    axis.set_title("Post-release behavior patterns")
    axis.grid(True, axis="x", alpha=0.25)
    for index, value in enumerate(ordered["count"]):
        axis.text(value, index, f" {int(value)}", va="center")
    _save_figure(figure, output_path)


def cliffs_delta(first: np.ndarray, second: np.ndarray):
    if len(first) == 0 or len(second) == 0:
        return np.nan
    differences = np.sign(first[:, np.newaxis] - second[np.newaxis, :])
    return float(differences.mean())


def _expit(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _fit_logistic_regression(
    design: np.ndarray,
    outcome: np.ndarray,
    *,
    maximum_iterations: int = 100,
    tolerance: float = 1e-10,
):
    """Fit an unpenalized logistic model by damped Newton iteration."""
    design = np.asarray(design, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    parameter_count = design.shape[1]
    coefficients = np.zeros(parameter_count, dtype=float)
    outcome_mean = np.clip(outcome.mean(), 1e-6, 1.0 - 1e-6)
    coefficients[0] = math.log(outcome_mean / (1.0 - outcome_mean))

    def log_likelihood(parameters):
        linear_predictor = design @ parameters
        return float(
            np.sum(
                outcome * linear_predictor
                - np.logaddexp(0.0, linear_predictor)
            )
        )

    current_log_likelihood = log_likelihood(coefficients)
    converged = False
    iteration = 0
    for iteration in range(1, maximum_iterations + 1):
        probability = _expit(design @ coefficients)
        weights = np.clip(probability * (1.0 - probability), 1e-12, None)
        information = design.T @ (weights[:, None] * design)
        score = design.T @ (outcome - probability)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            break

        damping = 1.0
        candidate = coefficients + step
        candidate_log_likelihood = log_likelihood(candidate)
        while (
            candidate_log_likelihood < current_log_likelihood
            and damping > 1e-6
        ):
            damping *= 0.5
            candidate = coefficients + damping * step
            candidate_log_likelihood = log_likelihood(candidate)
        coefficients = candidate
        current_log_likelihood = candidate_log_likelihood
        if np.max(np.abs(damping * step)) < tolerance:
            converged = True
            break

    probability = _expit(design @ coefficients)
    weights = np.clip(probability * (1.0 - probability), 1e-12, None)
    information = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        covariance = np.full((parameter_count, parameter_count), np.nan)
        converged = False
    return {
        "coefficients": coefficients,
        "covariance": covariance,
        "log_likelihood": current_log_likelihood,
        "aic": 2.0 * parameter_count - 2.0 * current_log_likelihood,
        "converged": converged,
        "iterations": iteration,
    }


def make_vxy_success_bins(trials: pd.DataFrame, bin_count: int = 5):
    selected = eligible_trials(trials).dropna(
        subset=["capture_state_vxy_mm_s", "release_xy_offset_mm"]
    ).copy()
    columns = [
        "bin",
        "vxy_min_mm_s",
        "vxy_max_mm_s",
        "vxy_mean_mm_s",
        "vxy_median_mm_s",
        "release_xy_offset_mean_mm",
        "release_xy_offset_median_mm",
        "trials",
        "successes",
        "failures",
        "success_rate",
        "success_ci95_low",
        "success_ci95_high",
        "interval_method",
    ]
    if selected.empty:
        return pd.DataFrame(columns=columns)
    usable_bin_count = min(
        int(bin_count),
        len(selected),
        int(selected["capture_state_vxy_mm_s"].nunique()),
    )
    if usable_bin_count <= 0:
        return pd.DataFrame(columns=columns)
    selected["vxy_bin"] = pd.qcut(
        selected["capture_state_vxy_mm_s"],
        q=usable_bin_count,
        labels=False,
        duplicates="drop",
    )
    rows = []
    for bin_index, group in selected.groupby("vxy_bin", sort=True):
        successes = int((group["status"] == "success").sum())
        total = int(len(group))
        lower, upper = wilson_interval(successes, total)
        rows.append(
            {
                "bin": int(bin_index) + 1,
                "vxy_min_mm_s": group["capture_state_vxy_mm_s"].min(),
                "vxy_max_mm_s": group["capture_state_vxy_mm_s"].max(),
                "vxy_mean_mm_s": group["capture_state_vxy_mm_s"].mean(),
                "vxy_median_mm_s": group["capture_state_vxy_mm_s"].median(),
                "release_xy_offset_mean_mm": group[
                    "release_xy_offset_mm"
                ].mean(),
                "release_xy_offset_median_mm": group[
                    "release_xy_offset_mm"
                ].median(),
                "trials": total,
                "successes": successes,
                "failures": total - successes,
                "success_rate": successes / total,
                "success_ci95_low": lower,
                "success_ci95_high": upper,
                "interval_method": "Wilson score, 95%",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def make_vxy_logistic_analysis(trials: pd.DataFrame):
    selected = eligible_trials(trials).dropna(
        subset=["capture_state_vxy_mm_s", "release_xy_offset_mm"]
    ).copy()
    outcome = (selected["status"] == "success").to_numpy(dtype=float)
    vxy = selected["capture_state_vxy_mm_s"].to_numpy(dtype=float)
    release_offset = selected["release_xy_offset_mm"].to_numpy(dtype=float)
    model_columns = [
        "model",
        "term",
        "odds_ratio_increment",
        "coefficient",
        "standard_error",
        "odds_ratio",
        "odds_ratio_ci95_low",
        "odds_ratio_ci95_high",
        "wald_p_value",
        "trials",
        "successes",
        "failures",
        "log_likelihood",
        "aic",
        "converged",
        "iterations",
    ]
    prediction_columns = [
        "vxy_mm_s",
        "release_xy_offset_reference_mm",
        "unadjusted_success_probability",
        "unadjusted_ci95_low",
        "unadjusted_ci95_high",
        "adjusted_success_probability",
        "adjusted_ci95_low",
        "adjusted_ci95_high",
        "interval_method",
    ]
    if (
        len(selected) < 10
        or len(np.unique(outcome)) < 2
        or min(int(outcome.sum()), int(len(outcome) - outcome.sum())) < 5
    ):
        return (
            pd.DataFrame(columns=model_columns),
            pd.DataFrame(columns=prediction_columns),
        )

    model_definitions = (
        (
            "unadjusted_vxy",
            ("intercept", "vxy_per_50_mm_s"),
            np.column_stack([np.ones(len(selected)), vxy / 50.0]),
            (np.nan, 50.0),
        ),
        (
            "adjusted_vxy_release_xy_offset",
            ("intercept", "vxy_per_50_mm_s", "release_xy_offset_per_10_mm"),
            np.column_stack(
                [np.ones(len(selected)), vxy / 50.0, release_offset / 10.0]
            ),
            (np.nan, 50.0, 10.0),
        ),
    )
    fitted = {}
    rows = []
    z_value = 1.959963984540054
    for model_name, terms, design, increments in model_definitions:
        result = _fit_logistic_regression(design, outcome)
        fitted[model_name] = result
        standard_errors = np.sqrt(np.diag(result["covariance"]))
        for term, increment, coefficient, standard_error in zip(
            terms,
            increments,
            result["coefficients"],
            standard_errors,
        ):
            wald_z = coefficient / standard_error
            rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "odds_ratio_increment": increment,
                    "coefficient": coefficient,
                    "standard_error": standard_error,
                    "odds_ratio": math.exp(coefficient),
                    "odds_ratio_ci95_low": math.exp(
                        coefficient - z_value * standard_error
                    ),
                    "odds_ratio_ci95_high": math.exp(
                        coefficient + z_value * standard_error
                    ),
                    "wald_p_value": math.erfc(abs(wald_z) / math.sqrt(2.0)),
                    "trials": len(selected),
                    "successes": int(outcome.sum()),
                    "failures": int(len(outcome) - outcome.sum()),
                    "log_likelihood": result["log_likelihood"],
                    "aic": result["aic"],
                    "converged": result["converged"],
                    "iterations": result["iterations"],
                }
            )

    grid = np.linspace(float(vxy.min()), float(vxy.max()), 200)
    offset_reference = float(np.median(release_offset))
    predictions = pd.DataFrame(
        {
            "vxy_mm_s": grid,
            "release_xy_offset_reference_mm": offset_reference,
        }
    )
    prediction_specs = (
        (
            "unadjusted_vxy",
            "unadjusted",
            np.column_stack([np.ones(len(grid)), grid / 50.0]),
        ),
        (
            "adjusted_vxy_release_xy_offset",
            "adjusted",
            np.column_stack(
                [
                    np.ones(len(grid)),
                    grid / 50.0,
                    np.full(len(grid), offset_reference / 10.0),
                ]
            ),
        ),
    )
    for model_name, prefix, prediction_design in prediction_specs:
        result = fitted[model_name]
        linear_predictor = prediction_design @ result["coefficients"]
        linear_variance = np.einsum(
            "ij,jk,ik->i",
            prediction_design,
            result["covariance"],
            prediction_design,
        )
        linear_standard_error = np.sqrt(np.maximum(linear_variance, 0.0))
        predictions[f"{prefix}_success_probability"] = _expit(
            linear_predictor
        )
        predictions[f"{prefix}_ci95_low"] = _expit(
            linear_predictor - z_value * linear_standard_error
        )
        predictions[f"{prefix}_ci95_high"] = _expit(
            linear_predictor + z_value * linear_standard_error
        )
    predictions["interval_method"] = "Wald interval on logit scale, 95%"
    return (
        pd.DataFrame(rows, columns=model_columns),
        predictions[prediction_columns],
    )


def plot_capture_success_vs_vxy(
    trials: pd.DataFrame,
    bins: pd.DataFrame,
    predictions: pd.DataFrame,
    output_directory: Path,
):
    output_path = output_directory / "capture_success_vs_vxy"
    selected = eligible_trials(trials).dropna(
        subset=["capture_state_vxy_mm_s", "release_xy_offset_mm"]
    )
    if selected.empty or bins.empty:
        _placeholder_figure(output_path, "No capture-onset horizontal speeds are available")
        return

    figure, axis = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
    x_values = bins["vxy_median_mm_s"].to_numpy(dtype=float)
    rates = bins["success_rate"].to_numpy(dtype=float)
    errors = np.vstack(
        [
            rates - bins["success_ci95_low"].to_numpy(dtype=float),
            bins["success_ci95_high"].to_numpy(dtype=float) - rates,
        ]
    )
    # Numerical round-off in a degenerate bin can put a bound a few ulps on
    # the wrong side of the point estimate; Matplotlib rejects negative yerr.
    errors = np.maximum(errors, 0.0)
    trials_per_bin = sorted(set(int(value) for value in bins["trials"]))
    if len(trials_per_bin) == 1:
        bin_size_label = f"n={trials_per_bin[0]}/bin; "
    else:
        bin_size_label = ""
    axis.errorbar(
        x_values,
        rates,
        yerr=errors,
        fmt="o",
        markersize=7.5,
        color="#2166ac",
        ecolor="#2166ac",
        elinewidth=1.5,
        capsize=4,
        label=f"Equal-count bins ({bin_size_label}Wilson 95% CI)",
        zorder=4,
    )

    if not predictions.empty:
        grid = predictions["vxy_mm_s"].to_numpy(dtype=float)
        adjusted = predictions["adjusted_success_probability"].to_numpy(
            dtype=float
        )
        adjusted_low = predictions["adjusted_ci95_low"].to_numpy(dtype=float)
        adjusted_high = predictions["adjusted_ci95_high"].to_numpy(dtype=float)
        axis.fill_between(
            grid,
            adjusted_low,
            adjusted_high,
            color="#d73027",
            alpha=0.16,
            linewidth=0.0,
            label="Adjusted model 95% CI",
        )
        axis.plot(
            grid,
            adjusted,
            color="#d73027",
            linewidth=2.1,
            label=(
                "Adjusted for release XY offset "
                f"(at median {predictions['release_xy_offset_reference_mm'].iloc[0]:.1f} mm)"
            ),
        )
        axis.plot(
            grid,
            predictions["unadjusted_success_probability"],
            color="#636363",
            linewidth=1.5,
            linestyle="--",
            label="Unadjusted model",
        )

    overall_rate = (selected["status"] == "success").mean()
    axis.axhline(
        overall_rate,
        color="#969696",
        linewidth=0.9,
        linestyle=":",
        label=f"Overall success rate ({overall_rate:.1%})",
    )
    axis.set_xlabel(r"Horizontal speed at capture onset, $v_{xy}$ [mm/s]", fontsize=13)
    axis.set_ylabel("Capture success probability", fontsize=13)
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, 1.0)
    axis.tick_params(axis="both", labelsize=11)
    axis.grid(True, alpha=0.22)
    axis.legend(loc="upper right", fontsize=9)
    _save_figure(figure, output_path)


def make_capture_success_vxy_outputs(
    trials: pd.DataFrame,
    output_directory: Path,
):
    bins = make_vxy_success_bins(trials)
    models, predictions = make_vxy_logistic_analysis(trials)
    selected = eligible_trials(trials).dropna(
        subset=["capture_state_vxy_mm_s", "release_xy_offset_mm"]
    )
    summary = {
        "analysis_trials": len(selected),
        "successes": int((selected["status"] == "success").sum()),
        "failures": int((selected["status"] == "failure").sum()),
        "overall_success_rate": (
            float((selected["status"] == "success").mean())
            if len(selected)
            else np.nan
        ),
        "vxy_min_mm_s": selected["capture_state_vxy_mm_s"].min(),
        "vxy_median_mm_s": selected["capture_state_vxy_mm_s"].median(),
        "vxy_max_mm_s": selected["capture_state_vxy_mm_s"].max(),
        "release_xy_offset_median_mm": selected[
            "release_xy_offset_mm"
        ].median(),
        "vxy_release_xy_offset_pearson_r": selected[
            ["capture_state_vxy_mm_s", "release_xy_offset_mm"]
        ].corr().iloc[0, 1],
        "binning_method": "Five equal-count quantile bins",
        "bin_interval_method": "Wilson score, 95%",
        "model_interval_method": "Wald interval on logit scale, 95%",
        "interpretation": (
            "Association analysis only; the fitted trend does not establish "
            "causality or a capture-speed threshold."
        ),
    }
    pd.DataFrame([summary]).to_csv(
        output_directory / "capture_success_vs_vxy_summary.csv",
        index=False,
    )
    bins.to_csv(
        output_directory / "capture_success_vs_vxy_bins.csv",
        index=False,
    )
    models.to_csv(
        output_directory / "capture_success_vs_vxy_logistic.csv",
        index=False,
    )
    predictions.to_csv(
        output_directory / "capture_success_vs_vxy_predictions.csv",
        index=False,
    )
    plot_capture_success_vs_vxy(
        trials,
        bins,
        predictions,
        output_directory,
    )


def plot_success_failure_effect_sizes(
    comparison: pd.DataFrame,
    output_directory: Path,
):
    output_path = output_directory / "success_failure_effect_sizes"
    if comparison.empty:
        _placeholder_figure(output_path, "No success/failure comparison is available")
        return

    labels = {
        "release_xy_offset_mm": "Release XY offset",
        "release_dz_mm": "Release Z offset",
        "release_to_field_ms": "Release-to-field delay",
        "capture_state_abs_vz_mm_s": "Onset $|v_z|$",
        "capture_state_vxy_mm_s": "Onset $v_{xy}$",
        "prediction_error_3d_mm": "3D prediction error",
        "stopping_distance_error_mm": "Stopping-distance error",
        "braking_end_vz_mm_s": "Braking-end $v_z$",
        "maximum_drop_from_onset_mm": "Maximum post-onset drop",
    }
    selected = comparison.dropna(
        subset=["cliffs_delta_success_minus_failure"]
    ).copy()
    selected["label"] = selected["metric"].map(labels).fillna(selected["metric"])
    selected = selected.iloc[::-1].reset_index(drop=True)
    delta = selected["cliffs_delta_success_minus_failure"].to_numpy(dtype=float)
    positions = np.arange(len(selected))

    figure, axis = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    axis.axvspan(-0.147, 0.147, color="#d9d9d9", alpha=0.45)
    axis.axvline(0.0, color="black", linewidth=0.9)
    axis.axvline(-0.147, color="#969696", linestyle=":", linewidth=0.9)
    axis.axvline(0.147, color="#969696", linestyle=":", linewidth=0.9)
    colors = ["#d73027" if value < 0.0 else "#1a9850" for value in delta]
    for position, value, color in zip(positions, delta, colors):
        axis.hlines(position, min(0.0, value), max(0.0, value), color=color, linewidth=2.5)
        axis.scatter([value], [position], color=color, s=70, zorder=3)
        offset = 0.016 if value >= 0.0 else -0.016
        axis.text(
            value + offset,
            position,
            f"{value:+.3f}",
            ha="left" if value >= 0.0 else "right",
            va="center",
            fontsize=12,
        )
    axis.set_yticks(positions, selected["label"], fontsize=14)
    axis.set_xlim(-0.36, 0.36)
    axis.set_xlabel(
        "Cliff's delta\n"
        r"$\leftarrow$ Larger in failures | Larger in successes $\rightarrow$",
        fontsize=15,
    )
    axis.tick_params(axis="x", labelsize=14)
    axis.grid(True, axis="x", alpha=0.2)
    _save_figure(figure, output_path)


def plot_success_failure_comparison(
    trials: pd.DataFrame,
    output_directory: Path,
):
    selected = eligible_trials(trials)

    metrics = [
        ("release_xy_offset_mm", "Release XY offset [mm]"),
        ("release_dz_mm", "Release Z offset [mm]"),
        ("release_to_field_ms", "Release-to-command [ms]"),
        ("capture_state_abs_vz_mm_s", "Estimated onset $|v_z|$ [mm/s]"),
        ("capture_state_vxy_mm_s", "Estimated onset $v_{xy}$ [mm/s]"),
        ("prediction_error_3d_mm", "Onset prediction error [mm]"),
        ("stopping_distance_error_mm", "Stopping-distance error [mm]"),
        ("braking_end_vz_mm_s", "Braking-end $v_z$ [mm/s]"),
        ("maximum_drop_from_onset_mm", "Maximum post-onset drop [mm]"),
    ]
    figure, axes = plt.subplots(3, 3, figsize=(11.2, 9.2), constrained_layout=True)
    rows = []
    random = np.random.default_rng(2026)
    for axis, (column, label) in zip(axes.flat, metrics):
        if column in selected:
            success_values = selected.loc[
                selected["status"] == "success", column
            ].dropna().to_numpy(dtype=float)
            failure_values = selected.loc[
                selected["status"] == "failure", column
            ].dropna().to_numpy(dtype=float)
        else:
            success_values = np.array([], dtype=float)
            failure_values = np.array([], dtype=float)
        groups = [success_values, failure_values]
        if any(len(group) for group in groups):
            axis.boxplot(groups, tick_labels=["Success", "Failure"], showfliers=False)
            for position, (values, color) in enumerate(
                zip(groups, ("#1a9850", "#d73027")), start=1
            ):
                if len(values):
                    jitter = random.normal(0.0, 0.035, size=len(values))
                    axis.scatter(
                        np.full(len(values), position) + jitter,
                        values,
                        color=color,
                        s=22,
                        alpha=0.75,
                        zorder=3,
                    )
        else:
            axis.text(
                0.5,
                0.5,
                "No usable values",
                ha="center",
                transform=axis.transAxes,
            )
        axis.set_ylabel(label)
        axis.grid(True, axis="y", alpha=0.25)
        rows.append(
            {
                "metric": column,
                "success_n": len(success_values),
                "success_median": np.median(success_values) if len(success_values) else np.nan,
                "success_q1": np.quantile(success_values, 0.25) if len(success_values) else np.nan,
                "success_q3": np.quantile(success_values, 0.75) if len(success_values) else np.nan,
                "failure_n": len(failure_values),
                "failure_median": np.median(failure_values) if len(failure_values) else np.nan,
                "failure_q1": np.quantile(failure_values, 0.25) if len(failure_values) else np.nan,
                "failure_q3": np.quantile(failure_values, 0.75) if len(failure_values) else np.nan,
                "cliffs_delta_success_minus_failure": cliffs_delta(
                    success_values,
                    failure_values,
                ),
            }
        )
    figure.suptitle(
        "Measured factors for successful and failed captures",
        fontsize=10,
    )
    _save_figure(figure, output_directory / "success_failure_comparison")
    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        output_directory / "success_failure_comparison.csv",
        index=False,
    )
    plot_success_failure_effect_sizes(comparison, output_directory)


def plot_failure_reasons(
    failure_summary: pd.DataFrame,
    output_directory: Path,
):
    output_path = output_directory / "failure_reason_counts"
    if failure_summary.empty:
        _placeholder_figure(output_path, "No failed trials are available")
        return
    category_counts = (
        failure_summary.groupby("category")["count"]
        .sum()
        .sort_values(ascending=True)
    )
    figure, axis = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    axis.barh(category_counts.index, category_counts.values, color="#d73027")
    axis.set_xlabel("Number of failed trials")
    axis.set_title("Recorded termination reasons before the 5 s criterion")
    axis.grid(True, axis="x", alpha=0.25)
    for index, value in enumerate(category_counts.values):
        axis.text(value, index, f" {int(value)}", va="center")
    _save_figure(figure, output_path)


def _save_figure(figure, path_without_suffix: Path):
    figure.savefig(path_without_suffix.with_suffix(".png"), dpi=300)
    figure.savefig(path_without_suffix.with_suffix(".svg"))
    plt.close(figure)


def build_all_outputs(
    data_directory: Path,
    output_directory: Path,
    *,
    representative_trial_id: str | None = None,
    center_x_mm: float | None = None,
    center_y_mm: float | None = None,
    center_z_mm: float | None = None,
    exclusions_file: Path | None = None,
    target_valid_trials: int = 100,
):
    output_directory.mkdir(parents=True, exist_ok=True)
    summary, timeseries = load_trial_data(data_directory)
    enriched = enrich_onset_metrics(summary, timeseries)
    exclusion_path = exclusions_file or (data_directory / "trial_exclusions.csv")
    exclusions = load_trial_exclusions(exclusion_path)
    enriched = apply_analysis_exclusions(enriched, exclusions)
    center = resolve_release_center(
        enriched,
        center_x_mm=center_x_mm,
        center_y_mm=center_y_mm,
        center_z_mm=center_z_mm,
    )
    enriched = add_centered_release_metrics(enriched, center)
    outcome_summary = make_outcome_summary(
        enriched,
        center,
        target_valid_trials=target_valid_trials,
    )
    failure_summary = make_failure_reason_summary(enriched)
    release_range_summary = make_release_range_summary(enriched)
    behavior_summary = make_behavior_pattern_summary(enriched)
    braking_summary = make_braking_validation_summary(enriched)
    failed_metric_summary = make_failed_trial_metric_summary(enriched)
    exclusion_summary = enriched.loc[
        enriched["analysis_excluded"],
        [
            "trial_id",
            "condition_id",
            "status",
            "reason",
            "analysis_exclusion_source",
            "analysis_exclusion_reason",
        ],
    ]
    enriched.to_csv(output_directory / "trial_metrics_enriched.csv", index=False)
    outcome_summary.to_csv(output_directory / "outcome_summary.csv", index=False)
    failure_summary.to_csv(
        output_directory / "failure_reason_summary.csv",
        index=False,
    )
    release_range_summary.to_csv(
        output_directory / "release_range_summary.csv",
        index=False,
    )
    behavior_summary.to_csv(
        output_directory / "behavior_pattern_summary.csv",
        index=False,
    )
    braking_summary.to_csv(
        output_directory / "braking_validation.csv",
        index=False,
    )
    failed_metric_summary.to_csv(
        output_directory / "failed_trial_metric_summary.csv",
        index=False,
    )
    exclusion_summary.to_csv(
        output_directory / "exclusion_summary.csv",
        index=False,
    )

    successful = eligible_trials(enriched)
    successful = successful[successful["status"] == "success"]
    if representative_trial_id is not None:
        match = successful[successful["trial_id"] == representative_trial_id]
        if match.empty:
            raise ValueError(
                "--trial-id must identify a successful 5 s hold trial"
            )
        representative = representative_trial_id
    elif successful.empty:
        representative = None
    else:
        representative = select_representative_trial(
            successful,
            None,
        )
    plot_successful_trajectory(
        enriched,
        timeseries,
        output_directory,
        representative,
    )
    plot_successful_trials_mean_response(
        enriched,
        timeseries,
        output_directory,
    )
    plot_successful_z_trajectories(
        enriched,
        timeseries,
        output_directory,
    )
    plot_failed_trials_mean_response(
        enriched,
        timeseries,
        output_directory,
    )
    plot_control_mechanism_validation(
        enriched,
        timeseries,
        output_directory,
        representative,
    )
    plot_outcome_trajectory_comparison(
        enriched,
        timeseries,
        output_directory,
    )
    plot_release_outcomes_3d(enriched, output_directory, center)
    make_capture_success_vxy_outputs(enriched, output_directory)
    plot_success_failure_comparison(enriched, output_directory)
    plot_failure_reasons(failure_summary, output_directory)
    plot_braking_validation(braking_summary, output_directory)
    plot_behavior_patterns(behavior_summary, output_directory)
    return representative, enriched, outcome_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate center-region SI2026 capture figures from trial logs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./manual_release_tracking/si2026_data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./manual_release_tracking/si2026_figures"),
    )
    parser.add_argument("--trial-id", default=None)
    parser.add_argument(
        "--center-x-mm",
        type=float,
        default=None,
        help="Nominal AUTD-center X; default is the logged AUTD center.",
    )
    parser.add_argument(
        "--center-y-mm",
        type=float,
        default=None,
        help="Nominal AUTD-center Y; default is the logged AUTD center.",
    )
    parser.add_argument(
        "--center-z-mm",
        type=float,
        default=None,
        help="Nominal AUTD-center Z; default is the logged AUTD center.",
    )
    parser.add_argument(
        "--exclusions-file",
        type=Path,
        default=None,
        help=(
            "CSV with trial_id,reason columns. Default: "
            "<data-dir>/trial_exclusions.csv when present."
        ),
    )
    parser.add_argument(
        "--target-valid-trials",
        type=int,
        default=100,
        help="Target number of non-excluded real releases (default: 100).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    representative_id, trials, result_summary = build_all_outputs(
        arguments.data_dir,
        arguments.output_dir,
        representative_trial_id=arguments.trial_id,
        center_x_mm=arguments.center_x_mm,
        center_y_mm=arguments.center_y_mm,
        center_z_mm=arguments.center_z_mm,
        exclusions_file=arguments.exclusions_file,
        target_valid_trials=arguments.target_valid_trials,
    )
    print(f"Generated figures in: {arguments.output_dir}")
    print(f"Representative trial: {representative_id}")
    print(f"Trials: {len(trials)}")
    print(result_summary.to_string(index=False))
