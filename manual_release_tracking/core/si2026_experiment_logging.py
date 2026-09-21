from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 4


TIMESERIES_HEADER = [
    "schema_version",
    "session_id",
    "trial_id",
    "condition_id",
    "sample_time_sec",
    "measurement_time_sec",
    "elapsed_from_release_ms",
    "elapsed_from_field_ms",
    "frame_xy_time_sec",
    "frame_z_time_sec",
    "frame_sync_skew_ms",
    "detected_xy",
    "detected_z",
    "raw_x_mm",
    "raw_y_mm",
    "raw_z_mm",
    "filtered_x_mm",
    "filtered_y_mm",
    "filtered_z_mm",
    "vx_mm_s",
    "vy_mm_s",
    "vz_mm_s",
    "velocity_ready",
    "release_gate_origin_z_mm",
    "release_gate_downward_displacement_mm",
    "release_gate_vz_threshold_mm_s",
    "release_gate_drop_threshold_mm",
    "release_gate_consecutive_count",
    "release_gate_consecutive_required",
    "release_gate_passed",
    "success_criterion_reached",
    "control_mode",
    "mode_transition_reason",
    "braking_elapsed_ms",
    "reference_z_mm",
    "reference_vz_mm_s",
    "reference_acceleration_mm_s2",
    "z_reference_error_mm",
    "vz_reference_error_mm_s",
    "anchor_x_mm",
    "anchor_y_mm",
    "setpoint_x_mm",
    "setpoint_y_mm",
    "setpoint_z_mm",
    "target_x_mm",
    "target_y_mm",
    "target_z_mm",
    "commanded_intensity",
    "xy_distance_to_target_mm",
    "required_force_mN",
    "force_saturated",
    "plan_feasible",
    "plan_workspace_limited",
    "plan_force_saturated",
    "plan_duration_ms",
    "plan_stop_z_mm",
    "plan_stopping_distance_mm",
]


SUMMARY_HEADER = [
    "schema_version",
    "session_id",
    "trial_id",
    "condition_id",
    "trial_index",
    "operator",
    "planned_release_x_mm",
    "planned_release_y_mm",
    "planned_release_z_mm",
    "autd_center_x_mm",
    "autd_center_y_mm",
    "autd_center_z_mm",
    "extra_field_delay_ms",
    "ball_mass_kg",
    "ball_diameter_mm",
    "release_timestamp_ms",
    "release_measurement_time_sec",
    "release_x_mm",
    "release_y_mm",
    "release_z_mm",
    "release_gate_vz_threshold_mm_s",
    "release_gate_drop_threshold_mm",
    "release_gate_consecutive_required",
    "release_gate_origin_z_mm",
    "release_gate_minimum_vz_mm_s",
    "release_gate_maximum_drop_mm",
    "release_gate_evidence_seen",
    "release_gate_passed_time_sec",
    "release_gate_elapsed_ms",
    "release_gate_pass_vz_mm_s",
    "release_gate_pass_drop_mm",
    "release_gate_pass_consecutive_count",
    "motion_ready_time_sec",
    "field_time_sec",
    "release_to_field_ms",
    "field_prediction_x_mm",
    "field_prediction_y_mm",
    "field_prediction_z_mm",
    "field_prediction_vz_mm_s",
    "onset_measurement_time_sec",
    "onset_measurement_offset_ms",
    "onset_x_mm",
    "onset_y_mm",
    "onset_z_mm",
    "onset_vx_mm_s",
    "onset_vy_mm_s",
    "onset_vz_mm_s",
    "onset_rxy_from_anchor_mm",
    "prediction_error_x_mm",
    "prediction_error_y_mm",
    "prediction_error_z_mm",
    "prediction_error_3d_mm",
    "anchor_x_mm",
    "anchor_y_mm",
    "plan_duration_ms",
    "plan_stop_z_mm",
    "plan_stopping_distance_mm",
    "plan_required_force_mN",
    "plan_maximum_force_mN",
    "plan_feasible",
    "plan_force_saturated",
    "plan_workspace_limited",
    "braking_end_time_sec",
    "braking_end_from_field_ms",
    "braking_end_reason",
    "braking_end_x_mm",
    "braking_end_y_mm",
    "braking_end_z_mm",
    "braking_end_vx_mm_s",
    "braking_end_vy_mm_s",
    "braking_end_vz_mm_s",
    "actual_stopping_distance_mm",
    "stopping_distance_error_mm",
    "plan_stop_z_error_mm",
    "local_hold_reached_time_sec",
    "capture_time_ms",
    "hold_duration_ms",
    "hold_sample_count",
    "hold_rms_xy_mm",
    "hold_rms_z_mm",
    "minimum_z_mm",
    "maximum_drop_from_onset_mm",
    "maximum_xy_distance_to_target_mm",
    "maximum_commanded_intensity",
    "sample_count",
    "success_hold_requirement_sec",
    "success_reached_time_sec",
    "success_reached_elapsed_ms",
    "observation_requirement_sec",
    "post_success_observation_ms",
    "retained_duration_ms",
    "status",
    "success",
    "reason",
    "notes",
]


@dataclass(frozen=True)
class SI2026Condition:
    condition_id: str = "baseline"
    operator: str = ""
    planned_release_x_mm: float | None = None
    planned_release_y_mm: float | None = None
    planned_release_z_mm: float | None = None
    extra_field_delay_ms: float = 0.0
    notes: str = ""


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _csv_number(value: Any, digits: int = 6):
    number = _finite_number(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _safe_json_value(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    return repr(value)


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            check=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def release_hold_success_reached(
    *,
    elapsed_from_release_sec: float | None,
    required_hold_sec: float,
    field_started: bool,
    automatic_hold_active: bool,
) -> bool:
    return bool(
        elapsed_from_release_sec is not None
        and field_started
        and automatic_hold_active
        and float(elapsed_from_release_sec) >= float(required_hold_sec)
    )


class SI2026ExperimentLogger:
    """Append-only, trial-oriented logger for the SI2026 experiments.

    The logger observes the existing controller. It never sends a command or
    changes a target. A trial starts at the first explicit release candidate
    and ends with one summary row, even when it is invalid or interrupted.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        config: Any,
        condition: SI2026Condition | None = None,
        project_root: str | os.PathLike[str] | None = None,
        enabled: bool = True,
        session_id: str | None = None,
    ):
        self.enabled = bool(enabled)
        self.directory = Path(directory)
        self.config = config
        self.condition = condition or SI2026Condition()
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.session_id = session_id or (
            f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        )
        self._trial_counter = 0
        self._timeseries_file = None
        self._timeseries_writer = None
        self._active: dict[str, Any] | None = None
        self._last_completed_trial: dict[str, Any] | None = None
        self._reference_center: tuple[float, float, float] | None = None

        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        self._open_timeseries()
        self._write_session_metadata()

    @property
    def active(self) -> bool:
        return self._active is not None

    @property
    def field_started(self) -> bool:
        return self.active and self._active.get("field_time_sec") is not None

    @property
    def success_reached(self) -> bool:
        return bool(
            self.active
            and self._active.get("success_reached_time_sec") is not None
        )

    @property
    def trial_id(self) -> str | None:
        return None if not self.active else str(self._active["trial_id"])

    def _ensure_header(self, path: Path, header: list[str]):
        if path.exists() and path.stat().st_size > 0:
            with path.open(newline="", encoding="utf-8") as file:
                existing = next(csv.reader(file), [])
            if existing != header:
                raise RuntimeError(
                    f"CSV schema mismatch: {path}. Move the old file or use "
                    "a new --paper-data-dir before recording more trials."
                )

    def _open_timeseries(self):
        path = self.directory / "trial_timeseries.csv"
        self._ensure_header(path, TIMESERIES_HEADER)
        exists = path.exists() and path.stat().st_size > 0
        self._timeseries_file = path.open("a", newline="", encoding="utf-8")
        self._timeseries_writer = csv.writer(self._timeseries_file)
        if not exists:
            self._timeseries_writer.writerow(TIMESERIES_HEADER)
            self._timeseries_file.flush()

    def _config_dict(self) -> dict[str, Any]:
        if is_dataclass(self.config):
            result = asdict(self.config)
            # Dynamically added experiment options are not returned by asdict.
            for key, value in vars(self.config).items():
                result.setdefault(key, value)
            return result
        return dict(vars(self.config))

    def _write_session_metadata(self):
        calibration_paths = {}
        for name in (
            "intrinsic_xy_npz",
            "intrinsic_z_npz",
            "stereo_npz",
            "stereo_camera_to_autd_npz",
        ):
            configured = getattr(self.config, name, "")
            if not configured:
                continue
            path = Path(configured)
            if not path.is_absolute():
                path = self.project_root / path
            calibration_paths[name] = {
                "path": str(path.resolve()),
                "sha256": _file_sha256(path),
            }

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_wall_time": time.time(),
            "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_commit": _git_commit(self.project_root),
            "condition": _safe_json_value(asdict(self.condition)),
            "config": _safe_json_value(self._config_dict()),
            "calibration_files": calibration_paths,
        }
        path = self.directory / f"session_{self.session_id}.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2, sort_keys=True)

    def start_trial(self, *, release_timestamp_ms: int, release_estimate=None):
        if not self.enabled:
            return None
        if self.active:
            self.finish_trial(
                status=(
                    "success"
                    if self.success_reached
                    else "failure"
                    if self.field_started
                    else "aborted"
                ),
                reason="new_release_candidate_before_previous_trial_finished",
            )

        self._trial_counter += 1
        trial_id = (
            f"{self.session_id}_T{self._trial_counter:04d}_"
            f"{int(release_timestamp_ms)}"
        )
        release_time_sec = float(release_timestamp_ms) / 1000.0
        self._active = {
            "trial_id": trial_id,
            "trial_index": self._trial_counter,
            "release_timestamp_ms": int(release_timestamp_ms),
            "release_time_sec": release_time_sec,
            "release_state": (
                None
                if release_estimate is None
                else {
                    "measurement_time_sec": _finite_number(
                        release_estimate.measurement_time_sec
                    ),
                    "x_mm": _finite_number(release_estimate.x_mm),
                    "y_mm": _finite_number(release_estimate.y_mm),
                    "z_mm": _finite_number(release_estimate.z_mm),
                }
            ),
            "release_gate_origin_z_mm": None,
            "release_gate_minimum_vz_mm_s": None,
            "release_gate_maximum_drop_mm": None,
            "release_gate_evidence_seen": False,
            "release_gate_passed_time_sec": None,
            "release_gate_pass_vz_mm_s": None,
            "release_gate_pass_drop_mm": None,
            "release_gate_pass_consecutive_count": None,
            "motion_ready_time_sec": None,
            "field_time_sec": None,
            "field_prediction": None,
            "anchor": None,
            "plan": None,
            "onset": None,
            "braking_end": None,
            "local_hold_reached_time_sec": None,
            "success_reached_time_sec": None,
            "sample_count": 0,
            "minimum_z_mm": None,
            "maximum_xy_distance_to_target_mm": None,
            "maximum_commanded_intensity": None,
            "hold_sample_count": 0,
            "hold_xy_error_sq_sum": 0.0,
            "hold_z_error_sq_sum": 0.0,
        }
        return trial_id

    def set_reference_center(self, x_mm: float, y_mm: float, z_mm: float):
        self._reference_center = (
            float(x_mm),
            float(y_mm),
            float(z_mm),
        )

    def elapsed_from_release_sec(self, now_sec: float | None = None):
        if not self.active:
            return None
        now = time.perf_counter() if now_sec is None else float(now_sec)
        return max(0.0, now - float(self._active["release_time_sec"]))

    def mark_motion_ready(self, event_time_sec: float):
        if self.active and self._active["motion_ready_time_sec"] is None:
            self._active["motion_ready_time_sec"] = float(event_time_sec)

    def update_release_gate_observation(
        self,
        *,
        origin_z_mm: float | None,
        estimate,
        downward_displacement_mm: float | None,
        evidence_seen: bool,
    ):
        if not self.active:
            return
        active = self._active
        origin = _finite_number(origin_z_mm)
        if origin is not None and active["release_gate_origin_z_mm"] is None:
            active["release_gate_origin_z_mm"] = origin
        if estimate is not None:
            vz = _finite_number(estimate.vz_mm_s)
            if vz is not None:
                previous = active["release_gate_minimum_vz_mm_s"]
                active["release_gate_minimum_vz_mm_s"] = (
                    vz if previous is None else min(previous, vz)
                )
        drop = _finite_number(downward_displacement_mm)
        if drop is not None:
            previous = active["release_gate_maximum_drop_mm"]
            active["release_gate_maximum_drop_mm"] = (
                drop if previous is None else max(previous, drop)
            )
        active["release_gate_evidence_seen"] = bool(
            active["release_gate_evidence_seen"] or evidence_seen
        )

    def mark_release_gate_passed(
        self,
        *,
        event_time_sec: float,
        estimate,
        downward_displacement_mm: float,
        consecutive_count: int,
    ):
        if not self.active or self._active["release_gate_passed_time_sec"] is not None:
            return
        self._active["release_gate_passed_time_sec"] = float(event_time_sec)
        self._active["release_gate_pass_vz_mm_s"] = _finite_number(
            None if estimate is None else estimate.vz_mm_s
        )
        self._active["release_gate_pass_drop_mm"] = _finite_number(
            downward_displacement_mm
        )
        self._active["release_gate_pass_consecutive_count"] = int(
            consecutive_count
        )

    def mark_success_reached(self, event_time_sec: float):
        if self.active and self._active["success_reached_time_sec"] is None:
            self._active["success_reached_time_sec"] = float(event_time_sec)

    def mark_braking_end(self, event_time_sec: float, estimate, reason: str):
        if not self.active or self._active["braking_end"] is not None:
            return
        self._active["braking_end"] = {
            "event_time_sec": float(event_time_sec),
            "reason": str(reason),
            "x_mm": _finite_number(None if estimate is None else estimate.x_mm),
            "y_mm": _finite_number(None if estimate is None else estimate.y_mm),
            "z_mm": _finite_number(None if estimate is None else estimate.z_mm),
            "vx_mm_s": _finite_number(None if estimate is None else estimate.vx_mm_s),
            "vy_mm_s": _finite_number(None if estimate is None else estimate.vy_mm_s),
            "vz_mm_s": _finite_number(None if estimate is None else estimate.vz_mm_s),
        }

    def mark_field_sent(
        self,
        *,
        field_time_sec: float,
        prediction,
        anchor: tuple[float, float] | None,
        plan,
    ):
        if not self.active:
            return
        self._active["field_time_sec"] = float(field_time_sec)
        self._active["field_prediction"] = prediction
        self._active["anchor"] = anchor
        self._active["plan"] = plan

    def mark_local_hold_reached(self, event_time_sec: float):
        if self.active and self._active["local_hold_reached_time_sec"] is None:
            self._active["local_hold_reached_time_sec"] = float(event_time_sec)

    def exclude_recent_failed_trial(
        self,
        *,
        reason: str,
        now_sec: float | None = None,
        maximum_age_sec: float = 5.0,
    ) -> str | None:
        """Register a just-finished false trigger without rewriting raw logs."""
        if not self.enabled or self._last_completed_trial is None:
            return None
        now = time.perf_counter() if now_sec is None else float(now_sec)
        completed = self._last_completed_trial
        if completed["status"] != "failure":
            return None
        if now - float(completed["event_time_sec"]) > max(0.0, maximum_age_sec):
            return None

        path = self.directory / "trial_exclusions.csv"
        rows: list[dict[str, str]] = []
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as file:
                rows = [
                    {
                        "trial_id": str(row.get("trial_id", "")),
                        "reason": str(row.get("reason", "")),
                    }
                    for row in csv.DictReader(file)
                    if row.get("trial_id")
                ]
        trial_id = str(completed["trial_id"])
        rows = [row for row in rows if row["trial_id"] != trial_id]
        rows.append({"trial_id": trial_id, "reason": str(reason)})
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["trial_id", "reason"])
            writer.writeheader()
            writer.writerows(rows)
        return trial_id

    def record_sample(
        self,
        *,
        sample_time_sec: float,
        frame_xy_time_sec: float,
        frame_z_time_sec: float,
        frame_sync_skew_sec: float,
        detected_xy: bool,
        detected_z: bool,
        raw_x_mm: float | None,
        raw_y_mm: float | None,
        raw_z_mm: float | None,
        estimate,
        control_mode: str,
        reference,
        anchor: tuple[float, float] | None,
        setpoint,
        target,
        commanded_intensity: float | None,
        xy_distance_to_target_mm: float | None,
        required_force_mN: float | None,
        force_saturated: bool,
        plan,
        release_gate_origin_z_mm: float | None = None,
        release_gate_downward_displacement_mm: float | None = None,
        release_gate_consecutive_count: int = 0,
        release_gate_passed: bool = False,
        mode_transition_reason: str = "",
    ):
        if not self.active or self._timeseries_writer is None:
            return

        active = self._active
        measurement_time = (
            None if estimate is None else _finite_number(estimate.measurement_time_sec)
        )
        field_time = active["field_time_sec"]
        elapsed_field_ms = (
            None
            if field_time is None
            else (float(sample_time_sec) - float(field_time)) * 1000.0
        )
        elapsed_release_ms = (
            float(sample_time_sec) - float(active["release_time_sec"])
        ) * 1000.0

        if active["release_state"] is None and estimate is not None:
            active["release_state"] = {
                "measurement_time_sec": measurement_time,
                "x_mm": _finite_number(estimate.x_mm),
                "y_mm": _finite_number(estimate.y_mm),
                "z_mm": _finite_number(estimate.z_mm),
            }

        filtered_z = None if estimate is None else _finite_number(estimate.z_mm)
        if filtered_z is not None:
            previous_min = active["minimum_z_mm"]
            active["minimum_z_mm"] = (
                filtered_z if previous_min is None else min(previous_min, filtered_z)
            )
        xy_distance = _finite_number(xy_distance_to_target_mm)
        if xy_distance is not None:
            previous_max = active["maximum_xy_distance_to_target_mm"]
            active["maximum_xy_distance_to_target_mm"] = (
                xy_distance if previous_max is None else max(previous_max, xy_distance)
            )
        intensity = _finite_number(commanded_intensity)
        if intensity is not None:
            previous_max = active["maximum_commanded_intensity"]
            active["maximum_commanded_intensity"] = (
                intensity if previous_max is None else max(previous_max, intensity)
            )

        reference_z = _finite_number(
            None if reference is None else reference.z_mm
        )
        reference_vz = _finite_number(
            None if reference is None else reference.vz_mm_s
        )
        reference_acceleration = _finite_number(
            None
            if reference is None
            else getattr(reference, "acceleration_mm_s2", None)
        )
        estimate_z = _finite_number(None if estimate is None else estimate.z_mm)
        estimate_vz = _finite_number(None if estimate is None else estimate.vz_mm_s)
        z_reference_error = (
            None
            if reference_z is None or estimate_z is None
            else reference_z - estimate_z
        )
        vz_reference_error = (
            None
            if reference_vz is None or estimate_vz is None
            else reference_vz - estimate_vz
        )

        if (
            active["onset"] is None
            and field_time is not None
            and measurement_time is not None
            and measurement_time >= float(field_time)
            and estimate is not None
        ):
            active["onset"] = {
                "measurement_time_sec": measurement_time,
                "x_mm": _finite_number(estimate.x_mm),
                "y_mm": _finite_number(estimate.y_mm),
                "z_mm": _finite_number(estimate.z_mm),
                "vx_mm_s": _finite_number(estimate.vx_mm_s),
                "vy_mm_s": _finite_number(estimate.vy_mm_s),
                "vz_mm_s": _finite_number(estimate.vz_mm_s),
            }

        if control_mode == "LOCAL_HOLD" and estimate is not None and setpoint is not None:
            ex = _finite_number(estimate.x_mm - setpoint.x)
            ey = _finite_number(estimate.y_mm - setpoint.y)
            ez = _finite_number(estimate.z_mm - setpoint.z)
            if ex is not None and ey is not None and ez is not None:
                active["hold_sample_count"] += 1
                active["hold_xy_error_sq_sum"] += ex * ex + ey * ey
                active["hold_z_error_sq_sum"] += ez * ez

        row = [
            SCHEMA_VERSION,
            self.session_id,
            active["trial_id"],
            self.condition.condition_id,
            _csv_number(sample_time_sec),
            _csv_number(measurement_time),
            _csv_number(elapsed_release_ms, 3),
            _csv_number(elapsed_field_ms, 3),
            _csv_number(frame_xy_time_sec),
            _csv_number(frame_z_time_sec),
            _csv_number(float(frame_sync_skew_sec) * 1000.0, 3),
            int(bool(detected_xy)),
            int(bool(detected_z)),
            _csv_number(raw_x_mm, 3),
            _csv_number(raw_y_mm, 3),
            _csv_number(raw_z_mm, 3),
            _csv_number(None if estimate is None else estimate.x_mm, 3),
            _csv_number(None if estimate is None else estimate.y_mm, 3),
            _csv_number(None if estimate is None else estimate.z_mm, 3),
            _csv_number(None if estimate is None else estimate.vx_mm_s, 3),
            _csv_number(None if estimate is None else estimate.vy_mm_s, 3),
            _csv_number(None if estimate is None else estimate.vz_mm_s, 3),
            int(bool(estimate is not None and estimate.velocity_ready)),
            _csv_number(release_gate_origin_z_mm, 3),
            _csv_number(release_gate_downward_displacement_mm, 3),
            _csv_number(
                getattr(
                    self.config,
                    "auto_release_gate_max_vz_mm_s",
                    -100.0,
                ),
                3,
            ),
            _csv_number(
                getattr(
                    self.config,
                    "auto_release_gate_min_drop_mm",
                    2.5,
                ),
                3,
            ),
            int(release_gate_consecutive_count),
            int(
                getattr(
                    self.config,
                    "auto_release_gate_consecutive_estimates",
                    3,
                )
            ),
            int(bool(release_gate_passed)),
            int(self.success_reached),
            str(control_mode),
            str(mode_transition_reason),
            _csv_number(
                None
                if reference is None
                else getattr(reference, "elapsed_sec", None) * 1000.0
                if getattr(reference, "elapsed_sec", None) is not None
                else None,
                3,
            ),
            _csv_number(reference_z, 3),
            _csv_number(reference_vz, 3),
            _csv_number(reference_acceleration, 3),
            _csv_number(z_reference_error, 3),
            _csv_number(vz_reference_error, 3),
            _csv_number(None if anchor is None else anchor[0], 3),
            _csv_number(None if anchor is None else anchor[1], 3),
            _csv_number(None if setpoint is None else setpoint.x, 3),
            _csv_number(None if setpoint is None else setpoint.y, 3),
            _csv_number(None if setpoint is None else setpoint.z, 3),
            _csv_number(None if target is None else target.x, 3),
            _csv_number(None if target is None else target.y, 3),
            _csv_number(None if target is None else target.z, 3),
            _csv_number(intensity, 4),
            _csv_number(xy_distance, 3),
            _csv_number(required_force_mN, 3),
            int(bool(force_saturated)),
            "" if plan is None else int(bool(plan.feasible)),
            "" if plan is None else int(bool(plan.workspace_limited)),
            "" if plan is None else int(bool(plan.force_saturated)),
            _csv_number(None if plan is None else plan.duration_sec * 1000.0, 3),
            _csv_number(None if plan is None else plan.stop_z_mm, 3),
            _csv_number(
                None if plan is None else plan.stopping_distance_mm,
                3,
            ),
        ]
        self._timeseries_writer.writerow(row)
        active["sample_count"] += 1

    def finish_trial(
        self,
        *,
        status: str,
        reason: str,
        event_time_sec: float | None = None,
    ):
        if not self.active:
            return
        status = str(status).lower()
        if status not in {"success", "failure", "invalid", "aborted"}:
            raise ValueError(f"Unknown trial status: {status}")
        now_sec = time.perf_counter() if event_time_sec is None else float(event_time_sec)
        active = self._active
        condition = self.condition
        reference_center = self._reference_center
        release_state = active["release_state"]
        field_time = active["field_time_sec"]
        prediction = active["field_prediction"]
        onset = active["onset"]
        anchor = active["anchor"]
        plan = active["plan"]
        braking_end = active["braking_end"]
        local_hold_time = active["local_hold_reached_time_sec"]
        success_reached_time = active["success_reached_time_sec"]

        def onset_value(name: str):
            return None if onset is None else onset.get(name)

        def release_value(name: str):
            return None if release_state is None else release_state.get(name)

        def braking_end_value(name: str):
            return None if braking_end is None else braking_end.get(name)

        prediction_error = [None, None, None]
        if prediction is not None and onset is not None:
            prediction_error = [
                onset_value("x_mm") - float(prediction.x_mm),
                onset_value("y_mm") - float(prediction.y_mm),
                onset_value("z_mm") - float(prediction.z_mm),
            ]
        finite_prediction_errors = [
            value for value in prediction_error if _finite_number(value) is not None
        ]
        prediction_error_3d = (
            float(np.linalg.norm(prediction_error))
            if len(finite_prediction_errors) == 3
            else None
        )
        onset_rxy = None
        if onset is not None and anchor is not None:
            onset_rxy = float(
                np.hypot(
                    onset_value("x_mm") - float(anchor[0]),
                    onset_value("y_mm") - float(anchor[1]),
                )
            )
        hold_count = int(active["hold_sample_count"])
        hold_rms_xy = (
            math.sqrt(active["hold_xy_error_sq_sum"] / hold_count)
            if hold_count > 0
            else None
        )
        hold_rms_z = (
            math.sqrt(active["hold_z_error_sq_sum"] / hold_count)
            if hold_count > 0
            else None
        )
        minimum_z = active["minimum_z_mm"]
        maximum_drop = (
            onset_value("z_mm") - minimum_z
            if onset is not None and minimum_z is not None
            else None
        )
        actual_stopping_distance = (
            onset_value("z_mm") - braking_end_value("z_mm")
            if onset is not None
            and braking_end is not None
            and onset_value("z_mm") is not None
            and braking_end_value("z_mm") is not None
            else None
        )
        stopping_distance_error = (
            actual_stopping_distance - float(plan.stopping_distance_mm)
            if actual_stopping_distance is not None and plan is not None
            else None
        )
        plan_stop_z_error = (
            braking_end_value("z_mm") - float(plan.stop_z_mm)
            if braking_end_value("z_mm") is not None and plan is not None
            else None
        )

        # Flush once per completed trial, not once per camera frame, so paper
        # logging does not add a disk-sync operation to the control loop.
        if self._timeseries_file is not None:
            self._timeseries_file.flush()

        summary_path = self.directory / "trial_summary.csv"
        self._ensure_header(summary_path, SUMMARY_HEADER)
        exists = summary_path.exists() and summary_path.stat().st_size > 0
        with summary_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not exists:
                writer.writerow(SUMMARY_HEADER)
            writer.writerow(
                [
                    SCHEMA_VERSION,
                    self.session_id,
                    active["trial_id"],
                    condition.condition_id,
                    active["trial_index"],
                    condition.operator,
                    _csv_number(condition.planned_release_x_mm, 3),
                    _csv_number(condition.planned_release_y_mm, 3),
                    _csv_number(condition.planned_release_z_mm, 3),
                    _csv_number(
                        None if reference_center is None else reference_center[0],
                        3,
                    ),
                    _csv_number(
                        None if reference_center is None else reference_center[1],
                        3,
                    ),
                    _csv_number(
                        None if reference_center is None else reference_center[2],
                        3,
                    ),
                    _csv_number(condition.extra_field_delay_ms, 3),
                    _csv_number(getattr(self.config, "ball_mass_kg", None), 9),
                    _csv_number(getattr(self.config, "ball_diameter_mm", None), 3),
                    active["release_timestamp_ms"],
                    _csv_number(release_value("measurement_time_sec")),
                    _csv_number(release_value("x_mm"), 3),
                    _csv_number(release_value("y_mm"), 3),
                    _csv_number(release_value("z_mm"), 3),
                    _csv_number(
                        getattr(
                            self.config,
                            "auto_release_gate_max_vz_mm_s",
                            -100.0,
                        ),
                        3,
                    ),
                    _csv_number(
                        getattr(
                            self.config,
                            "auto_release_gate_min_drop_mm",
                            2.5,
                        ),
                        3,
                    ),
                    int(
                        getattr(
                            self.config,
                            "auto_release_gate_consecutive_estimates",
                            3,
                        )
                    ),
                    _csv_number(active["release_gate_origin_z_mm"], 3),
                    _csv_number(active["release_gate_minimum_vz_mm_s"], 3),
                    _csv_number(active["release_gate_maximum_drop_mm"], 3),
                    int(bool(active["release_gate_evidence_seen"])),
                    _csv_number(active["release_gate_passed_time_sec"]),
                    _csv_number(
                        None
                        if active["release_gate_passed_time_sec"] is None
                        else (
                            active["release_gate_passed_time_sec"]
                            - active["release_time_sec"]
                        )
                        * 1000.0,
                        3,
                    ),
                    _csv_number(active["release_gate_pass_vz_mm_s"], 3),
                    _csv_number(active["release_gate_pass_drop_mm"], 3),
                    (
                        ""
                        if active["release_gate_pass_consecutive_count"] is None
                        else int(
                            active["release_gate_pass_consecutive_count"]
                        )
                    ),
                    _csv_number(active["motion_ready_time_sec"]),
                    _csv_number(field_time),
                    _csv_number(
                        None
                        if field_time is None
                        else (field_time - active["release_time_sec"]) * 1000.0,
                        3,
                    ),
                    _csv_number(None if prediction is None else prediction.x_mm, 3),
                    _csv_number(None if prediction is None else prediction.y_mm, 3),
                    _csv_number(None if prediction is None else prediction.z_mm, 3),
                    _csv_number(None if prediction is None else prediction.vz_mm_s, 3),
                    _csv_number(onset_value("measurement_time_sec")),
                    _csv_number(
                        None
                        if onset is None or field_time is None
                        else (onset_value("measurement_time_sec") - field_time) * 1000.0,
                        3,
                    ),
                    _csv_number(onset_value("x_mm"), 3),
                    _csv_number(onset_value("y_mm"), 3),
                    _csv_number(onset_value("z_mm"), 3),
                    _csv_number(onset_value("vx_mm_s"), 3),
                    _csv_number(onset_value("vy_mm_s"), 3),
                    _csv_number(onset_value("vz_mm_s"), 3),
                    _csv_number(onset_rxy, 3),
                    _csv_number(prediction_error[0], 3),
                    _csv_number(prediction_error[1], 3),
                    _csv_number(prediction_error[2], 3),
                    _csv_number(prediction_error_3d, 3),
                    _csv_number(None if anchor is None else anchor[0], 3),
                    _csv_number(None if anchor is None else anchor[1], 3),
                    _csv_number(None if plan is None else plan.duration_sec * 1000.0, 3),
                    _csv_number(None if plan is None else plan.stop_z_mm, 3),
                    _csv_number(None if plan is None else plan.stopping_distance_mm, 3),
                    _csv_number(None if plan is None else plan.required_force_mN, 3),
                    _csv_number(None if plan is None else plan.maximum_force_mN, 3),
                    "" if plan is None else int(bool(plan.feasible)),
                    "" if plan is None else int(bool(plan.force_saturated)),
                    "" if plan is None else int(bool(plan.workspace_limited)),
                    _csv_number(braking_end_value("event_time_sec")),
                    _csv_number(
                        None
                        if braking_end is None or field_time is None
                        else (
                            braking_end_value("event_time_sec") - field_time
                        )
                        * 1000.0,
                        3,
                    ),
                    "" if braking_end is None else braking_end_value("reason"),
                    _csv_number(braking_end_value("x_mm"), 3),
                    _csv_number(braking_end_value("y_mm"), 3),
                    _csv_number(braking_end_value("z_mm"), 3),
                    _csv_number(braking_end_value("vx_mm_s"), 3),
                    _csv_number(braking_end_value("vy_mm_s"), 3),
                    _csv_number(braking_end_value("vz_mm_s"), 3),
                    _csv_number(actual_stopping_distance, 3),
                    _csv_number(stopping_distance_error, 3),
                    _csv_number(plan_stop_z_error, 3),
                    _csv_number(local_hold_time),
                    _csv_number(
                        None
                        if local_hold_time is None or field_time is None
                        else (local_hold_time - field_time) * 1000.0,
                        3,
                    ),
                    _csv_number(
                        None
                        if local_hold_time is None
                        else (now_sec - local_hold_time) * 1000.0,
                        3,
                    ),
                    hold_count,
                    _csv_number(hold_rms_xy, 3),
                    _csv_number(hold_rms_z, 3),
                    _csv_number(minimum_z, 3),
                    _csv_number(maximum_drop, 3),
                    _csv_number(active["maximum_xy_distance_to_target_mm"], 3),
                    _csv_number(active["maximum_commanded_intensity"], 4),
                    active["sample_count"],
                    _csv_number(
                        getattr(self.config, "si2026_success_hold_sec", 5.0),
                        3,
                    ),
                    _csv_number(success_reached_time),
                    _csv_number(
                        None
                        if success_reached_time is None
                        else (
                            success_reached_time - active["release_time_sec"]
                        )
                        * 1000.0,
                        3,
                    ),
                    _csv_number(
                        getattr(
                            self.config,
                            "si2026_observation_sec",
                            10.0,
                        ),
                        3,
                    ),
                    _csv_number(
                        None
                        if success_reached_time is None
                        else max(0.0, now_sec - success_reached_time) * 1000.0,
                        3,
                    ),
                    _csv_number(
                        (now_sec - active["release_time_sec"]) * 1000.0,
                        3,
                    ),
                    status,
                    int(status == "success"),
                    str(reason),
                    condition.notes,
                ]
            )
        self._last_completed_trial = {
            "trial_id": active["trial_id"],
            "status": status,
            "reason": str(reason),
            "event_time_sec": now_sec,
        }
        self._active = None

    def close(self):
        if self.active:
            self.finish_trial(
                status=(
                    "success"
                    if self.success_reached
                    else "failure"
                    if self.field_started
                    else "aborted"
                ),
                reason=(
                    "application_closed_during_active_capture"
                    if self.field_started
                    else "application_closed_before_field_started"
                ),
            )
        try:
            if self._timeseries_file is not None:
                self._timeseries_file.close()
        finally:
            self._timeseries_file = None
            self._timeseries_writer = None
