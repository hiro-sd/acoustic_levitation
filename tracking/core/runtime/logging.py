import csv
import os
import time

import numpy as np

from ..config import AppConfig
from ..models import HomePosition, Target3D
from ..control.recovery import RecoveryTelemetry


LOG_HEADER = [
    "timestamp",
    "session_mode",
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
    "z_drop_mm",
    "vz_mm_s",
    "predicted_z_mm",
    "predicted_vz_mm_s",
    "required_force_mN",
    "commanded_intensity",
    "actual_intensity",
    "capture_force_saturated",
    "temporary_home_x",
    "temporary_home_y",
    "temporary_home_z",
    "mode_transition_reason",
]


class StabilityLogger:
    """Owns the lifecycle and duplicate suppression of a timed CSV session."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.file = None
        self.writer = None
        self.active = False
        self.mode = ""
        self.end_time = 0.0
        self.last_xy_time = None
        self.last_z_time = None

    def start(self, mode: str):
        if self.active:
            return

        directory = os.path.dirname(self.cfg.log_csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        file_exists = os.path.exists(self.cfg.log_csv_path)
        if file_exists and os.path.getsize(self.cfg.log_csv_path) > 0:
            try:
                with open(self.cfg.log_csv_path, newline="", encoding="utf-8") as existing:
                    existing_header = next(csv.reader(existing), [])
                if existing_header != LOG_HEADER:
                    backup_path = (
                        f"{self.cfg.log_csv_path}.bak_"
                        f"{time.strftime('%Y%m%d_%H%M%S')}"
                    )
                    os.replace(self.cfg.log_csv_path, backup_path)
                    file_exists = False
                    print(
                        "[LOG] Existing CSV header differs from current schema. "
                        f"Moved old log to: {backup_path}"
                    )
            except Exception as exc:
                print(f"[LOG] Warning: failed to inspect existing CSV header: {exc}")

        self.file = open(self.cfg.log_csv_path, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        if not file_exists or os.path.getsize(self.cfg.log_csv_path) == 0:
            self.writer.writerow(LOG_HEADER)

        self.mode = mode
        self.end_time = time.time() + self.cfg.log_duration_sec
        self.last_xy_time = None
        self.last_z_time = None
        self.active = True
        print(f"[LOG] Start {self.cfg.log_duration_sec:.1f}s logging: mode={mode}")

    def stop_if_finished(self, now: float | None = None):
        if self.active and (time.time() if now is None else now) >= self.end_time:
            self.close()
            print("[LOG] Finished logging.")

    def write(
        self,
        *,
        frame_xy_time: float,
        frame_z_time: float,
        u_xy: float,
        v_xy: float,
        v_z: float,
        x_mm: float | None,
        y_mm: float | None,
        z_mm: float | None,
        home: HomePosition,
        target: Target3D,
        control_mode: str = "",
        recovery: RecoveryTelemetry | None = None,
    ):
        if not self.active or self.writer is None:
            return
        if self.last_xy_time == frame_xy_time and self.last_z_time == frame_z_time:
            return

        self.last_xy_time = frame_xy_time
        self.last_z_time = frame_z_time
        temporary_home = recovery.temporary_home if recovery is not None else None
        self.writer.writerow(
            [
                f"{time.time():.4f}",
                self.mode,
                control_mode,
                f"{u_xy:.2f}" if np.isfinite(u_xy) else "",
                f"{v_xy:.2f}" if np.isfinite(v_xy) else "",
                f"{v_z:.2f}" if np.isfinite(v_z) else "",
                f"{x_mm:.3f}" if x_mm is not None else "",
                f"{y_mm:.3f}" if y_mm is not None else "",
                f"{z_mm:.3f}" if z_mm is not None else "",
                f"{home.x:.3f}",
                f"{home.y:.3f}",
                f"{home.z:.3f}",
                f"{target.x:.3f}",
                f"{target.y:.3f}",
                f"{target.z:.3f}",
                (
                    f"{recovery.z_drop_mm:.3f}"
                    if recovery is not None and recovery.z_drop_mm is not None
                    else ""
                ),
                (
                    f"{recovery.vz_mm_s:.3f}"
                    if recovery is not None and recovery.vz_mm_s is not None
                    else ""
                ),
                (
                    f"{recovery.predicted_z_mm:.3f}"
                    if recovery is not None and recovery.predicted_z_mm is not None
                    else ""
                ),
                (
                    f"{recovery.predicted_vz_mm_s:.3f}"
                    if recovery is not None and recovery.predicted_vz_mm_s is not None
                    else ""
                ),
                (
                    f"{recovery.required_force_mN:.3f}"
                    if recovery is not None and recovery.required_force_mN is not None
                    else ""
                ),
                (
                    f"{recovery.commanded_intensity:.3f}"
                    if recovery is not None and recovery.commanded_intensity is not None
                    else ""
                ),
                (
                    f"{recovery.actual_intensity:.3f}"
                    if recovery is not None and recovery.actual_intensity is not None
                    else ""
                ),
                (
                    "1"
                    if recovery is not None and recovery.capture_force_saturated
                    else "0"
                ),
                f"{temporary_home.x:.3f}" if temporary_home is not None else "",
                f"{temporary_home.y:.3f}" if temporary_home is not None else "",
                f"{temporary_home.z:.3f}" if temporary_home is not None else "",
                recovery.mode_transition_reason if recovery is not None else "",
            ]
        )

    def close(self):
        try:
            if self.file is not None:
                self.file.close()
        except Exception:
            pass
        finally:
            self.file = None
            self.writer = None
            self.active = False
