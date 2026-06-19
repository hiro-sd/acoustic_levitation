import csv
import os
import time

import numpy as np

from ..config import AppConfig
from ..models import HomePosition, Target3D


LOG_HEADER = [
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
    ):
        if not self.active or self.writer is None:
            return
        if self.last_xy_time == frame_xy_time and self.last_z_time == frame_z_time:
            return

        self.last_xy_time = frame_xy_time
        self.last_z_time = frame_z_time
        self.writer.writerow(
            [
                f"{time.time():.4f}",
                self.mode,
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
