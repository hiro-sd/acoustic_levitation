import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.config import AppConfig
from tracking.core.app import run_tracking_app


if __name__ == "__main__":
    cfg = AppConfig(
        radius=23.5,
        default_z=400.0,
        static_intensity_ratio=0.9,

        kp_xy=0.3,
        kd_xy=0.05,
        ki_xy=0.1,
        dt_pred_xy=0.01,

        kp_z=0.6,
        kd_z=0.15,
        ki_z=0.1,
        z_lpf_alpha=0.7,
        dt_pred_z=0.01,

        z_min=250.0,
        z_max=550.0,

        # [ENTER]でフィードバック開始後、[D]でデモ軌道を開始/停止する。
        enable_auto_demo=True,
        demo_toggle_key="d",
        demo_square_size_mm=60.0,
        demo_xy_speed_mm_s=20.0,
        demo_z_amplitude_mm=20.0,
        demo_z_period_sec=4.0,

        enable_base_move=False,
        use_gravity_prediction_z=True,

        log_csv_path="./tracking/stability_log_demo_square_z.csv",
    )

    run_tracking_app(cfg)
