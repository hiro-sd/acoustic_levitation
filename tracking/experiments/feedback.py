import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.config import AppConfig
from tracking.core.app import run_tracking_app


if __name__ == "__main__":
    cfg = AppConfig(
        radius=23.5,
        static_intensity_ratio=0.9,
        default_z=400.0,

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

        use_output_mask=False,
        enable_base_move=False,
        use_gravity_prediction_z=True,

        enable_z_intensity_boost=True,

        intensity_max_ratio=1.00,

        z_boost_full_drop_mm=8.0,
        z_boost_release_mm=3.5,

        intensity_lpf_alpha=0.90,
        intensity_update_eps=0.005,
    )

    run_tracking_app(cfg)
