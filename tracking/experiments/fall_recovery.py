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

        enable_base_move=False,
        enable_radius_change=False,
        use_output_mask=False,

        use_gravity_prediction_z=True,

        z_min=250.0,
        z_max=550.0,

        enable_z_intensity_boost=True,
        static_intensity_ratio=0.90,
        intensity_max_ratio=0.95,

        enable_fall_recovery=True,
    )

    run_tracking_app(cfg)
