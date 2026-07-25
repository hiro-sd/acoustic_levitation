import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.config import AppConfig
from tracking.core.app import run_tracking_app


if __name__ == "__main__":
    cfg = AppConfig(
        radius=19.0,
        default_z=400.0,
        static_intensity_ratio=0.6,

        # stereo_camera_to_autd.npz が作成済みの場合だけ、ステレオ3Dを制御に使う。
        enable_stereo_triangulation=True,
        use_stereo_position_for_control=True,
        stereo_camera_to_autd_npz="./tracking/calibration/stereo_camera_to_autd.npz",

        enable_base_move=True,
        enable_radius_change=False,
        enable_fall_recovery=False,
        enable_auto_demo=True,
        use_gravity_prediction_z=True,

        # dt_pred_xy=0.05,
        # dt_pred_z=0.05,

        # use_output_mask=True,
        # output_mask_radius_mm=170.0,

        log_csv_path="./tracking/stability_log.csv",
    )

    run_tracking_app(cfg)
