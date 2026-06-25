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
        static_intensity_ratio=0.0,

        # 球中心のステレオ三角測量を画面表示する。
        # 出力はまだcam1/XYカメラ座標系なので、AUTD制御には使わない。
        enable_stereo_triangulation=True,
        use_stereo_position_for_control=False,

        enable_base_move=False,
        enable_radius_change=False,
        enable_fall_recovery=False,

        log_csv_path="./tracking/stability_log_stereo_preview.csv",
    )

    run_tracking_app(cfg)
