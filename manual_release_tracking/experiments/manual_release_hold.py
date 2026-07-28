import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.config import AppConfig
from manual_release_tracking.manual_release_app import run_manual_release_app


if __name__ == "__main__":
    cfg = AppConfig(
        radius=19.0,
        default_z=400.0,
        static_intensity_ratio=0.6,

        # Use calibrated stereo 3D position as the feedback measurement.
        enable_stereo_triangulation=True,
        use_stereo_position_for_control=True,
        stereo_camera_to_autd_npz="./tracking/calibration/stereo_camera_to_autd.npz",

        # Keep this first experiment simple: hold at the release point only.
        enable_base_move=False,
        enable_radius_change=False,
        enable_fall_recovery=False,
        enable_auto_demo=False,
        use_gravity_prediction_z=False,

        log_csv_path="./manual_release_tracking/manual_release_log.csv",
    )

    # Manual-release handoff:
    # Before pressing R, the app only detects/tracks the sphere and emits no ultrasound.
    # Press R while pinching/holding the sphere in view, wait briefly, then release.
    cfg.manual_release_key = "r"
    cfg.manual_release_pre_hold_sec = 0.3

    run_manual_release_app(cfg)
