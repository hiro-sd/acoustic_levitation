# 半径を大きくすると物体は下降、小さくすると上昇する。しかしこれはあまり制御には使えなそうかなあ
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.config import AppConfig
from tracking.core.app import run_tracking_app


if __name__ == "__main__":
    cfg = AppConfig(
        # Radius change
        enable_radius_change=True,

        # 今回は矢印キーをradius変更に使うのでbase移動はOFF
        enable_base_move=False,

        # OutputMaskは使わない
        use_output_mask=False,
    )

    run_tracking_app(cfg)