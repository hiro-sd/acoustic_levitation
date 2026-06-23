"""
Capture ChArUco images for cam2/Z intrinsic calibration.

Zカメラ画像は tracking 実行時と同じ向きに回転してから保存する。
この画像で intrinsic_charuco_cam2.npz を作り直すと、ステレオ画像・実行時画像と
intrinsic の座標系を揃えられる。
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


TRACKING_ROOT = Path(__file__).resolve().parents[2]
if str(TRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKING_ROOT))

from core.config import AppConfig
from core.vision import load_ximea_api, rotate_frame_if_needed, safe_close_camera


@dataclass(frozen=True)
class CaptureCam2Settings:
    camera_z_sn: str = "43435351"
    exposure_us: int = 10000
    rotate_z_frame: bool = True
    rotate_z_code: int = cv2.ROTATE_90_CLOCKWISE
    max_images: int = 40
    auto_interval_sec: float = 5.0
    preview_width_px: int = 640
    preview_height_px: int = 480
    output_dir: str = str(
        TRACKING_ROOT / "calibration" / "intrinsic_images" / "cam2_rotated"
    )


def _open_camera(xiapi, settings: CaptureCam2Settings):
    cam = xiapi.Camera()
    cam.open_device_by_SN(settings.camera_z_sn)
    print(f"[INFO] Opened cam2/Z camera SN={settings.camera_z_sn}")

    cam.set_imgdataformat("XI_RGB24")
    try:
        cam.enable_auto_wb()
    except AttributeError:
        pass

    if hasattr(cam, "disable_aeag"):
        try:
            cam.disable_aeag()
        except Exception:
            pass

    cam.set_exposure(settings.exposure_us)
    cam.set_trigger_source("XI_TRG_SOFTWARE")
    cam.start_acquisition()
    return cam, xiapi.Image()


def _capture_frame(cam, img, settings: CaptureCam2Settings):
    cam.set_trigger_software(1)
    cam.get_image(img, timeout=1000)
    frame = img.get_image_data_numpy().copy()
    frame = rotate_frame_if_needed(
        frame,
        settings.rotate_z_frame,
        settings.rotate_z_code,
    )
    return frame


def _make_preview(frame: np.ndarray, text: str, settings: CaptureCam2Settings):
    preview = cv2.resize(
        frame,
        (settings.preview_width_px, settings.preview_height_px),
        interpolation=cv2.INTER_LINEAR,
    )
    cv2.putText(
        preview,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return preview


def main():
    settings = CaptureCam2Settings()
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "capture_settings.json").write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cfg = AppConfig(
        camera_z_sn=settings.camera_z_sn,
        exposure_us=settings.exposure_us,
        rotate_z_frame=settings.rotate_z_frame,
        rotate_z_code=settings.rotate_z_code,
    )
    xiapi = load_ximea_api(cfg)
    if xiapi is None:
        raise RuntimeError("XIMEA API is not available.")

    cam = None
    try:
        cam, img = _open_camera(xiapi, settings)

        print("\n" + "=" * 60)
        print("  cam2/Z ChArUco intrinsic capture")
        print("  [SPACE/C] save current image")
        print("  [A]       toggle auto capture")
        print("  [Q/ESC]   quit")
        print(f"  output: {output_dir}")
        print("=" * 60 + "\n")

        window_name = "cam2/Z ChArUco Intrinsic Capture"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        image_index = 0
        auto_capture = False
        last_auto_save_t = 0.0

        while image_index < settings.max_images:
            frame = _capture_frame(cam, img, settings)

            now = time.perf_counter()
            auto_remaining = 0.0
            if auto_capture and settings.auto_interval_sec > 0.0:
                auto_remaining = max(
                    0.0,
                    settings.auto_interval_sec - (now - last_auto_save_t),
                )

            text = (
                f"images {image_index}/{settings.max_images} | "
                f"auto {'ON' if auto_capture else 'OFF'}"
            )
            if auto_capture:
                text += f" | next {auto_remaining:.1f}s"

            cv2.imshow(window_name, _make_preview(frame, text, settings))

            should_save = False
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                print("[INFO] stopped by user.")
                break
            if key in (ord(" "), ord("c"), ord("C")):
                should_save = True
            if key in (ord("a"), ord("A")):
                auto_capture = not auto_capture
                last_auto_save_t = now
                print(f"[INFO] auto capture: {'ON' if auto_capture else 'OFF'}")

            if (
                auto_capture
                and settings.auto_interval_sec > 0.0
                and now - last_auto_save_t >= settings.auto_interval_sec
            ):
                should_save = True
                last_auto_save_t = now

            if should_save:
                path = output_dir / f"cam2_rotated_{image_index:03d}.png"
                cv2.imwrite(str(path), frame)
                print(f"[SAVE] {path}")
                image_index += 1

        print(f"[FINISH] saved {image_index} images.")

    finally:
        if cam is not None:
            safe_close_camera(cam)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
