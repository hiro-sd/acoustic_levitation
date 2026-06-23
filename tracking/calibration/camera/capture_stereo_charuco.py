"""
Capture synchronized-ish stereo image pairs for ChArUco stereo calibration.

保存形式:
    calibration/stereo_images/charuco/cam1/pair_000_xy.png
    calibration/stereo_images/charuco/cam2/pair_000_z.png

Zカメラ画像は、tracking本体と同じ向きに回転してから保存する。
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
class CaptureSettings:
    camera_xy_sn: str = "43430551"
    camera_z_sn: str = "43435351"
    exposure_us: int = 10000
    rotate_z_frame: bool = True
    rotate_z_code: int = cv2.ROTATE_90_CLOCKWISE
    max_pairs: int = 20
    auto_interval_sec: float = 0.0
    preview_width_px: int = 640
    preview_height_px: int = 480
    output_dir: str = str(TRACKING_ROOT / "calibration" / "stereo_images" / "charuco")


def _open_triggered_camera(xiapi, serial: str, exposure_us: int, role: str):
    cam = xiapi.Camera()
    cam.open_device_by_SN(serial)
    print(f"[INFO] Opened {role} camera SN={serial}")

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

    cam.set_exposure(exposure_us)
    cam.set_trigger_source("XI_TRG_SOFTWARE")
    cam.start_acquisition()
    return cam, xiapi.Image()


def _software_trigger_pair(cam_xy, img_xy, cam_z, img_z):
    t_trigger = time.perf_counter()
    cam_xy.set_trigger_software(1)
    cam_z.set_trigger_software(1)

    cam_xy.get_image(img_xy, timeout=1000)
    t_xy = time.perf_counter()
    cam_z.get_image(img_z, timeout=1000)
    t_z = time.perf_counter()

    return (
        img_xy.get_image_data_numpy().copy(),
        t_xy,
        img_z.get_image_data_numpy().copy(),
        t_z,
        t_trigger,
    )


def _make_preview(
    frame_xy: np.ndarray,
    frame_z: np.ndarray,
    text: str,
    settings: CaptureSettings,
):
    preview_size = (
        int(settings.preview_width_px),
        int(settings.preview_height_px),
    )
    xy_disp = cv2.resize(
        frame_xy,
        preview_size,
        interpolation=cv2.INTER_LINEAR,
    )
    z_disp = cv2.resize(
        frame_z,
        preview_size,
        interpolation=cv2.INTER_LINEAR,
    )
    tiled = np.hstack([xy_disp, z_disp])
    cv2.line(
        tiled,
        (settings.preview_width_px, 0),
        (settings.preview_width_px, settings.preview_height_px - 1),
        (255, 255, 255),
        1,
    )
    cv2.putText(
        tiled,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return tiled


def _save_pair(
    output_dir: Path,
    pair_index: int,
    frame_xy: np.ndarray,
    t_xy: float,
    frame_z: np.ndarray,
    t_z: float,
    settings: CaptureSettings,
):
    cam1_dir = output_dir / "cam1"
    cam2_dir = output_dir / "cam2"
    cam1_dir.mkdir(parents=True, exist_ok=True)
    cam2_dir.mkdir(parents=True, exist_ok=True)

    xy_path = cam1_dir / f"pair_{pair_index:03d}_xy.png"
    z_path = cam2_dir / f"pair_{pair_index:03d}_z.png"
    meta_path = output_dir / f"pair_{pair_index:03d}_meta.json"

    cv2.imwrite(str(xy_path), frame_xy)
    cv2.imwrite(str(z_path), frame_z)
    meta_path.write_text(
        json.dumps(
            {
                "pair_index": pair_index,
                "xy_path": str(xy_path),
                "z_path": str(z_path),
                "t_xy_perf_counter": t_xy,
                "t_z_perf_counter": t_z,
                "software_timestamp_skew_sec": abs(t_xy - t_z),
                "settings": asdict(settings),
                "note": (
                    "Z frame is saved after the same rotation used by the "
                    "tracking runtime when rotate_z_frame is true."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[SAVE] pair {pair_index:03d}: "
        f"{xy_path.name}, {z_path.name}, skew={abs(t_xy - t_z) * 1000.0:.2f} ms"
    )


def main():
    settings = CaptureSettings()
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "capture_settings.json").write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cfg = AppConfig(
        camera_xy_sn=settings.camera_xy_sn,
        camera_z_sn=settings.camera_z_sn,
        exposure_us=settings.exposure_us,
        rotate_z_frame=settings.rotate_z_frame,
        rotate_z_code=settings.rotate_z_code,
    )
    xiapi = load_ximea_api(cfg)
    if xiapi is None:
        raise RuntimeError("XIMEA API is not available.")

    cam_xy = cam_z = None
    try:
        cam_xy, img_xy = _open_triggered_camera(
            xiapi,
            settings.camera_xy_sn,
            settings.exposure_us,
            "xy",
        )
        cam_z, img_z = _open_triggered_camera(
            xiapi,
            settings.camera_z_sn,
            settings.exposure_us,
            "z",
        )

        print("\n" + "=" * 60)
        print("  ChArUco stereo capture")
        print("  [SPACE/C] save current pair")
        print("  [A]       toggle auto capture")
        print("  [Q/ESC]   quit")
        print(f"  output: {output_dir}")
        print("=" * 60 + "\n")

        window_name = "ChArUco Stereo Capture"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        pair_index = 0
        auto_capture = settings.auto_interval_sec > 0.0
        last_auto_save_t = 0.0

        while pair_index < settings.max_pairs:
            frame_xy, t_xy, frame_z, t_z, _ = _software_trigger_pair(
                cam_xy,
                img_xy,
                cam_z,
                img_z,
            )
            frame_z = rotate_frame_if_needed(
                frame_z,
                settings.rotate_z_frame,
                settings.rotate_z_code,
            )

            text = (
                f"pairs {pair_index}/{settings.max_pairs} | "
                f"skew {abs(t_xy - t_z) * 1000.0:.2f} ms | "
                f"auto {'ON' if auto_capture else 'OFF'}"
            )
            cv2.imshow(window_name, _make_preview(frame_xy, frame_z, text, settings))

            now = time.perf_counter()
            should_save = False
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                print("[INFO] stopped by user.")
                break
            if key in (ord(" "), ord("c")):
                should_save = True
            if key == ord("a"):
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
                _save_pair(
                    output_dir,
                    pair_index,
                    frame_xy,
                    t_xy,
                    frame_z,
                    t_z,
                    settings,
                )
                pair_index += 1

        print(f"[FINISH] saved {pair_index} stereo pairs.")

    finally:
        if cam_xy is not None:
            safe_close_camera(cam_xy)
        if cam_z is not None:
            safe_close_camera(cam_z)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
