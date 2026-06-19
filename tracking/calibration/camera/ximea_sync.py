import time
from pathlib import Path
import sys
import cv2
import numpy as np

# TODO: 画像をきちんと設定する (今は白黒っぽい)

# XIMEA設定
sys.path.append(r"C:\Users\Hiroto Yoshida\Desktop\XIMEA\API\Python\v3")
try:
    from ximea import xiapi
except ImportError:
    print("[WARN] ximea モジュールなし。")
    xiapi = None

# ユーザー設定
CAM1_SN = "43430551"
CAM2_SN = "43435351"

SAVE_ROOT = Path("./tracking/calibration/dual_usb_soft_sync")
PAIR_DIR = SAVE_ROOT / "pairs"
PAIR_DIR.mkdir(parents=True, exist_ok=True)

NUM_FRAMES = 100
TIMEOUT_MS = 2000

EXPOSURE_US = 5000
IMG_FORMAT = "XI_RGB24"   # 必要なら XI_RGB24
WIDTH = None              # 例: 1024
HEIGHT = None             # 例: 768

# 保存設定
SAVE_IMAGES = True
SAVE_EVERY_FRAME = True   # True: 毎フレーム保存 / False: sキーを押した時だけ保存

# 成功判定（固定オフセットを引いた後の相対ずれ）
PASS_STD_REL_MS = 0.2
PASS_MAX_ABS_REL_MS = 1.0

WINDOW_NAME_1 = "cam1"
WINDOW_NAME_2 = "cam2"


def open_camera_by_sn(sn: str) -> xiapi.Camera:
    cam = xiapi.Camera()
    cam.open_device_by_SN(sn)
    return cam


def apply_common_settings(cam: xiapi.Camera) -> None:
    cam.set_imgdataformat(IMG_FORMAT)
    cam.set_exposure(EXPOSURE_US)

    if WIDTH is not None:
        cam.set_width(WIDTH)
    if HEIGHT is not None:
        cam.set_height(HEIGHT)

    if hasattr(cam, "disable_aeag"):
        try:
            cam.disable_aeag()
        except Exception:
            pass

    if hasattr(cam, "set_gain"):
        try:
            cam.set_gain(0.0)
        except Exception:
            pass


def configure_software_trigger(cam: xiapi.Camera) -> None:
    cam.set_trigger_source("XI_TRG_SOFTWARE")


def trigger_camera(cam: xiapi.Camera) -> None:
    cam.set_trigger_software(1)


def image_timestamp_ns(img: xiapi.Image) -> int:
    return int(img.tsSec) * 1_000_000_000 + int(img.tsUSec) * 1_000


def to_numpy(img: xiapi.Image) -> np.ndarray:
    return img.get_image_data_numpy()


def save_frame(arr: np.ndarray, path: Path) -> None:
    if arr.ndim == 2:
        cv2.imwrite(str(path), arr)
    else:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)


def preview_image(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    cam1 = None
    cam2 = None
    img1 = None
    img2 = None

    dt_list_ms = []
    dt_rel_list_ms = []
    pair_count = 0
    dt0_ms = None

    try:
        print("[INFO] Opening cameras...")
        cam1 = open_camera_by_sn(CAM1_SN)
        cam2 = open_camera_by_sn(CAM2_SN)

        print(f"[INFO] cam1 SN={cam1.get_device_sn()}")
        print(f"[INFO] cam2 SN={cam2.get_device_sn()}")

        print("[INFO] Applying common settings...")
        apply_common_settings(cam1)
        apply_common_settings(cam2)

        print("[INFO] Setting software trigger mode...")
        configure_software_trigger(cam1)
        configure_software_trigger(cam2)

        img1 = xiapi.Image()
        img2 = xiapi.Image()

        cam1.start_acquisition()
        cam2.start_acquisition()
        print("[INFO] Acquisition started.")

        if not SAVE_EVERY_FRAME:
            cv2.namedWindow(WINDOW_NAME_1, cv2.WINDOW_NORMAL)
            cv2.namedWindow(WINDOW_NAME_2, cv2.WINDOW_NORMAL)
            print("[INFO] Press 's' to save current checkerboard pair.")
            print("[INFO] Press 'q' or ESC to quit.")

        time.sleep(0.2)

        for i in range(NUM_FRAMES):
            trigger_camera(cam1)
            trigger_camera(cam2)

            cam1.get_image(img1, timeout=TIMEOUT_MS)
            cam2.get_image(img2, timeout=TIMEOUT_MS)

            arr1 = to_numpy(img1)
            arr2 = to_numpy(img2)

            ts1 = image_timestamp_ns(img1)
            ts2 = image_timestamp_ns(img2)
            dt_ms = (ts2 - ts1) / 1e6

            if dt0_ms is None:
                dt0_ms = dt_ms

            dt_rel_ms = dt_ms - dt0_ms

            dt_list_ms.append(dt_ms)
            dt_rel_list_ms.append(dt_rel_ms)

            print(
                f"[{i:04d}] "
                f"cam1_frame={img1.nframe:>6} ts={ts1} | "
                f"cam2_frame={img2.nframe:>6} ts={ts2} | "
                f"dt={dt_ms:.3f} ms | "
                f"dt_rel={dt_rel_ms:+.3f} ms"
            )

            should_save = False

            if SAVE_EVERY_FRAME and SAVE_IMAGES:
                should_save = True
            elif not SAVE_EVERY_FRAME:
                show1 = preview_image(arr1)
                show2 = preview_image(arr2)
                cv2.imshow(WINDOW_NAME_1, show1)
                cv2.imshow(WINDOW_NAME_2, show2)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    should_save = True
                elif key == ord("q") or key == 27:
                    print("[INFO] Interrupted by user.")
                    break

            if should_save:
                f1 = PAIR_DIR / f"pair_{pair_count:04d}_cam1.png"
                f2 = PAIR_DIR / f"pair_{pair_count:04d}_cam2.png"
                save_frame(arr1, f1)
                save_frame(arr2, f2)
                print(f"[SAVE] Saved pair {pair_count:04d}")
                pair_count += 1

        dt_arr = np.array(dt_list_ms, dtype=np.float64)
        dt_rel_arr = np.array(dt_rel_list_ms, dtype=np.float64)

        mean_dt = float(np.mean(dt_arr))
        std_dt = float(np.std(dt_arr))
        min_dt = float(np.min(dt_arr))
        max_dt = float(np.max(dt_arr))

        mean_rel = float(np.mean(dt_rel_arr))
        std_rel = float(np.std(dt_rel_arr))
        min_rel = float(np.min(dt_rel_arr))
        max_rel = float(np.max(dt_rel_arr))
        max_abs_rel = float(np.max(np.abs(dt_rel_arr)))

        print("\n========== RESULT ==========")
        print(f"[INFO] frames              : {len(dt_list_ms)}")
        print(f"[INFO] saved pairs         : {pair_count}")
        print(f"[INFO] offset dt0 ms       : {dt0_ms:.3f}")
        print(f"[INFO] mean dt ms          : {mean_dt:.3f}")
        print(f"[INFO] std  dt ms          : {std_dt:.3f}")
        print(f"[INFO] min  dt ms          : {min_dt:.3f}")
        print(f"[INFO] max  dt ms          : {max_dt:.3f}")
        print(f"[INFO] mean dt_rel ms      : {mean_rel:.3f}")
        print(f"[INFO] std  dt_rel ms      : {std_rel:.3f}")
        print(f"[INFO] min  dt_rel ms      : {min_rel:.3f}")
        print(f"[INFO] max  dt_rel ms      : {max_rel:.3f}")
        print(f"[INFO] max |dt_rel| ms     : {max_abs_rel:.3f}")

        passed = (std_rel <= PASS_STD_REL_MS) and (max_abs_rel <= PASS_MAX_ABS_REL_MS)

        if passed:
            print("[PASS] 固定オフセット差し引き後の相対同期は安定しています")
            print("[PASS] この画像ペアでキャリブレーションへ進めます")
        else:
            print("[WARN] 相対同期の揺れが大きめです")
            print("[WARN] キャリブレーションは試せますが、撮影条件の見直しを推奨します")

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        for cam in (cam1, cam2):
            if cam is not None:
                try:
                    cam.stop_acquisition()
                except Exception:
                    pass
                try:
                    cam.close_device()
                except Exception:
                    pass
        print("[INFO] Cameras closed.")


if __name__ == "__main__":
    main()