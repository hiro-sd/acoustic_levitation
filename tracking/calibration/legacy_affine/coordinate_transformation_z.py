import os
import sys
import time
import csv
import math
from typing import Optional, Tuple, List
import numpy as np
import cv2
import keyboard

# XIMEA のパス設定
sys.path.append(r"C:\Users\Hiroto Yoshida\Desktop\XIMEA\API\Python\v3")
try:
    from ximea import xiapi
except ImportError:
    print("[WARN] ximea モジュールなし。")
    xiapi = None

# AUTD
from pyautd3 import AUTD3, Controller, FociSTM, Hz, Silencer, Static
from pyautd3.link.twincat import TwinCAT

# 設定
# 回転設定
ROTATE_FRAME = True
ROTATE_CODE = cv2.ROTATE_90_CLOCKWISE

# 真横カメラの内部パラメータ
INTRINSIC_NPZ = "./tracking/calibration/intrinsic_charuco_cam2.npz"

# 出力CSV
CSV_PATH = "./tracking/z_calib_points.csv"

# XIMEA
EXPOSURE_US = 5000
CAMERA_SN = "43435351"  # side camera serial number

# AUTD STM
POINT_NUM = 8
RADIUS = 23.5

# x, y は固定、zだけ動かす
X_FIXED = 0.0
Y_FIXED = 0.0
Z0 = 400.0
Z_STEP_MM = 2.0

# 取得設定
CAPTURE_SECONDS = 15.0
MIN_SAMPLES_TO_SAVE = 20

# 画像処理設定
USE_OTSU = False
FIXED_THRESH = 160
BLUR_KSIZE = 5
MIN_AREA_PX = 200
MAX_AREA_PX = 200000

ROI_INIT_SIZE = 240
ROI_MIN_SIZE = 120
ROI_MAX_SIZE = 640
ROI_MARGIN = 40
ROI_EXPAND_ON_LOST = 1.15

DISPLAY_EVERY_N_FRAMES = 3

os.environ.setdefault("OMP_NUM_THREADS", "4")

# AUTD 配置
autd_arrangement = [
    AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[0.0, (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[0.0, 2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, 2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[2 * AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[2 * AUTD3.DEVICE_WIDTH, (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[2 * AUTD3.DEVICE_WIDTH, 2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
]

def init_ximea_camera():
    if xiapi is None:
        raise RuntimeError("XIMEA API not loaded")

    cam = xiapi.Camera()
    cam.open_device_by_SN(CAMERA_SN)
    print(f"[INFO] XIMEA camera opened. SN={CAMERA_SN}")

    cam.set_imgdataformat("XI_RGB24")

    try:
        cam.enable_auto_wb()
    except AttributeError:
        try:
            cam.set_param("auto_wb", 1)
        except Exception:
            print("[WARN] auto white balance could not be enabled.")

    cam.set_exposure(EXPOSURE_US)
    print(f"[INFO] Exposure set to {EXPOSURE_US} us")

    img = xiapi.Image()
    cam.start_acquisition()
    print("[INFO] XIMEA acquisition started.")
    return cam, img


def load_intrinsic(npz_path: str):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Intrinsic npz not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    if "camera_matrix" in data and "dist_coeffs" in data:
        mtx = data["camera_matrix"].astype(np.float32)
        dist = data["dist_coeffs"].astype(np.float32)
    elif "mtx" in data and "dist" in data:
        mtx = data["mtx"].astype(np.float32)
        dist = data["dist"].astype(np.float32)
    else:
        keys = ", ".join(data.files)
        raise KeyError(
            "Intrinsic npz keys are unsupported. "
            f"Expected (camera_matrix, dist_coeffs) or (mtx, dist), got: {keys}"
        )

    print("[INFO] Loaded intrinsic parameters.")
    return mtx, dist


def ensure_csv_header(path: str):
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp",
            "v_mean",
            "n_samples",
            "z_cmd_mm",
            "u_mean",
            "r_mean",
        ])
    print(f"[INFO] Created CSV: {path}")


def clamp_roi(cx: int, cy: int, size: int, w: int, h: int):
    size = int(max(ROI_MIN_SIZE, min(ROI_MAX_SIZE, size)))
    half = size // 2
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))

    if (x2 - x1) < size:
        if x1 == 0:
            x2 = min(w, x1 + size)
        elif x2 == w:
            x1 = max(0, x2 - size)

    if (y2 - y1) < size:
        if y1 == 0:
            y2 = min(h, y1 + size)
        elif y2 == h:
            y1 = max(0, y2 - size)

    return x1, y1, x2, y2


def undistort_frame(frame: np.ndarray, mtx: np.ndarray, dist: np.ndarray) -> np.ndarray:
    return cv2.undistort(frame, mtx, dist)

def rotate_frame_if_needed(frame: np.ndarray) -> np.ndarray:
    if not ROTATE_FRAME:
        return frame
    return cv2.rotate(frame, ROTATE_CODE)

def track_ball_cv(frame_rgb: np.ndarray, roi_rect) -> Tuple[Optional[Tuple[float, float, float]], np.ndarray]:
    x1, y1, x2, y2 = roi_rect
    roi = frame_rgb[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi
    if BLUR_KSIZE > 1:
        gray = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)

    if USE_OTSU:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray, FIXED_THRESH, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, bw

    roi_cx, roi_cy = (x2 - x1) / 2.0, (y2 - y1) / 2.0
    best = None
    best_score = -1e18

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX or area > MAX_AREA_PX:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue

        M = cv2.moments(cnt)
        if M["m00"] <= 1e-6:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist2 = (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2

        score = area - 0.8 * dist2
        if score > best_score:
            best_score = score
            best = cnt

    if best is None:
        return None, bw

    (xc, yc), r = cv2.minEnclosingCircle(best)
    return (float(x1 + xc), float(y1 + yc), float(r)), bw


def send_stm(autd, center_xyz: np.ndarray):
    stm = FociSTM(
        foci=(
            center_xyz + RADIUS * np.array([np.cos(theta), np.sin(theta), 0.0])
            for theta in (
                np.pi / 8 + 2.0 * np.pi * i / POINT_NUM
                for i in range(POINT_NUM)
            )
        ),
        config=100 * Hz,
    ).into_nearest()
    autd.send(stm)


def main():
    ensure_csv_header(CSV_PATH)
    mtx, dist = load_intrinsic(INTRINSIC_NPZ)
    cam, img = init_ximea_camera()

    with Controller.open(
        autd_arrangement,
        TwinCAT(),
    ) as autd:

        autd.send(Silencer())
        autd.send(Static(intensity=int(0xFF * 0.9)))

        base_center = autd.center()

        x_cmd = X_FIXED
        y_cmd = Y_FIXED
        z_cmd = Z0

        center_xyz = base_center + np.array([x_cmd, y_cmd, z_cmd], dtype=np.float32)
        send_stm(autd, center_xyz)

        window_name = "Side Z Calibration | up/down: z move / c: capture / r: reset ROI / esc: quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        prev_time = time.time()
        fps = 0.0

        capturing = False
        capture_end_t = 0.0
        samples_uvr: List[Tuple[float, float, float]] = []

        prev_c_state = False
        prev_up = False
        prev_down = False
        prev_r_state = False

        cam.get_image(img)
        frame0 = img.get_image_data_numpy()
        frame0_ud = undistort_frame(frame0, mtx, dist)
        frame0_proc = rotate_frame_if_needed(frame0_ud)
        H, W = frame0_proc.shape[:2]

        roi_cx, roi_cy = W // 2, H // 2
        roi_size = ROI_INIT_SIZE
        frame_count = 0

        try:
            while True:
                if keyboard.is_pressed("esc"):
                    print("[INFO] Exit.")
                    break

                # z移動
                up = keyboard.is_pressed("up")
                down = keyboard.is_pressed("down")

                moved = False
                if up and not prev_up:
                    z_cmd += Z_STEP_MM
                    moved = True
                if down and not prev_down:
                    z_cmd -= Z_STEP_MM
                    moved = True

                prev_up, prev_down = up, down

                if moved:
                    center_xyz = base_center + np.array([x_cmd, y_cmd, z_cmd], dtype=np.float32)
                    send_stm(autd, center_xyz)
                    print(f"[INFO] Move target z -> {z_cmd:.2f} mm")

                # ROIリセット
                r_state = keyboard.is_pressed("r")
                if r_state and not prev_r_state:
                    roi_cx, roi_cy = W // 2, H // 2
                    roi_size = ROI_INIT_SIZE
                    print("[INFO] ROI reset.")
                prev_r_state = r_state

                # フレーム取得
                cam.get_image(img)
                frame = img.get_image_data_numpy()

                frame_ud = undistort_frame(frame, mtx, dist)
                frame_proc = rotate_frame_if_needed(frame_ud)

                frame_count += 1
                do_display = (frame_count % DISPLAY_EVERY_N_FRAMES == 0)
                annotated = frame_proc.copy() if do_display else None

                x1, y1, x2, y2 = clamp_roi(roi_cx, roi_cy, roi_size, W, H)
                track, bw = track_ball_cv(frame_proc, (x1, y1, x2, y2))

                detected = False
                u = v = r = 0.0

                if track is not None:
                    u, v, r = track
                    detected = True

                    roi_cx, roi_cy = int(u), int(v)
                    desired = int(2 * (2.5 * r + ROI_MARGIN))
                    roi_size = int(0.7 * roi_size + 0.3 * desired)
                    roi_size = int(max(ROI_MIN_SIZE, min(ROI_MAX_SIZE, roi_size)))

                    if capturing:
                        samples_uvr.append((u, v, r))

                    if do_display:
                        cv2.circle(annotated, (int(u), int(v)), int(max(2, r)), (0, 255, 0), 2)
                        cv2.circle(annotated, (int(u), int(v)), 4, (0, 0, 255), -1)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(
                            annotated,
                            f"u={u:.1f}, v={v:.1f}, r={r:.1f}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                else:
                    roi_size = int(min(ROI_MAX_SIZE, roi_size * ROI_EXPAND_ON_LOST))
                    if do_display:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # FPS
                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                # capture開始
                c_state = keyboard.is_pressed("c")
                if c_state and not prev_c_state and not capturing:
                    capturing = True
                    samples_uvr = []
                    capture_end_t = time.monotonic() + CAPTURE_SECONDS
                    print(f"[INFO] Start capturing for {CAPTURE_SECONDS:.1f}s at z={z_cmd:.2f} mm")
                prev_c_state = c_state

                # capture終了
                if capturing:
                    remaining = capture_end_t - time.monotonic()
                    if do_display:
                        cv2.putText(
                            annotated,
                            f"CAPTURING {max(0.0, remaining):.1f}s | samples={len(samples_uvr)}",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    if remaining <= 0.0:
                        capturing = False

                        if len(samples_uvr) < MIN_SAMPLES_TO_SAVE:
                            print(f"[WARN] Not enough samples: {len(samples_uvr)}. Not saved.")
                        else:
                            u_mean = float(np.mean([p[0] for p in samples_uvr]))
                            v_mean = float(np.mean([p[1] for p in samples_uvr]))
                            r_mean = float(np.mean([p[2] for p in samples_uvr]))
                            ts = time.strftime("%Y-%m-%d %H:%M:%S")

                            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                                w = csv.writer(f)
                                w.writerow([ts, v_mean, len(samples_uvr), z_cmd, u_mean, r_mean])

                            print(
                                f"[INFO] Saved: v_mean={v_mean:.2f}, z={z_cmd:.2f}, "
                                f"n={len(samples_uvr)}, u_mean={u_mean:.2f}, r_mean={r_mean:.2f}"
                            )

                if do_display:
                    cv2.putText(
                        annotated,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated,
                        f"Target z: {z_cmd:.1f} mm",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated,
                        "UP/DOWN: move z | c: capture | r: reset ROI | esc: quit",
                        (10, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, annotated)

                    if cv2.waitKey(1) & 0xFF == 27:
                        print("[INFO] Exit.")
                        break

        finally:
            print("[INFO] Cleanup.")
            try:
                cam.stop_acquisition()
            except Exception:
                pass
            try:
                cam.close_device()
            except Exception:
                pass
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()