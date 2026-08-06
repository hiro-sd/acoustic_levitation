import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import cv2

from .config import AppConfig


def load_ximea_api(cfg: AppConfig):
    """
    XIMEA API を import する。
    Windows上のSDKパスを cfg.ximea_python_path から追加する。
    """
    if cfg.ximea_python_path and cfg.ximea_python_path not in sys.path:
        sys.path.append(cfg.ximea_python_path)

    try:
        from ximea import xiapi
        return xiapi
    except ImportError:
        print("[WARN] ximea モジュールなし。")
        return None

def init_ximea_camera(xiapi, camera_sn: str, role: str, cfg: AppConfig):
    """
    XIMEAカメラを開き、露光・フレームレートを設定して acquisition を開始する。
    """
    if xiapi is None:
        raise RuntimeError("XIMEA API not loaded")

    cam = xiapi.Camera()
    cam.open_device_by_SN(camera_sn)
    print(f"[INFO] XIMEA camera opened ({role}) SN={camera_sn}.")

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

    cam.set_exposure(cfg.exposure_us)
    print(f"[INFO] Exposure set to {cfg.exposure_us} us")

    try:
        framerate_min = cam.get_framerate_minimum()
        framerate_max = cam.get_framerate_maximum()
        framerate_inc = cam.get_framerate_increment()

        target_framerate = framerate_max
        if framerate_inc > 0:
            target_framerate = np.floor(framerate_max / framerate_inc) * framerate_inc
            if target_framerate < framerate_min:
                target_framerate = framerate_max

        cam.set_framerate(target_framerate)
        applied_framerate = cam.get_framerate()

        print(
            f"[INFO] Frame rate set ({role}): "
            f"min={framerate_min:.2f}, max={framerate_max:.2f}, "
            f"inc={framerate_inc:.2f}, applied={applied_framerate:.2f} fps"
        )
    except AttributeError:
        print(f"[WARN] Framerate API unavailable ({role}); skipped frame rate configuration.")
    except Exception as e:
        print(f"[WARN] Failed to configure framerate ({role}): {e}")

    img = xiapi.Image()
    cam.start_acquisition()
    print(f"[INFO] XIMEA acquisition started ({role}).")

    return cam, img


def load_intrinsic(npz_path: str, role: str):
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
            f"Unsupported intrinsic keys for {role}: {keys}. "
            "Expected (camera_matrix, dist_coeffs) or (mtx, dist)."
        )
    
    print(f"[INFO] Loaded intrinsic parameters ({role}) from {npz_path}")
    return mtx, dist


def rotate_frame_if_needed(frame: np.ndarray, do_rotate: bool, rotate_code: int) -> np.ndarray:
    if not do_rotate:
        return frame
    return cv2.rotate(frame, rotate_code)


def build_undistort_maps(frame_shape, mtx: np.ndarray, dist: np.ndarray):
    """
    毎フレーム cv2.undistort() する代わりに、
    最初に1回だけ remap 用テーブルを作る。
    """
    h, w = frame_shape[:2]

    map1, map2 = cv2.initUndistortRectifyMap(
        mtx,
        dist,
        None,
        mtx,
        (w, h),
        cv2.CV_16SC2,
    )

    return map1, map2


def undistort_frame(frame: np.ndarray, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
    """
    事前計算済み remap テーブルで高速に歪み補正する。
    """
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


def clamp_roi(cx, cy, size, w, h, cfg: AppConfig):
    size = int(max(cfg.roi_min_size, min(cfg.roi_max_size, size)))
    half = size // 2

    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(w, cx + half))
    y2 = int(min(h, cy + half))

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


def track_ball_cv(frame_rgb: np.ndarray, roi_rect, cfg: AppConfig):
    """
    ROI内で白い球を検出する。
    戻り値:
        detection: (u, v, r) or None
        bw: 二値画像
    """
    x1, y1, x2, y2 = roi_rect
    roi = frame_rgb[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi

    if cfg.blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)

    if cfg.use_otsu:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray, cfg.fixed_thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, bw

    roi_cx = (x2 - x1) / 2.0
    roi_cy = (y2 - y1) / 2.0

    best = None
    best_score = -1e18

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < cfg.min_area_px or area > cfg.max_area_px:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.6:
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


@dataclass
class SharedFrameBuffer:
    """
    2台のカメラスレッドからフレームを受け取り、撮影時刻が近い組を作る。

    ソフトウェア同期なので露光開始そのものは一致させられないが、短い
    バッファ内から時刻差が最小の組だけをメインループへ渡す。古すぎて
    組にできないフレームは破棄する。
    """
    maxlen: int = 8

    def __post_init__(self):
        self.maxlen = max(1, int(self.maxlen))
        self._lock = threading.Lock()
        self._xy_frames = deque(maxlen=self.maxlen)
        self._z_frames = deque(maxlen=self.maxlen)
        self._latest_xy_time = 0.0
        self._latest_z_time = 0.0

    def set_xy(self, frame: np.ndarray, t: float):
        with self._lock:
            self._xy_frames.append((float(t), frame))
            self._latest_xy_time = float(t)

    def set_z(self, frame: np.ndarray, t: float):
        with self._lock:
            self._z_frames.append((float(t), frame))
            self._latest_z_time = float(t)

    def get_synced_pair(self, max_skew_sec: float):
        """
        バッファ内で時刻差が最小のXY/Zフレームを1組だけ返す。

        返したフレームと、それ以前のフレームは消費する。同じ画像を
        メインループで複数回処理しない。
        """
        max_skew_sec = max(0.0, float(max_skew_sec))

        with self._lock:
            while self._xy_frames and self._z_frames:
                best_xy_idx = 0
                best_z_idx = 0
                best_skew = float("inf")

                for xy_idx, (t_xy, _) in enumerate(self._xy_frames):
                    for z_idx, (t_z, _) in enumerate(self._z_frames):
                        skew = abs(t_xy - t_z)
                        if skew < best_skew:
                            best_xy_idx = xy_idx
                            best_z_idx = z_idx
                            best_skew = skew

                if best_skew <= max_skew_sec:
                    t_xy, frame_xy = self._xy_frames[best_xy_idx]
                    t_z, frame_z = self._z_frames[best_z_idx]

                    for _ in range(best_xy_idx + 1):
                        self._xy_frames.popleft()
                    for _ in range(best_z_idx + 1):
                        self._z_frames.popleft()

                    return (
                        frame_xy.copy(),
                        t_xy,
                        frame_z.copy(),
                        t_z,
                        best_skew,
                    )

                # 最古のフレームは今後到着する相手とも同期しにくいため破棄する。
                if self._xy_frames[0][0] < self._z_frames[0][0]:
                    self._xy_frames.popleft()
                else:
                    self._z_frames.popleft()

            return None

    def get_frame_ages(self, now: float):
        """各カメラで最後に画像を受信してからの経過時間を返す。"""
        with self._lock:
            age_xy = float("inf") if self._latest_xy_time <= 0 else now - self._latest_xy_time
            age_z = float("inf") if self._latest_z_time <= 0 else now - self._latest_z_time
        return age_xy, age_z


def camera_capture_loop(
    cam,
    img,
    role: str,
    cfg: AppConfig,
    frame_buffer: SharedFrameBuffer,
    running_event: threading.Event,
    use_undistort: bool,
    map1,
    map2,
):
    """
    カメラごとの取得スレッド。
    role は "xy" または "z"。
    """
    while running_event.is_set():
        try:
            cam.get_image(img)
            # get_image完了直後を取得時刻とし、後段の画像処理時間を含めない。
            now_t = time.perf_counter()
            frame = img.get_image_data_numpy()

            if (
                role == "z"
                and cfg.rotate_z_frame
                and cfg.z_intrinsic_is_rotated
            ):
                frame = rotate_frame_if_needed(frame, True, cfg.rotate_z_code)

            if use_undistort:
                frame = undistort_frame(frame, map1, map2)

            if (
                role == "z"
                and cfg.rotate_z_frame
                and not cfg.z_intrinsic_is_rotated
            ):
                frame = rotate_frame_if_needed(frame, True, cfg.rotate_z_code)

            if role == "xy":
                frame_buffer.set_xy(frame, now_t)
            elif role == "z":
                frame_buffer.set_z(frame, now_t)
            else:
                raise ValueError(f"Unknown camera role: {role}")

        except Exception as e:
            print(f"[CAM {role} Thread Error] {e}")
            time.sleep(0.01)


def safe_close_camera(cam):
    """Best-effort shutdown shared by normal exit and initialization errors."""
    try:
        cam.stop_acquisition()
    except Exception:
        pass

    try:
        cam.close_device()
    except Exception:
        pass
