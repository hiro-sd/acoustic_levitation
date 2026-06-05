import time
import threading
from dataclasses import dataclass

import numpy as np

from pyautd3 import AUTD3, FociSTM, Hz, OutputMask

from .config import AppConfig


def make_autd_arrangement():
    # 3行×3列のAUTD配置
    return [
        AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]),
        AUTD3(pos=[0.0, AUTD3.DEVICE_HEIGHT, 0.0], rot=[1, 0, 0, 0]),
        AUTD3(pos=[0.0, 2 * AUTD3.DEVICE_HEIGHT, 0.0], rot=[1, 0, 0, 0]),

        AUTD3(pos=[AUTD3.DEVICE_WIDTH, 2 * AUTD3.DEVICE_HEIGHT, 0.0], rot=[1, 0, 0, 0]),
        AUTD3(pos=[AUTD3.DEVICE_WIDTH, AUTD3.DEVICE_HEIGHT, 0.0], rot=[1, 0, 0, 0]),
        AUTD3(pos=[AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),

        AUTD3(pos=[2 * AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),
        AUTD3(pos=[2 * AUTD3.DEVICE_WIDTH, AUTD3.DEVICE_HEIGHT, 0.0], rot=[1, 0, 0, 0]),
        AUTD3(pos=[2 * AUTD3.DEVICE_WIDTH, 2 * AUTD3.DEVICE_HEIGHT, 0.0], rot=[1, 0, 0, 0]),
    ]


def make_circle_offsets(cfg: AppConfig):
    # STM円軌道の8点オフセットを作る
    return np.array(
        [
            [
                cfg.radius * np.cos(np.pi / 8 + 2.0 * np.pi * i / cfg.point_num),
                cfg.radius * np.sin(np.pi / 8 + 2.0 * np.pi * i / cfg.point_num),
                0.0,
            ]
            for i in range(cfg.point_num)
        ],
        dtype=np.float32,
    )


def _get_vec_coord(v, idx: int, name: str) -> float:
    if hasattr(v, name):
        return float(getattr(v, name))
    return float(v[idx])


def get_transducer_xy(dev, tr):
    """
    OutputMask用に振動子のXY座標を取得する。
    """
    if hasattr(tr, "position"):
        p_attr = tr.position
        p = p_attr() if callable(p_attr) else p_attr
        x = _get_vec_coord(p, 0, "x")
        y = _get_vec_coord(p, 1, "y")
        return x, y

    if hasattr(tr, "pos"):
        p_attr = tr.pos
        p = p_attr() if callable(p_attr) else p_attr
        x = _get_vec_coord(p, 0, "x")
        y = _get_vec_coord(p, 1, "y")
        return x, y

    raise AttributeError(
        "Transducer position API not found. "
        "Please check dir(tr) inside OutputMask."
    )


def make_circular_output_mask(center_x: float, center_y: float, radius_mm: float):
    """
    center_x, center_y を中心とした半径 radius_mm 以内の振動子だけONにする。
    """
    cx = float(center_x)
    cy = float(center_y)
    r2 = float(radius_mm) ** 2

    def device_mask(dev):
        def transducer_mask(tr):
            try:
                tx, ty = get_transducer_xy(dev, tr)
                d2 = (tx - cx) ** 2 + (ty - cy) ** 2
                return d2 <= r2
            except Exception:
                return False

        return transducer_mask

    return OutputMask(device_mask)


@dataclass
class TargetCommand:
    x: float
    y: float
    z: float
    mask_center_x: float | None = None
    mask_center_y: float | None = None


class AutdSender:
    """
    AUTD送信専用クラス。
    これにより shared_target_pos, shared_target_seq, pos_lock などを隠蔽する。
    """

    def __init__(self, autd, cfg: AppConfig):
        self.autd = autd
        self.cfg = cfg
        self.circle_offsets = make_circle_offsets(cfg)

        self._lock = threading.Lock()
        self._target: TargetCommand | None = None
        self._seq = 0

        self._running = threading.Event()
        self._thread: threading.Thread | None = None

        self.display_fps = 0.0
        self.build_time_ema_ms = 0.0
        self.send_time_ema_ms = 0.0

        self._last_mask_center_x = None
        self._last_mask_center_y = None

    def set_target(
        self,
        x: float,
        y: float,
        z: float,
        mask_center_x: float | None = None,
        mask_center_y: float | None = None,
    ):
        """
        メインスレッドから呼ぶ。
        STM中心と、必要ならOutputMask中心を更新する。
        """
        with self._lock:
            self._target = TargetCommand(
                x=float(x),
                y=float(y),
                z=float(z),
                mask_center_x=None if mask_center_x is None else float(mask_center_x),
                mask_center_y=None if mask_center_y is None else float(mask_center_y),
            )
            self._seq += 1

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return

        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        self._running.clear()

        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _get_latest_target(self):
        with self._lock:
            return self._target, self._seq

    def _send_output_mask_if_needed(self, target: TargetCommand):
        """
        OutputMaskを使う場合だけ、必要に応じて再送する。
        """
        if not self.cfg.use_output_mask:
            if self._last_mask_center_x is not None:
                self.autd.send(OutputMask(lambda _dev: lambda _tr: True))
                self._last_mask_center_x = None
                self._last_mask_center_y = None
            return

        if target.mask_center_x is None or target.mask_center_y is None:
            return

        mask_x = float(target.mask_center_x)
        mask_y = float(target.mask_center_y)

        need_update_mask = (
            self._last_mask_center_x is None
            or self._last_mask_center_y is None
            or np.hypot(mask_x - self._last_mask_center_x, mask_y - self._last_mask_center_y)
            >= self.cfg.output_mask_update_eps_mm
        )

        if need_update_mask:
            self.autd.send(
                make_circular_output_mask(
                    mask_x,
                    mask_y,
                    self.cfg.output_mask_radius_mm,
                )
            )
            self._last_mask_center_x = mask_x
            self._last_mask_center_y = mask_y

    def _loop(self):
        print("[THREAD] AUTD Control Thread Started.")

        fps_start_time = time.perf_counter()
        fps_frame_count = 0
        last_seq = -1

        while self._running.is_set():
            target, seq = self._get_latest_target()

            if target is None or seq == last_seq:
                time.sleep(self.cfg.autd_loop_sleep_sec)
                continue

            try:
                t0 = time.perf_counter()

                center_vec = np.array([target.x, target.y, target.z], dtype=np.float32)
                foci = center_vec[None, :] + self.circle_offsets

                stm = FociSTM(
                    foci=[foci[i] for i in range(foci.shape[0])],
                    config=self.cfg.stm_freq_hz * Hz,
                ).into_nearest()

                t1 = time.perf_counter()

                self._send_output_mask_if_needed(target)
                self.autd.send(stm)

                t2 = time.perf_counter()

                build_ms = (t1 - t0) * 1000.0
                send_ms = (t2 - t1) * 1000.0

                self.build_time_ema_ms = 0.9 * self.build_time_ema_ms + 0.1 * build_ms
                self.send_time_ema_ms = 0.9 * self.send_time_ema_ms + 0.1 * send_ms

                last_seq = seq
                fps_frame_count += 1

            except Exception as e:
                print(f"[AUTD Thread Error] {e}")
                time.sleep(0.01)

            now = time.perf_counter()
            if now - fps_start_time >= 1.0:
                self.display_fps = fps_frame_count / (now - fps_start_time)

                print(
                    f"[AUTD] fps={self.display_fps:.1f}, "
                    f"build_ema={self.build_time_ema_ms:.2f} ms, "
                    f"send_ema={self.send_time_ema_ms:.2f} ms"
                )

                fps_start_time = now
                fps_frame_count = 0

        print("[THREAD] AUTD Control Thread Stopped.")