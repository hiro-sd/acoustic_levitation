import time
import threading
from dataclasses import dataclass

import numpy as np

from pyautd3 import AUTD3, FociSTM, Hz, Intensity, OutputMask, Static
from pyautd3.gain.holo import GSPAT, EmissionConstraint, GSPATOption, Pa

from .config import AppConfig
from .models import HomePosition


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


def make_circle_offsets(cfg: AppConfig, radius: float | None = None):
    """
    STM円軌道のオフセットを作る。
    radiusを指定しない場合はcfg.radiusを使う。
    """
    r = float(cfg.radius if radius is None else radius)

    return np.array(
        [
            [
                r * np.cos(np.pi / 8 + 2.0 * np.pi * i / cfg.point_num),
                r * np.sin(np.pi / 8 + 2.0 * np.pi * i / cfg.point_num),
                0.0,
            ]
            for i in range(cfg.point_num)
        ],
        dtype=np.float32,
    )


def make_static_multi_focus_gain(
    cfg: AppConfig,
    center: np.ndarray,
    offsets: np.ndarray,
):
    """
    円周上に複数焦点を同時生成する静的な多焦点ゲインを作る。

    従来のFociSTMは「単焦点を時間的に8点周回」させるのに対し、
    このモードではGSPATで「8点を同時に存在」させる。
    """
    foci = [
        (
            center + offsets[i],
            float(cfg.multi_focus_pressure_pa) * Pa,
        )
        for i in range(offsets.shape[0])
    ]

    return GSPAT(
        foci=foci,
        option=GSPATOption(
            repeat=int(cfg.multi_focus_gspat_repeat),
            constraint=EmissionConstraint.Clamp(Intensity.MIN, Intensity.MAX),
        ),
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
    radius: float | None = None
    intensity_ratio: float | None = None


class AutdSender:
    """
    AUTD送信専用クラス。
    これにより shared_target_pos, shared_target_seq, pos_lock などを隠蔽する。
    """

    def __init__(self, autd, cfg: AppConfig):
        self.autd = autd
        self.cfg = cfg
        self._last_radius = float(cfg.radius)
        self.circle_offsets = make_circle_offsets(cfg, self._last_radius)

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

        self._last_intensity_ratio = None

    def set_target(
    self,
    x: float,
    y: float,
    z: float,
    mask_center_x: float | None = None,
    mask_center_y: float | None = None,
    radius: float | None = None,
    intensity_ratio: float | None = None,
    ):
        """
        メインスレッドから呼ぶ。
        STM中心、OutputMask中心、STM円軌道半径、出力強度を更新する。
        """
        with self._lock:
            self._target = TargetCommand(
                x=float(x),
                y=float(y),
                z=float(z),
                mask_center_x=None if mask_center_x is None else float(mask_center_x),
                mask_center_y=None if mask_center_y is None else float(mask_center_y),
                radius=None if radius is None else float(radius),
                intensity_ratio=None if intensity_ratio is None else float(intensity_ratio),
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

    def _send_intensity_if_needed(self, target: TargetCommand):
        """
        intensity_ratio が変わったときだけ Static を再送する。
        """ 
        if self.cfg.autd_field_mode == "static_multi_focus_circle":
            # 多焦点Staticモードでは Static と GSPAT を同時に送るため、
            # ここでStaticだけを単独再送しない。
            return

        if target.intensity_ratio is None:
            return

        ratio = float(np.clip(target.intensity_ratio, 0.0, 1.0))

        need_update = (
            self._last_intensity_ratio is None
            or abs(ratio - self._last_intensity_ratio) >= self.cfg.intensity_update_eps
        )

        if not need_update:
            return

        self.autd.send(
            Static(
                intensity=int(0xFF * ratio)
            )
        )

        self._last_intensity_ratio = ratio

    def _loop(self):
        print("[THREAD] AUTD Control Thread Started.")
        print(f"[AUTD] field_mode={self.cfg.autd_field_mode}")

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

                radius = float(self.cfg.radius if target.radius is None else target.radius)

                if abs(radius - self._last_radius) > 1e-6:
                    self.circle_offsets = make_circle_offsets(self.cfg, radius)
                    self._last_radius = radius

                center_vec = np.array([target.x, target.y, target.z], dtype=np.float32)
                foci = center_vec[None, :] + self.circle_offsets

                if self.cfg.autd_field_mode == "stm_circle":
                    datagram = FociSTM(
                        foci=[foci[i] for i in range(foci.shape[0])],
                        config=self.cfg.stm_freq_hz * Hz,
                    ).into_nearest()

                elif self.cfg.autd_field_mode == "static_multi_focus_circle":
                    gain = make_static_multi_focus_gain(
                        self.cfg,
                        center_vec,
                        self.circle_offsets,
                    )
                    intensity_ratio = (
                        self.cfg.static_intensity_ratio
                        if target.intensity_ratio is None
                        else target.intensity_ratio
                    )
                    datagram = (
                        Static(intensity=int(0xFF * float(np.clip(intensity_ratio, 0.0, 1.0)))),
                        gain,
                    )
                    self._last_intensity_ratio = float(np.clip(intensity_ratio, 0.0, 1.0))

                else:
                    raise ValueError(
                        "Unknown cfg.autd_field_mode: "
                        f"{self.cfg.autd_field_mode!r}. "
                        "Use 'stm_circle' or 'static_multi_focus_circle'."
                    )

                t1 = time.perf_counter()

                self._send_output_mask_if_needed(target)
                self._send_intensity_if_needed(target)
                self.autd.send(datagram)

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


def set_tracking_target(
    sender: AutdSender,
    cfg: AppConfig,
    target_x: float,
    target_y: float,
    target_z: float,
    home: HomePosition,
    radius: float | None = None,
    intensity_ratio: float | None = None,
):
    """Send a target while keeping OutputMask-specific arguments out of app.py."""
    if cfg.use_output_mask:
        sender.set_target(
            target_x,
            target_y,
            target_z,
            home.x,
            home.y,
            radius=radius,
            intensity_ratio=intensity_ratio,
        )
    else:
        sender.set_target(
            target_x,
            target_y,
            target_z,
            radius=radius,
            intensity_ratio=intensity_ratio,
        )
