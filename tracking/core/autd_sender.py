import csv
import time
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autd3.commands import FociStm, Modulation, SetOutputMask
from autd3.geometry import Autd3, Geometry
from autd3.units import Hz
from autd3.value import ControlPoint, ControlPoints, Nearest, SamplingConfig
from autd3_modulation import constant, modulation_buffer

from .config import AppConfig
from .models import HomePosition
from .weighted_stm import (
    cycle_frequency_preserving_slot_rate,
    expand_focus_offsets,
)


def make_autd_arrangement() -> Geometry:
    # 3行×3列のAUTD配置
    return Geometry(
        [
            Autd3([0.0, 0.0, 0.0]),
            Autd3([0.0, Autd3.DEVICE_HEIGHT, 0.0]),
            Autd3([0.0, 2 * Autd3.DEVICE_HEIGHT, 0.0]),

            Autd3([Autd3.DEVICE_WIDTH, 2 * Autd3.DEVICE_HEIGHT, 0.0]),
            Autd3([Autd3.DEVICE_WIDTH, Autd3.DEVICE_HEIGHT, 0.0]),
            Autd3([Autd3.DEVICE_WIDTH, 0.0, 0.0]),

            Autd3([2 * Autd3.DEVICE_WIDTH, 0.0, 0.0]),
            Autd3([2 * Autd3.DEVICE_WIDTH, Autd3.DEVICE_HEIGHT, 0.0]),
            Autd3([2 * Autd3.DEVICE_WIDTH, 2 * Autd3.DEVICE_HEIGHT, 0.0]),
        ]
    )


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


def make_circular_output_mask(
    geometry: Geometry,
    center_x: float,
    center_y: float,
    radius_mm: float,
) -> SetOutputMask:
    """
    center_x, center_y を中心とした半径 radius_mm 以内の振動子だけONにする。
    """
    cx = float(center_x)
    cy = float(center_y)
    r2 = float(radius_mm) ** 2

    masks = []
    for device in geometry:
        positions = device.positions()
        d2 = (positions[:, 0] - cx) ** 2 + (positions[:, 1] - cy) ** 2
        masks.append((d2 <= r2).tolist())

    return SetOutputMask(masks)


def make_constant_modulation(intensity_ratio: float) -> Modulation:
    ratio = float(np.clip(intensity_ratio, 0.0, 1.0))
    buffer = modulation_buffer()
    constant(quantize_intensity_ratio(ratio)[0], buffer)
    return Modulation(SamplingConfig.FREQ_4K, buffer)


def quantize_intensity_ratio(intensity_ratio: float) -> tuple[int, float]:
    """Return the exact 8-bit modulation command and its normalized value."""
    modulation_byte = int(0xFF * float(np.clip(intensity_ratio, 0.0, 1.0)))
    return modulation_byte, modulation_byte / 255.0


@dataclass
class TargetCommand:
    x: float
    y: float
    z: float
    enqueued_time_sec: float
    mask_center_x: float | None = None
    mask_center_y: float | None = None
    radius: float | None = None
    intensity_ratio: float | None = None
    focus_dwell_counts: tuple[int, ...] | None = None
    measurement_time_sec: float | None = None
    measurement_age_sec: float | None = None
    estimated_actuation_delay_sec: float | None = None
    prediction_horizon_sec: float | None = None
    control_mode: str = ""
    release_timestamp_ms: int | None = None
    is_initial_capture_command: bool = False


@dataclass(frozen=True)
class SentCommandTelemetry:
    sequence: int
    dequeued_time_sec: float
    send_started_time_sec: float
    sent_time_sec: float
    target: TargetCommand
    modulation_byte: int | None = None
    quantized_intensity_ratio: float | None = None


AUTD_UPDATE_LOG_HEADER = [
    "session_id",
    "wall_time_sec",
    "command_sequence",
    "control_mode",
    "release_timestamp_ms",
    "enqueued_time_sec",
    "dequeued_time_sec",
    "send_started_time_sec",
    "send_completed_time_sec",
    "enqueue_to_send_completed_ms",
    "measurement_time_sec",
    "measurement_age_ms",
    "measurement_age_at_send_completed_ms",
    "estimated_actuation_delay_ms",
    "prediction_horizon_ms",
    "prediction_horizon_shortfall_ms",
    "target_x_mm",
    "target_y_mm",
    "target_z_mm",
    "requested_intensity_ratio",
    "modulation_byte",
    "quantized_intensity_ratio",
]


class AutdSender:
    """
    AUTD送信専用クラス。
    これにより shared_target_pos, shared_target_seq, pos_lock などを隠蔽する。
    """

    def __init__(self, autd, cfg: AppConfig):
        self.autd = autd
        self.geometry = autd.geometry
        self.cfg = cfg
        self._last_radius = float(cfg.radius)
        self.circle_offsets = make_circle_offsets(cfg, self._last_radius)

        self._lock = threading.Lock()
        self._target: TargetCommand | None = None
        self._seq = 0
        self._last_sent: SentCommandTelemetry | None = None
        self._sent_history: deque[SentCommandTelemetry] = deque(maxlen=256)

        self._running = threading.Event()
        self._thread: threading.Thread | None = None

        self.display_fps = 0.0
        self.build_time_ema_ms = 0.0
        self.send_time_ema_ms = 0.0
        self.command_latency_ema_ms: float | None = None
        self.initial_capture_latency_ema_ms: float | None = None

        self._last_mask_center_x = None
        self._last_mask_center_y = None

        self._last_intensity_ratio = None
        self._last_modulation_byte: int | None = None
        self._update_log_file = None
        self._update_log_writer = None
        self._update_log_rows_since_flush = 0
        self._update_log_last_flush_sec = time.perf_counter()
        self._update_session_id = str(
            getattr(cfg, "autd_update_session_id", "")
        )
        update_log_path = str(getattr(cfg, "autd_update_log_path", "")).strip()
        if update_log_path:
            path = Path(update_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            exists = path.exists() and path.stat().st_size > 0
            if exists:
                with path.open(newline="", encoding="utf-8") as existing_file:
                    existing_header = next(csv.reader(existing_file), [])
                if existing_header != AUTD_UPDATE_LOG_HEADER:
                    raise RuntimeError(
                        f"CSV schema mismatch: {path}. Use a new data directory "
                        "or move the old AUTD update telemetry file."
                    )
            self._update_log_file = path.open("a", newline="", encoding="utf-8")
            self._update_log_writer = csv.writer(self._update_log_file)
            if not exists:
                self._update_log_writer.writerow(AUTD_UPDATE_LOG_HEADER)
                self._update_log_file.flush()

    def set_target(
        self,
        x: float,
        y: float,
        z: float,
        mask_center_x: float | None = None,
        mask_center_y: float | None = None,
        radius: float | None = None,
        intensity_ratio: float | None = None,
        focus_dwell_counts: tuple[int, ...] | None = None,
        measurement_time_sec: float | None = None,
        measurement_age_sec: float | None = None,
        estimated_actuation_delay_sec: float | None = None,
        prediction_horizon_sec: float | None = None,
        control_mode: str = "",
        release_timestamp_ms: int | None = None,
        is_initial_capture_command: bool = False,
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
                enqueued_time_sec=float(time.perf_counter()),
                mask_center_x=None if mask_center_x is None else float(mask_center_x),
                mask_center_y=None if mask_center_y is None else float(mask_center_y),
                radius=None if radius is None else float(radius),
                intensity_ratio=None if intensity_ratio is None else float(intensity_ratio),
                focus_dwell_counts=(
                    None
                    if focus_dwell_counts is None
                    else tuple(int(value) for value in focus_dwell_counts)
                ),
                measurement_time_sec=(
                    None if measurement_time_sec is None else float(measurement_time_sec)
                ),
                measurement_age_sec=(
                    None if measurement_age_sec is None else float(measurement_age_sec)
                ),
                estimated_actuation_delay_sec=(
                    None
                    if estimated_actuation_delay_sec is None
                    else float(estimated_actuation_delay_sec)
                ),
                prediction_horizon_sec=(
                    None
                    if prediction_horizon_sec is None
                    else float(prediction_horizon_sec)
                ),
                control_mode=str(control_mode),
                release_timestamp_ms=(
                    None if release_timestamp_ms is None else int(release_timestamp_ms)
                ),
                is_initial_capture_command=bool(is_initial_capture_command),
            )
            self._seq += 1
            return self._seq

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
        if self._update_log_file is not None:
            self._update_log_file.close()
            self._update_log_file = None
            self._update_log_writer = None

    def _get_latest_target(self):
        with self._lock:
            return self._target, self._seq

    def get_last_sent_telemetry(self) -> SentCommandTelemetry | None:
        with self._lock:
            return self._last_sent

    def get_first_sent_at_or_after(
        self,
        sequence: int,
    ) -> SentCommandTelemetry | None:
        with self._lock:
            for telemetry in self._sent_history:
                if telemetry.sequence >= int(sequence):
                    return telemetry
        return None

    def get_sent_telemetry(self, sequence: int) -> SentCommandTelemetry | None:
        """Return telemetry only when the exact queued command was emitted."""
        with self._lock:
            for telemetry in self._sent_history:
                if telemetry.sequence == int(sequence):
                    return telemetry
        return None

    def _write_update_telemetry(self, telemetry: SentCommandTelemetry):
        if self._update_log_writer is None:
            return
        target = telemetry.target

        def value_or_empty(value, scale=1.0):
            return "" if value is None else float(value) * float(scale)

        self._update_log_writer.writerow(
            [
                self._update_session_id,
                time.time(),
                telemetry.sequence,
                target.control_mode,
                "" if target.release_timestamp_ms is None else target.release_timestamp_ms,
                target.enqueued_time_sec,
                telemetry.dequeued_time_sec,
                telemetry.send_started_time_sec,
                telemetry.sent_time_sec,
                (telemetry.sent_time_sec - target.enqueued_time_sec) * 1000.0,
                value_or_empty(target.measurement_time_sec),
                value_or_empty(target.measurement_age_sec, 1000.0),
                (
                    ""
                    if target.measurement_time_sec is None
                    else (
                        telemetry.sent_time_sec - target.measurement_time_sec
                    )
                    * 1000.0
                ),
                value_or_empty(target.estimated_actuation_delay_sec, 1000.0),
                value_or_empty(target.prediction_horizon_sec, 1000.0),
                (
                    ""
                    if target.measurement_time_sec is None
                    or target.prediction_horizon_sec is None
                    else (
                        telemetry.sent_time_sec
                        - target.measurement_time_sec
                        - target.prediction_horizon_sec
                    )
                    * 1000.0
                ),
                target.x,
                target.y,
                target.z,
                value_or_empty(target.intensity_ratio),
                "" if telemetry.modulation_byte is None else telemetry.modulation_byte,
                value_or_empty(telemetry.quantized_intensity_ratio),
            ]
        )
        self._update_log_rows_since_flush += 1
        now_sec = time.perf_counter()
        if (
            self._update_log_rows_since_flush >= 100
            or now_sec - self._update_log_last_flush_sec >= 1.0
        ):
            self._update_log_file.flush()
            self._update_log_rows_since_flush = 0
            self._update_log_last_flush_sec = now_sec

    def estimated_command_latency_sec(self, fallback_sec: float) -> float:
        """Estimate enqueue-to-send-completion latency from this run.

        Before the first command has completed, the caller-provided fallback is
        used.  Thereafter the estimate follows the latency of commands that were
        actually emitted, so prediction horizons adapt to the current host and
        network rather than assuming a fixed delay for the whole session.
        """
        with self._lock:
            latency_ms = self.command_latency_ema_ms
        if latency_ms is None or not np.isfinite(latency_ms):
            return max(0.0, float(fallback_sec))
        return max(0.0, float(latency_ms) * 1e-3)

    def estimated_initial_capture_latency_sec(self, fallback_sec: float) -> float:
        """Estimate latency using only previously sent initial capture commands.

        Seed the EMA with the measured fallback before the first release.  A
        single first-command queue stall therefore cannot replace the estimate
        outright; it is incorporated with the same 0.1 EMA weight as later
        observations.
        """
        fallback_ms = max(0.0, float(fallback_sec)) * 1000.0
        with self._lock:
            latency_ms = self.initial_capture_latency_ema_ms
            if latency_ms is None or not np.isfinite(latency_ms):
                latency_ms = fallback_ms
                self.initial_capture_latency_ema_ms = latency_ms
        return max(0.0, float(latency_ms) * 1e-3)

    def _record_command_latency_ms(
        self,
        target: TargetCommand,
        command_latency_ms: float,
    ) -> None:
        """Update the global EMA and, when applicable, the initial-only EMA."""
        latency_ms = float(command_latency_ms)
        with self._lock:
            if self.command_latency_ema_ms is None:
                self.command_latency_ema_ms = latency_ms
            else:
                self.command_latency_ema_ms = (
                    0.9 * self.command_latency_ema_ms + 0.1 * latency_ms
                )
            if target.is_initial_capture_command:
                if self.initial_capture_latency_ema_ms is None:
                    self.initial_capture_latency_ema_ms = latency_ms
                else:
                    self.initial_capture_latency_ema_ms = (
                        0.9 * self.initial_capture_latency_ema_ms
                        + 0.1 * latency_ms
                    )

    def _send_output_mask_if_needed(self, target: TargetCommand):
        """
        OutputMaskを使う場合だけ、必要に応じて再送する。
        """
        if not self.cfg.use_output_mask:
            if self._last_mask_center_x is not None:
                self.autd.send(
                    SetOutputMask(self.geometry.output_mask_buffer())
                )
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
                    self.geometry,
                    mask_x,
                    mask_y,
                    self.cfg.output_mask_radius_mm,
                )
            )
            self._last_mask_center_x = mask_x
            self._last_mask_center_y = mask_y

    def _combine_stm_with_intensity_if_needed(self, stm, target: TargetCommand):
        """
        intensity_ratio が変わった場合は、定振幅変調と音場を1回の送信にまとめる。

        戻り値の intensity_ratio は送信成功後にだけ記録する。これにより、
        送信失敗時には次回も同じIntensity更新を再試行できる。
        """
        if target.intensity_ratio is None:
            return stm, None

        ratio = float(np.clip(target.intensity_ratio, 0.0, 1.0))

        need_update = (
            self._last_intensity_ratio is None
            or abs(ratio - self._last_intensity_ratio) >= self.cfg.intensity_update_eps
        )

        if not need_update:
            return stm, None

        return (
            make_constant_modulation(ratio),
            stm,
        ), ratio

    def _send_field_datagram(self, datagram, intensity_ratio: float | None):
        """音場を送信し、成功したIntensityだけを送信済みとして記録する。"""
        self.autd.send(datagram)
        if intensity_ratio is not None:
            self._last_intensity_ratio = float(intensity_ratio)
            self._last_modulation_byte = quantize_intensity_ratio(
                intensity_ratio
            )[0]

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
                sent_intensity_ratio = None

                radius = float(self.cfg.radius if target.radius is None else target.radius)

                if abs(radius - self._last_radius) > 1e-6:
                    self.circle_offsets = make_circle_offsets(self.cfg, radius)
                    self._last_radius = radius

                center_vec = np.array([target.x, target.y, target.z], dtype=np.float32)
                stm_offsets = expand_focus_offsets(
                    self.circle_offsets,
                    target.focus_dwell_counts,
                )
                foci = center_vec[None, :] + stm_offsets

                stm_cycle_frequency_hz = float(self.cfg.stm_freq_hz)
                if target.focus_dwell_counts is not None:
                    # FociStm's frequency is the frequency of one complete
                    # cycle, not the per-focus update rate. A weighted pattern
                    # contains more slots than the normal 8-point STM, so
                    # retain the existing slot period and Silencer setup.
                    stm_cycle_frequency_hz = (
                        cycle_frequency_preserving_slot_rate(
                            base_cycle_frequency_hz=self.cfg.stm_freq_hz,
                            base_point_num=len(self.circle_offsets),
                            total_slots=len(stm_offsets),
                        )
                    )
                stm = FociStm(
                    Nearest(stm_cycle_frequency_hz * Hz),
                    [
                        ControlPoints([ControlPoint(foci[i])])
                        for i in range(foci.shape[0])
                    ],
                )
                datagram, sent_intensity_ratio = (
                    self._combine_stm_with_intensity_if_needed(stm, target)
                )

                t1 = time.perf_counter()

                self._send_output_mask_if_needed(target)
                send_started_time_sec = time.perf_counter()
                self._send_field_datagram(datagram, sent_intensity_ratio)

                t2 = time.perf_counter()
                with self._lock:
                    telemetry = SentCommandTelemetry(
                        sequence=int(seq),
                        dequeued_time_sec=float(t0),
                        send_started_time_sec=float(send_started_time_sec),
                        sent_time_sec=float(t2),
                        target=target,
                        modulation_byte=self._last_modulation_byte,
                        quantized_intensity_ratio=(
                            None
                            if self._last_modulation_byte is None
                            else self._last_modulation_byte / 255.0
                        ),
                    )
                    self._last_sent = telemetry
                    self._sent_history.append(telemetry)
                # The physical send has completed. Mark the sequence before
                # file I/O so a telemetry-disk failure can never resend it.
                last_seq = seq
                try:
                    self._write_update_telemetry(telemetry)
                except Exception as log_error:
                    print(f"[AUTD Telemetry Warning] {log_error}")
                    if self._update_log_file is not None:
                        try:
                            self._update_log_file.close()
                        except Exception:
                            pass
                    self._update_log_file = None
                    self._update_log_writer = None

                build_ms = (t1 - t0) * 1000.0
                send_ms = (t2 - t1) * 1000.0
                command_latency_ms = (
                    t2 - float(target.enqueued_time_sec)
                ) * 1000.0

                self.build_time_ema_ms = 0.9 * self.build_time_ema_ms + 0.1 * build_ms
                self.send_time_ema_ms = 0.9 * self.send_time_ema_ms + 0.1 * send_ms
                self._record_command_latency_ms(target, command_latency_ms)

                fps_frame_count += 1

            except Exception as e:
                print(f"[AUTD Thread Error] {e}")
                time.sleep(0.01)

            now = time.perf_counter()
            if now - fps_start_time >= 1.0:
                self.display_fps = fps_frame_count / (now - fps_start_time)
                command_latency_ema_ms = (
                    0.0
                    if self.command_latency_ema_ms is None
                    else self.command_latency_ema_ms
                )

                print(
                    f"[AUTD] fps={self.display_fps:.1f}, "
                    f"build_ema={self.build_time_ema_ms:.2f} ms, "
                    f"send_ema={self.send_time_ema_ms:.2f} ms, "
                    f"command_latency_ema="
                    f"{command_latency_ema_ms:.2f} ms"
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
    *,
    measurement_time_sec: float | None = None,
    measurement_age_sec: float | None = None,
    estimated_actuation_delay_sec: float | None = None,
    prediction_horizon_sec: float | None = None,
    control_mode: str = "",
    release_timestamp_ms: int | None = None,
    is_initial_capture_command: bool = False,
):
    """Send a target while keeping OutputMask-specific arguments out of app.py."""
    if cfg.use_output_mask:
        return sender.set_target(
            target_x,
            target_y,
            target_z,
            home.x,
            home.y,
            radius=radius,
            intensity_ratio=intensity_ratio,
            measurement_time_sec=measurement_time_sec,
            measurement_age_sec=measurement_age_sec,
            estimated_actuation_delay_sec=estimated_actuation_delay_sec,
            prediction_horizon_sec=prediction_horizon_sec,
            control_mode=control_mode,
            release_timestamp_ms=release_timestamp_ms,
            is_initial_capture_command=is_initial_capture_command,
        )
    else:
        return sender.set_target(
            target_x,
            target_y,
            target_z,
            radius=radius,
            intensity_ratio=intensity_ratio,
            measurement_time_sec=measurement_time_sec,
            measurement_age_sec=measurement_age_sec,
            estimated_actuation_delay_sec=estimated_actuation_delay_sec,
            prediction_horizon_sec=prediction_horizon_sec,
            control_mode=control_mode,
            release_timestamp_ms=release_timestamp_ms,
            is_initial_capture_command=is_initial_capture_command,
        )
