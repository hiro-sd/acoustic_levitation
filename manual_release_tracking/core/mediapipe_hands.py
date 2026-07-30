from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


THUMB_TIP_INDEX = 4
INDEX_FINGER_TIP_INDEX = 8
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class HandTipConfig:
    max_input_width_px: int = 640
    fingertip_gap_normalized_max: float = 0.45
    min_opposition_angle_deg: float = 60.0
    min_hand_detection_confidence: float = 0.50
    min_hand_presence_confidence: float = 0.50
    min_tracking_confidence: float = 0.50
    pending_limit: int = 128


@dataclass(frozen=True)
class BallFrameMetadata:
    frame_width: int
    frame_height: int
    ball_center: tuple[float, float] | None
    ball_radius_px: float | None


@dataclass(frozen=True)
class HandTipObservation:
    sequence: int
    timestamp_ms: int
    received_time_ms: float
    result_latency_ms: float
    hand_detected: bool
    ball_detected: bool
    thumb_tip_px: tuple[float, float] | None
    index_tip_px: tuple[float, float] | None
    thumb_gap_px: float | None
    index_gap_px: float | None
    thumb_gap_normalized: float | None
    index_gap_normalized: float | None
    opposition_angle_deg: float | None
    contact_candidate: bool
    ball_center_px: tuple[float, float] | None
    ball_radius_px: float | None

    @property
    def valid_for_decision(self) -> bool:
        return self.ball_detected


def _surface_gap(
    point: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> tuple[float, float]:
    distance = float(np.hypot(point[0] - center[0], point[1] - center[1]))
    gap_px = max(0.0, distance - radius)
    return gap_px, gap_px / max(1.0, radius)


def _opposition_angle_deg(
    thumb: tuple[float, float],
    index: tuple[float, float],
    center: tuple[float, float],
) -> float | None:
    thumb_vec = np.asarray(thumb, dtype=float) - np.asarray(center, dtype=float)
    index_vec = np.asarray(index, dtype=float) - np.asarray(center, dtype=float)
    denominator = float(np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec))
    if denominator <= 1e-6:
        return None
    cosine = float(np.dot(thumb_vec, index_vec) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def build_hand_tip_observation(
    *,
    sequence: int,
    timestamp_ms: int,
    received_time_ms: float,
    metadata: BallFrameMetadata,
    thumb_tip_normalized: tuple[float, float] | None,
    index_tip_normalized: tuple[float, float] | None,
    cfg: HandTipConfig,
) -> HandTipObservation:
    has_hand = (
        thumb_tip_normalized is not None
        and index_tip_normalized is not None
    )
    has_ball = metadata.ball_center is not None and metadata.ball_radius_px is not None

    thumb_px = None
    index_px = None
    thumb_gap_px = None
    index_gap_px = None
    thumb_gap_norm = None
    index_gap_norm = None
    opposition_angle = None
    contact_candidate = False

    if has_hand:
        thumb_px = (
            float(thumb_tip_normalized[0] * metadata.frame_width),
            float(thumb_tip_normalized[1] * metadata.frame_height),
        )
        index_px = (
            float(index_tip_normalized[0] * metadata.frame_width),
            float(index_tip_normalized[1] * metadata.frame_height),
        )

    if has_hand and has_ball:
        center = metadata.ball_center
        radius = float(metadata.ball_radius_px)
        thumb_gap_px, thumb_gap_norm = _surface_gap(thumb_px, center, radius)
        index_gap_px, index_gap_norm = _surface_gap(index_px, center, radius)
        opposition_angle = _opposition_angle_deg(thumb_px, index_px, center)
        contact_candidate = bool(
            thumb_gap_norm <= cfg.fingertip_gap_normalized_max
            and index_gap_norm <= cfg.fingertip_gap_normalized_max
            and opposition_angle is not None
            and opposition_angle >= cfg.min_opposition_angle_deg
        )

    return HandTipObservation(
        sequence=int(sequence),
        timestamp_ms=int(timestamp_ms),
        received_time_ms=float(received_time_ms),
        result_latency_ms=max(0.0, float(received_time_ms - timestamp_ms)),
        hand_detected=bool(has_hand),
        ball_detected=bool(has_ball),
        thumb_tip_px=thumb_px,
        index_tip_px=index_px,
        thumb_gap_px=thumb_gap_px,
        index_gap_px=index_gap_px,
        thumb_gap_normalized=thumb_gap_norm,
        index_gap_normalized=index_gap_norm,
        opposition_angle_deg=opposition_angle,
        contact_candidate=contact_candidate,
        ball_center_px=metadata.ball_center,
        ball_radius_px=metadata.ball_radius_px,
    )


class AsyncHandLandmarker:
    """One MediaPipe LIVE_STREAM Hand Landmarker with latest-result semantics."""

    def __init__(
        self,
        role: str,
        model_path: str | Path,
        cfg: HandTipConfig | None = None,
    ):
        self.role = str(role)
        self.cfg = cfg or HandTipConfig()
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe hand model not found: {self.model_path}\n"
                f"Download {HAND_LANDMARKER_MODEL_URL}\n"
                f"and save it as {self.model_path}"
            )

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Install the optional dependency with:\n"
                "  python -m pip install -r "
                "manual_release_tracking/requirements-mediapipe.txt"
            ) from exc

        self._mp = mp
        self._lock = threading.Lock()
        self._pending: OrderedDict[int, BallFrameMetadata] = OrderedDict()
        self._latest: HandTipObservation | None = None
        self._sequence = 0
        self._last_submit_timestamp_ms = -1
        self._callback_times: deque[float] = deque()
        self._callback_fps = 0.0

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path),
            ),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=self.cfg.min_hand_detection_confidence,
            min_hand_presence_confidence=self.cfg.min_hand_presence_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
            result_callback=self._on_result,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    @property
    def callback_fps(self) -> float:
        with self._lock:
            return float(self._callback_fps)

    def latest(self) -> HandTipObservation | None:
        with self._lock:
            return self._latest

    def submit(
        self,
        frame_bgr: np.ndarray,
        capture_time_sec: float,
        ball_center: tuple[float, float] | None,
        ball_radius_px: float | None,
    ) -> int:
        height, width = frame_bgr.shape[:2]
        timestamp_ms = int(round(float(capture_time_sec) * 1000.0))
        with self._lock:
            timestamp_ms = max(timestamp_ms, self._last_submit_timestamp_ms + 1)
            self._last_submit_timestamp_ms = timestamp_ms
            self._pending[timestamp_ms] = BallFrameMetadata(
                frame_width=int(width),
                frame_height=int(height),
                ball_center=ball_center,
                ball_radius_px=ball_radius_px,
            )
            while len(self._pending) > self.cfg.pending_limit:
                self._pending.popitem(last=False)

        submit_frame = frame_bgr
        if self.cfg.max_input_width_px > 0 and width > self.cfg.max_input_width_px:
            scale = self.cfg.max_input_width_px / width
            submit_frame = cv2.resize(
                frame_bgr,
                (
                    self.cfg.max_input_width_px,
                    max(1, int(round(height * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )

        frame_rgb = cv2.cvtColor(submit_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        try:
            self._landmarker.detect_async(mp_image, timestamp_ms)
        except Exception:
            with self._lock:
                self._pending.pop(timestamp_ms, None)
            raise
        return timestamp_ms

    def _on_result(self, result, _output_image, timestamp_ms: int):
        now_sec = time.perf_counter()
        now_ms = now_sec * 1000.0
        with self._lock:
            metadata = self._pending.pop(int(timestamp_ms), None)
            if metadata is None:
                return

            self._sequence += 1
            thumb = None
            index = None
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                if len(landmarks) > INDEX_FINGER_TIP_INDEX:
                    thumb_landmark = landmarks[THUMB_TIP_INDEX]
                    index_landmark = landmarks[INDEX_FINGER_TIP_INDEX]
                    thumb = (float(thumb_landmark.x), float(thumb_landmark.y))
                    index = (float(index_landmark.x), float(index_landmark.y))

            self._latest = build_hand_tip_observation(
                sequence=self._sequence,
                timestamp_ms=int(timestamp_ms),
                received_time_ms=now_ms,
                metadata=metadata,
                thumb_tip_normalized=thumb,
                index_tip_normalized=index,
                cfg=self.cfg,
            )

            self._callback_times.append(now_sec)
            while self._callback_times and now_sec - self._callback_times[0] > 1.0:
                self._callback_times.popleft()
            if len(self._callback_times) >= 2:
                elapsed = max(
                    1e-6,
                    self._callback_times[-1] - self._callback_times[0],
                )
                self._callback_fps = (
                    len(self._callback_times) - 1
                ) / elapsed
            else:
                self._callback_fps = 0.0

    def close(self):
        self._landmarker.close()


class TipContactState:
    WAITING_FOR_GRASP = "WAITING_FOR_GRASP"
    GRASPED = "GRASPED"
    RELEASED = "RELEASED"


@dataclass(frozen=True)
class TipContactStateConfig:
    grasp_confirm_ms: float = 50.0
    release_confirm_ms: float = 35.0
    released_latch_ms: float = 500.0


class TipContactStateMachine:
    """Timestamp-based preview state machine for MediaPipe contact candidates."""

    def __init__(self, cfg: TipContactStateConfig | None = None):
        self.cfg = cfg or TipContactStateConfig()
        self.state = TipContactState.WAITING_FOR_GRASP
        self.transition_reason = ""
        self.just_released = False
        self._candidate_since_ms: float | None = None
        self._released_at_ms: float | None = None

    def reset(self):
        self.state = TipContactState.WAITING_FOR_GRASP
        self.transition_reason = "manual_reset"
        self.just_released = False
        self._candidate_since_ms = None
        self._released_at_ms = None

    def update(
        self,
        timestamp_ms: float,
        contact_candidate: bool,
        valid: bool = True,
    ) -> str:
        self.transition_reason = ""
        self.just_released = False
        timestamp_ms = float(timestamp_ms)

        if not valid:
            self._candidate_since_ms = None
            return self.state

        if self.state == TipContactState.WAITING_FOR_GRASP:
            if contact_candidate:
                if self._candidate_since_ms is None:
                    self._candidate_since_ms = timestamp_ms
                if (
                    timestamp_ms - self._candidate_since_ms
                    >= self.cfg.grasp_confirm_ms
                ):
                    self.state = TipContactState.GRASPED
                    self.transition_reason = "fingertips_near_sphere"
                    self._candidate_since_ms = None
            else:
                self._candidate_since_ms = None

        elif self.state == TipContactState.GRASPED:
            if not contact_candidate:
                if self._candidate_since_ms is None:
                    self._candidate_since_ms = timestamp_ms
                if (
                    timestamp_ms - self._candidate_since_ms
                    >= self.cfg.release_confirm_ms
                ):
                    self.state = TipContactState.RELEASED
                    self.transition_reason = "fingertips_left_sphere"
                    self.just_released = True
                    self._released_at_ms = timestamp_ms
                    self._candidate_since_ms = None
            else:
                self._candidate_since_ms = None

        elif self.state == TipContactState.RELEASED:
            if (
                self._released_at_ms is not None
                and timestamp_ms - self._released_at_ms
                >= self.cfg.released_latch_ms
            ):
                self.state = TipContactState.WAITING_FOR_GRASP
                self.transition_reason = "ready_for_regrasp"
                self._released_at_ms = None
                self._candidate_since_ms = None

        return self.state


@dataclass
class BinaryErrorStats:
    samples: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def update(self, predicted_grasped: bool, actual_grasped: bool):
        self.samples += 1
        if predicted_grasped and not actual_grasped:
            self.false_positive += 1
        elif actual_grasped and not predicted_grasped:
            self.false_negative += 1

    @property
    def errors(self) -> int:
        return self.false_positive + self.false_negative

    @property
    def error_rate(self) -> float | None:
        if self.samples <= 0:
            return None
        return self.errors / self.samples

    def reset(self):
        self.samples = 0
        self.false_positive = 0
        self.false_negative = 0
