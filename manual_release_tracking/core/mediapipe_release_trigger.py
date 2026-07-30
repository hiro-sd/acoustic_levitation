from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .mediapipe_hands import (
    AsyncHandLandmarker,
    HandTipConfig,
    HandTipObservation,
    TipContactState,
    TipContactStateConfig,
    TipContactStateMachine,
)


@dataclass(frozen=True)
class AutoReleaseEvent:
    """One-shot event emitted on the GRASPED -> RELEASED transition."""

    timestamp_ms: int
    result_age_ms: float
    result_pair_skew_ms: float


@dataclass(frozen=True)
class AutoReleaseStatus:
    xy_state: str
    z_state: str
    both_state: str
    decision_camera: str
    decision_state: str
    result_age_xy_ms: float | None
    result_age_z_ms: float | None
    result_pair_skew_ms: float | None
    candidate_event: AutoReleaseEvent | None = None
    candidate_cancelled: bool = False
    release_confirmed: bool = False
    event: AutoReleaseEvent | None = None


def fused_contact_candidate(
    state: str,
    xy_contact: bool,
    z_contact: bool,
) -> bool:
    """
    Fuse the two views asymmetrically for reliable automatic control.

    A grasp must first be visible in both cameras. Once grasped, either camera
    still seeing contact keeps the grasp alive, so release requires explicit
    separation in both cameras.
    """
    if state == TipContactState.WAITING_FOR_GRASP:
        return bool(xy_contact and z_contact)
    return bool(xy_contact or z_contact)


def observation_is_explicit(
    observation: HandTipObservation,
) -> bool:
    """Missing hand landmarks are unknown, not evidence of release."""
    return bool(
        observation.valid_for_decision
        and observation.hand_detected
    )


def should_stop_automatic_hold(
    auto_hold_active: bool,
    both_state: str,
) -> bool:
    """Only a re-grasp can cancel a hold started by automatic release."""
    return bool(
        auto_hold_active
        and both_state == TipContactState.GRASPED
    )


def normalize_decision_camera(value: str) -> str:
    camera = str(value).strip().lower()
    if camera not in {"xy", "z", "both"}:
        raise ValueError(
            "mediapipe_auto_release_camera must be 'xy', 'z', or 'both', "
            f"got {value!r}"
        )
    return camera


def build_auto_release_event(
    *,
    just_released: bool,
    event_timestamp_ms: int,
    now_ms: float,
    pair_skew_ms: float,
    maximum_age_ms: float,
) -> AutoReleaseEvent | None:
    """Apply the final freshness gate to a one-shot release transition."""
    event_age_ms = max(0.0, float(now_ms) - int(event_timestamp_ms))
    if not just_released or event_age_ms > float(maximum_age_ms):
        return None
    return AutoReleaseEvent(
        timestamp_ms=int(event_timestamp_ms),
        result_age_ms=float(event_age_ms),
        result_pair_skew_ms=float(pair_skew_ms),
    )


def update_release_candidate(
    *,
    candidate_active: bool,
    previous_state: str,
    just_released: bool,
    explicit_observation: bool,
    contact_candidate: bool,
    timestamp_ms: int,
    now_ms: float,
    pair_skew_ms: float,
    maximum_age_ms: float,
) -> tuple[bool, AutoReleaseEvent | None, bool, bool]:
    """
    Convert the first explicit separation into an early one-shot event.

    Returns ``(active, candidate_event, candidate_cancelled,
    release_confirmed)``. Missing landmarks leave the candidate unchanged.
    """
    active = bool(candidate_active)
    candidate_event = None
    candidate_cancelled = False
    release_confirmed = bool(just_released)

    if explicit_observation:
        if previous_state == TipContactState.GRASPED and not contact_candidate:
            if not active:
                candidate_event = build_auto_release_event(
                    just_released=True,
                    event_timestamp_ms=timestamp_ms,
                    now_ms=now_ms,
                    pair_skew_ms=pair_skew_ms,
                    maximum_age_ms=maximum_age_ms,
                )
            active = True
        elif active and contact_candidate:
            active = False
            candidate_cancelled = True

    if release_confirmed:
        active = False

    return active, candidate_event, candidate_cancelled, release_confirmed


class MediaPipeReleaseTrigger:
    """
    Non-blocking MediaPipe release trigger for the manual hold application.

    It only decides when a confirmed two-camera grasp changes to release.
    AUTD control remains owned by ``manual_release_app``.
    """

    def __init__(self, cfg):
        model_path = Path(
            getattr(
                cfg,
                "mediapipe_hand_model_path",
                "./manual_release_tracking/models/hand_landmarker.task",
            )
        )
        hand_cfg = HandTipConfig(
            max_input_width_px=int(
                getattr(cfg, "mediapipe_input_width_px", 640)
            ),
            fingertip_gap_normalized_max=float(
                getattr(cfg, "mediapipe_tip_gap_normalized_max", 0.45)
            ),
            min_opposition_angle_deg=float(
                getattr(cfg, "mediapipe_min_opposition_angle_deg", 60.0)
            ),
        )
        state_cfg = TipContactStateConfig(
            grasp_confirm_ms=float(
                getattr(cfg, "mediapipe_grasp_confirm_ms", 50.0)
            ),
            release_confirm_ms=float(
                getattr(cfg, "mediapipe_release_confirm_ms", 35.0)
            ),
            released_latch_ms=float(
                getattr(cfg, "mediapipe_released_latch_ms", 500.0)
            ),
        )

        self.decision_camera = normalize_decision_camera(
            getattr(cfg, "mediapipe_auto_release_camera", "both")
        )
        self.detector_xy = AsyncHandLandmarker("xy", model_path, hand_cfg)
        self.detector_z = None
        if self.decision_camera in {"z", "both"}:
            try:
                self.detector_z = AsyncHandLandmarker("z", model_path, hand_cfg)
            except Exception:
                self.detector_xy.close()
                raise

        self.machines = {
            "xy": TipContactStateMachine(state_cfg),
            "z": TipContactStateMachine(state_cfg),
            "both": TipContactStateMachine(state_cfg),
        }
        self.states = {
            name: TipContactState.WAITING_FOR_GRASP
            for name in self.machines
        }
        self.result_pair_tolerance_ms = float(
            getattr(cfg, "mediapipe_result_pair_tolerance_ms", 40.0)
        )
        self.maximum_trigger_age_ms = float(
            getattr(cfg, "mediapipe_auto_release_max_result_age_ms", 80.0)
        )
        submit_max_fps = float(
            getattr(cfg, "mediapipe_submit_max_fps", 60.0)
        )
        self.submit_interval_sec = (
            0.0 if submit_max_fps <= 0.0 else 1.0 / submit_max_fps
        )

        self.last_submit_xy_time = float("-inf")
        self.last_submit_z_time = float("-inf")
        self.last_xy_sequence = 0
        self.last_z_sequence = 0
        self.last_both_sequences = (0, 0)
        self.latest_xy: HandTipObservation | None = None
        self.latest_z: HandTipObservation | None = None
        self._release_candidate_active = False
        self.status = AutoReleaseStatus(
            xy_state=self.states["xy"],
            z_state=self.states["z"],
            both_state=self.states["both"],
            decision_camera=self.decision_camera,
            decision_state=self.states[self.decision_camera],
            result_age_xy_ms=None,
            result_age_z_ms=None,
            result_pair_skew_ms=None,
        )
        self._closed = False

    @staticmethod
    def _ball_metadata(detection):
        if detection.detected:
            return detection.center, detection.radius_px
        return None, None

    def update(
        self,
        *,
        frame_xy: np.ndarray,
        frame_xy_time: float,
        ball_xy,
        frame_z: np.ndarray,
        frame_z_time: float,
        ball_z,
    ) -> AutoReleaseStatus:
        if frame_xy_time - self.last_submit_xy_time >= self.submit_interval_sec:
            center, radius = self._ball_metadata(ball_xy)
            self.detector_xy.submit(
                frame_xy,
                frame_xy_time,
                center,
                radius,
            )
            self.last_submit_xy_time = frame_xy_time

        if (
            self.detector_z is not None
            and frame_z_time - self.last_submit_z_time >= self.submit_interval_sec
        ):
            center, radius = self._ball_metadata(ball_z)
            self.detector_z.submit(
                frame_z,
                frame_z_time,
                center,
                radius,
            )
            self.last_submit_z_time = frame_z_time

        self.latest_xy = self.detector_xy.latest()
        self.latest_z = (
            None if self.detector_z is None else self.detector_z.latest()
        )

        xy_updated = False
        xy_previous_state = self.machines["xy"].state
        if (
            self.latest_xy is not None
            and self.latest_xy.sequence != self.last_xy_sequence
        ):
            xy_updated = True
            self.last_xy_sequence = self.latest_xy.sequence
            self.states["xy"] = self.machines["xy"].update(
                self.latest_xy.timestamp_ms,
                self.latest_xy.contact_candidate,
                observation_is_explicit(self.latest_xy),
            )

        z_updated = False
        z_previous_state = self.machines["z"].state
        if (
            self.latest_z is not None
            and self.latest_z.sequence != self.last_z_sequence
        ):
            z_updated = True
            self.last_z_sequence = self.latest_z.sequence
            self.states["z"] = self.machines["z"].update(
                self.latest_z.timestamp_ms,
                self.latest_z.contact_candidate,
                observation_is_explicit(self.latest_z),
            )

        now_ms = time.perf_counter() * 1000.0
        age_xy = (
            None
            if self.latest_xy is None
            else max(0.0, now_ms - self.latest_xy.timestamp_ms)
        )
        age_z = (
            None
            if self.latest_z is None
            else max(0.0, now_ms - self.latest_z.timestamp_ms)
        )
        pair_skew = None
        event = None
        candidate_event = None
        candidate_cancelled = False
        release_confirmed = False

        if (
            self.decision_camera == "xy"
            and xy_updated
            and self.latest_xy is not None
        ):
            (
                self._release_candidate_active,
                candidate_event,
                candidate_cancelled,
                release_confirmed,
            ) = update_release_candidate(
                candidate_active=self._release_candidate_active,
                previous_state=xy_previous_state,
                just_released=self.machines["xy"].just_released,
                explicit_observation=observation_is_explicit(self.latest_xy),
                contact_candidate=self.latest_xy.contact_candidate,
                timestamp_ms=self.latest_xy.timestamp_ms,
                now_ms=now_ms,
                pair_skew_ms=0.0,
                maximum_age_ms=self.maximum_trigger_age_ms,
            )
            event = build_auto_release_event(
                just_released=self.machines["xy"].just_released,
                event_timestamp_ms=self.latest_xy.timestamp_ms,
                now_ms=now_ms,
                pair_skew_ms=0.0,
                maximum_age_ms=self.maximum_trigger_age_ms,
            )

        elif (
            self.decision_camera == "z"
            and z_updated
            and self.latest_z is not None
        ):
            (
                self._release_candidate_active,
                candidate_event,
                candidate_cancelled,
                release_confirmed,
            ) = update_release_candidate(
                candidate_active=self._release_candidate_active,
                previous_state=z_previous_state,
                just_released=self.machines["z"].just_released,
                explicit_observation=observation_is_explicit(self.latest_z),
                contact_candidate=self.latest_z.contact_candidate,
                timestamp_ms=self.latest_z.timestamp_ms,
                now_ms=now_ms,
                pair_skew_ms=0.0,
                maximum_age_ms=self.maximum_trigger_age_ms,
            )
            event = build_auto_release_event(
                just_released=self.machines["z"].just_released,
                event_timestamp_ms=self.latest_z.timestamp_ms,
                now_ms=now_ms,
                pair_skew_ms=0.0,
                maximum_age_ms=self.maximum_trigger_age_ms,
            )

        elif (
            self.decision_camera == "both"
            and self.latest_xy is not None
            and self.latest_z is not None
        ):
            pair_skew = abs(
                self.latest_xy.timestamp_ms - self.latest_z.timestamp_ms
            )
            sequences = (
                self.latest_xy.sequence,
                self.latest_z.sequence,
            )
            if (
                sequences != self.last_both_sequences
                and pair_skew <= self.result_pair_tolerance_ms
            ):
                self.last_both_sequences = sequences
                both_previous_state = self.machines["both"].state
                both_valid = (
                    observation_is_explicit(self.latest_xy)
                    and observation_is_explicit(self.latest_z)
                )
                both_contact = fused_contact_candidate(
                    self.machines["both"].state,
                    self.latest_xy.contact_candidate,
                    self.latest_z.contact_candidate,
                )
                event_timestamp_ms = max(
                    self.latest_xy.timestamp_ms,
                    self.latest_z.timestamp_ms,
                )
                self.states["both"] = self.machines["both"].update(
                    event_timestamp_ms,
                    both_contact,
                    both_valid,
                )
                (
                    self._release_candidate_active,
                    candidate_event,
                    candidate_cancelled,
                    release_confirmed,
                ) = update_release_candidate(
                    candidate_active=self._release_candidate_active,
                    previous_state=both_previous_state,
                    just_released=self.machines["both"].just_released,
                    explicit_observation=both_valid,
                    contact_candidate=both_contact,
                    timestamp_ms=event_timestamp_ms,
                    now_ms=now_ms,
                    pair_skew_ms=pair_skew,
                    maximum_age_ms=self.maximum_trigger_age_ms,
                )

                event = build_auto_release_event(
                    just_released=self.machines["both"].just_released,
                    event_timestamp_ms=event_timestamp_ms,
                    now_ms=now_ms,
                    pair_skew_ms=pair_skew,
                    maximum_age_ms=self.maximum_trigger_age_ms,
                )

        self.status = AutoReleaseStatus(
            xy_state=self.states["xy"],
            z_state=self.states["z"],
            both_state=self.states["both"],
            decision_camera=self.decision_camera,
            decision_state=self.states[self.decision_camera],
            result_age_xy_ms=age_xy,
            result_age_z_ms=age_z,
            result_pair_skew_ms=pair_skew,
            candidate_event=candidate_event,
            candidate_cancelled=candidate_cancelled,
            release_confirmed=release_confirmed,
            event=event,
        )
        return self.status

    def draw(self, frame_xy: np.ndarray, frame_z: np.ndarray):
        self._draw_camera(frame_xy, "XY", self.latest_xy, self.states["xy"])
        if self.detector_z is None:
            cv2.putText(
                frame_z,
                "MP Z: NOT USED FOR RELEASE",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (160, 160, 160),
                2,
            )
        else:
            self._draw_camera(frame_z, "Z", self.latest_z, self.states["z"])
        decision_state = self.states[self.decision_camera]
        both_color = (
            (0, 255, 0)
            if decision_state == TipContactState.GRASPED
            else (0, 165, 255)
        )
        for frame in (frame_xy, frame_z):
            cv2.putText(
                frame,
                (
                    f"AUTO RELEASE [{self.decision_camera.upper()}]: "
                    f"{decision_state}"
                ),
                (10, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                both_color,
                2,
            )

    @staticmethod
    def _draw_camera(
        frame: np.ndarray,
        role: str,
        observation: HandTipObservation | None,
        state: str,
    ):
        if observation is not None:
            if observation.thumb_tip_px is not None:
                thumb = tuple(
                    int(round(value)) for value in observation.thumb_tip_px
                )
                cv2.circle(frame, thumb, 6, (0, 165, 255), -1)
            if observation.index_tip_px is not None:
                index = tuple(
                    int(round(value)) for value in observation.index_tip_px
                )
                cv2.circle(frame, index, 6, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"MP {role}: {state}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )

    def reset(self):
        for machine in self.machines.values():
            machine.reset()
        self.states = {
            name: TipContactState.WAITING_FOR_GRASP
            for name in self.machines
        }
        self._release_candidate_active = False
        self.status = AutoReleaseStatus(
            xy_state=self.states["xy"],
            z_state=self.states["z"],
            both_state=self.states["both"],
            decision_camera=self.decision_camera,
            decision_state=self.states[self.decision_camera],
            result_age_xy_ms=None,
            result_age_z_ms=None,
            result_pair_skew_ms=None,
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.detector_xy.close()
        if self.detector_z is not None:
            self.detector_z.close()
