import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.core.ball_tracker import RoiBallTracker, draw_ball_detection
from tracking.core.config import AppConfig
from tracking.core.vision import (
    SharedFrameBuffer,
    build_undistort_maps,
    camera_capture_loop,
    init_ximea_camera,
    load_intrinsic,
    load_ximea_api,
    rotate_frame_if_needed,
    safe_close_camera,
    undistort_frame,
)
from manual_release_tracking.core.mediapipe_hands import (
    AsyncHandLandmarker,
    BinaryErrorStats,
    HandTipConfig,
    HandTipObservation,
    TipContactState,
    TipContactStateConfig,
    TipContactStateMachine,
)
from manual_release_tracking.experiments.release_detection_preview import (
    _put_lines,
    _resize_pair_to_same_size,
    _to_bgr,
)


def _format_optional(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _draw_tip(
    frame: np.ndarray,
    observation: HandTipObservation,
    point: tuple[float, float] | None,
    label: str,
    color: tuple[int, int, int],
):
    if point is None:
        return

    tip = (int(round(point[0])), int(round(point[1])))
    cv2.circle(frame, tip, 7, color, thickness=-1)
    cv2.putText(
        frame,
        label,
        (tip[0] + 8, tip[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        color,
        2,
    )

    center = observation.ball_center_px
    radius = observation.ball_radius_px
    if center is None or radius is None:
        return

    center_np = np.asarray(center, dtype=float)
    tip_np = np.asarray(point, dtype=float)
    direction = tip_np - center_np
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return
    surface = center_np + direction * (float(radius) / norm)
    surface_px = (int(round(surface[0])), int(round(surface[1])))
    cv2.line(frame, surface_px, tip, color, 2)
    cv2.circle(frame, surface_px, 4, color, thickness=-1)


def _draw_hand_observation(
    frame: np.ndarray,
    observation: HandTipObservation | None,
    role: str,
):
    if observation is None:
        cv2.putText(
            frame,
            f"{role}: waiting for MediaPipe result",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (180, 180, 180),
            2,
        )
        return

    _draw_tip(
        frame,
        observation,
        observation.thumb_tip_px,
        "THUMB",
        (0, 165, 255),
    )
    _draw_tip(
        frame,
        observation,
        observation.index_tip_px,
        "INDEX",
        (0, 255, 0),
    )
    if observation.thumb_tip_px is not None and observation.index_tip_px is not None:
        thumb = tuple(int(round(value)) for value in observation.thumb_tip_px)
        index = tuple(int(round(value)) for value in observation.index_tip_px)
        cv2.line(frame, thumb, index, (255, 255, 255), 1)

    status = "CONTACT" if observation.contact_candidate else "SEPARATED"
    color = (0, 255, 255) if observation.contact_candidate else (200, 200, 200)
    cv2.putText(
        frame,
        (
            f"{role}: {status} | "
            f"T gap/r={_format_optional(observation.thumb_gap_normalized)} | "
            f"I gap/r={_format_optional(observation.index_gap_normalized)} | "
            f"angle={_format_optional(observation.opposition_angle_deg, 0)}deg"
        ),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
    )


def _stats_label(name: str, stats: BinaryErrorStats) -> str:
    rate = "--" if stats.error_rate is None else f"{stats.error_rate * 100.0:.1f}%"
    return (
        f"{name} error={rate} n={stats.samples} "
        f"FP={stats.false_positive} FN={stats.false_negative}"
    )


def _update_error_stats(
    stats: BinaryErrorStats,
    state: str,
    ground_truth_grasped: bool | None,
):
    if ground_truth_grasped is None:
        return
    stats.update(
        predicted_grasped=state == TipContactState.GRASPED,
        actual_grasped=ground_truth_grasped,
    )


def run_mediapipe_release_detection_preview(cfg: AppConfig):
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
    result_pair_tolerance_ms = float(
        getattr(cfg, "mediapipe_result_pair_tolerance_ms", 40.0)
    )
    minimum_control_fps = float(
        getattr(cfg, "mediapipe_minimum_control_fps", 30.0)
    )
    maximum_control_age_ms = float(
        getattr(cfg, "mediapipe_maximum_control_age_ms", 80.0)
    )
    submit_max_fps = float(
        getattr(cfg, "mediapipe_submit_max_fps", 60.0)
    )
    submit_interval_sec = (
        0.0 if submit_max_fps <= 0.0 else 1.0 / submit_max_fps
    )

    detector_xy = detector_z = None
    try:
        detector_xy = AsyncHandLandmarker("xy", model_path, hand_cfg)
        detector_z = AsyncHandLandmarker("z", model_path, hand_cfg)
    except Exception as exc:
        if detector_xy is not None:
            detector_xy.close()
        print(f"[ERROR] MediaPipe initialization failed:\n{exc}")
        return

    try:
        mtx_cam_xy, dist_cam_xy = load_intrinsic(cfg.intrinsic_xy_npz, "xy")
        mtx_cam_z, dist_cam_z = load_intrinsic(cfg.intrinsic_z_npz, "z")
        use_undistort = True
        print("[INFO] Undistortion enabled for both MediaPipe preview cameras.")
    except Exception as exc:
        print(f"[WARN] Intrinsic parameters load failed: {exc}")
        mtx_cam_xy = dist_cam_xy = None
        mtx_cam_z = dist_cam_z = None
        use_undistort = False

    xiapi = load_ximea_api(cfg)
    if xiapi is None:
        print("[ERROR] XIMEA API not available.")
        detector_xy.close()
        detector_z.close()
        return

    cam_xy = cam_z = None
    running_event = threading.Event()
    cam_thread_xy = cam_thread_z = None

    machines = {
        "xy": TipContactStateMachine(state_cfg),
        "z": TipContactStateMachine(state_cfg),
        "both": TipContactStateMachine(state_cfg),
    }
    error_stats = {
        "xy": BinaryErrorStats(),
        "z": BinaryErrorStats(),
        "both": BinaryErrorStats(),
    }
    states = {
        "xy": TipContactState.WAITING_FOR_GRASP,
        "z": TipContactState.WAITING_FOR_GRASP,
        "both": TipContactState.WAITING_FOR_GRASP,
    }
    ground_truth_grasped: bool | None = None
    last_xy_sequence = 0
    last_z_sequence = 0
    last_both_sequences = (0, 0)

    try:
        cam_xy, img_xy = init_ximea_camera(xiapi, cfg.camera_xy_sn, "xy", cfg)
        cam_z, img_z = init_ximea_camera(xiapi, cfg.camera_z_sn, "z", cfg)

        cam_xy.get_image(img_xy)
        frame0_xy_time = time.perf_counter()
        frame0_raw_xy = img_xy.get_image_data_numpy()

        cam_z.get_image(img_z)
        frame0_z_time = time.perf_counter()
        frame0_raw_z = img_z.get_image_data_numpy()

        if use_undistort:
            map1_xy, map2_xy = build_undistort_maps(
                frame0_raw_xy.shape,
                mtx_cam_xy,
                dist_cam_xy,
            )
            frame0_xy = undistort_frame(frame0_raw_xy, map1_xy, map2_xy)

            if cfg.rotate_z_frame and cfg.z_intrinsic_is_rotated:
                frame0_z_for_intrinsic = rotate_frame_if_needed(
                    frame0_raw_z,
                    True,
                    cfg.rotate_z_code,
                )
                map1_z, map2_z = build_undistort_maps(
                    frame0_z_for_intrinsic.shape,
                    mtx_cam_z,
                    dist_cam_z,
                )
                frame0_z = undistort_frame(
                    frame0_z_for_intrinsic,
                    map1_z,
                    map2_z,
                )
            else:
                map1_z, map2_z = build_undistort_maps(
                    frame0_raw_z.shape,
                    mtx_cam_z,
                    dist_cam_z,
                )
                frame0_z = undistort_frame(frame0_raw_z, map1_z, map2_z)
        else:
            map1_xy = map2_xy = None
            map1_z = map2_z = None
            frame0_xy = frame0_raw_xy
            frame0_z = frame0_raw_z

        if not (use_undistort and cfg.z_intrinsic_is_rotated):
            frame0_z = rotate_frame_if_needed(
                frame0_z,
                cfg.rotate_z_frame,
                cfg.rotate_z_code,
            )

        height_xy, width_xy = frame0_xy.shape[:2]
        height_z, width_z = frame0_z.shape[:2]
        tracker_xy = RoiBallTracker(width_xy, height_xy, cfg)
        tracker_z = RoiBallTracker(width_z, height_z, cfg)

        frame_buffer = SharedFrameBuffer(maxlen=cfg.camera_sync_buffer_size)
        frame_buffer.set_xy(frame0_xy.copy(), frame0_xy_time)
        frame_buffer.set_z(frame0_z.copy(), frame0_z_time)

        running_event.set()
        cam_thread_xy = threading.Thread(
            target=camera_capture_loop,
            args=(
                cam_xy,
                img_xy,
                "xy",
                cfg,
                frame_buffer,
                running_event,
                use_undistort,
                map1_xy,
                map2_xy,
            ),
            daemon=True,
        )
        cam_thread_z = threading.Thread(
            target=camera_capture_loop,
            args=(
                cam_z,
                img_z,
                "z",
                cfg,
                frame_buffer,
                running_event,
                use_undistort,
                map1_z,
                map2_z,
            ),
            daemon=True,
        )
        cam_thread_xy.start()
        cam_thread_z.start()

        window_title = "MediaPipe Release Detection Preview"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        last_synced_pair_time = time.perf_counter()
        fps_started_at = time.perf_counter()
        pair_count = 0
        pair_fps = 0.0
        last_submit_xy_time = float("-inf")
        last_submit_z_time = float("-inf")

        print("=================================================")
        print("  MEDIAPIPE HAND LANDMARKER PREVIEW")
        print("  No AUTD connection and no ultrasound output.")
        print("  THUMB=orange, INDEX=green.")
        print(f"  MediaPipe submission cap: {submit_max_fps:.1f} fps/camera")
        print("  [G] ground truth GRASPED")
        print("  [N] ground truth NOT GRASPED / RELEASED")
        print("  [U] ground truth unknown (do not score)")
        print("  [C] clear comparison statistics")
        print("  [SPACE] reset states, [ESC] exit")
        print("=================================================")

        while True:
            now_sec = time.perf_counter()
            age_xy, age_z = frame_buffer.get_frame_ages(now_sec)
            if age_xy > cfg.camera_frame_timeout_sec or age_z > cfg.camera_frame_timeout_sec:
                raise RuntimeError(
                    "Camera watchdog timeout: "
                    f"XY age={age_xy:.3f}s, Z age={age_z:.3f}s"
                )

            synced_pair = frame_buffer.get_synced_pair(
                cfg.camera_sync_tolerance_sec
            )
            if synced_pair is None:
                if now_sec - last_synced_pair_time > cfg.camera_frame_timeout_sec:
                    raise RuntimeError(
                        "Camera synchronization timeout: no frame pair within "
                        f"{cfg.camera_sync_tolerance_sec * 1000.0:.1f} ms"
                    )
                time.sleep(0.001)
                continue

            (
                frame_xy,
                frame_xy_time,
                frame_z,
                frame_z_time,
                frame_sync_skew,
            ) = synced_pair
            last_synced_pair_time = now_sec
            pair_count += 1
            fps_elapsed = now_sec - fps_started_at
            if fps_elapsed >= 1.0:
                pair_fps = pair_count / fps_elapsed
                pair_count = 0
                fps_started_at = now_sec

            ball_xy = tracker_xy.detect(frame_xy)
            ball_z = tracker_z.detect(frame_z)
            if frame_xy_time - last_submit_xy_time >= submit_interval_sec:
                detector_xy.submit(
                    frame_xy,
                    frame_xy_time,
                    ball_xy.center if ball_xy.detected else None,
                    ball_xy.radius_px if ball_xy.detected else None,
                )
                last_submit_xy_time = frame_xy_time
            if frame_z_time - last_submit_z_time >= submit_interval_sec:
                detector_z.submit(
                    frame_z,
                    frame_z_time,
                    ball_z.center if ball_z.detected else None,
                    ball_z.radius_px if ball_z.detected else None,
                )
                last_submit_z_time = frame_z_time

            observation_xy = detector_xy.latest()
            observation_z = detector_z.latest()

            if (
                observation_xy is not None
                and observation_xy.sequence != last_xy_sequence
            ):
                last_xy_sequence = observation_xy.sequence
                states["xy"] = machines["xy"].update(
                    observation_xy.timestamp_ms,
                    observation_xy.contact_candidate,
                    observation_xy.valid_for_decision,
                )

            if (
                observation_z is not None
                and observation_z.sequence != last_z_sequence
            ):
                last_z_sequence = observation_z.sequence
                states["z"] = machines["z"].update(
                    observation_z.timestamp_ms,
                    observation_z.contact_candidate,
                    observation_z.valid_for_decision,
                )

            result_pair_skew_ms = None
            if observation_xy is not None and observation_z is not None:
                result_pair_skew_ms = abs(
                    observation_xy.timestamp_ms - observation_z.timestamp_ms
                )
                current_sequences = (
                    observation_xy.sequence,
                    observation_z.sequence,
                )
                if (
                    current_sequences != last_both_sequences
                    and result_pair_skew_ms <= result_pair_tolerance_ms
                ):
                    last_both_sequences = current_sequences
                    both_valid = (
                        observation_xy.valid_for_decision
                        and observation_z.valid_for_decision
                    )
                    both_contact = (
                        observation_xy.contact_candidate
                        and observation_z.contact_candidate
                    )
                    states["both"] = machines["both"].update(
                        max(
                            observation_xy.timestamp_ms,
                            observation_z.timestamp_ms,
                        ),
                        both_contact,
                        both_valid,
                    )
                    # Score all three modes on the same matched-result pair so their
                    # error rates have the same sample count and are directly comparable.
                    for name in ("xy", "z", "both"):
                        _update_error_stats(
                            error_stats[name],
                            states[name],
                            ground_truth_grasped,
                        )

            display_now_ms = time.perf_counter() * 1000.0
            result_age_xy_ms = (
                None
                if observation_xy is None
                else max(0.0, display_now_ms - observation_xy.timestamp_ms)
            )
            result_age_z_ms = (
                None
                if observation_z is None
                else max(0.0, display_now_ms - observation_z.timestamp_ms)
            )
            mp_fps_xy = detector_xy.callback_fps
            mp_fps_z = detector_z.callback_fps
            speed_ready = bool(
                mp_fps_xy >= minimum_control_fps
                and mp_fps_z >= minimum_control_fps
                and result_age_xy_ms is not None
                and result_age_z_ms is not None
                and result_age_xy_ms <= maximum_control_age_ms
                and result_age_z_ms <= maximum_control_age_ms
            )

            frame_xy_bgr = _to_bgr(frame_xy)
            frame_z_bgr = _to_bgr(frame_z)
            draw_ball_detection(frame_xy_bgr, ball_xy, tracking_active=False)
            draw_ball_detection(frame_z_bgr, ball_z, tracking_active=False)
            _draw_hand_observation(frame_xy_bgr, observation_xy, "XY")
            _draw_hand_observation(frame_z_bgr, observation_z, "Z")

            ground_truth_label = {
                None: "UNKNOWN",
                True: "GRASPED",
                False: "NOT_GRASPED",
            }[ground_truth_grasped]
            speed_label = (
                "SPEED READY"
                if speed_ready
                else "SPEED WARN (preview state only)"
            )
            common_lines = [
                (
                    f"State XY={states['xy']} | Z={states['z']} | "
                    f"BOTH={states['both']}"
                ),
                (
                    f"MP FPS XY={mp_fps_xy:.1f}, Z={mp_fps_z:.1f} | "
                    f"Pair FPS={pair_fps:.1f}"
                ),
                (
                    f"Result latency XY="
                    f"{_format_optional(None if observation_xy is None else observation_xy.result_latency_ms)}ms, "
                    f"Z={_format_optional(None if observation_z is None else observation_z.result_latency_ms)}ms"
                ),
                (
                    f"Result age XY={_format_optional(result_age_xy_ms)}ms, "
                    f"Z={_format_optional(result_age_z_ms)}ms | "
                    f"result skew={_format_optional(result_pair_skew_ms)}ms"
                ),
                (
                    f"Camera sync={frame_sync_skew * 1000.0:.2f}ms | "
                    f"{speed_label}"
                ),
                f"Ground truth={ground_truth_label} (G/N/U)",
                _stats_label("XY", error_stats["xy"]),
                _stats_label("Z", error_stats["z"]),
                _stats_label("BOTH", error_stats["both"]),
                "C: clear stats | SPACE: reset states | ESC: exit",
            ]
            _put_lines(
                frame_xy_bgr,
                common_lines,
                10,
                60,
                (255, 255, 255),
            )

            show_xy, show_z = _resize_pair_to_same_size(
                frame_xy_bgr,
                frame_z_bgr,
            )
            combined = np.hstack([show_xy, show_z])
            cv2.line(
                combined,
                (show_xy.shape[1], 0),
                (show_xy.shape[1], combined.shape[0] - 1),
                (255, 255, 255),
                1,
            )
            cv2.imshow(window_title, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("g"):
                ground_truth_grasped = True
                print("[GT] GRASPED")
            elif key == ord("n"):
                ground_truth_grasped = False
                print("[GT] NOT_GRASPED / RELEASED")
            elif key == ord("u"):
                ground_truth_grasped = None
                print("[GT] UNKNOWN")
            elif key == ord("c"):
                for stats in error_stats.values():
                    stats.reset()
                print("[INFO] comparison statistics cleared.")
            elif key == ord(" "):
                for machine in machines.values():
                    machine.reset()
                states = {
                    name: TipContactState.WAITING_FOR_GRASP
                    for name in machines
                }
                print("[INFO] MediaPipe contact states reset.")

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt.")
    except Exception as exc:
        print(f"[ERROR] Runtime error: {exc}")
    finally:
        running_event.clear()
        try:
            if cam_thread_xy is not None:
                cam_thread_xy.join(timeout=1.0)
            if cam_thread_z is not None:
                cam_thread_z.join(timeout=1.0)
        except Exception:
            pass
        safe_close_camera(cam_xy)
        safe_close_camera(cam_z)
        if detector_xy is not None:
            detector_xy.close()
        if detector_z is not None:
            detector_z.close()
        cv2.destroyAllWindows()
        print("[SUMMARY] Ground-truth-labeled frame comparison")
        for name in ("xy", "z", "both"):
            print(f"[SUMMARY] {_stats_label(name.upper(), error_stats[name])}")
        print("[INFO] MediaPipe preview stopped.")


if __name__ == "__main__":
    config = AppConfig(log_enabled=False)
    config.mediapipe_hand_model_path = (
        "./manual_release_tracking/models/hand_landmarker.task"
    )
    config.mediapipe_input_width_px = 640
    config.mediapipe_tip_gap_normalized_max = 0.45
    config.mediapipe_min_opposition_angle_deg = 60.0
    config.mediapipe_result_pair_tolerance_ms = 40.0
    config.mediapipe_submit_max_fps = 60.0
    config.mediapipe_minimum_control_fps = 30.0
    config.mediapipe_maximum_control_age_ms = 80.0
    run_mediapipe_release_detection_preview(config)
