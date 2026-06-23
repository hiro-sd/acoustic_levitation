"""
Fit single-camera intrinsic parameters from ChArUco images.

デフォルトでは cam2/Z の「回転後画像」から
calibration/intrinsic_charuco_cam2.npz を作り直す。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np


TRACKING_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class IntrinsicFitSettings:
    images_dir: str = str(TRACKING_ROOT / "calibration" / "intrinsic_images" / "cam2_rotated")
    output_npz: str = str(TRACKING_ROOT / "calibration" / "intrinsic_charuco_cam2.npz")
    report_json: str = str(TRACKING_ROOT / "calibration" / "intrinsic_charuco_cam2_report.json")

    # 現在の single_camera_calibration.py / stereo report に合わせたデフォルト。
    # 実際に印刷したボードの寸法に合わせて --square-length-mm / --marker-length-mm で上書きする。
    squares_x: int = 7
    squares_y: int = 5
    square_length_mm: float = 38.0
    marker_length_mm: float = 27.0
    aruco_dict: int = cv2.aruco.DICT_4X4_50

    min_charuco_corners: int = 8
    min_valid_images: int = 8


def _make_board(settings: IntrinsicFitSettings):
    dictionary = cv2.aruco.getPredefinedDictionary(settings.aruco_dict)
    return cv2.aruco.CharucoBoard(
        (settings.squares_x, settings.squares_y),
        settings.square_length_mm,
        settings.marker_length_mm,
        dictionary,
    )


def _detect_image_points(board, detector, image, min_corners: int):
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(image)
    if charuco_ids is None or len(charuco_ids) < min_corners:
        return None, None, 0

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is None or img_points is None or len(obj_points) < min_corners:
        return None, None, len(charuco_ids)

    return (
        obj_points.astype(np.float32),
        img_points.astype(np.float32),
        len(charuco_ids),
    )


def _compute_mean_reprojection_error(
    objpoints,
    imgpoints,
    rvecs,
    tvecs,
    camera_matrix,
    dist_coeffs,
) -> float:
    total_error = 0.0
    total_points = 0
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, projected, cv2.NORM_L2)
        total_error += error * error
        total_points += len(objp)
    if total_points == 0:
        return float("nan")
    return float(np.sqrt(total_error / total_points))


def main():
    parser = argparse.ArgumentParser(
        description="Fit intrinsic parameters from ChArUco images."
    )
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--square-length-mm", type=float, default=None)
    parser.add_argument("--marker-length-mm", type=float, default=None)
    parser.add_argument("--min-charuco-corners", type=int, default=None)
    args = parser.parse_args()

    settings = IntrinsicFitSettings()
    overrides = {}
    if args.images_dir is not None:
        overrides["images_dir"] = args.images_dir
    if args.output_npz is not None:
        overrides["output_npz"] = args.output_npz
    if args.report_json is not None:
        overrides["report_json"] = args.report_json
    if args.square_length_mm is not None:
        overrides["square_length_mm"] = args.square_length_mm
    if args.marker_length_mm is not None:
        overrides["marker_length_mm"] = args.marker_length_mm
    if args.min_charuco_corners is not None:
        overrides["min_charuco_corners"] = args.min_charuco_corners
    if overrides:
        settings = replace(settings, **overrides)

    images_dir = Path(settings.images_dir)
    images = sorted(images_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(
            f"No png images found in {images_dir}. "
            "Run calibration/camera/capture_charuco_cam2.py first."
        )

    print(
        "[INFO] ChArUco board: "
        f"{settings.squares_x}x{settings.squares_y}, "
        f"square={settings.square_length_mm:.3f} mm, "
        f"marker={settings.marker_length_mm:.3f} mm"
    )
    print(f"[INFO] Processing {len(images)} images from {images_dir}")

    board = _make_board(settings)
    detector_params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    objpoints = []
    imgpoints = []
    used_images = []
    rejected_images = []
    corner_counts = []
    image_size = None

    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            rejected_images.append({"path": str(path), "reason": "read_failed"})
            continue

        current_size = (img.shape[1], img.shape[0])
        image_size = current_size if image_size is None else image_size
        if current_size != image_size:
            rejected_images.append(
                {
                    "path": str(path),
                    "reason": "image_size_mismatch",
                    "image_size": current_size,
                }
            )
            continue

        objp, imgp, n_corners = _detect_image_points(
            board,
            detector,
            img,
            settings.min_charuco_corners,
        )
        if objp is None:
            rejected_images.append(
                {
                    "path": str(path),
                    "reason": "not_enough_charuco_corners",
                    "corners": n_corners,
                }
            )
            print(f"[WARN] rejected {path.name}: corners={n_corners}")
            continue

        objpoints.append(objp)
        imgpoints.append(imgp)
        used_images.append(str(path))
        corner_counts.append(n_corners)
        print(f"[OK] {path.name}: corners={n_corners}")

    print(f"[INFO] Valid images: {len(objpoints)}/{len(images)}")
    if len(objpoints) < settings.min_valid_images:
        raise RuntimeError(
            f"Not enough valid images: {len(objpoints)}. "
            f"Need at least {settings.min_valid_images}."
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    mean_reprojection_error = _compute_mean_reprojection_error(
        objpoints,
        imgpoints,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
    )

    print("\n=== ChArUco intrinsic calibration result ===")
    print(f"RMS reprojection error : {rms:.4f} px")
    print(f"Mean reprojection error: {mean_reprojection_error:.4f} px")
    print(f"Valid images used      : {len(objpoints)}")
    print(f"Image size             : {image_size}")
    print("camera_matrix:")
    print(camera_matrix)
    print("dist_coeffs:")
    print(dist_coeffs)

    output_npz = Path(settings.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms=float(rms),
        mean_reprojection_error=float(mean_reprojection_error),
        image_size=np.array(image_size, dtype=np.int32),
        board_squares=np.array([settings.squares_x, settings.squares_y], dtype=np.int32),
        square_length_mm=float(settings.square_length_mm),
        marker_length_mm=float(settings.marker_length_mm),
        used_images=np.array(used_images),
        corner_counts=np.array(corner_counts, dtype=np.int32),
        note=(
            "For cam2/Z, these intrinsics should be used with images in the same "
            "orientation as the captured calibration images."
        ),
    )

    report = {
        "settings": asdict(settings),
        "rms_px": float(rms),
        "mean_reprojection_error_px": float(mean_reprojection_error),
        "valid_images_used": len(objpoints),
        "total_images_found": len(images),
        "image_size": image_size,
        "corner_counts": corner_counts,
        "used_images": used_images,
        "rejected_images": rejected_images,
    }
    Path(settings.report_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[SUCCESS] Saved intrinsics: {output_npz}")
    print(f"[SUCCESS] Saved report: {settings.report_json}")


if __name__ == "__main__":
    main()
