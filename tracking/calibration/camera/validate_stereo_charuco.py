"""
Validate stereo calibration by triangulating ChArUco corners.

評価内容:
  - 2カメラで共通検出されたChArUco角を三角測量
  - 復元点を各画像へ再投影した誤差
  - 復元点同士の隣接距離が既知のsquare_lengthに近いか
  - 復元点が1枚の平面に乗っているか

この検証は「ステレオキャリブレーション結果が幾何的に妥当か」を見るためのもの。
AUTD座標系への変換精度は別途、既知3D点で確認する必要がある。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from fit_stereo_charuco import (
    StereoFitSettings,
    _extract_common_points,
    _find_image_pairs,
    _make_charuco_board,
    _make_detector,
)


TRACKING_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class StereoValidationSettings:
    image_root: str = str(TRACKING_ROOT / "calibration" / "stereo_images" / "charuco")
    stereo_npz: str = str(TRACKING_ROOT / "calibration" / "stereo_charuco_calibration_result.npz")
    report_json: str = str(TRACKING_ROOT / "calibration" / "stereo_charuco_validation_report.json")
    min_common_corners: int = 8


def _as_matrix(data, key: str) -> np.ndarray:
    value = data[key]
    if value.dtype == object and value.shape == ():
        raise ValueError(f"{key} is None in stereo npz.")
    return value.astype(np.float64)


def _load_stereo_params(path: str):
    data = np.load(path, allow_pickle=True)
    params = {
        "camera_matrix_l": _as_matrix(data, "camera_matrix_l"),
        "dist_l": _as_matrix(data, "dist_l"),
        "camera_matrix_r": _as_matrix(data, "camera_matrix_r"),
        "dist_r": _as_matrix(data, "dist_r"),
        "R": _as_matrix(data, "R"),
        "T": _as_matrix(data, "T").reshape(3, 1),
        "square_length_mm": float(data["square_length_mm"]),
        "marker_length_mm": float(data["marker_length_mm"]),
        "board_squares": tuple(int(v) for v in data["board_squares"]),
    }
    return params


def _triangulate_points(
    imgpoints_l: np.ndarray,
    imgpoints_r: np.ndarray,
    camera_matrix_l: np.ndarray,
    dist_l: np.ndarray,
    camera_matrix_r: np.ndarray,
    dist_r: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    undist_l = cv2.undistortPoints(imgpoints_l, camera_matrix_l, dist_l)
    undist_r = cv2.undistortPoints(imgpoints_r, camera_matrix_r, dist_r)

    proj_l = np.hstack([np.eye(3), np.zeros((3, 1))])
    proj_r = np.hstack([R, T.reshape(3, 1)])

    points_h = cv2.triangulatePoints(
        proj_l,
        proj_r,
        undist_l.reshape(-1, 2).T,
        undist_r.reshape(-1, 2).T,
    )
    points_3d = (points_h[:3] / points_h[3]).T
    return points_3d.astype(np.float64)


def _reprojection_errors(
    points_3d_l: np.ndarray,
    imgpoints_l: np.ndarray,
    imgpoints_r: np.ndarray,
    camera_matrix_l: np.ndarray,
    dist_l: np.ndarray,
    camera_matrix_r: np.ndarray,
    dist_r: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    proj_l, _ = cv2.projectPoints(
        points_3d_l.reshape(-1, 1, 3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        camera_matrix_l,
        dist_l,
    )
    points_3d_r = (R @ points_3d_l.T + T).T
    proj_r, _ = cv2.projectPoints(
        points_3d_r.reshape(-1, 1, 3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        camera_matrix_r,
        dist_r,
    )

    err_l = np.linalg.norm(proj_l.reshape(-1, 2) - imgpoints_l.reshape(-1, 2), axis=1)
    err_r = np.linalg.norm(proj_r.reshape(-1, 2) - imgpoints_r.reshape(-1, 2), axis=1)
    return err_l, err_r


def _board_ids_from_objpoints(objpoints: np.ndarray, square_length_mm: float) -> list[tuple[int, int]]:
    ids = []
    for p in objpoints.reshape(-1, 3):
        col = int(round(float(p[0]) / square_length_mm))
        row = int(round(float(p[1]) / square_length_mm))
        ids.append((col, row))
    return ids


def _neighbor_distance_errors(
    objpoints: np.ndarray,
    points_3d_l: np.ndarray,
    square_length_mm: float,
) -> list[float]:
    ids = _board_ids_from_objpoints(objpoints, square_length_mm)
    point_by_grid = {grid: p for grid, p in zip(ids, points_3d_l)}
    distance_errors = []

    for col, row in ids:
        p = point_by_grid[(col, row)]
        for neighbor in ((col + 1, row), (col, row + 1)):
            if neighbor not in point_by_grid:
                continue
            q = point_by_grid[neighbor]
            distance = float(np.linalg.norm(p - q))
            distance_errors.append(distance - square_length_mm)

    return distance_errors


def _plane_residuals(points_3d_l: np.ndarray) -> np.ndarray:
    if len(points_3d_l) < 3:
        return np.full(len(points_3d_l), np.nan, dtype=np.float64)
    centroid = points_3d_l.mean(axis=0)
    _, _, vh = np.linalg.svd(points_3d_l - centroid, full_matrices=False)
    normal = vh[-1]
    signed = (points_3d_l - centroid) @ normal
    return np.abs(signed)


def _summary(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "rmse": None,
            "p95_abs": None,
            "max_abs": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "p95_abs": float(np.percentile(np.abs(arr), 95)),
        "max_abs": float(np.max(np.abs(arr))),
    }


def _pair_key_from_path(path: str) -> str:
    return Path(path).stem


def main():
    parser = argparse.ArgumentParser(
        description="Validate stereo calibration by triangulating ChArUco corners."
    )
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--stereo-npz", default=None)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--min-common-corners", type=int, default=None)
    args = parser.parse_args()

    settings = StereoValidationSettings()
    overrides = {}
    if args.image_root is not None:
        overrides["image_root"] = args.image_root
    if args.stereo_npz is not None:
        overrides["stereo_npz"] = args.stereo_npz
    if args.report_json is not None:
        overrides["report_json"] = args.report_json
    if args.min_common_corners is not None:
        overrides["min_common_corners"] = args.min_common_corners
    if overrides:
        settings = replace(settings, **overrides)

    params = _load_stereo_params(settings.stereo_npz)
    square_length_mm = params["square_length_mm"]
    marker_length_mm = params["marker_length_mm"]
    squares_x, squares_y = params["board_squares"]

    fit_settings = StereoFitSettings(
        image_root=settings.image_root,
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_mm=square_length_mm,
        marker_length_mm=marker_length_mm,
        min_common_corners=settings.min_common_corners,
    )
    pairs = _find_image_pairs(fit_settings)

    board = _make_charuco_board(fit_settings)
    detector_l = _make_detector(board, params["camera_matrix_l"], params["dist_l"])
    detector_r = _make_detector(board, params["camera_matrix_r"], params["dist_r"])

    all_reproj_l = []
    all_reproj_r = []
    all_distance_errors = []
    all_plane_residuals = []
    pair_reports = []
    rejected_pairs = []

    for left_path, right_path in pairs:
        img_l = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        img_r = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if img_l is None or img_r is None:
            rejected_pairs.append(
                {
                    "left": str(left_path),
                    "right": str(right_path),
                    "reason": "read_failed",
                }
            )
            continue

        objp, imgp_l, imgp_r, n_l, n_r, n_common = _extract_common_points(
            board,
            detector_l,
            detector_r,
            img_l,
            img_r,
            settings.min_common_corners,
        )
        if objp is None:
            rejected_pairs.append(
                {
                    "left": str(left_path),
                    "right": str(right_path),
                    "reason": "not_enough_common_charuco_corners",
                    "left_corners": n_l,
                    "right_corners": n_r,
                    "common_corners": n_common,
                }
            )
            continue

        points_3d = _triangulate_points(
            imgp_l,
            imgp_r,
            params["camera_matrix_l"],
            params["dist_l"],
            params["camera_matrix_r"],
            params["dist_r"],
            params["R"],
            params["T"],
        )
        reproj_l, reproj_r = _reprojection_errors(
            points_3d,
            imgp_l,
            imgp_r,
            params["camera_matrix_l"],
            params["dist_l"],
            params["camera_matrix_r"],
            params["dist_r"],
            params["R"],
            params["T"],
        )
        distance_errors = _neighbor_distance_errors(objp, points_3d, square_length_mm)
        plane_residuals = _plane_residuals(points_3d)

        all_reproj_l.extend(reproj_l.tolist())
        all_reproj_r.extend(reproj_r.tolist())
        all_distance_errors.extend(distance_errors)
        all_plane_residuals.extend(plane_residuals.tolist())

        pair_report = {
            "left": str(left_path),
            "right": str(right_path),
            "common_corners": n_common,
            "reprojection_error_l_px": _summary(reproj_l),
            "reprojection_error_r_px": _summary(reproj_r),
            "neighbor_distance_error_mm": _summary(distance_errors),
            "plane_residual_mm": _summary(plane_residuals),
        }
        pair_reports.append(pair_report)

        print(
            f"[OK] {left_path.name}: common={n_common}, "
            f"reproj_l_mean={pair_report['reprojection_error_l_px']['mean']:.3f}px, "
            f"reproj_r_mean={pair_report['reprojection_error_r_px']['mean']:.3f}px, "
            f"dist_rmse={pair_report['neighbor_distance_error_mm']['rmse']:.3f}mm, "
            f"plane_rmse={pair_report['plane_residual_mm']['rmse']:.3f}mm"
        )

    report = {
        "settings": asdict(settings),
        "stereo_npz": settings.stereo_npz,
        "square_length_mm": square_length_mm,
        "marker_length_mm": marker_length_mm,
        "valid_pairs": len(pair_reports),
        "rejected_pairs": rejected_pairs,
        "overall": {
            "reprojection_error_l_px": _summary(all_reproj_l),
            "reprojection_error_r_px": _summary(all_reproj_r),
            "neighbor_distance_error_mm": _summary(all_distance_errors),
            "plane_residual_mm": _summary(all_plane_residuals),
        },
        "pairs": pair_reports,
    }

    output_path = Path(settings.report_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    overall = report["overall"]
    print("\n=== Stereo triangulation validation ===")
    print(f"Valid pairs: {len(pair_reports)}")
    print(
        "Reprojection L mean/RMSE: "
        f"{overall['reprojection_error_l_px']['mean']:.3f} / "
        f"{overall['reprojection_error_l_px']['rmse']:.3f} px"
    )
    print(
        "Reprojection R mean/RMSE: "
        f"{overall['reprojection_error_r_px']['mean']:.3f} / "
        f"{overall['reprojection_error_r_px']['rmse']:.3f} px"
    )
    print(
        "Neighbor distance error mean/RMSE: "
        f"{overall['neighbor_distance_error_mm']['mean']:.3f} / "
        f"{overall['neighbor_distance_error_mm']['rmse']:.3f} mm"
    )
    print(
        "Plane residual mean/RMSE: "
        f"{overall['plane_residual_mm']['mean']:.3f} / "
        f"{overall['plane_residual_mm']['rmse']:.3f} mm"
    )
    print(f"[SUCCESS] Saved validation report: {output_path}")


if __name__ == "__main__":
    main()
