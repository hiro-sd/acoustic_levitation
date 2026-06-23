"""
Fit stereo extrinsic parameters from captured ChArUco stereo image pairs.

入力:
    calibration/stereo_images/charuco/cam1/pair_000_xy.png
    calibration/stereo_images/charuco/cam2/pair_000_z.png

出力:
    calibration/stereo_charuco_calibration_result.npz

注意:
    Zカメラ画像は capture_stereo_charuco.py で保存した「回転後画像」を使う前提。
    既存の intrinsic_charuco_cam1.npz / intrinsic_charuco_cam2.npz も、
    同じ画像向きで作成されている必要がある。
"""

from __future__ import annotations

import json
import re
import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np


TRACKING_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class StereoFitSettings:
    image_root: str = str(TRACKING_ROOT / "calibration" / "stereo_images" / "charuco")
    cam1_dir_name: str = "cam1"
    cam2_dir_name: str = "cam2"
    left_intrinsic_npz: str = str(TRACKING_ROOT / "calibration" / "intrinsic_charuco_cam1.npz")
    right_intrinsic_npz: str = str(TRACKING_ROOT / "calibration" / "intrinsic_charuco_cam2.npz")
    output_npz: str = str(TRACKING_ROOT / "calibration" / "stereo_charuco_calibration_result.npz")
    report_json: str = str(TRACKING_ROOT / "calibration" / "stereo_charuco_calibration_report.json")

    # create_charuco.py と同じ設定。印刷後に実測値が違う場合はここを修正する。
    squares_x: int = 7
    squares_y: int = 5
    square_length_mm: float = 38.0
    marker_length_mm: float = 27.0
    aruco_dict: int = cv2.aruco.DICT_4X4_50

    min_common_corners: int = 8
    min_valid_pairs: int = 5
    outlier_std_scale: float = 2.5


def _load_intrinsic(path: str, label: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "camera_matrix" in data and "dist_coeffs" in data:
        return data["camera_matrix"].astype(np.float64), data["dist_coeffs"].astype(np.float64)
    if "mtx" in data and "dist" in data:
        return data["mtx"].astype(np.float64), data["dist"].astype(np.float64)
    raise KeyError(f"{label} intrinsic npz has unsupported keys: {data.files}")


def _make_charuco_board(settings: StereoFitSettings):
    dictionary = cv2.aruco.getPredefinedDictionary(settings.aruco_dict)
    return cv2.aruco.CharucoBoard(
        (settings.squares_x, settings.squares_y),
        settings.square_length_mm,
        settings.marker_length_mm,
        dictionary,
    )


def _make_detector(board, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    detector_params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.cameraMatrix = camera_matrix
    charuco_params.distCoeffs = dist_coeffs
    return cv2.aruco.CharucoDetector(board, charuco_params, detector_params)


def _flatten_ids(ids) -> list[int]:
    return [int(v) for v in np.asarray(ids).reshape(-1)]


def _pair_key(path: Path) -> str:
    """
    pair_000_xy.png / pair_000_z.png / img_00.png などから対応キーを作る。
    """
    stem = path.stem
    match = re.search(r"(\d+)", stem)
    if match:
        return match.group(1)
    return stem.replace("_xy", "").replace("_z", "")


def _find_image_pairs(settings: StereoFitSettings) -> list[tuple[Path, Path]]:
    root = Path(settings.image_root)
    cam1_dir = root / settings.cam1_dir_name
    cam2_dir = root / settings.cam2_dir_name

    if not cam1_dir.exists() or not cam2_dir.exists():
        raise FileNotFoundError(
            "Stereo image directories not found:\n"
            f"  {cam1_dir}\n"
            f"  {cam2_dir}\n"
            "Run calibration/camera/capture_stereo_charuco.py first, or edit image_root."
        )

    cam1_images = sorted(cam1_dir.glob("*.png"))
    cam2_images = sorted(cam2_dir.glob("*.png"))

    cam1_by_key = {_pair_key(p): p for p in cam1_images}
    cam2_by_key = {_pair_key(p): p for p in cam2_images}
    common_keys = sorted(set(cam1_by_key) & set(cam2_by_key), key=lambda s: int(s) if s.isdigit() else s)

    pairs = [(cam1_by_key[k], cam2_by_key[k]) for k in common_keys]
    if not pairs:
        raise FileNotFoundError(
            f"No matching png pairs found under {cam1_dir} and {cam2_dir}."
        )

    if len(pairs) != len(cam1_images) or len(pairs) != len(cam2_images):
        print(
            "[WARN] Some images do not have a matching pair: "
            f"cam1={len(cam1_images)}, cam2={len(cam2_images)}, matched={len(pairs)}"
        )

    return pairs


def _extract_common_points(
    board,
    detector_left,
    detector_right,
    left_img: np.ndarray,
    right_img: np.ndarray,
    min_common_corners: int,
):
    corners_l, ids_l, _, _ = detector_left.detectBoard(left_img)
    corners_r, ids_r, _, _ = detector_right.detectBoard(right_img)

    if ids_l is None or ids_r is None:
        return None, None, None, 0, 0, 0

    ids_l_list = _flatten_ids(ids_l)
    ids_r_list = _flatten_ids(ids_r)
    common_ids = sorted(set(ids_l_list) & set(ids_r_list))

    if len(common_ids) < min_common_corners:
        return None, None, None, len(ids_l_list), len(ids_r_list), len(common_ids)

    idx_l = {cid: i for i, cid in enumerate(ids_l_list)}
    idx_r = {cid: i for i, cid in enumerate(ids_r_list)}

    common_corners_l = np.array(
        [corners_l[idx_l[cid]][0] for cid in common_ids],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    common_corners_r = np.array(
        [corners_r[idx_r[cid]][0] for cid in common_ids],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    common_ids_arr = np.array(common_ids, dtype=np.int32).reshape(-1, 1)

    obj_points, img_points_l = board.matchImagePoints(common_corners_l, common_ids_arr)
    _, img_points_r = board.matchImagePoints(common_corners_r, common_ids_arr)

    if obj_points is None or img_points_l is None or img_points_r is None:
        return None, None, None, len(ids_l_list), len(ids_r_list), len(common_ids)

    return (
        obj_points.astype(np.float32),
        img_points_l.astype(np.float32),
        img_points_r.astype(np.float32),
        len(ids_l_list),
        len(ids_r_list),
        len(common_ids),
    )


def _stereo_calibrate(
    objpoints,
    imgpoints_l,
    imgpoints_r,
    camera_matrix_l,
    dist_l,
    camera_matrix_r,
    dist_r,
    image_size_l,
):
    criteria = (
        cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
        200,
        1e-7,
    )
    return cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        camera_matrix_l.copy(),
        dist_l.copy(),
        camera_matrix_r.copy(),
        dist_r.copy(),
        image_size_l,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )


def _pair_stereo_residuals(
    objpoints,
    imgpoints_l,
    imgpoints_r,
    camera_matrix_l,
    dist_l,
    camera_matrix_r,
    dist_r,
    R,
    T,
) -> np.ndarray:
    residuals = []
    for objp_i, imgp_l_i, imgp_r_i in zip(objpoints, imgpoints_l, imgpoints_r):
        ok_l, rvec_l, tvec_l = cv2.solvePnP(
            objp_i,
            imgp_l_i,
            camera_matrix_l,
            dist_l,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        ok_r, rvec_r, tvec_r = cv2.solvePnP(
            objp_i,
            imgp_r_i,
            camera_matrix_r,
            dist_r,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok_l or not ok_r:
            residuals.append(float("inf"))
            continue

        R_l, _ = cv2.Rodrigues(rvec_l)
        R_r, _ = cv2.Rodrigues(rvec_r)

        predicted_R_r = R @ R_l
        predicted_t_r = R @ tvec_l + T

        rot_delta = predicted_R_r @ R_r.T
        rot_vec, _ = cv2.Rodrigues(rot_delta)
        rot_err_deg = float(np.linalg.norm(rot_vec) * 180.0 / np.pi)
        trans_err_mm = float(np.linalg.norm(predicted_t_r - tvec_r))

        residuals.append(trans_err_mm + rot_err_deg)

    return np.asarray(residuals, dtype=np.float64)


def _filter_outliers(values: np.ndarray, std_scale: float) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return finite
    mean = float(np.mean(values[finite]))
    std = float(np.std(values[finite]))
    if std <= 1e-12:
        return finite
    return finite & (values <= mean + std_scale * std)


def main():
    parser = argparse.ArgumentParser(
        description="Fit stereo extrinsics from ChArUco stereo image pairs."
    )
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--square-length-mm", type=float, default=None)
    parser.add_argument("--marker-length-mm", type=float, default=None)
    parser.add_argument("--min-common-corners", type=int, default=None)
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    settings = StereoFitSettings()
    overrides = {}
    if args.image_root is not None:
        overrides["image_root"] = args.image_root
    if args.square_length_mm is not None:
        overrides["square_length_mm"] = args.square_length_mm
    if args.marker_length_mm is not None:
        overrides["marker_length_mm"] = args.marker_length_mm
    if args.min_common_corners is not None:
        overrides["min_common_corners"] = args.min_common_corners
    if args.output_npz is not None:
        overrides["output_npz"] = args.output_npz
    if args.report_json is not None:
        overrides["report_json"] = args.report_json
    if overrides:
        settings = replace(settings, **overrides)

    print(
        "[INFO] ChArUco board: "
        f"{settings.squares_x}x{settings.squares_y}, "
        f"square={settings.square_length_mm:.3f} mm, "
        f"marker={settings.marker_length_mm:.3f} mm"
    )
    pairs = _find_image_pairs(settings)
    print(f"[INFO] Found {len(pairs)} stereo image pairs.")

    camera_matrix_l, dist_l = _load_intrinsic(settings.left_intrinsic_npz, "left")
    camera_matrix_r, dist_r = _load_intrinsic(settings.right_intrinsic_npz, "right")
    print(f"[INFO] Loaded intrinsics:\n  {settings.left_intrinsic_npz}\n  {settings.right_intrinsic_npz}")

    board = _make_charuco_board(settings)
    detector_l = _make_detector(board, camera_matrix_l, dist_l)
    detector_r = _make_detector(board, camera_matrix_r, dist_r)

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []
    used_left_images = []
    used_right_images = []
    per_pair_common_counts = []
    rejected_pairs = []
    image_size_l = None
    image_size_r = None

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

        current_size_l = (img_l.shape[1], img_l.shape[0])
        current_size_r = (img_r.shape[1], img_r.shape[0])
        image_size_l = current_size_l if image_size_l is None else image_size_l
        image_size_r = current_size_r if image_size_r is None else image_size_r

        if current_size_l != image_size_l or current_size_r != image_size_r:
            rejected_pairs.append(
                {
                    "left": str(left_path),
                    "right": str(right_path),
                    "reason": "image_size_mismatch",
                    "left_size": current_size_l,
                    "right_size": current_size_r,
                }
            )
            continue

        objp_i, imgp_l_i, imgp_r_i, n_l, n_r, n_common = _extract_common_points(
            board,
            detector_l,
            detector_r,
            img_l,
            img_r,
            settings.min_common_corners,
        )

        if objp_i is None:
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
            print(
                f"[WARN] rejected {left_path.name}: "
                f"left={n_l}, right={n_r}, common={n_common}"
            )
            continue

        objpoints.append(objp_i)
        imgpoints_l.append(imgp_l_i)
        imgpoints_r.append(imgp_r_i)
        used_left_images.append(str(left_path))
        used_right_images.append(str(right_path))
        per_pair_common_counts.append(n_common)
        print(f"[OK] {left_path.name}: common corners={n_common}")

    print(f"[INFO] Valid pairs: {len(objpoints)}/{len(pairs)}")
    if len(objpoints) < settings.min_valid_pairs:
        raise RuntimeError(
            f"Not enough valid pairs: {len(objpoints)}. "
            f"Need at least {settings.min_valid_pairs}."
        )

    if image_size_l != image_size_r:
        print(
            "[WARN] Left and right image sizes differ. stereoCalibrate with fixed "
            "intrinsics can still estimate R/T, but stereoRectify preview is skipped."
        )

    print("[INFO] Running initial stereoCalibrate...")
    rms, _, _, _, _, R, T, E, F = _stereo_calibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        camera_matrix_l,
        dist_l,
        camera_matrix_r,
        dist_r,
        image_size_l,
    )

    residuals = _pair_stereo_residuals(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        camera_matrix_l,
        dist_l,
        camera_matrix_r,
        dist_r,
        R,
        T,
    )
    keep_mask = _filter_outliers(residuals, settings.outlier_std_scale)

    if keep_mask.sum() < len(objpoints) and keep_mask.sum() >= settings.min_valid_pairs:
        print(
            f"[INFO] Removing outlier pairs: keep={int(keep_mask.sum())}/{len(objpoints)}"
        )
        for keep, path, residual in zip(keep_mask, used_left_images, residuals):
            if not keep:
                print(f"[WARN] outlier removed: {Path(path).name}, residual={residual:.3f}")

        objpoints_f = [v for v, keep in zip(objpoints, keep_mask) if keep]
        imgpoints_l_f = [v for v, keep in zip(imgpoints_l, keep_mask) if keep]
        imgpoints_r_f = [v for v, keep in zip(imgpoints_r, keep_mask) if keep]
        used_left_f = [v for v, keep in zip(used_left_images, keep_mask) if keep]
        used_right_f = [v for v, keep in zip(used_right_images, keep_mask) if keep]
        common_counts_f = [v for v, keep in zip(per_pair_common_counts, keep_mask) if keep]

        print("[INFO] Running final stereoCalibrate...")
        rms, _, _, _, _, R, T, E, F = _stereo_calibrate(
            objpoints_f,
            imgpoints_l_f,
            imgpoints_r_f,
            camera_matrix_l,
            dist_l,
            camera_matrix_r,
            dist_r,
            image_size_l,
        )
        objpoints = objpoints_f
        imgpoints_l = imgpoints_l_f
        imgpoints_r = imgpoints_r_f
        used_left_images = used_left_f
        used_right_images = used_right_f
        per_pair_common_counts = common_counts_f
    else:
        print("[INFO] No outlier removal applied.")

    baseline_mm = float(np.linalg.norm(T))
    print("\n=== Stereo ChArUco calibration result ===")
    print(f"RMS reprojection error: {rms:.4f} px")
    print(f"Valid pairs used      : {len(objpoints)}")
    print(f"Mean common corners   : {float(np.mean(per_pair_common_counts)):.1f}")
    print(f"Baseline |T|          : {baseline_mm:.3f} mm")
    print("R:")
    print(R)
    print("T [mm]:")
    print(T)

    R1 = R2 = P1 = P2 = Q = None
    roi1 = roi2 = None
    if image_size_l == image_size_r:
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            camera_matrix_l,
            dist_l,
            camera_matrix_r,
            dist_r,
            image_size_l,
            R,
            T,
        )

    output_npz = Path(settings.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        camera_matrix_l=camera_matrix_l,
        dist_l=dist_l,
        camera_matrix_r=camera_matrix_r,
        dist_r=dist_r,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        roi1=roi1,
        roi2=roi2,
        rms=float(rms),
        baseline_mm=baseline_mm,
        image_size_l=np.array(image_size_l, dtype=np.int32),
        image_size_r=np.array(image_size_r, dtype=np.int32),
        board_squares=np.array([settings.squares_x, settings.squares_y], dtype=np.int32),
        square_length_mm=float(settings.square_length_mm),
        marker_length_mm=float(settings.marker_length_mm),
        used_left_images=np.array(used_left_images),
        used_right_images=np.array(used_right_images),
        common_corner_counts=np.array(per_pair_common_counts, dtype=np.int32),
    )

    report = {
        "settings": asdict(settings),
        "rms_px": float(rms),
        "baseline_mm": baseline_mm,
        "valid_pairs_used": len(objpoints),
        "total_pairs_found": len(pairs),
        "image_size_l": image_size_l,
        "image_size_r": image_size_r,
        "common_corner_counts": per_pair_common_counts,
        "used_left_images": used_left_images,
        "used_right_images": used_right_images,
        "rejected_pairs": rejected_pairs,
    }
    Path(settings.report_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[SUCCESS] Saved stereo parameters: {output_npz}")
    print(f"[SUCCESS] Saved report: {settings.report_json}")


if __name__ == "__main__":
    main()
