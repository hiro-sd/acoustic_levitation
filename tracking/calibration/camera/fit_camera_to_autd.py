"""
Fit a rigid transform from stereo cam1 coordinates to AUTD coordinates.

入力CSVの必須列:
    cam_x_mm, cam_y_mm, cam_z_mm, autd_x_mm, autd_y_mm, autd_z_mm

出力:
    calibration/stereo_camera_to_autd.npz

変換式:
    p_autd = R @ p_cam1 + t

注意:
    ステレオ復元結果はすでにmmスケールなので、ここではスケールなしの剛体変換を推定する。
    もし既知点とステレオ点のスケールが合わない場合は、先にステレオ/ボード寸法を疑う。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


TRACKING_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CameraToAutdFitSettings:
    input_csv: str = str(TRACKING_ROOT / "calibration" / "camera_to_autd_points.csv")
    output_npz: str = str(TRACKING_ROOT / "calibration" / "stereo_camera_to_autd.npz")
    report_json: str = str(TRACKING_ROOT / "calibration" / "stereo_camera_to_autd_report.json")


REQUIRED_COLUMNS = [
    "cam_x_mm",
    "cam_y_mm",
    "cam_z_mm",
    "autd_x_mm",
    "autd_y_mm",
    "autd_z_mm",
]


def _load_correspondences(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = []
    cam_points = []
    autd_points = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header.")

        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise KeyError(
                f"{csv_path} is missing required columns: {missing}. "
                f"Required columns are: {REQUIRED_COLUMNS}"
            )

        for idx, row in enumerate(reader, start=1):
            if not any((value or "").strip() for value in row.values()):
                continue

            try:
                cam = np.array(
                    [
                        float(row["cam_x_mm"]),
                        float(row["cam_y_mm"]),
                        float(row["cam_z_mm"]),
                    ],
                    dtype=np.float64,
                )
                autd = np.array(
                    [
                        float(row["autd_x_mm"]),
                        float(row["autd_y_mm"]),
                        float(row["autd_z_mm"]),
                    ],
                    dtype=np.float64,
                )
            except Exception as e:
                raise ValueError(f"Invalid numeric value at CSV row {idx}: {row}") from e

            cam_points.append(cam)
            autd_points.append(autd)
            rows.append(dict(row))

    if len(cam_points) < 3:
        raise ValueError(
            f"At least 3 non-collinear points are required, got {len(cam_points)}."
        )

    return np.vstack(cam_points), np.vstack(autd_points), rows


def _fit_rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Kabsch fit for dst ~= R @ src + t.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected src/dst shape (N,3), got {src.shape} and {dst.shape}")

    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid

    # 点がほぼ一直線/一点だと回転が安定しない。
    rank = np.linalg.matrix_rank(src_centered)
    if rank < 2:
        raise ValueError(
            "Source points are degenerate. Use points spread over a 3D volume, "
            "not only one point or a straight line."
        )

    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = dst_centroid - R @ src_centroid
    return R, t


def _summarize_errors(errors: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "count": int(errors.size),
        "mean_mm": float(np.mean(errors)),
        "median_mm": float(np.median(errors)),
        "rmse_mm": float(np.sqrt(np.mean(errors * errors))),
        "p95_mm": float(np.percentile(errors, 95)),
        "max_mm": float(np.max(errors)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fit rigid transform p_autd = R @ p_cam1 + t from known correspondences."
    )
    parser.add_argument("--input-csv", default=CameraToAutdFitSettings.input_csv)
    parser.add_argument("--output-npz", default=CameraToAutdFitSettings.output_npz)
    parser.add_argument("--report-json", default=CameraToAutdFitSettings.report_json)
    args = parser.parse_args()

    settings = CameraToAutdFitSettings(
        input_csv=args.input_csv,
        output_npz=args.output_npz,
        report_json=args.report_json,
    )

    cam_points, autd_points, rows = _load_correspondences(settings.input_csv)
    R, t = _fit_rigid_transform(cam_points, autd_points)

    predicted = (R @ cam_points.T).T + t
    residual_vectors = predicted - autd_points
    residual_norms = np.linalg.norm(residual_vectors, axis=1)
    summary = _summarize_errors(residual_norms)

    print("\n=== cam1 -> AUTD rigid transform ===")
    print(f"Input points : {len(cam_points)}")
    print(f"RMSE         : {summary['rmse_mm']:.3f} mm")
    print(f"95 percentile: {summary['p95_mm']:.3f} mm")
    print(f"Max error    : {summary['max_mm']:.3f} mm")
    print("R:")
    print(R)
    print("t [mm]:")
    print(t)

    output_npz = Path(settings.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        R=R,
        t=t,
        cam_points=cam_points,
        autd_points=autd_points,
        predicted_autd_points=predicted,
        residual_vectors=residual_vectors,
        residual_norms=residual_norms,
        rmse_mm=summary["rmse_mm"],
        equation="p_autd = R @ p_cam1 + t",
    )

    report = {
        "settings": asdict(settings),
        "equation": "p_autd = R @ p_cam1 + t",
        "error_summary": summary,
        "R": R.tolist(),
        "t": t.tolist(),
        "points": [
            {
                "row": row,
                "cam": cam.tolist(),
                "autd": autd.tolist(),
                "predicted_autd": pred.tolist(),
                "residual": residual.tolist(),
                "error_mm": float(err),
            }
            for row, cam, autd, pred, residual, err in zip(
                rows,
                cam_points,
                autd_points,
                predicted,
                residual_vectors,
                residual_norms,
            )
        ],
    }
    Path(settings.report_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[SUCCESS] Saved transform: {output_npz}")
    print(f"[SUCCESS] Saved report: {settings.report_json}")


if __name__ == "__main__":
    main()
