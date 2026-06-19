import os
import csv
import json
from typing import Tuple
import numpy as np


# 回転済み side camera の CSV から
# v_mean -> z_mm の線形変換を推定して JSON 保存する
#
# 前提:
#   - CSV中の v_mean は「歪み補正 + 回転後画像」の縦座標
#   - z_cmd_mm は AUTD に与えた高さ指令値 [mm]
#
# モデル:
#   z_mm = a * v + b

CSV_PATH = "./tracking/z_calib_points.csv"
OUT_JSON = "./tracking/affine_v_to_z.json"


def read_points(csv_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    v_list = []
    z_list = []
    rows = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {"v_mean", "z_cmd_mm"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV columns mismatch. required={sorted(required)}, actual={reader.fieldnames}"
            )

        for r in reader:
            try:
                v = float(r["v_mean"])
                z = float(r["z_cmd_mm"])
            except (ValueError, TypeError):
                continue

            if not np.isfinite([v, z]).all():
                continue

            v_list.append(v)
            z_list.append(z)
            rows.append(r)

    v_arr = np.array(v_list, dtype=np.float64)
    z_arr = np.array(z_list, dtype=np.float64)

    if len(v_arr) < 2:
        raise ValueError(f"Not enough valid points: {len(v_arr)}")

    return v_arr, z_arr, rows


def fit_linear(v: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """
    z = a * v + b
    """
    a, b = np.polyfit(v, z, deg=1)
    return float(a), float(b)


def predict_z(v: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * v + b


def error_stats(z_true: np.ndarray, z_pred: np.ndarray):
    err = z_pred - z_true
    abs_err = np.abs(err)
    stats = {
        "mean_abs_mm": float(np.mean(abs_err)),
        "median_abs_mm": float(np.median(abs_err)),
        "rmse_mm": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_mm": float(np.max(abs_err)),
        "min_abs_mm": float(np.min(abs_err)),
    }
    return stats, err, abs_err


def main():
    v, z, rows = read_points(CSV_PATH)
    print(f"[INFO] Loaded points: {len(v)}")
    print("[INFO] Assumption: v_mean is from undistorted + rotated image.")

    a, b = fit_linear(v, z)
    z_hat = predict_z(v, a, b)

    stats, err, abs_err = error_stats(z, z_hat)

    print("\n===== Linear fit =====")
    print(f"z_mm = a * v + b")
    print(f"a = {a:.8f}")
    print(f"b = {b:.8f}")

    print("\n===== Error stats =====")
    for k, val in stats.items():
        print(f"{k:>14}: {val:.4f}")

    # 外れ値っぽい点を確認しやすいように大きい順で表示
    order = np.argsort(-abs_err)

    print("\n===== Worst points =====")
    for rank, i in enumerate(order[: min(10, len(v))], start=1):
        r = rows[i]
        print(
            f"{rank:>2}. "
            f"v={v[i]:.2f}, "
            f"true_z={z[i]:.2f}, "
            f"pred_z={z_hat[i]:.2f}, "
            f"err={err[i]:+.3f} mm, "
            f"time={r.get('timestamp', '')}"
        )

    out = {
        "csv_path": os.path.abspath(CSV_PATH),
        "equation": "z_mm = a * v_rot_ud + b",
        "a": a,
        "b": b,
        "input_axis": "v",
        "input_v_type": "undistorted_rotated_pixel",
        "camera_role": "side_camera_for_z",
        "error_mm": stats,
        "note": (
            "This fit assumes the image was undistorted first and then rotated, "
            "so physical z corresponds to the image v-axis."
        ),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Saved JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()