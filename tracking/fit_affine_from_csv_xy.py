# import os
# import csv
# import json
# import math
# import random
# from typing import Tuple

# import numpy as np
# import cv2

# # (u,v)と(x,y)の対応csvからアフィン変換行列を推定し、JSON保存するスクリプト

# # ===== 設定 =====
# CSV_PATH = "calib_points_uv_xy.csv"     # 取得したCSV
# OUT_JSON = "affine_uv_to_xy.json"       # 推定結果の保存先
# RANSAC_REPROJ_THRESH_PX = 3.0           # RANSACのしきい値（大きいほど外れ値に寛容）
# TEST_RATIO = 0.2                        # 80/20で検証
# SEED = 42                               # 再現性


# def read_points(csv_path: str):
#     """
#     CSVから対応点を読み込む。
#     必要列: u_mean, v_mean, x_cmd_mm, y_cmd_mm
#     """
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")

#     uv = []
#     xy = []
#     rows = []

#     with open(csv_path, "r", encoding="utf-8", newline="") as f:
#         reader = csv.DictReader(f)
#         required = {"u_mean", "v_mean", "x_cmd_mm", "y_cmd_mm"}
#         if not required.issubset(set(reader.fieldnames or [])):
#             raise ValueError(
#                 f"CSV列が想定と違います。必要: {sorted(required)} / 実際: {reader.fieldnames}"
#             )
#         for r in reader:
#             try:
#                 u = float(r["u_mean"])
#                 v = float(r["v_mean"])
#                 x = float(r["x_cmd_mm"])
#                 y = float(r["y_cmd_mm"])
#             except ValueError:
#                 continue
#             uv.append([u, v])
#             xy.append([x, y])
#             rows.append(r)

#     uv = np.array(uv, dtype=np.float32)
#     xy = np.array(xy, dtype=np.float32)

#     if len(uv) < 6:
#         raise ValueError(f"点が少なすぎます（{len(uv)}点）。最低6点以上必要です。")

#     return uv, xy, rows


# def fit_affine(uv: np.ndarray, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     (u,v)->(x,y) のアフィン行列 A(2x3) を推定
#     """
#     A, inliers = cv2.estimateAffine2D(
#         uv,
#         xy,
#         method=cv2.RANSAC,
#         ransacReprojThreshold=RANSAC_REPROJ_THRESH_PX,
#         maxIters=5000,
#         confidence=0.99,
#         refineIters=10,
#     )
#     if A is None:
#         raise RuntimeError("estimateAffine2D が失敗しました。点配置が悪い/外れ値が多い可能性があります。")
#     if inliers is None:
#         inliers = np.ones((len(uv), 1), dtype=np.uint8)
#     return A, inliers


# def apply_affine(A: np.ndarray, uv: np.ndarray) -> np.ndarray:
#     """
#     A(2x3) と uv(Nx2) から xy_hat(Nx2) を計算
#     """
#     uv1 = np.hstack([uv, np.ones((len(uv), 1), dtype=np.float32)])  # Nx3
#     xy_hat = (A @ uv1.T).T  # Nx2
#     return xy_hat.astype(np.float32)


# def error_stats(xy_true: np.ndarray, xy_pred: np.ndarray):
#     """
#     2D誤差（mm）を集計
#     """
#     diff = xy_pred - xy_true
#     err = np.sqrt(np.sum(diff**2, axis=1))  # N
#     return {
#         "mean_mm": float(np.mean(err)),
#         "median_mm": float(np.median(err)),
#         "rmse_mm": float(np.sqrt(np.mean(err**2))),
#         "max_mm": float(np.max(err)),
#         "min_mm": float(np.min(err)),
#     }, err, diff


# def train_test_split(n: int, test_ratio: float, seed: int):
#     idx = list(range(n))
#     random.Random(seed).shuffle(idx)
#     n_test = max(1, int(round(n * test_ratio)))
#     test_idx = idx[:n_test]
#     train_idx = idx[n_test:]
#     return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


# def main():
#     uv, xy, rows = read_points(CSV_PATH)
#     n = len(uv)
#     print(f"[INFO] Loaded points: {n}")

#     # 80/20 ホールドアウト評価
#     train_idx, test_idx = train_test_split(n, TEST_RATIO, SEED)
#     uv_tr, xy_tr = uv[train_idx], xy[train_idx]
#     uv_te, xy_te = uv[test_idx], xy[test_idx]

#     # 学習で推定
#     A_tr, inliers_tr = fit_affine(uv_tr, xy_tr)
#     xy_tr_hat = apply_affine(A_tr, uv_tr)
#     tr_stats, tr_err, tr_diff = error_stats(xy_tr, xy_tr_hat)

#     # テスト評価
#     xy_te_hat = apply_affine(A_tr, uv_te)
#     te_stats, te_err, te_diff = error_stats(xy_te, xy_te_hat)

#     print("\n===== Affine matrix A (trained on 80%) =====")
#     print(A_tr)

#     print("\n===== Train error (mm) =====")
#     for k, v in tr_stats.items():
#         print(f"{k:>10}: {v:.4f}")

#     print("\n===== Test error (mm) =====")
#     for k, v in te_stats.items():
#         print(f"{k:>10}: {v:.4f}")

#     # 全点で最終推定（保存用）
#     A_all, inliers_all = fit_affine(uv, xy)
#     xy_all_hat = apply_affine(A_all, uv)
#     all_stats, all_err, all_diff = error_stats(xy, xy_all_hat)

#     print("\n===== Final A (fit on ALL points) =====")
#     print(A_all)

#     inlier_count = int(np.sum(inliers_all))
#     print(f"\n[INFO] Inliers (ALL fit): {inlier_count} / {n}")

#     print("\n===== ALL error (mm) =====")
#     for k, v in all_stats.items():
#         print(f"{k:>10}: {v:.4f}")

#     # 点ごとの残差表示（上位10）
#     order = np.argsort(-all_err)  # 大きい順
#     print("\n===== Worst 10 points (ALL fit) =====")
#     for rank, i in enumerate(order[: min(10, n)], start=1):
#         r = rows[i]
#         u, v = uv[i]
#         x_t, y_t = xy[i]
#         x_p, y_p = xy_all_hat[i]
#         print(
#             f"{rank:>2}. err={all_err[i]:.3f}mm | "
#             f"uv=({u:.1f},{v:.1f}) -> "
#             f"true=({x_t:.2f},{y_t:.2f}) pred=({x_p:.2f},{y_p:.2f}) | "
#             f"time={r.get('timestamp','')}"
#         )

#     # JSON保存（あとで制御コードで読み込めるように）
#     out = {
#         "csv_path": os.path.abspath(CSV_PATH),
#         "A_2x3": A_all.tolist(),
#         "inliers": inliers_all.reshape(-1).astype(int).tolist(),
#         "error_all_mm": all_stats,
#         "ransac_reproj_threshold_px": RANSAC_REPROJ_THRESH_PX,
#         "note": "xy(mm) = A @ [u, v, 1].T  (u,v are pixel coords)",
#     }
#     with open(OUT_JSON, "w", encoding="utf-8") as f:
#         json.dump(out, f, indent=2, ensure_ascii=False)

#     print(f"\n[INFO] Saved affine to: {OUT_JSON}")

#     # 追加：簡易チェック（推定値が極端でないか）
#     det = A_all[0, 0] * A_all[1, 1] - A_all[0, 1] * A_all[1, 0]
#     print(f"[DEBUG] Determinant of 2x2 submatrix: {det:.6f}")


# if __name__ == "__main__":
#     main()

import os
import csv
import json
import random
from typing import Tuple

import numpy as np
import cv2

# 補正済み(u,v)と AUTD平面(x,y) の対応CSVから
# アフィン変換行列を推定して JSON 保存するスクリプト
#
# 前提:
#   - CSV中の u_mean, v_mean は「歪み補正済み画像上のピクセル座標」
#   - よって、このスクリプトでは undistort は行わない
#   - 出力 A は
#         [x_mm, y_mm]^T = A @ [u_ud, v_ud, 1]^T
#     を表す

# 設定
CSV_PATH = "./tracking/xy_calib_points.csv"
OUT_JSON = "./tracking/affine_uv_to_xy.json"

# estimateAffine2D の RANSACしきい値
# OpenCV上は「出力座標系」での誤差閾値として扱われるので、
# 今回は mm のつもりで使う
RANSAC_REPROJ_THRESH_MM = 3.0

TEST_RATIO = 0.2
SEED = 42


def read_points(csv_path: str):
    """
    CSVから対応点を読み込む。
    必要列:
      - u_mean, v_mean : 補正済みピクセル座標
      - x_cmd_mm, y_cmd_mm : AUTD中心からのオフセット[mm]
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")

    uv = []
    xy = []
    rows = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"u_mean", "v_mean", "x_cmd_mm", "y_cmd_mm"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV列が想定と違います。必要: {sorted(required)} / 実際: {reader.fieldnames}"
            )

        for r in reader:
            try:
                u = float(r["u_mean"])
                v = float(r["v_mean"])
                x = float(r["x_cmd_mm"])
                y = float(r["y_cmd_mm"])
            except (ValueError, TypeError):
                continue

            if not np.isfinite([u, v, x, y]).all():
                continue

            uv.append([u, v])
            xy.append([x, y])
            rows.append(r)

    uv = np.array(uv, dtype=np.float32)
    xy = np.array(xy, dtype=np.float32)

    if len(uv) < 6:
        raise ValueError(f"点が少なすぎます（{len(uv)}点）。最低6点以上必要です。")

    return uv, xy, rows


def fit_affine(uv: np.ndarray, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    補正済み (u,v) -> AUTD平面 (x,y) のアフィン行列 A(2x3) を推定
    """
    A, inliers = cv2.estimateAffine2D(
        uv,
        xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESH_MM,
        maxIters=5000,
        confidence=0.99,
        refineIters=10,
    )

    if A is None:
        raise RuntimeError(
            "estimateAffine2D が失敗しました。"
            "点配置が悪いか、外れ値が多い可能性があります。"
        )

    if inliers is None:
        inliers = np.ones((len(uv), 1), dtype=np.uint8)

    return A.astype(np.float32), inliers.astype(np.uint8)


def apply_affine(A: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """
    A(2x3) と uv(Nx2) から xy_hat(Nx2) を計算
    uv は補正済みピクセル座標
    """
    uv1 = np.hstack([uv, np.ones((len(uv), 1), dtype=np.float32)])  # Nx3
    xy_hat = (A @ uv1.T).T  # Nx2
    return xy_hat.astype(np.float32)


def error_stats(xy_true: np.ndarray, xy_pred: np.ndarray):
    """
    2D誤差 [mm] を集計
    """
    diff = xy_pred - xy_true
    err = np.sqrt(np.sum(diff ** 2, axis=1))
    return {
        "mean_mm": float(np.mean(err)),
        "median_mm": float(np.median(err)),
        "rmse_mm": float(np.sqrt(np.mean(err ** 2))),
        "max_mm": float(np.max(err)),
        "min_mm": float(np.min(err)),
    }, err, diff


def train_test_split(n: int, test_ratio: float, seed: int):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = max(1, int(round(n * test_ratio)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def main():
    uv, xy, rows = read_points(CSV_PATH)
    n = len(uv)
    print(f"[INFO] Loaded points: {n}")
    print("[INFO] u,v are assumed to be UNDISTORTED pixel coordinates.")

    # 80/20 ホールドアウト評価
    train_idx, test_idx = train_test_split(n, TEST_RATIO, SEED)
    uv_tr, xy_tr = uv[train_idx], xy[train_idx]
    uv_te, xy_te = uv[test_idx], xy[test_idx]

    # 学習
    A_tr, inliers_tr = fit_affine(uv_tr, xy_tr)
    xy_tr_hat = apply_affine(A_tr, uv_tr)
    tr_stats, tr_err, tr_diff = error_stats(xy_tr, xy_tr_hat)

    # テスト
    xy_te_hat = apply_affine(A_tr, uv_te)
    te_stats, te_err, te_diff = error_stats(xy_te, xy_te_hat)

    print("\n===== Affine matrix A (trained on 80%) =====")
    print(A_tr)

    print("\n===== Train error (mm) =====")
    for k, v in tr_stats.items():
        print(f"{k:>10}: {v:.4f}")

    print("\n===== Test error (mm) =====")
    for k, v in te_stats.items():
        print(f"{k:>10}: {v:.4f}")

    # 全点で最終推定
    A_all, inliers_all = fit_affine(uv, xy)
    xy_all_hat = apply_affine(A_all, uv)
    all_stats, all_err, all_diff = error_stats(xy, xy_all_hat)

    print("\n===== Final A (fit on ALL points) =====")
    print(A_all)

    inlier_count = int(np.sum(inliers_all))
    print(f"\n[INFO] Inliers (ALL fit): {inlier_count} / {n}")

    print("\n===== ALL error (mm) =====")
    for k, v in all_stats.items():
        print(f"{k:>10}: {v:.4f}")

    # 残差の大きい点を表示
    order = np.argsort(-all_err)
    print("\n===== Worst 10 points (ALL fit) =====")
    for rank, i in enumerate(order[: min(10, n)], start=1):
        r = rows[i]
        u, v = uv[i]
        x_t, y_t = xy[i]
        x_p, y_p = xy_all_hat[i]
        print(
            f"{rank:>2}. err={all_err[i]:.3f}mm | "
            f"uv_ud=({u:.1f},{v:.1f}) -> "
            f"true=({x_t:.2f},{y_t:.2f}) pred=({x_p:.2f},{y_p:.2f}) | "
            f"time={r.get('timestamp', '')}"
        )

    # JSON保存
    out = {
        "csv_path": os.path.abspath(CSV_PATH),
        "A_2x3": A_all.tolist(),
        "inliers": inliers_all.reshape(-1).astype(int).tolist(),
        "error_all_mm": all_stats,
        "ransac_reproj_threshold_mm": RANSAC_REPROJ_THRESH_MM,
        "input_uv_type": "undistorted_pixel",
        "xy_definition": "AUTD center offset in mm",
        "equation": "[x_mm, y_mm]^T = A @ [u_ud, v_ud, 1]^T",
        "note": "This affine was fitted on undistorted pixel coordinates. Do NOT undistort again before applying A.",
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Saved affine to: {OUT_JSON}")

    det = A_all[0, 0] * A_all[1, 1] - A_all[0, 1] * A_all[1, 0]
    print(f"[DEBUG] Determinant of 2x2 submatrix: {det:.6f}")


if __name__ == "__main__":
    main()