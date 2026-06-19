# import cv2
# import numpy as np
# import glob
# import os

# ステレオキャリブレーション用のスクリプト (うまくいっていない)

# # 設定
# PATTERN_SIZE = (9, 6)
# SQUARE_SIZE_MM = 24.0

# LEFT_DIR = "./tracking/calibration/stereo_images/cam1"
# RIGHT_DIR = "./tracking/calibration/stereo_images/cam2"

# LEFT_INTRINSIC = "./tracking/calibration/intrinsic_cam1.npz"
# RIGHT_INTRINSIC = "./tracking/calibration/intrinsic_cam2.npz"

# OUTPUT_FILE = "./tracking/calibration/stereo_calibration_result.npz"

# # 外れペア除外のしきい値
# OUTLIER_STD_SCALE = 2.0

# # 内部パラメータ読み込み
# left_data = np.load(LEFT_INTRINSIC)
# right_data = np.load(RIGHT_INTRINSIC)

# mtx_l, dist_l = left_data["mtx"], left_data["dist"]
# mtx_r, dist_r = right_data["mtx"], right_data["dist"]

# # 3D座標
# objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
# objp *= SQUARE_SIZE_MM

# objpoints = []
# imgpoints_l = []
# imgpoints_r = []
# used_left_images = []
# used_right_images = []

# images_l = sorted(glob.glob(os.path.join(LEFT_DIR, "*.png")))
# images_r = sorted(glob.glob(os.path.join(RIGHT_DIR, "*.png")))

# assert len(images_l) == len(images_r), "左右画像の枚数が一致しません"

# print(f"[INFO] {len(images_l)} 組のペア画像を処理中...")

# def detect_chessboard_corners(gray, pattern_size):
#     if hasattr(cv2, "findChessboardCornersSB"):
#         ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
#         if ret:
#             return ret, corners
#     flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
#     return cv2.findChessboardCorners(gray, pattern_size, flags)

# for img_l_path, img_r_path in zip(images_l, images_r):
#     img_l = cv2.imread(img_l_path)
#     img_r = cv2.imread(img_r_path)

#     gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
#     gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

#     ret_l, corners_l = detect_chessboard_corners(gray_l, PATTERN_SIZE)
#     ret_r, corners_r = detect_chessboard_corners(gray_r, PATTERN_SIZE)

#     if ret_l and ret_r:
#         corners_l2 = cv2.cornerSubPix(
#             gray_l, corners_l, (11, 11), (-1, -1),
#             criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#         )
#         corners_r2 = cv2.cornerSubPix(
#             gray_r, corners_r, (11, 11), (-1, -1),
#             criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#         )

#         objpoints.append(objp)
#         imgpoints_l.append(corners_l2)
#         imgpoints_r.append(corners_r2)
#         used_left_images.append(img_l_path)
#         used_right_images.append(img_r_path)
#     else:
#         print(f"[WARN] ペアの検出に失敗しました: {os.path.basename(img_l_path)}")

# print(f"[INFO] 有効なペア数: {len(objpoints)}/{len(images_l)}")

# if len(objpoints) < 5:
#     print("[ERROR] ステレオキャリブレーションに必要な十分なペアがありません。")
#     exit()

# img_shape = gray_l.shape[::-1]

# # 1回目のステレオキャリブレーション
# print("[INFO] ステレオ計算中...")
# criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

# ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
#     objpoints, imgpoints_l, imgpoints_r,
#     mtx_l, dist_l, mtx_r, dist_r,
#     img_shape,
#     criteria=criteria_stereo,
#     flags=cv2.CALIB_FIX_INTRINSIC
# )

# # ペアごとの誤差を計算
# per_pair_errors = []
# rvec = np.zeros((3, 1), dtype=np.float64)
# tvec = np.zeros((3, 1), dtype=np.float64)

# for objp_i, corners_l_i, corners_r_i in zip(objpoints, imgpoints_l, imgpoints_r):
#     proj_l, _ = cv2.projectPoints(objp_i, rvec, tvec, mtx_l, dist_l)
#     proj_r, _ = cv2.projectPoints(objp_i, cv2.Rodrigues(R)[0], T, mtx_r, dist_r)

#     err_l = cv2.norm(corners_l_i, proj_l, cv2.NORM_L2) / len(proj_l)
#     err_r = cv2.norm(corners_r_i, proj_r, cv2.NORM_L2) / len(proj_r)
#     pair_err = (err_l + err_r) / 2.0
#     per_pair_errors.append(pair_err)

# per_pair_errors = np.array(per_pair_errors, dtype=np.float64)
# err_mean = float(np.mean(per_pair_errors))
# err_std = float(np.std(per_pair_errors))
# threshold = err_mean + OUTLIER_STD_SCALE * err_std

# print(f"[INFO] ペア誤差 平均: {err_mean:.4f}, 標準偏差: {err_std:.4f}, しきい値: {threshold:.4f}")

# filtered_objpoints = []
# filtered_imgpoints_l = []
# filtered_imgpoints_r = []
# filtered_left_images = []
# filtered_right_images = []

# for i, (objp_i, corners_l_i, corners_r_i, img_l_path, img_r_path, pair_err) in enumerate(
#     zip(objpoints, imgpoints_l, imgpoints_r, used_left_images, used_right_images, per_pair_errors)
# ):
#     if pair_err <= threshold:
#         filtered_objpoints.append(objp_i)
#         filtered_imgpoints_l.append(corners_l_i)
#         filtered_imgpoints_r.append(corners_r_i)
#         filtered_left_images.append(img_l_path)
#         filtered_right_images.append(img_r_path)
#     else:
#         print(f"[WARN] 外れペアを除外: {os.path.basename(img_l_path)} (error={pair_err:.4f})")

# print(f"[INFO] 外れ除外後の有効ペア数: {len(filtered_objpoints)}/{len(objpoints)}")

# if len(filtered_objpoints) < 5:
#     print("[ERROR] 外れ除外後に十分なペアが残りませんでした。")
#     exit()

# # 2回目のステレオキャリブレーション
# print("[INFO] 外れ除外後のステレオ再計算中...")
# ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
#     filtered_objpoints, filtered_imgpoints_l, filtered_imgpoints_r,
#     mtx_l, dist_l, mtx_r, dist_r,
#     img_shape,
#     criteria=criteria_stereo,
#     flags=cv2.CALIB_FIX_INTRINSIC
# )

# print("\n=== ステレオキャリブレーション結果 ===")
# print(f"ステレオ再投影誤差 (RMS): {ret:.4f}")
# print("回転行列 (R):")
# print(R)
# print("並進ベクトル (T) [mm]:")
# print(T)

# # Rectification
# R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
#     mtx_l, dist_l, mtx_r, dist_r, img_shape, R, T
# )

# # 保存
# np.savez(
#     OUTPUT_FILE,
#     R=R, T=T, E=E, F=F,
#     R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
#     roi1=roi1, roi2=roi2
# )
# print(f"\n[SUCCESS] 結果を {OUTPUT_FILE} に保存しました。")

# # Rectified preview
# sample_indices = [0, len(filtered_left_images) // 2, len(filtered_left_images) - 1]
# print("[INFO] 平行化プレビューを表示中。緑の線にチェッカーボードの角が左右で揃っていれば成功です。")

# for idx in sample_indices:
#     img_l = cv2.imread(filtered_left_images[idx])
#     img_r = cv2.imread(filtered_right_images[idx])

#     map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_shape, cv2.CV_16SC2)
#     map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_shape, cv2.CV_16SC2)

#     rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
#     rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

#     combined = np.hstack((rect_l, rect_r))
#     h, w = combined.shape[:2]

#     for y in range(0, h, 40):
#         cv2.line(combined, (0, y), (w, y), (0, 255, 0), 1)

#     cv2.imshow(f"Rectified Pair Preview {idx}", combined)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np
import glob
import os

# ステレオキャリブレーション用のスクリプト (ChArUco, うまくいっていない)

# 設定
LEFT_DIR = "./tracking/calibration/stereo_images/cam1"
RIGHT_DIR = "./tracking/calibration/stereo_images/cam2"

LEFT_INTRINSIC = "./tracking/calibration/intrinsic_charuco_cam1.npz"
RIGHT_INTRINSIC = "./tracking/calibration/intrinsic_charuco_cam2.npz"

OUTPUT_FILE = "./tracking/calibration/stereo_charuco_calibration_result.npz"

# 印刷した ChArUco ボードの設定
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH_MM = 38.0
MARKER_LENGTH_MM = 27.0
ARUCO_DICT = cv2.aruco.DICT_4X4_50

MIN_COMMON_CORNERS = 8
OUTLIER_STD_SCALE = 2.0


def flatten_ids(ids):
    return [int(x) for x in ids.flatten()]


def main():
    # 内部パラメータ
    left_data = np.load(LEFT_INTRINSIC, allow_pickle=True)
    right_data = np.load(RIGHT_INTRINSIC, allow_pickle=True)

    mtx_l = left_data["camera_matrix"]
    dist_l = left_data["dist_coeffs"]
    mtx_r = right_data["camera_matrix"]
    dist_r = right_data["dist_coeffs"]

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH_MM,
        MARKER_LENGTH_MM,
        dictionary
    )

    detector_params = cv2.aruco.DetectorParameters()

    # 内部パラメータを与えると detectBoard の補間が安定しやすい
    charuco_params_l = cv2.aruco.CharucoParameters()
    charuco_params_l.cameraMatrix = mtx_l
    charuco_params_l.distCoeffs = dist_l

    charuco_params_r = cv2.aruco.CharucoParameters()
    charuco_params_r.cameraMatrix = mtx_r
    charuco_params_r.distCoeffs = dist_r

    detector_l = cv2.aruco.CharucoDetector(board, charuco_params_l, detector_params)
    detector_r = cv2.aruco.CharucoDetector(board, charuco_params_r, detector_params)

    images_l = sorted(glob.glob(os.path.join(LEFT_DIR, "*.png")))
    images_r = sorted(glob.glob(os.path.join(RIGHT_DIR, "*.png")))
    assert len(images_l) == len(images_r), "左右画像の枚数が一致しません"

    print(f"[INFO] {len(images_l)} 組のペア画像を処理中...")

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []
    used_left_images = []
    used_right_images = []
    image_size = None

    for img_l_path, img_r_path in zip(images_l, images_r):
        img_l = cv2.imread(img_l_path)
        img_r = cv2.imread(img_r_path)

        if img_l is None or img_r is None:
            print(f"[WARN] 読み込み失敗: {os.path.basename(img_l_path)}")
            continue

        if image_size is None:
            image_size = (img_l.shape[1], img_l.shape[0])

        corners_l, ids_l, _, _ = detector_l.detectBoard(img_l)
        corners_r, ids_r, _, _ = detector_r.detectBoard(img_r)

        if ids_l is None or ids_r is None:
            print(f"[WARN] ChArUco検出失敗: {os.path.basename(img_l_path)}")
            continue

        ids_l_list = flatten_ids(ids_l)
        ids_r_list = flatten_ids(ids_r)

        common_ids = sorted(list(set(ids_l_list) & set(ids_r_list)))

        if len(common_ids) < MIN_COMMON_CORNERS:
            print(f"[WARN] 共通角不足: {os.path.basename(img_l_path)} ({len(common_ids)})")
            continue

        # 共通IDだけ抽出
        common_corners_l = []
        common_corners_r = []

        for cid in common_ids:
            idx_l = ids_l_list.index(cid)
            idx_r = ids_r_list.index(cid)
            common_corners_l.append(corners_l[idx_l][0])
            common_corners_r.append(corners_r[idx_r][0])

        common_corners_l = np.array(common_corners_l, dtype=np.float32).reshape(-1, 1, 2)
        common_corners_r = np.array(common_corners_r, dtype=np.float32).reshape(-1, 1, 2)
        common_ids_arr = np.array(common_ids, dtype=np.int32).reshape(-1, 1)

        objp_i, imgp_l_i = board.matchImagePoints(common_corners_l, common_ids_arr)
        _, imgp_r_i = board.matchImagePoints(common_corners_r, common_ids_arr)

        if objp_i is None or imgp_l_i is None or imgp_r_i is None:
            print(f"[WARN] 対応点生成失敗: {os.path.basename(img_l_path)}")
            continue

        objpoints.append(objp_i.astype(np.float32))
        imgpoints_l.append(imgp_l_i.astype(np.float32))
        imgpoints_r.append(imgp_r_i.astype(np.float32))
        used_left_images.append(img_l_path)
        used_right_images.append(img_r_path)

    print(f"[INFO] 有効なペア数: {len(objpoints)}/{len(images_l)}")

    if len(objpoints) < 5:
        print("[ERROR] ステレオキャリブレーションに十分なペアがありません。")
        return

    criteria_stereo = (
        cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
        100,
        1e-5
    )

    print("[INFO] 1回目のステレオ計算中...")
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        criteria=criteria_stereo,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    # 外れペア除外（簡易）
    per_pair_errors = []
    for objp_i, imgp_l_i, imgp_r_i in zip(objpoints, imgpoints_l, imgpoints_r):
        proj_l, _ = cv2.projectPoints(
            objp_i, np.zeros((3, 1)), np.zeros((3, 1)), mtx_l, dist_l
        )
        proj_r, _ = cv2.projectPoints(
            objp_i, cv2.Rodrigues(R)[0], T, mtx_r, dist_r
        )

        err_l = cv2.norm(imgp_l_i, proj_l, cv2.NORM_L2) / len(proj_l)
        err_r = cv2.norm(imgp_r_i, proj_r, cv2.NORM_L2) / len(proj_r)
        per_pair_errors.append((err_l + err_r) / 2.0)

    per_pair_errors = np.array(per_pair_errors, dtype=np.float64)
    err_mean = float(np.mean(per_pair_errors))
    err_std = float(np.std(per_pair_errors))
    threshold = err_mean + OUTLIER_STD_SCALE * err_std

    print(f"[INFO] ペア誤差 平均: {err_mean:.4f}, 標準偏差: {err_std:.4f}, しきい値: {threshold:.4f}")

    filtered_objpoints = []
    filtered_imgpoints_l = []
    filtered_imgpoints_r = []
    filtered_left_images = []
    filtered_right_images = []

    for objp_i, imgp_l_i, imgp_r_i, lp, rp, pe in zip(
        objpoints, imgpoints_l, imgpoints_r, used_left_images, used_right_images, per_pair_errors
    ):
        if pe <= threshold:
            filtered_objpoints.append(objp_i)
            filtered_imgpoints_l.append(imgp_l_i)
            filtered_imgpoints_r.append(imgp_r_i)
            filtered_left_images.append(lp)
            filtered_right_images.append(rp)
        else:
            print(f"[WARN] 外れペア除外: {os.path.basename(lp)} (error={pe:.4f})")

    print(f"[INFO] 外れ除外後の有効ペア数: {len(filtered_objpoints)}/{len(objpoints)}")

    if len(filtered_objpoints) < 5:
        print("[ERROR] 外れ除外後に十分なペアが残りません。")
        return

    print("[INFO] 2回目のステレオ計算中...")
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        filtered_objpoints,
        filtered_imgpoints_l,
        filtered_imgpoints_r,
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        criteria=criteria_stereo,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print("\n=== ChArUco ステレオキャリブレーション結果 ===")
    print(f"ステレオ再投影誤差 (RMS): {rms:.4f}")
    print("回転行列 (R):")
    print(R)
    print("並進ベクトル (T) [mm]:")
    print(T)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, image_size, R, T
    )

    np.savez(
        OUTPUT_FILE,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        roi1=roi1, roi2=roi2,
        rms=rms,
        used_left_images=np.array(filtered_left_images),
        used_right_images=np.array(filtered_right_images)
    )
    print(f"\n[SUCCESS] 結果を {OUTPUT_FILE} に保存しました。")

    # rectification preview
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2)

    sample_indices = [0, len(filtered_left_images)//2, len(filtered_left_images)-1]
    print("[INFO] Rectification preview を表示中。緑線に左右の対応点が揃っていれば良好です。")

    for idx in sample_indices:
        img_l = cv2.imread(filtered_left_images[idx])
        img_r = cv2.imread(filtered_right_images[idx])

        rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

        combined = np.hstack((rect_l, rect_r))
        h, w = combined.shape[:2]

        for y in range(0, h, 40):
            cv2.line(combined, (0, y), (w, y), (0, 255, 0), 1)

        cv2.imshow(f"Rectified Pair Preview {idx}", combined)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()