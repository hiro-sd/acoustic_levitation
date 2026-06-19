# import cv2
# import numpy as np
# import glob
# import os

# # 単眼カメラのキャリブレーションスクリプト

# # 設定項目
# PATTERN_SIZE = (9, 6)       # チェッカーボードの内側の交点数 (cols, rows)
# SQUARE_SIZE_MM = 24.0       # 1マスのサイズ [mm]
# IMAGES_DIR = "./tracking/calibration/cam2_calibration_images" # キャリブレーション画像が入ったフォルダ
# OUTPUT_FILE = "./tracking/calibration/intrinsic_cam2.npz"  # 保存するファイル名

# # チェッカーボードの3D座標を準備
# objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
# objp *= SQUARE_SIZE_MM

# objpoints = []  # 3D点
# imgpoints = []  # 2D点

# images = glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) # 画像の拡張子に合わせて変更

# print(f"[INFO] {len(images)} 枚の画像を処理中...")

# def detect_chessboard_corners(gray, pattern_size):
#     if hasattr(cv2, "findChessboardCornersSB"):
#         ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
#         if ret:
#             return ret, corners
#     flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
#     return cv2.findChessboardCorners(gray, pattern_size, flags)

# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     ret, corners = detect_chessboard_corners(gray, PATTERN_SIZE)

#     if ret:
#         corners2 = cv2.cornerSubPix(
#             gray, corners, (11, 11), (-1, -1),
#             criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#         )
#         objpoints.append(objp)
#         imgpoints.append(corners2)

# print(f"[INFO] 検出成功: {len(objpoints)}/{len(images)} 枚")

# if len(objpoints) < 5:
#     print("[ERROR] キャリブレーションに十分な画像がありません。")
#     exit()

# # カメラキャリブレーション
# print("[INFO] 計算中...")
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
#     objpoints, imgpoints, gray.shape[::-1], None, None
# )

# print("\n=== キャリブレーション結果 ===")
# print(f"再投影誤差 (RMS): {ret:.4f}")
# print("カメラ行列 (mtx):")
# print(mtx)
# print("歪み係数 (dist):")
# print(dist)

# # 結果を保存
# np.savez(OUTPUT_FILE, mtx=mtx, dist=dist)
# print(f"\n[SUCCESS] 結果を {OUTPUT_FILE} に保存しました。")

# # 歪み補正の確認表示
# sample_img = cv2.imread(images[0])
# h, w = sample_img.shape[:2]
# new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
# undistorted = cv2.undistort(sample_img, mtx, dist, None, new_camera_mtx)

# x, y, w_roi, h_roi = roi
# if w_roi > 0 and h_roi > 0:
#     undistorted = undistorted[y:y+h_roi, x:x+w_roi]

# sample_resized = cv2.resize(sample_img, (640, 480))
# undist_resized = cv2.resize(undistorted, (640, 480))
# combined = np.hstack((sample_resized, undist_resized))

# cv2.imshow("Before (Left) vs After (Right) Undistortion", combined)
# print("[INFO] 歪み補正のプレビューを表示しています。何かキーを押すと終了します。")
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np
import glob
import os

# 単眼カメラのキャリブレーションスクリプト (ChArUco)

# 設定
IMAGES_DIR = "./tracking/calibration/stereo_images/cam2"
OUTPUT_FILE = "./tracking/calibration/intrinsic_charuco_cam2.npz"

# 印刷した ChArUco ボードの設定
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH_MM = 38.0
MARKER_LENGTH_MM = 27.0
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# 最低でもこれ以上の角が見えている画像だけ使う
MIN_CHARUCO_CORNERS = 8


def main():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH_MM,
        MARKER_LENGTH_MM,
        dictionary
    )

    detector_params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    print(f"[INFO] {len(images)} 枚の画像を処理中...")

    all_obj_points = []
    all_img_points = []
    image_size = None
    used_images = []

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"[WARN] 読み込み失敗: {fname}")
            continue

        if image_size is None:
            image_size = (img.shape[1], img.shape[0])

        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)

        if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
            print(f"[WARN] ChArUco角が不足: {os.path.basename(fname)}")
            continue

        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)

        if obj_points is None or img_points is None or len(obj_points) < MIN_CHARUCO_CORNERS:
            print(f"[WARN] 対応点不足: {os.path.basename(fname)}")
            continue

        all_obj_points.append(obj_points.astype(np.float32))
        all_img_points.append(img_points.astype(np.float32))
        used_images.append(fname)

    print(f"[INFO] 使用画像数: {len(all_obj_points)}/{len(images)}")

    if len(all_obj_points) < 5:
        print("[ERROR] キャリブレーションに十分な画像がありません。")
        return

    print("[INFO] 計算中...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points,
        all_img_points,
        image_size,
        None,
        None
    )

    print("\n=== ChArUco 単眼キャリブレーション結果 ===")
    print(f"再投影誤差 (RMS): {rms:.4f}")
    print("カメラ行列 (camera_matrix):")
    print(camera_matrix)
    print("歪み係数 (dist_coeffs):")
    print(dist_coeffs)

    np.savez(
        OUTPUT_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms=rms,
        used_images=np.array(used_images)
    )
    print(f"\n[SUCCESS] 結果を {OUTPUT_FILE} に保存しました。")

    # 簡易プレビュー
    sample_img = cv2.imread(used_images[0])
    h, w = sample_img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 0, (w, h)
    )
    undistorted = cv2.undistort(sample_img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]

    sample_resized = cv2.resize(sample_img, (640, 480))
    undist_resized = cv2.resize(undistorted, (640, 480))
    combined = np.hstack((sample_resized, undist_resized))

    cv2.imshow("Before (Left) vs After (Right) Undistortion", combined)
    print("[INFO] 歪み補正プレビュー表示中。何かキーを押すと終了します。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()