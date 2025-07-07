import cv2
import numpy as np
from tqdm import tqdm
from ximea import xiapi
import yaml
from scipy.spatial.transform import Rotation as R

# ---------- ユーティリティ ----------
def load_camera_params(yaml_path: str):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return (np.array(data['camera_matrix']),
            np.array(data['dist_coeffs']))

def detect_markers_bgr(frame_bgr, hsv_ranges):
    """HSB 色閾値でマーカー中心を抽出（複数色対応）"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    centers = []
    for (lower, upper) in hsv_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 30:   # ノイズ除去
                continue
            M = cv2.moments(c)
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            centers.append((cx, cy))
    return centers          # [(x,y),...]

def solve_orientation(centers_img, centers_obj, K, dist):
    """対応点 ≥3 で姿勢推定 (PnP) → 回転行列"""
    ok, rvec, tvec = cv2.solvePnP(centers_obj, centers_img,
                                  K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    return cv2.Rodrigues(rvec)[0]     # 3×3 回転行列

# ---------- キャプチャ/解析 ----------
def main():
    # ---- 撮影条件 ----
    N_FRAMES   = 3000           # 6 秒間
    OUTPUT_AVI = 'capture.avi'
    CAM_YAML   = 'camera_params.yaml'

    # 3D 座標 (球半径 r=1; 単位球で定義し実スケール不要)
    obj_pts = np.array([
        [ 0.0,  0.0,  3.8],     # 北極
        [ 0.0,  3.8,  0.0],     # 赤道
        [ 3.8,  0.0,  0.0],     # 赤道
    ], dtype=np.float32)

    # ---- マーカー色範囲 (要調整・複数可) ----
    red  = (np.array([0,160,120]), np.array([10,255,255]))
    blue = (np.array([100,100,100]), np.array([130,255,255]))
    hsv_ranges = [red, blue]

    # ---- カメラ初期化 ----
    cam = xiapi.Camera()
    cam.open_device()
    cam.set_param('width',  648)
    cam.set_param('height', 488)
    cam.set_param('exposure', 1000)       # µs
    cam.set_param('framerate', 500.0)
    cam.set_param('gain', 0.0)
    cam.start_acquisition()

    img = xiapi.Image()

    # ---- 動画保存 (MJPG) ----
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vw = cv2.VideoWriter(OUTPUT_AVI, fourcc, 500,
                         (648,488), True)

    K, dist = load_camera_params(CAM_YAML)
    rotations = []
    times     = []

    print('Capturing…')
    for i in tqdm(range(N_FRAMES)):
        cam.get_image(img)
        raw = img.get_image_data_numpy()
        bgr = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)

        # マーカー検出
        centers = detect_markers_bgr(bgr, hsv_ranges)

        # PnP (≥3 点で実行)
        if len(centers) >= len(obj_pts):
            img_pts = np.array(centers[:len(obj_pts)], dtype=np.float32)
            Rmat = solve_orientation(img_pts, obj_pts, K, dist)
            if Rmat is not None:
                rotations.append(R.from_matrix(Rmat))
                times.append(img.timestamp / 1e6)   # µs → 秒

        vw.write(bgr)

    cam.stop_acquisition()
    cam.close_device()
    vw.release()
    print('Capture done.')

    # ---- 角速度計算 ----
    if len(rotations) < 2:
        print('十分な姿勢データが取れていません。')
        return

    ang_speeds = []
    axes_world = []
    for r1, r2, t1, t2 in zip(rotations[:-1], rotations[1:],
                              times[:-1], times[1:]):
        dR = r2 * r1.inv()              # 差分回転
        angle = dR.magnitude()
        axis  = dR.as_rotvec() / angle  # 正規化回転軸
        ang_speeds.append(angle / (t2 - t1))  # rad/s
        axes_world.append(axis)

    rpm = np.mean(ang_speeds) * 60 / (2*np.pi)
    print(f'平均角速度: {rpm:.1f} rpm  '
          f'({np.mean(ang_speeds):.2f} rad/s)')

    # 主回転軸（平均）
    axis_mean = np.mean(axes_world, axis=0)
    axis_mean /= np.linalg.norm(axis_mean)
    print(f'推定回転軸 (カメラ座標系): {axis_mean}')

if __name__ == '__main__':
    main()
