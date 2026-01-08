import os
import time
import sys
import numpy as np
import cv2
import keyboard
from ultralytics import YOLO

# フィードバック制御で球を追従し、中心座標を更新するスクリプト
# TODO: 制御をちゃんと行えるようにする

# ===== XIMEA =====
sys.path.append(r"C:\Users\Hiroto Yoshida\Desktop\XIMEA\API\Python\v3")
from ximea import xiapi

# ===== AUTD =====
from pyautd3 import AUTD3, Controller, FociSTM, Hz, Silencer, Static
from pyautd3_link_soem import SOEM, SOEMOption, Status

# ===================== ユーザー設定 =====================
MODEL_PATH = "tracking/train/weights/best.pt"  # 独自モデル
TARGET_CLASS_NAME = "levitatedball"            # 独自モデル内のクラス名
IMG_SIZE = 640
CONF_THRES = 0.5

EXPOSURE_US = 5000

# 円軌道STM
POINT_NUM = 8
RADIUS_MM = 23.0
STM_INNER_HZ = 100

# 基準中心
X_REF_MM, Y_REF_MM, Z_REF_MM = 0.0, 0.0, 400.0

# ★ フィードバック開始遅延（起動してから何秒後に閉ループを有効化するか）
FEEDBACK_DELAY_S = 20.0

# ===== キャリブレーション行列 =====
A_UV_TO_XY = np.array([
    [0.4828603396891242, -0.0049182869868897175, -159.34300668267267],
    [-0.012420190414686443, -0.47713028907367255, 115.62492376472873],
], dtype=np.float32)

# ===== 制御（PD推奨）=====
CONTROL_HZ = 20.0
KP_X, KP_Y = 0.6, 0.6
KD_X, KD_Y = 0.05, 0.05
KI_X, KI_Y = 0.0, 0.0

MAX_DELTA_MM = 8.0

MAX_ABS_X_MM = 25.0
MAX_ABS_Y_MM = 25.0

LOST_TIMEOUT_S = 0.25
RETURN_RATE_MM_PER_S = 20.0

EMA_ALPHA = 0.35

os.environ.setdefault("OMP_NUM_THREADS", "4")

# ===================== AUTD 配置 =====================
autd_arrangement = [
    AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[0.0, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[0.0, -2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, -2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
    AUTD3(pos=[AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),
]

def err_handler(slave: int, status: Status) -> None:
    print(f"slave [{slave}]: {status}")
    if status == Status.Lost():
        os._exit(-1)

def init_ximea_camera():
    cam = xiapi.Camera()
    cam.open_device()
    cam.set_imgdataformat('XI_RGB24')
    try:
        cam.enable_auto_wb()
    except AttributeError:
        try:
            cam.set_param('auto_wb', 1)
        except Exception:
            pass
    cam.set_exposure(EXPOSURE_US)
    img = xiapi.Image()
    cam.start_acquisition()
    return cam, img

def find_target_class_id(model: YOLO, target_name: str):
    names = model.model.names
    for cid, name in names.items():
        if str(name).lower() == target_name.lower():
            return int(cid)
    return None

def best_detection_center(results, target_cls_id: int | None, conf_thres: float):
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = None
    best_score = -1.0
    for b in boxes:
        score = float(b.conf[0].item())
        if score < conf_thres:
            continue
        cls_id = int(b.cls[0].item())
        if target_cls_id is not None and cls_id != target_cls_id:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        if score > best_score:
            best_score = score
            best = (u, v, score, (x1, y1, x2, y2))
    return best

def uv_to_xy_mm(u: float, v: float, A: np.ndarray):
    uv1 = np.array([u, v, 1.0], dtype=np.float32)
    xy = A @ uv1
    return float(xy[0]), float(xy[1])

class PID:
    def __init__(self, kp, ki, kd, out_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit
        self.i = 0.0
        self.prev_e = 0.0
        self.prev_t = None

    def reset(self):
        self.i = 0.0
        self.prev_e = 0.0
        self.prev_t = None

    def update(self, e: float, t_now: float):
        if self.prev_t is None:
            self.prev_t = t_now
            self.prev_e = e
            return 0.0
        dt = t_now - self.prev_t
        if dt <= 1e-6:
            return 0.0
        self.i += e * dt
        de = (e - self.prev_e) / dt
        u = self.kp * e + self.ki * self.i + self.kd * de
        if self.out_limit is not None:
            u = max(min(u, self.out_limit), -self.out_limit)
        self.prev_t = t_now
        self.prev_e = e
        return u

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def main():
    print("[INFO] Loading YOLO...")
    model = YOLO(MODEL_PATH)
    target_cls_id = find_target_class_id(model, TARGET_CLASS_NAME)
    print(f"[INFO] target class: {TARGET_CLASS_NAME} -> id={target_cls_id}")

    print("[INFO] Init XIMEA...")
    cam, img = init_ximea_camera()

    thetas = [np.pi/8 + 2.0*np.pi*i/POINT_NUM for i in range(POINT_NUM)]
    unit_circle = np.array([[np.cos(th), np.sin(th), 0.0] for th in thetas], dtype=np.float32)

    pid_x = PID(KP_X, KI_X, KD_X, out_limit=MAX_DELTA_MM)
    pid_y = PID(KP_Y, KI_Y, KD_Y, out_limit=MAX_DELTA_MM)

    ema_x = None
    ema_y = None
    last_detect_t = None

    x_cmd, y_cmd, z_cmd = X_REF_MM, Y_REF_MM, Z_REF_MM

    next_control_t = time.monotonic()
    control_dt = 1.0 / CONTROL_HZ

    # ★ 起動時刻（monotonic）
    start_mono = time.monotonic()
    feedback_started = False

    window_name = "Feedback delayed 20s | ESC to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with Controller.open(
        autd_arrangement,
        SOEM(err_handler=err_handler, option=SOEMOption()),
    ) as autd:
        autd.send(Silencer())
        m = Static(intensity=int(0xFF))

        prev_vis_t = time.time()
        fps = 0.0

        try:
            while True:
                if keyboard.is_pressed("esc"):
                    break

                cam.get_image(img)
                frame = img.get_image_data_numpy()
                if frame.ndim == 2:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    frame_bgr = frame

                results = model.predict(
                    source=frame_bgr,
                    imgsz=IMG_SIZE,
                    conf=CONF_THRES,
                    device="cpu",
                    verbose=False,
                )

                annotated = frame_bgr.copy()

                det = best_detection_center(results, target_cls_id, CONF_THRES)

                if det is not None:
                    u, v, score, (x1, y1, x2, y2) = det
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(annotated, (int(u), int(v)), 5, (0, 255, 0), -1)

                    x_meas, y_meas = uv_to_xy_mm(u, v, A_UV_TO_XY)

                    if abs(x_meas) <= MAX_ABS_X_MM and abs(y_meas) <= MAX_ABS_Y_MM:
                        last_detect_t = time.monotonic()
                        if ema_x is None:
                            ema_x, ema_y = x_meas, y_meas
                        else:
                            ema_x = EMA_ALPHA * x_meas + (1.0 - EMA_ALPHA) * ema_x
                            ema_y = EMA_ALPHA * y_meas + (1.0 - EMA_ALPHA) * ema_y

                        cv2.putText(
                            annotated,
                            f"xy(mm)={x_meas:+.2f},{y_meas:+.2f}  ema={ema_x:+.2f},{ema_y:+.2f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA
                        )
                    else:
                        cv2.putText(
                            annotated,
                            f"[OUT OF CALIB RANGE] xy(mm)={x_meas:+.1f},{y_meas:+.1f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA
                        )

                # ==== 制御周期 ====
                now_mono = time.monotonic()

                # ★ フィードバック有効判定
                feedback_enabled = (now_mono - start_mono) >= FEEDBACK_DELAY_S
                if feedback_enabled and (not feedback_started):
                    feedback_started = True
                    # 開始時にPIDをリセットしてキックを防ぐ
                    pid_x.reset()
                    pid_y.reset()
                    print(f"[INFO] Feedback started at t={now_mono - start_mono:.1f}s")

                if now_mono >= next_control_t:
                    next_control_t += control_dt

                    # フィードバックがまだなら中心固定
                    if not feedback_enabled:
                        pid_x.reset()
                        pid_y.reset()
                        x_cmd, y_cmd, z_cmd = X_REF_MM, Y_REF_MM, Z_REF_MM

                    else:
                        # ロスト判定
                        lost = True
                        if last_detect_t is not None and (now_mono - last_detect_t) <= LOST_TIMEOUT_S:
                            lost = False

                        if (not lost) and (ema_x is not None):
                            ex = X_REF_MM - ema_x
                            ey = Y_REF_MM - ema_y

                            dx = pid_x.update(ex, now_mono)
                            dy = pid_y.update(ey, now_mono)

                            # もし逆に動くなら dx,dy を反転（必要なら）
                            x_cmd = clip(X_REF_MM + dx, X_REF_MM - MAX_DELTA_MM, X_REF_MM + MAX_DELTA_MM)
                            y_cmd = clip(Y_REF_MM + dy, Y_REF_MM - MAX_DELTA_MM, Y_REF_MM + MAX_DELTA_MM)
                            z_cmd = Z_REF_MM
                        else:
                            pid_x.reset()
                            pid_y.reset()

                            step = RETURN_RATE_MM_PER_S * control_dt
                            if x_cmd > X_REF_MM: x_cmd = max(X_REF_MM, x_cmd - step)
                            if x_cmd < X_REF_MM: x_cmd = min(X_REF_MM, x_cmd + step)
                            if y_cmd > Y_REF_MM: y_cmd = max(Y_REF_MM, y_cmd - step)
                            if y_cmd < Y_REF_MM: y_cmd = min(Y_REF_MM, y_cmd + step)
                            z_cmd = Z_REF_MM

                    # STM生成＆送信（常に送信は行う）
                    center = autd.center() + np.array([x_cmd, y_cmd, z_cmd], dtype=np.float32)
                    stm = FociSTM(
                        foci=(center + RADIUS_MM * unit_circle[i] for i in range(POINT_NUM)),
                        config=int(STM_INNER_HZ) * Hz,
                    ).into_nearest()
                    autd.send((m, stm))

                # FPS表示など
                now_vis = time.time()
                dt_vis = now_vis - prev_vis_t
                prev_vis_t = now_vis
                if dt_vis > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt_vis)

                t_since = now_mono - start_mono
                status = "FEEDBACK ON" if feedback_enabled else f"FEEDBACK OFF ({FEEDBACK_DELAY_S - t_since:.1f}s)"
                cv2.putText(
                    annotated,
                    f"FPS:{fps:.1f} | {status} | center_cmd=({x_cmd:+.2f},{y_cmd:+.2f},{z_cmd:+.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(window_name, annotated)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

        finally:
            try:
                cam.stop_acquisition()
            except Exception:
                pass
            try:
                cam.close_device()
            except Exception:
                pass
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
