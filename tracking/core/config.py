from dataclasses import dataclass
import cv2


@dataclass
class AppConfig:

    # paths
    affine_xy_json: str = "./tracking/ball1_calibration_data/affine_uv_to_xy.json"
    affine_z_json: str = "./tracking/ball1_calibration_data/affine_v_to_z.json"

    intrinsic_xy_npz: str = "./tracking/calibration/intrinsic_charuco_cam1.npz"
    intrinsic_z_npz: str = "./tracking/calibration/intrinsic_charuco_cam2.npz"

    ximea_python_path: str = r"C:\Users\Hiroto Yoshida\Desktop\XIMEA\API\Python\v3"

    # camera settings
    camera_xy_sn: str = "43430551"
    camera_z_sn: str = "43435351"
    rotate_z_frame: bool = True
    rotate_z_code: int = cv2.ROTATE_90_CLOCKWISE
    exposure_us: int = 5000

    # cv settings
    use_otsu: bool = False
    fixed_thresh: int = 160
    blur_ksize: int = 5
    min_area_px: int = 200
    max_area_px: int = 200000

    # roi settings
    roi_init_size: int = 240
    roi_min_size: int = 120
    roi_max_size: int = 640
    roi_margin: int = 40
    roi_expand_on_lost: float = 1.15

    # display settings
    display_every_n_frames: int = 3

    # AUTD / stm settings
    point_num: int = 8
    radius: float = 23.5
    default_z: float = 400.0

    autd_loop_sleep_sec: float = 0.001
    stm_freq_hz: float = 100.0
    static_intensity_ratio: float = 0.9

    # Base movement by keyboard
    enable_base_move: bool = False
    base_move_speed_mm_s: float = 60.0
    base_z_move_speed_mm_s: float = 80.0

    # outputmask settings
    use_output_mask: bool = False
    output_mask_radius_mm: float = 170.0
    output_mask_update_eps_mm: float = 0.5

    # XY prediction PID
    kp_xy: float = 0.3 # [P] 中心に引き戻す強さ (0.0 なら自然な復元力のみ)
    kd_xy: float = 0.05 # [D] 揺れを抑えるブレーキの強さ (速度に対する抵抗)
    ki_xy: float = 0.1 # [I] ゆっくりと中心に引き戻す力 (積分項)
    dt_pred_xy: float = 0.01 # 10 ms 先を予測
    xy_integral_clamp: float = 150.0 # PIDの積分項が暴走しないようにするためのクランプ値 (mm * s)

    # Z prediction PID
    kp_z: float = 0.6 # [P] 高さを維持する強さ (0.0 なら自然な復元力のみ)
    kd_z: float = 0.15 # [D] 高さの揺れを抑えるブレーキの強さ (速度に対する抵抗)
    ki_z: float = 0.1 # [I] ゆっくりと高さを維持する力 (積分項)
    z_lpf_alpha: float = 0.7 # [LPF] Z方向の低周波数フィルタ係数
    dt_pred_z: float = 0.01 # 10 ms 先を予測
    z_integral_clamp: float = 150.0 # PIDの積分項が暴走しないようにするためのクランプ値 (mm * s)

    gravity_mm_s2: float = 9.80665 * 1000.0
    use_gravity_prediction_z: bool = False

    z_min: float = 250.0
    z_max: float = 550.0

    # Logging
    log_enabled: bool = True
    log_csv_path: str = "./tracking/stability_log.csv"
    log_duration_sec: float = 30.0
    log_trigger_key: str = "l"