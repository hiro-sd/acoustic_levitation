from dataclasses import dataclass
import cv2


@dataclass
class AppConfig:

    # paths
    affine_xy_json: str = "./tracking/calibration/ball1_calibration_data/affine_uv_to_xy.json"
    affine_z_json: str = "./tracking/calibration/ball1_calibration_data/affine_v_to_z.json"

    intrinsic_xy_npz: str = "./tracking/calibration/intrinsic_charuco_cam1.npz"
    intrinsic_z_npz: str = "./tracking/calibration/intrinsic_charuco_cam2.npz"

    ximea_python_path: str = r"C:\Users\Hiroto Yoshida\Desktop\XIMEA\API\Python\v3"

    # camera settings
    camera_xy_sn: str = "43430551"
    camera_z_sn: str = "43435351"
    rotate_z_frame: bool = True
    rotate_z_code: int = cv2.ROTATE_90_CLOCKWISE
    exposure_us: int = 5000

    # Software synchronization / camera watchdog
    # 2台の取得時刻差がこの値以内のフレームだけを1組として処理する。
    camera_sync_tolerance_sec: float = 0.010
    camera_sync_buffer_size: int = 8
    # どちらかのカメラから画像が届かない状態が続いたら安全停止する。
    camera_frame_timeout_sec: float = 0.5

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

    # Runtime radius change settings
    enable_radius_change: bool = False
    radius_min: float = 20.0
    radius_max: float = 28.0
    radius_change_speed_mm_s: float = 1.0 # 矢印キーを押し続けたときの半径変更速度 [mm/s]

    # Base movement by keyboard
    enable_base_move: bool = False
    base_move_speed_mm_s: float = 50.0
    base_z_move_speed_mm_s: float = 50.0

    # Automatic demo trajectory for the circular STM center.
    # demo_toggle_key を押すと、固定AUTD原点を基準にXYは指定範囲の四角形、
    # Zは指定範囲内で上下動する。
    enable_auto_demo: bool = False
    demo_toggle_key: str = "d"
    demo_x_min_mm: float = -40.0
    demo_x_max_mm: float = 30.0
    demo_y_min_mm: float = -30.0
    demo_y_max_mm: float = 30.0
    demo_xy_speed_mm_s: float = 20.0
    demo_z_min_mm: float = 330.0
    demo_z_max_mm: float = 430.0
    demo_z_corner_mm: float = 400.0

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

    # Z intensity boost
    enable_z_intensity_boost: bool = False

    intensity_max_ratio: float = 1.00

    # zがこの値以上下がったら最大ブースト
    z_boost_full_drop_mm: float = 5.0

    # z偏差がこの値以内なら通常強度へ戻す
    z_boost_release_mm: float = 1.5

    # intensity_ratio のローパス
    # 大きいほど変化がゆっくり
    intensity_lpf_alpha: float = 0.90

    # この差以上変わったときだけAUTDへStaticを再送する
    intensity_update_eps: float = 0.005

    # Logging
    log_enabled: bool = True
    log_csv_path: str = "./tracking/stability_log.csv"
    log_duration_sec: float = 30.0
    log_trigger_key: str = "l"

    # Fall recovery / return-to-home settings
    enable_fall_recovery: bool = False

    fall_drop_threshold_mm: float = 6.0
    fall_vz_threshold_mm_s: float = -50.0

    fall_recovery_dt_pred_z: float = 0.02
    fall_recovery_z_offset_mm: float = 0.0

    fall_recovery_intensity_ratio: float = 1.0

    fall_recovery_dt_pred_xy: float = 0
    fall_recovery_target_xy_step_mm: float = 10.0
    fall_recovery_target_z_step_mm: float = 10.0

    fall_recovery_hold_time_s: float = 1.0

    # 捕捉できたと判断する条件
    # fall_captured_speed_xy_mm_s: float = 50.0
    # fall_captured_vz_mm_s: float = 40.0
    # fall_captured_stable_time_s: float = 0.05

    # RETURN_TO_HOME中、一時目標をhomeへ戻す速度
    return_home_speed_xy_mm_s: float = 8.0
    return_home_speed_z_mm_s: float = 5.0

    # RETURN_TO_HOME完了条件
    return_home_done_error_xy_mm: float = 2.0
    return_home_done_error_z_mm: float = 2.0
    return_home_done_speed_xy_mm_s: float = 30.0
    return_home_done_vz_mm_s: float = 30.0
