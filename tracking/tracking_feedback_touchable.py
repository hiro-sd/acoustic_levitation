from tracking_fast_feedback import *


# Touchable / drop-resistant settings
Z_FAILSAFE_DROP_MM = 3.0
Z_FAILSAFE_LOSS_FRAMES = 2
Z_EMERGENCY_LIFT_MM = 2.0
Z_EMERGENCY_HOLD_SEC = 0.18
Z_EMERGENCY_KP_SCALE = 1.4
Z_EMERGENCY_KD_SCALE = 1.6
Z_INTEGRAL_FREEZE_ON_DISTURB = True
XY_DISTURB_R_MM = 6.0
XY_DISTURB_V_MM_S = 100.0
XY_INTEGRAL_FREEZE_ON_DISTURB = True
XY_RETURN_RATE = 0.04
USE_TARGET_STEP_LIMIT = False


# Additional drop-resistance constants (restored from copied variant)
XY_DISTURB_R_MM = 6.0
XY_DISTURB_V_MM_S = 100.0
Z_DISTURB_MM = 4.0
Z_DISTURB_V_MM_S = 90.0
Z_SAFETY_DROP_MM = 2.5
Z_SAFETY_BOOST_MM = 1.5
Z_EMERGENCY_DROP_MM = 6.0
Z_EMERGENCY_VZ_MM_S = 140.0
Z_EMERGENCY_LOST_FRAMES = 2
Z_EMERGENCY_HOLD_SEC = 0.35
Z_EMERGENCY_LIFT_STEP_MM = 8.0
FREEZE_INTEGRAL_ON_DISTURB = True
INTEGRAL_DECAY_DISTURBED = 0.95
DISTURB_KP_SCALE_XY = 0.9
DISTURB_KD_SCALE_XY = 1.4
DISTURB_KP_SCALE_Z = 1.1
DISTURB_KD_SCALE_Z = 1.3
TARGET_STEP_XY_NORMAL = 1.0
TARGET_STEP_Z_NORMAL = 0.7
TARGET_STEP_XY_DISTURBED = 2.0
TARGET_STEP_Z_DISTURBED = 4.0


def _clamp01(value: float) -> float:
	return float(max(0.0, min(1.0, value)))


def _limit_step(new_val: float, old_val: float, max_step: float) -> float:
	return old_val + float(np.clip(new_val - old_val, -max_step, max_step))


def main():
	global shared_target_pos, program_running
	global latest_frame_xy, latest_frame_z

	try:
		A_affine, affine_uv_type = load_affine_matrix(AFFINE_XY_JSON)
		use_affine = True
		print(f"[INFO] Loaded affine matrix from {AFFINE_XY_JSON}")
		print(f"[INFO] affine input_uv_type = {affine_uv_type}")
	except Exception as e:
		print(f"[WARN] Affine matrix load failed: {e}")
		A_affine = None
		affine_uv_type = "unknown"
		use_affine = False

	try:
		mtx_cam_xy, dist_cam_xy = load_intrinsic(INTRINSIC_XY_NPZ, "xy")
		mtx_cam_z, dist_cam_z = load_intrinsic(INTRINSIC_Z_NPZ, "z")
		use_undistort = True
		print("[INFO] Vision pipeline: undistort each camera frame, then detect.")
	except Exception as e:
		print(f"[WARN] Intrinsic parameters load failed: {e}")
		mtx_cam_xy = None
		dist_cam_xy = None
		mtx_cam_z = None
		dist_cam_z = None
		use_undistort = False

	try:
		z_a, z_b = load_z_model(AFFINE_Z_JSON)
		use_z_model = True
	except Exception as e:
		print(f"[ERROR] z model load failed: {e}")
		print("[ERROR] 先に coordinate_transformation_z.py と fit_affine_from_csv_z.py を実行して、affine_v_to_z.json を生成してください。")
		return

	try:
		cam_xy, img_xy = init_ximea_camera(CAMERA_XY_SN, "xy")
		cam_z, img_z = init_ximea_camera(CAMERA_Z_SN, "z")
	except Exception as e:
		print(f"[ERROR] Camera Init Failed: {e}")
		return

	print("[INFO] Opening AUTD Controller...")
	try:
		with Controller.open(autd_arrangement, TwinCAT()) as autd:
			autd.send(Silencer())
			autd.send(Static(intensity=int(0xFF * 0.9)))

			base_center = autd.center()
			home_x = float(base_center[0])
			home_y = float(base_center[1])
			home_z = float(DEFAULT_Z)

			set_shared_target_pos(home_x, home_y, home_z)

			t = threading.Thread(target=autd_control_loop, args=(autd,))
			t.start()

			print("[INFO] Vision loop started.")
			print("=================================================")
			print("  READY TO LEVITATE.")
			print("  Press [ENTER] to START dynamic tracking.")
			print("  Press [ENTER] again to PAUSE (Return to center).")
			print(f"  Press [{LOG_TRIGGER_KEY.upper()}] to START {LOG_DURATION_SEC:.0f}s logging in current mode.")
			print("  Press [ESC] to EXIT and STOP ultrasound.")
			print("=================================================")

			window_tile = "Tracking"
			cv2.namedWindow(window_tile, cv2.WINDOW_NORMAL)

			fps_start_time = time.time()
			loop_fps_count = 0
			new_xy_fps_count = 0
			new_z_fps_count = 0
			loop_display_fps = 0.0
			new_xy_display_fps = 0.0
			new_z_display_fps = 0.0
			frame_count = 0

			cam_xy.get_image(img_xy)
			frame0_raw_xy = img_xy.get_image_data_numpy()
			cam_z.get_image(img_z)
			frame0_raw_z = img_z.get_image_data_numpy()

			if use_undistort:
				map1_xy, map2_xy = build_undistort_maps(frame0_raw_xy.shape, mtx_cam_xy, dist_cam_xy)
				map1_z, map2_z = build_undistort_maps(frame0_raw_z.shape, mtx_cam_z, dist_cam_z)
				frame0 = undistort_frame(frame0_raw_xy, map1_xy, map2_xy)
				frame0_z = undistort_frame(frame0_raw_z, map1_z, map2_z)
			else:
				map1_xy = map2_xy = None
				map1_z = map2_z = None
				frame0 = frame0_raw_xy
				frame0_z = frame0_raw_z

			frame0_z = rotate_frame_if_needed(frame0_z, ROTATE_Z_FRAME, ROTATE_Z_CODE)

			H_xy, W_xy = frame0.shape[:2]
			H_z, W_z = frame0_z.shape[:2]

			with frame_xy_lock:
				latest_frame_xy = frame0.copy()
			with frame_z_lock:
				latest_frame_z = frame0_z.copy()

			t_cam_xy = threading.Thread(
				target=camera_capture_loop,
				args=(cam_xy, img_xy, "xy", use_undistort, map1_xy, map2_xy, False, None),
				daemon=True,
			)
			t_cam_z = threading.Thread(
				target=camera_capture_loop,
				args=(cam_z, img_z, "z", use_undistort, map1_z, map2_z, ROTATE_Z_FRAME, ROTATE_Z_CODE),
				daemon=True,
			)
			t_cam_xy.start()
			t_cam_z.start()

			roi_cx_xy, roi_cy_xy = W_xy // 2, H_xy // 2
			roi_size_xy = ROI_INIT_SIZE
			roi_cx_z, roi_cy_z = W_z // 2, H_z // 2
			roi_size_z = ROI_INIT_SIZE

			tracking_active = False
			prev_enter_state = False
			prev_log_trigger_state = False

			prev_particle_x = None
			prev_particle_y = None
			prev_vx = 0.0
			prev_vy = 0.0
			prev_xy_meas_time = None

			prev_particle_z = None
			prev_particle_z_filt = None
			prev_vz = 0.0
			prev_z_meas_time = None
			prev_z_error_int = 0.0

			last_valid_target_x = home_x
			last_valid_target_y = home_y
			last_valid_target_z = home_z

			z_lost_frames = 0
			emergency_lift_until = 0.0

			last_processed_xy_time = -1.0
			last_processed_z_time = -1.0

			log_file_initialized = False
			log_session_active = False
			log_session_end_time = 0.0
			log_session_mode = ""
			log_file = None
			log_writer = None

			while True:
				if keyboard.is_pressed("esc"):
					break

				current_enter_state = keyboard.is_pressed("enter")
				if current_enter_state and not prev_enter_state:
					tracking_active = not tracking_active

					if tracking_active:
						print("[INFO] >>> TRACKING ACTIVATED <<<")
						prev_particle_x = None
						prev_particle_y = None
						prev_particle_z = None
						prev_particle_z_filt = None
						prev_vx = 0.0
						prev_vy = 0.0
						prev_vz = 0.0
						prev_xy_meas_time = None
						prev_z_meas_time = None
						prev_xy_error_int_x = 0.0
						prev_xy_error_int_y = 0.0
						prev_z_error_int = 0.0
						last_valid_target_x = home_x
						last_valid_target_y = home_y
						last_valid_target_z = home_z
						z_lost_frames = 0
						emergency_lift_until = 0.0
					else:
						print("[INFO] >>> TRACKING PAUSED (Center Fixed) <<<")
						set_shared_target_pos(home_x, home_y, home_z)
				prev_enter_state = current_enter_state

				current_log_trigger_state = keyboard.is_pressed(LOG_TRIGGER_KEY)
				if LOG_ENABLED and current_log_trigger_state and not prev_log_trigger_state:
					if log_session_active:
						remain = max(0.0, log_session_end_time - time.time())
						print(f"[LOG] Recording in progress ({log_session_mode}), remaining {remain:.1f}s")
					else:
						if not log_file_initialized:
							with open(LOG_CSV_PATH, "w", newline="", encoding="utf-8") as f_init:
								w_init = csv.writer(f_init)
								w_init.writerow([
									"timestamp", "mode",
									"u_xy_px", "v_xy_px", "v_z_px",
									"x_mm", "y_mm", "z_mm",
									"center_x_mm", "center_y_mm", "center_z_mm",
									"autd_target_x_mm", "autd_target_y_mm", "autd_target_z_mm",
									"z_lost_frames", "emergency_lift_active"
								])
							log_file_initialized = True
							print(f"[LOG] Log file initialized: {LOG_CSV_PATH}")

						log_file = open(LOG_CSV_PATH, "a", newline="", encoding="utf-8")
						log_writer = csv.writer(log_file)
						log_session_active = True
						log_session_mode = "PID" if tracking_active else "FIXED"
						now_log = time.time()
						log_session_end_time = now_log + LOG_DURATION_SEC
						print(f"[LOG] START {log_session_mode} logging for {LOG_DURATION_SEC:.0f}s")
				prev_log_trigger_state = current_log_trigger_state

				if log_session_active and time.time() >= log_session_end_time:
					log_session_active = False
					if log_file is not None:
						log_file.close()
						log_file = None
						log_writer = None
					print(f"[LOG] DONE {log_session_mode} logging ({LOG_DURATION_SEC:.0f}s)")

				with frame_xy_lock:
					frame_xy = None if latest_frame_xy is None else latest_frame_xy.copy()
					frame_xy_time = latest_frame_xy_time

				with frame_z_lock:
					frame_z = None if latest_frame_z is None else latest_frame_z.copy()
					frame_z_time = latest_frame_z_time

				if frame_xy is None or frame_z is None:
					time.sleep(0.001)
					continue

				is_new_xy_frame = frame_xy_time > last_processed_xy_time
				is_new_z_frame = frame_z_time > last_processed_z_time
				if is_new_xy_frame:
					last_processed_xy_time = frame_xy_time
					new_xy_fps_count += 1
				if is_new_z_frame:
					last_processed_z_time = frame_z_time
					new_z_fps_count += 1

				frame_count += 1
				loop_fps_count += 1
				do_display = (frame_count % DISPLAY_EVERY_N_FRAMES == 0)

				if do_display:
					frame_xy_bgr = cv2.cvtColor(frame_xy, cv2.COLOR_GRAY2BGR) if frame_xy.ndim == 2 else frame_xy.copy()
					frame_z_bgr = cv2.cvtColor(frame_z, cv2.COLOR_GRAY2BGR) if frame_z.ndim == 2 else frame_z.copy()
				else:
					frame_xy_bgr = None
					frame_z_bgr = None

				x1_xy, y1_xy, x2_xy, y2_xy = clamp_roi(roi_cx_xy, roi_cy_xy, roi_size_xy, W_xy, H_xy)
				track_xy, _ = track_ball_cv(frame_xy, (x1_xy, y1_xy, x2_xy, y2_xy))
				x1_z, y1_z, x2_z, y2_z = clamp_roi(roi_cx_z, roi_cy_z, roi_size_z, W_z, H_z)
				track_z, _ = track_ball_cv(frame_z, (x1_z, y1_z, x2_z, y2_z))

				detected_xy = False
				detected_z = False
				u_xy = v_xy = r_xy = 0.0
				u_z = v_z = r_z = 0.0

				if track_xy is not None:
					u_xy, v_xy, r_xy = track_xy
					roi_cx_xy, roi_cy_xy = int(u_xy), int(v_xy)
					desired_xy = int(2 * (2.5 * r_xy + ROI_MARGIN))
					roi_size_xy = int(0.7 * roi_size_xy + 0.3 * desired_xy)
					roi_size_xy = int(max(ROI_MIN_SIZE, min(ROI_MAX_SIZE, roi_size_xy)))
					detected_xy = True
					if do_display:
						color = (0, 255, 0) if tracking_active else (200, 200, 200)
						cv2.circle(frame_xy_bgr, (int(u_xy), int(v_xy)), int(max(2, r_xy)), color, 2)
						cv2.rectangle(frame_xy_bgr, (x1_xy, y1_xy), (x2_xy, y2_xy), (255, 255, 0), 2)
				else:
					roi_size_xy = int(min(ROI_MAX_SIZE, roi_size_xy * ROI_EXPAND_ON_LOST))
					if do_display:
						cv2.rectangle(frame_xy_bgr, (x1_xy, y1_xy), (x2_xy, y2_xy), (0, 255, 255), 2)

				if track_z is not None:
					u_z, v_z, r_z = track_z
					roi_cx_z, roi_cy_z = int(u_z), int(v_z)
					desired_z = int(2 * (2.5 * r_z + ROI_MARGIN))
					roi_size_z = int(0.7 * roi_size_z + 0.3 * desired_z)
					roi_size_z = int(max(ROI_MIN_SIZE, min(ROI_MAX_SIZE, roi_size_z)))
					detected_z = True
					z_lost_frames = 0
					if do_display:
						color = (0, 255, 0) if tracking_active else (200, 200, 200)
						cv2.circle(frame_z_bgr, (int(u_z), int(v_z)), int(max(2, r_z)), color, 2)
						cv2.rectangle(frame_z_bgr, (x1_z, y1_z), (x2_z, y2_z), (255, 255, 0), 2)
				else:
					roi_size_z = int(min(ROI_MAX_SIZE, roi_size_z * ROI_EXPAND_ON_LOST))
					z_lost_frames += 1
					if do_display:
						cv2.rectangle(frame_z_bgr, (x1_z, y1_z), (x2_z, y2_z), (0, 255, 255), 2)

				method = "XY+Z" if (detected_xy and detected_z) else "PARTIAL"

				loop_target_x = float(home_x)
				loop_target_y = float(home_y)
				loop_target_z = float(last_valid_target_z)

				emergency_lift_active = time.time() < emergency_lift_until

				if tracking_active:
					target_x = last_valid_target_x
					target_y = last_valid_target_y
					target_z = last_valid_target_z

					if use_affine and detected_xy and is_new_xy_frame:
						uv_homo = np.array([[u_xy, v_xy, 1.0]], dtype=np.float32).T
						xy_affine = (A_affine @ uv_homo).flatten()
						current_x = home_x + xy_affine[0]
						current_y = home_y + xy_affine[1]

						if prev_particle_x is not None and prev_xy_meas_time is not None:
							dt_xy = max(1e-3, frame_xy_time - prev_xy_meas_time)
							raw_vx = (current_x - prev_particle_x) / dt_xy
							raw_vy = (current_y - prev_particle_y) / dt_xy
							vx = 0.5 * prev_vx + 0.5 * raw_vx
							vy = 0.5 * prev_vy + 0.5 * raw_vy
						else:
							dt_xy = 0.0
							vx, vy = 0.0, 0.0

						prev_particle_x = current_x
						prev_particle_y = current_y
						prev_xy_meas_time = frame_xy_time
						prev_vx = vx
						prev_vy = vy

						xy_disturbed = (
							np.hypot(current_x - home_x, current_y - home_y) > XY_DISTURB_R_MM
							or np.hypot(vx, vy) > XY_DISTURB_V_MM_S
						)

						x_pred = current_x + vx * DT_PRED_XY
						y_pred = current_y + vy * DT_PRED_XY

						if xy_disturbed:
							target_x = current_x
							target_y = current_y
						else:
							target_x = home_x
							target_y = home_y

						x_error = target_x - x_pred
						y_error = target_y - y_pred

						if xy_disturbed and XY_INTEGRAL_FREEZE_ON_DISTURB:
							prev_xy_error_int_x *= 0.95
							prev_xy_error_int_y *= 0.95
						else:
							prev_xy_error_int_x = float(
								np.clip(prev_xy_error_int_x + x_error * dt_xy, -XY_INTEGRAL_CLAMP, XY_INTEGRAL_CLAMP)
							)
							prev_xy_error_int_y = float(
								np.clip(prev_xy_error_int_y + y_error * dt_xy, -XY_INTEGRAL_CLAMP, XY_INTEGRAL_CLAMP)
							)

						kp_xy_eff = K_P_XY
						ki_xy_eff = K_I_XY
						kd_xy_eff = K_D_XY
						if xy_disturbed:
							kp_xy_eff *= 1.0
							kd_xy_eff *= 1.3
							ki_xy_eff = 0.0 if XY_INTEGRAL_FREEZE_ON_DISTURB else K_I_XY

						target_x = target_x + kp_xy_eff * x_error + ki_xy_eff * prev_xy_error_int_x - kd_xy_eff * vx
						target_y = target_y + kp_xy_eff * y_error + ki_xy_eff * prev_xy_error_int_y - kd_xy_eff * vy

					if use_z_model and (detected_z or z_lost_frames >= 0):
						if detected_z and is_new_z_frame:
							current_z = z_a * v_z + z_b
							current_z_meas_time = frame_z_time

							if prev_particle_z_filt is None:
								z_filt = current_z
							else:
								z_filt = Z_LP_ALPHA * prev_particle_z_filt + (1.0 - Z_LP_ALPHA) * current_z

							if prev_particle_z is not None and prev_z_meas_time is not None:
								dt_z = max(1e-3, frame_z_time - prev_z_meas_time)
								raw_vz = (z_filt - prev_particle_z) / dt_z
								vz = 0.5 * prev_vz + 0.5 * raw_vz
							else:
								vz = 0.0

							prev_particle_z = z_filt
							prev_particle_z_filt = z_filt
							prev_vz = vz

							z_disturbed = (
								abs(current_z - home_z) > Z_FAILSAFE_DROP_MM
								or abs(vz) > 120.0
								or z_lost_frames >= Z_FAILSAFE_LOSS_FRAMES
							)

							if z_disturbed:
								emergency_lift_until = max(emergency_lift_until, time.time() + Z_EMERGENCY_HOLD_SEC)

							setpoint_z = home_z
							z_error = setpoint_z - current_z

							if prev_z_meas_time is None:
								dt_int = 0.0
							else:
								dt_int = max(1e-3, current_z_meas_time - prev_z_meas_time)

							if z_disturbed and Z_INTEGRAL_FREEZE_ON_DISTURB:
								prev_z_error_int *= 0.95
							else:
								prev_z_error_int = float(
									np.clip(prev_z_error_int + z_error * dt_int, -Z_INTEGRAL_CLAMP, Z_INTEGRAL_CLAMP)
								)
							prev_z_meas_time = current_z_meas_time

							kp_z_eff = K_P_Z * (Z_EMERGENCY_KP_SCALE if z_disturbed else 1.0)
							ki_z_eff = 0.0 if (z_disturbed and Z_INTEGRAL_FREEZE_ON_DISTURB) else K_I_Z
							kd_z_eff = K_D_Z * (Z_EMERGENCY_KD_SCALE if z_disturbed else 1.0)

							z_pred = z_filt + vz * DT_PRED_Z - 0.5 * GRAVITY_MM_S2 * (DT_PRED_Z ** 2)
							target_z = (
								setpoint_z
								+ kp_z_eff * (setpoint_z - z_pred)
								+ ki_z_eff * prev_z_error_int
								- kd_z_eff * vz
							)

							if z_disturbed or emergency_lift_active:
								lift_target = max(home_z + Z_EMERGENCY_LIFT_MM, z_filt + Z_EMERGENCY_LIFT_MM)
								target_z = max(target_z, lift_target)

							target_z = float(np.clip(target_z, Z_MIN, Z_MAX))
							last_valid_target_z = target_z
						else:
							if z_lost_frames >= Z_FAILSAFE_LOSS_FRAMES:
								emergency_lift_until = max(emergency_lift_until, time.time() + Z_EMERGENCY_HOLD_SEC)
								target_z = max(last_valid_target_z, home_z + Z_EMERGENCY_LIFT_MM)
							else:
								target_z = last_valid_target_z

					if emergency_lift_active:
						target_z = max(target_z, home_z + Z_EMERGENCY_LIFT_MM)

					target_x = _limit_step(target_x, last_valid_target_x, 2.5 if emergency_lift_active else 1.0)
					target_y = _limit_step(target_y, last_valid_target_y, 2.5 if emergency_lift_active else 1.0)
					target_z = _limit_step(target_z, last_valid_target_z, 5.0 if emergency_lift_active else 0.7)

					last_valid_target_x = float(target_x)
					last_valid_target_y = float(target_y)
					last_valid_target_z = float(target_z)

					set_shared_target_pos(target_x, target_y, target_z)
					loop_target_x = float(target_x)
					loop_target_y = float(target_y)
					loop_target_z = float(target_z)

					if do_display:
						cv2.putText(frame_xy_bgr, f"TGT XY: {target_x:.1f}, {target_y:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
						cv2.putText(frame_z_bgr, f"TGT Z: {target_z:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

				if not tracking_active:
					set_shared_target_pos(home_x, home_y, home_z)
					last_valid_target_x = home_x
					last_valid_target_y = home_y
					last_valid_target_z = home_z

				if log_session_active and log_writer is not None and (is_new_xy_frame or is_new_z_frame):
					x_log = np.nan
					y_log = np.nan
					z_log = np.nan

					if use_affine and detected_xy:
						uv_homo_log = np.array([[u_xy, v_xy, 1.0]], dtype=np.float32).T
						xy_log = (A_affine @ uv_homo_log).flatten()
						x_log = float(home_x + xy_log[0])
						y_log = float(home_y + xy_log[1])

					if use_z_model and detected_z:
						z_log = float(z_a * v_z + z_b)

					log_writer.writerow([
						f"{time.time():.4f}",
						log_session_mode,
						f"{u_xy:.2f}", f"{v_xy:.2f}", f"{v_z:.2f}",
						f"{x_log:.3f}" if np.isfinite(x_log) else "",
						f"{y_log:.3f}" if np.isfinite(y_log) else "",
						f"{z_log:.3f}" if np.isfinite(z_log) else "",
						f"{home_x:.3f}", f"{home_y:.3f}", f"{home_z:.3f}",
						f"{loop_target_x:.3f}", f"{loop_target_y:.3f}", f"{loop_target_z:.3f}",
						f"{z_lost_frames}", f"{int(emergency_lift_active)}",
					])

				now = time.time()
				if now - fps_start_time >= 1.0:
					elapsed = now - fps_start_time
					loop_display_fps = loop_fps_count / elapsed
					new_xy_display_fps = new_xy_fps_count / elapsed
					new_z_display_fps = new_z_fps_count / elapsed
					fps_start_time = now
					loop_fps_count = 0
					new_xy_fps_count = 0
					new_z_fps_count = 0

				status_text = "ACTIVE" if tracking_active else "WAIT (Press ENTER)"
				if emergency_lift_active:
					status_text = "EMERGENCY LIFT"
				status_color = (0, 255, 0) if tracking_active else (0, 165, 255)
				if emergency_lift_active:
					status_color = (0, 0, 255)

				if do_display:
					cv2.putText(
						frame_xy_bgr,
						f"Loop FPS: {loop_display_fps:.1f} | NewXY FPS: {new_xy_display_fps:.1f} | AUTD FPS: {autd_display_fps:.1f} | {method}",
						(10, 30),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.6,
						(255, 255, 255),
						2,
					)
					cv2.putText(frame_xy_bgr, f"STATUS: {status_text}", (10, H_xy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
					cv2.putText(
						frame_z_bgr,
						f"Loop FPS: {loop_display_fps:.1f} | NewZ FPS: {new_z_display_fps:.1f} | v_z={v_z:.1f}",
						(10, 30),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.6,
						(255, 255, 255),
						2,
					)
					cv2.putText(frame_z_bgr, f"STATUS: {status_text}", (10, H_z - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

					display_h = max(frame_xy_bgr.shape[0], frame_z_bgr.shape[0])
					display_w = max(frame_xy_bgr.shape[1], frame_z_bgr.shape[1])
					frame_xy_disp = cv2.resize(frame_xy_bgr, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
					frame_z_disp = cv2.resize(frame_z_bgr, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

					tiled = np.hstack([frame_xy_disp, frame_z_disp])
					split_x = frame_xy_disp.shape[1]
					cv2.line(tiled, (split_x, 0), (split_x, tiled.shape[0] - 1), (255, 255, 255), 1)
					cv2.putText(tiled, "XY", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
					cv2.putText(tiled, "Z", (split_x + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
					cv2.imshow(window_tile, tiled)
					if cv2.waitKey(1) & 0xFF == 27:
						break

			program_running = False
			if log_file is not None:
				log_file.close()
				print(f"[LOG] Saved (closed session): {LOG_CSV_PATH}")

			print("[INFO] Waiting for AUTD thread to close...")
			t.join(timeout=2.0)

	except Exception as e:
		print(f"[ERROR] Runtime Error: {e}")
	finally:
		program_running = False

		if "t" in locals() and t.is_alive():
			t.join(timeout=1.0)
		if "t_cam_xy" in locals() and t_cam_xy.is_alive():
			t_cam_xy.join(timeout=1.0)
		if "t_cam_z" in locals() and t_cam_z.is_alive():
			t_cam_z.join(timeout=1.0)
		if "log_file" in locals() and log_file is not None:
			try:
				log_file.close()
				print(f"[LOG] Saved (on exit): {LOG_CSV_PATH}")
			except Exception:
				pass

		for cam_obj in (locals().get("cam_xy"), locals().get("cam_z")):
			if cam_obj is None:
				continue
			try:
				cam_obj.stop_acquisition()
			except Exception:
				pass
			try:
				cam_obj.close_device()
			except Exception:
				pass

		cv2.destroyAllWindows()
		print("[INFO] Finished.")


if __name__ == "__main__":
	main()

def load_intrinsic(npz_path: str, role: str):
	data = np.load(npz_path, allow_pickle=True)
	if "camera_matrix" in data and "dist_coeffs" in data:
		mtx = data["camera_matrix"].astype(np.float32)
		dist = data["dist_coeffs"].astype(np.float32)
	elif "mtx" in data and "dist" in data:
		mtx = data["mtx"].astype(np.float32)
		dist = data["dist"].astype(np.float32)
	else:
		keys = ", ".join(data.files)
		raise KeyError(
			f"Unsupported intrinsic keys for {role}: {keys}. Expected (camera_matrix, dist_coeffs) or (mtx, dist)."
		)
	print(f"[INFO] Loaded intrinsic parameters ({role}) from {npz_path}")
	return mtx, dist


def rotate_frame_if_needed(frame: np.ndarray, do_rotate: bool, rotate_code: int) -> np.ndarray:
	return cv2.rotate(frame, rotate_code) if do_rotate else frame


def load_z_model(json_path: str):
	if not os.path.exists(json_path):
		raise FileNotFoundError(f"Z model JSON not found: {json_path}")
	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)
	a = float(data["a"])
	b = float(data["b"])
	print(f"[INFO] Loaded z model from {json_path}: z = {a:.6f} * v + {b:.6f}")
	return a, b


def clamp_roi(cx, cy, size, w, h):
	size = int(max(ROI_MIN_SIZE, min(ROI_MAX_SIZE, size)))
	half = size // 2
	x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
	x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))
	if (x2 - x1) < size:
		if x1 == 0:
			x2 = min(w, x1 + size)
		elif x2 == w:
			x1 = max(0, x2 - size)
	if (y2 - y1) < size:
		if y1 == 0:
			y2 = min(h, y1 + size)
		elif y2 == h:
			y1 = max(0, y2 - size)
	return x1, y1, x2, y2


def limit_step(new_val, old_val, max_step):
	return old_val + float(np.clip(new_val - old_val, -max_step, max_step))


def build_undistort_maps(frame_shape, mtx: np.ndarray, dist: np.ndarray):
	h, w = frame_shape[:2]
	map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (w, h), cv2.CV_16SC2)
	return map1, map2


def undistort_frame(frame: np.ndarray, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
	return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


def track_ball_cv(frame_rgb: np.ndarray, roi_rect):
	x1, y1, x2, y2 = roi_rect
	roi = frame_rgb[y1:y2, x1:x2]
	gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi
	if BLUR_KSIZE > 1:
		gray = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
	if USE_OTSU:
		_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	else:
		_, bw = cv2.threshold(gray, FIXED_THRESH, 255, cv2.THRESH_BINARY)
	kernel = np.ones((3, 3), np.uint8)
	bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
	bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
	contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return None, bw

	roi_cx, roi_cy = (x2 - x1) / 2.0, (y2 - y1) / 2.0
	best = None
	best_score = -1e18
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area < MIN_AREA_PX or area > MAX_AREA_PX:
			continue
		perimeter = cv2.arcLength(cnt, True)
		if perimeter <= 0:
			continue
		circularity = 4 * np.pi * (area / (perimeter * perimeter))
		if circularity < 0.6:
			continue
		M = cv2.moments(cnt)
		if M["m00"] <= 1e-6:
			continue
		cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
		dist2 = (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2
		score = area - 0.8 * dist2
		if score > best_score:
			best_score = score
			best = cnt
	if best is None:
		return None, bw
	(xc, yc), r = cv2.minEnclosingCircle(best)
	return (float(x1 + xc), float(y1 + yc), float(r)), bw


def camera_capture_loop(cam, img, role, use_undistort, map1, map2, do_rotate=False, rotate_code=None):
	global program_running, latest_frame_xy, latest_frame_z, latest_frame_xy_time, latest_frame_z_time
	while program_running:
		try:
			cam.get_image(img)
			frame = img.get_image_data_numpy()
			if use_undistort:
				frame = undistort_frame(frame, map1, map2)
			if do_rotate:
				frame = rotate_frame_if_needed(frame, True, rotate_code)
			now_t = time.time()
			if role == "xy":
				with frame_xy_lock:
					latest_frame_xy = frame
					latest_frame_xy_time = now_t
			else:
				with frame_z_lock:
					latest_frame_z = frame
					latest_frame_z_time = now_t
		except Exception as e:
			print(f"[CAM {role} Thread Error] {e}")
			time.sleep(0.01)


def autd_control_loop(autd):
	global shared_target_pos, shared_target_seq, program_running, autd_display_fps
	print("[THREAD] AUTD Control Thread Started.")
	fps_start_time = time.perf_counter()
	fps_frame_count = 0
	last_seq = -1
	send_time_ema_ms = 0.0
	build_time_ema_ms = 0.0
	while program_running:
		tgt = None
		seq = last_seq
		with pos_lock:
			if shared_target_pos is not None:
				tgt = shared_target_pos
				seq = shared_target_seq
		if tgt is None or seq == last_seq:
			time.sleep(AUTD_LOOP_SLEEP_SEC)
			continue
		tx, ty, tz = tgt
		try:
			t0 = time.perf_counter()
			center_vec = np.array([tx, ty, tz], dtype=np.float32)
			foci = center_vec[None, :] + CIRCLE_OFFSETS
			t1 = time.perf_counter()
			stm = FociSTM(foci=[foci[i] for i in range(foci.shape[0])], config=100 * Hz).into_nearest()
			t2 = time.perf_counter()
			autd.send(stm)
			t3 = time.perf_counter()
			build_ms = (t2 - t0) * 1000.0
			send_ms = (t3 - t2) * 1000.0
			build_time_ema_ms = 0.9 * build_time_ema_ms + 0.1 * build_ms
			send_time_ema_ms = 0.9 * send_time_ema_ms + 0.1 * send_ms
			last_seq = seq
			fps_frame_count += 1
		except Exception as e:
			print(f"[AUTD Thread Error] {e}")
			time.sleep(0.01)
		now = time.perf_counter()
		if now - fps_start_time >= 1.0:
			autd_display_fps = fps_frame_count / (now - fps_start_time)
			print(f"[AUTD] fps={autd_display_fps:.1f}, build_ema={build_time_ema_ms:.2f} ms, send_ema={send_time_ema_ms:.2f} ms")
			fps_start_time = now
			fps_frame_count = 0
	print("[THREAD] AUTD Control Thread Stopped.")


def load_affine_matrix(json_path):
	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)
	A = np.array(data["A_2x3"], dtype=np.float32)
	uv_type = data.get("input_uv_type", "unknown")
	return A, uv_type


def set_shared_target_pos(x: float, y: float, z: float):
	global shared_target_pos, shared_target_seq
	with pos_lock:
		shared_target_pos = (float(x), float(y), float(z))
		shared_target_seq += 1


def main():
	global shared_target_pos, program_running, latest_frame_xy, latest_frame_z

	try:
		A_affine, affine_uv_type = load_affine_matrix(AFFINE_XY_JSON)
		use_affine = True
		print(f"[INFO] Loaded affine matrix from {AFFINE_XY_JSON}")
		print(f"[INFO] affine input_uv_type = {affine_uv_type}")
	except Exception as e:
		print(f"[WARN] Affine matrix load failed: {e}")
		A_affine = None
		use_affine = False

	try:
		mtx_cam_xy, dist_cam_xy = load_intrinsic(INTRINSIC_XY_NPZ, "xy")
		mtx_cam_z, dist_cam_z = load_intrinsic(INTRINSIC_Z_NPZ, "z")
		use_undistort = True
		print("[INFO] Vision pipeline: undistort each camera frame, then detect.")
	except Exception as e:
		print(f"[WARN] Intrinsic parameters load failed: {e}")
		mtx_cam_xy = dist_cam_xy = mtx_cam_z = dist_cam_z = None
		use_undistort = False

	try:
		z_a, z_b = load_z_model(AFFINE_Z_JSON)
		use_z_model = True
	except Exception as e:
		print(f"[ERROR] z model load failed: {e}")
		return

	try:
		cam_xy, img_xy = init_ximea_camera(CAMERA_XY_SN, "xy")
		cam_z, img_z = init_ximea_camera(CAMERA_Z_SN, "z")
	except Exception as e:
		print(f"[ERROR] Camera Init Failed: {e}")
		return

	print("[INFO] Opening AUTD Controller...")
	try:
		with Controller.open(autd_arrangement, TwinCAT()) as autd:
			autd.send(Silencer())
			autd.send(Static(intensity=int(0xFF * 0.9)))

			base_center = autd.center()
			home_x = float(base_center[0])
			home_y = float(base_center[1])
			home_z = float(DEFAULT_Z)
			set_shared_target_pos(home_x, home_y, home_z)

			t = threading.Thread(target=autd_control_loop, args=(autd,))
			t.start()

			print("[INFO] Vision loop started.")
			print("=================================================")
			print("  READY TO LEVITATE.")
			print("  Press [ENTER] to START dynamic tracking.")
			print("  Press [ENTER] again to PAUSE (Return to center).")
			print(f"  Press [{LOG_TRIGGER_KEY.upper()}] to START {LOG_DURATION_SEC:.0f}s logging in current mode.")
			print("  Press [ESC] to EXIT and STOP ultrasound.")
			print("=================================================")

			window_tile = "Tracking"
			cv2.namedWindow(window_tile, cv2.WINDOW_NORMAL)

			fps_start_time = time.time()
			loop_fps_count = 0
			new_xy_fps_count = 0
			new_z_fps_count = 0
			loop_display_fps = 0.0
			new_xy_display_fps = 0.0
			new_z_display_fps = 0.0
			frame_count = 0

			cam_xy.get_image(img_xy)
			frame0_raw_xy = img_xy.get_image_data_numpy()
			cam_z.get_image(img_z)
			frame0_raw_z = img_z.get_image_data_numpy()

			if use_undistort:
				map1_xy, map2_xy = build_undistort_maps(frame0_raw_xy.shape, mtx_cam_xy, dist_cam_xy)
				map1_z, map2_z = build_undistort_maps(frame0_raw_z.shape, mtx_cam_z, dist_cam_z)
				frame0 = undistort_frame(frame0_raw_xy, map1_xy, map2_xy)
				frame0_z = undistort_frame(frame0_raw_z, map1_z, map2_z)
			else:
				map1_xy = map2_xy = map1_z = map2_z = None
				frame0 = frame0_raw_xy
				frame0_z = frame0_raw_z

			frame0_z = rotate_frame_if_needed(frame0_z, ROTATE_Z_FRAME, ROTATE_Z_CODE)
			H_xy, W_xy = frame0.shape[:2]
			H_z, W_z = frame0_z.shape[:2]

			with frame_xy_lock:
				latest_frame_xy = frame0.copy()
			with frame_z_lock:
				latest_frame_z = frame0_z.copy()

			t_cam_xy = threading.Thread(target=camera_capture_loop, args=(cam_xy, img_xy, "xy", use_undistort, map1_xy, map2_xy, False, None), daemon=True)
			t_cam_z = threading.Thread(target=camera_capture_loop, args=(cam_z, img_z, "z", use_undistort, map1_z, map2_z, ROTATE_Z_FRAME, ROTATE_Z_CODE), daemon=True)
			t_cam_xy.start()
			t_cam_z.start()

			roi_cx_xy, roi_cy_xy = W_xy // 2, H_xy // 2
			roi_size_xy = ROI_INIT_SIZE
			roi_cx_z, roi_cy_z = W_z // 2, H_z // 2
			roi_size_z = ROI_INIT_SIZE
			method = "CV"

			tracking_active = False
			prev_enter_state = False
			prev_log_trigger_state = False

			prev_particle_x = None
			prev_particle_y = None
			prev_vx = 0.0
			prev_vy = 0.0
			prev_xy_meas_time = None

			prev_particle_z = None
			prev_particle_z_filt = None
			prev_vz = 0.0
			prev_z_meas_time = None
			prev_z_error_int = 0.0

			last_valid_target_x = home_x
			last_valid_target_y = home_y
			last_valid_target_z = home_z
			last_target_x = home_x
			last_target_y = home_y
			last_target_z = home_z

			lost_xy_frames = 0
			lost_z_frames = 0
			emergency_lift_until = 0.0
			emergency_reason = ""

			last_processed_xy_time = -1.0
			last_processed_z_time = -1.0

			log_file_initialized = False
			log_session_active = False
			log_session_end_time = 0.0
			log_session_mode = ""
			log_file = None
			log_writer = None

			while True:
				if keyboard.is_pressed("esc"):
					break

				current_enter_state = keyboard.is_pressed("enter")
				if current_enter_state and not prev_enter_state:
					tracking_active = not tracking_active
					if tracking_active:
						print("[INFO] >>> TRACKING ACTIVATED <<<")
						prev_particle_x = prev_particle_y = None
						prev_particle_z = prev_particle_z_filt = None
						prev_vx = prev_vy = prev_vz = 0.0
						prev_xy_meas_time = prev_z_meas_time = None
						prev_xy_error_int_x = 0.0
						prev_xy_error_int_y = 0.0
						prev_z_error_int = 0.0
						lost_xy_frames = 0
						lost_z_frames = 0
						emergency_lift_until = 0.0
						emergency_reason = ""
						last_valid_target_x = home_x
						last_valid_target_y = home_y
						last_valid_target_z = home_z
						last_target_x = home_x
						last_target_y = home_y
						last_target_z = home_z
					else:
						print("[INFO] >>> TRACKING PAUSED (Center Fixed) <<<")
						set_shared_target_pos(home_x, home_y, home_z)
				prev_enter_state = current_enter_state

				current_log_trigger_state = keyboard.is_pressed(LOG_TRIGGER_KEY)
				if LOG_ENABLED and current_log_trigger_state and not prev_log_trigger_state:
					if log_session_active:
						remain = max(0.0, log_session_end_time - time.time())
						print(f"[LOG] Recording in progress ({log_session_mode}), remaining {remain:.1f}s")
					else:
						if not log_file_initialized:
							with open(LOG_CSV_PATH, "w", newline="", encoding="utf-8") as f_init:
								w_init = csv.writer(f_init)
								w_init.writerow([
									"timestamp", "mode",
									"u_xy_px", "v_xy_px", "v_z_px",
									"x_mm", "y_mm", "z_mm",
									"home_x_mm", "home_y_mm", "home_z_mm",
									"autd_target_x_mm", "autd_target_y_mm", "autd_target_z_mm",
									"emergency_active", "emergency_reason",
									"lost_xy_frames", "lost_z_frames",
								])
							log_file_initialized = True
							print(f"[LOG] Log file initialized: {LOG_CSV_PATH}")
						log_file = open(LOG_CSV_PATH, "a", newline="", encoding="utf-8")
						log_writer = csv.writer(log_file)
						log_session_active = True
						log_session_mode = "PID" if tracking_active else "FIXED"
						now_log = time.time()
						log_session_end_time = now_log + LOG_DURATION_SEC
						print(f"[LOG] START {log_session_mode} logging for {LOG_DURATION_SEC:.0f}s")
				prev_log_trigger_state = current_log_trigger_state

				if log_session_active and time.time() >= log_session_end_time:
					log_session_active = False
					if log_file is not None:
						log_file.close()
						log_file = None
						log_writer = None
					print(f"[LOG] DONE {log_session_mode} logging ({LOG_DURATION_SEC:.0f}s)")

				with frame_xy_lock:
					frame_xy = None if latest_frame_xy is None else latest_frame_xy.copy()
					frame_xy_time = latest_frame_xy_time
				with frame_z_lock:
					frame_z = None if latest_frame_z is None else latest_frame_z.copy()
					frame_z_time = latest_frame_z_time

				if frame_xy is None or frame_z is None:
					time.sleep(0.001)
					continue

				is_new_xy_frame = frame_xy_time > last_processed_xy_time
				is_new_z_frame = frame_z_time > last_processed_z_time
				if is_new_xy_frame:
					last_processed_xy_time = frame_xy_time
					new_xy_fps_count += 1
				if is_new_z_frame:
					last_processed_z_time = frame_z_time
					new_z_fps_count += 1

				frame_count += 1
				loop_fps_count += 1
				do_display = (frame_count % DISPLAY_EVERY_N_FRAMES == 0)

				if do_display:
					frame_xy_bgr = cv2.cvtColor(frame_xy, cv2.COLOR_GRAY2BGR) if frame_xy.ndim == 2 else frame_xy.copy()
					frame_z_bgr = cv2.cvtColor(frame_z, cv2.COLOR_GRAY2BGR) if frame_z.ndim == 2 else frame_z.copy()
				else:
					frame_xy_bgr = None
					frame_z_bgr = None

				x1_xy, y1_xy, x2_xy, y2_xy = clamp_roi(roi_cx_xy, roi_cy_xy, roi_size_xy, W_xy, H_xy)
				track_xy, _ = track_ball_cv(frame_xy, (x1_xy, y1_xy, x2_xy, y2_xy))
				x1_z, y1_z, x2_z, y2_z = clamp_roi(roi_cx_z, roi_cy_z, roi_size_z, W_z, H_z)
				track_z, _ = track_ball_cv(frame_z, (x1_z, y1_z, x2_z, y2_z))

				detected_xy = False
				detected_z = False
				u_xy = v_xy = r_xy = 0.0
				u_z = v_z = r_z = 0.0

				if track_xy is not None:
					u_xy, v_xy, r_xy = track_xy
					roi_cx_xy, roi_cy_xy = int(u_xy), int(v_xy)
					desired_xy = int(2 * (2.5 * r_xy + ROI_MARGIN))
					roi_size_xy = int(0.7 * roi_size_xy + 0.3 * desired_xy)
					roi_size_xy = int(max(ROI_MIN_SIZE, min(ROI_MAX_SIZE, roi_size_xy)))
					detected_xy = True
					if do_display:
						color = (0, 255, 0) if tracking_active else (200, 200, 200)
						cv2.circle(frame_xy_bgr, (int(u_xy), int(v_xy)), int(max(2, r_xy)), color, 2)
						cv2.rectangle(frame_xy_bgr, (x1_xy, y1_xy), (x2_xy, y2_xy), (255, 255, 0), 2)
				else:
					roi_size_xy = int(min(ROI_MAX_SIZE, roi_size_xy * ROI_EXPAND_ON_LOST))
					if do_display:
						cv2.rectangle(frame_xy_bgr, (x1_xy, y1_xy), (x2_xy, y2_xy), (0, 255, 255), 2)

				if track_z is not None:
					u_z, v_z, r_z = track_z
					roi_cx_z, roi_cy_z = int(u_z), int(v_z)
					desired_z = int(2 * (2.5 * r_z + ROI_MARGIN))
					roi_size_z = int(0.7 * roi_size_z + 0.3 * desired_z)
					roi_size_z = int(max(ROI_MIN_SIZE, min(ROI_MAX_SIZE, roi_size_z)))
					detected_z = True
					if do_display:
						color = (0, 255, 0) if tracking_active else (200, 200, 200)
						cv2.circle(frame_z_bgr, (int(u_z), int(v_z)), int(max(2, r_z)), color, 2)
						cv2.rectangle(frame_z_bgr, (x1_z, y1_z), (x2_z, y2_z), (255, 255, 0), 2)
				else:
					roi_size_z = int(min(ROI_MAX_SIZE, roi_size_z * ROI_EXPAND_ON_LOST))
					if do_display:
						cv2.rectangle(frame_z_bgr, (x1_z, y1_z), (x2_z, y2_z), (0, 255, 255), 2)

				method = "XY+Z" if (detected_xy and detected_z) else "PARTIAL"

				loop_target_x = home_x
				loop_target_y = home_y
				loop_target_z = last_valid_target_z

				emergency_now = False
				emergency_reason_now = ""

				if tracking_active:
					target_x = last_valid_target_x
					target_y = last_valid_target_y
					target_z = last_valid_target_z

					xy_disturbed = False
					z_disturbed = False

					if use_affine and detected_xy and is_new_xy_frame:
						uv_homo = np.array([[u_xy, v_xy, 1.0]], dtype=np.float32).T
						xy_affine = (A_affine @ uv_homo).flatten()
						current_x = home_x + xy_affine[0]
						current_y = home_y + xy_affine[1]

						if prev_particle_x is not None and prev_xy_meas_time is not None:
							dt_xy = max(1e-3, frame_xy_time - prev_xy_meas_time)
							raw_vx = (current_x - prev_particle_x) / dt_xy
							raw_vy = (current_y - prev_particle_y) / dt_xy
							vx = 0.5 * prev_vx + 0.5 * raw_vx
							vy = 0.5 * prev_vy + 0.5 * raw_vy
						else:
							vx = vy = 0.0

						prev_particle_x = current_x
						prev_particle_y = current_y
						prev_xy_meas_time = frame_xy_time
						prev_vx = vx
						prev_vy = vy

						x_pred = current_x + vx * DT_PRED_XY
						y_pred = current_y + vy * DT_PRED_XY

						setpoint_x = home_x
						setpoint_y = home_y
						x_error = setpoint_x - x_pred
						y_error = setpoint_y - y_pred

						xy_disturbed = (
							np.hypot(current_x - home_x, current_y - home_y) > XY_DISTURB_R_MM
							or np.hypot(vx, vy) > XY_DISTURB_V_MM_S
						)

						if xy_disturbed and FREEZE_INTEGRAL_ON_DISTURB:
							prev_xy_error_int_x *= INTEGRAL_DECAY_DISTURBED
							prev_xy_error_int_y *= INTEGRAL_DECAY_DISTURBED
						else:
							prev_xy_error_int_x = float(np.clip(prev_xy_error_int_x + x_error * max(1e-3, frame_xy_time - prev_xy_meas_time), -XY_INTEGRAL_CLAMP, XY_INTEGRAL_CLAMP))
							prev_xy_error_int_y = float(np.clip(prev_xy_error_int_y + y_error * max(1e-3, frame_xy_time - prev_xy_meas_time), -XY_INTEGRAL_CLAMP, XY_INTEGRAL_CLAMP))

						kp_xy_eff = K_P_XY * (DISTURB_KP_SCALE_XY if xy_disturbed else 1.0)
						ki_xy_eff = 0.0 if (xy_disturbed and FREEZE_INTEGRAL_ON_DISTURB) else K_I_XY
						kd_xy_eff = K_D_XY * (DISTURB_KD_SCALE_XY if xy_disturbed else 1.0)

						target_x = setpoint_x + kp_xy_eff * x_error + ki_xy_eff * prev_xy_error_int_x - kd_xy_eff * vx
						target_y = setpoint_y + kp_xy_eff * y_error + ki_xy_eff * prev_xy_error_int_y - kd_xy_eff * vy

					if use_z_model and detected_z and is_new_z_frame:
						current_z = z_a * v_z + z_b
						if prev_particle_z_filt is None:
							z_filt = current_z
						else:
							z_filt = Z_LP_ALPHA * prev_particle_z_filt + (1.0 - Z_LP_ALPHA) * current_z

						if prev_particle_z is not None and prev_z_meas_time is not None:
							dt_z = max(1e-3, frame_z_time - prev_z_meas_time)
							raw_vz = (z_filt - prev_particle_z) / dt_z
							vz = 0.5 * prev_vz + 0.5 * raw_vz
						else:
							vz = 0.0

						prev_particle_z = z_filt
						prev_particle_z_filt = z_filt
						prev_vz = vz

						z_disturbed = (
							abs(current_z - home_z) > Z_DISTURB_MM
							or abs(vz) > Z_DISTURB_V_MM_S
						)

						if prev_z_meas_time is None:
							dt_int = 0.0
						else:
							dt_int = max(1e-3, frame_z_time - prev_z_meas_time)

						if z_disturbed and FREEZE_INTEGRAL_ON_DISTURB:
							prev_z_error_int *= INTEGRAL_DECAY_DISTURBED
						else:
							z_error = home_z - current_z
							prev_z_error_int = float(np.clip(prev_z_error_int + z_error * dt_int, -Z_INTEGRAL_CLAMP, Z_INTEGRAL_CLAMP))

						prev_z_meas_time = frame_z_time

						kp_z_eff = K_P_Z * (DISTURB_KP_SCALE_Z if z_disturbed else 1.0)
						ki_z_eff = 0.0 if (z_disturbed and FREEZE_INTEGRAL_ON_DISTURB) else K_I_Z
						kd_z_eff = K_D_Z * (DISTURB_KD_SCALE_Z if z_disturbed else 1.0)

						z_pred = z_filt + vz * DT_PRED_Z - 0.5 * GRAVITY_MM_S2 * (DT_PRED_Z ** 2)
						target_z = home_z + kp_z_eff * (home_z - z_pred) + ki_z_eff * prev_z_error_int - kd_z_eff * vz

						if current_z < home_z - Z_SAFETY_DROP_MM:
							target_z = max(target_z, last_valid_target_z + Z_SAFETY_BOOST_MM)

						if z_disturbed:
							lost_z_frames += 1
						else:
							lost_z_frames = 0

						if current_z < home_z - Z_EMERGENCY_DROP_MM or vz < -Z_EMERGENCY_VZ_MM_S or lost_z_frames >= Z_EMERGENCY_LOST_FRAMES:
							emergency_now = True
							emergency_reason_now = "z_drop/lost"

						target_z = float(np.clip(target_z, Z_MIN, Z_MAX))

					if not detected_xy:
						lost_xy_frames += 1
					else:
						lost_xy_frames = 0

					if not detected_z:
						lost_z_frames += 1
					else:
						if not emergency_now:
							lost_z_frames = 0

					if lost_z_frames >= Z_EMERGENCY_LOST_FRAMES:
						emergency_now = True
						emergency_reason_now = emergency_reason_now or "z_lost"

					if emergency_now:
						emergency_lift_until = time.time() + Z_EMERGENCY_HOLD_SEC
						emergency_reason = emergency_reason_now
					elif time.time() < emergency_lift_until:
						emergency_now = True

					if emergency_now:
						prev_xy_error_int_x *= INTEGRAL_DECAY_DISTURBED
						prev_xy_error_int_y *= INTEGRAL_DECAY_DISTURBED
						prev_z_error_int *= INTEGRAL_DECAY_DISTURBED
						target_x = home_x
						target_y = home_y
						target_z = max(target_z, last_valid_target_z + Z_EMERGENCY_LIFT_STEP_MM, home_z + Z_EMERGENCY_LIFT_STEP_MM)
						target_z = float(np.clip(target_z, Z_MIN, Z_MAX))

					if USE_TARGET_STEP_LIMIT:
						if emergency_now:
							max_step_xy = TARGET_STEP_XY_DISTURBED
							max_step_z = Z_EMERGENCY_LIFT_STEP_MM
						elif xy_disturbed or z_disturbed:
							max_step_xy = TARGET_STEP_XY_DISTURBED
							max_step_z = TARGET_STEP_Z_DISTURBED
						else:
							max_step_xy = TARGET_STEP_XY_NORMAL
							max_step_z = TARGET_STEP_Z_NORMAL
						target_x = limit_step(target_x, last_valid_target_x, max_step_xy)
						target_y = limit_step(target_y, last_valid_target_y, max_step_xy)
						target_z = limit_step(target_z, last_valid_target_z, max_step_z)

					last_valid_target_x = float(target_x)
					last_valid_target_y = float(target_y)
					last_valid_target_z = float(target_z)
					last_target_x = float(target_x)
					last_target_y = float(target_y)
					last_target_z = float(target_z)

					set_shared_target_pos(target_x, target_y, target_z)
					loop_target_x = float(target_x)
					loop_target_y = float(target_y)
					loop_target_z = float(target_z)

					if do_display:
						cv2.putText(frame_xy_bgr, f"TGT XY: {target_x:.1f}, {target_y:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
						cv2.putText(frame_z_bgr, f"TGT Z: {target_z:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

				if not tracking_active:
					set_shared_target_pos(home_x, home_y, home_z)

				if log_session_active and log_writer is not None and (is_new_xy_frame or is_new_z_frame):
					x_log = np.nan
					y_log = np.nan
					z_log = np.nan
					if use_affine and detected_xy:
						uv_homo_log = np.array([[u_xy, v_xy, 1.0]], dtype=np.float32).T
						xy_log = (A_affine @ uv_homo_log).flatten()
						x_log = float(home_x + xy_log[0])
						y_log = float(home_y + xy_log[1])
					if use_z_model and detected_z:
						z_log = float(z_a * v_z + z_b)
					log_writer.writerow([
						f"{time.time():.4f}",
						log_session_mode,
						f"{u_xy:.2f}", f"{v_xy:.2f}", f"{v_z:.2f}",
						f"{x_log:.3f}" if np.isfinite(x_log) else "",
						f"{y_log:.3f}" if np.isfinite(y_log) else "",
						f"{z_log:.3f}" if np.isfinite(z_log) else "",
						f"{home_x:.3f}", f"{home_y:.3f}", f"{home_z:.3f}",
						f"{loop_target_x:.3f}", f"{loop_target_y:.3f}", f"{loop_target_z:.3f}",
						int(emergency_now), emergency_reason,
						lost_xy_frames, lost_z_frames,
					])

				now = time.time()
				if now - fps_start_time >= 1.0:
					elapsed = now - fps_start_time
					loop_display_fps = loop_fps_count / elapsed
					new_xy_display_fps = new_xy_fps_count / elapsed
					new_z_display_fps = new_z_fps_count / elapsed
					fps_start_time = now
					loop_fps_count = new_xy_fps_count = new_z_fps_count = 0

				status_text = "ACTIVE" if tracking_active else "WAIT (Press ENTER)"
				if emergency_now:
					status_text = f"EMERGENCY {emergency_reason}"
				status_color = (0, 255, 0) if tracking_active and not emergency_now else (0, 0, 255) if emergency_now else (0, 165, 255)

				if do_display:
					cv2.putText(frame_xy_bgr, f"Loop FPS: {loop_display_fps:.1f} | NewXY FPS: {new_xy_display_fps:.1f} | AUTD FPS: {autd_display_fps:.1f} | {method}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
					cv2.putText(frame_xy_bgr, f"STATUS: {status_text}", (10, H_xy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
					cv2.putText(frame_z_bgr, f"Loop FPS: {loop_display_fps:.1f} | NewZ FPS: {new_z_display_fps:.1f} | v_z={prev_vz:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
					cv2.putText(frame_z_bgr, f"STATUS: {status_text}", (10, H_z - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
					display_h = max(frame_xy_bgr.shape[0], frame_z_bgr.shape[0])
					display_w = max(frame_xy_bgr.shape[1], frame_z_bgr.shape[1])
					frame_xy_disp = cv2.resize(frame_xy_bgr, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
					frame_z_disp = cv2.resize(frame_z_bgr, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
					tiled = np.hstack([frame_xy_disp, frame_z_disp])
					split_x = frame_xy_disp.shape[1]
					cv2.line(tiled, (split_x, 0), (split_x, tiled.shape[0] - 1), (255, 255, 255), 1)
					cv2.putText(tiled, "XY", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
					cv2.putText(tiled, "Z", (split_x + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
					cv2.imshow(window_tile, tiled)
					if cv2.waitKey(1) & 0xFF == 27:
						break

			program_running = False
			if log_file is not None:
				log_file.close()
				print(f"[LOG] Saved (closed session): {LOG_CSV_PATH}")
			print("[INFO] Waiting for AUTD thread to close...")
			t.join(timeout=2.0)
	except Exception as e:
		print(f"[ERROR] Runtime Error: {e}")
	finally:
		program_running = False
		if "t" in locals() and t.is_alive():
			t.join(timeout=1.0)
		if "t_cam_xy" in locals() and t_cam_xy.is_alive():
			t_cam_xy.join(timeout=1.0)
		if "t_cam_z" in locals() and t_cam_z.is_alive():
			t_cam_z.join(timeout=1.0)
		if "log_file" in locals() and log_file is not None:
			try:
				log_file.close()
				print(f"[LOG] Saved (on exit): {LOG_CSV_PATH}")
			except Exception:
				pass
		for cam_obj in (locals().get("cam_xy"), locals().get("cam_z")):
			if cam_obj is None:
				continue
			try:
				cam_obj.stop_acquisition()
			except Exception:
				pass
			try:
				cam_obj.close_device()
			except Exception:
				pass
		cv2.destroyAllWindows()
		print("[INFO] Finished.")


if __name__ == "__main__":
	main()
