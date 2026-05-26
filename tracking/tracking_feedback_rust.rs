use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use opencv::core::{self, Point, Point2f, Rect, Scalar, Size, Vector};
use opencv::highgui;
use opencv::imgproc;
use opencv::prelude::*;
use opencv::videoio;
use serde::Deserialize;

const AFFINE_XY_JSON: &str = "./tracking/ball1_calibration_data/affine_uv_to_xy.json";
const AFFINE_Z_JSON: &str = "./tracking/ball1_calibration_data/affine_v_to_z.json";

const MIN_AREA_PX: f64 = 200.0;
const MAX_AREA_PX: f64 = 200_000.0;
const FIXED_THRESH: f64 = 160.0;
const BLUR_KSIZE: i32 = 5;

const ROI_INIT_SIZE: i32 = 240;
const ROI_MIN_SIZE: i32 = 120;
const ROI_MAX_SIZE: i32 = 640;
const ROI_MARGIN: f32 = 40.0;
const ROI_EXPAND_ON_LOST: f32 = 1.15;

const DISPLAY_EVERY_N_FRAMES: u32 = 3;
const LOG_ENABLED: bool = true;
const LOG_CSV_PATH: &str = "./tracking/stability_log.csv";
const LOG_DURATION_SEC: f64 = 30.0;

const POINT_NUM: usize = 8;
const RADIUS: f32 = 23.5;
const DEFAULT_Z: f32 = 400.0;
const AUTD_LOOP_SLEEP_SEC: f64 = 0.001;

const K_P_XY: f32 = 0.3;
const K_D_XY: f32 = 0.05;
const K_I_XY: f32 = 0.1;
const DT_PRED_XY: f32 = 0.01;
const XY_INTEGRAL_CLAMP: f32 = 150.0;

const K_P_Z: f32 = 0.6;
const K_D_Z: f32 = 0.15;
const K_I_Z: f32 = 0.1;
const Z_LP_ALPHA: f32 = 0.7;
const DT_PRED_Z: f32 = 0.01;
const Z_INTEGRAL_CLAMP: f32 = 150.0;
const GRAVITY_MM_S2: f32 = 9.80665 * 1000.0;
const Z_MIN: f32 = 330.0;
const Z_MAX: f32 = 470.0;

const CAMERA_XY_INDEX: i32 = 0;
const CAMERA_Z_INDEX: i32 = 1;

#[derive(Debug, Deserialize)]
struct AffineFile {
	#[serde(rename = "A_2x3")]
	a_2x3: [[f32; 3]; 2],
	input_uv_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ZModelFile {
	a: f32,
	b: f32,
}

#[derive(Clone)]
struct FramePacket {
	frame: Mat,
	t_sec: f64,
}

#[derive(Clone, Copy)]
struct RoiState {
	cx: i32,
	cy: i32,
	size: i32,
}

#[derive(Clone, Copy)]
struct TargetState {
	pos: [f32; 3],
	seq: u64,
}

trait AutdController: Send {
	fn center(&self) -> [f32; 3];
	fn send_static(&mut self) -> Result<()>;
	fn send_foci_stm(&mut self, foci: &[[f32; 3]]) -> Result<()>;
}

struct DummyAutdController {
	center_mm: [f32; 3],
}

impl DummyAutdController {
	fn new() -> Self {
		Self {
			center_mm: [192.0, 144.0, DEFAULT_Z],
		}
	}
}

impl AutdController for DummyAutdController {
	fn center(&self) -> [f32; 3] {
		self.center_mm
	}

	fn send_static(&mut self) -> Result<()> {
		println!("[AUTD] static emission configured (dummy)");
		Ok(())
	}

	fn send_foci_stm(&mut self, _foci: &[[f32; 3]]) -> Result<()> {
		Ok(())
	}
}

fn now_sec() -> f64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or(Duration::from_secs(0))
		.as_secs_f64()
}

fn clamp_roi(cx: i32, cy: i32, size: i32, w: i32, h: i32) -> (i32, i32, i32, i32) {
	let size = size.clamp(ROI_MIN_SIZE, ROI_MAX_SIZE);
	let half = size / 2;
	let mut x1 = (cx - half).max(0);
	let mut y1 = (cy - half).max(0);
	let mut x2 = (cx + half).min(w);
	let mut y2 = (cy + half).min(h);

	if (x2 - x1) < size {
		if x1 == 0 {
			x2 = (x1 + size).min(w);
		} else if x2 == w {
			x1 = (x2 - size).max(0);
		}
	}
	if (y2 - y1) < size {
		if y1 == 0 {
			y2 = (y1 + size).min(h);
		} else if y2 == h {
			y1 = (y2 - size).max(0);
		}
	}
	(x1, y1, x2, y2)
}

fn circle_offsets() -> Vec<[f32; 3]> {
	(0..POINT_NUM)
		.map(|i| {
			let a = std::f32::consts::PI / 8.0 + 2.0 * std::f32::consts::PI * (i as f32) / (POINT_NUM as f32);
			[RADIUS * a.cos(), RADIUS * a.sin(), 0.0]
		})
		.collect()
}

fn load_affine(path: &str) -> Result<([[f32; 3]; 2], String)> {
	let file = File::open(path).with_context(|| format!("open {}", path))?;
	let reader = BufReader::new(file);
	let data: AffineFile = serde_json::from_reader(reader).with_context(|| format!("parse {}", path))?;
	Ok((
		data.a_2x3,
		data.input_uv_type.unwrap_or_else(|| "unknown".to_string()),
	))
}

fn load_z_model(path: &str) -> Result<(f32, f32)> {
	let file = File::open(path).with_context(|| format!("open {}", path))?;
	let reader = BufReader::new(file);
	let data: ZModelFile = serde_json::from_reader(reader).with_context(|| format!("parse {}", path))?;
	Ok((data.a, data.b))
}

fn apply_affine(a: &[[f32; 3]; 2], u: f32, v: f32) -> (f32, f32) {
	let x = a[0][0] * u + a[0][1] * v + a[0][2];
	let y = a[1][0] * u + a[1][1] * v + a[1][2];
	(x, y)
}

fn init_camera(index: i32, role: &str) -> Result<videoio::VideoCapture> {
	let mut cap = videoio::VideoCapture::new(index, videoio::CAP_ANY)
		.with_context(|| format!("open camera {} ({})", index, role))?;
	if !videoio::VideoCapture::is_opened(&cap)? {
		anyhow::bail!("camera {} ({}) is not opened", index, role);
	}
	let _ = cap.set(videoio::CAP_PROP_FPS, 120.0);
	let _ = cap.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0);
	let _ = cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 1024.0);
	println!("[INFO] camera opened: role={}, index={}", role, index);
	Ok(cap)
}

fn camera_capture_loop(
	mut cap: videoio::VideoCapture,
	role: &'static str,
	latest: Arc<Mutex<Option<FramePacket>>>,
	running: Arc<AtomicBool>,
) {
	while running.load(Ordering::Relaxed) {
		let mut frame = Mat::default();
		match cap.read(&mut frame) {
			Ok(ok) if ok => {
				if let Ok(size) = frame.size() {
					if size.width > 0 && size.height > 0 {
						let packet = FramePacket {
							frame,
							t_sec: now_sec(),
						};
						if let Ok(mut guard) = latest.lock() {
							*guard = Some(packet);
						}
					}
				}
			}
			Ok(_) => {
				thread::sleep(Duration::from_millis(2));
			}
			Err(e) => {
				eprintln!("[CAM {}] read error: {}", role, e);
				thread::sleep(Duration::from_millis(10));
			}
		}
	}
}

fn track_ball_cv(frame: &Mat, roi_rect: (i32, i32, i32, i32)) -> Result<Option<(f32, f32, f32)>> {
	let (x1, y1, x2, y2) = roi_rect;
	let roi_w = (x2 - x1).max(1);
	let roi_h = (y2 - y1).max(1);
	let roi = Mat::roi(frame, Rect::new(x1, y1, roi_w, roi_h))?;

	let mut gray = Mat::default();
	if roi.channels() == 3 {
		imgproc::cvt_color(&roi, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
	} else {
		gray = roi.try_clone()?;
	}

	if BLUR_KSIZE > 1 {
		let mut tmp = Mat::default();
		imgproc::gaussian_blur(
			&gray,
			&mut tmp,
			Size::new(BLUR_KSIZE, BLUR_KSIZE),
			0.0,
			0.0,
			core::BORDER_DEFAULT,
		)?;
		gray = tmp;
	}

	let mut bw = Mat::default();
	imgproc::threshold(&gray, &mut bw, FIXED_THRESH, 255.0, imgproc::THRESH_BINARY)?;

	let kernel = imgproc::get_structuring_element(
		imgproc::MORPH_RECT,
		Size::new(3, 3),
		Point::new(-1, -1),
	)?;
	let mut opened = Mat::default();
	imgproc::morphology_ex(
		&bw,
		&mut opened,
		imgproc::MORPH_OPEN,
		&kernel,
		Point::new(-1, -1),
		1,
		core::BORDER_CONSTANT,
		Scalar::all(0.0),
	)?;
	let mut closed = Mat::default();
	imgproc::morphology_ex(
		&opened,
		&mut closed,
		imgproc::MORPH_CLOSE,
		&kernel,
		Point::new(-1, -1),
		2,
		core::BORDER_CONSTANT,
		Scalar::all(0.0),
	)?;

	let mut contours: Vector<Vector<Point>> = Vector::new();
	imgproc::find_contours(
		&closed,
		&mut contours,
		imgproc::RETR_EXTERNAL,
		imgproc::CHAIN_APPROX_SIMPLE,
		Point::new(0, 0),
	)?;
	if contours.is_empty() {
		return Ok(None);
	}

	let roi_cx = (roi_w as f32) / 2.0;
	let roi_cy = (roi_h as f32) / 2.0;

	let mut best_score = f64::NEG_INFINITY;
	let mut best: Option<Vector<Point>> = None;
	for i in 0..contours.len() {
		let cnt = contours.get(i)?;
		let area = imgproc::contour_area(&cnt, false)?;
		if !(MIN_AREA_PX..=MAX_AREA_PX).contains(&area) {
			continue;
		}

		let perimeter = imgproc::arc_length(&cnt, true)?;
		if perimeter <= 0.0 {
			continue;
		}

		let circularity = 4.0 * std::f64::consts::PI * (area / (perimeter * perimeter));
		if circularity < 0.6 {
			continue;
		}

		let m = imgproc::moments(&cnt, false)?;
		if m.m00.abs() < 1e-6 {
			continue;
		}
		let cx = (m.m10 / m.m00) as f32;
		let cy = (m.m01 / m.m00) as f32;
		let dx = (cx - roi_cx) as f64;
		let dy = (cy - roi_cy) as f64;
		let score = area - 0.8 * (dx * dx + dy * dy);

		if score > best_score {
			best_score = score;
			best = Some(cnt);
		}
	}

	let Some(best_cnt) = best else {
		return Ok(None);
	};

	let mut center = Point2f::default();
	let mut radius = 0.0f32;
	imgproc::min_enclosing_circle(&best_cnt, &mut center, &mut radius)?;

	Ok(Some((x1 as f32 + center.x, y1 as f32 + center.y, radius)))
}

fn autd_control_loop(
	mut autd: Box<dyn AutdController>,
	shared_target: Arc<Mutex<TargetState>>,
	running: Arc<AtomicBool>,
) {
	println!("[THREAD] AUTD control thread started");
	let offsets = circle_offsets();
	let mut last_seq = u64::MAX;
	let mut fps_count = 0u32;
	let mut fps_start = Instant::now();

	while running.load(Ordering::Relaxed) {
		let target = {
			let guard = shared_target.lock();
			match guard {
				Ok(g) => *g,
				Err(_) => {
					thread::sleep(Duration::from_millis(1));
					continue;
				}
			}
		};

		if target.seq == last_seq {
			thread::sleep(Duration::from_secs_f64(AUTD_LOOP_SLEEP_SEC));
			continue;
		}

		let center = target.pos;
		let mut foci = Vec::with_capacity(offsets.len());
		for o in &offsets {
			foci.push([center[0] + o[0], center[1] + o[1], center[2] + o[2]]);
		}
		if let Err(e) = autd.send_foci_stm(&foci) {
			eprintln!("[AUTD] send error: {}", e);
			thread::sleep(Duration::from_millis(10));
			continue;
		}

		last_seq = target.seq;
		fps_count += 1;
		let elapsed = fps_start.elapsed().as_secs_f64();
		if elapsed >= 1.0 {
			println!("[AUTD] fps={:.1}", (fps_count as f64) / elapsed);
			fps_start = Instant::now();
			fps_count = 0;
		}
	}
	println!("[THREAD] AUTD control thread stopped");
}

fn set_target(shared_target: &Arc<Mutex<TargetState>>, x: f32, y: f32, z: f32) {
	if let Ok(mut t) = shared_target.lock() {
		t.pos = [x, y, z];
		t.seq = t.seq.wrapping_add(1);
	}
}

fn main() -> Result<()> {
	println!("[INFO] Rust tracking feedback started");

	let (affine, uv_type) = load_affine(AFFINE_XY_JSON)?;
	let (z_a, z_b) = load_z_model(AFFINE_Z_JSON)?;
	println!("[INFO] loaded affine={}, uv_type={}", AFFINE_XY_JSON, uv_type);
	println!("[INFO] loaded z model={}, z = {:.6} * v + {:.6}", AFFINE_Z_JSON, z_a, z_b);

	let cap_xy = init_camera(CAMERA_XY_INDEX, "xy")?;
	let cap_z = init_camera(CAMERA_Z_INDEX, "z")?;

	let mut autd_impl = Box::new(DummyAutdController::new()) as Box<dyn AutdController>;
	autd_impl.send_static()?;
	let base_center = autd_impl.center();

	let running = Arc::new(AtomicBool::new(true));
	let latest_xy: Arc<Mutex<Option<FramePacket>>> = Arc::new(Mutex::new(None));
	let latest_z: Arc<Mutex<Option<FramePacket>>> = Arc::new(Mutex::new(None));
	let shared_target = Arc::new(Mutex::new(TargetState {
		pos: [base_center[0], base_center[1], DEFAULT_Z],
		seq: 0,
	}));

	let t_cam_xy = {
		let running = Arc::clone(&running);
		let latest = Arc::clone(&latest_xy);
		thread::spawn(move || camera_capture_loop(cap_xy, "xy", latest, running))
	};
	let t_cam_z = {
		let running = Arc::clone(&running);
		let latest = Arc::clone(&latest_z);
		thread::spawn(move || camera_capture_loop(cap_z, "z", latest, running))
	};
	let t_autd = {
		let running = Arc::clone(&running);
		let shared_target = Arc::clone(&shared_target);
		thread::spawn(move || autd_control_loop(autd_impl, shared_target, running))
	};

	highgui::named_window("Tracking", highgui::WINDOW_NORMAL)?;
	println!("[INFO] ENTER: start/pause, L: log, ESC: exit");

	let mut tracking_active = false;
	let mut roi_xy: Option<RoiState> = None;
	let mut roi_z: Option<RoiState> = None;

	let mut prev_particle_x: Option<f32> = None;
	let mut prev_particle_y: Option<f32> = None;
	let mut prev_vx = 0.0f32;
	let mut prev_vy = 0.0f32;
	let mut prev_xy_t: Option<f64> = None;
	let mut prev_xy_int_x = 0.0f32;
	let mut prev_xy_int_y = 0.0f32;

	let mut prev_particle_z: Option<f32> = None;
	let mut prev_particle_z_filt: Option<f32> = None;
	let mut prev_vz = 0.0f32;
	let mut prev_z_t: Option<f64> = None;
	let mut prev_z_error_int = 0.0f32;

	let mut last_valid_target_x = base_center[0];
	let mut last_valid_target_y = base_center[1];
	let mut last_valid_target_z = DEFAULT_Z;

	let mut loop_count = 0u32;
	let mut loop_fps_count = 0u32;
	let mut fps_start = Instant::now();
	let mut loop_fps = 0.0f64;

	let mut log_until = 0.0f64;
	let mut log_mode = String::new();
	let mut log_writer: Option<csv::Writer<BufWriter<File>>> = None;

	loop {
		if !running.load(Ordering::Relaxed) {
			break;
		}

		let packet_xy = latest_xy.lock().ok().and_then(|g| g.clone());
		let packet_z = latest_z.lock().ok().and_then(|g| g.clone());
		let (Some(packet_xy), Some(packet_z)) = (packet_xy, packet_z) else {
			thread::sleep(Duration::from_millis(1));
			continue;
		};

		let mut frame_xy = packet_xy.frame;
		let mut frame_z = packet_z.frame;
		let sz_xy = frame_xy.size()?;
		let sz_z = frame_z.size()?;
		if roi_xy.is_none() {
			roi_xy = Some(RoiState {
				cx: sz_xy.width / 2,
				cy: sz_xy.height / 2,
				size: ROI_INIT_SIZE,
			});
		}
		if roi_z.is_none() {
			roi_z = Some(RoiState {
				cx: sz_z.width / 2,
				cy: sz_z.height / 2,
				size: ROI_INIT_SIZE,
			});
		}

		let mut roi_xy_s = roi_xy.unwrap();
		let mut roi_z_s = roi_z.unwrap();

		let rect_xy = clamp_roi(roi_xy_s.cx, roi_xy_s.cy, roi_xy_s.size, sz_xy.width, sz_xy.height);
		let rect_z = clamp_roi(roi_z_s.cx, roi_z_s.cy, roi_z_s.size, sz_z.width, sz_z.height);

		let det_xy = track_ball_cv(&frame_xy, rect_xy)?;
		let det_z = track_ball_cv(&frame_z, rect_z)?;

		let mut u_xy = 0.0f32;
		let mut v_xy = 0.0f32;
		let mut v_z = 0.0f32;

		let detected_xy = if let Some((u, v, r)) = det_xy {
			u_xy = u;
			v_xy = v;
			roi_xy_s.cx = u as i32;
			roi_xy_s.cy = v as i32;
			let desired = (2.0 * (2.5 * r + ROI_MARGIN)) as i32;
			roi_xy_s.size = ((0.7 * roi_xy_s.size as f32) + (0.3 * desired as f32)) as i32;
			roi_xy_s.size = roi_xy_s.size.clamp(ROI_MIN_SIZE, ROI_MAX_SIZE);
			true
		} else {
			roi_xy_s.size = ((roi_xy_s.size as f32) * ROI_EXPAND_ON_LOST) as i32;
			roi_xy_s.size = roi_xy_s.size.clamp(ROI_MIN_SIZE, ROI_MAX_SIZE);
			false
		};

		let detected_z = if let Some((_, v, r)) = det_z {
			v_z = v;
			roi_z_s.cx = v as i32;
			roi_z_s.cy = v as i32;
			let desired = (2.0 * (2.5 * r + ROI_MARGIN)) as i32;
			roi_z_s.size = ((0.7 * roi_z_s.size as f32) + (0.3 * desired as f32)) as i32;
			roi_z_s.size = roi_z_s.size.clamp(ROI_MIN_SIZE, ROI_MAX_SIZE);
			true
		} else {
			roi_z_s.size = ((roi_z_s.size as f32) * ROI_EXPAND_ON_LOST) as i32;
			roi_z_s.size = roi_z_s.size.clamp(ROI_MIN_SIZE, ROI_MAX_SIZE);
			false
		};

		roi_xy = Some(roi_xy_s);
		roi_z = Some(roi_z_s);

		let mut target_x = last_valid_target_x;
		let mut target_y = last_valid_target_y;
		let mut target_z = last_valid_target_z;

		if tracking_active {
			if detected_xy {
				let (dx, dy) = apply_affine(&affine, u_xy, v_xy);
				let current_x = base_center[0] + dx;
				let current_y = base_center[1] + dy;

				let dt_xy = prev_xy_t
					.map(|pt| (packet_xy.t_sec - pt).max(1e-3))
					.unwrap_or(1e-3) as f32;
				let vx = if let Some(px) = prev_particle_x {
					let raw = (current_x - px) / dt_xy;
					0.5 * prev_vx + 0.5 * raw
				} else {
					0.0
				};
				let vy = if let Some(py) = prev_particle_y {
					let raw = (current_y - py) / dt_xy;
					0.5 * prev_vy + 0.5 * raw
				} else {
					0.0
				};
				prev_particle_x = Some(current_x);
				prev_particle_y = Some(current_y);
				prev_vx = vx;
				prev_vy = vy;

				let x_pred = current_x + vx * DT_PRED_XY;
				let y_pred = current_y + vy * DT_PRED_XY;
				let x_err = base_center[0] - x_pred;
				let y_err = base_center[1] - y_pred;

				prev_xy_int_x = (prev_xy_int_x + x_err * dt_xy).clamp(-XY_INTEGRAL_CLAMP, XY_INTEGRAL_CLAMP);
				prev_xy_int_y = (prev_xy_int_y + y_err * dt_xy).clamp(-XY_INTEGRAL_CLAMP, XY_INTEGRAL_CLAMP);

				target_x = base_center[0] + K_P_XY * x_err + K_I_XY * prev_xy_int_x - K_D_XY * vx;
				target_y = base_center[1] + K_P_XY * y_err + K_I_XY * prev_xy_int_y - K_D_XY * vy;

				last_valid_target_x = target_x;
				last_valid_target_y = target_y;
				prev_xy_t = Some(packet_xy.t_sec);
			}

			if detected_z {
				let current_z = z_a * v_z + z_b;
				let z_filt = prev_particle_z_filt
					.map(|p| Z_LP_ALPHA * p + (1.0 - Z_LP_ALPHA) * current_z)
					.unwrap_or(current_z);
				let dt_z = prev_z_t
					.map(|pt| (packet_z.t_sec - pt).max(1e-3))
					.unwrap_or(1e-3) as f32;
				let vz = if let Some(pz) = prev_particle_z {
					let raw = (z_filt - pz) / dt_z;
					0.5 * prev_vz + 0.5 * raw
				} else {
					0.0
				};

				prev_particle_z = Some(z_filt);
				prev_particle_z_filt = Some(z_filt);
				prev_vz = vz;

				let z_err = DEFAULT_Z - current_z;
				prev_z_error_int = (prev_z_error_int + z_err * dt_z).clamp(-Z_INTEGRAL_CLAMP, Z_INTEGRAL_CLAMP);
				prev_z_t = Some(packet_z.t_sec);

				let z_pred = z_filt + vz * DT_PRED_Z - 0.5 * GRAVITY_MM_S2 * DT_PRED_Z * DT_PRED_Z;
				target_z = DEFAULT_Z + K_P_Z * (DEFAULT_Z - z_pred) + K_I_Z * prev_z_error_int - K_D_Z * vz;
				target_z = target_z.clamp(Z_MIN, Z_MAX);
				last_valid_target_z = target_z;
			}

			set_target(&shared_target, target_x, target_y, target_z);
		}

		let now = now_sec();
		if LOG_ENABLED && log_until > now {
			if let Some(w) = log_writer.as_mut() {
				let _ = w.write_record(&[
					format!("{:.4}", now),
					log_mode.clone(),
					format!("{:.2}", u_xy),
					format!("{:.2}", v_xy),
					format!("{:.2}", v_z),
					format!("{:.3}", target_x),
					format!("{:.3}", target_y),
					format!("{:.3}", target_z),
					format!("{:.3}", base_center[0]),
					format!("{:.3}", base_center[1]),
					format!("{:.3}", DEFAULT_Z),
				]);
				let _ = w.flush();
			}
		} else if log_until > 0.0 {
			log_until = 0.0;
			log_writer = None;
			println!("[LOG] session completed");
		}

		loop_count += 1;
		loop_fps_count += 1;
		let elapsed = fps_start.elapsed().as_secs_f64();
		if elapsed >= 1.0 {
			loop_fps = (loop_fps_count as f64) / elapsed;
			loop_fps_count = 0;
			fps_start = Instant::now();
		}

		if loop_count % DISPLAY_EVERY_N_FRAMES == 0 {
			imgproc::put_text(
				&mut frame_xy,
				&format!("Loop FPS: {:.1}", loop_fps),
				Point::new(10, 30),
				imgproc::FONT_HERSHEY_SIMPLEX,
				0.7,
				Scalar::new(255.0, 255.0, 255.0, 0.0),
				2,
				imgproc::LINE_AA,
				false,
			)?;
			imgproc::put_text(
				&mut frame_xy,
				if tracking_active { "STATUS: ACTIVE" } else { "STATUS: WAIT" },
				Point::new(10, sz_xy.height - 20),
				imgproc::FONT_HERSHEY_SIMPLEX,
				0.8,
				if tracking_active {
					Scalar::new(0.0, 255.0, 0.0, 0.0)
				} else {
					Scalar::new(0.0, 165.0, 255.0, 0.0)
				},
				2,
				imgproc::LINE_AA,
				false,
			)?;

			imgproc::put_text(
				&mut frame_z,
				&format!("v_z={:.1}", v_z),
				Point::new(10, 30),
				imgproc::FONT_HERSHEY_SIMPLEX,
				0.7,
				Scalar::new(255.0, 255.0, 255.0, 0.0),
				2,
				imgproc::LINE_AA,
				false,
			)?;

			let mut tiled = Mat::default();
			let mut mats = Vector::<Mat>::new();
			mats.push(frame_xy.try_clone()?);
			mats.push(frame_z.try_clone()?);
			core::hconcat(&mats, &mut tiled)?;
			highgui::imshow("Tracking", &tiled)?;
		}

		let key = highgui::wait_key(1)?;
		if key == 27 {
			break;
		}
		if key == 13 {
			tracking_active = !tracking_active;
			if tracking_active {
				println!("[INFO] tracking activated");
				prev_particle_x = None;
				prev_particle_y = None;
				prev_particle_z = None;
				prev_particle_z_filt = None;
				prev_vx = 0.0;
				prev_vy = 0.0;
				prev_vz = 0.0;
				prev_xy_t = None;
				prev_z_t = None;
				prev_xy_int_x = 0.0;
				prev_xy_int_y = 0.0;
				prev_z_error_int = 0.0;
			} else {
				println!("[INFO] tracking paused");
				set_target(&shared_target, base_center[0], base_center[1], DEFAULT_Z);
			}
		}
		if key == 'l' as i32 || key == 'L' as i32 {
			if LOG_ENABLED && log_until <= now_sec() {
				if let Some(parent) = Path::new(LOG_CSV_PATH).parent() {
					std::fs::create_dir_all(parent).ok();
				}
				let f = File::create(LOG_CSV_PATH)?;
				let mut w = csv::Writer::from_writer(BufWriter::new(f));
				w.write_record([
					"timestamp",
					"mode",
					"u_xy_px",
					"v_xy_px",
					"v_z_px",
					"autd_target_x_mm",
					"autd_target_y_mm",
					"autd_target_z_mm",
					"center_x_mm",
					"center_y_mm",
					"center_z_mm",
				])?;
				w.flush()?;
				log_writer = Some(w);
				log_mode = if tracking_active { "PID" } else { "FIXED" }.to_string();
				log_until = now_sec() + LOG_DURATION_SEC;
				println!("[LOG] started mode={} duration={}s", log_mode, LOG_DURATION_SEC as i32);
			}
		}
	}

	running.store(false, Ordering::Relaxed);
	let _ = t_cam_xy.join();
	let _ = t_cam_z.join();
	let _ = t_autd.join();
	highgui::destroy_all_windows()?;
	println!("[INFO] finished");
	std::io::stdout().flush().ok();
	Ok(())
}
