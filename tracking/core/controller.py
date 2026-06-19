import numpy as np

from .config import AppConfig
from .models import ControllerDebug, HomePosition, Measurement3D, Target3D


class PredictionPIDController:
    """
    XY: 速度推定 + 予測PID
    Z : 位置LPF + 速度推定 + 予測PID

    これまで main loop 内に直書きしていた制御部分をここへ移す。
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.reset()

    def reset(self, initial_target: Target3D | None = None):
        # XY state
        self.prev_particle_x = None
        self.prev_particle_y = None
        self.prev_xy_meas_time = None
        self.prev_vx = 0.0
        self.prev_vy = 0.0

        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_xy_integral_time = None

        # Z state
        self.prev_particle_z = None
        self.prev_particle_z_filt = None
        self.prev_z_meas_time = None
        self.prev_vz = 0.0

        self.integral_z = 0.0
        self.prev_z_integral_time = None

        if initial_target is None:
            self.last_valid_target = None
        else:
            self.last_valid_target = Target3D(
                float(initial_target.x),
                float(initial_target.y),
                float(initial_target.z),
            )

        self.debug = ControllerDebug()

    def ensure_target_initialized(self, home: HomePosition):
        if self.last_valid_target is None:
            self.last_valid_target = Target3D(home.x, home.y, home.z)

    def update(self, meas: Measurement3D, home: HomePosition) -> tuple[Target3D, ControllerDebug]:
        """
        新しい測定値に基づいて、AUTDへ送るSTM中心 target を計算する。

        測定がない軸は last_valid_target を保持する。
        """
        self.ensure_target_initialized(home)

        target_x = self.last_valid_target.x
        target_y = self.last_valid_target.y
        target_z = self.last_valid_target.z

        # XY prediction PID
        if meas.detected_xy and meas.x is not None and meas.y is not None and meas.t_xy is not None:
            current_x = float(meas.x)
            current_y = float(meas.y)

            if self.prev_particle_x is not None and self.prev_xy_meas_time is not None:
                dt_xy = max(1e-3, float(meas.t_xy - self.prev_xy_meas_time))

                raw_vx = (current_x - self.prev_particle_x) / dt_xy
                raw_vy = (current_y - self.prev_particle_y) / dt_xy

                vx = 0.5 * self.prev_vx + 0.5 * raw_vx
                vy = 0.5 * self.prev_vy + 0.5 * raw_vy
            else:
                dt_xy = 0.0
                vx = 0.0
                vy = 0.0

            self.prev_particle_x = current_x
            self.prev_particle_y = current_y
            self.prev_xy_meas_time = float(meas.t_xy)
            self.prev_vx = vx
            self.prev_vy = vy

            x_pred = current_x + vx * self.cfg.dt_pred_xy
            y_pred = current_y + vy * self.cfg.dt_pred_xy

            setpoint_x = float(home.x)
            setpoint_y = float(home.y)

            x_error = setpoint_x - x_pred
            y_error = setpoint_y - y_pred

            if self.prev_xy_integral_time is not None:
                dt_int_xy = max(1e-3, float(meas.t_xy - self.prev_xy_integral_time))
            else:
                dt_int_xy = 0.0

            self.prev_xy_integral_time = float(meas.t_xy)

            if dt_int_xy > 0:
                self.integral_x = float(
                    np.clip(
                        self.integral_x + x_error * dt_int_xy,
                        -self.cfg.xy_integral_clamp,
                        self.cfg.xy_integral_clamp,
                    )
                )
                self.integral_y = float(
                    np.clip(
                        self.integral_y + y_error * dt_int_xy,
                        -self.cfg.xy_integral_clamp,
                        self.cfg.xy_integral_clamp,
                    )
                )

            target_x = (
                setpoint_x
                + self.cfg.kp_xy * x_error
                + self.cfg.ki_xy * self.integral_x
                - self.cfg.kd_xy * vx
            )

            target_y = (
                setpoint_y
                + self.cfg.kp_xy * y_error
                + self.cfg.ki_xy * self.integral_y
                - self.cfg.kd_xy * vy
            )

            self.debug.current_x = current_x
            self.debug.current_y = current_y
            self.debug.vx = vx
            self.debug.vy = vy
            self.debug.x_pred = x_pred
            self.debug.y_pred = y_pred
            self.debug.error_x = x_error
            self.debug.error_y = y_error
            self.debug.integral_x = self.integral_x
            self.debug.integral_y = self.integral_y

        # Z prediction PID
        if meas.detected_z and meas.z is not None and meas.t_z is not None:
            current_z = float(meas.z)

            # z位置そのものをローパス
            if self.prev_particle_z_filt is None:
                z_filt = current_z
            else:
                z_filt = (
                    self.cfg.z_lpf_alpha * self.prev_particle_z_filt
                    + (1.0 - self.cfg.z_lpf_alpha) * current_z
                )

            # ローパス後の位置から速度を推定し、速度も平滑化
            if self.prev_particle_z is not None and self.prev_z_meas_time is not None:
                dt_z = max(1e-3, float(meas.t_z - self.prev_z_meas_time))
                raw_vz = (z_filt - self.prev_particle_z) / dt_z
                vz = 0.5 * self.prev_vz + 0.5 * raw_vz
            else:
                dt_z = 0.0
                vz = 0.0

            self.prev_particle_z = z_filt
            self.prev_particle_z_filt = z_filt
            self.prev_z_meas_time = float(meas.t_z)
            self.prev_vz = vz

            if self.cfg.use_gravity_prediction_z:
                # 自由飛行ではないので通常はOFF推奨。
                # 試す場合のみ使う。
                z_pred = (
                    z_filt
                    + vz * self.cfg.dt_pred_z
                    - 0.5 * self.cfg.gravity_mm_s2 * (self.cfg.dt_pred_z ** 2)
                )
            else:
                z_pred = z_filt + vz * self.cfg.dt_pred_z

            setpoint_z = float(home.z)
            z_error = setpoint_z - z_pred

            if self.prev_z_integral_time is not None:
                dt_int_z = max(1e-3, float(meas.t_z - self.prev_z_integral_time))
            else:
                dt_int_z = 0.0

            self.prev_z_integral_time = float(meas.t_z)

            if dt_int_z > 0:
                self.integral_z = float(
                    np.clip(
                        self.integral_z + z_error * dt_int_z,
                        -self.cfg.z_integral_clamp,
                        self.cfg.z_integral_clamp,
                    )
                )

            target_z = (
                setpoint_z
                + self.cfg.kp_z * z_error
                + self.cfg.ki_z * self.integral_z
                - self.cfg.kd_z * vz
            )

            target_z = float(np.clip(target_z, self.cfg.z_min, self.cfg.z_max))

            self.debug.current_z = current_z
            self.debug.vz = vz
            self.debug.z_pred = z_pred
            self.debug.error_z = z_error
            self.debug.integral_z = self.integral_z

        self.last_valid_target = Target3D(
            float(target_x),
            float(target_y),
            float(target_z),
        )

        return self.last_valid_target, self.debug
