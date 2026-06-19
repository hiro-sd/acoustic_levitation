from dataclasses import dataclass


@dataclass
class HomePosition:
    x: float
    y: float
    z: float


@dataclass
class Measurement3D:
    """2台のカメラから得た、時刻の異なる3次元測定値。"""

    detected_xy: bool
    detected_z: bool
    x: float | None = None
    y: float | None = None
    z: float | None = None
    t_xy: float | None = None
    t_z: float | None = None


@dataclass
class Target3D:
    x: float
    y: float
    z: float


@dataclass
class ControllerDebug:
    """画面表示とログに公開するコントローラ内部状態。"""

    current_x: float | None = None
    current_y: float | None = None
    current_z: float | None = None
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    x_pred: float | None = None
    y_pred: float | None = None
    z_pred: float | None = None
    error_x: float | None = None
    error_y: float | None = None
    error_z: float | None = None
    integral_x: float = 0.0
    integral_y: float = 0.0
    integral_z: float = 0.0
