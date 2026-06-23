from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoPoint3D:
    """Triangulated point in cam1/XY-camera coordinates."""

    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass(frozen=True)
class RigidTransform:
    """p_dst = R @ p_src + t"""

    R: np.ndarray
    t: np.ndarray

    @classmethod
    def from_npz(cls, path: str | Path) -> "RigidTransform":
        data = np.load(path, allow_pickle=True)
        if "R" not in data or "t" not in data:
            raise KeyError(f"{path} must contain R and t.")
        return cls(
            R=data["R"].astype(np.float64),
            t=data["t"].astype(np.float64).reshape(3),
        )

    def apply(self, point: StereoPoint3D | np.ndarray) -> np.ndarray:
        p = point.as_array() if isinstance(point, StereoPoint3D) else np.asarray(point, dtype=np.float64)
        return self.R @ p.reshape(3) + self.t


class StereoTriangulator:
    """
    Convert a pair of 2D detections into a 3D point in cam1 coordinates.

    OpenCV stereoCalibrate returns R/T such that:
        p_cam2 = R @ p_cam1 + T
    """

    def __init__(
        self,
        camera_matrix_l: np.ndarray,
        dist_l: np.ndarray,
        camera_matrix_r: np.ndarray,
        dist_r: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        input_is_undistorted: bool,
    ):
        self.camera_matrix_l = camera_matrix_l.astype(np.float64)
        self.dist_l = dist_l.astype(np.float64)
        self.camera_matrix_r = camera_matrix_r.astype(np.float64)
        self.dist_r = dist_r.astype(np.float64)
        self.R = R.astype(np.float64)
        self.T = T.astype(np.float64).reshape(3, 1)
        self.input_is_undistorted = bool(input_is_undistorted)

        self._proj_l = np.hstack([np.eye(3), np.zeros((3, 1))])
        self._proj_r = np.hstack([self.R, self.T])

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        input_is_undistorted: bool,
    ) -> "StereoTriangulator":
        data = np.load(path, allow_pickle=True)
        return cls(
            camera_matrix_l=data["camera_matrix_l"],
            dist_l=data["dist_l"],
            camera_matrix_r=data["camera_matrix_r"],
            dist_r=data["dist_r"],
            R=data["R"],
            T=data["T"],
            input_is_undistorted=input_is_undistorted,
        )

    def triangulate_pixels(
        self,
        uv_l: tuple[float, float],
        uv_r: tuple[float, float],
    ) -> StereoPoint3D:
        pts_l = np.array([[uv_l]], dtype=np.float64)
        pts_r = np.array([[uv_r]], dtype=np.float64)

        if self.input_is_undistorted:
            zeros_l = np.zeros_like(self.dist_l)
            zeros_r = np.zeros_like(self.dist_r)
            norm_l = cv2.undistortPoints(pts_l, self.camera_matrix_l, zeros_l)
            norm_r = cv2.undistortPoints(pts_r, self.camera_matrix_r, zeros_r)
        else:
            norm_l = cv2.undistortPoints(pts_l, self.camera_matrix_l, self.dist_l)
            norm_r = cv2.undistortPoints(pts_r, self.camera_matrix_r, self.dist_r)

        point_h = cv2.triangulatePoints(
            self._proj_l,
            self._proj_r,
            norm_l.reshape(-1, 2).T,
            norm_r.reshape(-1, 2).T,
        )
        point = (point_h[:3] / point_h[3]).reshape(3)
        return StereoPoint3D(
            x=float(point[0]),
            y=float(point[1]),
            z=float(point[2]),
        )
