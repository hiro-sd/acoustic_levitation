"""Reusable AUTD3 SDK 0.11 command builders for standalone experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autd3.commands import FociStm, Pattern, PatternStm
from autd3.geometry import Geometry
from autd3.units import Hz, m, s
from autd3.value import ControlPoint, ControlPoints, Intensity, Nearest
from autd3_pattern import focus, set_intensity, wavelength
from autd3_pattern_holo import (
    AmplitudeTarget,
    GspatOption,
    IntensityConstraint,
    Pa,
    gspat,
)


DEFAULT_WAVELENGTH = wavelength(340.0 * m / s)


def make_single_focus_stm(
    points: Sequence[np.ndarray],
    cycle_frequency_hz: float,
) -> FociStm:
    return FociStm(
        Nearest(float(cycle_frequency_hz) * Hz),
        [ControlPoints([ControlPoint(point)]) for point in points],
    )


def make_multi_focus_stm(
    frames: Sequence[Sequence[np.ndarray]],
    cycle_frequency_hz: float,
    intensity: Intensity | int = Intensity.MAX,
) -> FociStm:
    return FociStm(
        Nearest(float(cycle_frequency_hz) * Hz),
        [
            ControlPoints(
                [ControlPoint(point) for point in points],
                intensity=intensity,
            )
            for points in frames
        ],
    )


def make_gspat_pattern(
    geometry: Geometry,
    points: Sequence[np.ndarray],
    *,
    pressure_pa: float,
    repeat: int = 100,
) -> Pattern:
    phases = geometry.phase_buffer()
    intensities = geometry.intensity_buffer()
    gspat(
        geometry,
        [
            AmplitudeTarget(point, float(pressure_pa) * Pa)
            for point in points
        ],
        DEFAULT_WAVELENGTH,
        GspatOption(
            repeat=int(repeat),
            constraint=IntensityConstraint.Clamp(Intensity.MIN, Intensity.MAX),
        ),
        phases,
        intensities,
    )
    return Pattern(phases, intensities)


def make_gspat_pattern_stm(
    geometry: Geometry,
    frames: Sequence[Sequence[np.ndarray]],
    cycle_frequency_hz: float,
    *,
    pressure_pa: float,
    repeat: int = 100,
) -> PatternStm:
    phases = []
    intensities = []
    option = GspatOption(
        repeat=int(repeat),
        constraint=IntensityConstraint.Clamp(Intensity.MIN, Intensity.MAX),
    )

    for points in frames:
        phase = geometry.phase_buffer()
        intensity = geometry.intensity_buffer()
        gspat(
            geometry,
            [
                AmplitudeTarget(point, float(pressure_pa) * Pa)
                for point in points
            ],
            DEFAULT_WAVELENGTH,
            option,
            phase,
            intensity,
        )
        phases.append(phase)
        intensities.append(intensity)

    return PatternStm(
        Nearest(float(cycle_frequency_hz) * Hz),
        phases,
        intensities,
    )


def make_masked_focus_pattern_stm(
    geometry: Geometry,
    points: Sequence[np.ndarray],
    cycle_frequency_hz: float,
    *,
    mask_center_xy: Sequence[float],
    mask_radius_mm: float,
    intensity: Intensity | int = Intensity.MAX,
) -> PatternStm:
    phases = []
    for point in points:
        phase = geometry.phase_buffer()
        focus(geometry, point, DEFAULT_WAVELENGTH, phase)
        phases.append(phase)

    intensities = geometry.intensity_buffer()
    set_intensity(intensity, intensities)
    values = intensities.to_numpy()
    cx, cy = float(mask_center_xy[0]), float(mask_center_xy[1])
    radius2 = float(mask_radius_mm) ** 2
    for device in geometry:
        positions = device.positions()
        enabled = (
            (positions[:, 0] - cx) ** 2
            + (positions[:, 1] - cy) ** 2
            <= radius2
        )
        values[device.idx(), ~enabled] = 0
    intensities.copy_from(values)

    return PatternStm(
        Nearest(float(cycle_frequency_hz) * Hz),
        phases,
        [intensities] * len(phases),
    )
