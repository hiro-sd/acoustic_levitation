from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeightedDwellPattern:
    """Discrete dwell allocation for the points of one circular STM cycle."""

    bias_angle_rad: float
    bias_level: float
    slot_counts: tuple[int, ...]

    @property
    def total_slots(self) -> int:
        return int(sum(self.slot_counts))


def make_weighted_dwell_pattern(
    *,
    point_num: int,
    total_slots: int,
    bias_angle_rad: float,
    bias_level: float,
    phase_offset_rad: float = np.pi / 8.0,
    minimum_slots_per_point: int = 1,
) -> WeightedDwellPattern:
    """Allocate a fixed number of STM slots with a cosine-shaped bias.

    ``bias_angle_rad`` names the side of the ring receiving the longest dwell.
    It does not assume that the resulting sphere force points in that direction;
    the force-characterization experiment determines that sign empirically.
    """

    n = int(point_num)
    slots = int(total_slots)
    minimum = int(minimum_slots_per_point)
    level = float(bias_level)
    if n < 2:
        raise ValueError("point_num must be at least 2")
    if minimum < 0:
        raise ValueError("minimum_slots_per_point must be non-negative")
    if slots < n * minimum:
        raise ValueError(
            "total_slots is too small for minimum_slots_per_point"
        )
    if not 0.0 <= level < 1.0:
        raise ValueError("bias_level must satisfy 0.0 <= bias_level < 1.0")

    angles = (
        float(phase_offset_rad)
        + 2.0 * np.pi * np.arange(n, dtype=float) / float(n)
    )
    weights = 1.0 + level * np.cos(angles - float(bias_angle_rad))
    weights = np.maximum(weights, 0.0)

    remaining = slots - n * minimum
    if remaining == 0:
        counts = np.full(n, minimum, dtype=int)
    else:
        quotas = remaining * weights / float(np.sum(weights))
        additions = np.floor(quotas).astype(int)
        remainder = remaining - int(np.sum(additions))
        if remainder:
            # Stable sorting keeps the result deterministic when fractions tie.
            order = np.argsort(-(quotas - additions), kind="stable")
            additions[order[:remainder]] += 1
        counts = additions + minimum

    return WeightedDwellPattern(
        bias_angle_rad=float(bias_angle_rad),
        bias_level=level,
        slot_counts=tuple(int(value) for value in counts),
    )


def expand_focus_offsets(
    offsets: np.ndarray,
    slot_counts: tuple[int, ...] | list[int] | np.ndarray | None,
) -> np.ndarray:
    """Repeat each circular focus according to its dwell allocation.

    ``None`` returns the original offsets unchanged, which is the normal
    application path. Consecutive repeats hold one focus for longer while the
    configured STM cycle frequency remains unchanged.
    """

    values = np.asarray(offsets, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("offsets must have shape (point_num, 3)")
    if slot_counts is None:
        return values

    counts = np.asarray(slot_counts, dtype=int)
    if counts.ndim != 1 or counts.size != values.shape[0]:
        raise ValueError("slot_counts must have one entry per focus")
    if np.any(counts < 1):
        raise ValueError("each focus must receive at least one dwell slot")
    return np.repeat(values, counts, axis=0)
