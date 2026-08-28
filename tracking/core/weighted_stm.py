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


def cycle_frequency_preserving_slot_rate(
    *,
    base_cycle_frequency_hz: float,
    base_point_num: int,
    total_slots: int,
) -> float:
    """Keep the AUTD STM slot update rate unchanged after slot expansion.

    ``FociSTM`` interprets its frequency as the frequency of one complete STM
    cycle.  Therefore, expanding an 8-point, 100 Hz STM to 64 dwell slots
    without changing the cycle frequency would increase the slot update rate
    from 800 Hz to 6400 Hz.  The default Silencer cannot complete its phase
    interpolation in that short sampling period.  Scaling the cycle frequency
    by ``base_point_num / total_slots`` preserves the original slot period.
    """

    frequency = float(base_cycle_frequency_hz)
    point_num = int(base_point_num)
    slots = int(total_slots)
    if frequency <= 0.0:
        raise ValueError("base_cycle_frequency_hz must be positive")
    if point_num <= 0:
        raise ValueError("base_point_num must be positive")
    if slots <= 0:
        raise ValueError("total_slots must be positive")
    return frequency * float(point_num) / float(slots)


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
    """Schedule each circular focus according to its slot allocation.

    ``None`` returns the original offsets unchanged, which is the normal
    application path. Allocated presentations are spread over several circular
    passes instead of grouping every repetition consecutively. In particular,
    equal counts produce the original point order repeated several times, so a
    zero-bias control pulse retains the normal 8-point STM timing.
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

    round_count = int(np.max(counts))
    scheduled_rounds: list[set[int]] = []
    for count in counts:
        # Place each occurrence near the center of an equal subdivision of the
        # available rounds.  This is a deterministic, evenly spaced schedule.
        rounds = np.floor(
            (np.arange(int(count), dtype=float) + 0.5)
            * float(round_count)
            / float(count)
        ).astype(int)
        scheduled_rounds.append(set(int(value) for value in rounds))

    indices = [
        focus_index
        for round_index in range(round_count)
        for focus_index, rounds in enumerate(scheduled_rounds)
        if round_index in rounds
    ]
    return values[np.asarray(indices, dtype=int)]
