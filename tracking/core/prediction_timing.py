from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionTiming:
    measurement_age_sec: float
    actuation_delay_sec: float
    horizon_sec: float


def prediction_timing_from_timestamps(
    *,
    measurement_time_sec: float,
    now_sec: float,
    actuation_delay_sec: float,
) -> PredictionTiming:
    """Build a prediction horizon from measurement age and AUTD latency."""
    measurement_age = max(
        0.0,
        float(now_sec) - float(measurement_time_sec),
    )
    actuation_delay = max(0.0, float(actuation_delay_sec))
    return PredictionTiming(
        measurement_age_sec=measurement_age,
        actuation_delay_sec=actuation_delay,
        horizon_sec=measurement_age + actuation_delay,
    )
