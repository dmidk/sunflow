#!/usr/bin/env python3
from datetime import datetime, timedelta


def generate_time_steps(
    time_step: datetime,
    n_steps: int,
    step_minutes: int = 15,
    type: str = "past_observations",
) -> list[datetime]:
    """Generate an ordered list of datetime objects for data loading.

    Supports two modes controlled by 'type':

    - 'past_observations': Returns n_steps timesteps ending at
      time_step (inclusive), spaced step_minutes apart, in
      chronological order.
    - 'previous_day': Returns n_steps timesteps starting one
      step_minutes interval after time_step minus one day,
      used for clearsky lookups on the previous day.

    Args:
        time_step: Reference datetime (the current time step).
        n_steps: Number of timesteps to generate.
        step_minutes: Interval between timesteps in minutes. Default 15.
        type: Generation mode — 'past_observations' or
            'previous_day'. Default 'past_observations'.

    Returns:
        List of datetimes in chronological order.
    """
    if type == "past_observations":
        return [
            time_step - timedelta(minutes=i * step_minutes) for i in range(0, n_steps)
        ][::-1]
    else:  # previous_day
        return [
            time_step + timedelta(minutes=i * step_minutes) - timedelta(days=1)
            for i in range(1, n_steps + 1)
        ]


def round_time(dt: datetime, delay: int, frequency: int) -> datetime:
    """Round a datetime to the latest available data interval.

    First subtracts delay minutes to account for data availability lag,
    then rounds up to the nearest multiple of frequency minutes while
    zeroing out seconds and microseconds.

    Args:
        dt: Current datetime (typically UTC now).
        delay: Data availability delay in minutes to subtract before rounding.
        frequency: Rounding interval in minutes.

    Returns:
        Rounded datetime representing the latest available data timestamp.
    """
    offset_time = dt - timedelta(minutes=delay)
    rounded_time = offset_time + timedelta(
        minutes=frequency - offset_time.minute % frequency,
        seconds=-offset_time.second,
        microseconds=-offset_time.microsecond,
    )
    return rounded_time
