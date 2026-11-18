#!/usr/bin/env python3
from datetime import datetime, timedelta

import pytest

from sunflow.time_handler import generate_time_steps, round_time


class TestGenerateTimeSteps:
    def test_past_observations_are_chronological_and_end_at_time_step(self):
        time_step = datetime(2026, 3, 9, 12, 0)

        steps = generate_time_steps(time_step, n_steps=4, step_minutes=15)

        assert steps == [
            datetime(2026, 3, 9, 11, 15),
            datetime(2026, 3, 9, 11, 30),
            datetime(2026, 3, 9, 11, 45),
            datetime(2026, 3, 9, 12, 0),
        ]

    def test_past_observations_respect_step_minutes(self):
        steps = generate_time_steps(
            datetime(2026, 3, 9, 12, 0), n_steps=3, step_minutes=10
        )

        assert steps == [
            datetime(2026, 3, 9, 11, 40),
            datetime(2026, 3, 9, 11, 50),
            datetime(2026, 3, 9, 12, 0),
        ]

    def test_single_past_observation_is_the_reference_time(self):
        time_step = datetime(2026, 3, 9, 12, 0)

        assert generate_time_steps(time_step, n_steps=1) == [time_step]

    def test_previous_day_starts_one_step_after_reference_minus_a_day(self):
        time_step = datetime(2026, 3, 9, 12, 0)

        steps = generate_time_steps(
            time_step, n_steps=3, step_minutes=15, type="previous_day"
        )

        assert steps == [
            datetime(2026, 3, 8, 12, 15),
            datetime(2026, 3, 8, 12, 30),
            datetime(2026, 3, 8, 12, 45),
        ]

    def test_previous_day_excludes_the_reference_time_itself(self):
        time_step = datetime(2026, 3, 9, 12, 0)

        steps = generate_time_steps(time_step, n_steps=4, type="previous_day")

        assert time_step - timedelta(days=1) not in steps

    @pytest.mark.parametrize("mode", ["past_observations", "previous_day"])
    def test_requested_number_of_steps_is_returned(self, mode):
        steps = generate_time_steps(datetime(2026, 3, 9, 12, 0), n_steps=24, type=mode)

        assert len(steps) == 24
        assert steps == sorted(steps)


class TestRoundTime:
    def test_rounds_up_to_next_frequency_interval_after_subtracting_delay(self):
        # 12:40 - 24 min delay = 12:16, which rounds up to 12:30.
        rounded = round_time(datetime(2026, 3, 9, 12, 40), delay=24, frequency=15)

        assert rounded == datetime(2026, 3, 9, 12, 30)

    def test_seconds_and_microseconds_are_zeroed(self):
        rounded = round_time(
            datetime(2026, 3, 9, 12, 40, 37, 123456), delay=24, frequency=15
        )

        assert rounded == datetime(2026, 3, 9, 12, 30)
        assert rounded.second == 0
        assert rounded.microsecond == 0

    def test_time_already_on_interval_advances_a_full_interval(self):
        # Documents current behaviour: an exact interval boundary is not a no-op.
        rounded = round_time(datetime(2026, 3, 9, 12, 15), delay=0, frequency=15)

        assert rounded == datetime(2026, 3, 9, 12, 30)

    def test_rounding_can_cross_an_hour_boundary(self):
        rounded = round_time(datetime(2026, 3, 9, 12, 55), delay=0, frequency=15)

        assert rounded == datetime(2026, 3, 9, 13, 0)

    def test_delay_can_push_the_result_into_the_previous_hour(self):
        rounded = round_time(datetime(2026, 3, 9, 12, 10), delay=24, frequency=15)

        assert rounded == datetime(2026, 3, 9, 12, 0)

    def test_ten_minute_frequency(self):
        rounded = round_time(datetime(2026, 3, 9, 12, 33), delay=5, frequency=10)

        assert rounded == datetime(2026, 3, 9, 12, 30)

    def test_result_is_never_more_than_one_interval_past_the_offset_time(self):
        dt = datetime(2026, 3, 9, 12, 7, 42)
        frequency = 15

        rounded = round_time(dt, delay=24, frequency=frequency)

        offset = dt - timedelta(minutes=24)
        assert offset < rounded <= offset + timedelta(minutes=frequency)
