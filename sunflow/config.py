#!/usr/bin/env python3
import os
from dataclasses import dataclass
from typing import Self

# Predefined domain options
# Format: lon_min,lat_min,lon_max,lat_max
DOMAIN_OPTIONS: dict[str, str | None] = {
    "DENMARK": "4,50,18,62",
    "NW_EUROPE": "-10.75,47.25,20,63.5",
    "NW_EUROPE_SATELLITE": "-20.75,37.25,30,73.5",
    "CUSTOM": None,
}


@dataclass
class S3Config:
    """S3 storage configuration."""

    endpoint_url: str
    bucket: str
    input_prefix: str
    output_prefix: str

    @classmethod
    def from_env(cls) -> Self:
        """Load S3 configuration from environment variables with defaults.

        Reads the following environment variables:

        - S3_ENDPOINT_URL (default: None)
        - S3_BUCKET (default: None)
        - S3_INPUT_PREFIX (default: satellite_data)
        - S3_OUTPUT_PREFIX (default: nowcasts)
        """
        return cls(
            endpoint_url=os.getenv("S3_ENDPOINT_URL", None),
            bucket=os.getenv("S3_BUCKET", None),
            input_prefix=os.getenv("S3_INPUT_PREFIX", "satellite_data"),
            output_prefix=os.getenv("S3_OUTPUT_PREFIX", "nowcasts"),
        )


@dataclass
class NowcastConfig:
    """Application configuration from environment variables."""

    nowcast_directory: str
    ens_members: int
    past_steps: int
    future_steps: int
    input_data_availability_delay_minutes: int
    input_data_frequency_minutes: int
    max_waiting_time_minutes: int
    satellite_data_directory: str
    max_clearsky_fallback_days: int

    @classmethod
    def from_env(cls) -> Self:
        """Load nowcast configuration from environment variables with defaults.

        Reads the following environment variables:

        - NOWCAST_DIRECTORY (default: .)
        - ENS_MEMBERS (default: 1)
        - PAST_STEPS (default: 4)
        - FUTURE_STEPS (default: 24)
        - INPUT_DATA_AVAILABILITY_DELAY_MINUTES (default: 24)
        - INPUT_DATA_FREQUENCY_MINUTES (default: 15)
        - MAX_WAITING_TIME_MINUTES (default: 27)
        - SATELLITE_DATA_DIRECTORY (default: .)
        - MAX_CLEARSKY_FALLBACK_DAYS (default: 3)
        """
        return cls(
            nowcast_directory=os.getenv("NOWCAST_DIRECTORY", "."),
            ens_members=int(os.getenv("ENS_MEMBERS", "1")),
            past_steps=int(os.getenv("PAST_STEPS", "4")),
            future_steps=int(os.getenv("FUTURE_STEPS", "24")),
            input_data_availability_delay_minutes=int(
                os.getenv("INPUT_DATA_AVAILABILITY_DELAY_MINUTES", "24")
            ),
            input_data_frequency_minutes=int(
                os.getenv("INPUT_DATA_FREQUENCY_MINUTES", "15")
            ),
            max_waiting_time_minutes=int(os.getenv("MAX_WAITING_TIME_MINUTES", "27")),
            satellite_data_directory=os.getenv("SATELLITE_DATA_DIRECTORY", "."),
            max_clearsky_fallback_days=int(os.getenv("MAX_CLEARSKY_FALLBACK_DAYS", "3")),
        )
