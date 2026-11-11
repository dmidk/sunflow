#!/usr/bin/env python3
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

DEFAULT_ENSEMBLE_STATISTICS = "median,mean,p10,p25,p75,p90"
DEFAULT_OUTPUT_FILENAME_FORMAT = "SolarNowcast_{iso8601_compact_timestamp}.nc"
_ALLOWED_STATISTICS = {
    "median",
    "mean",
    "p10",
    "p25",
    "p75",
    "p90",
    "10th_percentile",
    "25th_percentile",
    "75th_percentile",
    "90th_percentile",
}
_STATISTIC_ALIASES = {
    "10th_percentile": "p10",
    "25th_percentile": "p25",
    "75th_percentile": "p75",
    "90th_percentile": "p90",
}


# Predefined domain options
# Format: lon_min,lat_min,lon_max,lat_max
DOMAIN_OPTIONS: dict[str, str | None] = {
    "DENMARK": "4,50,18,62",
    "NW_EUROPE": "-10.75,47.25,20,63.5",
    "NW_EUROPE_SATELLITE": "-20.75,37.25,30,73.5",
    "CUSTOM": None,
}

# Tiny absolute tolerance for floating-point boundary comparisons in degrees.
COVERAGE_ABS_TOL_DEGREES = 1e-9


@dataclass
class S3Config:
    """S3 storage configuration."""

    endpoint_url: str
    bucket: str
    input_prefix: str
    output_prefix: str
    verify_ssl: bool = True

    @classmethod
    def from_env(cls) -> Self:
        """Load S3 configuration from environment variables with defaults.

        Reads the following environment variables:

        - S3_ENDPOINT_URL (default: None)
        - S3_BUCKET (default: None)
        - S3_INPUT_PREFIX (default: satellite_data)
        - S3_OUTPUT_PREFIX (default: nowcasts)
        - AWS_VERIFY_SSL (default: true; set to 0/false/no/off to disable)
        """
        return cls(
            endpoint_url=os.getenv("S3_ENDPOINT_URL", None),
            bucket=os.getenv("S3_BUCKET", None),
            input_prefix=os.getenv("S3_INPUT_PREFIX", "satellite_data"),
            output_prefix=os.getenv("S3_OUTPUT_PREFIX", "nowcasts"),
            verify_ssl=os.getenv("AWS_VERIFY_SSL", "true").strip().lower()
            in {"1", "true", "yes", "on"},
        )


@dataclass
class NowcastConfig:
    """Application configuration from environment variables."""

    nowcast_directory: str
    ens_members: int
    alpha: float
    beta: float
    past_steps: int
    future_steps: int
    input_data_availability_delay_minutes: int
    input_data_frequency_minutes: int
    max_waiting_time_minutes: int
    satellite_data_directory: str
    max_clearsky_fallback_days: int
    min_solar_elevation_degrees: float
    ensemble_statistics: list[str]

    @classmethod
    def from_env(
        cls,
        ensemble_members: int = 1,
        overrides: Mapping[str, object] | None = None,
    ) -> Self:
        """Load nowcast configuration from environment variables with defaults.

        Optional ``overrides`` can provide fallback values (for example, from YAML).
        Environment variables always take priority over ``overrides``.

        Reads the following environment variables:

        - NOWCAST_DIRECTORY (default: .)
        - ALPHA (default: 0.0 for ENS_MEMBERS=1, 9.23 for ENS_MEMBERS>1)
        - BETA (default: 0.0 for ENS_MEMBERS=1, 0.15 for ENS_MEMBERS>1)
        - PAST_STEPS (default: 4)
        - FUTURE_STEPS (default: 24)
        - INPUT_DATA_AVAILABILITY_DELAY_MINUTES (default: 24)
        - INPUT_DATA_FREQUENCY_MINUTES (default: 15)
        - MAX_WAITING_TIME_MINUTES (default: 27)
        - SATELLITE_DATA_DIRECTORY (default: .)
        - MAX_CLEARSKY_FALLBACK_DAYS (default: 3)
        - MIN_SOLAR_ELEVATION_DEGREES (default: 6)
        - ENSEMBLE_STATISTICS (default: median,mean,p10,p25,p75,p90)
        """

        overrides = overrides or {}

        def _get_value(name: str, default: str) -> str:
            env_value = os.getenv(name)
            if env_value is not None:
                return env_value

            override_value = overrides.get(name.lower())
            if override_value is None:
                return default

            if name == "ENSEMBLE_STATISTICS":
                if isinstance(override_value, str):
                    return override_value
                if isinstance(override_value, (list, tuple)):
                    return ",".join(str(item) for item in override_value)

            return str(override_value)

        ens_members = ensemble_members
        # Reference for default noise values:
        # A. Carpentieri, D. Folini, D. Nerini, S. Pulkkinen, M. Wild, A. Meyer,
        # "Intraday probabilistic forecasts of surface solar radiation with cloud
        # scale-dependent autoregressive advection,"
        # Applied Energy, Volume 351, 2023
        default_alpha = 0.0 if ens_members == 1 else 9.23
        default_beta = 0.0 if ens_members == 1 else 0.15
        statistics = _parse_ensemble_statistics(
            _get_value("ENSEMBLE_STATISTICS", DEFAULT_ENSEMBLE_STATISTICS)
        )

        return cls(
            nowcast_directory=_get_value("NOWCAST_DIRECTORY", "."),
            ens_members=ens_members,
            alpha=float(_get_value("ALPHA", str(default_alpha))),
            beta=float(_get_value("BETA", str(default_beta))),
            past_steps=int(_get_value("PAST_STEPS", "4")),
            future_steps=int(_get_value("FUTURE_STEPS", "24")),
            input_data_availability_delay_minutes=int(
                _get_value("INPUT_DATA_AVAILABILITY_DELAY_MINUTES", "24")
            ),
            input_data_frequency_minutes=int(
                _get_value("INPUT_DATA_FREQUENCY_MINUTES", "15")
            ),
            max_waiting_time_minutes=int(_get_value("MAX_WAITING_TIME_MINUTES", "27")),
            satellite_data_directory=_get_value("SATELLITE_DATA_DIRECTORY", "."),
            max_clearsky_fallback_days=int(_get_value("MAX_CLEARSKY_FALLBACK_DAYS", "3")),
            min_solar_elevation_degrees=float(
                _get_value("MIN_SOLAR_ELEVATION_DEGREES", "6")
            ),
            ensemble_statistics=statistics,
        )


def _parse_ensemble_statistics(raw_statistics: str) -> list[str]:
    """Parse and validate requested ensemble statistics from environment."""
    statistics = [
        token.strip().lower() for token in raw_statistics.split(",") if token.strip()
    ]
    if not statistics:
        raise ValueError("ENSEMBLE_STATISTICS must contain at least one statistic")

    invalid = [stat for stat in statistics if stat not in _ALLOWED_STATISTICS]
    if invalid:
        raise ValueError(
            "Invalid ENSEMBLE_STATISTICS value(s): "
            f"{', '.join(invalid)}. Allowed values are: "
            "median, mean, p10, p25, p75, p90, "
            "10th_percentile, 25th_percentile, 75th_percentile, 90th_percentile"
        )

    return [_STATISTIC_ALIASES.get(stat, stat) for stat in statistics]
