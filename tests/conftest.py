#!/usr/bin/env python3
"""Fixtures shared across the sunflow test suite."""
import numpy as np
import pytest
import xarray as xr

from sunflow.config import NowcastConfig

_NOWCAST_CONFIG_DEFAULTS = {
    "nowcast_directory": ".",
    "ens_members": 1,
    "alpha": 0.0,
    "beta": 0.0,
    "past_steps": 4,
    "future_steps": 24,
    "input_data_availability_delay_minutes": 24,
    "input_data_frequency_minutes": 15,
    "max_waiting_time_minutes": 27,
    "satellite_data_directory": ".",
    "max_clearsky_fallback_days": 3,
    "min_solar_elevation_degrees": 6.0,
    "ensemble_statistics": ["median"],
    "input_filename_format": "{dataset_name}_{timestamp}.nc",
    "output_filename_format": "SolarNowcast_{dataset_name}_{timestamp}.nc",
}


@pytest.fixture
def make_nowcast_config():
    """Build a NowcastConfig from explicit values, bypassing the environment."""

    def _make(**overrides) -> NowcastConfig:
        return NowcastConfig(**{**_NOWCAST_CONFIG_DEFAULTS, **overrides})

    return _make


@pytest.fixture
def make_gridded_dataset():
    """Build a minimal satellite dataset over a lat/lon grid."""

    def _make(
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        lat_name: str = "lat",
        lon_name: str = "lon",
        sds: np.ndarray | None = None,
        sds_cs: np.ndarray | None = None,
    ) -> xr.Dataset:
        shape = (len(latitudes), len(longitudes))
        return xr.Dataset(
            {
                "sds": (
                    [lat_name, lon_name],
                    np.full(shape, 400.0) if sds is None else sds,
                ),
                "sds_cs": (
                    [lat_name, lon_name],
                    np.full(shape, 800.0) if sds_cs is None else sds_cs,
                ),
            },
            coords={lat_name: latitudes, lon_name: longitudes},
        )

    return _make


@pytest.fixture
def write_input_file(make_gridded_dataset):
    """Write a minimal satellite input file to disk, creating parent directories."""

    def _write(path, latitudes, longitudes, sds=None, sds_cs=None):
        ds = make_gridded_dataset(latitudes, longitudes, sds=sds, sds_cs=sds_cs)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path)
        return path

    return _write


@pytest.fixture
def make_clearsky_dataset():
    """Build a clearsky dataset covering the given time steps."""

    def _make(time_steps, shape=(3, 4), variable="sds_cs") -> xr.Dataset:
        return xr.Dataset(
            {
                variable: (
                    ["time", "lat", "lon"],
                    np.full((len(time_steps), *shape), 800.0),
                )
            },
            coords={
                "time": [ts.replace(tzinfo=None) for ts in time_steps],
                "lat": np.arange(shape[0], dtype=float),
                "lon": np.arange(shape[1], dtype=float),
            },
        )

    return _make
