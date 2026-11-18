#!/usr/bin/env python3
"""Fixtures shared across the sunflow test suite.

Everything here is offline: datasets are synthesised in memory or written to
pytest's temporary directories, and the S3 double replaces fsspec entirely.
"""
from io import BytesIO

import numpy as np
import pytest
import xarray as xr

from sunflow.config import NowcastConfig

# Default clear-sky irradiance used by the synthetic datasets, in W m-2.
CLEARSKY = 800.0

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


@pytest.fixture(scope="session")
def make_nowcast_config():
    """Build a NowcastConfig from explicit values, bypassing the environment."""

    def _make(**overrides) -> NowcastConfig:
        return NowcastConfig(**{**_NOWCAST_CONFIG_DEFAULTS, **overrides})

    return _make


@pytest.fixture(scope="session")
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
                    np.full(shape, CLEARSKY / 2) if sds is None else sds,
                ),
                "sds_cs": (
                    [lat_name, lon_name],
                    np.full(shape, CLEARSKY) if sds_cs is None else sds_cs,
                ),
            },
            coords={lat_name: latitudes, lon_name: longitudes},
        )

    return _make


@pytest.fixture(scope="session")
def write_input_file(make_gridded_dataset):
    """Write a minimal satellite input file to disk, creating parent directories."""

    def _write(path, latitudes, longitudes, sds=None, sds_cs=None):
        ds = make_gridded_dataset(latitudes, longitudes, sds=sds, sds_cs=sds_cs)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path)
        return path

    return _write


@pytest.fixture(scope="session")
def make_clearsky_dataset():
    """Build a clearsky dataset covering the given time steps."""

    def _make(time_steps, shape=(3, 4), variable="sds_cs", values=None) -> xr.Dataset:
        if values is None:
            values = np.full((len(time_steps), *shape), CLEARSKY)
        return xr.Dataset(
            {variable: (["time", "lat", "lon"], values)},
            coords={
                "time": [ts.replace(tzinfo=None) for ts in time_steps],
                "lat": np.arange(shape[0], dtype=float),
                "lon": np.arange(shape[1], dtype=float),
            },
        )

    return _make


@pytest.fixture(scope="session")
def to_netcdf_bytes():
    """Serialise a dataset to in-memory NetCDF bytes, as stored in S3."""

    def _serialise(dataset: xr.Dataset) -> bytes:
        buffer = BytesIO()
        dataset.to_netcdf(buffer, engine="h5netcdf")
        return buffer.getvalue()

    return _serialise


class FakeS3FileSystem:
    """In-memory stand-in for the fsspec S3 filesystem used by `data_io`.

    Only the handful of operations sunflow actually calls are implemented:
    `exists` and `open` for reading and writing. Files live in a dict keyed by
    the full ``s3://bucket/key`` path, so tests can seed inputs and inspect
    whatever was uploaded.
    """

    def __init__(self, files=None, client_kwargs=None):
        self.files: dict[str, bytes] = dict(files or {})
        self.client_kwargs = client_kwargs or {}
        self.open_failures: dict[str, Exception] = {}

    def exists(self, path: str) -> bool:
        return path in self.files

    def open(self, path: str, mode: str = "rb"):
        if path in self.open_failures:
            raise self.open_failures[path]

        if "w" in mode:
            return _S3WriteBuffer(self, path)

        if path not in self.files:
            raise FileNotFoundError(f"No such object: {path}")
        return BytesIO(self.files[path])


class _S3WriteBuffer(BytesIO):
    """Writable buffer that stores its contents on the filesystem when closed."""

    def __init__(self, filesystem: FakeS3FileSystem, path: str):
        super().__init__()
        self._filesystem = filesystem
        self._path = path

    def close(self):
        if not self.closed:
            self._filesystem.files[self._path] = self.getvalue()
        super().close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


@pytest.fixture
def fake_s3(monkeypatch):
    """Replace the fsspec S3 filesystem with an in-memory double.

    Returns the FakeS3FileSystem instance that `data_io` will receive, so a
    test can seed objects before acting and inspect uploads afterwards.
    """
    filesystem = FakeS3FileSystem()
    calls = []

    def fake_filesystem(protocol, **kwargs):
        calls.append((protocol, kwargs))
        filesystem.client_kwargs = kwargs.get("client_kwargs", {})
        return filesystem

    monkeypatch.setattr("sunflow.data_io.fsspec.filesystem", fake_filesystem)
    filesystem.calls = calls
    return filesystem
