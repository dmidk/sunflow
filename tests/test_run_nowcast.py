#!/usr/bin/env python3
"""End-to-end tests driving the full nowcast pipeline in 'files' mode.

These tests write synthetic NetCDF input to a temporary directory, run
`run_nowcast` against it, and inspect the forecast file that comes out.
No network access, credentials or real satellite data are involved.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sunflow.config import S3Config
from sunflow.main import RunStatus, run_nowcast

DOMAIN = "4,50,18,62"
DATASET_CONFIG = {
    "nc_variable_names": {"sds": "sds", "sds_cs": "sds_cs"},
    "satellite": "Meteosat Second Generation (MSG)",
}
INPUT_FILENAME_FORMAT = "{dataset_name}_{timestamp}.nc"

# Midsummer noon keeps the sun well above the minimum elevation over Denmark.
NOWCAST_TIME = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
PAST_STEPS = 2
FUTURE_STEPS = 2
FREQUENCY_MINUTES = 15


def cloud_field(shape, offset):
    """A diagonal cloud pattern displaced by `offset`, as a fraction of clearsky."""
    rows, cols = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return 0.5 + 0.35 * np.sin((rows + cols - 3.0 * offset) / 5.0)


def write_timestep_file(directory, time_step, latitudes, longitudes, offset):
    """Write one synthetic satellite file holding observed and clearsky irradiance."""
    shape = (len(latitudes), len(longitudes))
    clearsky = np.full(shape, 800.0)
    ds = xr.Dataset(
        {
            "sds": (["lat", "lon"], cloud_field(shape, offset) * clearsky),
            "sds_cs": (["lat", "lon"], clearsky),
        },
        coords={"lat": latitudes, "lon": longitudes},
    )
    filename = f"DWD_{time_step.strftime('%Y%m%d%H%M')}.nc"
    ds.to_netcdf(directory / filename)


@pytest.fixture
def grid():
    # Spans well beyond the requested domain so coverage validation passes.
    return np.arange(48.0, 64.0, 0.4), np.arange(2.0, 20.0, 0.4)


@pytest.fixture
def satellite_data(tmp_path, grid):
    """Populate a directory with every file a single nowcast run needs."""
    latitudes, longitudes = grid
    directory = tmp_path / "satellite_data"
    directory.mkdir()

    # Past observations, ending at the nowcast time.
    for i in range(PAST_STEPS):
        time_step = NOWCAST_TIME - timedelta(minutes=FREQUENCY_MINUTES * i)
        write_timestep_file(directory, time_step, latitudes, longitudes, offset=-i)

    # Clearsky is taken from the previous day: the analysis step plus one file
    # per forecast step.
    clearsky_t0 = NOWCAST_TIME - timedelta(days=1)
    write_timestep_file(directory, clearsky_t0, latitudes, longitudes, offset=0)
    for i in range(1, FUTURE_STEPS + 1):
        time_step = clearsky_t0 + timedelta(minutes=FREQUENCY_MINUTES * i)
        write_timestep_file(directory, time_step, latitudes, longitudes, offset=i)

    return directory


@pytest.fixture
def nowcast_config(tmp_path, satellite_data, make_nowcast_config):
    output_directory = tmp_path / "nowcasts"
    output_directory.mkdir()

    def _make(**overrides):
        return make_nowcast_config(
            nowcast_directory=str(output_directory),
            satellite_data_directory=str(satellite_data),
            past_steps=PAST_STEPS,
            future_steps=FUTURE_STEPS,
            input_data_frequency_minutes=FREQUENCY_MINUTES,
            input_filename_format=INPUT_FILENAME_FORMAT,
            output_filename_format="SolarNowcast_{dataset_name}_{timestamp}.nc",
            **overrides,
        )

    return _make


def run(nowcast_config, time_step=NOWCAST_TIME, full_ensemble=False):
    return run_nowcast(
        time_step,
        "files",
        DATASET_CONFIG,
        DOMAIN,
        DOMAIN,
        "DWD",
        "NW_EUROPE",
        nowcast_config,
        S3Config.from_env(),
        full_ensemble=full_ensemble,
        custom_time=True,
    )


class TestDeterministicRun:
    def test_run_reports_success_and_names_the_output_file(self, nowcast_config):
        result = run(nowcast_config())

        assert result.status == RunStatus.SUCCESS
        assert result.time_step == NOWCAST_TIME
        assert result.filename == "SolarNowcast_DWD_202606211200.nc"

    def test_forecast_file_is_written(self, nowcast_config):
        config = nowcast_config()

        result = run(config)

        assert (Path(config.nowcast_directory) / result.filename).is_file()

    def test_output_has_the_analysis_step_plus_every_forecast_step(
        self, nowcast_config
    ):
        config = nowcast_config()

        result = run(config)

        with xr.open_dataset(f"{config.nowcast_directory}/{result.filename}") as ds:
            assert ds.sizes["time"] == FUTURE_STEPS + 1
            assert ds.sizes["ensemble"] == 1

    def test_output_is_cropped_to_the_nowcast_domain(self, nowcast_config):
        config = nowcast_config()

        result = run(config)

        with xr.open_dataset(f"{config.nowcast_directory}/{result.filename}") as ds:
            assert ds.lat.min() >= 49.0 and ds.lat.max() <= 63.0
            assert ds.lon.min() >= 3.0 and ds.lon.max() <= 19.0

    def test_output_holds_plausible_irradiance_values(self, nowcast_config):
        config = nowcast_config()

        result = run(config)

        with xr.open_dataset(f"{config.nowcast_directory}/{result.filename}") as ds:
            values = ds["GHI_probabilistic_advection"].values
            finite = values[np.isfinite(values)]
            assert finite.size > 0
            # Ratios lie in roughly 0.15-0.85 and clearsky is a flat 800 W/m2.
            assert finite.min() >= 0.0
            assert finite.max() <= 800.0

    def test_analysis_step_equals_the_latest_observation_times_clearsky(
        self, nowcast_config, grid
    ):
        config = nowcast_config()

        result = run(config)

        with xr.open_dataset(f"{config.nowcast_directory}/{result.filename}") as ds:
            analysis = ds["GHI_probabilistic_advection"].isel(ensemble=0, time=0).values

        latitudes, longitudes = grid
        with xr.open_dataset(
            f"{config.satellite_data_directory}/DWD_202606211200.nc"
        ) as observed:
            expected_ratio = (observed["sds"] / observed["sds_cs"]).sel(
                lat=slice(ds.lat.min(), ds.lat.max()),
                lon=slice(ds.lon.min(), ds.lon.max()),
            )

        np.testing.assert_allclose(
            analysis, expected_ratio.values * 800.0, rtol=1e-5
        )

    def test_global_attributes_describe_the_run(self, nowcast_config):
        config = nowcast_config()

        result = run(config)

        with xr.open_dataset(f"{config.nowcast_directory}/{result.filename}") as ds:
            assert ds.attrs["satellite_data_source"] == "DWD"
            assert ds.attrs["satellite"] == "Meteosat Second Generation (MSG)"
            assert ds.attrs["output_mode"] == "Deterministic"


class TestEnsembleRun:
    def test_ensemble_statistics_are_written_by_default(self, nowcast_config):
        config = nowcast_config(
            ens_members=4,
            alpha=9.23,
            beta=0.15,
            ensemble_statistics=["median", "p90"],
        )

        result = run(config)

        with xr.open_dataset(f"{config.nowcast_directory}/{result.filename}") as ds:
            assert set(ds.data_vars) == {
                "median_GHI_probabilistic_advection",
                "p90_GHI_probabilistic_advection",
            }
            assert ds.attrs["output_mode"] == "Ensemble statistics"
            assert ds.sizes["ensemble"] == 1

    def test_full_ensemble_keeps_every_member(self, nowcast_config):
        config = nowcast_config(ens_members=4, alpha=9.23, beta=0.15)

        result = run(config, full_ensemble=True)

        with xr.open_dataset(f"{config.nowcast_directory}/{result.filename}") as ds:
            assert ds.sizes["ensemble"] == 4
            assert ds.attrs["output_mode"] == "Full ensemble"
            assert set(ds.data_vars) == {"GHI_probabilistic_advection"}

    def test_ensemble_members_differ_once_noise_is_enabled(self, nowcast_config):
        config = nowcast_config(ens_members=4, alpha=9.23, beta=0.15)

        result = run(config, full_ensemble=True)

        with xr.open_dataset(f"{config.nowcast_directory}/{result.filename}") as ds:
            # The analysis step is shared, so compare the last forecast step.
            members = ds["GHI_probabilistic_advection"].isel(time=-1).values

        assert not np.allclose(members[0], members[1], equal_nan=True)


class TestSkipAndFailureBehaviour:
    def test_a_run_is_skipped_when_the_sun_is_too_low(self, tmp_path, grid, nowcast_config):
        latitudes, longitudes = grid
        night = datetime(2026, 12, 21, 0, 0, tzinfo=timezone.utc)
        config = nowcast_config()
        write_timestep_file(
            tmp_path / "satellite_data", night, latitudes, longitudes, offset=0
        )

        result = run(config, time_step=night)

        assert result.status == RunStatus.SKIPPED
        assert result.reason == "sun too low"
        assert result.filename is None

    def test_missing_input_for_the_nowcast_time_is_reported(self, nowcast_config):
        from sunflow.validation import DataNotAvailableError

        config = nowcast_config()
        unavailable = NOWCAST_TIME + timedelta(days=1)

        with pytest.raises(DataNotAvailableError):
            run(config, time_step=unavailable)

    def test_missing_clearsky_data_aborts_the_run(self, satellite_data, nowcast_config):
        config = nowcast_config()
        clearsky_t0 = NOWCAST_TIME - timedelta(days=1)
        for i in range(FUTURE_STEPS + 1):
            path = satellite_data / (
                f"DWD_{(clearsky_t0 + timedelta(minutes=FREQUENCY_MINUTES * i))
                      .strftime('%Y%m%d%H%M')}.nc"
            )
            path.unlink()

        with pytest.raises(RuntimeError, match="Missing clearsky data"):
            run(config)
