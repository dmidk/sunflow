#!/usr/bin/env python3
"""End-to-end tests driving the full nowcast pipeline in 'files' mode.

These tests write synthetic NetCDF input to a temporary directory, run
`run_nowcast` against it, and inspect the forecast file that comes out.
No network access, credentials or real satellite data are involved.

The happy-path pipeline is expensive, so each configuration is executed once
per module by a module-scoped fixture. Those fixtures hand out already-loaded
data, and tests only read from them. Tests that need to modify the input
directory build their own, function-scoped, copy.
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
OUTPUT_FILENAME_FORMAT = "SolarNowcast_{dataset_name}_{timestamp}.nc"

# The specMAGIC_MTG_pvlib configuration derives clearsky from pvlib instead of
# reading it from the previous day's files.
PVLIB_DATASET_CONFIG = {
    "nc_variable_names": {"sds": "sds", "sds_cs": "sds_cs"},
    "satellite": "Meteosat Third Generation (MTG)",
    "clearsky": {"method": "pvlib"},
}

# Midsummer noon keeps the sun well above the minimum elevation over Denmark.
NOWCAST_TIME = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
PAST_STEPS = 2
FUTURE_STEPS = 2
FREQUENCY_MINUTES = 15
CLEARSKY = 800.0

# Spans well beyond the requested domain so coverage validation passes.
LATITUDES = np.arange(48.0, 64.0, 0.4)
LONGITUDES = np.arange(2.0, 20.0, 0.4)


def cloud_field(shape, offset):
    """A diagonal cloud pattern displaced by `offset`, as a fraction of clearsky."""
    rows, cols = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return 0.5 + 0.35 * np.sin((rows + cols - 3.0 * offset) / 5.0)


def write_timestep_file(directory, time_step, offset=0.0):
    """Write one synthetic satellite file holding observed and clearsky irradiance."""
    shape = (len(LATITUDES), len(LONGITUDES))
    clearsky = np.full(shape, CLEARSKY)
    ds = xr.Dataset(
        {
            "sds": (["lat", "lon"], cloud_field(shape, offset) * clearsky),
            "sds_cs": (["lat", "lon"], clearsky),
        },
        coords={"lat": LATITUDES, "lon": LONGITUDES},
    )
    ds.to_netcdf(directory / f"DWD_{time_step.strftime('%Y%m%d%H%M')}.nc")


def populate_input_directory(directory, nowcast_time=NOWCAST_TIME):
    """Write every file a single nowcast run needs for `nowcast_time`."""
    directory.mkdir(parents=True, exist_ok=True)

    # Past observations, ending at the nowcast time.
    for i in range(PAST_STEPS):
        write_timestep_file(
            directory,
            nowcast_time - timedelta(minutes=FREQUENCY_MINUTES * i),
            offset=-i,
        )

    # Clearsky comes from the previous day: the analysis step plus one file per
    # forecast step.
    clearsky_t0 = nowcast_time - timedelta(days=1)
    for i in range(FUTURE_STEPS + 1):
        write_timestep_file(
            directory, clearsky_t0 + timedelta(minutes=FREQUENCY_MINUTES * i), offset=i
        )

    return directory


def build_config(input_directory, output_directory, **overrides):
    """Build a NowcastConfig for the end-to-end runs."""
    from sunflow.config import NowcastConfig

    settings = {
        "nowcast_directory": str(output_directory),
        "satellite_data_directory": str(input_directory),
        "ens_members": 1,
        "alpha": 0.0,
        "beta": 0.0,
        "past_steps": PAST_STEPS,
        "future_steps": FUTURE_STEPS,
        "input_data_availability_delay_minutes": 24,
        "input_data_frequency_minutes": FREQUENCY_MINUTES,
        "max_waiting_time_minutes": 27,
        "max_clearsky_fallback_days": 3,
        "min_solar_elevation_degrees": 6.0,
        "ensemble_statistics": ["median"],
        "input_filename_format": INPUT_FILENAME_FORMAT,
        "output_filename_format": OUTPUT_FILENAME_FORMAT,
    }
    settings.update(overrides)
    return NowcastConfig(**settings)


def execute(config, time_step=NOWCAST_TIME, full_ensemble=False):
    return run_nowcast(
        time_step,
        "files",
        DATASET_CONFIG,
        DOMAIN,
        DOMAIN,
        "DWD",
        "NW_EUROPE",
        config,
        S3Config.from_env(),
        full_ensemble=full_ensemble,
        custom_time=True,
    )


class PipelineRun:
    """The outcome of one pipeline execution, with the output already read."""

    def __init__(self, result, config, dataset):
        self.result = result
        self.config = config
        self.dataset = dataset

    @property
    def forecast(self):
        return self.dataset["GHI_probabilistic_advection"]


def run_pipeline(tmp_path_factory, name, full_ensemble=False, **config_overrides):
    """Run the pipeline once in an isolated directory and read the output."""
    base = tmp_path_factory.mktemp(name)
    input_directory = populate_input_directory(base / "satellite_data")
    output_directory = base / "nowcasts"
    output_directory.mkdir()

    config = build_config(input_directory, output_directory, **config_overrides)
    result = execute(config, full_ensemble=full_ensemble)

    with xr.open_dataset(Path(config.nowcast_directory) / result.filename) as ds:
        dataset = ds.load()

    return PipelineRun(result, config, dataset)


@pytest.fixture(scope="module")
def deterministic_run(tmp_path_factory):
    return run_pipeline(tmp_path_factory, "deterministic")


@pytest.fixture(scope="module")
def statistics_run(tmp_path_factory):
    return run_pipeline(
        tmp_path_factory,
        "statistics",
        ens_members=4,
        alpha=9.23,
        beta=0.15,
        ensemble_statistics=["median", "p90"],
    )


@pytest.fixture(scope="module")
def full_ensemble_run(tmp_path_factory):
    return run_pipeline(
        tmp_path_factory,
        "full_ensemble",
        full_ensemble=True,
        ens_members=4,
        alpha=9.23,
        beta=0.15,
    )


@pytest.fixture
def input_directory(tmp_path):
    """A writable input directory for tests that modify or remove files."""
    return populate_input_directory(tmp_path / "satellite_data")


@pytest.fixture
def mutable_config(tmp_path, input_directory):
    output_directory = tmp_path / "nowcasts"
    output_directory.mkdir()

    def _make(**overrides):
        return build_config(input_directory, output_directory, **overrides)

    return _make


class TestDeterministicRun:
    def test_the_run_reports_success(self, deterministic_run):
        assert deterministic_run.result.status == RunStatus.SUCCESS
        assert deterministic_run.result.time_step == NOWCAST_TIME

    def test_the_output_filename_follows_the_configured_format(self, deterministic_run):
        assert deterministic_run.result.filename == "SolarNowcast_DWD_202606211200.nc"

    def test_the_output_file_is_written_to_the_nowcast_directory(self, deterministic_run):
        output_path = (
            Path(deterministic_run.config.nowcast_directory)
            / deterministic_run.result.filename
        )

        assert output_path.is_file()

    def test_the_forecast_starts_at_the_nowcast_time(self, deterministic_run):
        assert deterministic_run.dataset.time.values[0] == np.datetime64(
            NOWCAST_TIME.replace(tzinfo=None)
        )

    def test_the_analysis_step_is_followed_by_every_forecast_step(
        self, deterministic_run
    ):
        assert deterministic_run.dataset.sizes["time"] == FUTURE_STEPS + 1

    def test_forecast_steps_are_evenly_spaced_at_the_input_frequency(
        self, deterministic_run
    ):
        times = deterministic_run.dataset.time.values
        spacing = np.diff(times).astype("timedelta64[m]").astype(int)

        assert list(spacing) == [FREQUENCY_MINUTES] * FUTURE_STEPS

    def test_a_deterministic_run_has_a_single_ensemble_member(self, deterministic_run):
        assert deterministic_run.dataset.sizes["ensemble"] == 1

    def test_the_output_holds_one_forecast_variable(self, deterministic_run):
        assert set(deterministic_run.dataset.data_vars) == {"GHI_probabilistic_advection"}

    def test_the_forecast_variable_keeps_its_units_and_description(
        self, deterministic_run
    ):
        attributes = deterministic_run.forecast.attrs

        assert attributes["units"] == "W m-2"
        assert attributes["long_name"] == "Surface downwelling solar radiation"
        assert attributes["description"] == "Probabilistic advection solar forecast"

    def test_global_attributes_describe_the_run(self, deterministic_run):
        attributes = deterministic_run.dataset.attrs

        assert attributes["satellite_data_source"] == "DWD"
        assert attributes["satellite"] == "Meteosat Second Generation (MSG)"
        assert attributes["output_mode"] == "Deterministic"
        assert attributes["model_version"]

    def test_dimensions_are_ordered_ensemble_time_lat_lon(self, deterministic_run):
        assert deterministic_run.forecast.dims == ("ensemble", "time", "lat", "lon")


class TestSpatialCropping:
    def test_no_grid_cell_inside_the_domain_is_dropped(self, deterministic_run):
        expected_inside = LATITUDES[(LATITUDES >= 50.0) & (LATITUDES <= 62.0)]

        assert set(expected_inside) <= set(deterministic_run.dataset.lat.values)

    def test_output_cells_stay_within_one_grid_step_of_the_domain(
        self, deterministic_run
    ):
        step = 0.4
        latitudes = deterministic_run.dataset.lat.values
        longitudes = deterministic_run.dataset.lon.values

        assert latitudes.min() >= 50.0 - step
        assert latitudes.max() <= 62.0 + step
        assert longitudes.min() >= 4.0 - step
        assert longitudes.max() <= 18.0 + step

    def test_output_coordinates_come_from_the_input_grid(self, deterministic_run):
        assert set(deterministic_run.dataset.lat.values) <= set(LATITUDES)
        assert set(deterministic_run.dataset.lon.values) <= set(LONGITUDES)

    def test_the_grid_spacing_is_unchanged(self, deterministic_run):
        spacing = np.diff(deterministic_run.dataset.lat.values)

        np.testing.assert_allclose(spacing, 0.4)

    def test_coordinates_are_strictly_increasing(self, deterministic_run):
        latitudes = deterministic_run.dataset.lat.values
        longitudes = deterministic_run.dataset.lon.values

        assert (np.diff(latitudes) > 0).all()
        assert (np.diff(longitudes) > 0).all()


class TestForecastValues:
    def test_the_analysis_step_is_the_latest_observation_times_clearsky(
        self, deterministic_run
    ):
        # Computed from the synthetic input directly, independently of the
        # pipeline: the cloud field at offset 0 scaled by the clearsky value.
        dataset = deterministic_run.dataset
        lat_index = np.isin(LATITUDES, dataset.lat.values)
        lon_index = np.isin(LONGITUDES, dataset.lon.values)
        expected = (
            cloud_field((len(LATITUDES), len(LONGITUDES)), 0.0)[lat_index][:, lon_index]
            * CLEARSKY
        )

        analysis = dataset["GHI_probabilistic_advection"].isel(ensemble=0, time=0).values

        np.testing.assert_allclose(analysis, expected, rtol=1e-5)

    def test_irradiance_values_are_physically_plausible(self, deterministic_run):
        values = deterministic_run.forecast.values
        finite = values[np.isfinite(values)]

        assert finite.size > 0
        assert finite.min() >= 0.0
        # Ratios stay below 0.85 and clearsky is a flat 800 W/m2.
        assert finite.max() <= CLEARSKY

    def test_no_infinite_values_are_written(self, deterministic_run):
        assert not np.isinf(deterministic_run.forecast.values).any()

    def test_the_analysis_step_is_fully_defined(self, deterministic_run):
        # Nothing is advected yet at t=0, so every cell must have a value.
        analysis = deterministic_run.forecast.isel(ensemble=0, time=0).values

        assert not np.isnan(analysis).any()

    def test_the_run_is_reproducible(self, tmp_path_factory, deterministic_run):
        repeat = run_pipeline(tmp_path_factory, "deterministic_repeat")

        np.testing.assert_array_equal(
            repeat.forecast.values, deterministic_run.forecast.values
        )


class TestEnsembleStatisticsRun:
    def test_one_variable_is_written_per_requested_statistic(self, statistics_run):
        assert set(statistics_run.dataset.data_vars) == {
            "median_GHI_probabilistic_advection",
            "p90_GHI_probabilistic_advection",
        }

    def test_the_output_mode_is_recorded(self, statistics_run):
        assert statistics_run.dataset.attrs["output_mode"] == "Ensemble statistics"

    def test_the_requested_statistics_are_listed(self, statistics_run):
        assert statistics_run.dataset.attrs["statistics"] == "median,p90"

    def test_statistics_collapse_the_ensemble_dimension(self, statistics_run):
        assert statistics_run.dataset.sizes["ensemble"] == 1

    def test_statistics_keep_the_full_time_axis(self, statistics_run):
        assert statistics_run.dataset.sizes["time"] == FUTURE_STEPS + 1

    def test_the_upper_percentile_is_never_below_the_median(self, statistics_run):
        median = statistics_run.dataset["median_GHI_probabilistic_advection"].values
        p90 = statistics_run.dataset["p90_GHI_probabilistic_advection"].values
        comparable = np.isfinite(median) & np.isfinite(p90)

        assert (p90[comparable] >= median[comparable]).all()

    def test_statistics_carry_the_forecast_units(self, statistics_run):
        variable = statistics_run.dataset["median_GHI_probabilistic_advection"]

        assert variable.attrs["units"] == "W m-2"


class TestFullEnsembleRun:
    def test_every_member_is_written(self, full_ensemble_run):
        assert full_ensemble_run.dataset.sizes["ensemble"] == 4

    def test_members_share_a_single_forecast_variable(self, full_ensemble_run):
        assert set(full_ensemble_run.dataset.data_vars) == {"GHI_probabilistic_advection"}

    def test_the_output_mode_is_recorded(self, full_ensemble_run):
        assert full_ensemble_run.dataset.attrs["output_mode"] == "Full ensemble"

    def test_the_analysis_step_is_shared_by_every_member(self, full_ensemble_run):
        analysis = full_ensemble_run.forecast.isel(time=0).values

        for member in range(1, analysis.shape[0]):
            np.testing.assert_allclose(analysis[member], analysis[0])

    def test_members_diverge_once_perturbations_are_applied(self, full_ensemble_run):
        members = full_ensemble_run.forecast.isel(time=-1).values

        assert not np.allclose(members[0], members[1], equal_nan=True)

    def test_the_ensemble_coordinate_counts_from_zero(self, full_ensemble_run):
        np.testing.assert_array_equal(
            full_ensemble_run.dataset.ensemble.values, np.arange(4)
        )

    def test_the_ensemble_is_reproducible(self, tmp_path_factory, full_ensemble_run):
        repeat = run_pipeline(
            tmp_path_factory,
            "full_ensemble_repeat",
            full_ensemble=True,
            ens_members=4,
            alpha=9.23,
            beta=0.15,
        )

        np.testing.assert_array_equal(
            repeat.forecast.values, full_ensemble_run.forecast.values
        )


@pytest.fixture(scope="module")
def pvlib_run(tmp_path_factory):
    """Run the pipeline once with pvlib-generated clearsky."""
    base = tmp_path_factory.mktemp("pvlib")
    input_directory = base / "satellite_data"
    input_directory.mkdir()
    output_directory = base / "nowcasts"
    output_directory.mkdir()

    # Only observed irradiance is stored; clearsky is computed, so writing an
    # sds_cs variable here would collide with the generated one.
    for i in range(PAST_STEPS):
        time_step = NOWCAST_TIME - timedelta(minutes=FREQUENCY_MINUTES * i)
        shape = (len(LATITUDES), len(LONGITUDES))
        xr.Dataset(
            {"sds": (["lat", "lon"], cloud_field(shape, -i) * CLEARSKY)},
            coords={"lat": LATITUDES, "lon": LONGITUDES},
        ).to_netcdf(input_directory / f"DWD_{time_step.strftime('%Y%m%d%H%M')}.nc")

    config = build_config(input_directory, output_directory)
    result = run_nowcast(
        NOWCAST_TIME,
        "files",
        PVLIB_DATASET_CONFIG,
        DOMAIN,
        DOMAIN,
        "DWD",
        "NW_EUROPE",
        config,
        S3Config.from_env(),
        custom_time=True,
    )

    with xr.open_dataset(Path(config.nowcast_directory) / result.filename) as ds:
        dataset = ds.load()

    return PipelineRun(result, config, dataset)


class TestPvlibClearskyRun:
    """The pvlib clearsky method needs no previous-day files at all."""

    def test_the_run_succeeds_without_any_clearsky_files(self, pvlib_run):
        assert pvlib_run.result.status == RunStatus.SUCCESS

    def test_the_forecast_has_the_expected_shape(self, pvlib_run):
        assert pvlib_run.dataset.sizes["time"] == FUTURE_STEPS + 1
        assert pvlib_run.dataset.sizes["ensemble"] == 1

    def test_the_satellite_attribute_follows_the_dataset_config(self, pvlib_run):
        assert pvlib_run.dataset.attrs["satellite"] == "Meteosat Third Generation (MTG)"

    def test_irradiance_values_are_physically_plausible(self, pvlib_run):
        values = pvlib_run.forecast.values
        finite = values[np.isfinite(values)]

        assert finite.size > 0
        assert finite.min() >= 0.0
        # Midsummer clearsky GHI over Denmark stays well under the solar constant.
        assert finite.max() <= 1400.0

    def test_no_infinite_values_are_written(self, pvlib_run):
        assert not np.isinf(pvlib_run.forecast.values).any()


class TestSkipAndFailureBehaviour:
    def test_a_run_is_skipped_when_the_sun_is_too_low(
        self, input_directory, mutable_config
    ):
        night = datetime(2026, 12, 21, 0, 0, tzinfo=timezone.utc)
        write_timestep_file(input_directory, night)

        result = execute(mutable_config(), time_step=night)

        assert result.status == RunStatus.SKIPPED
        assert result.reason == "sun too low"
        assert result.filename is None

    def test_a_skipped_run_writes_no_output(self, input_directory, mutable_config):
        night = datetime(2026, 12, 21, 0, 0, tzinfo=timezone.utc)
        write_timestep_file(input_directory, night)
        config = mutable_config()

        execute(config, time_step=night)

        assert list(Path(config.nowcast_directory).iterdir()) == []

    def test_the_elevation_threshold_is_configurable(
        self, input_directory, mutable_config
    ):
        # The same timestep runs to completion once the threshold is lowered.
        night = datetime(2026, 12, 21, 0, 0, tzinfo=timezone.utc)
        populate_input_directory(input_directory, nowcast_time=night)

        result = execute(
            mutable_config(min_solar_elevation_degrees=-90.0), time_step=night
        )

        assert result.status == RunStatus.SUCCESS
        assert result.filename == "SolarNowcast_DWD_202612210000.nc"

    def test_missing_input_for_the_nowcast_time_is_reported(self, mutable_config):
        from sunflow.validation import DataNotAvailableError

        unavailable = NOWCAST_TIME + timedelta(days=1)

        with pytest.raises(DataNotAvailableError):
            execute(mutable_config(), time_step=unavailable)

    def test_a_missing_past_observation_aborts_the_run(
        self, input_directory, mutable_config
    ):
        earliest_past = NOWCAST_TIME - timedelta(
            minutes=FREQUENCY_MINUTES * (PAST_STEPS - 1)
        )
        (input_directory / f"DWD_{earliest_past.strftime('%Y%m%d%H%M')}.nc").unlink()

        with pytest.raises(FileNotFoundError, match="Past data file not found"):
            execute(mutable_config())

    def test_missing_clearsky_data_aborts_the_run(self, input_directory, mutable_config):
        clearsky_t0 = NOWCAST_TIME - timedelta(days=1)
        for i in range(FUTURE_STEPS + 1):
            time_step = clearsky_t0 + timedelta(minutes=FREQUENCY_MINUTES * i)
            (input_directory / f"DWD_{time_step.strftime('%Y%m%d%H%M')}.nc").unlink()

        with pytest.raises(RuntimeError, match="Missing clearsky data"):
            execute(mutable_config())

    def test_a_single_missing_clearsky_step_aborts_the_run(
        self, input_directory, mutable_config
    ):
        # The fallback searches earlier days, not earlier timesteps, so a
        # clearsky file missing on every candidate day is fatal.
        missing = NOWCAST_TIME - timedelta(days=1) + timedelta(minutes=FREQUENCY_MINUTES)
        (input_directory / f"DWD_{missing.strftime('%Y%m%d%H%M')}.nc").unlink()

        with pytest.raises(RuntimeError, match="Missing clearsky data"):
            execute(mutable_config())

    def test_an_input_file_that_is_not_netcdf_is_reported(
        self, input_directory, mutable_config
    ):
        path = input_directory / f"DWD_{NOWCAST_TIME.strftime('%Y%m%d%H%M')}.nc"
        path.write_bytes(b"not a NetCDF file")

        with pytest.raises(ValueError, match="did not find a match"):
            execute(mutable_config())

    def test_a_domain_larger_than_the_input_grid_is_rejected(self, mutable_config):
        config = mutable_config()

        with pytest.raises(RuntimeError, match="does not cover requested"):
            run_nowcast(
                NOWCAST_TIME,
                "files",
                DATASET_CONFIG,
                "-30,30,40,75",
                "-30,30,40,75",
                "DWD",
                "NW_EUROPE",
                config,
                S3Config.from_env(),
                custom_time=True,
            )
