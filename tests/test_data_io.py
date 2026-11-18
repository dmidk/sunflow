#!/usr/bin/env python3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr

from sunflow.config import S3Config
from sunflow.data_io import (
    check_current_data_existence_file,
    check_current_data_existence_s3,
    fetch_clearsky_with_fallback,
    generate_filename,
    load_data_from_files,
    load_data_from_s3,
    save_forecast,
    save_forecast_to_s3,
)

TIME_STEP = datetime(2026, 3, 9, 12, 30, tzinfo=timezone.utc)
CLEARSKY = 800.0


class TestGenerateFilename:
    @pytest.mark.parametrize(
        "template,expected",
        [
            ("{dataset_name}_{timestamp}.nc", "DWD_202603091230.nc"),
            ("NetCDF4_sds_{pds_timestamp}.nc", "NetCDF4_sds_2026-03-09T12_30_00Z.nc"),
            (
                "SolarNowcast_{iso8601_compact_timestamp}.nc",
                "SolarNowcast_2026-03-09T1230Z.nc",
            ),
            ("GHI/GHI{year}{month}{day}{hour}{minute}.nc", "GHI/GHI202603091230.nc"),
            ("{year}-{month}-{day}T{hour}:{minute}:{second}", "2026-03-09T12:30:00"),
        ],
    )
    def test_template_variables_are_substituted(self, template, expected):
        assert generate_filename(TIME_STEP, "DWD", template) == expected

    def test_domain_satellite_name_is_available_when_provided(self):
        filename = generate_filename(
            TIME_STEP,
            "DWD",
            "{dataset_name}_{timestamp}_{domain_satellite_name}.nc",
            "NW_EUROPE_SATELLITE",
        )

        assert filename == "DWD_202603091230_NW_EUROPE_SATELLITE.nc"

    def test_domain_satellite_name_is_unavailable_for_output_filenames(self):
        with pytest.raises(KeyError, match="domain_satellite_name"):
            generate_filename(TIME_STEP, "DWD", "{domain_satellite_name}.nc")

    def test_unknown_template_variable_raises(self):
        with pytest.raises(KeyError, match="orbit"):
            generate_filename(TIME_STEP, "DWD", "{orbit}.nc")

    def test_template_without_variables_is_returned_unchanged(self):
        assert generate_filename(TIME_STEP, "DWD", "fixed.nc") == "fixed.nc"

    def test_single_digit_date_parts_are_zero_padded(self):
        time_step = datetime(2026, 3, 9, 4, 5, 6, tzinfo=timezone.utc)

        filename = generate_filename(
            time_step, "DWD", "{year}{month}{day}{hour}{minute}{second}"
        )

        assert filename == "20260309040506"

    def test_path_separators_produce_nested_paths(self):
        filename = generate_filename(
            TIME_STEP, "KNMI_MSG", "{year}/{month}/{day}/{dataset_name}_{timestamp}.nc"
        )

        assert filename == "2026/03/09/KNMI_MSG_202603091230.nc"


class TestCheckCurrentDataExistenceFile:
    def test_passes_when_the_file_exists(self, tmp_path, write_input_file):
        write_input_file(
            tmp_path / "DWD_202603091230_NW_EUROPE.nc",
            np.arange(45.0, 65.0),
            np.arange(-5.0, 15.0),
        )

        check_current_data_existence_file(
            TIME_STEP,
            "DWD",
            "NW_EUROPE",
            str(tmp_path),
            "{dataset_name}_{timestamp}_{domain_satellite_name}.nc",
        )

    def test_missing_file_raises_with_the_expected_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="DWD_202603091230.nc"):
            check_current_data_existence_file(
                TIME_STEP,
                "DWD",
                "NW_EUROPE",
                str(tmp_path),
                "{dataset_name}_{timestamp}.nc",
            )

    def test_domain_coverage_is_validated_when_required(self, tmp_path, write_input_file):
        write_input_file(
            tmp_path / "DWD_202603091230.nc",
            np.arange(50.0, 55.0),
            np.arange(0.0, 5.0),
        )

        with pytest.raises(RuntimeError, match="does not cover requested"):
            check_current_data_existence_file(
                TIME_STEP,
                "DWD",
                "NW_EUROPE",
                str(tmp_path),
                "{dataset_name}_{timestamp}.nc",
                required_domain="-10.75,47.25,20,63.5",
            )

    def test_sufficient_coverage_passes_the_domain_check(
        self, tmp_path, write_input_file
    ):
        write_input_file(
            tmp_path / "DWD_202603091230.nc",
            np.arange(40.0, 70.0),
            np.arange(-25.0, 35.0),
        )

        check_current_data_existence_file(
            TIME_STEP,
            "DWD",
            "NW_EUROPE",
            str(tmp_path),
            "{dataset_name}_{timestamp}.nc",
            required_domain="-10.75,47.25,20,63.5",
        )

    def test_nested_filename_formats_are_resolved(self, tmp_path, write_input_file):
        write_input_file(
            tmp_path / "2026" / "03" / "09" / "DWD_202603091230.nc",
            np.arange(50.0, 55.0),
            np.arange(0.0, 5.0),
        )

        check_current_data_existence_file(
            TIME_STEP,
            "DWD",
            "NW_EUROPE",
            str(tmp_path),
            "{year}/{month}/{day}/{dataset_name}_{timestamp}.nc",
        )


class TestLoadDataFromFiles:
    @pytest.fixture
    def time_steps(self):
        return [
            datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 9, 12, 15, tzinfo=timezone.utc),
        ]

    def test_files_are_concatenated_along_time(
        self, tmp_path, write_input_file, time_steps
    ):
        for i, ts in enumerate(time_steps):
            write_input_file(
                tmp_path / f"DWD_{ts.strftime('%Y%m%d%H%M')}.nc",
                np.arange(45.0, 65.0),
                np.arange(-5.0, 15.0),
                sds=np.full((20, 20), 100.0 * (i + 1)),
            )

        ds = load_data_from_files(
            time_steps,
            "DWD",
            "NW_EUROPE",
            str(tmp_path),
            "past data",
            "{dataset_name}_{timestamp}.nc",
            bbox="0,50,5,55",
        )

        assert ds.sizes["time"] == 2
        assert list(ds.time.values) == [
            np.datetime64(ts.replace(tzinfo=None)) for ts in time_steps
        ]
        assert ds["sds"].isel(time=0).values.max() == 100.0
        assert ds["sds"].isel(time=1).values.max() == 200.0

    def test_loaded_data_is_subset_to_the_bbox(
        self, tmp_path, write_input_file, time_steps
    ):
        for ts in time_steps:
            write_input_file(
                tmp_path / f"DWD_{ts.strftime('%Y%m%d%H%M')}.nc",
                np.arange(45.0, 65.0),
                np.arange(-5.0, 15.0),
            )

        ds = load_data_from_files(
            time_steps,
            "DWD",
            "NW_EUROPE",
            str(tmp_path),
            "past data",
            "{dataset_name}_{timestamp}.nc",
            bbox="0,50,5,55",
        )

        assert ds.lat.min() >= 49.0 and ds.lat.max() <= 56.0
        assert ds.lon.min() >= -1.0 and ds.lon.max() <= 6.0

    def test_a_missing_file_raises_and_names_the_data_type(
        self, tmp_path, write_input_file, time_steps
    ):
        write_input_file(
            tmp_path / f"DWD_{time_steps[0].strftime('%Y%m%d%H%M')}.nc",
            np.arange(50.0, 55.0),
            np.arange(0.0, 5.0),
        )

        with pytest.raises(FileNotFoundError, match="Clearsky data file not found"):
            load_data_from_files(
                time_steps,
                "DWD",
                "NW_EUROPE",
                str(tmp_path),
                "clearsky data",
                "{dataset_name}_{timestamp}.nc",
                bbox="0,50,5,55",
            )

    def test_no_time_steps_yields_an_empty_dataset(self, tmp_path):
        ds = load_data_from_files(
            [],
            "DWD",
            "NW_EUROPE",
            str(tmp_path),
            "past data",
            "{dataset_name}_{timestamp}.nc",
            bbox="0,50,5,55",
        )

        assert len(ds.data_vars) == 0


class TestSaveForecast:
    @pytest.fixture
    def grid(self):
        return np.arange(50.0, 55.0), np.arange(0.0, 6.0)

    def test_deterministic_forecast_round_trips(
        self, tmp_path, grid, make_nowcast_config
    ):
        latitudes, longitudes = grid
        forecast = np.random.default_rng(0).random(
            (1, 3, len(latitudes), len(longitudes))
        )
        config = make_nowcast_config(nowcast_directory=str(tmp_path))

        filename = save_forecast(
            forecast,
            TIME_STEP,
            n_steps=3,
            latitudes=latitudes,
            longitudes=longitudes,
            dataset_name="DWD",
            satellite="Meteosat Second Generation (MSG)",
            nowcast_config=config,
            model_version="1.4.0",
            output_mode="deterministic",
        )

        assert filename == "SolarNowcast_DWD_202603091230.nc"
        with xr.open_dataset(tmp_path / filename) as ds:
            np.testing.assert_allclose(ds["GHI_probabilistic_advection"].values, forecast)
            assert ds["GHI_probabilistic_advection"].attrs["units"] == "W m-2"
            assert ds.attrs["output_mode"] == "deterministic"
            assert ds.attrs["satellite_data_source"] == "DWD"
            assert ds.attrs["model_version"] == "1.4.0"

    def test_time_axis_uses_the_configured_data_frequency(
        self, tmp_path, grid, make_nowcast_config
    ):
        latitudes, longitudes = grid
        config = make_nowcast_config(
            nowcast_directory=str(tmp_path), input_data_frequency_minutes=10
        )

        filename = save_forecast(
            np.zeros((1, 3, len(latitudes), len(longitudes))),
            TIME_STEP,
            n_steps=3,
            latitudes=latitudes,
            longitudes=longitudes,
            dataset_name="DWD",
            satellite="MTG",
            nowcast_config=config,
            model_version="1.4.0",
            output_mode="deterministic",
        )

        with xr.open_dataset(tmp_path / filename) as ds:
            assert [str(t) for t in ds.time.values] == [
                "2026-03-09T12:30:00.000000000",
                "2026-03-09T12:40:00.000000000",
                "2026-03-09T12:50:00.000000000",
            ]

    def test_statistics_are_written_as_separate_variables(
        self, tmp_path, grid, make_nowcast_config
    ):
        latitudes, longitudes = grid
        shape = (1, 2, len(latitudes), len(longitudes))
        statistics = {
            "median": np.full(shape, 1.0),
            "p90": np.full(shape, 2.0),
        }
        config = make_nowcast_config(nowcast_directory=str(tmp_path))

        filename = save_forecast(
            statistics,
            TIME_STEP,
            n_steps=2,
            latitudes=latitudes,
            longitudes=longitudes,
            dataset_name="DWD",
            satellite="MSG",
            nowcast_config=config,
            model_version="1.4.0",
            output_mode="ensemble_statistics",
        )

        with xr.open_dataset(tmp_path / filename) as ds:
            assert set(ds.data_vars) == {
                "median_GHI_probabilistic_advection",
                "p90_GHI_probabilistic_advection",
            }
            assert ds.attrs["statistics"] == "median,p90"

    def test_nested_output_formats_create_directories(
        self, tmp_path, grid, make_nowcast_config
    ):
        latitudes, longitudes = grid
        config = make_nowcast_config(
            nowcast_directory=str(tmp_path),
            output_filename_format="{year}/{month}/SolarNowcast_{timestamp}.nc",
        )

        filename = save_forecast(
            np.zeros((1, 1, len(latitudes), len(longitudes))),
            TIME_STEP,
            n_steps=1,
            latitudes=latitudes,
            longitudes=longitudes,
            dataset_name="DWD",
            satellite="MSG",
            nowcast_config=config,
            model_version="1.4.0",
            output_mode="deterministic",
        )

        assert filename == "2026/03/SolarNowcast_202603091230.nc"
        assert (tmp_path / filename).exists()

    def test_wrong_dimensionality_is_rejected(self, tmp_path, grid, make_nowcast_config):
        latitudes, longitudes = grid
        config = make_nowcast_config(nowcast_directory=str(tmp_path))

        with pytest.raises(ValueError, match="must have shape"):
            save_forecast(
                np.zeros((3, len(latitudes), len(longitudes))),
                TIME_STEP,
                n_steps=3,
                latitudes=latitudes,
                longitudes=longitudes,
                dataset_name="DWD",
                satellite="MSG",
                nowcast_config=config,
                model_version="1.4.0",
                output_mode="deterministic",
            )

    def test_ensemble_size_must_match_the_configuration(
        self, tmp_path, grid, make_nowcast_config
    ):
        latitudes, longitudes = grid
        config = make_nowcast_config(nowcast_directory=str(tmp_path), ens_members=5)

        with pytest.raises(ValueError, match="does not match expected 5"):
            save_forecast(
                np.zeros((3, 2, len(latitudes), len(longitudes))),
                TIME_STEP,
                n_steps=2,
                latitudes=latitudes,
                longitudes=longitudes,
                dataset_name="DWD",
                satellite="MSG",
                nowcast_config=config,
                model_version="1.4.0",
                output_mode="full_ensemble",
            )

    def test_empty_statistics_mapping_is_rejected(
        self, tmp_path, grid, make_nowcast_config
    ):
        latitudes, longitudes = grid
        config = make_nowcast_config(nowcast_directory=str(tmp_path))

        with pytest.raises(ValueError, match="cannot be empty"):
            save_forecast(
                {},
                TIME_STEP,
                n_steps=1,
                latitudes=latitudes,
                longitudes=longitudes,
                dataset_name="DWD",
                satellite="MSG",
                nowcast_config=config,
                model_version="1.4.0",
                output_mode="ensemble_statistics",
            )

    def test_statistic_arrays_must_share_a_shape(
        self, tmp_path, grid, make_nowcast_config
    ):
        latitudes, longitudes = grid
        config = make_nowcast_config(nowcast_directory=str(tmp_path))
        statistics = {
            "median": np.zeros((1, 2, len(latitudes), len(longitudes))),
            "p90": np.zeros((1, 3, len(latitudes), len(longitudes))),
        }

        with pytest.raises(ValueError, match="share the same shape"):
            save_forecast(
                statistics,
                TIME_STEP,
                n_steps=2,
                latitudes=latitudes,
                longitudes=longitudes,
                dataset_name="DWD",
                satellite="MSG",
                nowcast_config=config,
                model_version="1.4.0",
                output_mode="ensemble_statistics",
            )


class TestCorruptAndIncompleteInput:
    """Failure modes for input files that exist but cannot be used."""

    @pytest.fixture
    def time_steps(self):
        base = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        return [base + timedelta(minutes=15 * i) for i in range(3)]

    def test_a_file_that_is_not_netcdf_raises_when_loaded(self, tmp_path, time_steps):
        # xarray reports an unreadable file by exhausting its backends.
        (tmp_path / "DWD_202603091200.nc").write_bytes(b"this is not a NetCDF file")

        with pytest.raises(ValueError, match="did not find a match"):
            load_data_from_files(
                time_steps[:1],
                "DWD",
                "NW_EUROPE",
                str(tmp_path),
                "past data",
                "{dataset_name}_{timestamp}.nc",
                bbox="0,50,5,55",
            )

    def test_a_file_that_is_not_netcdf_raises_during_the_existence_check(
        self, tmp_path, time_steps
    ):
        (tmp_path / "DWD_202603091200.nc").write_bytes(b"this is not a NetCDF file")

        with pytest.raises(ValueError, match="did not find a match"):
            check_current_data_existence_file(
                time_steps[0],
                "DWD",
                "NW_EUROPE",
                str(tmp_path),
                "{dataset_name}_{timestamp}.nc",
                required_domain="0,50,5,55",
            )

    def test_a_gap_in_an_otherwise_complete_sequence_is_reported(
        self, tmp_path, write_input_file, time_steps
    ):
        # Write the first and last step, leaving the middle one missing.
        for time_step in (time_steps[0], time_steps[2]):
            write_input_file(
                tmp_path / f"DWD_{time_step.strftime('%Y%m%d%H%M')}.nc",
                np.arange(50.0, 55.0),
                np.arange(0.0, 5.0),
            )

        with pytest.raises(FileNotFoundError, match="DWD_202603091215.nc"):
            load_data_from_files(
                time_steps,
                "DWD",
                "NW_EUROPE",
                str(tmp_path),
                "past data",
                "{dataset_name}_{timestamp}.nc",
                bbox="0,50,5,55",
            )

    def test_a_file_without_usable_coordinates_fails_the_coverage_check(self, tmp_path):
        xr.Dataset(
            {"sds": (["row", "col"], np.zeros((3, 4)))},
            coords={"row": np.arange(3.0), "col": np.arange(4.0)},
        ).to_netcdf(tmp_path / "DWD_202603091200.nc")

        with pytest.raises(RuntimeError, match="Could not find coordinates"):
            check_current_data_existence_file(
                datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc),
                "DWD",
                "NW_EUROPE",
                str(tmp_path),
                "{dataset_name}_{timestamp}.nc",
                required_domain="0,50,5,55",
            )

    def test_a_grid_that_is_too_small_fails_the_coverage_check(
        self, tmp_path, write_input_file
    ):
        write_input_file(
            tmp_path / "DWD_202603091200.nc",
            np.arange(50.0, 52.0, 0.5),
            np.arange(0.0, 2.0, 0.5),
        )

        with pytest.raises(RuntimeError, match="does not cover requested"):
            check_current_data_existence_file(
                datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc),
                "DWD",
                "NW_EUROPE",
                str(tmp_path),
                "{dataset_name}_{timestamp}.nc",
                required_domain="-10.75,47.25,20,63.5",
            )


class TestFetchClearskyWithFallback:
    """The documented fallback: try the target day, then earlier days."""

    @pytest.fixture
    def time_steps(self):
        return [datetime(2026, 3, 8, 12, 15, tzinfo=timezone.utc)]

    def fetch(self, config, time_steps, max_fallback_days=3):
        return fetch_clearsky_with_fallback(
            time_steps,
            "files",
            max_fallback_days,
            "{dataset_name}_{timestamp}.nc",
            {"nc_variable_names": {"sds": "sds", "sds_cs": "sds_cs"}},
            "0,50,5,55",
            "DWD",
            "NW_EUROPE",
            config,
            S3Config.from_env(),
        )

    def test_data_for_the_requested_day_is_used_directly(
        self, tmp_path, write_input_file, make_nowcast_config, time_steps
    ):
        write_input_file(
            tmp_path / "DWD_202603081215.nc",
            np.arange(45.0, 65.0),
            np.arange(-5.0, 15.0),
        )
        config = make_nowcast_config(satellite_data_directory=str(tmp_path))

        clearsky = self.fetch(config, time_steps)

        assert clearsky.sizes["time"] == 1
        np.testing.assert_allclose(clearsky["sds_cs"].values.max(), CLEARSKY)

    def test_an_earlier_day_is_used_when_the_target_day_is_missing(
        self, tmp_path, write_input_file, make_nowcast_config, time_steps
    ):
        # Only the file from two days earlier exists.
        write_input_file(
            tmp_path / "DWD_202603061215.nc",
            np.arange(45.0, 65.0),
            np.arange(-5.0, 15.0),
            sds_cs=np.full((20, 20), 500.0),
        )
        config = make_nowcast_config(satellite_data_directory=str(tmp_path))

        clearsky = self.fetch(config, time_steps)

        assert clearsky.sizes["time"] == 1
        np.testing.assert_allclose(clearsky["sds_cs"].values.max(), 500.0)

    def test_fallback_data_keeps_the_originally_requested_time_coordinate(
        self, tmp_path, write_input_file, make_nowcast_config, time_steps
    ):
        write_input_file(
            tmp_path / "DWD_202603061215.nc",
            np.arange(45.0, 65.0),
            np.arange(-5.0, 15.0),
        )
        config = make_nowcast_config(satellite_data_directory=str(tmp_path))

        clearsky = self.fetch(config, time_steps)

        assert list(clearsky.time.values) == [np.datetime64("2026-03-08T12:15:00")]

    def test_a_time_step_with_no_data_within_the_window_is_omitted(
        self, tmp_path, make_nowcast_config, time_steps
    ):
        config = make_nowcast_config(satellite_data_directory=str(tmp_path))

        clearsky = self.fetch(config, time_steps)

        assert len(clearsky.data_vars) == 0

    def test_the_search_stops_after_max_fallback_days(
        self, tmp_path, write_input_file, make_nowcast_config, time_steps
    ):
        # Data exists three days back, but only two days are searched.
        write_input_file(
            tmp_path / "DWD_202603051215.nc",
            np.arange(45.0, 65.0),
            np.arange(-5.0, 15.0),
        )
        config = make_nowcast_config(satellite_data_directory=str(tmp_path))

        clearsky = self.fetch(config, time_steps, max_fallback_days=2)

        assert len(clearsky.data_vars) == 0


class TestS3FilesystemConfiguration:
    def test_the_configured_endpoint_is_passed_to_fsspec(self, fake_s3):
        s3_config = S3Config(
            endpoint_url="https://s3.example.org",
            bucket="solar",
            input_prefix="satellite_data",
            output_prefix="nowcasts",
        )

        check_existence = pytest.raises(FileNotFoundError)
        with check_existence:
            check_current_data_existence_s3(
                TIME_STEP, "DWD", "NW_EUROPE", s3_config, "{dataset_name}_{timestamp}.nc"
            )

        assert fake_s3.calls[0][0] == "s3"
        assert fake_s3.client_kwargs["endpoint_url"] == "https://s3.example.org"

    def test_ssl_verification_is_disabled_when_configured(self, fake_s3):
        s3_config = S3Config(
            endpoint_url="https://s3.example.org",
            bucket="solar",
            input_prefix="satellite_data",
            output_prefix="nowcasts",
            verify_ssl=False,
        )

        with pytest.raises(FileNotFoundError):
            check_current_data_existence_s3(
                TIME_STEP, "DWD", "NW_EUROPE", s3_config, "{dataset_name}_{timestamp}.nc"
            )

        assert fake_s3.client_kwargs["verify"] is False

    def test_ssl_verification_is_left_alone_by_default(self, fake_s3):
        s3_config = S3Config(
            endpoint_url="https://s3.example.org",
            bucket="solar",
            input_prefix="satellite_data",
            output_prefix="nowcasts",
        )

        with pytest.raises(FileNotFoundError):
            check_current_data_existence_s3(
                TIME_STEP, "DWD", "NW_EUROPE", s3_config, "{dataset_name}_{timestamp}.nc"
            )

        assert "verify" not in fake_s3.client_kwargs


@pytest.fixture
def s3_config():
    return S3Config(
        endpoint_url="https://s3.example.org",
        bucket="solar",
        input_prefix="satellite_data",
        output_prefix="nowcasts",
    )


class TestCheckCurrentDataExistenceS3:
    def test_an_existing_object_passes(
        self, fake_s3, s3_config, make_gridded_dataset, to_netcdf_bytes
    ):
        fake_s3.files["s3://solar/satellite_data/DWD_202603091230.nc"] = to_netcdf_bytes(
            make_gridded_dataset(np.arange(45.0, 65.0), np.arange(-5.0, 15.0))
        )

        check_current_data_existence_s3(
            TIME_STEP, "DWD", "NW_EUROPE", s3_config, "{dataset_name}_{timestamp}.nc"
        )

    def test_the_object_key_is_built_from_bucket_prefix_and_filename(
        self, fake_s3, s3_config, make_gridded_dataset, to_netcdf_bytes
    ):
        key = "s3://solar/satellite_data/2026/03/DWD_202603091230.nc"
        fake_s3.files[key] = to_netcdf_bytes(
            make_gridded_dataset(np.arange(45.0, 65.0), np.arange(-5.0, 15.0))
        )

        check_current_data_existence_s3(
            TIME_STEP,
            "DWD",
            "NW_EUROPE",
            s3_config,
            "{year}/{month}/{dataset_name}_{timestamp}.nc",
        )

    def test_a_missing_object_raises_file_not_found(self, fake_s3, s3_config):
        with pytest.raises(FileNotFoundError, match="not yet found in S3"):
            check_current_data_existence_s3(
                TIME_STEP, "DWD", "NW_EUROPE", s3_config, "{dataset_name}_{timestamp}.nc"
            )

    def test_domain_coverage_is_validated_for_objects_in_s3(
        self, fake_s3, s3_config, make_gridded_dataset, to_netcdf_bytes
    ):
        fake_s3.files["s3://solar/satellite_data/DWD_202603091230.nc"] = to_netcdf_bytes(
            make_gridded_dataset(np.arange(50.0, 55.0), np.arange(0.0, 5.0))
        )

        with pytest.raises(RuntimeError, match="does not cover requested"):
            check_current_data_existence_s3(
                TIME_STEP,
                "DWD",
                "NW_EUROPE",
                s3_config,
                "{dataset_name}_{timestamp}.nc",
                required_domain="-10.75,47.25,20,63.5",
            )

    def test_authentication_failures_propagate(self, fake_s3, s3_config):
        key = "s3://solar/satellite_data/DWD_202603091230.nc"
        fake_s3.files[key] = b""
        fake_s3.open_failures[key] = PermissionError("Access Denied")

        with pytest.raises(PermissionError, match="Access Denied"):
            check_current_data_existence_s3(
                TIME_STEP,
                "DWD",
                "NW_EUROPE",
                s3_config,
                "{dataset_name}_{timestamp}.nc",
                required_domain="0,50,5,55",
            )


class TestLoadDataFromS3:
    @pytest.fixture
    def time_steps(self):
        base = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        return [base + timedelta(minutes=15 * i) for i in range(2)]

    def seed(self, fake_s3, to_netcdf_bytes, make_gridded_dataset, time_steps, values):
        # Objects in S3 are the files produced by the downloaders, which carry
        # their own time dimension.
        for time_step, value in zip(time_steps, values):
            key = f"s3://solar/satellite_data/DWD_{time_step.strftime('%Y%m%d%H%M')}.nc"
            dataset = make_gridded_dataset(
                np.arange(45.0, 65.0),
                np.arange(-5.0, 15.0),
                sds=np.full((20, 20), value),
            ).expand_dims(time=[time_step.replace(tzinfo=None)])
            fake_s3.files[key] = to_netcdf_bytes(dataset)

    def load(self, s3_config, time_steps):
        return load_data_from_s3(
            time_steps,
            "DWD",
            "NW_EUROPE",
            s3_config,
            "past data",
            "{dataset_name}_{timestamp}.nc",
            bbox="0,50,5,55",
        )

    def test_objects_are_concatenated_along_time(
        self, fake_s3, s3_config, to_netcdf_bytes, make_gridded_dataset, time_steps
    ):
        self.seed(
            fake_s3, to_netcdf_bytes, make_gridded_dataset, time_steps, [100.0, 200.0]
        )

        ds = self.load(s3_config, time_steps)

        assert ds.sizes["time"] == 2
        assert ds["sds"].isel(time=0).values.max() == 100.0
        assert ds["sds"].isel(time=1).values.max() == 200.0

    def test_loaded_objects_are_subset_to_the_bbox(
        self, fake_s3, s3_config, to_netcdf_bytes, make_gridded_dataset, time_steps
    ):
        self.seed(
            fake_s3, to_netcdf_bytes, make_gridded_dataset, time_steps, [100.0, 200.0]
        )

        ds = self.load(s3_config, time_steps)

        assert ds.lat.min() >= 49.0 and ds.lat.max() <= 56.0
        assert ds.lon.min() >= -1.0 and ds.lon.max() <= 6.0

    def test_a_missing_object_is_reported_with_its_path(
        self, fake_s3, s3_config, time_steps
    ):
        with pytest.raises(FileNotFoundError, match="s3://solar/satellite_data"):
            self.load(s3_config, time_steps)

    def test_no_time_steps_yields_an_empty_dataset(self, fake_s3, s3_config):
        assert len(self.load(s3_config, []).data_vars) == 0


class TestSaveForecastToS3:
    @pytest.fixture
    def forecast_dataset(self):
        return xr.Dataset(
            {
                "GHI_probabilistic_advection": (
                    ["ensemble", "time", "lat", "lon"],
                    np.full((1, 2, 3, 4), 250.0),
                )
            },
            coords={
                "time": [datetime(2026, 3, 9, 12, 30), datetime(2026, 3, 9, 12, 45)],
                "ensemble": np.arange(1, dtype=np.int32),
                "lat": np.arange(3.0),
                "lon": np.arange(4.0),
            },
        )

    @pytest.fixture
    def time_encoding(self):
        return {
            "units": "minutes since 2026-03-09 12:30:00",
            "calendar": "standard",
            "dtype": np.int32,
        }

    def test_the_object_is_written_under_the_output_prefix(
        self, fake_s3, s3_config, forecast_dataset, time_encoding
    ):
        save_forecast_to_s3(
            forecast_dataset, "SolarNowcast_DWD.nc", s3_config, time_encoding
        )

        assert "s3://solar/nowcasts/SolarNowcast_DWD.nc" in fake_s3.files

    def test_the_uploaded_bytes_are_a_readable_netcdf_file(
        self, fake_s3, s3_config, forecast_dataset, time_encoding
    ):
        save_forecast_to_s3(
            forecast_dataset, "SolarNowcast_DWD.nc", s3_config, time_encoding
        )

        uploaded = fake_s3.files["s3://solar/nowcasts/SolarNowcast_DWD.nc"]
        with xr.open_dataset(uploaded, engine="h5netcdf") as ds:
            np.testing.assert_allclose(ds["GHI_probabilistic_advection"].values, 250.0)
            assert list(ds.time.values) == [
                np.datetime64("2026-03-09T12:30:00"),
                np.datetime64("2026-03-09T12:45:00"),
            ]

    def test_upload_failures_propagate(
        self, fake_s3, s3_config, forecast_dataset, time_encoding
    ):
        fake_s3.open_failures["s3://solar/nowcasts/SolarNowcast_DWD.nc"] = OSError(
            "connection reset by peer"
        )

        with pytest.raises(OSError, match="connection reset"):
            save_forecast_to_s3(
                forecast_dataset, "SolarNowcast_DWD.nc", s3_config, time_encoding
            )

    def test_save_forecast_routes_to_s3_in_s3_run_mode(
        self, fake_s3, s3_config, make_nowcast_config, tmp_path
    ):
        config = make_nowcast_config(nowcast_directory=str(tmp_path))

        filename = save_forecast(
            np.full((1, 2, 3, 4), 300.0),
            TIME_STEP,
            n_steps=2,
            latitudes=np.arange(3.0),
            longitudes=np.arange(4.0),
            dataset_name="DWD",
            satellite="MSG",
            nowcast_config=config,
            model_version="1.4.0",
            output_mode="deterministic",
            run_mode="s3",
            s3_config=s3_config,
        )

        assert filename == "SolarNowcast_DWD_202603091230.nc"
        assert f"s3://solar/nowcasts/{filename}" in fake_s3.files
        # Nothing should have been written to the local output directory.
        assert list(tmp_path.iterdir()) == []
