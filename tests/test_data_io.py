#!/usr/bin/env python3
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from sunflow.data_io import (
    check_current_data_existence_file,
    generate_filename,
    load_data_from_files,
    save_forecast,
)

TIME_STEP = datetime(2026, 3, 9, 12, 30, tzinfo=timezone.utc)


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
