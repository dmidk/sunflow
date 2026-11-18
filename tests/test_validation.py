#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr

from sunflow.validation import (
    MissingClearskyDataError,
    validate_clearsky_completeness,
    validate_clearsky_shapes,
    validate_config,
    validate_custom_domain,
    validate_data_shape,
    validate_nowcast_config,
    validate_run_mode,
    verify_environment_variables,
)

NC_VARIABLE_NAMES = {"sds": "sds", "sds_cs": "sds_cs"}


@pytest.fixture
def parser():
    return argparse.ArgumentParser()


class TestValidateCustomDomain:
    def test_valid_custom_domain_passes(self, parser):
        validate_custom_domain(
            parser, "CUSTOM", "4,50,18,62", "--domain_satellite", "--custom_domain_satellite"
        )

    def test_predefined_domain_without_a_custom_value_passes(self, parser):
        validate_custom_domain(
            parser, "DENMARK", None, "--domain_satellite", "--custom_domain_satellite"
        )

    def test_none_domain_without_a_custom_value_passes(self, parser):
        validate_custom_domain(
            parser, None, None, "--domain_nowcast", "--custom_domain_nowcast"
        )

    def test_custom_domain_requires_a_value(self, parser, capsys):
        with pytest.raises(SystemExit):
            validate_custom_domain(
                parser, "CUSTOM", None, "--domain_satellite", "--custom_domain_satellite"
            )

        assert "--custom_domain_satellite is required" in capsys.readouterr().err

    def test_malformed_custom_domain_is_reported(self, parser, capsys):
        with pytest.raises(SystemExit):
            validate_custom_domain(
                parser, "CUSTOM", "4,50,18", "--domain_satellite", "--custom_domain_satellite"
            )

        assert "Invalid --custom_domain_satellite format" in capsys.readouterr().err

    def test_reversed_bounds_are_reported(self, parser, capsys):
        with pytest.raises(SystemExit):
            validate_custom_domain(
                parser, "CUSTOM", "18,50,4,62", "--domain_satellite", "--custom_domain_satellite"
            )

        assert "lon_min must be smaller" in capsys.readouterr().err

    def test_custom_value_without_custom_domain_is_rejected(self, parser, capsys):
        with pytest.raises(SystemExit):
            validate_custom_domain(
                parser,
                "DENMARK",
                "4,50,18,62",
                "--domain_satellite",
                "--custom_domain_satellite",
            )

        assert "only valid when --domain_satellite=CUSTOM" in capsys.readouterr().err


class TestValidateConfig:
    def test_config_with_all_required_keys_passes(self):
        validate_config({"satellite": "Meteosat Second Generation (MSG)"}, "DWD")

    def test_missing_satellite_key_exits(self):
        with pytest.raises(SystemExit) as excinfo:
            validate_config({"base_url": "https://example.org"}, "DWD")

        assert excinfo.value.code == 1


class TestValidateNowcastConfig:
    def test_valid_configuration_passes(self, make_nowcast_config):
        validate_nowcast_config(make_nowcast_config())

    @pytest.mark.parametrize("ens_members", [0, -1])
    def test_non_positive_ensemble_size_exits(self, ens_members, make_nowcast_config):
        with pytest.raises(SystemExit) as excinfo:
            validate_nowcast_config(make_nowcast_config(ens_members=ens_members))

        assert excinfo.value.code == 1

    @pytest.mark.parametrize("filename_format", [None, ""])
    def test_missing_input_filename_format_exits(
        self, filename_format, make_nowcast_config
    ):
        with pytest.raises(SystemExit) as excinfo:
            validate_nowcast_config(
                make_nowcast_config(input_filename_format=filename_format)
            )

        assert excinfo.value.code == 1


class TestValidateRunMode:
    def test_download_mode_is_unsupported_for_dwd(self):
        with pytest.raises(SystemExit) as excinfo:
            validate_run_mode("download", "DWD")

        assert excinfo.value.code == 1

    @pytest.mark.parametrize("run_mode", ["files", "s3"])
    def test_other_modes_are_supported_for_dwd(self, run_mode):
        validate_run_mode(run_mode, "DWD")

    @pytest.mark.parametrize("dataset", ["KNMI_MSG", "KNMI_MTG", "specMAGIC_MTG"])
    def test_download_mode_is_supported_for_other_datasets(self, dataset):
        validate_run_mode("download", dataset)


class TestVerifyEnvironmentVariables:
    @pytest.fixture(autouse=True)
    def clean_environment(self, monkeypatch):
        for name in (
            "KNMI_API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "S3_ENDPOINT_URL",
            "S3_BUCKET",
        ):
            monkeypatch.delenv(name, raising=False)

    @pytest.mark.parametrize("dataset", ["KNMI_MSG", "KNMI_MTG"])
    def test_knmi_download_requires_an_api_key(self, dataset):
        with pytest.raises(SystemExit) as excinfo:
            verify_environment_variables("download", dataset)

        assert excinfo.value.code == 1

    @pytest.mark.parametrize("dataset", ["KNMI_MSG", "KNMI_MTG"])
    def test_knmi_download_passes_with_an_api_key(self, monkeypatch, dataset):
        monkeypatch.setenv("KNMI_API_KEY", "secret")

        verify_environment_variables("download", dataset)

    def test_non_knmi_download_needs_no_api_key(self):
        verify_environment_variables("download", "specMAGIC_MTG")

    def test_files_mode_needs_no_environment_variables(self):
        verify_environment_variables("files", "DWD")

    def test_s3_mode_requires_credentials_and_endpoint(self):
        with pytest.raises(SystemExit) as excinfo:
            verify_environment_variables("s3", "DWD")

        assert excinfo.value.code == 1

    def test_s3_mode_passes_once_everything_is_set(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.example.org")
        monkeypatch.setenv("S3_BUCKET", "solar")

        verify_environment_variables("s3", "DWD")

    def test_s3_mode_fails_when_only_some_variables_are_set(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

        with pytest.raises(SystemExit):
            verify_environment_variables("s3", "DWD")


class TestValidateDataShape:
    def test_matching_shape_passes(self):
        time_steps = [datetime(2026, 3, 9, 12, 0), datetime(2026, 3, 9, 12, 15)]
        latitudes = np.arange(5.0)
        longitudes = np.arange(4.0)

        validate_data_shape(np.zeros((2, 5, 4)), time_steps, latitudes, longitudes)

    def test_mismatched_shape_raises_and_reports_both_shapes(self):
        time_steps = [datetime(2026, 3, 9, 12, 0)]

        with pytest.raises(RuntimeError, match=r"Expected \(1, 5, 4\), got \(2, 5, 4\)"):
            validate_data_shape(
                np.zeros((2, 5, 4)), time_steps, np.arange(5.0), np.arange(4.0)
            )


class TestValidateClearskyCompleteness:
    @pytest.fixture
    def time_steps(self):
        return [
            datetime(2026, 3, 8, 12, 15, tzinfo=timezone.utc),
            datetime(2026, 3, 8, 12, 30, tzinfo=timezone.utc),
        ]

    def test_complete_clearsky_data_passes(self, time_steps, make_clearsky_dataset):
        validate_clearsky_completeness(make_clearsky_dataset(time_steps), time_steps)

    def test_missing_time_step_raises(self, time_steps, make_clearsky_dataset):
        clearsky = make_clearsky_dataset(time_steps[:1])

        with pytest.raises(MissingClearskyDataError, match="2026-03-08T12:30:00Z"):
            validate_clearsky_completeness(clearsky, time_steps)

    def test_dataset_without_a_time_coordinate_reports_every_step(self, time_steps):
        with pytest.raises(MissingClearskyDataError, match="2 time steps"):
            validate_clearsky_completeness(xr.Dataset(), time_steps)

    def test_only_the_first_five_missing_steps_are_listed(self):
        base = datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc)
        time_steps = [base + timedelta(minutes=15 * i) for i in range(8)]

        with pytest.raises(MissingClearskyDataError, match=r"\.\.\."):
            validate_clearsky_completeness(xr.Dataset(), time_steps)


class TestValidateClearskyShapes:
    @pytest.fixture
    def time_steps(self):
        return [
            datetime(2026, 3, 8, 12, 15, tzinfo=timezone.utc),
            datetime(2026, 3, 8, 12, 30, tzinfo=timezone.utc),
        ]

    def test_matching_shapes_pass(self, time_steps, make_clearsky_dataset):
        clearsky = make_clearsky_dataset(time_steps, shape=(3, 4))

        validate_clearsky_shapes(clearsky, time_steps, (3, 4), NC_VARIABLE_NAMES)

    def test_mismatched_spatial_shape_raises(self, time_steps, make_clearsky_dataset):
        clearsky = make_clearsky_dataset(time_steps, shape=(3, 4))

        with pytest.raises(RuntimeError, match="Clearsky data shape mismatch"):
            validate_clearsky_shapes(clearsky, time_steps, (5, 6), NC_VARIABLE_NAMES)

    def test_the_configured_clearsky_variable_name_is_used(
        self, time_steps, make_clearsky_dataset
    ):
        clearsky = make_clearsky_dataset(time_steps, shape=(3, 4), variable="SISc")

        validate_clearsky_shapes(
            clearsky, time_steps, (3, 4), {"sds": "SIS", "sds_cs": "SISc"}
        )
