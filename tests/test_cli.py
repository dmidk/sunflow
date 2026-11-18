#!/usr/bin/env python3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from sunflow.main import parse_arguments
from sunflow.validation import validate_config


@pytest.fixture
def run_cli(monkeypatch):
    """Invoke parse_arguments with the given command line arguments."""

    def _run(*arguments):
        monkeypatch.setattr(sys, "argv", ["sunflow", *arguments])
        return parse_arguments()

    return _run


class TestArgumentDefaults:
    def test_defaults_match_the_documented_behaviour(self, run_cli):
        args = run_cli()

        assert args.run_mode == "download"
        assert args.dataset == "KNMI_MSG"
        assert args.domain_satellite == "NW_EUROPE"
        assert args.domain_nowcast is None
        assert args.time is None
        assert args.start_time is None
        assert args.end_time is None
        assert args.ensemble_members == 1
        assert args.full_ensemble is False

    @pytest.mark.parametrize("run_mode", ["download", "files", "s3"])
    def test_supported_run_modes_are_accepted(self, run_cli, run_mode):
        assert run_cli("--run_mode", run_mode).run_mode == run_mode

    @pytest.mark.parametrize(
        "dataset",
        ["KNMI_MSG", "KNMI_MTG", "DWD", "specMAGIC_MTG", "specMAGIC_MTG_pvlib"],
    )
    def test_supported_datasets_are_accepted(self, run_cli, dataset):
        assert run_cli("--dataset", dataset).dataset == dataset

    def test_unknown_run_mode_is_rejected(self, run_cli):
        with pytest.raises(SystemExit):
            run_cli("--run_mode", "ftp")

    def test_unknown_dataset_is_rejected(self, run_cli):
        with pytest.raises(SystemExit):
            run_cli("--dataset", "GOES")

    def test_full_ensemble_is_a_flag(self, run_cli):
        assert run_cli("--full_ensemble").full_ensemble is True


class TestTimeParsing:
    def test_naive_times_are_assumed_to_be_utc(self, run_cli):
        args = run_cli("--time", "2026-03-09T12:30")

        assert args.time == datetime(2026, 3, 9, 12, 30, tzinfo=timezone.utc)
        assert args.time.tzinfo is timezone.utc

    def test_explicit_utc_is_preserved(self, run_cli):
        args = run_cli("--time", "2026-03-09T12:30Z")

        assert args.time == datetime(2026, 3, 9, 12, 30, tzinfo=timezone.utc)

    def test_other_time_zones_are_converted_to_utc(self, run_cli):
        args = run_cli("--time", "2026-03-09T14:30+02:00")

        assert args.time == datetime(2026, 3, 9, 12, 30, tzinfo=timezone.utc)
        assert args.time.tzinfo is timezone.utc

    def test_malformed_times_are_rejected(self, run_cli):
        with pytest.raises(SystemExit):
            run_cli("--time", "yesterday")

    def test_a_time_span_is_accepted(self, run_cli):
        args = run_cli("--start_time", "2026-03-09T12:00Z", "--end_time", "2026-03-09T15:00Z")

        assert args.start_time == datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        assert args.end_time == datetime(2026, 3, 9, 15, 0, tzinfo=timezone.utc)

    def test_a_single_time_cannot_be_combined_with_a_span(self, run_cli, capsys):
        with pytest.raises(SystemExit):
            run_cli(
                "--time",
                "2026-03-09T12:00Z",
                "--start_time",
                "2026-03-09T12:00Z",
                "--end_time",
                "2026-03-09T15:00Z",
            )

        assert "cannot be combined" in capsys.readouterr().err

    @pytest.mark.parametrize("argument", ["--start_time", "--end_time"])
    def test_span_bounds_must_be_given_together(self, run_cli, capsys, argument):
        with pytest.raises(SystemExit):
            run_cli(argument, "2026-03-09T12:00Z")

        assert "must be provided together" in capsys.readouterr().err

    def test_the_span_must_not_run_backwards(self, run_cli, capsys):
        with pytest.raises(SystemExit):
            run_cli(
                "--start_time", "2026-03-09T15:00Z", "--end_time", "2026-03-09T12:00Z"
            )

        assert "must be before" in capsys.readouterr().err

    def test_an_empty_span_of_one_instant_is_allowed(self, run_cli):
        args = run_cli(
            "--start_time", "2026-03-09T12:00Z", "--end_time", "2026-03-09T12:00Z"
        )

        assert args.start_time == args.end_time


class TestDomainArguments:
    def test_custom_satellite_domain_is_accepted(self, run_cli):
        args = run_cli(
            "--domain_satellite", "CUSTOM", "--custom_domain_satellite", "4,50,18,62"
        )

        assert args.domain_satellite == "CUSTOM"
        assert args.custom_domain_satellite == "4,50,18,62"

    def test_custom_satellite_domain_requires_a_bbox(self, run_cli, capsys):
        with pytest.raises(SystemExit):
            run_cli("--domain_satellite", "CUSTOM")

        assert "--custom_domain_satellite is required" in capsys.readouterr().err

    def test_a_bbox_without_custom_mode_is_rejected(self, run_cli, capsys):
        with pytest.raises(SystemExit):
            run_cli("--custom_domain_satellite", "4,50,18,62")

        assert "only valid when --domain_satellite=CUSTOM" in capsys.readouterr().err

    def test_an_invalid_bbox_is_rejected(self, run_cli, capsys):
        with pytest.raises(SystemExit):
            run_cli(
                "--domain_satellite", "CUSTOM", "--custom_domain_satellite", "4,50,18"
            )

        assert "Invalid --custom_domain_satellite format" in capsys.readouterr().err

    def test_nowcast_domain_can_be_set_independently(self, run_cli):
        args = run_cli(
            "--domain_satellite", "NW_EUROPE_SATELLITE", "--domain_nowcast", "DENMARK"
        )

        assert args.domain_satellite == "NW_EUROPE_SATELLITE"
        assert args.domain_nowcast == "DENMARK"

    def test_custom_nowcast_bbox_requires_custom_nowcast_domain(self, run_cli, capsys):
        with pytest.raises(SystemExit):
            run_cli("--custom_domain_nowcast", "4,50,18,62")

        assert "requires --domain_nowcast=CUSTOM" in capsys.readouterr().err

    def test_custom_nowcast_domain_is_accepted(self, run_cli):
        args = run_cli(
            "--domain_nowcast", "CUSTOM", "--custom_domain_nowcast", "4,50,18,62"
        )

        assert args.custom_domain_nowcast == "4,50,18,62"


class TestShippedConfigYaml:
    """The shipped config.yaml is part of the deliverable and must stay loadable."""

    @pytest.fixture
    def config(self):
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with config_path.open() as handle:
            return yaml.safe_load(handle)

    def test_every_cli_dataset_choice_has_a_configuration_entry(self, config):
        datasets = [
            "KNMI_MSG",
            "KNMI_MTG",
            "DWD",
            "specMAGIC_MTG",
            "specMAGIC_MTG_pvlib",
        ]

        assert set(datasets) <= set(config)

    @pytest.mark.parametrize(
        "dataset",
        ["KNMI_MSG", "KNMI_MTG", "DWD", "specMAGIC_MTG", "specMAGIC_MTG_pvlib"],
    )
    def test_every_entry_passes_validation(self, config, dataset):
        validate_config(config[dataset], dataset)

    @pytest.mark.parametrize(
        "dataset",
        ["KNMI_MSG", "KNMI_MTG", "DWD", "specMAGIC_MTG", "specMAGIC_MTG_pvlib"],
    )
    def test_every_entry_maps_both_netcdf_variable_names(self, config, dataset):
        assert set(config[dataset]["nc_variable_names"]) == {"sds", "sds_cs"}

    @pytest.mark.parametrize(
        "dataset",
        ["KNMI_MSG", "KNMI_MTG", "DWD", "specMAGIC_MTG", "specMAGIC_MTG_pvlib"],
    )
    def test_every_entry_defines_an_input_filename_format(self, config, dataset):
        assert config[dataset]["input_filename_format"]
