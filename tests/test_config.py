#!/usr/bin/env python3
import pytest

from sunflow.config import (
    DEFAULT_OUTPUT_FILENAME_FORMAT,
    NowcastConfig,
    S3Config,
    _parse_ensemble_statistics,
)

# Every environment variable these tests touch, cleared before each test so the
# developer's own shell cannot influence the outcome.
_ENV_VARS = (
    "S3_ENDPOINT_URL",
    "S3_BUCKET",
    "S3_INPUT_PREFIX",
    "S3_OUTPUT_PREFIX",
    "AWS_VERIFY_SSL",
    "NOWCAST_DIRECTORY",
    "ALPHA",
    "BETA",
    "PAST_STEPS",
    "FUTURE_STEPS",
    "INPUT_DATA_AVAILABILITY_DELAY_MINUTES",
    "INPUT_DATA_FREQUENCY_MINUTES",
    "MAX_WAITING_TIME_MINUTES",
    "SATELLITE_DATA_DIRECTORY",
    "MAX_CLEARSKY_FALLBACK_DAYS",
    "MIN_SOLAR_ELEVATION_DEGREES",
    "ENSEMBLE_STATISTICS",
    "INPUT_FILENAME_FORMAT",
    "OUTPUT_FILENAME_FORMAT",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class TestS3Config:
    def test_defaults_when_nothing_is_set(self):
        config = S3Config.from_env()

        assert config.endpoint_url is None
        assert config.bucket is None
        assert config.input_prefix == "satellite_data"
        assert config.output_prefix == "nowcasts"
        assert config.verify_ssl is True

    def test_values_are_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.example.org")
        monkeypatch.setenv("S3_BUCKET", "solar")
        monkeypatch.setenv("S3_INPUT_PREFIX", "in")
        monkeypatch.setenv("S3_OUTPUT_PREFIX", "out")

        config = S3Config.from_env()

        assert config.endpoint_url == "https://s3.example.org"
        assert config.bucket == "solar"
        assert config.input_prefix == "in"
        assert config.output_prefix == "out"

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " True "])
    def test_ssl_verification_stays_enabled_for_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("AWS_VERIFY_SSL", value)

        assert S3Config.from_env().verify_ssl is True

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", ""])
    def test_ssl_verification_is_disabled_for_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("AWS_VERIFY_SSL", value)

        assert S3Config.from_env().verify_ssl is False


class TestNowcastConfigDefaults:
    def test_deterministic_run_defaults_to_no_noise(self):
        config = NowcastConfig.from_env(ensemble_members=1)

        assert config.ens_members == 1
        assert config.alpha == 0.0
        assert config.beta == 0.0

    def test_ensemble_run_defaults_to_published_noise_parameters(self):
        config = NowcastConfig.from_env(ensemble_members=20)

        assert config.ens_members == 20
        assert config.alpha == pytest.approx(9.23)
        assert config.beta == pytest.approx(0.15)

    def test_remaining_defaults(self):
        config = NowcastConfig.from_env()

        assert config.nowcast_directory == "."
        assert config.past_steps == 4
        assert config.future_steps == 24
        assert config.input_data_availability_delay_minutes == 24
        assert config.input_data_frequency_minutes == 15
        assert config.max_waiting_time_minutes == 27
        assert config.satellite_data_directory == "."
        assert config.max_clearsky_fallback_days == 3
        assert config.min_solar_elevation_degrees == 6
        assert config.ensemble_statistics == ["median", "mean", "p10", "p25", "p75", "p90"]
        assert config.input_filename_format is None
        assert config.output_filename_format == DEFAULT_OUTPUT_FILENAME_FORMAT


class TestNowcastConfigPrecedence:
    def test_environment_variables_override_defaults(self, monkeypatch):
        monkeypatch.setenv("PAST_STEPS", "6")
        monkeypatch.setenv("FUTURE_STEPS", "12")
        monkeypatch.setenv("ALPHA", "1.5")
        monkeypatch.setenv("MIN_SOLAR_ELEVATION_DEGREES", "3.5")

        config = NowcastConfig.from_env()

        assert config.past_steps == 6
        assert config.future_steps == 12
        assert config.alpha == pytest.approx(1.5)
        assert config.min_solar_elevation_degrees == pytest.approx(3.5)

    def test_overrides_are_used_when_the_environment_is_unset(self):
        config = NowcastConfig.from_env(
            overrides={
                "input_filename_format": "{dataset_name}_{timestamp}.nc",
                "input_data_frequency_minutes": 10,
            }
        )

        assert config.input_filename_format == "{dataset_name}_{timestamp}.nc"
        assert config.input_data_frequency_minutes == 10

    def test_environment_wins_over_overrides(self, monkeypatch):
        monkeypatch.setenv("INPUT_FILENAME_FORMAT", "from_env_{timestamp}.nc")

        config = NowcastConfig.from_env(
            overrides={"input_filename_format": "from_yaml_{timestamp}.nc"}
        )

        assert config.input_filename_format == "from_env_{timestamp}.nc"

    def test_none_valued_override_falls_back_to_the_default(self):
        config = NowcastConfig.from_env(overrides={"past_steps": None})

        assert config.past_steps == 4

    def test_ensemble_statistics_override_accepts_a_list(self):
        config = NowcastConfig.from_env(
            overrides={"ensemble_statistics": ["mean", "p90"]}
        )

        assert config.ensemble_statistics == ["mean", "p90"]

    def test_non_string_overrides_are_coerced(self):
        config = NowcastConfig.from_env(overrides={"min_solar_elevation_degrees": 2.5})

        assert config.min_solar_elevation_degrees == pytest.approx(2.5)


class TestParseEnsembleStatistics:
    def test_percentile_aliases_are_normalised(self):
        statistics = _parse_ensemble_statistics(
            "10th_percentile,25th_percentile,75th_percentile,90th_percentile"
        )

        assert statistics == ["p10", "p25", "p75", "p90"]

    def test_whitespace_and_case_are_tolerated(self):
        assert _parse_ensemble_statistics(" Median , MEAN ") == ["median", "mean"]

    def test_empty_tokens_are_dropped(self):
        assert _parse_ensemble_statistics("mean,,p90,") == ["mean", "p90"]

    def test_order_is_preserved(self):
        assert _parse_ensemble_statistics("p90,mean,median") == ["p90", "mean", "median"]

    @pytest.mark.parametrize("raw", ["", "   ", ",,,"])
    def test_empty_selection_is_rejected(self, raw):
        with pytest.raises(ValueError, match="at least one statistic"):
            _parse_ensemble_statistics(raw)

    def test_unknown_statistic_is_rejected_and_named(self):
        with pytest.raises(ValueError, match="p99"):
            _parse_ensemble_statistics("mean,p99")

    def test_invalid_statistic_from_the_environment_propagates(self, monkeypatch):
        monkeypatch.setenv("ENSEMBLE_STATISTICS", "mode")

        with pytest.raises(ValueError, match="Invalid ENSEMBLE_STATISTICS"):
            NowcastConfig.from_env()
