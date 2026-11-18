#!/usr/bin/env python3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr

from sunflow.forecast import (
    compute_ensemble_statistics,
    make_pvlib_clearsky_dataset,
    multiply_clearsky,
    prepend_t0,
    preprocess_data,
    probabilistic_advection_forecast,
)

NC_VARIABLE_NAMES = {"sds": "sds", "sds_cs": "sds_cs"}


def make_observation_dataset(time_steps, sds, sds_cs):
    """Build a past-observation dataset with a time dimension."""
    return xr.Dataset(
        {
            "sds": (["time", "lat", "lon"], sds),
            "sds_cs": (["time", "lat", "lon"], sds_cs),
        },
        coords={
            "time": [ts.replace(tzinfo=None) for ts in time_steps],
            "lat": np.arange(sds.shape[1], dtype=float),
            "lon": np.arange(sds.shape[2], dtype=float),
        },
    )


class TestPreprocessData:
    @pytest.fixture
    def time_steps(self):
        base = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        return [base + timedelta(minutes=15 * i) for i in range(2)]

    def test_ratio_is_observed_over_clearsky(self, time_steps):
        sds = np.full((2, 3, 4), 400.0)
        sds_cs = np.full((2, 3, 4), 800.0)
        data = make_observation_dataset(time_steps, sds, sds_cs)

        ratio, latitudes, longitudes = preprocess_data(
            data, time_steps, NC_VARIABLE_NAMES
        )

        np.testing.assert_allclose(ratio, np.full((2, 3, 4), 0.5))
        assert ratio.shape == (len(time_steps), 3, 4)
        np.testing.assert_array_equal(latitudes, np.arange(3.0))
        np.testing.assert_array_equal(longitudes, np.arange(4.0))

    def test_time_steps_are_stacked_in_the_requested_order(self, time_steps):
        sds = np.stack([np.full((3, 4), 100.0), np.full((3, 4), 600.0)])
        sds_cs = np.full((2, 3, 4), 1000.0)
        data = make_observation_dataset(time_steps, sds, sds_cs)

        ratio, _, _ = preprocess_data(data, time_steps, NC_VARIABLE_NAMES)

        np.testing.assert_allclose(ratio[0], 0.1)
        np.testing.assert_allclose(ratio[1], 0.6)

    def test_night_time_zero_clearsky_becomes_nan_instead_of_dividing_by_zero(
        self, time_steps
    ):
        # Clear-sky irradiance is zero at night; the ratio is undefined there.
        sds = np.full((2, 3, 4), 0.0)
        sds_cs = np.full((2, 3, 4), 0.0)
        data = make_observation_dataset(time_steps, sds, sds_cs)

        ratio, _, _ = preprocess_data(data, time_steps, NC_VARIABLE_NAMES)

        assert np.isnan(ratio).all()

    def test_negative_clearsky_values_are_also_treated_as_undefined(self, time_steps):
        sds = np.full((2, 3, 4), 400.0)
        sds_cs = np.full((2, 3, 4), -1.0)
        data = make_observation_dataset(time_steps, sds, sds_cs)

        ratio, _, _ = preprocess_data(data, time_steps, NC_VARIABLE_NAMES)

        assert np.isnan(ratio).all()

    def test_mixed_day_and_night_pixels(self, time_steps):
        sds = np.full((2, 3, 4), 400.0)
        sds_cs = np.full((2, 3, 4), 800.0)
        sds_cs[:, 0, 0] = 0.0
        data = make_observation_dataset(time_steps, sds, sds_cs)

        ratio, _, _ = preprocess_data(data, time_steps, NC_VARIABLE_NAMES)

        assert np.isnan(ratio[:, 0, 0]).all()
        np.testing.assert_allclose(ratio[:, 1, 1], 0.5)

    def test_configured_variable_names_are_honoured(self, time_steps):
        data = make_observation_dataset(
            time_steps, np.full((2, 3, 4), 300.0), np.full((2, 3, 4), 600.0)
        ).rename({"sds": "SIS", "sds_cs": "SISc"})

        ratio, _, _ = preprocess_data(data, time_steps, {"sds": "SIS", "sds_cs": "SISc"})

        np.testing.assert_allclose(ratio, 0.5)

    def test_ratios_are_computed_pixel_by_pixel(self, time_steps):
        # Each pixel has its own observed and clearsky value, so a transposed
        # or flattened array would not reproduce the expected result.
        sds = np.array([[[100.0, 200.0], [300.0, 400.0]]] * 2)
        sds_cs = np.array([[[400.0, 400.0], [600.0, 800.0]]] * 2)
        data = make_observation_dataset(time_steps, sds, sds_cs)

        ratio, _, _ = preprocess_data(data, time_steps, NC_VARIABLE_NAMES)

        expected = np.array([[0.25, 0.5], [0.5, 0.5]])
        np.testing.assert_allclose(ratio[0], expected)
        np.testing.assert_allclose(ratio[1], expected)

    def test_non_square_spatial_dimensions_are_preserved_in_order(self, time_steps):
        sds = np.full((2, 3, 7), 400.0)
        sds_cs = np.full((2, 3, 7), 800.0)
        data = make_observation_dataset(time_steps, sds, sds_cs)

        ratio, latitudes, longitudes = preprocess_data(
            data, time_steps, NC_VARIABLE_NAMES
        )

        assert ratio.shape == (2, 3, 7)
        assert len(latitudes) == 3
        assert len(longitudes) == 7

    def test_time_steps_are_selected_by_label_not_by_storage_order(self, time_steps):
        # The dataset is stored in reverse order; selection must still follow
        # the order of the requested time steps.
        sds = np.stack([np.full((3, 4), 100.0), np.full((3, 4), 800.0)])
        sds_cs = np.full((2, 3, 4), 1000.0)
        data = make_observation_dataset(time_steps, sds, sds_cs).isel(
            time=slice(None, None, -1)
        )

        ratio, _, _ = preprocess_data(data, time_steps, NC_VARIABLE_NAMES)

        np.testing.assert_allclose(ratio[0], 0.1)
        np.testing.assert_allclose(ratio[1], 0.8)

    def test_missing_observations_propagate_as_nan(self, time_steps):
        sds = np.full((2, 3, 4), 400.0)
        sds[0, 1, 1] = np.nan
        sds_cs = np.full((2, 3, 4), 800.0)
        data = make_observation_dataset(time_steps, sds, sds_cs)

        ratio, _, _ = preprocess_data(data, time_steps, NC_VARIABLE_NAMES)

        assert np.isnan(ratio[0, 1, 1])
        np.testing.assert_allclose(ratio[0, 0, 0], 0.5)

    def test_a_missing_clearsky_variable_is_reported(self, time_steps):
        data = make_observation_dataset(
            time_steps, np.full((2, 3, 4), 400.0), np.full((2, 3, 4), 800.0)
        ).drop_vars("sds_cs")

        with pytest.raises(KeyError, match="sds_cs"):
            preprocess_data(data, time_steps, NC_VARIABLE_NAMES)

    def test_a_dataset_without_a_time_dimension_is_reported(self, time_steps):
        data = xr.Dataset(
            {
                "sds": (["lat", "lon"], np.full((3, 4), 400.0)),
                "sds_cs": (["lat", "lon"], np.full((3, 4), 800.0)),
            },
            coords={"lat": np.arange(3.0), "lon": np.arange(4.0)},
        )

        with pytest.raises(KeyError, match="time"):
            preprocess_data(data, time_steps, NC_VARIABLE_NAMES)


class TestMultiplyClearsky:
    @pytest.fixture
    def time_steps(self):
        base = datetime(2026, 3, 8, 12, 15, tzinfo=timezone.utc)
        return [base + timedelta(minutes=15 * i) for i in range(3)]

    @pytest.fixture
    def clearsky(self, time_steps, make_clearsky_dataset):
        return make_clearsky_dataset(time_steps, shape=(3, 4))

    def test_deterministic_forecast_gains_an_ensemble_dimension(
        self, clearsky, time_steps
    ):
        ratio_forecast = np.full((3, 3, 4), 0.5)

        solar = multiply_clearsky(ratio_forecast, clearsky, time_steps, NC_VARIABLE_NAMES)

        assert solar.shape == (1, 3, 3, 4)
        np.testing.assert_allclose(solar, 400.0)

    def test_ensemble_forecast_keeps_its_ensemble_dimension(self, clearsky, time_steps):
        ratio_forecast = np.full((5, 3, 3, 4), 0.25)

        solar = multiply_clearsky(ratio_forecast, clearsky, time_steps, NC_VARIABLE_NAMES)

        assert solar.shape == (5, 3, 3, 4)
        np.testing.assert_allclose(solar, 200.0)

    def test_each_step_is_multiplied_by_its_own_clearsky_field(self, time_steps):
        clearsky = xr.Dataset(
            {
                "sds_cs": (
                    ["time", "lat", "lon"],
                    np.stack([np.full((3, 4), v) for v in (100.0, 200.0, 400.0)]),
                )
            },
            coords={"time": [ts.replace(tzinfo=None) for ts in time_steps]},
        )
        ratio_forecast = np.full((3, 3, 4), 0.5)

        solar = multiply_clearsky(ratio_forecast, clearsky, time_steps, NC_VARIABLE_NAMES)

        np.testing.assert_allclose(solar[0, 0], 50.0)
        np.testing.assert_allclose(solar[0, 1], 100.0)
        np.testing.assert_allclose(solar[0, 2], 200.0)

    def test_nan_ratios_propagate(self, clearsky, time_steps):
        ratio_forecast = np.full((3, 3, 4), np.nan)

        solar = multiply_clearsky(ratio_forecast, clearsky, time_steps, NC_VARIABLE_NAMES)

        assert np.isnan(solar).all()

    def test_time_dimension_mismatch_is_rejected_for_deterministic_forecasts(
        self, clearsky, time_steps
    ):
        with pytest.raises(ValueError, match="does not match clearsky timesteps"):
            multiply_clearsky(
                np.full((2, 3, 4), 0.5), clearsky, time_steps, NC_VARIABLE_NAMES
            )

    def test_time_dimension_mismatch_is_rejected_for_ensemble_forecasts(
        self, clearsky, time_steps
    ):
        with pytest.raises(ValueError, match="does not match clearsky timesteps"):
            multiply_clearsky(
                np.full((5, 2, 3, 4), 0.5), clearsky, time_steps, NC_VARIABLE_NAMES
            )

    @pytest.mark.parametrize("shape", [(3,), (3, 4), (2, 3, 4, 5, 6)])
    def test_unsupported_dimensionality_is_rejected(self, clearsky, time_steps, shape):
        with pytest.raises(ValueError, match="must have shape"):
            multiply_clearsky(np.zeros(shape), clearsky, time_steps, NC_VARIABLE_NAMES)

    def test_missing_clearsky_time_step_is_reported(self, clearsky, time_steps):
        extra_step = time_steps[-1] + timedelta(minutes=15)

        with pytest.raises(RuntimeError, match="No clearsky data for 2026-03-08T13:00"):
            multiply_clearsky(
                np.full((4, 3, 4), 0.5),
                clearsky,
                time_steps + [extra_step],
                NC_VARIABLE_NAMES,
            )

    def test_clearsky_varies_per_pixel_as_well_as_per_step(
        self, time_steps, make_clearsky_dataset
    ):
        # A clearsky field that differs in every cell catches broadcasting
        # mistakes that a spatially uniform field would hide.
        values = np.arange(3 * 3 * 4, dtype=float).reshape(3, 3, 4) * 10.0
        clearsky = make_clearsky_dataset(time_steps, shape=(3, 4), values=values)
        ratio_forecast = np.full((3, 3, 4), 0.5)

        solar = multiply_clearsky(ratio_forecast, clearsky, time_steps, NC_VARIABLE_NAMES)

        np.testing.assert_allclose(solar[0], values * 0.5)

    def test_every_ensemble_member_uses_the_same_clearsky_field(
        self, time_steps, make_clearsky_dataset
    ):
        values = np.arange(3 * 3 * 4, dtype=float).reshape(3, 3, 4) * 10.0
        clearsky = make_clearsky_dataset(time_steps, shape=(3, 4), values=values)
        # Distinct ratio per member, so a mis-broadcast member is detectable.
        ratio_forecast = np.stack([np.full((3, 3, 4), 0.2), np.full((3, 3, 4), 0.4)])

        solar = multiply_clearsky(ratio_forecast, clearsky, time_steps, NC_VARIABLE_NAMES)

        np.testing.assert_allclose(solar[0], values * 0.2)
        np.testing.assert_allclose(solar[1], values * 0.4)

    def test_a_zero_clearsky_field_yields_zero_irradiance(
        self, time_steps, make_clearsky_dataset
    ):
        clearsky = make_clearsky_dataset(
            time_steps, shape=(3, 4), values=np.zeros((3, 3, 4))
        )

        solar = multiply_clearsky(
            np.full((3, 3, 4), 0.5), clearsky, time_steps, NC_VARIABLE_NAMES
        )

        np.testing.assert_array_equal(solar, np.zeros((1, 3, 3, 4)))


class TestPrependT0:
    @pytest.fixture
    def t0_time(self):
        return datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def clearsky(self, t0_time, make_clearsky_dataset):
        return make_clearsky_dataset([t0_time], shape=(3, 4))

    def test_analysis_step_is_inserted_at_the_front(self, clearsky, t0_time):
        ratio_data = np.full((4, 3, 4), 0.5)
        solar_forecast = np.full((2, 6, 3, 4), 123.0)
        config = {"nc_variable_names": NC_VARIABLE_NAMES}

        result = prepend_t0(clearsky, ratio_data, solar_forecast, config, t0_time)

        assert result.shape == (2, 7, 3, 4)
        np.testing.assert_allclose(result[:, 0], 400.0)
        np.testing.assert_allclose(result[:, 1:], 123.0)

    def test_the_latest_observed_ratio_is_used_for_t0(self, clearsky, t0_time):
        ratio_data = np.stack(
            [np.full((3, 4), 0.1), np.full((3, 4), 0.2), np.full((3, 4), 0.9)]
        )
        config = {"nc_variable_names": NC_VARIABLE_NAMES}

        result = prepend_t0(clearsky, ratio_data, np.zeros((1, 2, 3, 4)), config, t0_time)

        np.testing.assert_allclose(result[:, 0], 0.9 * 800.0)

    def test_the_analysis_field_is_shared_by_every_ensemble_member(
        self, clearsky, t0_time
    ):
        ratio_data = np.full((2, 3, 4), 0.5)
        solar_forecast = np.random.default_rng(0).random((4, 3, 3, 4))
        config = {"nc_variable_names": NC_VARIABLE_NAMES}

        result = prepend_t0(clearsky, ratio_data, solar_forecast, config, t0_time)

        for member in range(1, 4):
            np.testing.assert_allclose(result[member, 0], result[0, 0])

    def test_spatial_shape_mismatch_is_rejected(self, clearsky, t0_time):
        ratio_data = np.full((2, 5, 6), 0.5)
        config = {"nc_variable_names": NC_VARIABLE_NAMES}

        with pytest.raises(ValueError, match="Spatial shape mismatch"):
            prepend_t0(clearsky, ratio_data, np.zeros((1, 2, 5, 6)), config, t0_time)


class TestComputeEnsembleStatistics:
    @pytest.fixture
    def forecast(self):
        # 11 members holding values 0, 100, ..., 1000 at every grid point, so
        # percentiles are exact and easy to reason about.
        return np.stack([np.full((2, 3, 4), 100.0 * member) for member in range(11)])

    def test_requested_statistics_are_returned(self, forecast):
        computed = compute_ensemble_statistics(forecast, ["median", "mean", "p90"])

        assert set(computed) == {"median", "mean", "p90"}

    def test_statistics_keep_the_ensemble_axis_for_concatenation(self, forecast):
        computed = compute_ensemble_statistics(forecast, ["median"])

        assert computed["median"].shape == (1, 2, 3, 4)

    @pytest.mark.parametrize(
        "statistic,expected",
        [
            ("median", 500.0),
            ("mean", 500.0),
            ("p10", 100.0),
            ("p25", 250.0),
            ("p75", 750.0),
            ("p90", 900.0),
        ],
    )
    def test_values_are_computed_across_members(self, forecast, statistic, expected):
        computed = compute_ensemble_statistics(forecast, [statistic])

        np.testing.assert_allclose(computed[statistic], expected)

    def test_statistics_are_computed_per_grid_point_and_time_step(self):
        rng = np.random.default_rng(0)
        forecast = rng.random((7, 2, 3, 4))

        computed = compute_ensemble_statistics(forecast, ["mean"])

        np.testing.assert_allclose(computed["mean"], forecast.mean(axis=0, keepdims=True))

    def test_a_single_member_reduces_to_itself(self):
        forecast = np.random.default_rng(0).random((1, 2, 3, 4))

        computed = compute_ensemble_statistics(forecast, ["median", "mean", "p10"])

        for values in computed.values():
            np.testing.assert_allclose(values, forecast)

    def test_unsupported_statistic_is_rejected(self, forecast):
        with pytest.raises(ValueError, match="Unsupported ensemble statistic"):
            compute_ensemble_statistics(forecast, ["p99"])

    def test_no_statistics_yields_an_empty_mapping(self, forecast):
        assert compute_ensemble_statistics(forecast, []) == {}

    @pytest.mark.parametrize(
        "statistic,expected",
        [
            # numpy's default linear interpolation places the requested
            # percentile at position q/100 * (n - 1) along the sorted members.
            # With four members 0, 1, 2, 3 that position is:
            #   p10 -> 0.30, p25 -> 0.75, p75 -> 2.25, p90 -> 2.70
            ("p10", 0.30),
            ("p25", 0.75),
            ("p75", 2.25),
            ("p90", 2.70),
            ("median", 1.5),
            ("mean", 1.5),
        ],
    )
    def test_percentiles_use_linear_interpolation_between_members(
        self, statistic, expected
    ):
        # Four members makes every percentile fall between two members, which
        # distinguishes linear interpolation from nearest/lower/higher.
        forecast = np.arange(4, dtype=float).reshape(4, 1, 1, 1)

        computed = compute_ensemble_statistics(forecast, [statistic])

        np.testing.assert_allclose(computed[statistic], expected)

    def test_statistics_reduce_only_the_ensemble_axis(self):
        # Values vary along every axis; only axis 0 may be collapsed.
        forecast = np.arange(3 * 2 * 4 * 5, dtype=float).reshape(3, 2, 4, 5)

        computed = compute_ensemble_statistics(forecast, ["median", "p25"])

        for values in computed.values():
            assert values.shape == (1, 2, 4, 5)
        # Members are spaced by a constant stride, so the median is the middle one.
        np.testing.assert_allclose(computed["median"][0], forecast[1])

    def test_time_and_spatial_dimensions_keep_their_order(self):
        forecast = np.zeros((3, 2, 4, 5))

        computed = compute_ensemble_statistics(forecast, ["mean"])

        assert computed["mean"].shape == (1, 2, 4, 5)

    def test_nan_members_propagate_rather_than_being_ignored(self):
        # np.median/np.mean/np.percentile are not NaN-aware, so a single
        # missing member makes the whole grid point missing.
        forecast = np.array([1.0, np.nan, 3.0]).reshape(3, 1, 1, 1)

        computed = compute_ensemble_statistics(forecast, ["median", "mean", "p10", "p90"])

        for values in computed.values():
            assert np.isnan(values).all()

    def test_a_nan_at_one_grid_point_does_not_affect_its_neighbours(self):
        forecast = np.full((3, 1, 1, 2), 2.0)
        forecast[1, 0, 0, 0] = np.nan

        computed = compute_ensemble_statistics(forecast, ["mean"])

        assert np.isnan(computed["mean"][0, 0, 0, 0])
        assert computed["mean"][0, 0, 0, 1] == 2.0


class TestMakePvlibClearskyDataset:
    @pytest.fixture
    def times(self):
        base = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
        return [base + timedelta(minutes=15 * i) for i in range(2)]

    def test_dataset_shape_and_coordinates(self, times):
        latitudes = np.array([54.0, 55.0, 56.0])
        longitudes = np.array([10.0, 11.0])

        ds = make_pvlib_clearsky_dataset(times, latitudes, longitudes, "sds_cs")

        assert ds["sds_cs"].shape == (2, 3, 2)
        assert ds["sds_cs"].dims == ("time", "latitude", "longitude")
        np.testing.assert_array_equal(ds.latitude.values, latitudes)
        np.testing.assert_array_equal(ds.longitude.values, longitudes)

    def test_time_coordinate_is_stored_without_timezone(self, times):
        ds = make_pvlib_clearsky_dataset(
            times, np.array([55.0]), np.array([12.0]), "sds_cs"
        )

        assert list(ds.time.values) == [
            np.datetime64(t.replace(tzinfo=None)) for t in times
        ]

    def test_variable_name_is_configurable(self, times):
        ds = make_pvlib_clearsky_dataset(
            times, np.array([55.0]), np.array([12.0]), "GHI_clearsky"
        )

        assert "GHI_clearsky" in ds.data_vars

    def test_daytime_clearsky_irradiance_is_physically_plausible(self, times):
        ds = make_pvlib_clearsky_dataset(
            times, np.array([55.7]), np.array([12.6]), "sds_cs"
        )

        values = ds["sds_cs"].values
        assert (values > 400.0).all()
        assert (values < 1200.0).all()

    def test_night_time_clearsky_irradiance_is_zero(self):
        night = [datetime(2026, 12, 21, 0, 0, tzinfo=timezone.utc)]

        ds = make_pvlib_clearsky_dataset(
            night, np.array([55.7]), np.array([12.6]), "sds_cs"
        )

        np.testing.assert_allclose(ds["sds_cs"].values, 0.0)

    def test_longitudes_above_180_are_wrapped(self, times):
        wrapped = make_pvlib_clearsky_dataset(
            times, np.array([55.0]), np.array([347.4]), "sds_cs"
        )
        direct = make_pvlib_clearsky_dataset(
            times, np.array([55.0]), np.array([-12.6]), "sds_cs"
        )

        np.testing.assert_allclose(wrapped["sds_cs"].values, direct["sds_cs"].values)


class TestProbabilisticAdvectionForecast:
    @pytest.fixture
    def ratio_data(self):
        # A moving diagonal band gives the advection model a real signal to carry.
        rows, cols = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
        return np.stack(
            [0.5 + 0.4 * np.sin((rows + cols - 3.0 * step) / 6.0) for step in range(4)]
        ).astype(np.float64)

    @pytest.fixture
    def motion_field(self):
        return np.zeros((2, 32, 32))

    @pytest.fixture
    def moving_motion_field(self):
        return np.full((2, 32, 32), 2.0)

    def test_deterministic_forecast_shape(self, ratio_data, motion_field):
        forecast = probabilistic_advection_forecast(
            ratio_data, motion_field, n_steps=3, ens_members=1, alpha=0.0, beta=0.0
        )

        assert forecast.shape[-3:] == (3, 32, 32)

    def test_ensemble_forecast_has_a_member_dimension(self, ratio_data, motion_field):
        forecast = probabilistic_advection_forecast(
            ratio_data, motion_field, n_steps=3, ens_members=4, alpha=9.23, beta=0.15
        )

        assert forecast.shape == (4, 3, 32, 32)

    def test_a_zero_motion_field_persists_the_latest_observation(
        self, ratio_data, motion_field
    ):
        forecast = probabilistic_advection_forecast(
            ratio_data, motion_field, n_steps=2, ens_members=1, alpha=0.0, beta=0.0
        )

        np.testing.assert_allclose(np.squeeze(forecast)[0], ratio_data[-1], atol=1e-6)

    def test_noise_free_ensemble_members_are_identical(self, ratio_data, motion_field):
        forecast = probabilistic_advection_forecast(
            ratio_data, motion_field, n_steps=2, ens_members=3, alpha=0.0, beta=0.0
        )

        np.testing.assert_allclose(forecast[1], forecast[0], atol=1e-6)
        np.testing.assert_allclose(forecast[2], forecast[0], atol=1e-6)

    def test_no_infinite_values_are_produced(self, ratio_data, moving_motion_field):
        forecast = probabilistic_advection_forecast(
            ratio_data, moving_motion_field, n_steps=3, ens_members=1, alpha=0.0, beta=0.0
        )

        assert not np.isinf(forecast).any()

    def test_a_zero_motion_field_introduces_no_missing_values(
        self, ratio_data, motion_field
    ):
        # With no advection nothing is carried in from outside the grid, so a
        # NaN here would mean the model invented one.
        forecast = probabilistic_advection_forecast(
            ratio_data, motion_field, n_steps=3, ens_members=1, alpha=0.0, beta=0.0
        )

        assert not np.isnan(forecast).any()

    def test_advection_only_loses_data_at_the_inflow_edge(
        self, ratio_data, moving_motion_field
    ):
        # Cells advected in from beyond the domain are unknown, so NaNs are
        # expected to grow with lead time but never fill the grid.
        forecast = probabilistic_advection_forecast(
            ratio_data, moving_motion_field, n_steps=3, ens_members=1, alpha=0.0, beta=0.0
        )

        missing_per_step = [int(np.isnan(step).sum()) for step in forecast]
        assert missing_per_step[0] > 0
        assert missing_per_step == sorted(missing_per_step)
        assert missing_per_step[-1] < forecast[0].size


class TestEnsembleReproducibility:
    """The model seeds each member with its own index (see SolarSTEPS'
    ProbabilisticAdvection.maps_forecast), so forecasts are reproducible
    without the test suite touching the global random state."""

    @pytest.fixture
    def ratio_data(self):
        rows, cols = np.meshgrid(np.arange(24), np.arange(24), indexing="ij")
        return np.stack(
            [0.5 + 0.4 * np.sin((rows + cols - 3.0 * step) / 6.0) for step in range(4)]
        ).astype(np.float64)

    @pytest.fixture
    def motion_field(self):
        return np.full((2, 24, 24), 0.5)

    def forecast(self, ratio_data, motion_field, ens_members=4):
        return probabilistic_advection_forecast(
            ratio_data,
            motion_field,
            n_steps=3,
            ens_members=ens_members,
            alpha=9.23,
            beta=0.15,
        )

    def test_repeated_runs_produce_identical_ensembles(self, ratio_data, motion_field):
        first = self.forecast(ratio_data, motion_field)
        second = self.forecast(ratio_data, motion_field)

        np.testing.assert_array_equal(first, second)

    def test_the_result_does_not_depend_on_the_global_random_state(
        self, ratio_data, motion_field
    ):
        baseline = self.forecast(ratio_data, motion_field)

        np.random.seed(12345)
        after_seeding = self.forecast(ratio_data, motion_field)

        np.testing.assert_array_equal(baseline, after_seeding)

    def test_deterministic_runs_are_reproducible(self, ratio_data, motion_field):
        first = probabilistic_advection_forecast(
            ratio_data, motion_field, n_steps=3, ens_members=1, alpha=0.0, beta=0.0
        )
        second = probabilistic_advection_forecast(
            ratio_data, motion_field, n_steps=3, ens_members=1, alpha=0.0, beta=0.0
        )

        np.testing.assert_array_equal(first, second)

    def test_every_requested_member_is_generated(self, ratio_data, motion_field):
        forecast = self.forecast(ratio_data, motion_field, ens_members=6)

        assert forecast.shape == (6, 3, 24, 24)

    def test_members_seeded_differently_give_different_forecasts(
        self, ratio_data, motion_field
    ):
        forecast = self.forecast(ratio_data, motion_field)

        for member in range(1, forecast.shape[0]):
            assert not np.allclose(forecast[member], forecast[0], equal_nan=True)

    def test_perturbed_members_stay_finite(self, ratio_data, motion_field):
        forecast = self.forecast(ratio_data, motion_field)

        assert not np.isinf(forecast).any()
        assert np.isfinite(forecast).any()
