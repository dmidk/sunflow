#!/usr/bin/env python3
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from sunflow.geospatial import (
    check_solar_elevation,
    crop_forecast_to_domain,
    domain_contains,
    get_coordinates,
    infer_coordinate_edges,
    parse_bbox,
    resolve_domain_bbox,
    subset_to_bbox,
    validate_dataset_covers_domain,
)


def make_dataset(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.Dataset:
    """Build a minimal gridded dataset with one variable over lat/lon."""
    data = np.arange(len(latitudes) * len(longitudes), dtype=float).reshape(
        len(latitudes), len(longitudes)
    )
    return xr.Dataset(
        {"sds": ([lat_name, lon_name], data)},
        coords={lat_name: latitudes, lon_name: longitudes},
    )


class TestParseBbox:
    def test_returns_bounds_in_order(self):
        assert parse_bbox("4,50,18,62") == (4.0, 50.0, 18.0, 62.0)

    def test_accepts_negative_and_fractional_bounds(self):
        assert parse_bbox("-10.75,47.25,20,63.5") == (-10.75, 47.25, 20.0, 63.5)

    @pytest.mark.parametrize("bbox", ["1,2,3", "1,2,3,4,5", ""])
    def test_wrong_number_of_values_is_rejected(self, bbox):
        with pytest.raises(ValueError, match="exactly 4"):
            parse_bbox(bbox)

    def test_non_numeric_values_are_rejected(self):
        with pytest.raises(ValueError):
            parse_bbox("a,b,c,d")

    def test_longitudes_must_increase(self):
        with pytest.raises(ValueError, match="lon_min must be smaller"):
            parse_bbox("18,50,4,62")

    def test_latitudes_must_increase(self):
        with pytest.raises(ValueError, match="lat_min must be smaller"):
            parse_bbox("4,62,18,50")

    def test_degenerate_bounds_are_rejected(self):
        with pytest.raises(ValueError):
            parse_bbox("4,50,4,62")


class TestResolveDomainBbox:
    @pytest.mark.parametrize(
        "domain_choice,expected",
        [
            ("DENMARK", "4,50,18,62"),
            ("NW_EUROPE", "-10.75,47.25,20,63.5"),
            ("NW_EUROPE_SATELLITE", "-20.75,37.25,30,73.5"),
        ],
    )
    def test_predefined_domains(self, domain_choice, expected):
        assert resolve_domain_bbox(domain_choice) == expected

    def test_custom_domain_is_returned_verbatim(self):
        assert resolve_domain_bbox("CUSTOM", "1,2,3,4") == "1,2,3,4"

    def test_custom_without_a_value_returns_none(self):
        assert resolve_domain_bbox("CUSTOM") is None

    def test_custom_value_is_ignored_for_predefined_domains(self):
        assert resolve_domain_bbox("DENMARK", "1,2,3,4") == "4,50,18,62"

    def test_unknown_domain_raises(self):
        with pytest.raises(KeyError):
            resolve_domain_bbox("ATLANTIS")


class TestDomainContains:
    def test_strictly_larger_domain_contains_smaller_one(self):
        assert domain_contains("-20.75,37.25,30,73.5", "4,50,18,62") is True

    def test_identical_domains_contain_each_other(self):
        assert domain_contains("4,50,18,62", "4,50,18,62") is True

    def test_smaller_domain_does_not_contain_larger_one(self):
        assert domain_contains("4,50,18,62", "-20.75,37.25,30,73.5") is False

    @pytest.mark.parametrize(
        "inner",
        [
            "3,50,18,62",  # west of outer
            "4,49,18,62",  # south of outer
            "4,50,19,62",  # east of outer
            "4,50,18,63",  # north of outer
        ],
    )
    def test_overhang_in_any_direction_is_not_contained(self, inner):
        assert domain_contains("4,50,18,62", inner) is False


class TestGetCoordinates:
    @pytest.mark.parametrize(
        "lat_name,lon_name",
        [("y", "x"), ("lat", "lon"), ("latitude", "longitude")],
    )
    def test_supported_coordinate_namings(self, lat_name, lon_name):
        latitudes = np.array([50.0, 51.0, 52.0])
        longitudes = np.array([4.0, 5.0])
        ds = make_dataset(latitudes, longitudes, lat_name, lon_name)

        found_lats, found_lons = get_coordinates(ds)

        np.testing.assert_array_equal(found_lats, latitudes)
        np.testing.assert_array_equal(found_lons, longitudes)

    def test_unknown_coordinate_names_raise(self):
        ds = make_dataset(np.array([1.0, 2.0]), np.array([3.0, 4.0]), "row", "col")

        with pytest.raises(RuntimeError, match="Could not find coordinates"):
            get_coordinates(ds)

    def test_empty_coordinates_raise(self):
        ds = make_dataset(np.array([]), np.array([3.0, 4.0]))

        with pytest.raises(RuntimeError, match="cannot be empty"):
            get_coordinates(ds)


class TestInferCoordinateEdges:
    def test_edges_extend_half_a_grid_step_beyond_the_centres(self):
        coords = np.array([50.0, 51.0, 52.0])

        assert infer_coordinate_edges(coords) == (49.5, 52.5)

    def test_descending_coordinates_give_the_same_edges(self):
        coords = np.array([52.0, 51.0, 50.0])

        assert infer_coordinate_edges(coords) == (49.5, 52.5)

    def test_single_coordinate_has_no_resolvable_spacing(self):
        assert infer_coordinate_edges(np.array([50.0])) == (50.0, 50.0)

    def test_irregular_spacing_uses_the_median_step(self):
        coords = np.array([0.0, 1.0, 2.0, 10.0])

        edge_min, edge_max = infer_coordinate_edges(coords)

        assert edge_min == pytest.approx(-0.5)
        assert edge_max == pytest.approx(10.5)

    def test_empty_coordinates_raise(self):
        with pytest.raises(RuntimeError, match="cannot be empty"):
            infer_coordinate_edges(np.array([]))


class TestSubsetToBbox:
    def test_subsets_ascending_coordinates(self):
        ds = make_dataset(np.arange(45.0, 65.0), np.arange(-5.0, 15.0))

        subset = subset_to_bbox(ds, "0,50,5,55")

        assert subset.lat.min() >= 49.0
        assert subset.lat.max() <= 56.0
        assert subset.lon.min() >= -1.0
        assert subset.lon.max() <= 6.0

    def test_edge_cells_are_kept(self):
        # Centres at 50.0 and 55.0 lie exactly on the bbox edges and must survive.
        ds = make_dataset(np.arange(45.0, 65.0), np.arange(-5.0, 15.0))

        subset = subset_to_bbox(ds, "0,50,5,55")

        assert 50.0 in subset.lat.values
        assert 55.0 in subset.lat.values
        assert 0.0 in subset.lon.values
        assert 5.0 in subset.lon.values

    def test_descending_latitudes_are_handled(self):
        ds = make_dataset(np.arange(64.0, 44.0, -1.0), np.arange(-5.0, 15.0))

        subset = subset_to_bbox(ds, "0,50,5,55")

        assert subset.sizes["lat"] > 0
        assert subset.lat.min() >= 49.0
        assert subset.lat.max() <= 56.0

    def test_dataset_is_returned_unchanged_when_coordinates_are_unrecognised(self):
        ds = make_dataset(np.array([1.0, 2.0]), np.array([3.0, 4.0]), "row", "col")

        subset = subset_to_bbox(ds, "0,50,5,55")

        assert subset is ds


class TestValidateDatasetCoversDomain:
    def test_passes_when_the_grid_covers_the_domain(self):
        ds = make_dataset(np.arange(45.0, 65.0), np.arange(-5.0, 15.0))

        validate_dataset_covers_domain(ds, "0,50,5,55", "Input file")

    def test_passes_when_the_domain_lands_exactly_on_the_inferred_edges(self):
        # Centres 50..52 with 1 degree spacing infer edges at 49.5 and 52.5.
        ds = make_dataset(np.array([50.0, 51.0, 52.0]), np.array([4.0, 5.0, 6.0]))

        validate_dataset_covers_domain(ds, "3.5,49.5,6.5,52.5", "Input file")

    def test_fails_when_the_domain_extends_beyond_the_grid(self):
        ds = make_dataset(np.arange(50.0, 55.0), np.arange(0.0, 5.0))

        with pytest.raises(RuntimeError, match="does not cover requested"):
            validate_dataset_covers_domain(ds, "-10,40,20,60", "Input file")

    def test_error_message_names_the_context_and_bounds(self):
        ds = make_dataset(np.arange(50.0, 55.0), np.arange(0.0, 5.0))

        with pytest.raises(RuntimeError, match="Input file /data/x.nc"):
            validate_dataset_covers_domain(ds, "-10,40,20,60", "Input file /data/x.nc")


class TestCropForecastToDomain:
    def test_crops_a_time_lat_lon_array(self):
        latitudes = np.arange(45.0, 65.0)
        longitudes = np.arange(-5.0, 15.0)
        forecast = np.random.default_rng(0).random((3, len(latitudes), len(longitudes)))

        cropped, lats, lons = crop_forecast_to_domain(
            forecast, latitudes, longitudes, "0,50,5,55"
        )

        assert cropped.shape == (3, len(lats), len(lons))
        assert lats.min() >= 49.0 and lats.max() <= 56.0
        assert lons.min() >= -1.0 and lons.max() <= 6.0

    def test_crops_an_ensemble_array_without_touching_leading_dimensions(self):
        latitudes = np.arange(45.0, 65.0)
        longitudes = np.arange(-5.0, 15.0)
        forecast = np.random.default_rng(0).random(
            (5, 3, len(latitudes), len(longitudes))
        )

        cropped, lats, lons = crop_forecast_to_domain(
            forecast, latitudes, longitudes, "0,50,5,55"
        )

        assert cropped.shape == (5, 3, len(lats), len(lons))

    def test_cropped_values_match_the_original_grid(self):
        latitudes = np.arange(50.0, 56.0)
        longitudes = np.arange(0.0, 6.0)
        forecast = np.arange(
            1 * len(latitudes) * len(longitudes), dtype=float
        ).reshape(1, len(latitudes), len(longitudes))

        cropped, lats, lons = crop_forecast_to_domain(
            forecast, latitudes, longitudes, "1,51,3,53"
        )

        lat_idx = [int(np.where(latitudes == lat)[0][0]) for lat in lats]
        lon_idx = [int(np.where(longitudes == lon)[0][0]) for lon in lons]
        expected = forecast[:, lat_idx, :][:, :, lon_idx]
        np.testing.assert_array_equal(cropped, expected)

    @pytest.mark.parametrize("shape", [(10,), (4, 10), (2, 3, 4, 5, 6)])
    def test_unsupported_dimensionality_is_rejected(self, shape):
        with pytest.raises(RuntimeError, match="Expected forecast shape"):
            crop_forecast_to_domain(
                np.zeros(shape), np.arange(10.0), np.arange(10.0), "0,0.5,1,1.5"
            )

    def test_domain_outside_the_grid_is_rejected(self):
        latitudes = np.arange(50.0, 55.0)
        longitudes = np.arange(0.0, 5.0)

        with pytest.raises(RuntimeError, match="does not overlap forecast grid"):
            crop_forecast_to_domain(
                np.zeros((2, len(latitudes), len(longitudes))),
                latitudes,
                longitudes,
                "100,80,110,85",
            )


class TestCheckSolarElevation:
    def test_sun_is_high_at_local_noon_in_midsummer(self):
        elevation = check_solar_elevation(
            datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc), lat=55.7, lon=12.6
        )

        assert elevation > 50.0

    def test_sun_is_below_the_horizon_at_local_midnight(self):
        elevation = check_solar_elevation(
            datetime(2026, 12, 21, 0, 0, tzinfo=timezone.utc), lat=55.7, lon=12.6
        )

        assert elevation < 0.0

    def test_longitudes_above_180_are_wrapped(self):
        time = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)

        wrapped = check_solar_elevation(time, lat=55.7, lon=347.4)
        direct = check_solar_elevation(time, lat=55.7, lon=-12.6)

        assert wrapped == pytest.approx(direct)
