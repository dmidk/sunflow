#!/usr/bin/env python3
from datetime import datetime

import numpy as np
import pvlib
import xarray as xr

from .config import DOMAIN_OPTIONS

# Tiny absolute tolerance for floating-point boundary comparisons in degrees.
COVERAGE_ABS_TOL_DEGREES = 1e-9


def resolve_domain_bbox(
    domain_choice: str,
    custom_domain: str | None = None,
) -> str | None:
    """Return the bbox string for a given domain choice.

    For predefined choices ('DENMARK', 'NW_EUROPE'), looks up the value in
    DOMAIN_OPTIONS. For 'CUSTOM', returns custom_domain directly.

    Args:
        domain_choice: One of 'DENMARK', 'NW_EUROPE', or 'CUSTOM'.
        custom_domain: Bbox string lon_min,lat_min,lon_max,lat_max used
            when domain_choice='CUSTOM'.

    Returns:
        Bounding box string, or None if the predefined choice has no
        associated value.
    """
    return custom_domain if domain_choice == "CUSTOM" else DOMAIN_OPTIONS[domain_choice]


def parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    """Parse and validate a bbox string.

    Args:
        bbox: Bounding box string in format lon_min,lat_min,lon_max,lat_max.

    Returns:
        Tuple (lon_min, lat_min, lon_max, lat_max).

    Raises:
        ValueError: If the format is invalid or bounds are not ordered.
    """
    parts = bbox.split(",")
    if len(parts) != 4:
        raise ValueError("Must have exactly 4 comma-separated values")

    lon_min, lat_min, lon_max, lat_max = [float(x) for x in parts]

    if lon_min >= lon_max:
        raise ValueError("lon_min must be smaller than lon_max")
    if lat_min >= lat_max:
        raise ValueError("lat_min must be smaller than lat_max")

    return lon_min, lat_min, lon_max, lat_max


def domain_contains(outer_bbox: str, inner_bbox: str) -> bool:
    """Return True if outer_bbox fully contains inner_bbox."""
    outer_lon_min, outer_lat_min, outer_lon_max, outer_lat_max = parse_bbox(outer_bbox)
    inner_lon_min, inner_lat_min, inner_lon_max, inner_lat_max = parse_bbox(inner_bbox)

    return (
        outer_lon_min <= inner_lon_min
        and outer_lat_min <= inner_lat_min
        and outer_lon_max >= inner_lon_max
        and outer_lat_max >= inner_lat_max
    )


def get_coordinates(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Extract latitude and longitude coordinate arrays from an xarray Dataset.

    Supports both (y, x) and (lat, lon) coordinate naming conventions.

    Args:
        ds: xarray Dataset containing the coordinates.

    Returns:
        Tuple of (latitudes, longitudes) as 1-D numpy arrays.

    Raises:
        RuntimeError: If neither (y, x) nor (lat, lon) coordinates are found,
            or if either resulting array is empty.
    """
    if "y" in ds.coords and "x" in ds.coords:
        latitudes = ds.y.values
        longitudes = ds.x.values
    elif "lat" in ds.coords and "lon" in ds.coords:
        latitudes = ds.lat.values
        longitudes = ds.lon.values
    else:
        raise RuntimeError("Could not find coordinates in dataset.")

    if len(latitudes) == 0 or len(longitudes) == 0:
        raise RuntimeError(
            f"Coordinate arrays cannot be empty: lat={len(latitudes)},"
            f" lon={len(longitudes)}."
        )

    return latitudes, longitudes


def infer_coordinate_edges(coords: np.ndarray) -> tuple[float, float]:
    """Infer min/max coordinate edges from coordinate center values.

    For regularly gridded coordinates represented by cell centers, this
    returns the outer cell edges by extending half a grid step beyond the
    min/max center values.

    Args:
        coords: 1-D coordinate center values.

    Returns:
        Tuple (edge_min, edge_max).

    Raises:
        RuntimeError: If coordinates are empty.
    """
    if len(coords) == 0:
        raise RuntimeError("Coordinate array cannot be empty.")

    coord_min = float(np.min(coords))
    coord_max = float(np.max(coords))

    # A single coordinate has no resolvable spacing, so edge == center.
    if len(coords) < 2:
        return coord_min, coord_max

    # Use median absolute spacing to remain robust to ascending/descending order.
    spacing = float(np.median(np.abs(np.diff(coords))))
    half_step = 0.5 * spacing
    return coord_min - half_step, coord_max + half_step


def validate_dataset_covers_domain(
    ds: xr.Dataset,
    domain_bbox: str,
    context: str,
) -> None:
    """Validate that dataset coordinates fully cover a requested domain.

    Args:
        ds: Input dataset containing latitude and longitude coordinates.
        domain_bbox: Requested bbox string lon_min,lat_min,lon_max,lat_max.
        context: Human-readable context for error messages.

    Raises:
        RuntimeError: If dataset bounds do not fully contain requested domain.
    """
    latitudes, longitudes = get_coordinates(ds)
    lon_min, lat_min, lon_max, lat_max = parse_bbox(domain_bbox)

    data_lon_min, data_lon_max = infer_coordinate_edges(longitudes)
    data_lat_min, data_lat_max = infer_coordinate_edges(latitudes)

    if (
        data_lon_min > lon_min + COVERAGE_ABS_TOL_DEGREES
        or data_lat_min > lat_min + COVERAGE_ABS_TOL_DEGREES
        or data_lon_max < lon_max - COVERAGE_ABS_TOL_DEGREES
        or data_lat_max < lat_max - COVERAGE_ABS_TOL_DEGREES
    ):
        raise RuntimeError(
            f"{context} does not cover requested domain_satellite={domain_bbox}. "
            f"Available bounds are "
            f"{data_lon_min},{data_lat_min},{data_lon_max},{data_lat_max}."
        )


def crop_forecast_to_domain(
    forecast: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    domain_bbox: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop a [time, lat, lon] forecast and its coordinates to a domain bbox.

    Args:
        forecast: Forecast array with shape [time, lat, lon].
        latitudes: 1-D latitude array.
        longitudes: 1-D longitude array.
        domain_bbox: Requested bbox string lon_min,lat_min,lon_max,lat_max.

    Returns:
        Tuple (cropped_forecast, cropped_latitudes, cropped_longitudes).

    Raises:
        RuntimeError: If forecast dimensionality is not [time, lat, lon] or the
            requested domain has no overlap with the provided coordinates.
    """
    if forecast.ndim != 3:
        raise RuntimeError(
            f"Expected forecast shape [time, lat, lon], got {forecast.shape}."
        )

    lon_min, lat_min, lon_max, lat_max = parse_bbox(domain_bbox)
    lat_idx = np.where((latitudes >= lat_min) & (latitudes <= lat_max))[0]
    lon_idx = np.where((longitudes >= lon_min) & (longitudes <= lon_max))[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise RuntimeError(
            f"Requested domain_nowcast={domain_bbox} does not overlap forecast grid."
        )

    cropped_forecast = forecast[:, lat_idx, :][:, :, lon_idx]
    return cropped_forecast, latitudes[lat_idx], longitudes[lon_idx]


def check_solar_elevation(
    time: datetime,
    lat: float = 55.6761,
    lon: float = 12.5683,
) -> float:
    """Compute the solar elevation angle at a given time and location.

    Uses pvlib to calculate the solar position. Defaults to Copenhagen,
    Denmark (55.68°N, 12.57°E).

    Args:
        time: Datetime (timezone-aware) for which to compute the elevation.
        lat: Latitude in decimal degrees. Default 55.6761 (Copenhagen).
        lon: Longitude in decimal degrees. Default 12.5683 (Copenhagen).

    Returns:
        Solar elevation angle in degrees above the horizon. Negative values
        indicate the sun is below the horizon.
    """
    location = pvlib.location.Location(lat, lon)
    solar_elevation = location.get_solarposition(time)["elevation"].values[0]
    return solar_elevation
