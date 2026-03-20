#!/usr/bin/env python3
from datetime import datetime

import numpy as np
import pvlib
import xarray as xr

from .config import BBOX_OPTIONS


def get_bbox(bbox_choice: str, custom_bbox: str | None = None) -> str | None:
    """Return the bounding box string for a given bbox choice.

    For predefined choices ('DENMARK', 'NW_EUROPE'), looks up
    the value in BBOX_OPTIONS. For 'CUSTOM', returns custom_bbox
    directly.

    Args:
        bbox_choice: One of 'DENMARK', 'NW_EUROPE', or 'CUSTOM'.
        custom_bbox: Bbox string lon_min,lat_min,lon_max,lat_max used
            when bbox_choice='CUSTOM'. Default None.

    Returns:
        Bounding box string, or None if the predefined choice has no
        associated value.
    """
    return (
        custom_bbox if bbox_choice == "CUSTOM" else BBOX_OPTIONS[bbox_choice]
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
