#!/usr/bin/env python3
from datetime import datetime

import numpy as np
import xarray as xr
from Models.ProbabilisticAdvection import ProbabilisticAdvection
from loguru import logger

from .geospatial import get_coordinates


def preprocess_data(
    data: xr.Dataset,
    time_steps: list[datetime],
    nc_variable_names: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract solar irradiance ratio arrays and grid coordinates from input data.

    For each timestep, computes the ratio SDS / SDS_CS (observed surface
    downwelling solar radiation divided by its clear-sky equivalent). The
    result is a 3-D array of shape (time, lat, lon) suitable as input
    to the advection model. Coordinate arrays are extracted from the last
    processed dataset, supporting both (y, x) and (lat, lon) naming.

    Args:
        data: xarray Dataset with a 'time' dimension containing the SDS
            and SDS_CS variables for all past timesteps.
        time_steps: Ordered list of datetimes corresponding to entries in data.
        nc_variable_names: Dictionary with keys 'sds' and 'sds_cs' mapping
            to the NetCDF variable names in the datasets.

    Returns:
        Tuple of (ratio_array, latitudes, longitudes) where
        ratio_array has shape (len(time_steps), n_lat, n_lon) and
        the coordinate arrays are 1-D.

    Raises:
        ValueError: If latitude/longitude coordinates cannot be found.
    """
    ratio_arrays = []

    for time_step in time_steps:
        ds = data.sel(time=time_step.replace(tzinfo=None))

        sds = ds[nc_variable_names["sds"]].values
        sds_cs = ds[nc_variable_names["sds_cs"]].values

        ratio = np.divide(
            sds,
            sds_cs,
            out=np.full_like(sds, np.nan, dtype=float),
            where=sds_cs > 0,
        )
        ratio_arrays.append(ratio)

    latitudes, longitudes = get_coordinates(ds)

    return (
        np.stack(ratio_arrays),
        latitudes,
        longitudes,
    )


def simple_advection_forecast(
    ratio_data: np.ndarray, motion_field: np.ndarray, n_steps: int, ens_members: int
) -> np.ndarray:
    """Run a deterministic advection forecast on solar irradiance ratios.

    Uses ProbabilisticAdvection with noise parameters alpha=0 and beta=0,
    which disables Gaussian noise on the motion field norm and
    von Mises noise on the direction, yielding a purely deterministic
    advection result. The ensemble dimension added by the model is removed
    before returning.

    Args:
        ratio_data: Input array of shape (time, lat, lon) containing
            SDS/SDS_CS ratios for the past timesteps.
        motion_field: Optical flow field of shape (2, lat, lon) as
            produced by dense_lucaskanade.
        n_steps: Number of forecast timesteps to produce.

    Returns:
        Forecast array of shape (n_steps, lat, lon).
    """

    # Initialize ProbabilisticAdvection with NO noise (alpha=0, beta=0)
    pa = ProbabilisticAdvection(
        alpha=0.0,  # No Gaussian noise on motion field norm
        beta=0.0,  # No von Mises noise on motion field angle
        return_motion_field=False,
        ens_members=ens_members,
    )
    # Run probabilistic advection using the correct method name
    forecast = pa.maps_forecast(n_steps, ratio_data, motion_field)

    return forecast


def multiply_clearsky(
    ratio_forecast: np.ndarray,
    clearsky_data: xr.Dataset,
    previous_day_time_steps: list[datetime],
    nc_variable_names: dict[str, str],
) -> np.ndarray:
    """Convert a ratio forecast to solar irradiance by multiplying by clearsky values.

    Each forecast step is multiplied element-wise by the corresponding
    clearsky irradiance value from clearsky_data, converting the
    dimensionless SDS/SDS_CS ratio forecast into actual surface downwelling
    solar radiation in W m⁻².

    Args:
        ratio_forecast: Forecast array of shape (n_steps, lat, lon)
            or (ensemble, n_steps, lat, lon) containing SDS/SDS_CS ratios.
        clearsky_data: xarray Dataset with a 'time' dimension containing
            the clearsky variable for each forecast step.
        previous_day_time_steps: List of datetimes (one per forecast step)
            used to select timesteps from clearsky_data.
        nc_variable_names: Dictionary with key 'sds_cs' mapping to the clearsky
            NetCDF variable name in the datasets.

    Returns:
        Solar irradiance forecast array with the same shape as
        ratio_forecast, in W m⁻².

    Raises:
        RuntimeError: If clearsky data is missing for any forecast timestep.
    """
    clearsky_steps: list[np.ndarray] = []

    for time_step in previous_day_time_steps:
        try:
            sds_cs = clearsky_data.sel(time=time_step.replace(tzinfo=None))[
                nc_variable_names["sds_cs"]
            ].values
            clearsky_steps.append(sds_cs)
        except KeyError:
            raise RuntimeError(
                f"No clearsky data for {time_step.strftime('%Y-%m-%dT%H:%M:%SZ')}, "
                "cannot compute solar forecast for this step."
            )

    clearsky_stack = np.stack(clearsky_steps, axis=0)

    if ratio_forecast.ndim == 3:
        if ratio_forecast.shape[0] != clearsky_stack.shape[0]:
            raise ValueError(
                "ratio_forecast time dimension does not match clearsky timesteps "
                f"({ratio_forecast.shape[0]} != {clearsky_stack.shape[0]})."
            )
        return ratio_forecast * clearsky_stack

    if ratio_forecast.ndim == 4:
        if ratio_forecast.shape[1] != clearsky_stack.shape[0]:
            raise ValueError(
                "ratio_forecast time dimension does not match clearsky timesteps "
                f"({ratio_forecast.shape[1]} != {clearsky_stack.shape[0]})."
            )
        return ratio_forecast * clearsky_stack[np.newaxis, :, :, :]

    raise ValueError(
        "ratio_forecast must have shape (time, lat, lon) or "
        "(ensemble, time, lat, lon)."
    )


def prepend_t0(clearsky_data: xr.Dataset, ratio_data: np.ndarray, solar_forecast: np.ndarray, config: dict, clearsky_t0_time: datetime) -> np.ndarray:
    # Prepend timestep 0: current observation (ratio_data[-1]) × clearsky at t=0
    sds_cs_t0 = clearsky_data.sel(time=clearsky_t0_time.replace(tzinfo=None))[
        config["nc_variable_names"]["sds_cs"]
    ].values
    solar_t0 = ratio_data[-1] * sds_cs_t0

    if solar_forecast.ndim == 3:
        solar_forecast = np.concatenate(
            [solar_t0[np.newaxis, :, :], solar_forecast],
            axis=0,
        )
    elif solar_forecast.ndim == 4:
        # Broadcast the same t=0 clearsky-based analysis field to all ensembles.
        solar_t0_ens = np.broadcast_to(
            solar_t0,
            (solar_forecast.shape[0],) + solar_t0.shape,
        )
        solar_forecast = np.concatenate(
            [solar_t0_ens[:, np.newaxis, :, :], solar_forecast],
            axis=1,
        )
    else:
        raise ValueError(
            "solar_forecast must have shape (time, lat, lon) or "
            "(ensemble, time, lat, lon)."
        )
    return solar_forecast
