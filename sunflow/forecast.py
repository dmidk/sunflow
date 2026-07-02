#!/usr/bin/env python3
from datetime import datetime

import numpy as np
import xarray as xr
from loguru import logger
from Models.ProbabilisticAdvection import ProbabilisticAdvection
from Models.SolarSTEPS import SolarSTEPS
from pysteps.cascade.bandpass_filters import _gaussweights_1d

from .config import TARGET_SMALLEST_RESOLUTION
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


def probabilistic_advection_forecast(
    ratio_data: np.ndarray,
    motion_field: np.ndarray,
    n_steps: int,
    ens_members: int,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Run a probabilistic advection forecast on solar irradiance ratios.

    Uses ProbabilisticAdvection with configurable noise parameters:
    alpha controls Gaussian noise on motion field norm and beta controls
    von Mises noise on motion field direction.

    Args:
        ratio_data: Input array of shape (time, lat, lon) containing
            SDS/SDS_CS ratios for the past timesteps.
        motion_field: Optical flow field of shape (2, lat, lon) as
            produced by dense_lucaskanade.
        n_steps: Number of forecast timesteps to produce.
        ens_members: Number of ensemble members.
        alpha: Gaussian noise strength on motion field norm.
        beta: von Mises noise strength on motion field angle.

    Returns:
        Forecast array of shape (n_steps, lat, lon).
    """

    # Initialize ProbabilisticAdvection with configured noise settings.
    pa = ProbabilisticAdvection(
        alpha=alpha,
        beta=beta,
        return_motion_field=False,
        ens_members=ens_members,
    )
    # Run probabilistic advection using the correct method name
    forecast = pa.maps_forecast(n_steps, ratio_data, motion_field)

    return forecast


def solarsteps_forecast(
    ratio_data: np.ndarray,
    motion_field: np.ndarray,
    n_steps: int,
    ens_members: int,
    noise_win_size: float,
    noise_std_win_size: float,
) -> np.ndarray:
    """Run a SolarSTEPS ensemble forecast on solar irradiance ratios.

    Configures SolarSTEPS with fixed stochastic-noise and normalization
    options, determines the number of cascade levels from the input grid
    size, and generates an ensemble forecast for the requested lead times.

    Args:
        ratio_data: Input array of shape (time, lat, lon) containing
            SDS/SDS_CS ratios for the past timesteps.
        motion_field: Optical flow field of shape (2, lat, lon) as
            produced by dense_lucaskanade.
        n_steps: Number of forecast timesteps to produce.
        ens_members: Number of ensemble members.


    Returns:
        Forecast array of shape (ensemble, n_steps, lat, lon).
    """

    n_cascade_levels = determine_cascade_levels(ratio_data)
    solarsteps = SolarSTEPS(
        ar_order=1,
        n_cascade_levels=n_cascade_levels,
        probmatching=True,
        norm=True,
        local=False,
        noise_kwargs={
            "noise_win_size": noise_win_size,
            "noise_std_win_size": noise_std_win_size,
            "noise_method": "local-SSFT",
        },
        norm_kwargs={"extra_normalization": True},
        verbose=False,
    )
    forecast = solarsteps.ensemble_forecast(
        ratio_data, motion_field, n_steps, seeds=np.arange(ens_members)
    )

    return forecast


def determine_cascade_levels(ratio_data: np.ndarray) -> int:
    """Determine the number of cascade levels for SolarSTEPS based on grid size.

    Increases the number of cascade levels until the coarsest Gaussian
    cascade resolution reaches or undershoots TARGET_SMALLEST_RESOLUTION
    for the largest horizontal grid dimension.

    Args:
        ratio_data: Input array of shape (time, lat, lon) containing
            SDS/SDS_CS ratios for the past timesteps.

    Returns:
        Number of cascade levels.

    Raises:
        ValueError: If TARGET_SMALLEST_RESOLUTION is less than or equal to 1.
    """

    if TARGET_SMALLEST_RESOLUTION <= 1:
        raise ValueError("TARGET_SMALLEST_RESOLUTION must be greater than 1.")

    largest_dimension = max(ratio_data.shape[1], ratio_data.shape[2])
    n_cascade_levels = 1
    while True:
        _, resolution_centers = _gaussweights_1d(
            largest_dimension, n_cascade_levels, gauss_scale=0.5
        )
        if resolution_centers[0] <= TARGET_SMALLEST_RESOLUTION:
            logger.info(
                f"Optimal number of cascades identified to be: {n_cascade_levels}"
            )
            return n_cascade_levels
        n_cascade_levels += 1


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
        solar_forecast = ratio_forecast * clearsky_stack
        return solar_forecast[
            np.newaxis, :, :, :
        ]  # Add ensemble dimension for consistency

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


def prepend_t0(
    clearsky_data: xr.Dataset,
    ratio_data: np.ndarray,
    solar_forecast: np.ndarray,
    config: dict,
    clearsky_t0_time: datetime,
) -> np.ndarray:
    """Prepend analysis timestep (t=0) to an ensemble solar forecast.

    Computes the t=0 solar field as the latest observed ratio
    (ratio_data[-1]) multiplied by clearsky irradiance at clearsky_t0_time,
    then prepends that field to all ensemble members in solar_forecast.

    Args:
        clearsky_data: Dataset containing clearsky irradiance values on
            a time axis.
        ratio_data: Ratio history array with shape (time, lat, lon).
        solar_forecast: Forecast array with shape
            (ensemble, forecast_time, lat, lon).
        config: Runtime configuration dict containing
            config["nc_variable_names"]["sds_cs"].
        clearsky_t0_time: Timestamp for the analysis clearsky field,
            typically one day before the nowcast time.

    Returns:
        Array of shape (ensemble, forecast_time + 1, lat, lon)
        with the analysis field inserted at index 0 along the time axis.
    """
    # Prepend timestep 0: current observation (ratio_data[-1]) × clearsky at t=0
    sds_cs_t0 = clearsky_data.sel(time=clearsky_t0_time.replace(tzinfo=None))[
        config["nc_variable_names"]["sds_cs"]
    ].values
    solar_t0 = ratio_data[-1] * sds_cs_t0

    # Broadcast the same t=0 clearsky-based analysis field to all ensembles.
    solar_t0_ens = np.broadcast_to(
        solar_t0,
        (solar_forecast.shape[0],) + solar_t0.shape,
    )
    solar_forecast = np.concatenate(
        [solar_t0_ens[:, np.newaxis, :, :], solar_forecast],
        axis=1,
    )
    return solar_forecast
