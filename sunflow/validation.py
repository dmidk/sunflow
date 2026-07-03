#!/usr/bin/env python3
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


class MissingClearskyDataError(RuntimeError):
    pass


class DataNotAvailableError(RuntimeError):
    pass


def validate_config(config: dict[str, Any], dataset_name: str) -> None:
    """Validate required keys are present in the dataset config.

    Checks that all keys needed at runtime are present in the config dict
    loaded from config.yaml. Exits immediately with a descriptive error
    if any required key is missing.

    Args:
        config: Dataset configuration dict loaded from config.yaml.
        Requires the full config for the dataset to check for all required keys.
        dataset_name: Name of the dataset (used in error messages).

    Raises:
        SystemExit: If any required key is absent.
    """
    required_keys = ["filename_format"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        logger.error(
            f"Missing required key(s) in config for dataset '{dataset_name}': "
            f"{', '.join(missing)}. "
            "Check config.yaml and ensure all required fields are present. Exiting.\n"
        )
        sys.exit(1)


def validate_run_mode(run_mode: str, dataset_name: str) -> None:
    """Validate that the run mode is compatible with the dataset.

    Exits immediately if the combination of run mode and dataset name
    is not supported.

    Args:
        run_mode: The requested run mode ('download', 'files', or 's3').
        dataset_name: Name of the dataset.

    Raises:
        SystemExit: If the run mode is incompatible with the dataset.
    """
    if run_mode == "download" and dataset_name == "DWD":
        logger.error(
            "Currently data from DWD is only available online for about half a day. "
            "Thus, run_mode 'download' is not supported for dataset 'DWD'. "
            "Use run_mode 'files' or 's3' instead. Exiting.\n"
        )
        sys.exit(1)


def verify_environment_variables(run_mode: str, dataset_name: str) -> None:
    """Verify environment variables based on run mode.

    Args:
        run_mode: The mode in which the application is running
        ('download', 'files', 's3').
        dataset_name: Name of the dataset being used.

    Raises:
        SystemExit: If any required environment variable is missing.
    """
    if run_mode == "download" and dataset_name == "KNMI":
        if not os.getenv("KNMI_API_KEY"):
            logger.error("KNMI_API_KEY environment variable not set. Exiting.\n")
            sys.exit(1)

    if run_mode == "s3":
        required = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "S3_ENDPOINT_URL",
            "S3_BUCKET",
        ]
        missing = [v for v in required if not os.getenv(v)]

        if missing:
            logger.error(
                f"S3 mode requires the following environment variables:"
                f" {', '.join(missing)}. Exiting.\n"
            )
            sys.exit(1)


def validate_data_shape(
    ratio_data: np.ndarray,
    past_time_steps: list[datetime],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> None:
    """Validate that preprocessed data has expected shape and valid coordinates.

    Args:
        ratio_data: Preprocessed ratio array
        past_time_steps: List of past timesteps
        latitudes: Array of latitude coordinates
        longitudes: Array of longitude coordinates

    Raises:
        SystemExit: If validation fails
    """
    logger.info("Validating data shape...")
    expected_shape = (len(past_time_steps), len(latitudes), len(longitudes))

    if ratio_data.shape != expected_shape:
        raise RuntimeError(
            f"Data shape validation failed. Expected {expected_shape}, "
            f"got {ratio_data.shape}."
        )

    logger.info(
        f"Data shape validated: {ratio_data.shape} "
        f"(time={len(past_time_steps)}, lat={len(latitudes)}, lon={len(longitudes)})"
    )


def validate_clearsky_completeness(
    clearsky_data: xr.Dataset,
    time_steps: list[datetime],
) -> None:
    """Validate that clearsky data exists for all required timesteps.

    Args:
        clearsky_data: xarray Dataset with a 'time' dimension containing
            clearsky data for each forecast step.
        time_steps: List of timesteps that should be present

    Raises:
        MissingClearskyDataError: If any timesteps are missing
    """
    logger.info("Validating clearsky data completeness...")
    missing_clearsky_times = []

    available = (
        pd.DatetimeIndex(clearsky_data.time.values)
        if "time" in clearsky_data.coords
        else pd.DatetimeIndex([])
    )

    for time_step in time_steps:
        if pd.Timestamp(time_step.replace(tzinfo=None)) not in available:
            missing_clearsky_times.append(time_step.strftime("%Y-%m-%dT%H:%M:%SZ"))

    if missing_clearsky_times:
        raise MissingClearskyDataError(
            f"Missing clearsky data for {len(missing_clearsky_times)} time steps: "
            f"{', '.join(missing_clearsky_times[:5])}"
            f"{'...' if len(missing_clearsky_times) > 5 else ''}. "
            "Cannot produce forecast without complete clearsky data. Exiting.\n"
        )

    logger.info(f"Clearsky data complete for all {len(time_steps)} forecast steps")


def validate_clearsky_shapes(
    clearsky_data: xr.Dataset,
    time_steps: list[datetime],
    expected_spatial_shape: tuple[int, int],
    nc_variable_names: dict[str, str],
) -> None:
    """Validate that clearsky data arrays have expected spatial dimensions.

    Args:
        clearsky_data: xarray Dataset with a 'time' dimension containing
            clearsky data for each forecast step.
        time_steps: List of timesteps to validate
        expected_spatial_shape: Tuple of (lat_size, lon_size)
        nc_variable_names: Dictionary mapping canonical keys to NetCDF variable names

    Raises:
        SystemExit: If any shape mismatches are found
    """
    logger.info("Validating clearsky data shapes...")

    for time_step in time_steps:
        clearsky_ds = clearsky_data.sel(time=time_step.replace(tzinfo=None))
        clearsky_array = clearsky_ds[nc_variable_names["sds_cs"]].values

        if clearsky_array.shape != expected_spatial_shape:
            raise RuntimeError(
                f"Clearsky data shape mismatch at "
                f"{time_step.strftime('%Y-%m-%dT%H:%M:%SZ')}. "
                f"Expected {expected_spatial_shape}, "
                f"got {clearsky_array.shape}."
            )

    logger.info(f"Clearsky data shapes validated: all match {expected_spatial_shape}")
