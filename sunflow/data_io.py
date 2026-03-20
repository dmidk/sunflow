import os
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any

import fsspec
import numpy as np
import requests.exceptions
import xarray as xr
from loguru import logger

from .config import NowcastConfig, S3Config
from .validation import DataNotAvailableError


def fetch_current_data_with_retry(
    time_step: datetime,
    run_mode: str,
    config: dict[str, Any],
    bbox: str,
    dataset_name: str,
    bbox_choice: str,
    nowcast_config: NowcastConfig,
    s3_config: S3Config,
    custom_time: bool = False,
) -> None:
    """Fetch current data with retry logic for operational mode.

    Args:
        time_step: Datetime object for data to fetch
        run_mode: One of 'download', 'files', or 's3'
        config: Dataset configuration dict
        bbox: Bounding box string
        dataset_name: Name of dataset (options: KNMI, DWD)
        bbox_choice: Bounding box choice string
        nowcast_config: NowcastConfig object
        s3_config: S3Config object
        custom_time: Whether a custom time was specified (no retry if True)

    Raises:
        DataNotAvailableError: If data is missing and custom_time is True (no retry).
        RuntimeError: If data cannot be fetched within the maximum wait time.
    """
    from .downloaders import download_current_data

    time_step_str = time_step.strftime("%Y-%m-%dT%H:%M:%SZ")
    start_time = time.time()

    while True:
        try:
            logger.info(f"Attempting to fetch data for: {time_step_str}")

            match run_mode:
                case "download":
                    download_current_data(
                        time_step,
                        config,
                        bbox,
                        dataset_name,
                        bbox_choice,
                        nowcast_config.satellite_data_directory,
                    )
                case "files":
                    check_current_data_existence_file(
                        time_step,
                        dataset_name,
                        bbox_choice,
                        nowcast_config.satellite_data_directory,
                        config["filename_format"],
                    )
                case "s3":
                    check_current_data_existence_s3(
                        time_step,
                        dataset_name,
                        bbox_choice,
                        s3_config,
                        config["filename_format"],
                    )

            logger.info(f"Data successfully retrieved for {time_step_str}")
            break  # Exit the loop if successful

        except (FileNotFoundError, requests.exceptions.HTTPError) as e:
            # Data not ready yet - this is expected during the wait loop
            logger.info(f"Data not yet ready: {e}")

            # If using custom time, don't retry - just raise
            if custom_time:
                logger.info("Custom time specified. Not retrying.")
                raise DataNotAvailableError(
                    f"Data not available for {time_step_str}."
                )

            time.sleep(30)  # Wait 30 sec before retrying

            # Check if the maximum waiting time has been reached
            if (
                time.time() - start_time
                > nowcast_config.max_waiting_time_minutes * 60
            ):
                raise RuntimeError(
                    f"Maximum wait time of {nowcast_config.max_waiting_time_minutes}"
                    f" minutes reached for {time_step_str}."
                )

        except Exception as e:
            logger.error(f"Unexpected error while fetching data: {e}")
            raise


def generate_input_filename(
    time_step: datetime,
    dataset_name: str,
    bbox_choice: str,
    filename_format: str,
) -> str:
    """Generate input filename based on a format template string.

    Works uniformly across all modes (download, files, s3).

    Args:
        time_step: Datetime of the data timestep.
        dataset_name: Name of dataset (options: KNMI, DWD).
        bbox_choice: Bounding box identifier.
        filename_format: Template string from config['filename_format'].

    Returns:
        Filename string with all template variables substituted.

    Template variables supported:
        {dataset_name}: Name of the dataset
        {bbox_choice}: Bounding box identifier
        {timestamp}: Compact format YYYYMMDDHHMM
        {pds_timestamp}: PDS format YYYY-MM-DDTHH_MM_SSZ
        {year}: Four-digit year (e.g. 2026)
        {month}: Two-digit month (e.g. 03)
        {day}: Two-digit day (e.g. 09)
        {hour}: Two-digit hour (e.g. 12)

    Path separators in the result are supported, so a format like
    ``{year}/{month}/{day}/{dataset_name}_{timestamp}.nc`` resolves to a
    file inside a date-structured subdirectory of *satellite_data_directory*.
    """
    format_template = filename_format

    # Generate both time formats
    timestamp_compact = time_step.strftime("%Y%m%d%H%M")
    timestamp_pds = time_step.strftime("%Y-%m-%dT%H_%M_%SZ")

    # Substitute template variables
    filename = format_template.format(
        dataset_name=dataset_name,
        bbox_choice=bbox_choice,
        timestamp=timestamp_compact,
        pds_timestamp=timestamp_pds,
        year=time_step.strftime("%Y"),
        month=time_step.strftime("%m"),
        day=time_step.strftime("%d"),
        hour=time_step.strftime("%H"),
    )
    return filename


def check_current_data_existence_file(
    request_time: datetime,
    dataset_name: str,
    bbox_choice: str,
    satellite_data_directory: str,
    filename_format: str,
) -> None:
    """Check for existence of current data file.

    Args:
        request_time: Python datetime object.
        dataset_name: Name of dataset (options: KNMI, DWD).
        bbox_choice: Bounding box identifier.
        satellite_data_directory: Directory containing data files.
        filename_format: Template string from config['filename_format'].

    Raises:
        FileNotFoundError: If the expected file does not exist.
    """
    filename = generate_input_filename(
        request_time, dataset_name, bbox_choice, filename_format
    )
    filepath = os.path.join(satellite_data_directory, filename)
    logger.info(f"Checking existence of data at {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")


def load_data_from_files(
    time_steps: list[datetime],
    dataset_name: str,
    bbox_choice: str,
    satellite_data_directory: str,
    data_type: str,
    filename_format: str,
) -> xr.Dataset:
    """Load time steps from files (operational mode).

    Args:
        time_steps: List of timesteps to load.
        dataset_name: Name of dataset (options: KNMI, DWD).
        bbox_choice: Bounding box identifier.
        satellite_data_directory: Directory containing data files.
        data_type: Type of data for logging (options: past data, clearsky data).
        filename_format: Template string from config['filename_format'].

    Returns:
        xarray Dataset concatenated along a 'time' dimension.

    Raises:
        FileNotFoundError: If any expected file does not exist.
    """
    collected: list[xr.Dataset] = []

    for time_step in time_steps:
        filename = generate_input_filename(
            time_step, dataset_name, bbox_choice, filename_format
        )
        filepath = os.path.join(satellite_data_directory, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"{data_type.capitalize()} file not found: {filepath}"
            )

        try:
            ds = xr.open_dataset(filepath)
            collected.append(
                ds.assign_coords(time=[time_step.replace(tzinfo=None)])
            )
            logger.info(f"Loaded {data_type} from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load {data_type} {filepath}: {e}")
            raise

    if not collected:
        return xr.Dataset()
    return xr.concat(collected, dim="time", data_vars=None)


def check_current_data_existence_s3(
    request_time: datetime,
    dataset_name: str,
    bbox_choice: str,
    s3_config: S3Config,
    filename_format: str,
) -> None:
    """Check for existence of current data file in S3.

    Args:
        request_time: Python datetime object.
        dataset_name: Name of dataset (options: KNMI, DWD).
        bbox_choice: Bounding box identifier.
        s3_config: S3 configuration object.
        filename_format: Template string from config['filename_format'].

    Raises:
        FileNotFoundError: If the expected file does not exist in S3.
    """
    filename = generate_input_filename(
        request_time, dataset_name, bbox_choice, filename_format
    )
    s3_path = f"s3://{s3_config.bucket}/{s3_config.input_prefix}/{filename}"
    logger.info(f"Checking existence of data at {s3_path}")

    try:
        fs = fsspec.filesystem(
            "s3",
            client_kwargs={"endpoint_url": s3_config.endpoint_url},
        )
        if not fs.exists(s3_path):
            raise FileNotFoundError(
                f"Input file not yet found in S3: {s3_path}"
            )
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error checking file existence in S3: {e}")
        raise


def load_data_from_s3(
    time_steps: list[datetime],
    dataset_name: str,
    bbox_choice: str,
    s3_config: S3Config,
    data_type: str,
    filename_format: str,
) -> xr.Dataset:
    """Load time steps from S3.

    Args:
        time_steps: List of timesteps to load.
        dataset_name: Name of dataset (options: KNMI, DWD).
        bbox_choice: Bounding box identifier.
        s3_config: S3 configuration object.
        data_type: Type of data for logging (options: past data, clearsky data).
        filename_format: Template string from config['filename_format'].

    Returns:
        xarray Dataset concatenated along a 'time' dimension.

    Raises:
        FileNotFoundError: If any expected file does not exist in S3.
    """
    collected: list[xr.Dataset] = []
    fs = fsspec.filesystem(
        "s3",
        client_kwargs={"endpoint_url": s3_config.endpoint_url},
    )

    for time_step in time_steps:
        filename = generate_input_filename(
            time_step, dataset_name, bbox_choice, filename_format
        )
        s3_path = (
            f"s3://{s3_config.bucket}/{s3_config.input_prefix}/{filename}"
        )

        try:
            with fs.open(s3_path, "rb") as f:
                ds = xr.open_dataset(
                    f, engine="h5netcdf"
                ).load()  # Load into memory
            collected.append(
                ds.assign_coords(time=[time_step.replace(tzinfo=None)])
            )
            logger.info(f"Loaded {data_type} from {s3_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{data_type.capitalize()} file not found in S3: {s3_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load {data_type} {s3_path}: {e}")
            raise

    if not collected:
        return xr.Dataset()
    return xr.concat(collected, dim="time", data_vars=None)


def fetch_clearsky_with_fallback(
    time_steps: list[datetime],
    run_mode: str,
    max_fallback_days: int,
    config: dict[str, Any],
    bbox: str,
    dataset_name: str,
    bbox_choice: str,
    nowcast_config: NowcastConfig,
    s3_config: S3Config,
) -> xr.Dataset:
    """Fetch clearsky data, falling back to earlier days if a file is missing.

    For each requested time step, tries the target time first, then one day
    earlier, two days earlier, and so on up to max_fallback_days attempts.
    Data is always stored under the original target time coordinate so that
    downstream code is unaffected.

    Args:
        time_steps: Requested clearsky times (typically forecast times - 1 day).
        run_mode: One of 'download', 'files', or 's3'.
        max_fallback_days: Maximum number of days back to search per time step.
        config: Dataset configuration dict.
        bbox: Bounding box string.
        dataset_name: Name of dataset (options: KNMI, DWD).
        bbox_choice: Bounding box identifier.
        nowcast_config: NowcastConfig object.
        s3_config: S3Config object.

    Returns:
        xarray Dataset concatenated along 'time', keyed by the original
        target time steps. Time steps for which no data was found are omitted
        with a warning.
    """
    from .downloaders import download_clearsky_data

    collected: list[xr.Dataset] = []

    for ts in time_steps:
        ts_str = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        fetched = None

        for offset in range(max_fallback_days):
            source_time = ts - timedelta(days=offset)
            source_str = source_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                match run_mode:
                    case "download":
                        result = download_clearsky_data(
                            [source_time], config, bbox, dataset_name
                        )
                        if result.sizes.get("time", 0) == 0:
                            raise FileNotFoundError(
                                f"No clearsky data downloaded for {source_str}"
                            )
                        fetched = result
                    case "files":
                        fetched = load_data_from_files(
                            [source_time],
                            dataset_name,
                            bbox_choice,
                            nowcast_config.satellite_data_directory,
                            "clearsky data",
                            config["filename_format"],
                        )
                    case "s3":
                        fetched = load_data_from_s3(
                            [source_time],
                            dataset_name,
                            bbox_choice,
                            s3_config,
                            "clearsky data",
                            config["filename_format"],
                        )

                if offset > 0:
                    logger.warning(
                        f"Clearsky for {ts_str}: using data from "
                        f"{offset + 1} day(s) back ({source_str})"
                    )
                break

            except FileNotFoundError as e:
                logger.info(f"Clearsky not available at {source_str}: {e}")
                continue

        if fetched is not None:
            collected.append(
                fetched.assign_coords(time=[ts.replace(tzinfo=None)])
            )
        else:
            logger.warning(
                f"No clearsky data found for {ts_str} "
                f"within {max_fallback_days} day(s)"
            )

    if not collected:
        return xr.Dataset()
    return xr.concat(collected, dim="time", data_vars=None)


def save_forecast(
    forecast: np.ndarray,
    time_step: datetime,
    n_steps: int,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    dataset_name: str,
    nowcast_config: NowcastConfig,
    model_version: str,
    run_mode: str = "files",
    s3_config: S3Config | None = None,
) -> str:
    """Save forecast array to a CF-compliant NetCDF4 file.

    Writes the probabilistic advection forecast to either a local file or S3,
    depending on `run_mode`. The time coordinate is stored as CF-convention
    numeric values (float64, minutes since the forecast reference time).

    Args:
        forecast: Forecast array, shape [time, lat, lon] or
            [ensemble, time, lat, lon].
        time_step: Forecast reference time (start of the forecast window).
        n_steps: Number of forecast time steps to write.
        latitudes: 1-D array of latitude values (degrees).
        longitudes: 1-D array of longitude values (degrees).
        dataset_name: Name of the source dataset (options: KNMI, DWD).
        nowcast_config: NowcastConfig object supplying output directory,
            ensemble size, and input data frequency.
        model_version: Model version string written as a global attribute.
        run_mode: One of 'files' (local) or 's3'. Defaults to 'files'.
        s3_config: S3Config object; required when run_mode is 's3'.

    Returns:
        Filename (basename only) of the written NetCDF file.
    """
    input_data_frequency_minutes = nowcast_config.input_data_frequency_minutes
    ens_members = nowcast_config.ens_members
    filename = f"SolarNowcast_{time_step.strftime('%Y%m%d%H%M')}.nc"

    # Add ensemble dimension if needed (forecast should be [ensemble, time, lat, lon])
    if forecast.ndim == 3:
        forecast = forecast[np.newaxis, :, :, :]  # Now [1, time, lat, lon]

    # Build time coordinate (CF-convention: minutes since forecast reference time)

    time_step_naive = time_step.replace(tzinfo=None)
    _time_units = (
        f"minutes since {time_step_naive.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    time_datetimes = [
        time_step_naive + timedelta(minutes=input_data_frequency_minutes * i)
        for i in range(0, n_steps)
    ]

    ds = xr.Dataset(
        {
            "probabilistic_advection": (
                ["ensemble", "time", "lat", "lon"],
                forecast,
                {
                    "description": "Probabilistic advection solar forecast",
                    "long_name": "Surface downwelling solar radiation",
                    "units": "W m-2",
                },
            ),
        },
        coords={
            "time": (
                ["time"],
                time_datetimes,
                {"long_name": "time"},
            ),
            "ensemble": (
                ["ensemble"],
                np.arange(ens_members, dtype=np.int32),
                {"long_name": "Ensemble member index"},
            ),
            "lat": (
                ["lat"],
                latitudes,
                {"long_name": "latitude", "units": "degrees"},
            ),
            "lon": (
                ["lon"],
                longitudes,
                {"long_name": "longitude", "units": "degrees"},
            ),
        },
        attrs={
            "description": (
                f"Simple Probabilistic Advection solar forecast "
                f"using {dataset_name} data"
            ),
            "history": (
                f"Created "
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
            ),
            "model_version": model_version,
        },
    )

    # CF-compliant time encoding so ncview can read the axis
    time_encoding = {
        "units": _time_units,
        "calendar": "standard",
        "dtype": np.int32,
    }

    if run_mode == "s3":
        save_forecast_to_s3(ds, filename, s3_config, time_encoding)
    else:
        output_path = os.path.join(nowcast_config.nowcast_directory, filename)
        ds.to_netcdf(output_path, encoding={"time": time_encoding})

    return filename


def save_forecast_to_s3(
    dataset: xr.Dataset,
    filename: str,
    s3_config: S3Config,
    time_encoding: dict,
) -> None:
    """Upload a forecast dataset to S3 as a NetCDF4 file.

    Serialises the dataset to a BytesIO buffer first, then uploads the buffer
    to the configured S3 bucket. The provided `time_encoding` is passed to
    `xr.Dataset.to_netcdf` to ensure CF-compliant numeric time values.

    Args:
        dataset: xarray Dataset to write, as produced by `save_forecast`.
        filename: Basename of the destination file (e.g.
            'SolarNowcast_202603091200.nc').
        s3_config: S3Config object supplying bucket, output prefix, and
            endpoint URL.
        time_encoding: Encoding dict for the 'time' variable (must include at
            least 'units' and 'calendar' keys).

    Raises:
        Exception: Re-raises any error that occurs during serialisation or
            upload, after logging it.
    """
    s3_path = f"s3://{s3_config.bucket}/{s3_config.output_prefix}/{filename}"

    logger.info(f"Saving forecast to {s3_path}")

    try:
        fs = fsspec.filesystem(
            "s3",
            client_kwargs={"endpoint_url": s3_config.endpoint_url},
        )
        # Write to BytesIO buffer first
        buffer = BytesIO()
        dataset.to_netcdf(buffer, encoding={"time": time_encoding})
        buffer.seek(0)

        # Upload to S3
        with fs.open(s3_path, "wb") as f:
            f.write(buffer.getvalue())

        logger.info(f"Successfully saved forecast to S3: {s3_path}")
    except Exception as e:
        logger.error(f"Failed to save to S3: {e}")
        raise
