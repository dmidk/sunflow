import os
from datetime import datetime
from io import BytesIO
from typing import Any

import requests
import xarray as xr
from loguru import logger

from .data_io import generate_input_filename


def download_netcdf_knmi(url: str, api_key: str) -> BytesIO:
    """Download a NetCDF file from the KNMI WCS API.

    Sends a GET request with the API key in the Authorization header and
    returns the response body as an in-memory byte buffer.

    Args:
        url: Full WCS request URL including coverage, format, CRS, BBOX and time.
        api_key: KNMI API key sent as the Authorization header value.

    Returns:
        BytesIO buffer containing the raw NetCDF file content.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status code.
    """
    response = requests.get(url, headers={"Authorization": api_key})
    response.raise_for_status()
    return BytesIO(response.content)


def download_netcdf_dwd(url: str) -> BytesIO:
    """Download a NetCDF file from the DWD open-data server.

    Sends an unauthenticated GET request and returns the response body as
    an in-memory byte buffer.

    Args:
        url: Full URL to the DWD NetCDF file.

    Returns:
        BytesIO buffer containing the raw NetCDF file content.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status code.
    """
    response = requests.get(url)
    response.raise_for_status()
    return BytesIO(response.content)


def generate_dwd_filename(timestamp: datetime) -> str:
    """Generate the DWD SIS filename for a given timestamp.

    Produces filenames following the DWD naming convention:
    SISin<YYYYMMDDHHMM>FDv3.nc, e.g. SISin202501151230FDv3.nc.

    Args:
        timestamp: Datetime of the requested observation.

    Returns:
        string of the filename for the requested timestamp.
    """
    return f"SISin{timestamp.strftime('%Y%m%d%H%M')}FDv3.nc"


def subset_to_bbox(ds: xr.Dataset, bbox: str) -> xr.Dataset:
    """Subset an xarray Dataset to a geographic bounding box.

    Detects the latitude and longitude coordinate names automatically by
    matching against common aliases ('lat', 'latitude', 'y' for
    latitude; 'lon', 'longitude', 'x' for longitude). If coordinates
    cannot be identified, the original dataset is returned unchanged with
    a warning.

    Args:
        ds: Input dataset to subset.
        bbox: Bounding box string in the format lon_min,lat_min,lon_max,lat_max.

    Returns:
        Dataset sliced to the requested bounding box, or the original
        dataset if lat/lon coordinates could not be found.
    """
    lon_min, lat_min, lon_max, lat_max = map(float, bbox.split(","))

    lat_coord = lon_coord = None
    for coord in ds.coords:
        if coord.lower() in ["lat", "latitude", "y"]:
            lat_coord = coord
        elif coord.lower() in ["lon", "longitude", "x"]:
            lon_coord = coord

    if lat_coord is None or lon_coord is None:
        logger.warning("Could not find lat/lon coordinates")
        return ds

    return ds.sel(
        {
            lat_coord: slice(lat_min, lat_max),
            lon_coord: slice(lon_min, lon_max),
        }
    )


def download_current_data(
    request_time: datetime,
    config: dict[str, Any],
    bbox: str,
    dataset_name: str,
    bbox_choice: str,
    satellite_data_directory: str,
) -> None:
    """Download satellite data for a single timestep and save it to disk.

    Downloads all configured variables for the requested time, merges them
    into a single dataset, and writes the result as a NetCDF file using
    the filename format defined in config['filename_format']. The actual
    timestamp is taken from the dataset (not request_time) to ensure
    the filename reflects the true observation time.

    Supports KNMI (WCS API, multiple variables merged) and DWD (single file
    with bounding-box subsetting).

    Args:
        request_time: Requested observation time.
        config: Dataset configuration dict from config.yaml, including
            base_url, variables, format, crs, and filename_format.
        bbox: Bounding box string lon_min,lat_min,lon_max,lat_max.
        dataset_name: Name of the dataset source (options: KNMI, DWD).
        bbox_choice: Bounding box identifier used in the output filename.
        satellite_data_directory: Directory where the NetCDF file is saved.
    """
    if dataset_name == "KNMI":
        datasets = []
        for variable in config["variables"]:
            url = (
                f"{config['base_url']}coverage={variable}"
                f"&FORMAT={config['format']}&CRS={config['crs']}"
                f"&BBOX={bbox}&time={request_time.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            )
            netcdf_file = download_netcdf_knmi(url, os.getenv("KNMI_API_KEY"))
            ds = xr.open_dataset(netcdf_file, engine="h5netcdf")
            datasets.append(ds)
        merged_ds = xr.merge(datasets, compat="override")
        current_time = merged_ds.time.values[0]

    elif dataset_name == "DWD":
        filename = generate_dwd_filename(request_time)
        url = f"{config['base_url']}{filename}"
        netcdf_file = download_netcdf_dwd(url)
        ds = xr.open_dataset(netcdf_file)
        merged_ds = subset_to_bbox(ds, bbox)
        current_time = merged_ds.time.values[0]

    # Save data
    current_time_dt = current_time.astype("datetime64[s]").astype(datetime)
    filename = generate_input_filename(
        current_time_dt, dataset_name, bbox_choice, config["filename_format"]
    )
    output_path = os.path.join(satellite_data_directory, filename)
    merged_ds.to_netcdf(output_path)
    logger.info(f"Saved current data to {output_path}")


def download_past_data(
    time_steps: list[datetime],
    config: dict[str, Any],
    bbox: str,
    dataset_name: str,
) -> xr.Dataset:
    """Download satellite data for a list of past timesteps.

    Iterates over the provided timesteps and downloads each one. For KNMI,
    all configured variables are downloaded separately and merged; a timestep
    is only stored if all variables were retrieved successfully. For DWD,
    the file is subsetted to the bounding box before storing. Failed
    individual timesteps are logged as warnings and skipped, so the
    returned Dataset may contain fewer timesteps than time_steps.

    Args:
        time_steps: Ordered list of datetimes to download.
        config: Dataset configuration dict from config.yaml.
        bbox: Bounding box string lon_min,lat_min,lon_max,lat_max.
        dataset_name: Name of the dataset source (options: KNMI, DWD).

    Returns:
        xarray Dataset concatenated along a 'time' dimension, with one
        entry per successfully downloaded timestep.
    """
    collected: list[xr.Dataset] = []

    for time_step in time_steps:
        time_str = time_step.strftime("%Y-%m-%dT%H:%M:%SZ")

        if dataset_name == "KNMI":
            # Download each variable separately, then merge
            var_datasets: list[xr.Dataset] = []
            failed_vars = []
            for var in config["variables"]:
                base_url = (
                    f"{config['base_url']}FORMAT={config['format']}"
                    f"&CRS={config['crs']}&BBOX={bbox}&time={time_str}"
                )
                url = f"{base_url}&coverage={var}"
                try:
                    netcdf_file = download_netcdf_knmi(
                        url, os.getenv("KNMI_API_KEY")
                    )
                    ds = xr.open_dataset(netcdf_file, engine="h5netcdf")
                    var_datasets.append(ds)
                except Exception as e:
                    logger.warning(
                        f"Failed to download data for variable {var} at time {time_str}: "
                        f"{e}"
                    )
                    failed_vars.append(var)
                    continue

            # Only store if we got all variables
            if len(var_datasets) == len(config["variables"]):
                merged_ds = xr.merge(var_datasets, compat="override")
                collected.append(
                    merged_ds.assign_coords(
                        time=[time_step.replace(tzinfo=None)]
                    )
                )
            else:
                logger.warning(
                    f"Incomplete data for {time_str}, missing variables: {failed_vars}"
                )

        elif dataset_name == "DWD":
            filename = generate_dwd_filename(time_step)
            url = f"{config['base_url']}{filename}"
            try:
                netcdf_file = download_netcdf_dwd(url)
                ds = xr.open_dataset(netcdf_file)
                ds_subset = subset_to_bbox(ds, bbox)
                collected.append(
                    ds_subset.assign_coords(
                        time=[time_step.replace(tzinfo=None)]
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to download {filename}: {e}")
                continue

    if not collected:
        return xr.Dataset()
    return xr.concat(collected, dim="time", data_vars=None)


def download_clearsky_data(
    time_steps: list[datetime],
    config: dict[str, Any],
    bbox: str,
    dataset_name: str,
) -> xr.Dataset:
    """Download clearsky irradiance data for a list of timesteps.

    The time_steps are expected to already be shifted to the previous day
    by the caller. For KNMI, only the clearsky variable
    (config['variables'][1]) is fetched. For DWD, the full file is
    downloaded and subsetted to the bounding box. Failed timesteps are
    logged as warnings and skipped.

    Args:
        time_steps: List of datetimes for which to fetch clearsky data
            (typically forecast lead times offset by one day).
        config: Dataset configuration dict from config.yaml.
        bbox: Bounding box string lon_min,lat_min,lon_max,lat_max.
        dataset_name: Name of the dataset source (options: KNMI, DWD).

    Returns:
        xarray Dataset concatenated along a 'time' dimension, with one
        entry per successfully downloaded timestep.
    """
    collected: list[xr.Dataset] = []

    if dataset_name == "KNMI":
        var = config["variables"][1]  # clearsky variable
        for time_step in time_steps:
            clearsky_time_str = time_step.strftime("%Y-%m-%dT%H:%M:%SZ")
            url = (
                f"{config['base_url']}coverage={var}"
                f"&FORMAT={config['format']}&CRS={config['crs']}"
                f"&BBOX={bbox}&time={clearsky_time_str}"
            )
            try:
                netcdf_file = download_netcdf_knmi(
                    url, os.getenv("KNMI_API_KEY")
                )
                ds = xr.open_dataset(netcdf_file, engine="h5netcdf")
                collected.append(
                    ds.assign_coords(time=[time_step.replace(tzinfo=None)])
                )
            except Exception as e:
                logger.warning(f"Failed to download clearsky data: {e}")
                continue

    elif dataset_name == "DWD":
        for time_step in time_steps:
            clearsky_time_str = time_step.strftime("%Y-%m-%dT%H:%M:%SZ")
            filename = generate_dwd_filename(time_step)
            url = f"{config['base_url']}{filename}"
            try:
                netcdf_file = download_netcdf_dwd(url)
                ds = xr.open_dataset(netcdf_file)
                ds_subset = subset_to_bbox(ds, bbox)
                collected.append(
                    ds_subset.assign_coords(
                        time=[time_step.replace(tzinfo=None)]
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Failed to download clearsky data {filename}: {e}"
                )
                continue

    if not collected:
        return xr.Dataset()
    return xr.concat(collected, dim="time", data_vars=None)
