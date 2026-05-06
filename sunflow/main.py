#!/usr/bin/env python3
import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

import isodate
import numpy as np
import yaml
from loguru import logger
from pysteps.motion.lucaskanade import dense_lucaskanade

from . import __version__
from .config import NowcastConfig, S3Config
from .data_io import (
    fetch_clearsky_with_fallback,
    fetch_current_data_with_retry,
    load_data_from_files,
    load_data_from_s3,
    save_forecast,
)
from .downloaders import download_past_data
from .forecast import multiply_clearsky, preprocess_data, simple_advection_forecast
from .geospatial import check_solar_elevation, get_bbox
from .time_handler import generate_time_steps, round_time
from .validation import (
    MissingClearskyDataError,
    validate_clearsky_completeness,
    validate_clearsky_shapes,
    validate_config,
    validate_data_shape,
    validate_run_mode,
    verify_environment_variables,
)


class RunStatus(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"


@dataclass
class RunResult:
    status: RunStatus
    time_step: datetime
    filename: str | None = None
    reason: str | None = None


# Model version
model_version = __version__


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments.

    Defines arguments for run mode, dataset, bounding box, and an optional
    custom time override. Validates that --custom-bbox is provided and
    correctly formatted when --bbox=CUSTOM is selected.

    Returns:
        Parsed argument namespace with attributes run_mode, dataset,
        bbox, custom_bbox, and time.
    """

    def parse_datetime_with_timezone(datetime_str: str) -> datetime:
        """Parse an ISO 8601 datetime string and normalise to UTC.

        If no timezone is specified, UTC is assumed. Any other timezone
        is converted to UTC. The returned datetime always carries
        standard datetime.timezone.utc, regardless of the input tzinfo
        object (e.g. isodate.tzinfo.Utc is replaced with timezone.utc).
        """
        dt = isodate.parse_datetime(datetime_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt

    parser = argparse.ArgumentParser(
        description="Simple Solar Nowcasting with Probabilistic Advection only",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--run_mode",
        choices=["download", "files", "s3"],
        default="download",
        help="Run mode (default: download): \n"
        "download: fetch from API. For KNMI data, this requires "
        "KNMI_API_KEY environment variable\n"
        "files: read from local files\n"
        "s3: read from s3. This requires AWS_ACCESS_KEY_ID and "
        "AWS_SECRET_ACCESS_KEY environment variables",
    )
    parser.add_argument(
        "--dataset",
        choices=["KNMI", "DWD"],
        default="KNMI",
        help="Choose dataset: Currently, only KNMI and DWD data are supported "
        "(default: KNMI)",
    )
    parser.add_argument(
        "--bbox",
        choices=["DENMARK", "NW_EUROPE", "CUSTOM"],
        default="NW_EUROPE",
        help="Choose bounding box (default: NW_EUROPE)",
    )
    parser.add_argument(
        "--custom-bbox",
        type=str,
        help='Custom bbox in format "lon_min,lat_min,lon_max,lat_max"',
        default=None,
    )
    parser.add_argument(
        "--time",
        type=parse_datetime_with_timezone,
        help="Specific time step in ISO8601 format (e.g., 2025-01-15T12:30Z).",
        default=None,
    )
    parser.add_argument(
        "--start-time",
        type=parse_datetime_with_timezone,
        help="Start of time span in ISO8601 format. Use with --end-time.",
        default=None,
    )
    parser.add_argument(
        "--end-time",
        type=parse_datetime_with_timezone,
        help="End of time span in ISO8601 format (inclusive). Use with --start-time.",
        default=None,
    )

    args = parser.parse_args()

    # Validate time arguments
    if args.time and (args.start_time or args.end_time):
        parser.error("--time cannot be combined with --start-time/--end-time")
    if bool(args.start_time) != bool(args.end_time):
        parser.error("--start-time and --end-time must be provided together")
    if args.start_time and args.end_time and args.start_time > args.end_time:
        parser.error("--start-time must be before --end-time")

    # Validate custom bbox
    if args.bbox == "CUSTOM":
        if not args.custom_bbox:
            parser.error("--custom-bbox is required when --bbox=CUSTOM")
        try:
            # Validate format by trying to parse
            bbox_parts = args.custom_bbox.split(",")
            if len(bbox_parts) != 4:
                raise ValueError("Must have exactly 4 comma-separated values")
            [float(x) for x in bbox_parts]  # Ensure all are numeric
        except ValueError as e:
            parser.error(
                f"Invalid --custom-bbox format: {e}. "
                "Use format 'lon_min,lat_min,lon_max,lat_max'"
            )

    return args


def run_nowcast(
    time_step: datetime,
    run_mode: str,
    config: dict,
    bbox: str,
    dataset_name: str,
    bbox_choice: str,
    nowcast_config: NowcastConfig,
    s3_config: S3Config,
    custom_time: bool = True,
) -> RunResult:
    """Run a single nowcast for the given (already-rounded) time step.

    Args:
        time_step: The time step to produce a forecast for.
        run_mode: One of 'download', 'files', or 's3'.
        config: Dataset configuration dict.
        bbox: Bounding box string.
        dataset_name: Name of dataset.
        bbox_choice: Bounding box identifier.
        nowcast_config: NowcastConfig object.
        s3_config: S3Config object.
        custom_time: If True, skip the retry wait loop on missing data.

    Returns:
        Output filename on success, or None if the run was skipped
        (e.g. sun too low or missing data in a time-span run).
    """
    time_step_str = time_step.strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info(f"--- Running nowcast for {time_step_str} ---")

    # Fetch current data (with retry loop in operational mode)
    fetch_current_data_with_retry(
        time_step,
        run_mode,
        config,
        bbox,
        dataset_name,
        bbox_choice,
        nowcast_config,
        s3_config,
        custom_time=custom_time,
    )

    # Check solar elevation
    try:
        solar_elevation = check_solar_elevation(time_step)
        logger.info(f"Solar elevation: {solar_elevation:.2f} degrees")
        if solar_elevation < 1:
            reason = "sun too low"
            logger.warning(f"{reason.capitalize()}. Skipping.\n")
            return RunResult(
                status=RunStatus.SKIPPED,
                time_step=time_step,
                reason=reason,
            )
    except Exception as e:
        logger.error(f"Error checking solar elevation: {e}")
        raise

    # Get past data
    past_time_steps = generate_time_steps(
        time_step,
        nowcast_config.past_steps,
        nowcast_config.input_data_frequency_minutes,
        "past_observations",
    )

    logger.info(f"Loading past data for {len(past_time_steps)} time steps...")
    match run_mode:
        case "download":
            data = download_past_data(past_time_steps, config, bbox, dataset_name)
        case "files":
            data = load_data_from_files(
                past_time_steps,
                dataset_name,
                bbox_choice,
                nowcast_config.satellite_data_directory,
                "past data",
                config["filename_format"],
            )
        case "s3":
            data = load_data_from_s3(
                past_time_steps,
                dataset_name,
                bbox_choice,
                s3_config,
                "past data",
                config["filename_format"],
            )

    n_loaded = len(data.time) if "time" in data.coords else 0
    logger.info(f"Loaded {n_loaded} past data timesteps")
    if n_loaded == 0:
        raise RuntimeError("No past data loaded. Cannot proceed.")

    # Preprocess
    logger.info("Preprocessing data...")
    ratio_data, latitudes, longitudes = preprocess_data(
        data, past_time_steps, config["nc_variable_names"]
    )

    # Validate data shape
    validate_data_shape(ratio_data, past_time_steps, latitudes, longitudes)

    logger.info("Starting forecast computation...")
    # Compute motion field
    motion_field = dense_lucaskanade(ratio_data)

    # Simple forecast (ratio forecast)
    ratio_forecast = simple_advection_forecast(
        ratio_data, motion_field, nowcast_config.future_steps, ens_members=nowcast_config.ens_members
    )

    # Generate previous day time steps for clearsky lookup
    previous_day_time_steps = generate_time_steps(
        time_step,
        nowcast_config.future_steps,
        nowcast_config.input_data_frequency_minutes,
        "previous_day",
    )

    # Include clearsky at t=0 (time_step - 1 day) for the analysis timestep
    clearsky_t0_time = time_step - timedelta(days=1)
    all_clearsky_time_steps = [clearsky_t0_time] + previous_day_time_steps

    # Fetch clearsky data with fallback to earlier days if a file is missing
    logger.info("Fetching clearsky data...")
    clearsky_data = fetch_clearsky_with_fallback(
        all_clearsky_time_steps,
        run_mode,
        nowcast_config.max_clearsky_fallback_days,
        config,
        bbox,
        dataset_name,
        bbox_choice,
        nowcast_config,
        s3_config,
    )

    # Validate clearsky data
    try:
        validate_clearsky_completeness(clearsky_data, previous_day_time_steps)
    except MissingClearskyDataError as e:
        raise RuntimeError(f"Missing clearsky data: {e}") from e

    expected_spatial_shape = (len(latitudes), len(longitudes))
    validate_clearsky_shapes(
        clearsky_data,
        previous_day_time_steps,
        expected_spatial_shape,
        config["nc_variable_names"],
    )

    # Multiply by clearsky to get actual solar irradiance
    logger.info("Multiplying by clearsky values...")
    solar_forecast = multiply_clearsky(
        ratio_forecast,
        clearsky_data,
        previous_day_time_steps,
        config["nc_variable_names"],
    )

    # Prepend timestep 0: current observation (ratio_data[-1]) × clearsky at t=0
    sds_cs_t0 = clearsky_data.sel(time=clearsky_t0_time.replace(tzinfo=None))[
        config["nc_variable_names"]["sds_cs"]
    ].values
    solar_t0 = ratio_data[-1] * sds_cs_t0
    solar_forecast = np.concatenate([solar_t0[np.newaxis, :, :], solar_forecast], axis=0)

    # Save forecast (now contains actual solar irradiance, not ratios)
    filename = save_forecast(
        solar_forecast,
        time_step,
        nowcast_config.future_steps + 1,  # +1 for the t=0 analysis step
        latitudes,
        longitudes,
        dataset_name,
        nowcast_config,
        model_version,
        run_mode,
        s3_config,
    )

    logger.info(f"Nowcast completed: {filename}")
    return RunResult(
        status=RunStatus.SUCCESS,
        time_step=time_step,
        filename=filename,
    )


def print_run_summary(
    results: list[RunResult], failures: list[tuple[datetime, str]]
) -> None:
    n_success = sum(r.status == RunStatus.SUCCESS for r in results)
    n_skipped = sum(r.status == RunStatus.SKIPPED for r in results)
    n_failed = len(failures)

    logger.info(
        f"Run summary: {n_success} success, {n_skipped} skipped, {n_failed} failed"
    )

    if n_skipped:
        skipped_lines = "\n  ".join(
            r.time_step.strftime("%Y-%m-%dT%H:%M:%SZ")
            for r in results
            if r.status == RunStatus.SKIPPED
        )
        logger.info(f"Skipped timesteps:\n  {skipped_lines}")

    if n_failed:
        failed_lines = "\n  ".join(
            f"{t.strftime('%Y-%m-%dT%H:%M:%SZ')} ({msg})" for t, msg in failures
        )
        logger.error(f"Failed timesteps:\n  {failed_lines}")


def cli() -> None:
    """Command line interface for running the solarnowcasting.main module."""
    logger.info(
        f"Starting solarnowcasting.main (with configuration version: {model_version})..."
    )

    # Load configuration
    nowcast_config = NowcastConfig.from_env()
    s3_config = S3Config.from_env()
    args = parse_arguments()

    run_mode = args.run_mode
    dataset_name = args.dataset
    bbox_choice = args.bbox
    bbox = get_bbox(bbox_choice, args.custom_bbox)

    config = yaml.safe_load(open("config.yaml"))[dataset_name]

    if run_mode != "s3":
        os.makedirs(nowcast_config.nowcast_directory, exist_ok=True)

    logger.info(f"Running in {run_mode} mode")
    logger.info(f"Using {dataset_name} dataset")
    logger.info(f"Using {bbox_choice} bbox: {bbox}")

    validate_run_mode(run_mode, dataset_name)
    validate_config(config, dataset_name)
    verify_environment_variables(run_mode, dataset_name)

    # Determine the time steps to run
    if args.start_time and args.end_time:
        step = timedelta(minutes=nowcast_config.input_data_frequency_minutes)
        time_steps = []
        t = args.start_time
        while t <= args.end_time:
            time_steps.append(t)
            t += step
        logger.info(
            f"Running time span: {args.start_time.strftime('%Y-%m-%dT%H:%M:%SZ')} "
            f"to {args.end_time.strftime('%Y-%m-%dT%H:%M:%SZ')} "
            f"({len(time_steps)} time steps)"
        )
        custom_time = True
    elif args.time:
        time_steps = [args.time]
        logger.info(f"Using custom time: {args.time.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        custom_time = True
    else:
        now = datetime.now(timezone.utc)
        rounded_time = round_time(
            now,
            nowcast_config.input_data_availability_delay_minutes,
            nowcast_config.input_data_frequency_minutes,
        )
        logger.info(f"Script start time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info(
            f"Rounded time for data request: "
            f"{rounded_time.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )
        time_steps = [rounded_time]
        custom_time = False

    results: list[RunResult] = []
    failures: list[tuple[datetime, str]] = []

    # Run nowcast for each time step
    for time_step in time_steps:
        try:
            result = run_nowcast(
                time_step,
                run_mode,
                config,
                bbox,
                dataset_name,
                bbox_choice,
                nowcast_config,
                s3_config,
                custom_time=custom_time,
            )
            results.append(result)
        except Exception as e:
            logger.error(
                f"Nowcast failed for {time_step.strftime('%Y-%m-%dT%H:%M:%SZ')}: {e}"
            )
            failures.append((time_step, str(e)))
            if len(time_steps) == 1:
                print_run_summary(results, failures)
                sys.exit(1)  # Single-step run: propagate failure
            # Time-span run: log and continue to next step

    print_run_summary(results, failures)

    now = datetime.now(timezone.utc)
    logger.info(f"Script end time = {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")


if __name__ == "__main__":
    cli()
