# Sunflow

Framework for real-time satellite-based solar irradiance nowcasting at the Danish Meteorological Institute.

## Overview

This application performs short-term solar irradiance forecasting using:
- Real-time gridded surface solar irradiance products from processed geostationary satellite-data: [KNMI MSG-CPP](https://dataplatform.knmi.nl/dataset/msg-cpp-products-1-0) and [DWD SIS](https://opendata.dwd.de/weather/satellite/radiation/sis/)
- Optical flow motion field computation with the Lucas-Kanade method applied to computed clear-sky indices
- Probabilistic advection for forecast generation, based on [SolarSTEPS](https://github.com/EnergyWeatherAI/SolarSTEPS) in deterministic mode
- Solar irradiance nowcasts calculated using clear-sky data from the previous day

## Fetching the source code
```shell
git clone https://github.com/dmidk/sunflow.git
cd sunflow
```

To contribute, fork the repository first — see [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Local development
This project uses [pdm](https://pdm.fming.dev/latest/) for dependency management.
Other package managers, such as [uv](https://docs.astral.sh/uv/getting-started/), also work.
`pdm` can be installed with:

```shell
curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -
```

After this you can create a virtualenv and install `sunflow` with

```shell
pdm venv create
pdm install
```

The suite can then be run by
```shell
pdm run sunflow
```
or, after activating the virtualenv (`source .venv/bin/activate`):
```shell
sunflow
```

## Container Usage

### Building the Image

```shell
 podman build -f ContainerFile -t sunflow .
```

### Running the Container

To start a basic run, simply type the command below. This will start the default behavior where the script awaits the data and does the nowcasting when the data becomes available.

```shell
podman run --rm sunflow
```

To run for a specific time, specify this when calling the container in the ISO8601 format. Only multiples of 15 minutes are available. Input data from KNMI is available going back one week. Other input can be specified in the same way.

```shell
podman run --rm sunflow --time 2025-10-27T10:00Z
```

If you want the input and output files saved, you need to specify a volume to mount with "-v" when calling the container and specify the corresponding directory as environment variables to the code. E.g. to save both the input and output files you can run the following command.

```shell
podman run --rm -v <local/path/for/input>:/data/satellite_data -v <local/path/for/output>:/data/nowcasts -e SATELLITE_DATA_DIRECTORY=/data/satellite_data -e NOWCAST_DIRECTORY=/data/nowcasts sunflow
```

To fetch data from KNMI, an API key is needed (https://dataplatform.knmi.nl/dataset/access/msg-cpp-products-1-0). This needs to be given as an environment variable upon running the container. Assuming that it is defined as the environment variable KNMI_API_KEY, the container can be called like this:

```shell
podman run --rm -e KNMI_API_KEY=$KNMI_API_KEY sunflow
```

To use S3, AWS_ACCESS_KEY and AWS_SECRET_ACCESS_KEY need to be supplied as environment variables:
```shell
podman run --rm -e AWS_ACCESS_KEY_ID=<key> -e AWS_SECRET_ACCESS_KEY=<key> sunflow --time 2025-10-27T10:00Z --run_mode s3
```

### Interactive shell for debugging
```shell
podman run -it --rm --entrypoint="" sunflow bash
```

### Environment Variables

#### Application Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NOWCAST_DIRECTORY` | `.` | Directory for forecast output files |
| `SATELLITE_DATA_DIRECTORY` | `.` | Directory for input satellite data archive |
| `ENS_MEMBERS` | `1` | Number of ensemble members |
| `PAST_STEPS` | `4` | Number of past time steps for motion field |
| `FUTURE_STEPS` | `24` | Number of forecast time steps |
| `INPUT_DATA_AVAILABILITY_DELAY_MINUTES` | `24` | Data availability delay (minutes) |
| `INPUT_DATA_FREQUENCY_MINUTES` | `15` | Data frequency (minutes) |
| `MAX_WAITING_TIME_MINUTES` | `27` | Maximum wait time for data (minutes) |
| `MAX_CLEARSKY_FALLBACK_DAYS` | `3` | Days back to search for fallback clear-sky data |

#### Data Source Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `KNMI_API_KEY` | None | API key to fetch data from the KNMI API (required for `--run_mode download` with KNMI data) |

#### S3 Configuration (for `--run_mode s3`)

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_ACCESS_KEY_ID` | None | AWS access key ID for S3 authentication (required) |
| `AWS_SECRET_ACCESS_KEY` | None | AWS secret access key for S3 authentication (required) |
| `S3_ENDPOINT_URL` | None | S3 endpoint URL (e.g. `https://s3.amazonaws.com`) |
| `S3_BUCKET` | None | S3 bucket name |
| `S3_INPUT_PREFIX` | `satellite_data` | S3 prefix for input data |
| `S3_OUTPUT_PREFIX` | `nowcasts` | S3 prefix for forecast output |

### Volume Mounts

- `/data/nowcasts` - Forecast output files (NetCDF format)
- `/data/satellite_data` - Input satellite data archive

### Arguments

- `--dataset` - Choose between KNMI or DWD data sources
- `--bbox` - Predefined bounding boxes (DENMARK, NW_EUROPE, CUSTOM)
- `--custom-bbox` - Custom bounding box (lon_min,lat_min,lon_max,lat_max)
- `--time` - Specific time for processing in ISO8601 format
- `--start-time` - Start of a time range in ISO8601 format (use with `--end-time`)
- `--end-time` - End of a time range in ISO8601 format, inclusive (use with `--start-time`)
- `--run_mode` - Specify run mode: `download` (fetch from API), `files` (local files), or `s3` (object storage)

## Data Sources

- **KNMI**: MSG-CPP products — requires a free API key from the [KNMI Data Platform](https://dataplatform.knmi.nl/dataset/access/msg-cpp-products-1-0)
- **DWD**: Surface Incoming Shortwave radiation (SIS) — publicly available, no API key required

## Installing locally
Building pysteps can sometimes fail if a pre-built wheel is not available for your Python version. If the environment variables `CC` and `CXX` are not set, building pysteps falls back to clang, which may only work on Intel-based macOS. To use GNU compilers instead:
```shell
CC=gcc CXX=g++ pdm install
```
