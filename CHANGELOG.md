# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added checks of invalid configuration combinations [!22](https://github.com/dmidk/sunflow/pull/22), @KristianHMoller
- Added `{second}` and `{munite}` as a filename formatting variable [!15](https://github.com/dmidk/sunflow/pull/15), @JoachimKoenigslieb
- Added a `clearsky` config object which can load clear-sky data from files via `clearsky.path` or generate it with `pvlib` via the `simplified_solis` method [!15](https://github.com/dmidk/sunflow/pull/15), @JoachimKoenigslieb
- Added `MIN_SOLAR_ELEVATION_DEGREES` (defaulting to 6 degrees) to configure the minimum maximum-corner solar elevation required to run [!15](https://github.com/dmidk/sunflow/pull/15), @JoachimKoenigslieb

### Changed

- Use `.expand_dims` instead of `.assign_coords` to make sure we have both time dimension and time coordinates when loading from files [!15](https://github.com/dmidk/sunflow/pull/15), @JoachimKoenigslieb
- `check_solar_elevation` now does not assume location is in Copenhagen by default [!15](https://github.com/dmidk/sunflow/pull/15), @JoachimKoenigslieb

## [v1.2.0]

### Added

- Added support for running an ensemble. Default output is ensemble statistics, but --full_ensemble can be specified [!17](https://github.com/dmidk/sunflow/pull/17), @KristianHMoller
- Added support for using a different, smaller output domain than the satellite input domain [!10](https://github.com/dmidk/sunflow/pull/10), @KristianHMoller
- Added a check for the number of ensemble members, as the code currently supports only one [!13](https://github.com/dmidk/sunflow/pull/13), @KristianHMoller
- Subsetting to bounding box is now also done in the `s3` and `files` code paths [!11](https://github.com/dmidk/sunflow/pull/11), @JoachimKoenigslieb
- Subsetting to bounding box now correctly handles both ascending and descending lat/lon coordinates [!11](https://github.com/dmidk/sunflow/pull/11), @JoachimKoenigslieb

### Changed

- Change number of ensemble members from being read from environment variable to being a CLI argument. [!17](https://github.com/dmidk/sunflow/pull/17), @KristianHMoller
- Ensemble output statistics are now configurable via the `ENSEMBLE_STATISTICS` environment variable, and selected statistics are written as separate NetCDF variables. [!17](https://github.com/dmidk/sunflow/pull/17), @KristianHMoller
- Update SolarSTEPS dependency to use the version with arctan2 [!19](https://github.com/dmidk/sunflow/pull/19), @KristianHMoller
- Modified subset_to_bbox function to slice based on edges rather than centers [!10](https://github.com/dmidk/sunflow/pull/10), @KristianHMoller
- Pass down number of ensembles from configurations to `ProbabilisticAdvection`. This gives a roughly 3x speedup in the no ensembles case [!12](https://github.com/dmidk/sunflow/pull/12), @JoachimKoenigslieb

## [v1.1.0]

### Added

- Added pre-commit configuration for linting and formatting [!6](https://github.com/dmidk/sunflow/pull/6), @khintz
- Add build and publish container workflow [!8](https://github.com/dmidk/sunflow/pull/8), @khintz

### Changed

- Add .gitignore file and remove unnecessary files from the repository [!1](https://github.com/dmidk/sunflow/pull/1), @khintz
- Optimize container image size using multi-stage build and removing build-time dependencies [#9](https://github.com/dmidk/sunflow/pull/9), @khintz

## [v1.0.0]

Initial release of Sunflow. The application performs real-time or historical short-term solar irradiance nowcasting using probabilistic advection (via [SolarSTEPS](https://github.com/EnergyWeatherAI/SolarSTEPS)) with satellite data from KNMI MSG-CPP or DWD SIS. It supports local file, API download, and S3 run modes, produces CF-compliant NetCDF4 forecast files, and can be run as a container via Podman/Docker.
