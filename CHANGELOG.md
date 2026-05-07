# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added a check for the number of ensemble members, as the code currently supports only one [!13](https://github.com/dmidk/sunflow/pull/13), @KristianHMoller

## [v1.1.0]

### Added

- Added pre-commit configuration for linting and formatting [!6](https://github.com/dmidk/sunflow/pull/6), @khintz
- Add build and publish container workflow [!8](https://github.com/dmidk/sunflow/pull/8), @khintz

### Changed

- Add .gitignore file and remove unnecessary files from the repository [!1](https://github.com/dmidk/sunflow/pull/1), @khintz
- Optimize container image size using multi-stage build and removing build-time dependencies [#9](https://github.com/dmidk/sunflow/pull/9), @khintz

## [v1.0.0]

Initial release of Sunflow. The application performs real-time or historical short-term solar irradiance nowcasting using probabilistic advection (via [SolarSTEPS](https://github.com/EnergyWeatherAI/SolarSTEPS)) with satellite data from KNMI MSG-CPP or DWD SIS. It supports local file, API download, and S3 run modes, produces CF-compliant NetCDF4 forecast files, and can be run as a container via Podman/Docker.
