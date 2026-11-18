# Contributing to sunflow

Thank you for your interest in contributing to **sunflow**! This is a collaborative project based at the Danish Meteorological Institute for implementing open-source satellite-based solar nowcasting.

## Getting Started

Before contributing, follow the local development setup steps in the README.

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/sunflow.git
   cd sunflow
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```


## Development Guidelines

### Tests

Run `uv run pytest` and `uv run pre-commit run --all-files` before opening a
pull request. See [Running the tests](README.md#running-the-tests) in the README
for the available commands and for what the offline suite does and does not cover.

When adding tests:

- Put them in the module mirroring the code you touched (`tests/test_forecast.py`
  for `sunflow/forecast.py`, and so on). Full-pipeline runs belong in
  `tests/test_run_nowcast.py`.
- Reuse the fixtures in `tests/conftest.py` for configurations, synthetic
  datasets, temporary NetCDF files and the mocked S3 filesystem rather than
  rebuilding them.
- Keep tests offline and deterministic: no network, no credentials, no reliance
  on the current date or on test execution order. Use `tmp_path` for files.
- Prefer small numerical examples whose expected result you can work out by
  hand, instead of re-implementing the function under test in the assertion.
- Name tests after the behaviour they pin down, and assert on that behaviour
  rather than on incidental details.

### Commits

- Write clear and concise commit messages.
- Use the imperative mood in the subject line (e.g., "Add solar irradiance model").
- Reference relevant issues or pull requests where applicable.
- Add an entry to CHANGELOG.md describing your change under the [Unreleased] section.

## Submitting Changes

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Open a **Pull Request** against the `main` branch of `dmidk/sunflow`.
3. Describe your changes clearly in the pull request description.
4. Ensure all tests pass and there are no linting errors.
5. A maintainer will review your pull request and may request changes.

## Reporting Issues

If you find a bug or have a feature request, please [open an issue](https://github.com/dmidk/sunflow/issues) on GitHub with a clear description and, where relevant, a minimal reproducible example.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
