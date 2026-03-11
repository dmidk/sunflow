# Contributing to sunflow

Thank you for your interest in contributing to **sunflow**! This is a collaborative project based at the Danish Meteorological Institute.

## Getting Started

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

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code style.
- Use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:
  ```bash
  ruff check .
  ruff format .
  ```

### Testing

- Write tests for all new functionality using [pytest](https://pytest.org).
- Place tests in the `tests/` directory, mirroring the package structure.
- Run the test suite before submitting:
  ```bash
  pytest
  ```

### Commits

- Write clear and concise commit messages.
- Use the imperative mood in the subject line (e.g., "Add solar irradiance model").
- Reference relevant issues or pull requests where applicable.

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
