# Contributing to MSig

Thanks for your interest in MSig. This document explains how to set up a
development environment and submit changes.

## Development setup

```bash
git clone https://github.com/MiguelGarcaoSilva/msig.git
cd msig
uv sync --extra experiments --extra dev   # installs everything you need
```

## Running tests

```bash
uv run pytest tests/ -q                                 # unit tests
uv run pytest -m integration -v                         # smoke tests for experiment scripts
uv run pytest -m slow -v                                # paper-reproducibility regression suite
uv run pytest --cov=msig --cov-report=term-missing      # with coverage
```

## Formatting and linting

```bash
uv run black msig/ tests/ examples/ experiments/ scripts/
uv run isort msig/ tests/ examples/ experiments/ scripts/
uv run mypy msig/
```

## Pull requests

- Branch off `main`. Use a short, kebab-case branch name (`fix/2d-gaussian-cdf`).
- Keep commits focused; conventional-commit prefixes (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Reference any related issue in the PR description.
- Include tests for behaviour changes.

## Reproducing the paper

See `REPRODUCING_EXPERIMENTS.md`.

## License

By contributing, you agree that your contributions are licensed under the MIT
License (see `LICENSE`).
