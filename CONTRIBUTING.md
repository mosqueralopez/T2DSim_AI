# Contributor Guide

## Installation

```bash
pip install -e ".[dev]"
pre-commit install
```

## Before pushing

```bash
pre-commit run --all-files
python -m pytest tests/
```

Refresh bundled digital twins from the research codebase:

```bash
python scripts/sync_digital_twins.py
```

## Pull requests

Fork the repository, apply changes, and open a PR at
https://github.com/mosqueralopez/T2DSim_AI/pulls
