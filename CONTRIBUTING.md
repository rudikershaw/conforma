# Contributing to conformal

## Prerequisites

### uv

This project uses [uv](https://github.com/astral-sh/uv) for dependency management, virtual environment management, and task running. To install uv, see the [official installation documentation](https://docs.astral.sh/uv/getting-started/installation/).

The project was initialised with uv 0.11.6. You can verify that you have 0.11 or later using `uv --version`.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_FORK/conformal.git
   cd conformal
   ```

2. Install dependencies (including dev dependencies):
   ```bash
   uv sync --group dev
   ```

## Development Workflow

This project uses [Nox](https://nox.thea.codes/) for task running. All development tasks (linting, formatting, testing) are defined as Nox sessions.

### Running checks

```bash
# Run all default checks.
uv run nox
```
You can view the individual nox tasks in the `noxfile.py`.
