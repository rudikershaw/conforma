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

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Run commands using `uv run`:
   ```bash
   uv run python -m pytest
   uv run python
   ```
