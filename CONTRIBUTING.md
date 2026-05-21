# Contributing to conforma

## What this project is

`conforma` is a library for making conformal prediction accessible to ML practitioners who are not researchers. It has a deliberately narrow scope and a small API surface. Before opening a PR, it is worth understanding the principles that guide what gets included.

**Framework agnostic.** Most ML uncertainty libraries are tied to a specific framework. We think that is a mistake. `conforma` accepts any callable that returns scores or predictions, so it works with scikit-learn, PyTorch, LightGBM, or anything else.

**Minimal dependencies.** Every dependency is a maintenance burden and a potential source of breakage for users. numpy is the only runtime dependency, and we are not looking to add more. If a feature would require pulling in another package, it does not belong in the core.

**Opinionated and small.** The conformal prediction literature contains dozens of score functions, predictor variants, and extensions. We deliberately do not implement all of them. We would rather do a small number of things extremely well than become a research toolkit. If a feature adds complexity without a clear benefit to the typical user, it probably does not belong here.

Contributions that align with these principles are very welcome. If you are unsure whether something fits, open an issue to discuss it first.

## Prerequisites

### uv

This project uses [uv](https://github.com/astral-sh/uv) (0.11+) for dependency management, virtual environment management, and task running. To install uv, see the [official installation documentation](https://docs.astral.sh/uv/getting-started/installation/).

We recommend using `uv run` to run any project commands, to maintain a consistent python version and environment.

## Getting Started

1. Fork the repository on GitHub, then clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/conforma.git
   cd conforma
   ```

2. Install dependencies (including dev dependencies):
   ```bash
   uv sync --group dev
   ```

3. Verify everything works:
   ```bash
   uv run nox
   ```

## Development Workflow

This project uses [Nox](https://nox.thea.codes/) for task running. All development tasks (linting, formatting, testing) are defined as Nox sessions. While you work on this project, we recommend you use `uv run nox` to run any linting or checks that you need. For specific tasks, review the `noxfile.py` or use `uv run nox -l` in your terminal.

## Submitting Changes

If you are fixing a backward compatible bug then create a branch from `main`, otherwise create your branch off `develop`. Then make your changes, and open a pull request. Before submitting, make sure `uv run nox` passes. This runs linting, type checking, and the full test suite. If it passes, your PR is ready for review.
