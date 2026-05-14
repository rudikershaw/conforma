"""Shared types and utilities for diagnostic functions."""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DiagnosticConfig:
    """Configuration for diagnostic functions.

    Parameters
    ----------
    sizes : Iterable[int] | None
        Calibration set sizes to evaluate. If ``None``, a sensible
        default range is inferred from the data size and coverage level.
    n_repetitions : int
        Number of random subsamples per size. Defaults to 100.
    rng : np.random.Generator | int | None
        Random number generator or seed for reproducibility.
        If ``None``, a default random number generator will be used.

    """

    sizes: Iterable[int] | None = None
    n_repetitions: int = 100
    rng: np.random.Generator | int | None = None


@dataclass(frozen=True, slots=True)
class CoverageStabilityResult:
    """Results from a coverage stability analysis.

    Parameters
    ----------
    sizes : NDArray[np.intp]
        Calibration set sizes that were evaluated.
    mean_coverage : NDArray[np.float64]
        Mean empirical coverage at each size, averaged over repetitions.
    std_coverage : NDArray[np.float64]
        Standard deviation of empirical coverage at each size.
    mean_set_size : NDArray[np.float64]
        Mean prediction set size at each size, averaged over repetitions.
    std_set_size : NDArray[np.float64]
        Standard deviation of prediction set size at each size.

    """

    sizes: NDArray[np.intp]
    mean_coverage: NDArray[np.float64]
    std_coverage: NDArray[np.float64]
    mean_set_size: NDArray[np.float64]
    std_set_size: NDArray[np.float64]


def _minimum_size(coverage: float) -> int:
    min_cal = math.ceil(coverage / (1 - coverage))
    return max(min_cal * 2 + 2, 4)


def _default_sizes(n_data: int, coverage: float) -> NDArray[np.intp]:
    min_size = _minimum_size(coverage)
    max_size = n_data

    if min_size > max_size:
        msg = (
            f"Not enough data: {n_data} examples is too few to evaluate coverage={coverage}. Need at least {min_size}."
        )
        raise ValueError(msg)

    n_points = min(10, max_size - min_size + 1)
    sizes = np.unique(np.geomspace(min_size, max_size, num=n_points).astype(np.intp))
    return sizes


def _resolve_sizes(
    sizes: Iterable[int] | None,
    n_data: int,
    coverage: float,
) -> NDArray[np.intp]:
    """Resolve and validate the sizes to evaluate."""
    if sizes is None:
        return _default_sizes(n_data, coverage)

    evaluated = np.unique(np.fromiter(sizes, dtype=np.intp))
    min_size = _minimum_size(coverage)

    for size in evaluated:
        if size < min_size:
            msg = f"Size {size} is too small for coverage={coverage}. Minimum size is {min_size}."
            raise ValueError(msg)
        if size > n_data:
            msg = f"Size {size} exceeds the number of available examples ({n_data})."
            raise ValueError(msg)

    return evaluated


def _validate_diagnostic_inputs(n_predictions: int, n_labels: int, coverage: float, config: DiagnosticConfig) -> None:
    if n_labels != n_predictions:
        msg = f"Shape mismatch: got {n_predictions} model outputs but {n_labels} true values."
        raise ValueError(msg)

    if coverage <= 0 or coverage >= 1:
        msg = f"coverage must be in (0, 1), got {coverage}."
        raise ValueError(msg)

    if config.n_repetitions < 1:
        msg = f"n_repetitions must be a positive integer, got {config.n_repetitions}."
        raise ValueError(msg)


def _run_coverage_stability[F: np.floating[Any]](
    n_data: int,
    coverage: float,
    config: DiagnosticConfig,
    trial_fn: Callable[[NDArray[np.intp], NDArray[np.intp], float], tuple[F, F]],
) -> CoverageStabilityResult:
    """Run the shared subsampling loop for coverage stability diagnostics.

    Parameters
    ----------
    n_data : int
        Total number of examples in the dataset.
    coverage : float
        Target coverage level, already validated.
    config : DiagnosticConfig
        Configuration, already validated.
    trial_fn : Callable
        Callback that receives ``(cal_idx, test_idx, coverage)`` and
        returns ``(coverage_value, set_size_value)`` for one trial.

    """
    rng = config.rng
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    evaluated_sizes = _resolve_sizes(config.sizes, n_data, coverage)

    mean_coverages: list[F] = []
    std_coverages: list[F] = []
    mean_set_sizes: list[F] = []
    std_set_sizes: list[F] = []

    for size in evaluated_sizes:
        trial_coverages: list[F] = []
        trial_set_sizes: list[F] = []
        n_cal = int(size) // 2

        for _ in range(config.n_repetitions):
            indices = generator.choice(n_data, size=int(size), replace=False)
            cal_idx, test_idx = indices[:n_cal], indices[n_cal:]
            cov, set_size = trial_fn(cal_idx, test_idx, coverage)
            trial_coverages.append(cov)
            trial_set_sizes.append(set_size)

        mean_coverages.append(np.mean(trial_coverages))
        std_coverages.append(np.std(trial_coverages))
        mean_set_sizes.append(np.mean(trial_set_sizes))
        std_set_sizes.append(np.std(trial_set_sizes))

    return CoverageStabilityResult(
        sizes=evaluated_sizes,
        mean_coverage=np.array(mean_coverages),
        std_coverage=np.array(std_coverages),
        mean_set_size=np.array(mean_set_sizes),
        std_set_size=np.array(std_set_sizes),
    )
