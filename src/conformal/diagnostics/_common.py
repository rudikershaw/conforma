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
class _CoverageStabilityResult:
    """Base fields shared by coverage stability results.

    Parameters
    ----------
    sizes : NDArray[np.intp]
        Calibration set sizes that were evaluated.
    mean_coverage : NDArray[np.float64]
        Mean empirical coverage at each size, averaged over repetitions.
    std_coverage : NDArray[np.float64]
        Standard deviation of empirical coverage at each size.

    """

    sizes: NDArray[np.intp]
    mean_coverage: NDArray[np.float64]
    std_coverage: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _CalibrationPlan:
    """Base fields shared by calibration plan results.

    The top-level ``coverage`` and ``cal_size`` attributes are the
    recommended operating point: the highest empirical coverage where
    the second metric (prediction set size or interval width) stays
    within the user's constraint, and the smallest calibration set size
    that achieves it reliably.

    The remaining attributes represent the columns in a table, indexed
    by ``mean_coverage``. A row is represented by any index ``i`` across
    each array.

    Parameters
    ----------
    coverage : float
        Recommended coverage level (empirical, measured over trials).
    cal_size : int
        Recommended minimum calibration set size.
    mean_coverage : NDArray[np.float64]
        Mean empirical coverage at each operating point.
        Shape: ``(n_rows,)``
    std_coverage : NDArray[np.float64]
        Standard deviation of empirical coverage at each operating point.
        Shape: ``(n_rows,)``
    cal_sizes : NDArray[np.intp]
        Smallest calibration set size that reliably achieves each row's
        coverage level. Shape: ``(n_rows,)``

    """

    coverage: float
    cal_size: int
    mean_coverage: NDArray[np.float64]
    std_coverage: NDArray[np.float64]
    cal_sizes: NDArray[np.intp]


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


def _validate_shapes(n_predictions: int, n_labels: int) -> None:
    if n_labels != n_predictions:
        msg = f"Shape mismatch: got {n_predictions} model outputs but {n_labels} true values."
        raise ValueError(msg)


def _validate_n_repetitions(n_repetitions: int) -> None:
    if n_repetitions < 1:
        msg = f"n_repetitions must be a positive integer, got {n_repetitions}."
        raise ValueError(msg)


def _validate_diagnostic_inputs(n_predictions: int, n_labels: int, coverage: float, config: DiagnosticConfig) -> None:
    _validate_shapes(n_predictions, n_labels)
    _validate_n_repetitions(config.n_repetitions)

    if coverage <= 0 or coverage >= 1:
        msg = f"coverage must be in (0, 1), got {coverage}."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class _SweepSpec:
    """How to subsample and trial: the "sampling plan" half of a sweep call."""

    n_data: int
    sizes: NDArray[np.intp]
    n_repetitions: int
    generator: np.random.Generator


@dataclass(frozen=True, slots=True)
class _SweepSummary:
    """Summary of a size sweep: coverage and a second metric per size."""

    mean_coverage: NDArray[np.float64]
    std_coverage: NDArray[np.float64]
    mean_metric: NDArray[np.float64]
    std_metric: NDArray[np.float64]


def _sweep_sizes[F: np.floating[Any]](
    spec: _SweepSpec,
    coverage: float,
    trial_fn: Callable[[NDArray[np.intp], NDArray[np.intp], float], tuple[F, F]],
) -> _SweepSummary:
    """Run ``n_repetitions`` trials at each size and summarise results."""
    mean_coverages: list[float] = []
    std_coverages: list[float] = []
    mean_metrics: list[float] = []
    std_metrics: list[float] = []

    for size in spec.sizes:
        trial_coverages: list[F] = []
        trial_metrics: list[F] = []
        n_cal = int(size) // 2

        for _ in range(spec.n_repetitions):
            indices = spec.generator.choice(spec.n_data, size=int(size), replace=False)
            cal_idx, test_idx = indices[:n_cal], indices[n_cal:]
            cov, metric = trial_fn(cal_idx, test_idx, coverage)
            trial_coverages.append(cov)
            trial_metrics.append(metric)

        mean_coverages.append(float(np.mean(trial_coverages)))
        std_coverages.append(float(np.std(trial_coverages)))
        mean_metrics.append(float(np.mean(trial_metrics)))
        std_metrics.append(float(np.std(trial_metrics)))

    return _SweepSummary(
        mean_coverage=np.array(mean_coverages),
        std_coverage=np.array(std_coverages),
        mean_metric=np.array(mean_metrics),
        std_metric=np.array(std_metrics),
    )


def _run_coverage_stability[F: np.floating[Any]](
    n_data: int,
    coverage: float,
    config: DiagnosticConfig,
    trial_fn: Callable[[NDArray[np.intp], NDArray[np.intp], float], tuple[F, F]],
) -> tuple[NDArray[np.intp], _SweepSummary]:
    """Run the shared subsampling loop for coverage stability diagnostics."""
    rng = config.rng
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    evaluated_sizes = _resolve_sizes(config.sizes, n_data, coverage)
    spec = _SweepSpec(n_data, evaluated_sizes, config.n_repetitions, generator)

    summary = _sweep_sizes(spec, coverage, trial_fn)
    return evaluated_sizes, summary


_BINARY_SEARCH_STEPS = 10


@dataclass(frozen=True, slots=True)
class _PlanTable:
    """Raw calibration-plan table, ready to be wrapped in a concrete type."""

    best_coverage: float
    best_cal_size: int
    mean_coverage: NDArray[np.float64]
    std_coverage: NDArray[np.float64]
    cal_sizes: NDArray[np.intp]
    mean_metric: NDArray[np.float64]
    std_metric: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _SearchResults:
    coverages: list[float]
    mean_coverage: list[NDArray[np.float64]]
    std_coverage: list[NDArray[np.float64]]
    mean_metric: list[NDArray[np.float64]]
    std_metric: list[NDArray[np.float64]]


def _run_calibration_plan_search[F: np.floating[Any]](
    n_data: int,
    max_metric: float,
    config: DiagnosticConfig,
    trial_fn: Callable[[NDArray[np.intp], NDArray[np.intp], float], tuple[F, F]],
) -> _PlanTable:
    """Binary-search coverage and sweep sizes to build a calibration plan table."""
    _validate_n_repetitions(config.n_repetitions)

    rng = config.rng
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    low, high = 0.5, 1.0
    # _minimum_size(1.0) divides by zero, so resolve sizes at the first midpoint.
    # Coverage=1.0 is also degenerate (every class is included) so it's not a useful search boundary.
    evaluated_sizes = _resolve_sizes(config.sizes, n_data, (low + high) / 2)

    # Cap the search ceiling so every evaluated cal size can support every tried coverage.
    # For the regressor, coverage > n_cal / (n_cal + 1) is infeasible; classifier is unaffected.
    smallest_n_cal = int(evaluated_sizes[0]) // 2
    high = min(high, smallest_n_cal / (smallest_n_cal + 1))

    spec = _SweepSpec(n_data, evaluated_sizes, config.n_repetitions, generator)

    results = _SearchResults([], [], [], [], [])

    for _ in range(_BINARY_SEARCH_STEPS):
        mid = (low + high) / 2
        summary = _sweep_sizes(spec, mid, trial_fn)

        results.coverages.append(mid)
        results.mean_coverage.append(summary.mean_coverage)
        results.std_coverage.append(summary.std_coverage)
        results.mean_metric.append(summary.mean_metric)
        results.std_metric.append(summary.std_metric)

        # Use the best (smallest) mean metric across cal sizes to drive the search:
        # larger cal sizes tighten sets/intervals, so this asks "could any cal size meet the budget?"
        if min(summary.mean_metric) <= max_metric:
            low = mid
        else:
            high = mid

    return _build_plan_table(results, evaluated_sizes, max_metric)


def _build_plan_table(
    results: _SearchResults,
    evaluated_sizes: NDArray[np.intp],
    max_metric: float,
) -> _PlanTable:
    """Select the best calibration size per coverage level and build the table."""
    table_cal_sizes: list[int] = []
    table_mean_coverage: list[float] = []
    table_std_coverage: list[float] = []
    table_mean_metric: list[float] = []
    table_std_metric: list[float] = []

    for idx in range(len(results.coverages)):
        mean_cov_at_sizes = results.mean_coverage[idx]
        # Pick the smallest cal size whose empirical coverage meets the binary-search target.
        meets = mean_cov_at_sizes >= results.coverages[idx]
        best_idx = int(np.argmax(meets)) if np.any(meets) else int(np.argmax(mean_cov_at_sizes))
        table_cal_sizes.append(int(evaluated_sizes[best_idx]))
        table_mean_coverage.append(float(mean_cov_at_sizes[best_idx]))
        table_std_coverage.append(float(results.std_coverage[idx][best_idx]))
        table_mean_metric.append(float(results.mean_metric[idx][best_idx]))
        table_std_metric.append(float(results.std_metric[idx][best_idx]))

    order = np.argsort(table_mean_coverage)
    sorted_mean_coverage = np.array(table_mean_coverage)[order]
    sorted_cal_sizes = np.array(table_cal_sizes, dtype=np.intp)[order]
    sorted_std_coverage = np.array(table_std_coverage)[order]
    sorted_mean_metric = np.array(table_mean_metric)[order]
    sorted_std_metric = np.array(table_std_metric)[order]

    # Default to the lowest-coverage row as a fallback if no row meets the budget
    # (the lowest coverage has the smallest prediction sets, so it's the closest feasible option).
    best_coverage = float(sorted_mean_coverage[0])
    best_size = int(sorted_cal_sizes[0])
    for i in range(len(sorted_mean_coverage) - 1, -1, -1):
        if sorted_mean_metric[i] <= max_metric:
            best_coverage = float(sorted_mean_coverage[i])
            best_size = int(sorted_cal_sizes[i])
            break

    return _PlanTable(
        best_coverage=best_coverage,
        best_cal_size=best_size,
        mean_coverage=sorted_mean_coverage,
        std_coverage=sorted_std_coverage,
        cal_sizes=sorted_cal_sizes,
        mean_metric=sorted_mean_metric,
        std_metric=sorted_std_metric,
    )
