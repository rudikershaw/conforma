"""Diagnostics for regressors: coverage stability and calibration planning."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from conforma.calibration import calibrate_regressor
from conforma.core import compute_quantile
from conforma.diagnostics._common import (
    DiagnosticConfig,
    _CalibrationPlan,
    _CoverageStabilityResult,
    _run_calibration_plan_search,
    _run_coverage_stability,
    _validate_diagnostic_inputs,
    _validate_shapes,
)


@dataclass(frozen=True, slots=True)
class RegressorCoverageStability(_CoverageStabilityResult):
    """Results from a regressor coverage stability analysis.

    Parameters
    ----------
    sizes : NDArray[np.intp]
        Calibration set sizes that were evaluated.
    mean_coverage : NDArray[np.float64]
        Mean empirical coverage at each size, averaged over repetitions.
    std_coverage : NDArray[np.float64]
        Standard deviation of empirical coverage at each size.
    mean_interval_width : NDArray[np.float64]
        Mean prediction interval width at each size, averaged over repetitions.
    std_interval_width : NDArray[np.float64]
        Standard deviation of prediction interval width at each size.

    """

    mean_interval_width: NDArray[np.float64]
    std_interval_width: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RegressorCalibrationPlan(_CalibrationPlan):
    """Results from a regressor calibration plan analysis.

    The top-level ``coverage`` and ``cal_size`` attributes are the
    recommended operating point: the highest empirical coverage where
    the mean prediction interval width stays within the user's
    constraint, and the smallest calibration set size that achieves it
    reliably.

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
    mean_interval_width : NDArray[np.float64]
        Mean prediction interval width at each operating point.
        Shape: ``(n_rows,)``
    std_interval_width : NDArray[np.float64]
        Standard deviation of prediction interval width at each operating point.
        Shape: ``(n_rows,)``

    """

    mean_interval_width: NDArray[np.float64]
    std_interval_width: NDArray[np.float64]


def regressor_coverage_stability[F: np.floating[Any]](
    calibration_predictions: NDArray[F],
    true_values: NDArray[F],
    coverage: float = 0.90,
    config: DiagnosticConfig | None = None,
) -> RegressorCoverageStability:
    """Measure how calibration set size affects prediction interval quality.

    Subsamples the provided data at various sizes, runs the full
    calibrate-and-predict loop at each size, and reports how empirical
    coverage and interval width vary. Use this to decide how much
    calibration data you need.

    Parameters
    ----------
    calibration_predictions : NDArray[F]
        Predicted values on the calibration set.
        Shape: ``(n_examples,)`` for univariate or
        ``(n_examples, n_outputs)`` for multi-output regression.
    true_values : NDArray[F]
        True values for the calibration set. Same shape as
        ``calibration_predictions``.
    coverage : float
        Target coverage level in (0, 1). Defaults to 0.90.
    config : DiagnosticConfig
        Configuration for the diagnostic. Controls which sizes to
        evaluate, how many repetitions to run, and the random seed.

    Returns
    -------
    RegressorCoverageStability
        A frozen dataclass containing the sizes evaluated and the mean
        and standard deviation of empirical coverage and prediction
        interval width at each.

    Examples
    --------
    >>> import numpy as np
    >>> from conforma.diagnostics import regressor_coverage_stability, DiagnosticConfig
    >>> preds = np.array([2.1, 5.3, 7.8, 3.2, 1.0, 4.5, 6.2, 8.1, 2.5, 3.8])
    >>> true = np.array([2.0, 5.0, 8.0, 3.5, 1.2, 4.0, 6.0, 8.5, 2.8, 3.5])
    >>> config = DiagnosticConfig(sizes=[6, 8], n_repetitions=10, rng=42)
    >>> result = regressor_coverage_stability(preds, true, coverage=0.5, config=config)

    """
    if config is None:
        config = DiagnosticConfig()

    n_data = calibration_predictions.shape[0]
    _validate_diagnostic_inputs(n_data, true_values.shape[0], coverage, config)

    trial_fn = _make_regressor_trial_fn(calibration_predictions, true_values)
    sizes, summary = _run_coverage_stability(n_data, coverage, config, trial_fn)
    return RegressorCoverageStability(
        sizes=sizes,
        mean_coverage=summary.mean_coverage,
        std_coverage=summary.std_coverage,
        mean_interval_width=summary.mean_metric,
        std_interval_width=summary.std_metric,
    )


def regressor_calibration_plan[F: np.floating[Any]](
    calibration_predictions: NDArray[F],
    true_values: NDArray[F],
    max_interval_width: float,
    config: DiagnosticConfig | None = None,
) -> RegressorCalibrationPlan:
    """Find the best coverage level and calibration set size for a regressor.

    Searches over coverage levels to find the highest coverage where mean
    prediction interval width stays within ``max_interval_width``. At each
    coverage level, evaluates multiple calibration set sizes to find the
    smallest that achieves stable coverage.

    Parameters
    ----------
    calibration_predictions : NDArray[F]
        Predicted values on the calibration set.
        Shape: ``(n_examples,)`` for univariate or
        ``(n_examples, n_outputs)`` for multi-output regression.
    true_values : NDArray[F]
        True values for the calibration set. Same shape as
        ``calibration_predictions``.
    max_interval_width : float
        Maximum acceptable mean prediction interval width, in the
        target's units. There is no sensible default — pick a value
        that reflects the precision you need for your use case.
    config : DiagnosticConfig | None
        Configuration for the diagnostic. Controls which calibration sizes
        to evaluate, how many repetitions to run, and the random seed.

    Returns
    -------
    RegressorCalibrationPlan
        The recommended coverage and calibration size, plus the full grid
        of results from the search for inspection.

    """
    if config is None:
        config = DiagnosticConfig()

    n_data = calibration_predictions.shape[0]
    _validate_shapes(n_data, true_values.shape[0])

    trial_fn = _make_regressor_trial_fn(calibration_predictions, true_values)
    table = _run_calibration_plan_search(n_data, max_interval_width, config, trial_fn)

    return RegressorCalibrationPlan(
        coverage=table.best_coverage,
        cal_size=table.best_cal_size,
        mean_coverage=table.mean_coverage,
        std_coverage=table.std_coverage,
        cal_sizes=table.cal_sizes,
        mean_interval_width=table.mean_metric,
        std_interval_width=table.std_metric,
    )


def _make_regressor_trial_fn[F: np.floating[Any]](
    calibration_predictions: NDArray[F],
    true_values: NDArray[F],
) -> Callable[[NDArray[np.intp], NDArray[np.intp], float], tuple[F, F]]:
    """Build a trial function that scores a single cal/test split at a given coverage."""

    def trial_fn(cal_idx: NDArray[np.intp], test_idx: NDArray[np.intp], coverage: float) -> tuple[F, F]:
        scores = calibrate_regressor(
            calibration_predictions[cal_idx],
            true_values[cal_idx],
        )
        quantile = compute_quantile(scores, coverage)

        test_preds = calibration_predictions[test_idx]
        test_true = true_values[test_idx]
        lower = test_preds - quantile
        upper = test_preds + quantile

        covered = (test_true >= lower) & (test_true <= upper)
        widths = upper - lower
        return covered.mean(), widths.mean()

    return trial_fn
