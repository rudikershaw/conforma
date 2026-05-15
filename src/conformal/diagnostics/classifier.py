"""Diagnostics for classifiers: coverage stability and calibration planning."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from conformal.calibration import calibrate_classifier
from conformal.core import compute_p_values
from conformal.diagnostics._common import (
    DiagnosticConfig,
    _CalibrationPlan,
    _CoverageStabilityResult,
    _run_calibration_plan_search,
    _run_coverage_stability,
    _validate_diagnostic_inputs,
    _validate_shapes,
)


@dataclass(frozen=True, slots=True)
class ClassifierCoverageStability(_CoverageStabilityResult):
    """Results from a classifier coverage stability analysis.

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

    mean_set_size: NDArray[np.float64]
    std_set_size: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ClassifierCalibrationPlan(_CalibrationPlan):
    """Results from a classifier calibration plan analysis.

    The top-level ``coverage`` and ``cal_size`` attributes are the
    recommended operating point: the highest empirical coverage where
    the mean prediction set size stays within the user's constraint,
    and the smallest calibration set size that achieves it reliably.

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
    mean_set_size : NDArray[np.float64]
        Mean prediction set size at each operating point.
        Shape: ``(n_rows,)``
    std_set_size : NDArray[np.float64]
        Standard deviation of prediction set size at each operating point.
        Shape: ``(n_rows,)``

    """

    mean_set_size: NDArray[np.float64]
    std_set_size: NDArray[np.float64]


def classifier_coverage_stability[F: np.floating[Any], I: np.integer[Any]](
    calibration_probabilities: NDArray[F],
    true_labels: NDArray[I],
    coverage: float = 0.90,
    config: DiagnosticConfig | None = None,
) -> ClassifierCoverageStability:
    """Measure how calibration set size affects prediction set quality.

    Subsamples the provided data at various sizes, runs the full
    calibrate-and-predict loop at each size, and reports how empirical
    coverage and prediction set size vary. Use this to decide how much
    calibration data you need.

    Parameters
    ----------
    calibration_probabilities : NDArray[F]
        Model output scores on the calibration set (e.g. probabilities, logits).
        Shape: ``(n_examples, n_classes)``
    true_labels : NDArray[I]
        True class labels for the calibration set.
        Shape: ``(n_examples,)``
    coverage : float
        Target coverage level in (0, 1). Defaults to 0.90.
    config : DiagnosticConfig
        Configuration for the diagnostic. Controls which sizes to
        evaluate, how many repetitions to run, and the random seed.

    Returns
    -------
    ClassifierCoverageStability
        A frozen dataclass containing the sizes evaluated and the mean
        and standard deviation of empirical coverage and prediction set
        size at each.

    Examples
    --------
    >>> import numpy as np
    >>> from conformal.diagnostics import classifier_coverage_stability, DiagnosticConfig
    >>> probs = np.array([[0.9, 0.1]] * 30 + [[0.5, 0.5]] * 20)
    >>> labels = np.array([0] * 30 + [1] * 20)
    >>> config = DiagnosticConfig(sizes=[20, 40], n_repetitions=10, rng=42)
    >>> result = classifier_coverage_stability(probs, labels, coverage=0.5, config=config)

    """
    if config is None:
        config = DiagnosticConfig()

    n_data = calibration_probabilities.shape[0]
    _validate_diagnostic_inputs(n_data, true_labels.shape[0], coverage, config)

    trial_fn = _make_classifier_trial_fn(calibration_probabilities, true_labels)
    sizes, summary = _run_coverage_stability(n_data, coverage, config, trial_fn)
    return ClassifierCoverageStability(
        sizes=sizes,
        mean_coverage=summary.mean_coverage,
        std_coverage=summary.std_coverage,
        mean_set_size=summary.mean_metric,
        std_set_size=summary.std_metric,
    )


def classifier_calibration_plan[F: np.floating[Any], I: np.integer[Any]](
    calibration_probabilities: NDArray[F],
    true_labels: NDArray[I],
    max_set_size: float = 1.05,
    config: DiagnosticConfig | None = None,
) -> ClassifierCalibrationPlan:
    """Find the best coverage level and calibration set size for a classifier.

    Searches over coverage levels to find the highest coverage where mean
    prediction set size stays within ``max_set_size``. At each coverage
    level, evaluates multiple calibration set sizes to find the smallest
    that achieves stable coverage.

    Parameters
    ----------
    calibration_probabilities : NDArray[F]
        Model output scores on the calibration set (e.g. probabilities, logits).
        Shape: ``(n_examples, n_classes)``
    true_labels : NDArray[I]
        True class labels for the calibration set.
        Shape: ``(n_examples,)``
    max_set_size : float
        Maximum acceptable mean prediction set size. Defaults to 1.05,
        meaning at most ~5% of predictions will include more than one class.
    config : DiagnosticConfig | None
        Configuration for the diagnostic. Controls which calibration sizes
        to evaluate, how many repetitions to run, and the random seed.

    Returns
    -------
    ClassifierCalibrationPlan
        The recommended coverage and calibration size, plus the full grid
        of results from the search for inspection.

    """
    if config is None:
        config = DiagnosticConfig()

    n_data = calibration_probabilities.shape[0]
    _validate_shapes(n_data, true_labels.shape[0])

    trial_fn = _make_classifier_trial_fn(calibration_probabilities, true_labels)
    table = _run_calibration_plan_search(n_data, max_set_size, config, trial_fn)

    return ClassifierCalibrationPlan(
        coverage=table.best_coverage,
        cal_size=table.best_cal_size,
        mean_coverage=table.mean_coverage,
        std_coverage=table.std_coverage,
        cal_sizes=table.cal_sizes,
        mean_set_size=table.mean_metric,
        std_set_size=table.std_metric,
    )


def _make_classifier_trial_fn[F: np.floating[Any], I: np.integer[Any]](
    calibration_probabilities: NDArray[F],
    true_labels: NDArray[I],
) -> Callable[[NDArray[np.intp], NDArray[np.intp], float], tuple[F, F]]:
    """Build a trial function that scores a single cal/test split at a given coverage."""
    one = np.array(1, dtype=calibration_probabilities.dtype)

    def trial_fn(cal_idx: NDArray[np.intp], test_idx: NDArray[np.intp], coverage: float) -> tuple[F, F]:
        scores = calibrate_classifier(calibration_probabilities[cal_idx], true_labels[cal_idx])
        p_values = compute_p_values(scores, one - calibration_probabilities[test_idx])
        prediction_sets = p_values >= 1 - coverage

        hits = prediction_sets[np.arange(len(test_idx)), true_labels[test_idx]]
        return hits.mean(), prediction_sets.sum(axis=1).mean()

    return trial_fn
