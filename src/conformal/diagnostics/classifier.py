"""Coverage stability diagnostic for classifiers."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from conformal.calibration import calibrate_classifier
from conformal.core import compute_p_values
from conformal.diagnostics._common import (
    CalibrationPlan,
    CoverageStabilityResult,
    DiagnosticConfig,
    _resolve_sizes,
    _run_coverage_stability,
    _validate_diagnostic_inputs,
)


def classifier_coverage_stability[F: np.floating[Any], I: np.integer[Any]](
    calibration_probabilities: NDArray[F],
    true_labels: NDArray[I],
    coverage: float = 0.90,
    config: DiagnosticConfig | None = None,
) -> CoverageStabilityResult:
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
    CoverageStabilityResult
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

    one = np.array(1, dtype=calibration_probabilities.dtype)

    def trial_fn(cal_idx: NDArray[np.intp], test_idx: NDArray[np.intp], coverage: float) -> tuple[F, F]:
        scores = calibrate_classifier(
            calibration_probabilities[cal_idx],
            true_labels[cal_idx],
        )
        p_values = compute_p_values(scores, one - calibration_probabilities[test_idx])
        prediction_sets = p_values >= 1 - coverage

        hits = prediction_sets[np.arange(len(test_idx)), true_labels[test_idx]]
        return hits.mean(), prediction_sets.sum(axis=1).mean()

    return _run_coverage_stability(n_data, coverage, config, trial_fn)


_BINARY_SEARCH_STEPS = 10


def _evaluate_coverage_at_sizes[F: np.floating[Any], I: np.integer[Any]](
    calibration_probabilities: NDArray[F],
    true_labels: NDArray[I],
    coverage: float,
    config: tuple[NDArray[np.intp], int, np.random.Generator],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run trials at each calibration size for a single coverage level."""
    sizes, n_repetitions, generator = config
    n_data = calibration_probabilities.shape[0]
    one = np.array(1, dtype=calibration_probabilities.dtype)
    mean_coverages: list[float] = []
    std_coverages: list[float] = []
    mean_set_sizes: list[float] = []
    std_set_sizes: list[float] = []

    for size in sizes:
        trial_coverages: list[float] = []
        trial_set_sizes: list[float] = []
        n_cal = int(size) // 2

        for _ in range(n_repetitions):
            indices = generator.choice(n_data, size=int(size), replace=False)
            cal_idx, test_idx = indices[:n_cal], indices[n_cal:]

            scores = calibrate_classifier(calibration_probabilities[cal_idx], true_labels[cal_idx])
            p_values = compute_p_values(scores, one - calibration_probabilities[test_idx])
            prediction_sets = p_values >= 1 - coverage

            hits = prediction_sets[np.arange(len(test_idx)), true_labels[test_idx]]
            trial_coverages.append(hits.mean())
            trial_set_sizes.append(prediction_sets.sum(axis=1).mean())

        mean_coverages.append(float(np.mean(trial_coverages)))
        std_coverages.append(float(np.std(trial_coverages)))
        mean_set_sizes.append(float(np.mean(trial_set_sizes)))
        std_set_sizes.append(float(np.std(trial_set_sizes)))

    return np.array(mean_coverages), np.array(std_coverages), np.array(mean_set_sizes), np.array(std_set_sizes)


def classifier_calibration_plan[F: np.floating[Any], I: np.integer[Any]](
    calibration_probabilities: NDArray[F],
    true_labels: NDArray[I],
    max_set_size: float = 1.05,
    config: DiagnosticConfig | None = None,
) -> CalibrationPlan:
    """Find the best coverage level and calibration set size for a classifier.

    Search over coverage levels to find the highest coverage where mean prediction 
    set size stays within ``max_set_size``. At each coverage level, evaluates 
    multiple calibration set sizes to find the smallest that achieves stable coverage.

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
    CalibrationPlan
        The recommended coverage and calibration size, plus the full grid
        of results from the search for inspection.

    """
    if config is None:
        config = DiagnosticConfig()

    n_data = calibration_probabilities.shape[0]
    if true_labels.shape[0] != n_data:
        msg = f"Shape mismatch: got {n_data} model outputs but {true_labels.shape[0]} true values."
        raise ValueError(msg)

    rng = config.rng
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    low, high = 0.5, 1.0
    all_coverages: list[float] = []
    all_mean_coverage: list[NDArray[np.float64]] = []
    all_std_coverage: list[NDArray[np.float64]] = []
    all_mean_set_size: list[NDArray[np.float64]] = []
    all_std_set_size: list[NDArray[np.float64]] = []

    # Coverage=1.0 is degenerate (every class is included), so resolve sizes at the first midpoint
    evaluated_sizes = _resolve_sizes(config.sizes, n_data, (low + high) / 2)

    for _ in range(_BINARY_SEARCH_STEPS):
        mid = (low + high) / 2
        mean_cov, std_cov, mean_sets, std_sets = _evaluate_coverage_at_sizes(
            calibration_probabilities,
            true_labels,
            mid,
            (evaluated_sizes, config.n_repetitions, generator),
        )

        all_coverages.append(mid)
        all_mean_coverage.append(mean_cov)
        all_std_coverage.append(std_cov)
        all_mean_set_size.append(mean_sets)
        all_std_set_size.append(std_sets)

        if min(mean_sets) <= max_set_size:
            low = mid
        else:
            high = mid

    search_results = _SearchResults(
        all_coverages,
        all_mean_coverage,
        all_std_coverage,
        all_mean_set_size,
        all_std_set_size,
    )
    return _build_calibration_plan(search_results, evaluated_sizes, max_set_size)


@dataclass(slots=True)
class _SearchResults:
    coverages: list[float]
    mean_coverage: list[NDArray[np.float64]]
    std_coverage: list[NDArray[np.float64]]
    mean_set_size: list[NDArray[np.float64]]
    std_set_size: list[NDArray[np.float64]]


def _build_calibration_plan(
    results: _SearchResults,
    evaluated_sizes: NDArray[np.intp],
    max_set_size: float,
) -> CalibrationPlan:
    """Select the best calibration size per coverage level and build the result."""
    table_cal_sizes: list[int] = []
    table_mean_coverage: list[float] = []
    table_std_coverage: list[float] = []
    table_mean_set_size: list[float] = []
    table_std_set_size: list[float] = []

    for idx in range(len(results.coverages)):
        mean_cov_at_sizes = results.mean_coverage[idx]
        meets = mean_cov_at_sizes >= results.coverages[idx]
        best_idx = int(np.argmax(meets)) if np.any(meets) else int(np.argmax(mean_cov_at_sizes))
        table_cal_sizes.append(int(evaluated_sizes[best_idx]))
        table_mean_coverage.append(float(mean_cov_at_sizes[best_idx]))
        table_std_coverage.append(float(results.std_coverage[idx][best_idx]))
        table_mean_set_size.append(float(results.mean_set_size[idx][best_idx]))
        table_std_set_size.append(float(results.std_set_size[idx][best_idx]))

    order = np.argsort(table_mean_coverage)

    sorted_mean_coverage = np.array(table_mean_coverage)[order]
    sorted_cal_sizes = np.array(table_cal_sizes, dtype=np.intp)[order]
    sorted_std_coverage = np.array(table_std_coverage)[order]
    sorted_mean_set_size = np.array(table_mean_set_size)[order]
    sorted_std_set_size = np.array(table_std_set_size)[order]

    best_coverage = float(sorted_mean_coverage[0])
    best_size = int(sorted_cal_sizes[0])
    for i in range(len(sorted_mean_coverage) - 1, -1, -1):
        if sorted_mean_set_size[i] <= max_set_size:
            best_coverage = float(sorted_mean_coverage[i])
            best_size = int(sorted_cal_sizes[i])
            break

    return CalibrationPlan(
        best_coverage,
        best_size,
        sorted_mean_coverage,
        sorted_std_coverage,
        sorted_cal_sizes,
        sorted_mean_set_size,
        sorted_std_set_size,
    )
