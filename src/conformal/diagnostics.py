"""Diagnostic tools for analysing conformal predictor behaviour.

This module provides functions that help practitioners understand how
calibration set size affects prediction quality.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from conformal.calibration import calibrate_classifier
from conformal.core import compute_p_values


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
    >>> len(result.sizes)
    2

    """
    if config is None:
        config = DiagnosticConfig()

    n_data = calibration_probabilities.shape[0]
    if true_labels.shape[0] != n_data:
        msg = (
            f"Shape mismatch: calibration_probabilities has {n_data} examples "
            f"but true_labels has {true_labels.shape[0]}."
        )
        raise ValueError(msg)

    if coverage <= 0 or coverage >= 1:
        msg = f"coverage must be in (0, 1), got {coverage}."
        raise ValueError(msg)

    if config.n_repetitions < 1:
        msg = f"n_repetitions must be a positive integer, got {config.n_repetitions}."
        raise ValueError(msg)

    rng = config.rng
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    evaluated_sizes = _resolve_sizes(config.sizes, n_data, coverage)

    mean_coverages: list[F] = []
    std_coverages: list[F] = []
    mean_set_sizes: list[F] = []
    std_set_sizes: list[F] = []

    one = np.array(1, dtype=calibration_probabilities.dtype)

    for size in evaluated_sizes:
        trial_coverages: list[F] = []
        trial_set_sizes: list[F] = []
        n_cal = int(size) // 2

        for _ in range(config.n_repetitions):
            indices = generator.choice(n_data, size=int(size), replace=False)
            cal_idx, test_idx = indices[:n_cal], indices[n_cal:]

            scores = calibrate_classifier(
                calibration_probabilities[cal_idx],
                true_labels[cal_idx],
            )
            p_values = compute_p_values(scores, one - calibration_probabilities[test_idx])
            prediction_sets = p_values >= coverage

            hits = prediction_sets[np.arange(len(test_idx)), true_labels[test_idx]]
            trial_coverages.append(hits.mean())
            trial_set_sizes.append(prediction_sets.sum(axis=1).mean())

        mean_coverages.append(np.mean(trial_coverages))
        std_coverages.append(np.std(trial_coverages))
        mean_set_sizes.append(np.mean(trial_set_sizes))
        std_set_sizes.append(np.std(trial_set_sizes))

    return CoverageStabilityResult(
        sizes=evaluated_sizes,
        mean_coverage=np.array(mean_coverages, dtype=np.float64),
        std_coverage=np.array(std_coverages, dtype=np.float64),
        mean_set_size=np.array(mean_set_sizes, dtype=np.float64),
        std_set_size=np.array(std_set_sizes, dtype=np.float64),
    )
