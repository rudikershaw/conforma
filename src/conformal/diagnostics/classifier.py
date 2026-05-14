"""Coverage stability diagnostic for classifiers."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from conformal.calibration import calibrate_classifier
from conformal.core import compute_p_values
from conformal.diagnostics._common import (
    CoverageStabilityResult,
    DiagnosticConfig,
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
