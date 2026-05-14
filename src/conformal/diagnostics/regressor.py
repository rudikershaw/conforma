"""Coverage stability diagnostic for regressors."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from conformal.calibration import calibrate_regressor
from conformal.core import compute_quantile
from conformal.diagnostics._common import (
    CoverageStabilityResult,
    DiagnosticConfig,
    _run_coverage_stability,
    _validate_diagnostic_inputs,
)


def regressor_coverage_stability[F: np.floating[Any]](
    calibration_predictions: NDArray[F],
    true_values: NDArray[F],
    coverage: float = 0.90,
    config: DiagnosticConfig | None = None,
) -> CoverageStabilityResult:
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
    CoverageStabilityResult
        A frozen dataclass containing the sizes evaluated and the mean
        and standard deviation of empirical coverage and prediction
        interval width at each.

    Examples
    --------
    >>> import numpy as np
    >>> from conformal.diagnostics import regressor_coverage_stability, DiagnosticConfig
    >>> preds = np.array([2.1, 5.3, 7.8, 3.2, 1.0, 4.5, 6.2, 8.1, 2.5, 3.8])
    >>> true = np.array([2.0, 5.0, 8.0, 3.5, 1.2, 4.0, 6.0, 8.5, 2.8, 3.5])
    >>> config = DiagnosticConfig(sizes=[6, 8], n_repetitions=10, rng=42)
    >>> result = regressor_coverage_stability(preds, true, coverage=0.5, config=config)

    """
    if config is None:
        config = DiagnosticConfig()

    n_data = calibration_predictions.shape[0]
    _validate_diagnostic_inputs(n_data, true_values.shape[0], coverage, config)

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

    return _run_coverage_stability(n_data, coverage, config, trial_fn)
