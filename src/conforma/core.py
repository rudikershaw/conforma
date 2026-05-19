"""Core functions for conformal prediction.

This module provides shared prediction-time primitives used across
classification and regression conformal predictors.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Expected array dimension constants for validation
TWO_DIMENSIONS = 2
ONE_DIMENSION = 1


def compute_p_values[F: np.floating[Any]](
    calibration_scores: NDArray[F],
    prediction_scores: NDArray[F],
) -> NDArray[F]:
    """Compute conformal p-values by comparing prediction scores against calibration scores.

    For each prediction nonconformity score, the p-value is the proportion of
    calibration scores that are greater than or equal to the prediction score. A higher
    p-value means the prediction is more consistent with the calibration data (the model
    was less surprised).

    Parameters
    ----------
    calibration_scores : NDArray
        Sorted nonconformity scores from a calibration set, as returned by
        ``calibrate_classifier`` or ``calibrate_regressor``.
        Shape: (n_calibration,) for single-output or
        (n_calibration, n_outputs) for multi-output.
    prediction_scores : NDArray
        Nonconformity scores for the new predictions.
        Shape: (n_predictions,) or (n_predictions, n_columns) for single-output
        calibration, or (n_predictions, n_outputs) for multi-output calibration
        where n_columns must equal n_outputs.

    Returns
    -------
    NDArray
        P-values with the same shape as prediction_scores.
        Each value is in [0, 1] and represents the proportion of calibration
        scores that are at least as extreme as the corresponding prediction score.

    Examples
    --------
    With classifier calibration scores and per-class prediction scores:

    >>> import numpy as np
    >>> from conforma.core import compute_p_values
    >>> calibration_scores = np.array([0.1, 0.3, 0.5, 0.7])
    >>> prediction_scores = np.array([[0.2, 0.8], [0.6, 0.4]])
    >>> p_values = compute_p_values(calibration_scores, prediction_scores)
    >>> p_values
    array([[0.8, 0.2],
           [0.4, 0.6]])

    With multi-output regression calibration scores:

    >>> calibration_scores = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    >>> prediction_scores = np.array([[0.2, 0.5], [0.4, 0.3]])
    >>> p_values = compute_p_values(calibration_scores, prediction_scores)
    >>> p_values
    array([[0.75, 0.5 ],
           [0.5 , 0.75]])

    Notes
    -----
    The p-value is computed using the finite-sample corrected formula
    ``(n_calibration - rank + 1) / (n_calibration + 1)`` where rank is
    determined by binary search (``np.searchsorted``) into the sorted
    calibration scores. The +1 correction ensures p-values are never exactly
    zero and provides exact finite-sample coverage guarantees.

    """
    if calibration_scores.ndim not in (ONE_DIMENSION, TWO_DIMENSIONS):
        msg = f"calibration_scores must be 1D or 2D, got shape {calibration_scores.shape}."
        raise ValueError(msg)

    if prediction_scores.ndim not in (ONE_DIMENSION, TWO_DIMENSIONS):
        msg = f"prediction_scores must be 1D or 2D, got shape {prediction_scores.shape}."
        raise ValueError(msg)

    both_2d = calibration_scores.ndim == TWO_DIMENSIONS and prediction_scores.ndim == TWO_DIMENSIONS
    if both_2d and calibration_scores.shape[1] != prediction_scores.shape[1]:
        msg = (
            f"Column mismatch: calibration_scores has {calibration_scores.shape[1]} outputs "
            f"but prediction_scores has {prediction_scores.shape[1]}."
        )
        raise ValueError(msg)

    out_dtype = np.result_type(calibration_scores, prediction_scores)
    n_cal_plus_one = np.array(calibration_scores.shape[0] + 1, dtype=out_dtype)

    if calibration_scores.ndim == TWO_DIMENSIONS:
        columns = []
        for col in range(calibration_scores.shape[1]):
            columns.append(np.searchsorted(calibration_scores[:, col], prediction_scores[:, col], side="left"))
        ranks = np.column_stack(columns).astype(out_dtype)
    else:
        ranks = np.searchsorted(calibration_scores, prediction_scores, side="left").astype(out_dtype)

    p_values: NDArray[F] = (n_cal_plus_one - ranks) / n_cal_plus_one
    return p_values


def compute_quantile[F: np.floating[Any]](
    calibration_scores: NDArray[F],
    coverage: float,
) -> NDArray[F]:
    """Select the calibration quantile for a given coverage level.

    Returns the smallest nonconformity score threshold such that the
    resulting prediction intervals achieve the requested coverage.

    Parameters
    ----------
    calibration_scores : NDArray
        Sorted nonconformity scores from ``calibrate_regressor``.
        Shape: ``(n_calibration,)`` for univariate or
        ``(n_calibration, n_outputs)`` for multi-output.
    coverage : float
        Target coverage level in (0, 1).

    Returns
    -------
    NDArray
        The quantile value(s). Scalar-like for univariate, or shape
        ``(n_outputs,)`` for multi-output.

    Raises
    ------
    ValueError
        If coverage is outside (0, 1) or the calibration set is too
        small to achieve the requested coverage.

    Examples
    --------
    >>> import numpy as np
    >>> from conforma.core import compute_quantile
    >>> scores = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    >>> compute_quantile(scores, coverage=0.5)
    array([0.3, 0.4])

    """
    if coverage <= 0 or coverage >= 1:
        msg = f"coverage must be in (0, 1), got {coverage}."
        raise ValueError(msg)

    n_cal = calibration_scores.shape[0]
    index = int(np.ceil(coverage * (n_cal + 1))) - 1
    if index >= n_cal:
        msg = (
            f"Calibration set too small: {n_cal} examples cannot guarantee "
            f"coverage={coverage}. Maximum achievable coverage is "
            f"{n_cal / (n_cal + 1):.5f}."
        )
        raise ValueError(msg)

    result: NDArray[F] = calibration_scores[index]
    return result
