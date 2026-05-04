"""Calibration functions for conformal prediction.

This module provides functions to calibrate conformal predictors by computing
nonconformity scores from a calibration dataset.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Expected array dimensions for validation
REQUIRED_PROBABILITIES_DIMENSIONS = 2
REQUIRED_LABELS_DIMENSIONS = 1


def _calibrate[F: np.floating[Any], T: np.generic](
    cal_predictions: NDArray[F],
    y_cal: NDArray[T],
    score_fn: Callable[[NDArray[F], NDArray[T]], NDArray[F]],
) -> NDArray[F]:
    """Compute and sort nonconformity scores from a calibration set.

    This is an internal function that allows custom nonconformity score functions.
    Most users should use `calibrate_classifier` or `calibrate_regressor` instead.

    Parameters
    ----------
    cal_predictions : NDArray
        Model predictions on the calibration set.
        Shape depends on the task (see specific calibration functions).
    y_cal : NDArray
        True values for the calibration set.
        Shape must be compatible with cal_predictions for the score function.
    score_fn : Callable
        Function that computes nonconformity scores given predictions and true values.
        Signature: (predictions, true_values) -> scores
        Must return a 1D array of shape (n_calibration_examples,).

    Returns
    -------
    NDArray
        Sorted nonconformity scores from the calibration set.
        Shape: (n_calibration_examples,)

    Notes
    -----
    Lower nonconformity scores indicate the model was less surprised by
    the true value. The specific score function depends on the use case
    (classification vs regression).

    """
    # Basic validation: ensure first dimension matches
    n_cal = cal_predictions.shape[0]
    if y_cal.shape[0] != n_cal:
        msg = f"Shape mismatch: cal_predictions has {n_cal} examples but y_cal has {y_cal.shape[0]}"
        raise ValueError(msg)

    # Compute nonconformity scores
    scores: NDArray[F] = score_fn(cal_predictions, y_cal)

    # Validate scores shape
    if scores.shape != (n_cal,):
        msg = f"score_fn must return 1D array of shape ({n_cal},), got {scores.shape}"
        raise ValueError(msg)

    # Sort scores in ascending order and return
    sorted_scores: NDArray[F] = np.sort(scores)
    return sorted_scores


def calibrate_classifier[F: np.floating[Any], I: np.integer[Any]](
    cal_probabilities: NDArray[F], y_cal: NDArray[I]
) -> NDArray[F]:
    """Calibrate a conformal classifier using the standard nonconformity score.

    Computes nonconformity scores as `1 - p` where p is the predicted probability
    for the true class, then returns them sorted in ascending order.

    Parameters
    ----------
    cal_probabilities : NDArray
        Predicted probabilities on the calibration set.
        Shape: (n_calibration_examples, n_classes)
        Must sum to 1 across axis 1 (i.e., valid probability distributions).
    y_cal : NDArray
        True class labels for the calibration set.
        Shape: (n_calibration_examples,)
        Must contain integer class indices in range [0, n_classes).

    Returns
    -------
    NDArray
        Sorted nonconformity scores from the calibration set.
        Shape: (n_calibration_examples,)

    Examples
    --------
    >>> import numpy as np
    >>> from conformal.calibration import calibrate_classifier
    >>> cal_probs = np.array([[0.8, 0.1, 0.1],
    ...                       [0.3, 0.6, 0.1],
    ...                       [0.2, 0.3, 0.5]])
    >>> y_cal = np.array([0, 1, 2])
    >>> scores = calibrate_classifier(cal_probs, y_cal)
    >>> scores
    array([0.2, 0.4, 0.5])

    Notes
    -----
    The calibration set should be held-out data not used during model training.
    The exchangeability assumption requires that calibration and test data are
    drawn from the same distribution.

    """
    # Validate shapes for classification
    if cal_probabilities.ndim != REQUIRED_PROBABILITIES_DIMENSIONS:
        msg = f"cal_probabilities must be 2D, got shape {cal_probabilities.shape}"
        raise ValueError(msg)

    if y_cal.ndim != REQUIRED_LABELS_DIMENSIONS:
        msg = f"y_cal must be 1D, got shape {y_cal.shape}"
        raise ValueError(msg)

    # Default classifier score function: 1 - probability of true class
    def classifier_score_fn(probs: NDArray[F], labels: NDArray[I]) -> NDArray[F]:
        # Use the input's dtype for the constant to prevent precision upcasting
        one = np.array(1, dtype=probs.dtype)
        score: NDArray[F] = one - probs[np.arange(len(labels)), labels]
        return score

    return _calibrate(cal_probabilities, y_cal, score_fn=classifier_score_fn)


def calibrate_regressor[F: np.floating[Any]](cal_predictions: NDArray[F], y_cal: NDArray[F]) -> NDArray[F]:
    """Calibrate a conformal regressor using the standard nonconformity score.

    Computes nonconformity scores as the absolute residual |y_pred - y_true|,
    then returns them sorted in ascending order.

    Parameters
    ----------
    cal_predictions : NDArray
        Predicted values on the calibration set.
        Shape: (n_calibration_examples,)
    y_cal : NDArray
        True values for the calibration set.
        Shape: (n_calibration_examples,)

    Returns
    -------
    NDArray
        Sorted nonconformity scores from the calibration set.
        Shape: (n_calibration_examples,)

    Examples
    --------
    >>> import numpy as np
    >>> from conformal.calibration import calibrate_regressor
    >>> cal_preds = np.array([2.1, 5.3, 7.8])
    >>> y_cal = np.array([2.0, 5.0, 8.0])
    >>> scores = calibrate_regressor(cal_preds, y_cal)
    >>> scores
    array([0.1, 0.2, 0.3])

    Notes
    -----
    The calibration set should be held-out data not used during model training.
    The exchangeability assumption requires that calibration and test data are
    drawn from the same distribution.

    For regression, larger residuals indicate the model was more surprised by
    the true value. The sorted scores are used to construct prediction intervals
    that achieve the desired coverage level.

    """
    # Validate shapes for regression
    if cal_predictions.ndim != REQUIRED_LABELS_DIMENSIONS:
        msg = f"cal_predictions must be 1D, got shape {cal_predictions.shape}"
        raise ValueError(msg)

    if y_cal.ndim != REQUIRED_LABELS_DIMENSIONS:
        msg = f"y_cal must be 1D, got shape {y_cal.shape}"
        raise ValueError(msg)

    # Default regressor score function: absolute residual
    def regressor_score_fn(preds: NDArray[F], true_values: NDArray[F]) -> NDArray[F]:
        score: NDArray[F] = np.abs(preds - true_values)
        return score

    return _calibrate(cal_predictions, y_cal, score_fn=regressor_score_fn)
