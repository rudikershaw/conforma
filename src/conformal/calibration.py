"""Calibration functions for conformal prediction.

This module provides functions to calibrate conformal predictors by computing
nonconformity scores from a calibration dataset.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from conformal.core import ONE_DIMENSION, TWO_DIMENSIONS


def _calibrate[F: np.floating[Any], T: np.generic](
    calibration_predictions: NDArray[F],
    true_labels: NDArray[T],
    score_fn: Callable[[NDArray[F], NDArray[T]], NDArray[F]],
) -> NDArray[F]:
    """Compute and sort nonconformity scores from a calibration set.

    This is an internal function that allows custom nonconformity score functions.
    Most users should use `calibrate_classifier` or `calibrate_regressor` instead.

    Parameters
    ----------
    calibration_predictions : NDArray
        Model predictions on the calibration set.
        Shape depends on the task (see specific calibration functions).
    true_labels : NDArray
        True values for the calibration set.
        Shape must be compatible with calibration_predictions for the score function.
    score_fn : Callable
        Function that computes nonconformity scores given predictions and true values.
        Signature: (predictions, true_values) -> scores
        Must return a 1D array of shape (n_calibration_examples,) for single-output
        tasks, or a 2D array of shape (n_calibration_examples, n_outputs) for
        multi-output tasks.

    Returns
    -------
    NDArray
        Sorted nonconformity scores from the calibration set.
        Shape: (n_calibration_examples,) for single-output or
        (n_calibration_examples, n_outputs) for multi-output.
        Scores are sorted independently along axis 0 (per output dimension).

    Notes
    -----
    Lower nonconformity scores indicate the model was less surprised by
    the true value. The specific score function depends on the use case
    (classification vs regression).

    """
    n_cal = calibration_predictions.shape[0]
    if true_labels.shape[0] != n_cal:
        msg = f"Shape mismatch: calibration_predictions has {n_cal} examples but true_labels has {true_labels.shape[0]}"
        raise ValueError(msg)

    nonconformity_scores: NDArray[F] = score_fn(calibration_predictions, true_labels)

    if nonconformity_scores.ndim not in (1, 2):
        msg = f"score_fn must return 1D or 2D array, got shape {nonconformity_scores.shape}"
        raise ValueError(msg)
    if nonconformity_scores.shape[0] != n_cal:
        msg = f"score_fn must return array with {n_cal} examples, got {nonconformity_scores.shape[0]}"
        raise ValueError(msg)

    # Sort scores in ascending order along axis 0 (per column for 2D)
    sorted_scores: NDArray[F] = np.sort(nonconformity_scores, axis=0)
    return sorted_scores


def calibrate_classifier[F: np.floating[Any], I: np.integer[Any]](
    calibration_probabilities: NDArray[F], true_labels: NDArray[I]
) -> NDArray[F]:
    """Produce standard nonconformity scores for use with a conformal classifier.

    Computes nonconformity scores as `1 - p` where p is the predicted probability
    for the true class. The nonconformity score acts as a measure of how surprised
    the model was by the real answer. Scores are returned sorted in ascending order.

    Parameters
    ----------
    calibration_probabilities : NDArray
        Model output scores on the calibration set (e.g. probabilities, logits).
        Shape: (n_calibration_examples, n_classes)
    true_labels : NDArray
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
    >>> calibration_probabilities = np.array([[0.8, 0.1, 0.1],
    ...                                        [0.3, 0.6, 0.1],
    ...                                        [0.2, 0.3, 0.5]])
    >>> true_labels = np.array([0, 1, 2])
    >>> scores = calibrate_classifier(calibration_probabilities, true_labels)
    >>> scores
    array([0.2, 0.4, 0.5])

    Notes
    -----
    The calibration set should be held-out data not used during model training.
    The exchangeability assumption states that calibration and test data must be
    drawn from the same distribution, or distributions that are sufficiently similar.

    """
    if calibration_probabilities.ndim != TWO_DIMENSIONS:
        msg = f"calibration_probabilities must be 2D, got shape {calibration_probabilities.shape}."
        raise ValueError(msg)

    if true_labels.ndim != ONE_DIMENSION:
        msg = f"true_labels must be 1D, got shape {true_labels.shape}."
        raise ValueError(msg)

    n_classes = calibration_probabilities.shape[1]
    if np.any(true_labels < 0) or np.any(true_labels >= n_classes):
        msg = (
            f"true_labels contains values outside [0, {n_classes}). "
            f"Labels must be valid class indices for the {n_classes} classes "
            f"in calibration_probabilities."
        )
        raise ValueError(msg)

    def classifier_score_fn(probs: NDArray[F], labels: NDArray[I]) -> NDArray[F]:
        # Use the input's dtype for the constant to prevent precision upcasting
        one = np.array(1, dtype=probs.dtype)
        score: NDArray[F] = one - probs[np.arange(len(labels)), labels]
        return score

    return _calibrate(calibration_probabilities, true_labels, score_fn=classifier_score_fn)


def calibrate_regressor[F: np.floating[Any]](
    calibration_predictions: NDArray[F], true_values: NDArray[F]
) -> NDArray[F]:
    """Produce standard nonconformity scores for use with a conformal regressor.

    Computes nonconformity scores as the absolute difference between the prediction
    and the true value. The nonconformity score acts as a measure of how surprised
    the model was by the real answer. Scores are returned sorted in ascending order.
    Supports both univariate and multi-output regression.

    Parameters
    ----------
    calibration_predictions : NDArray
        Predicted values on the calibration set.
        Shape: (n_calibration_examples,) for univariate regression or
        (n_calibration_examples, n_outputs) for multi-output regression.
    true_values : NDArray
        True values for the calibration set.
        Shape: (n_calibration_examples,) for univariate regression or
        (n_calibration_examples, n_outputs) for multi-output regression.

    Returns
    -------
    NDArray
        Sorted nonconformity scores from the calibration set.
        Shape: (n_calibration_examples,) for univariate regression or
        (n_calibration_examples, n_outputs) for multi-output regression.
        For multi-output, each output dimension is calibrated independently.

    Examples
    --------
    Univariate regression:

    >>> import numpy as np
    >>> from conformal.calibration import calibrate_regressor
    >>> calibration_predictions = np.array([2.1, 5.3, 7.8])
    >>> true_values = np.array([2.0, 5.0, 8.0])
    >>> scores = calibrate_regressor(calibration_predictions, true_values)
    >>> scores
    array([0.1, 0.2, 0.3])

    Multi-output regression:

    >>> calibration_predictions = np.array([[1.1, 2.2], [3.4, 4.3], [5.6, 6.5]])
    >>> true_values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> scores = calibrate_regressor(calibration_predictions, true_values)
    >>> scores
    array([[0.1, 0.2],
           [0.4, 0.3],
           [0.6, 0.5]])

    Notes
    -----
    The calibration set should be held-out data not used during model training.
    The exchangeability assumption states that calibration and test data must be
    drawn from the same distribution, or distributions that are sufficiently similar.

    For multi-output regression, each output dimension is calibrated independently,
    providing separate prediction intervals for each output.

    """
    if calibration_predictions.ndim not in (ONE_DIMENSION, TWO_DIMENSIONS):
        msg = f"calibration_predictions must be 1D or 2D, got shape {calibration_predictions.shape}."
        raise ValueError(msg)

    if true_values.ndim not in (ONE_DIMENSION, TWO_DIMENSIONS):
        msg = f"true_values must be 1D or 2D, got shape {true_values.shape}."
        raise ValueError(msg)

    if calibration_predictions.shape != true_values.shape:
        msg = (
            f"Shape mismatch: calibration_predictions has shape {calibration_predictions.shape} "
            f"but true_values has shape {true_values.shape}."
        )
        raise ValueError(msg)

    def regressor_score_fn(preds: NDArray[F], true_values: NDArray[F]) -> NDArray[F]:
        score: NDArray[F] = np.abs(preds - true_values)
        return score

    return _calibrate(calibration_predictions, true_values, score_fn=regressor_score_fn)
