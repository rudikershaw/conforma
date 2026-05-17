"""Conformal classifier wrapper.

Wraps any model callable with conformal calibration to produce p-values or
prediction sets with exact coverage guarantees at any sample size.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from conforma.core import ONE_DIMENSION, TWO_DIMENSIONS, compute_p_values


class ConformalClassifier[X, F: np.floating[Any]]:
    """Framework-agnostic conformal classifier.

    Wraps any callable that returns per-class scores and uses calibration
    data to produce p-values or prediction sets with coverage guarantees.

    Parameters
    ----------
    predict_fn : Callable[[X], NDArray[F]]
        Any callable that takes inputs and returns per-class scores
        (probabilities, logits, or any per-class output).
        Must return an array of shape ``(n_examples, n_classes)``.
    calibration : NDArray[F]
        Sorted nonconformity scores from ``calibrate_classifier``.
        Shape: ``(n_calibration_examples,)``

    Examples
    --------
    >>> import numpy as np
    >>> from conforma import ConformalClassifier, calibrate_classifier
    >>> cal_probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.9, 0.1]])
    >>> cal_labels = np.array([0, 0, 1, 0])
    >>> calibration = calibrate_classifier(cal_probs, cal_labels)
    >>> wrapper = ConformalClassifier(
    ...     predict_fn=lambda X: X,  # identity for demonstration
    ...     calibration=calibration,
    ... )
    >>> test_scores = np.array([[0.7, 0.3], [0.5, 0.5]])
    >>> p_values = wrapper.predict_p_values(test_scores)
    >>> prediction_sets = wrapper.predict(test_scores, coverage=0.9)

    """

    def __init__(self, predict_fn: Callable[[X], NDArray[F]], calibration: NDArray[F]) -> None:
        """Create a conformal classifier from a model callable and calibration scores."""
        if calibration.ndim != ONE_DIMENSION:
            msg = f"calibration must be 1D (from calibrate_classifier), got shape {calibration.shape}."
            raise ValueError(msg)
        self._predict_fn = predict_fn
        self._calibration = calibration

    def predict(self, inputs: X, coverage: float) -> NDArray[np.bool_]:
        """Produce prediction sets at a given coverage level.

        A prediction set is a boolean mask where True means the class is
        included. Multiple Trues indicate the model considers more than
        one class plausible. An empty set (all False) indicates the model
        has no confident prediction for that example.

        Parameters
        ----------
        inputs : X
            Input data, passed directly to ``predict_fn``.
        coverage : float
            Target coverage level in (0, 1]. Classes with p-values at or
            above this threshold are included in the prediction set.

        Returns
        -------
        NDArray[np.bool_]
            Boolean array of shape ``(n_examples, n_classes)``.

        """
        if coverage <= 0 or coverage > 1:
            msg = f"coverage must be in (0, 1], got {coverage}."
            raise ValueError(msg)
        return self.predict_p_values(inputs) >= 1 - coverage

    def predict_p_values(self, inputs: X) -> NDArray[F]:
        """Compute conformal p-values for each class.

        Each p-value is the minimum coverage level at which that class enters
        the prediction set. Higher means more plausible.

        Parameters
        ----------
        inputs : X
            Input data, passed directly to ``predict_fn``.

        Returns
        -------
        NDArray[F]
            P-values of shape ``(n_examples, n_classes)``.

        """
        probabilities = self._predict_fn(inputs)
        if probabilities.ndim != TWO_DIMENSIONS:
            msg = f"predict_fn must return a 2D array, got shape {probabilities.shape}."
            raise ValueError(msg)

        one = np.array(1, dtype=probabilities.dtype)
        prediction_scores = one - probabilities
        return compute_p_values(self._calibration, prediction_scores)
