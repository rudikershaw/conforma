"""Conformal regressor wrapper.

Wraps any model callable with conformal calibration to produce prediction
intervals with exact coverage guarantees at any sample size.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from conforma.core import ONE_DIMENSION, TWO_DIMENSIONS, compute_quantile


class ConformalRegressor[X, F: np.floating[Any]]:
    """Framework-agnostic conformal regressor.

    Wraps any callable that returns predictions and uses calibration
    data to produce prediction intervals with coverage guarantees.

    Parameters
    ----------
    predict_fn : Callable[[X], NDArray[F]]
        Any callable that takes inputs and returns predictions.
        Must return an array of shape ``(n_examples,)`` for univariate
        or ``(n_examples, n_outputs)`` for multi-output regression.
    calibration : NDArray[F]
        Sorted nonconformity scores from ``calibrate_regressor``.
        Shape: ``(n_calibration_examples,)`` for univariate or
        ``(n_calibration_examples, n_outputs)`` for multi-output.

    Examples
    --------
    >>> import numpy as np
    >>> from conforma import ConformalRegressor, calibrate_regressor
    >>> cal_preds = np.array([2.1, 5.3, 7.8, 3.2])
    >>> cal_true = np.array([2.0, 5.0, 8.0, 3.5])
    >>> calibration = calibrate_regressor(cal_preds, cal_true)
    >>> wrapper = ConformalRegressor(
    ...     predict_fn=lambda X: X,  # identity for demonstration
    ...     calibration=calibration,
    ... )
    >>> intervals = wrapper.predict(np.array([4.0, 6.0]), coverage=0.5)

    """

    def __init__(self, predict_fn: Callable[[X], NDArray[F]], calibration: NDArray[F]) -> None:
        """Create a conformal regressor from a model callable and calibration scores."""
        if calibration.ndim not in (ONE_DIMENSION, TWO_DIMENSIONS):
            msg = f"calibration must be 1D or 2D (from calibrate_regressor), got shape {calibration.shape}."
            raise ValueError(msg)
        self._predict_fn = predict_fn
        self._calibration = calibration

    def predict(self, inputs: X, coverage: float) -> NDArray[F]:
        """Produce prediction intervals at a given coverage level.

        Parameters
        ----------
        inputs : X
            Input data, passed directly to ``predict_fn``.
        coverage : float
            Target coverage level in (0, 1]. The returned interval is
            guaranteed to contain the true value at least this fraction
            of the time under exchangeability.

        Returns
        -------
        NDArray[F]
            Prediction intervals. Shape ``(n_examples, 2)`` for univariate
            or ``(n_examples, n_outputs, 2)`` for multi-output, where the
            last axis is ``[lower, upper]``.

        """
        quantile = compute_quantile(self._calibration, coverage)

        predictions = self._predict_fn(inputs)
        if predictions.ndim not in (ONE_DIMENSION, TWO_DIMENSIONS):
            msg = f"predict_fn must return a 1D or 2D array, got shape {predictions.shape}."
            raise ValueError(msg)

        if self._calibration.ndim != predictions.ndim:
            msg = (
                f"Dimension mismatch: calibration is {self._calibration.ndim}D "
                f"but predict_fn returned {predictions.ndim}D predictions."
            )
            raise ValueError(msg)

        return np.stack([predictions - quantile, predictions + quantile], axis=-1)
