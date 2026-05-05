"""Tests for ConformalRegressor construction and shared validation."""

import numpy as np
import pytest

from conformal.regressor import ConformalRegressor


def test_calibration_must_be_1d_or_2d() -> None:
    """Test that 3D calibration scores are rejected."""
    calibration_3d = np.array([[[0.1, 0.2]]])
    with pytest.raises(ValueError, match="calibration must be 1D or 2D"):
        ConformalRegressor(predict_fn=lambda x: x, calibration=calibration_3d)


def test_predict_fn_must_return_1d_or_2d() -> None:
    """Test that a predict_fn returning 3D is rejected."""
    calibration = np.array([0.1, 0.2, 0.3])
    wrapper = ConformalRegressor(predict_fn=lambda x: x.reshape(1, 1, -1), calibration=calibration)

    with pytest.raises(ValueError, match="predict_fn must return a 1D or 2D array"):
        wrapper.predict(np.array([1.0, 2.0, 3.0]), coverage=0.5)


def test_calibration_prediction_dimension_mismatch() -> None:
    """Test that 1D calibration with 2D predictions is rejected."""
    calibration = np.array([0.1, 0.2, 0.3])
    wrapper = ConformalRegressor(predict_fn=lambda x: np.column_stack([x, x]), calibration=calibration)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        wrapper.predict(np.array([1.0, 2.0]), coverage=0.5)
