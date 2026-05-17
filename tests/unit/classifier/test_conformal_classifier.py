"""Tests for ConformalClassifier construction and shared validation."""

import numpy as np
import pytest

from conforma.classifier import ConformalClassifier


def test_calibration_must_be_1d() -> None:
    """Test that 2D calibration scores are rejected."""
    calibration_2d = np.array([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="calibration must be 1D"):
        ConformalClassifier(predict_fn=lambda x: x, calibration=calibration_2d)


def test_predict_fn_must_return_2d() -> None:
    """Test that a predict_fn returning 1D is rejected."""
    calibration = np.array([0.2, 0.4, 0.5])
    wrapper = ConformalClassifier(predict_fn=lambda x: x.flatten(), calibration=calibration)

    with pytest.raises(ValueError, match="predict_fn must return a 2D array"):
        wrapper.predict_p_values(np.array([[0.7, 0.3]]))
