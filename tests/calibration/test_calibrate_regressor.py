"""Tests for the calibrate_regressor function."""

import numpy as np
import pytest

from conformal.calibration import calibrate_regressor


def test_calibrate_regressor_univariate_basic() -> None:
    """Test basic univariate regression calibration."""
    cal_preds = np.array([2.1, 5.3, 7.8])
    y_cal = np.array([2.0, 5.0, 8.0])

    scores = calibrate_regressor(cal_preds, y_cal)

    # Expected scores: abs([2.1-2.0, 5.3-5.0, 7.8-8.0]) = [0.1, 0.3, 0.2], sorted
    expected = np.array([0.1, 0.2, 0.3])
    np.testing.assert_array_almost_equal(scores, expected)


def test_calibrate_regressor_univariate_returns_sorted() -> None:
    """Test that scores are returned in sorted order for univariate case."""
    cal_preds = np.array([1.0, 4.0, 2.5])
    y_cal = np.array([2.0, 3.0, 2.0])

    scores = calibrate_regressor(cal_preds, y_cal)

    # Expected scores: abs([1.0-2.0, 4.0-3.0, 2.5-2.0]) = [1.0, 1.0, 0.5], sorted
    expected = np.array([0.5, 1.0, 1.0])
    np.testing.assert_array_almost_equal(scores, expected)
    # Verify sorted
    assert np.all(scores[:-1] <= scores[1:])


def test_calibrate_regressor_multioutput_basic() -> None:
    """Test basic multi-output regression calibration."""
    cal_preds = np.array([[1.1, 2.2], [3.4, 4.3], [5.6, 6.5]])
    y_cal = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    scores = calibrate_regressor(cal_preds, y_cal)

    # Expected scores per column:
    # Column 0: abs([1.1-1.0, 3.4-3.0, 5.6-5.0]) = [0.1, 0.4, 0.6], sorted
    # Column 1: abs([2.2-2.0, 4.3-4.0, 6.5-6.0]) = [0.2, 0.3, 0.5], sorted
    expected = np.array([[0.1, 0.2], [0.4, 0.3], [0.6, 0.5]])
    np.testing.assert_array_almost_equal(scores, expected)


def test_calibrate_regressor_multioutput_returns_sorted() -> None:
    """Test that each column is sorted independently for multi-output case."""
    cal_preds = np.array([[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]])
    y_cal = np.array([[2.0, 6.0], [4.0, 3.0], [3.0, 5.0]])

    scores = calibrate_regressor(cal_preds, y_cal)

    # Verify each column is sorted independently
    assert np.all(scores[:-1, 0] <= scores[1:, 0])
    assert np.all(scores[:-1, 1] <= scores[1:, 1])


def test_calibrate_regressor_shape_validation() -> None:
    """Test that calibrate_regressor validates input shapes."""
    # Wrong dimensionality for predictions (3D)
    with pytest.raises(ValueError, match="must be 1D or 2D"):
        calibrate_regressor(np.array([[[1.0]]]), np.array([1.0]))

    # Wrong dimensionality for true values (3D)
    with pytest.raises(ValueError, match="must be 1D or 2D"):
        calibrate_regressor(np.array([1.0]), np.array([[[1.0]]]))

    # Mismatched shapes
    with pytest.raises(ValueError, match="Shape mismatch"):
        calibrate_regressor(np.array([1.0, 2.0]), np.array([1.0]))

    # Mismatched shapes for multi-output
    with pytest.raises(ValueError, match="Shape mismatch"):
        calibrate_regressor(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0, 3.0]]))


def test_calibrate_regressor_with_perfect_predictions() -> None:
    """Test calibration when predictions are perfect."""
    cal_preds = np.array([1.0, 2.0, 3.0])
    y_cal = np.array([1.0, 2.0, 3.0])

    scores = calibrate_regressor(cal_preds, y_cal)

    # All residuals should be zero
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(scores, expected)


def test_calibrate_regressor_multioutput_with_different_errors() -> None:
    """Test multi-output case where different outputs have different error magnitudes."""
    cal_preds = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    y_cal = np.array([[1.5, 15.0], [2.5, 25.0], [3.5, 35.0]])

    scores = calibrate_regressor(cal_preds, y_cal)

    # Column 0 errors: [0.5, 0.5, 0.5]
    # Column 1 errors: [5.0, 5.0, 5.0]
    expected = np.array([[0.5, 5.0], [0.5, 5.0], [0.5, 5.0]])
    np.testing.assert_array_almost_equal(scores, expected)
