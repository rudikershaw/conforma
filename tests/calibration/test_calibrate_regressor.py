"""Tests for the calibrate_regressor function."""

import numpy as np
import pytest

from conformal.calibration import calibrate_regressor
from tests.conftest import default_benchmark


@default_benchmark
def test_calibrate_regressor_univariate_basic(benchmark) -> None:
    """Test basic univariate regression calibration."""
    calibration_predictions = np.array([2.1, 5.3, 7.8])
    true_labels = np.array([2.0, 5.0, 8.0])

    scores = benchmark(calibrate_regressor, calibration_predictions, true_labels)

    # Expected scores: abs([2.1-2.0, 5.3-5.0, 7.8-8.0]) = [0.1, 0.3, 0.2], sorted
    expected = np.array([0.1, 0.2, 0.3])
    np.testing.assert_array_almost_equal(scores, expected)


@default_benchmark
def test_calibrate_regressor_univariate_returns_sorted(benchmark) -> None:
    """Test that scores are returned in sorted order for univariate case."""
    calibration_predictions = np.array([1.0, 4.0, 2.5])
    true_labels = np.array([2.0, 3.0, 2.0])

    scores = benchmark(calibrate_regressor, calibration_predictions, true_labels)

    # Expected scores: abs([1.0-2.0, 4.0-3.0, 2.5-2.0]) = [1.0, 1.0, 0.5], sorted
    expected = np.array([0.5, 1.0, 1.0])
    np.testing.assert_array_almost_equal(scores, expected)
    # Verify sorted
    assert np.all(scores[:-1] <= scores[1:])


@default_benchmark
def test_calibrate_regressor_multioutput_basic(benchmark) -> None:
    """Test basic multi-output regression calibration."""
    calibration_predictions = np.array([[1.1, 2.2], [3.4, 4.3], [5.6, 6.5]])
    true_labels = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    scores = benchmark(calibrate_regressor, calibration_predictions, true_labels)

    # Expected scores per column:
    # Column 0: abs([1.1-1.0, 3.4-3.0, 5.6-5.0]) = [0.1, 0.4, 0.6], sorted
    # Column 1: abs([2.2-2.0, 4.3-4.0, 6.5-6.0]) = [0.2, 0.3, 0.5], sorted
    expected = np.array([[0.1, 0.2], [0.4, 0.3], [0.6, 0.5]])
    np.testing.assert_array_almost_equal(scores, expected)


@default_benchmark
def test_calibrate_regressor_multioutput_returns_sorted(benchmark) -> None:
    """Test that each column is sorted independently for multi-output case."""
    calibration_predictions = np.array([[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]])
    true_labels = np.array([[2.0, 6.0], [4.0, 3.0], [3.0, 5.0]])

    scores = benchmark(calibrate_regressor, calibration_predictions, true_labels)

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


@default_benchmark
def test_calibrate_regressor_with_perfect_predictions(benchmark) -> None:
    """Test calibration when predictions are perfect."""
    calibration_predictions = np.array([1.0, 2.0, 3.0])
    true_labels = np.array([1.0, 2.0, 3.0])

    scores = benchmark(calibrate_regressor, calibration_predictions, true_labels)

    # All residuals should be zero
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(scores, expected)


@default_benchmark
def test_calibrate_regressor_multioutput_with_different_errors(benchmark) -> None:
    """Test multi-output case where different outputs have different error magnitudes."""
    calibration_predictions = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    true_labels = np.array([[1.5, 15.0], [2.5, 25.0], [3.5, 35.0]])

    scores = benchmark(calibrate_regressor, calibration_predictions, true_labels)

    # Column 0 errors: [0.5, 0.5, 0.5]
    # Column 1 errors: [5.0, 5.0, 5.0]
    expected = np.array([[0.5, 5.0], [0.5, 5.0], [0.5, 5.0]])
    np.testing.assert_array_almost_equal(scores, expected)
