"""Tests for the calibrate_classifier function."""

import numpy as np
import pytest

from conformal.calibration import calibrate_classifier


def test_calibrate_classifier_basic() -> None:
    """Test basic calibration with simple probabilities."""
    cal_probs = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])
    y_cal = np.array([0, 1, 2])

    scores = calibrate_classifier(cal_probs, y_cal)

    # Expected scores: 1 - [0.8, 0.6, 0.5] = [0.2, 0.4, 0.5], sorted
    expected = np.array([0.2, 0.4, 0.5])
    np.testing.assert_array_almost_equal(scores, expected)


def test_calibrate_classifier_returns_sorted() -> None:
    """Test that scores are returned in sorted order."""
    cal_probs = np.array([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]])
    y_cal = np.array([1, 0, 1])

    scores = calibrate_classifier(cal_probs, y_cal)

    # Expected scores: 1 - [0.9, 0.9, 0.5] = [0.1, 0.1, 0.5], sorted
    expected = np.array([0.1, 0.1, 0.5])
    np.testing.assert_array_almost_equal(scores, expected)
    # Verify sorted
    assert np.all(scores[:-1] <= scores[1:])


def test_calibrate_classifier_shape_validation() -> None:
    """Test that calibrate_classifier validates input shapes."""
    # Wrong dimensionality for probabilities
    with pytest.raises(ValueError, match="must be 2D"):
        calibrate_classifier(np.array([0.8, 0.1, 0.1]), np.array([0]))

    # Wrong dimensionality for labels
    with pytest.raises(ValueError, match="must be 1D"):
        calibrate_classifier(np.array([[0.8, 0.1, 0.1]]), np.array([[0]]))

    # Mismatched lengths
    with pytest.raises(ValueError, match="Shape mismatch"):
        calibrate_classifier(np.array([[0.8, 0.1, 0.1]]), np.array([0, 1]))


def test_calibrate_classifier_with_perfect_predictions() -> None:
    """Test calibration when model predicts perfectly."""
    cal_probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    y_cal = np.array([0, 1, 0])

    scores = calibrate_classifier(cal_probs, y_cal)

    # Expected scores: 1 - [1.0, 1.0, 1.0] = [0.0, 0.0, 0.0]
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(scores, expected)


def test_calibrate_classifier_with_uniform_predictions() -> None:
    """Test calibration when model predicts uniformly."""
    cal_probs = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    y_cal = np.array([0, 1, 0])

    scores = calibrate_classifier(cal_probs, y_cal)

    # Expected scores: 1 - [0.5, 0.5, 0.5] = [0.5, 0.5, 0.5]
    expected = np.array([0.5, 0.5, 0.5])
    np.testing.assert_array_almost_equal(scores, expected)
