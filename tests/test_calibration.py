"""Tests for calibration functions."""

import numpy as np
import pytest

from conformal.calibration import _calibrate, calibrate_classifier


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


def test_internal_calibrate_with_custom_score_fn() -> None:
    """Test _calibrate with a custom score function."""
    cal_probs = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1]])
    y_cal = np.array([0, 1])

    # Custom score function that returns constant scores
    def custom_score(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.array([0.5, 0.3])

    scores = _calibrate(cal_probs, y_cal, score_fn=custom_score)

    # Should return sorted custom scores
    expected = np.array([0.3, 0.5])
    np.testing.assert_array_almost_equal(scores, expected)


def test_internal_calibrate_validates_score_fn_output() -> None:
    """Test that _calibrate validates score function output shape."""
    cal_probs = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1]])
    y_cal = np.array([0, 1])

    # Score function that returns wrong shape
    def bad_score_fn(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.array([[0.5, 0.3]])  # 2D instead of 1D

    with pytest.raises(ValueError, match="score_fn must return 1D array"):
        _calibrate(cal_probs, y_cal, score_fn=bad_score_fn)
