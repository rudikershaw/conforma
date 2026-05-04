"""Tests for the internal _calibrate function."""

import numpy as np
import pytest

from conformal.calibration import _calibrate


def test_calibrate_with_custom_score_fn() -> None:
    """Test _calibrate with a custom score function."""
    calibration_probabilities = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1]])
    true_labels = np.array([0, 1])

    # Custom score function that returns constant scores
    def custom_score(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.array([0.5, 0.3])

    scores = _calibrate(calibration_probabilities, true_labels, score_fn=custom_score)

    # Should return sorted custom scores
    expected = np.array([0.3, 0.5])
    np.testing.assert_array_almost_equal(scores, expected)


def test_calibrate_validates_score_fn_output() -> None:
    """Test that _calibrate validates score function output shape."""
    calibration_probabilities = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1]])
    true_labels = np.array([0, 1])

    # Score function that returns 3D array (invalid)
    def bad_score_fn(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.array([[[0.5, 0.3]]])

    with pytest.raises(ValueError, match="score_fn must return 1D or 2D array"):
        _calibrate(calibration_probabilities, true_labels, score_fn=bad_score_fn)


def test_calibrate_validates_first_dimension_mismatch() -> None:
    """Test that _calibrate validates matching first dimensions."""
    calibration_probabilities = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1]])
    true_labels = np.array([0])  # Wrong length

    def score_fn(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.array([0.5])

    with pytest.raises(ValueError, match="Shape mismatch"):
        _calibrate(calibration_probabilities, true_labels, score_fn=score_fn)


def test_calibrate_validates_score_fn_output_length() -> None:
    """Test that _calibrate validates score function returns correct length."""
    calibration_probabilities = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1]])
    true_labels = np.array([0, 1])

    # Score function that returns wrong number of scores
    def bad_score_fn(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.array([0.5])  # Only 1 score instead of 2

    with pytest.raises(ValueError, match="score_fn must return array with 2 examples"):
        _calibrate(calibration_probabilities, true_labels, score_fn=bad_score_fn)


def test_calibrate_with_2d_scores() -> None:
    """Test _calibrate with 2D score output (multi-output case)."""
    calibration_predictions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    true_labels = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Score function that returns 2D scores
    def score_fn_2d(preds: np.ndarray, true_vals: np.ndarray) -> np.ndarray:
        return np.array([[0.5, 0.3], [0.7, 0.1], [0.2, 0.6]])

    scores = _calibrate(calibration_predictions, true_labels, score_fn=score_fn_2d)

    # Should sort along axis 0 (per column)
    expected = np.array([[0.2, 0.1], [0.5, 0.3], [0.7, 0.6]])
    np.testing.assert_array_almost_equal(scores, expected)

    # Verify each column is sorted
    assert np.all(scores[:-1, 0] <= scores[1:, 0])
    assert np.all(scores[:-1, 1] <= scores[1:, 1])
