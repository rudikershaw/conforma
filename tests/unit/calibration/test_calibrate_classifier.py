"""Tests for the calibrate_classifier function."""

import numpy as np
import pytest

from conformal.calibration import calibrate_classifier
from tests.unit.conftest import default_benchmark


@default_benchmark
def test_calibrate_classifier_basic(benchmark) -> None:
    """Test basic calibration with simple probabilities."""
    calibration_probabilities = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])
    true_labels = np.array([0, 1, 2])

    scores = benchmark(calibrate_classifier, calibration_probabilities, true_labels)

    # Expected scores: 1 - [0.8, 0.6, 0.5] = [0.2, 0.4, 0.5], sorted
    expected = np.array([0.2, 0.4, 0.5])
    np.testing.assert_array_almost_equal(scores, expected)


@default_benchmark
def test_calibrate_classifier_returns_sorted(benchmark) -> None:
    """Test that scores are returned in sorted order."""
    calibration_probabilities = np.array([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]])
    true_labels = np.array([1, 0, 1])

    scores = benchmark(calibrate_classifier, calibration_probabilities, true_labels)

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

    # Label index too high
    with pytest.raises(ValueError, match="valid class indices"):
        calibrate_classifier(np.array([[0.8, 0.1, 0.1]]), np.array([3]))

    # Negative label index
    with pytest.raises(ValueError, match="valid class indices"):
        calibrate_classifier(np.array([[0.8, 0.1, 0.1]]), np.array([-1]))


@default_benchmark
def test_calibrate_classifier_with_perfect_predictions(benchmark) -> None:
    """Test calibration when model predicts perfectly."""
    calibration_probabilities = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    true_labels = np.array([0, 1, 0])

    scores = benchmark(calibrate_classifier, calibration_probabilities, true_labels)

    # Expected scores: 1 - [1.0, 1.0, 1.0] = [0.0, 0.0, 0.0]
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(scores, expected)


@default_benchmark
def test_calibrate_classifier_with_uniform_predictions(benchmark) -> None:
    """Test calibration when model predicts uniformly."""
    calibration_probabilities = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    true_labels = np.array([0, 1, 0])

    scores = benchmark(calibrate_classifier, calibration_probabilities, true_labels)

    # Expected scores: 1 - [0.5, 0.5, 0.5] = [0.5, 0.5, 0.5]
    expected = np.array([0.5, 0.5, 0.5])
    np.testing.assert_array_almost_equal(scores, expected)
