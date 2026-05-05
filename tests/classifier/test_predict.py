"""Tests for ConformalClassifier.predict."""

import numpy as np
import pytest

from conformal.classifier import ConformalClassifier
from tests.conftest import default_benchmark


@default_benchmark
def test_predict_basic(benchmark, wrapper: ConformalClassifier) -> None:
    """Test that predict produces boolean prediction sets."""
    test_probs = np.array([[0.9, 0.05, 0.05]])

    prediction_set = benchmark(wrapper.predict, test_probs, 0.5)

    assert prediction_set.dtype == np.bool_
    assert prediction_set.shape == (1, 3)


@default_benchmark
def test_predict_lower_threshold_includes_more_classes(benchmark, wrapper: ConformalClassifier) -> None:
    """Test that a lower coverage threshold includes more classes in the prediction set."""
    test_probs = np.array([[0.7, 0.2, 0.1]])

    narrow = wrapper.predict(test_probs, 0.9)
    wide = benchmark(wrapper.predict, test_probs, 0.1)

    assert wide.sum() >= narrow.sum()


@default_benchmark
def test_predict_matches_p_value_threshold(benchmark, wrapper: ConformalClassifier) -> None:
    """Test that predict is equivalent to thresholding p-values."""
    test_probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    coverage = 0.5

    prediction_set = benchmark(wrapper.predict, test_probs, coverage)
    p_values = wrapper.predict_p_values(test_probs)

    np.testing.assert_array_equal(prediction_set, p_values >= coverage)


def test_predict_coverage_validation(wrapper: ConformalClassifier) -> None:
    """Test that coverage outside (0, 1] is rejected."""
    test_probs = np.array([[0.7, 0.2, 0.1]])

    with pytest.raises(ValueError, match="coverage must be in"):
        wrapper.predict(test_probs, 0.0)

    with pytest.raises(ValueError, match="coverage must be in"):
        wrapper.predict(test_probs, 1.5)

    with pytest.raises(ValueError, match="coverage must be in"):
        wrapper.predict(test_probs, -0.1)
