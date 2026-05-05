"""Tests for ConformalClassifier.predict_p_values."""

import numpy as np

from conformal.classifier import ConformalClassifier
from tests.conftest import default_benchmark


@default_benchmark
def test_predict_p_values_basic(benchmark, wrapper: ConformalClassifier) -> None:
    """Test that p-values are correctly computed through the wrapper."""
    test_probs = np.array([[0.7, 0.3, 0.0]])
    p_values = benchmark(wrapper.predict_p_values, test_probs)

    # e.g. class 0: nonconformity = 1 - 0.7 = 0.3, p = (3 + 1 - 1) / (3 + 1) = 0.75
    expected = np.array([[0.75, 0.25, 0.25]])
    np.testing.assert_array_almost_equal(p_values, expected)


@default_benchmark
def test_predict_p_values_shape(benchmark, wrapper: ConformalClassifier) -> None:
    """Test that output shape matches (n_examples, n_classes)."""
    test_probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])

    p_values = benchmark(wrapper.predict_p_values, test_probs)

    assert p_values.shape == (3, 3)


@default_benchmark
def test_predict_p_values_highest_for_predicted_class(benchmark, wrapper: ConformalClassifier) -> None:
    """Test that the most probable class has the highest p-value."""
    test_probs = np.array([[0.9, 0.05, 0.05]])

    p_values = benchmark(wrapper.predict_p_values, test_probs)

    assert p_values[0, 0] > p_values[0, 1]
    assert p_values[0, 0] > p_values[0, 2]
