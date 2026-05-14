"""Tests for the compute_p_values function."""

import numpy as np
import pytest

from conformal.core import compute_p_values
from tests.unit.conftest import default_benchmark


@default_benchmark
def test_compute_p_values_1d_calibration_2d_predictions(benchmark) -> None:
    """Test p-values with 1D calibration scores and 2D prediction scores (classification use case)."""
    calibration_scores = np.array([0.1, 0.3, 0.5, 0.7])
    prediction_scores = np.array([[0.2, 0.8], [0.6, 0.4]])

    p_values = benchmark(compute_p_values, calibration_scores, prediction_scores)

    # Formula: (n_cal + 1 - rank) / (n_cal + 1), n_cal=4, denominator=5
    # score 0.2: rank=1, (5-1)/5 = 0.8
    # score 0.8: rank=4, (5-4)/5 = 0.2
    # score 0.6: rank=3, (5-3)/5 = 0.4
    # score 0.4: rank=2, (5-2)/5 = 0.6
    expected = np.array([[0.8, 0.2], [0.4, 0.6]])
    np.testing.assert_array_almost_equal(p_values, expected)


@default_benchmark
def test_compute_p_values_2d_calibration_2d_predictions(benchmark) -> None:
    """Test p-values with 2D calibration and prediction scores (multi-output regression)."""
    calibration_scores = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    prediction_scores = np.array([[0.2, 0.5], [0.4, 0.3]])

    p_values = benchmark(compute_p_values, calibration_scores, prediction_scores)

    # Formula: (n_cal + 1 - rank) / (n_cal + 1), n_cal=3, denominator=4
    # Column 0: cal=[0.1, 0.3, 0.5], score 0.2 rank=1 → 3/4, score 0.4 rank=2 → 2/4
    # Column 1: cal=[0.2, 0.4, 0.6], score 0.5 rank=2 → 2/4, score 0.3 rank=1 → 3/4
    expected = np.array([[3 / 4, 2 / 4], [2 / 4, 3 / 4]])
    np.testing.assert_array_almost_equal(p_values, expected)


@default_benchmark
def test_compute_p_values_1d_both(benchmark) -> None:
    """Test p-values with 1D calibration and 1D prediction scores (univariate regression)."""
    calibration_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    prediction_scores = np.array([0.0, 0.25, 0.5, 0.6])

    p_values = benchmark(compute_p_values, calibration_scores, prediction_scores)

    # Formula: (n_cal + 1 - rank) / (n_cal + 1), n_cal=5, denominator=6
    # score 0.0: rank=0, 6/6 = 1.0
    # score 0.25: rank=2, 4/6 = 2/3
    # score 0.5: rank=4, 2/6 = 1/3
    # score 0.6: rank=5, 1/6
    expected = np.array([1.0, 2 / 3, 1 / 3, 1 / 6])
    np.testing.assert_array_almost_equal(p_values, expected)


@default_benchmark
def test_compute_p_values_all_below_calibration(benchmark) -> None:
    """Test that prediction scores below all calibration scores give p-value of 1.0."""
    calibration_scores = np.array([0.5, 0.6, 0.7, 0.8])
    prediction_scores = np.array([0.1, 0.2])

    p_values = benchmark(compute_p_values, calibration_scores, prediction_scores)

    expected = np.array([1.0, 1.0])
    np.testing.assert_array_almost_equal(p_values, expected)


@default_benchmark
def test_compute_p_values_all_above_calibration(benchmark) -> None:
    """Test that prediction scores above all calibration scores give minimum p-value of 1/(n+1)."""
    calibration_scores = np.array([0.1, 0.2, 0.3])
    prediction_scores = np.array([0.5, 0.9])

    p_values = benchmark(compute_p_values, calibration_scores, prediction_scores)

    expected = np.array([0.25, 0.25])
    np.testing.assert_array_almost_equal(p_values, expected)


@default_benchmark
def test_compute_p_values_exact_match_with_calibration(benchmark) -> None:
    """Test that prediction scores exactly matching calibration scores are counted."""
    calibration_scores = np.array([0.2, 0.4, 0.6, 0.8])
    prediction_scores = np.array([0.4])

    p_values = benchmark(compute_p_values, calibration_scores, prediction_scores)

    # n_cal=4, score 0.4: rank=1, (5-1)/5 = 0.8
    expected = np.array([0.8])
    np.testing.assert_array_almost_equal(p_values, expected)


@default_benchmark
def test_compute_p_values_preserves_float32(benchmark) -> None:
    """Test that float32 inputs produce float32 output without upcasting to float64."""
    calibration_scores = np.array([0.1, 0.3, 0.5], dtype=np.float32)
    prediction_scores = np.array([[0.2, 0.4]], dtype=np.float32)

    p_values = benchmark(compute_p_values, calibration_scores, prediction_scores)

    assert p_values.dtype == np.float32


@default_benchmark
def test_compute_p_values_mixed_precision_upcasts(benchmark) -> None:
    """Test that mixed float32/float64 inputs upcast to float64."""
    calibration_scores = np.array([0.1, 0.3, 0.5], dtype=np.float32)
    prediction_scores = np.array([[0.2, 0.4]], dtype=np.float64)

    p_values = benchmark(compute_p_values, calibration_scores, prediction_scores)

    assert p_values.dtype == np.float64


@default_benchmark
def test_compute_p_values_output_shape_matches_input(benchmark) -> None:
    """Test that output shape matches prediction_scores shape."""
    calibration_scores = np.array([0.1, 0.3, 0.5])
    prediction_scores = np.array([[0.2, 0.4, 0.6], [0.1, 0.5, 0.7]])

    p_values = benchmark(compute_p_values, calibration_scores, prediction_scores)

    assert p_values.shape == prediction_scores.shape


def test_compute_p_values_calibration_shape_validation() -> None:
    """Test that calibration_scores must be 1D or 2D."""
    with pytest.raises(ValueError, match="calibration_scores must be 1D or 2D"):
        compute_p_values(np.array([[[0.1]]]), np.array([0.2]))


def test_compute_p_values_prediction_shape_validation() -> None:
    """Test that prediction_scores must be 1D or 2D."""
    with pytest.raises(ValueError, match="prediction_scores must be 1D or 2D"):
        compute_p_values(np.array([0.1]), np.array([[[0.2]]]))


def test_compute_p_values_column_mismatch_validation() -> None:
    """Test that 2D calibration and prediction scores must have matching columns."""
    with pytest.raises(ValueError, match="Column mismatch"):
        compute_p_values(
            np.array([[0.1, 0.2], [0.3, 0.4]]),
            np.array([[0.1, 0.2, 0.3]]),
        )
