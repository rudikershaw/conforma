"""Tests for ConformalRegressor.predict."""

import numpy as np
import pytest

from conformal.regressor import ConformalRegressor
from tests.unit.conftest import default_benchmark


@default_benchmark
def test_predict_univariate_basic(benchmark, wrapper: ConformalRegressor) -> None:
    """Test that univariate prediction intervals have correct shape and structure."""
    intervals = benchmark(wrapper.predict, np.array([4.0, 6.0]), 0.5)

    assert intervals.shape == (2, 2)
    assert np.all(intervals[:, 0] <= intervals[:, 1])


@default_benchmark
def test_predict_univariate_symmetric(benchmark, wrapper: ConformalRegressor) -> None:
    """Test that intervals are symmetric around the point prediction."""
    point_predictions = np.array([4.0, 6.0])
    intervals = benchmark(wrapper.predict, point_predictions, 0.5)

    midpoints = (intervals[:, 0] + intervals[:, 1]) / 2
    np.testing.assert_array_almost_equal(midpoints, point_predictions)


@default_benchmark
def test_predict_univariate_values(benchmark, wrapper: ConformalRegressor) -> None:
    """Test that interval bounds match the expected quantile."""
    # index = ceil(coverage * (n + 1)) - 1; interval = [prediction ± quantile]
    intervals = benchmark(wrapper.predict, np.array([4.0]), 0.5)

    expected = np.array([[3.7, 4.3]])
    np.testing.assert_array_almost_equal(intervals, expected)


@default_benchmark
def test_predict_higher_coverage_wider_intervals(benchmark, wrapper: ConformalRegressor) -> None:
    """Test that higher coverage produces wider or equal intervals."""
    point = np.array([4.0])
    narrow = wrapper.predict(point, 0.25)
    wide = benchmark(wrapper.predict, point, 0.75)

    narrow_width = narrow[0, 1] - narrow[0, 0]
    wide_width = wide[0, 1] - wide[0, 0]
    assert wide_width >= narrow_width


@default_benchmark
def test_predict_multioutput_basic(benchmark, multioutput_wrapper: ConformalRegressor) -> None:
    """Test that multi-output prediction intervals have correct shape."""
    intervals = benchmark(multioutput_wrapper.predict, np.array([[2.0, 3.0], [4.0, 5.0]]), 0.5)

    assert intervals.shape == (2, 2, 2)
    assert np.all(intervals[:, :, 0] <= intervals[:, :, 1])


@default_benchmark
def test_predict_multioutput_symmetric(benchmark, multioutput_wrapper: ConformalRegressor) -> None:
    """Test that multi-output intervals are symmetric around point predictions."""
    point_predictions = np.array([[2.0, 3.0], [4.0, 5.0]])
    intervals = benchmark(multioutput_wrapper.predict, point_predictions, 0.5)

    midpoints = (intervals[:, :, 0] + intervals[:, :, 1]) / 2
    np.testing.assert_array_almost_equal(midpoints, point_predictions)


def test_predict_coverage_validation(wrapper: ConformalRegressor) -> None:
    """Test that coverage outside (0, 1] is rejected."""
    point = np.array([4.0])

    with pytest.raises(ValueError, match="coverage must be in"):
        wrapper.predict(point, 0.0)

    with pytest.raises(ValueError, match="coverage must be in"):
        wrapper.predict(point, 1.5)

    with pytest.raises(ValueError, match="coverage must be in"):
        wrapper.predict(point, -0.1)


def test_predict_coverage_too_high_for_calibration_size(wrapper: ConformalRegressor) -> None:
    """Test that unreachable coverage raises with a helpful message."""
    with pytest.raises(ValueError, match="Calibration set too small"):
        wrapper.predict(np.array([4.0]), coverage=1.0)
