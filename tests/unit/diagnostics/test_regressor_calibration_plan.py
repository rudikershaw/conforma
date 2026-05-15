"""Tests for regressor_calibration_plan."""

import numpy as np
import pytest

from conformal.diagnostics import (
    DiagnosticConfig,
    RegressorCalibrationPlan,
    regressor_calibration_plan,
)

SEARCH_LOWER_BOUND = 0.5


@pytest.fixture(scope="module")
def synthetic_data():
    """Synthetic regression predictions and true values for diagnostics tests."""
    rng = np.random.default_rng(42)
    n = 200
    true_values = rng.standard_normal(n).astype(np.float64)
    predictions = true_values + rng.normal(0, 0.3, size=n)
    return predictions, true_values


def test_returns_calibration_plan(synthetic_data) -> None:
    preds, true_vals = synthetic_data
    config = DiagnosticConfig(sizes=[20, 50], n_repetitions=5, rng=0)
    result = regressor_calibration_plan(preds, true_vals, max_interval_width=5.0, config=config)
    assert isinstance(result, RegressorCalibrationPlan)


def test_recommendation_respects_max_interval_width(synthetic_data) -> None:
    preds, true_vals = synthetic_data
    config = DiagnosticConfig(sizes=[20, 50, 100], n_repetitions=5, rng=0)
    result = regressor_calibration_plan(preds, true_vals, max_interval_width=2.0, config=config)
    assert result.coverage > SEARCH_LOWER_BOUND
    assert result.cal_size in result.cal_sizes


def test_table_shapes_are_consistent(synthetic_data) -> None:
    preds, true_vals = synthetic_data
    config = DiagnosticConfig(sizes=[20, 50], n_repetitions=5, rng=0)
    result = regressor_calibration_plan(preds, true_vals, max_interval_width=2.0, config=config)
    n_rows = len(result.mean_coverage)
    assert result.cal_sizes.shape == (n_rows,)
    assert result.std_coverage.shape == (n_rows,)
    assert result.mean_interval_width.shape == (n_rows,)
    assert result.std_interval_width.shape == (n_rows,)


def test_mean_coverage_is_sorted(synthetic_data) -> None:
    preds, true_vals = synthetic_data
    config = DiagnosticConfig(sizes=[20, 50], n_repetitions=5, rng=0)
    result = regressor_calibration_plan(preds, true_vals, max_interval_width=2.0, config=config)
    assert np.all(np.diff(result.mean_coverage) >= 0)


def test_tighter_max_interval_width_lowers_coverage(synthetic_data) -> None:
    preds, true_vals = synthetic_data
    config_loose = DiagnosticConfig(sizes=[20, 50, 100], n_repetitions=5, rng=0)
    config_tight = DiagnosticConfig(sizes=[20, 50, 100], n_repetitions=5, rng=0)
    loose = regressor_calibration_plan(preds, true_vals, max_interval_width=10.0, config=config_loose)
    tight = regressor_calibration_plan(preds, true_vals, max_interval_width=0.5, config=config_tight)
    assert loose.coverage >= tight.coverage


def test_reproducible_with_seed(synthetic_data) -> None:
    preds, true_vals = synthetic_data
    config1 = DiagnosticConfig(sizes=[20, 50], n_repetitions=5, rng=42)
    config2 = DiagnosticConfig(sizes=[20, 50], n_repetitions=5, rng=42)
    r1 = regressor_calibration_plan(preds, true_vals, max_interval_width=2.0, config=config1)
    r2 = regressor_calibration_plan(preds, true_vals, max_interval_width=2.0, config=config2)
    assert r1.coverage == r2.coverage
    assert r1.cal_size == r2.cal_size
    np.testing.assert_array_equal(r1.mean_coverage, r2.mean_coverage)


def test_shape_mismatch_raises(synthetic_data) -> None:
    preds, _true_vals = synthetic_data
    wrong_length_true = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="Shape mismatch"):
        regressor_calibration_plan(preds, wrong_length_true, max_interval_width=2.0)
