"""Tests for regressor_coverage_stability."""

import numpy as np
import pytest

from conformal.diagnostics import (
    CoverageStabilityResult,
    DiagnosticConfig,
    regressor_coverage_stability,
)
from tests.unit.conftest import default_benchmark


@pytest.fixture(scope="module")
def synthetic_data():
    """Synthetic regression predictions and true values for diagnostics tests."""
    rng = np.random.default_rng(42)
    n = 200
    true_values = rng.standard_normal(n).astype(np.float64)
    predictions = true_values + rng.normal(0, 0.3, size=n)
    return predictions, true_values


def test_returns_coverage_stability_result(synthetic_data) -> None:
    preds, true_vals = synthetic_data
    config = DiagnosticConfig(sizes=[20, 50], n_repetitions=5, rng=0)
    result = regressor_coverage_stability(preds, true_vals, coverage=0.5, config=config)
    assert isinstance(result, CoverageStabilityResult)


def test_result_shapes_match_sizes(synthetic_data) -> None:
    preds, true_vals = synthetic_data
    requested_sizes = [20, 50, 100]
    config = DiagnosticConfig(sizes=requested_sizes, n_repetitions=5, rng=0)
    result = regressor_coverage_stability(preds, true_vals, coverage=0.5, config=config)
    expected_length = len(requested_sizes)
    assert len(result.sizes) == expected_length
    assert len(result.mean_coverage) == expected_length
    assert len(result.std_coverage) == expected_length
    assert len(result.mean_set_size) == expected_length
    assert len(result.std_set_size) == expected_length


@default_benchmark
def test_custom_sizes(benchmark, synthetic_data) -> None:
    preds, true_vals = synthetic_data
    config = DiagnosticConfig(sizes=[20, 50, 100], n_repetitions=5, rng=0)
    result = benchmark(regressor_coverage_stability, preds, true_vals, coverage=0.5, config=config)
    np.testing.assert_array_equal(result.sizes, [20, 50, 100])


@default_benchmark
def test_coverage_improves_with_size(benchmark, synthetic_data) -> None:
    preds, true_vals = synthetic_data
    config = DiagnosticConfig(sizes=[20, 50, 100, 180], n_repetitions=5, rng=42)
    result = benchmark(regressor_coverage_stability, preds, true_vals, coverage=0.50, config=config)
    assert result.mean_coverage[-1] >= result.mean_coverage[0] - 0.05


@default_benchmark
def test_stability_improves_with_size(benchmark, synthetic_data) -> None:
    preds, true_vals = synthetic_data
    config = DiagnosticConfig(sizes=[20, 50, 100, 180], n_repetitions=5, rng=42)
    result = benchmark(regressor_coverage_stability, preds, true_vals, coverage=0.50, config=config)
    assert result.std_coverage[-1] <= result.std_coverage[0] + 0.01


@default_benchmark
def test_perfect_model(benchmark) -> None:
    n = 100
    values = np.linspace(0, 10, n, dtype=np.float64)
    config = DiagnosticConfig(sizes=[20, 60], n_repetitions=5, rng=0)
    result = benchmark(regressor_coverage_stability, values, values, coverage=0.5, config=config)
    np.testing.assert_array_less(0.99, result.mean_coverage)
    np.testing.assert_allclose(result.mean_set_size, 0.0, atol=0.01)


@default_benchmark
def test_default_sizes_inferred(benchmark, synthetic_data) -> None:
    preds, true_vals = synthetic_data
    config = DiagnosticConfig(n_repetitions=5, rng=0)
    result = benchmark(regressor_coverage_stability, preds, true_vals, coverage=0.80, config=config)
    assert len(result.sizes) > 0
    minimum_viable_size = 4
    assert result.sizes[0] >= minimum_viable_size
    assert result.sizes[-1] <= len(preds)


def test_default_sizes_respects_minimum(synthetic_data) -> None:
    preds, true_vals = synthetic_data
    target_coverage = 0.90
    config = DiagnosticConfig(n_repetitions=5, rng=0)
    result = regressor_coverage_stability(preds, true_vals, coverage=target_coverage, config=config)
    n_cal = int(result.sizes[0]) // 2
    assert n_cal / (n_cal + 1) >= target_coverage


def test_reproducible_with_seed(synthetic_data) -> None:
    preds, true_vals = synthetic_data
    config = DiagnosticConfig(sizes=[30, 60], n_repetitions=5, rng=123)
    r1 = regressor_coverage_stability(preds, true_vals, config=config)
    config2 = DiagnosticConfig(sizes=[30, 60], n_repetitions=5, rng=123)
    r2 = regressor_coverage_stability(preds, true_vals, config=config2)
    np.testing.assert_array_equal(r1.mean_coverage, r2.mean_coverage)
    np.testing.assert_array_equal(r1.mean_set_size, r2.mean_set_size)


def test_single_size(synthetic_data) -> None:
    preds, true_vals = synthetic_data
    config = DiagnosticConfig(sizes=[40], n_repetitions=5, rng=0)
    result = regressor_coverage_stability(preds, true_vals, config=config)
    assert len(result.sizes) == 1


def test_shape_mismatch_raises(synthetic_data) -> None:
    preds, _true_vals = synthetic_data
    wrong_length_true = np.array([0.0, 1.0, 2.0])
    config = DiagnosticConfig(sizes=[20], n_repetitions=1, rng=0)
    with pytest.raises(ValueError, match="Shape mismatch"):
        regressor_coverage_stability(preds, wrong_length_true, coverage=0.5, config=config)


def test_coverage_out_of_range() -> None:
    preds = np.array([1.0] * 10)
    true_vals = np.array([1.0] * 10)
    with pytest.raises(ValueError, match="coverage must be in"):
        regressor_coverage_stability(preds, true_vals, coverage=0.0)
    with pytest.raises(ValueError, match="coverage must be in"):
        regressor_coverage_stability(preds, true_vals, coverage=1.0)
    with pytest.raises(ValueError, match="coverage must be in"):
        regressor_coverage_stability(preds, true_vals, coverage=1.5)


def test_size_too_small_raises(synthetic_data) -> None:
    preds, true_vals = synthetic_data
    config = DiagnosticConfig(sizes=[4])
    with pytest.raises(ValueError, match="too small"):
        regressor_coverage_stability(preds, true_vals, coverage=0.90, config=config)


def test_size_exceeds_data_raises() -> None:
    preds = np.array([1.0] * 10)
    true_vals = np.array([1.0] * 10)
    config = DiagnosticConfig(sizes=[20])
    with pytest.raises(ValueError, match="exceeds"):
        regressor_coverage_stability(preds, true_vals, coverage=0.5, config=config)


def test_invalid_n_repetitions(synthetic_data) -> None:
    preds, true_vals = synthetic_data
    config = DiagnosticConfig(sizes=[30], n_repetitions=0)
    with pytest.raises(ValueError, match="n_repetitions"):
        regressor_coverage_stability(preds, true_vals, config=config)
