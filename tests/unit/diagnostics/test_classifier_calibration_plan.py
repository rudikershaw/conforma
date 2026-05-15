"""Tests for classifier_calibration_plan."""

import numpy as np
import pytest

from conformal.diagnostics import (
    ClassifierCalibrationPlan,
    DiagnosticConfig,
    classifier_calibration_plan,
)

SEARCH_LOWER_BOUND = 0.5


@pytest.fixture(scope="module")
def synthetic_data():
    """Synthetic classification probabilities and labels for diagnostics tests."""
    rng = np.random.default_rng(42)
    n = 200
    n_classes = 3
    true_labels = rng.integers(0, n_classes, size=n)

    probs = rng.dirichlet(np.ones(n_classes), size=n).astype(np.float64)
    for i in range(n):
        probs[i, true_labels[i]] += 1.0
    probs = probs / probs.sum(axis=1, keepdims=True)

    return probs, true_labels


def test_returns_calibration_plan(synthetic_data) -> None:
    probs, labels = synthetic_data
    config = DiagnosticConfig(sizes=[20, 50], n_repetitions=5, rng=0)
    result = classifier_calibration_plan(probs, labels, config=config)
    assert isinstance(result, ClassifierCalibrationPlan)


def test_recommendation_respects_max_set_size(synthetic_data) -> None:
    probs, labels = synthetic_data
    config = DiagnosticConfig(sizes=[20, 50, 100], n_repetitions=5, rng=0)
    result = classifier_calibration_plan(probs, labels, max_set_size=1.05, config=config)
    assert result.coverage > SEARCH_LOWER_BOUND
    assert result.cal_size in result.cal_sizes


def test_table_shapes_are_consistent(synthetic_data) -> None:
    probs, labels = synthetic_data
    config = DiagnosticConfig(sizes=[20, 50], n_repetitions=5, rng=0)
    result = classifier_calibration_plan(probs, labels, config=config)
    n_rows = len(result.mean_coverage)
    assert result.cal_sizes.shape == (n_rows,)
    assert result.std_coverage.shape == (n_rows,)
    assert result.mean_set_size.shape == (n_rows,)
    assert result.std_set_size.shape == (n_rows,)


def test_mean_coverage_is_sorted(synthetic_data) -> None:
    probs, labels = synthetic_data
    config = DiagnosticConfig(sizes=[20, 50], n_repetitions=5, rng=0)
    result = classifier_calibration_plan(probs, labels, config=config)
    assert np.all(np.diff(result.mean_coverage) >= 0)


def test_tighter_max_set_size_lowers_coverage(synthetic_data) -> None:
    probs, labels = synthetic_data
    config_loose = DiagnosticConfig(sizes=[20, 50, 100], n_repetitions=5, rng=0)
    config_tight = DiagnosticConfig(sizes=[20, 50, 100], n_repetitions=5, rng=0)
    loose = classifier_calibration_plan(probs, labels, max_set_size=2.0, config=config_loose)
    tight = classifier_calibration_plan(probs, labels, max_set_size=1.01, config=config_tight)
    assert loose.coverage >= tight.coverage


def test_reproducible_with_seed(synthetic_data) -> None:
    probs, labels = synthetic_data
    config1 = DiagnosticConfig(sizes=[20, 50], n_repetitions=5, rng=42)
    config2 = DiagnosticConfig(sizes=[20, 50], n_repetitions=5, rng=42)
    r1 = classifier_calibration_plan(probs, labels, config=config1)
    r2 = classifier_calibration_plan(probs, labels, config=config2)
    assert r1.coverage == r2.coverage
    assert r1.cal_size == r2.cal_size
    np.testing.assert_array_equal(r1.mean_coverage, r2.mean_coverage)


def test_shape_mismatch_raises(synthetic_data) -> None:
    probs, _labels = synthetic_data
    wrong_length_labels = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="Shape mismatch"):
        classifier_calibration_plan(probs, wrong_length_labels)
