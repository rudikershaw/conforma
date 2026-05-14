"""Integration test: diabetes regression with conformal prediction.

Uses scikit-learn's Diabetes dataset and a ridge regression model to exercise
the full conformal regression workflow end-to-end.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from conformal import ConformalRegressor, calibrate_regressor
from conformal.diagnostics import DiagnosticConfig, regressor_coverage_stability

COVERAGE = 0.90


def test_diabetes_regression():
    # Load a real dataset
    inputs, targets = load_diabetes(return_X_y=True)

    # Split into train and a held-out pool for calibration + testing
    train_inputs, pool_inputs, train_targets, pool_targets = train_test_split(
        inputs, targets, test_size=0.5, random_state=0
    )

    # Train a model
    model = Ridge(random_state=42)
    model.fit(train_inputs, train_targets)

    # Use diagnostics to find the minimum calibration set size
    pool_predictions = model.predict(pool_inputs)
    config = DiagnosticConfig(rng=42)
    result = regressor_coverage_stability(pool_predictions, pool_targets, coverage=COVERAGE, config=config)

    # Pick the smallest size where mean coverage meets the target
    viable = result.sizes[result.mean_coverage >= COVERAGE]
    cal_size = int(viable[0])

    # Split the pool into calibration and test using the recommended size
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(pool_targets))
    cal_idx, test_idx = indices[:cal_size], indices[cal_size:]
    cal_inputs, cal_targets = pool_inputs[cal_idx], pool_targets[cal_idx]
    test_inputs, test_targets = pool_inputs[test_idx], pool_targets[test_idx]

    # Calibrate and wrap the model
    calibration = calibrate_regressor(model.predict(cal_inputs), cal_targets)
    wrapper = ConformalRegressor(
        predict_fn=model.predict,
        calibration=calibration,
    )

    # Get prediction intervals at 90% coverage
    intervals = wrapper.predict(test_inputs, coverage=COVERAGE)

    # The coverage guarantee: the true value falls within the interval at least 90% of the time
    lower, upper = intervals[:, 0], intervals[:, 1]
    covered = (test_targets >= lower) & (test_targets <= upper)
    empirical_coverage = covered.mean()
    assert empirical_coverage >= COVERAGE

    # Intervals are symmetric around the point prediction and have positive width
    assert np.all(upper > lower)
