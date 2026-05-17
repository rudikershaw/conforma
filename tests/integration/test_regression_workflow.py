"""Predicting diabetes progression measures with scikit-learn and conformal prediction."""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from conforma import ConformalRegressor, calibrate_regressor
from conforma.diagnostics import DiagnosticConfig, regressor_calibration_plan

MAX_INTERVAL_WIDTH = 250.0


def test_diabetes_regression():
    # Load a real dataset
    inputs, targets = load_diabetes(return_X_y=True)

    # Split into train and a held-out reserve for calibration + testing
    train_inputs, reserve_inputs, train_targets, reserve_targets = train_test_split(
        inputs, targets, test_size=0.5, random_state=0
    )

    # Train a model
    model = Ridge(random_state=42)
    model.fit(train_inputs, train_targets)

    # Use diagnostics to recommend a coverage level and calibration set size
    reserve_predictions = model.predict(reserve_inputs)
    config = DiagnosticConfig(rng=42)
    plan = regressor_calibration_plan(
        reserve_predictions, reserve_targets, max_interval_width=MAX_INTERVAL_WIDTH, config=config
    )

    # Split the reserve into calibration and test using the recommended size
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(reserve_targets))
    cal_idx, test_idx = indices[: plan.cal_size], indices[plan.cal_size :]
    cal_inputs, cal_targets = reserve_inputs[cal_idx], reserve_targets[cal_idx]
    test_inputs, test_targets = reserve_inputs[test_idx], reserve_targets[test_idx]

    # Calibrate and wrap the model
    calibration = calibrate_regressor(model.predict(cal_inputs), cal_targets)
    wrapper = ConformalRegressor(
        predict_fn=model.predict,
        calibration=calibration,
    )

    # Get prediction intervals at the recommended coverage
    intervals = wrapper.predict(test_inputs, coverage=plan.coverage)

    # The coverage guarantee: the true value falls within the interval at least `plan.coverage` of the time
    lower, upper = intervals[:, 0], intervals[:, 1]
    covered = (test_targets >= lower) & (test_targets <= upper)
    empirical_coverage = covered.mean()
    assert empirical_coverage >= plan.coverage

    # Widths respect the budget requested via max_interval_width
    widths = upper - lower
    assert widths.mean() <= MAX_INTERVAL_WIDTH

    # Intervals are symmetric around the point prediction and have positive width
    assert np.all(upper > lower)
