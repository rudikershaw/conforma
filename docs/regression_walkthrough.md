# Using conformal with a Regressor

The following code is an example of using conformal to calibrate a scikit-learn regressor (Ridge) from the project's integration tests. You can find the code file in this project under [tests/integration/test_regression_workflow.py](tests/integration/test_regression_workflow.py). After reading through the example below, you can find a discussion of what each step achieves below the code example.

<!-- INSERT_CODE:tests/integration/test_regression_workflow.py -->
```py
"""Predicting diabetes progression measures with scikit-learn and conformal prediction. 
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from conformal import ConformalRegressor, calibrate_regressor
from conformal.diagnostics import DiagnosticConfig, regressor_calibration_plan

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
```
<!--CODE_END -->

First we pull in our dataset, split our data into training and a reserve sets, and then train our model on the training set. This is all achieved outside of the scope of the conformal library (in this case using scikit-learn).

Now, we need to choose the size of our calibration set and the coverage level. Typically calibration sets can be effective with a surprisingly small number of entries, and the best coverage depends on your desired interval width. Rather than choosing the calibration set size and coverage manually through trial and error, this example uses the `regressor_calibration_plan` function to recommend the best achievable coverage given a maximum average interval width.

The `regressor_calibration_plan` examines the model's predictions on the reserve set and returns a `plan` containing both the recommended coverage and the smallest calibration set size needed to meet the `max_interval_width` budget. Once we have that plan, we split the reserve set into calibration and held-out test subsets using the suggested calibration size.

Next, we calibrate the model using the calibration set. The `calibrate_regressor` function takes the model's point predictions and the true calibration targets, and computes the calibration data needed to form prediction intervals. That calibration data is then used to create a `ConformalRegressor` wrapper around the original model's `predict` function.

With the calibrated wrapper, we can now make conformal prediction intervals. The `predict` method returns two-sided intervals at the specified coverage level. The coverage guarantee means that the true target value falls within the predicted interval at least as often as the requested coverage. At the same time, the intervals are constructed to respect the width budget imposed by `max_interval_width` on average.

In this regression setting, the wrapper produces intervals instead of class sets. These intervals are symmetric around the point prediction and have positive width, so you can use them directly to quantify uncertainty around numerical predictions.