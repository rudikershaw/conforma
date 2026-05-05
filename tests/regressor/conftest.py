"""Shared fixtures for regressor tests."""

import numpy as np
import pytest

from conformal.calibration import calibrate_regressor
from conformal.regressor import ConformalRegressor


@pytest.fixture
def wrapper() -> ConformalRegressor:
    """A univariate ConformalRegressor with an identity predict_fn."""
    cal_preds = np.array([2.1, 5.3, 7.8, 3.2])
    cal_true = np.array([2.0, 5.0, 8.0, 3.5])
    calibration = calibrate_regressor(cal_preds, cal_true)
    return ConformalRegressor(predict_fn=lambda x: x, calibration=calibration)


@pytest.fixture
def multioutput_wrapper() -> ConformalRegressor:
    """A multi-output ConformalRegressor with an identity predict_fn."""
    cal_preds = np.array([[1.1, 2.2], [3.4, 4.3], [5.6, 6.5], [7.1, 8.2]])
    cal_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    calibration = calibrate_regressor(cal_preds, cal_true)
    return ConformalRegressor(predict_fn=lambda x: x, calibration=calibration)
