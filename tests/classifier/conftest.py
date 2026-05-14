"""Shared fixtures for classifier tests."""

import numpy as np
import pytest

from conformal.calibration import calibrate_classifier
from conformal.classifier import ConformalClassifier


@pytest.fixture(scope="module")
def wrapper() -> ConformalClassifier:
    """A ConformalClassifier with an identity predict_fn for direct score control."""
    cal_probs = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])
    cal_labels = np.array([0, 1, 2])
    calibration = calibrate_classifier(cal_probs, cal_labels)
    return ConformalClassifier(predict_fn=lambda x: x, calibration=calibration)
