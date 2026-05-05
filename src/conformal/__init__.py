"""Conformal prediction for practitioners.

A minimal, framework-agnostic library for producing statistically guaranteed
uncertainty estimates from ML models.
"""

from importlib.metadata import version

from conformal.calibration import calibrate_classifier, calibrate_regressor
from conformal.classifier import ConformalClassifier
from conformal.regressor import ConformalRegressor

__version__ = version("conformal")

__all__ = [
    "ConformalClassifier",
    "ConformalRegressor",
    "__version__",
    "calibrate_classifier",
    "calibrate_regressor",
]
