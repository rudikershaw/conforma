"""Conformal prediction for practitioners.

A minimal, framework-agnostic library for producing statistically guaranteed
uncertainty estimates from ML models.
"""

from importlib.metadata import version

from conforma.calibration import calibrate_classifier, calibrate_regressor
from conforma.classifier import ConformalClassifier
from conforma.regressor import ConformalRegressor

__version__ = version("conforma")

__all__ = [
    "ConformalClassifier",
    "ConformalRegressor",
    "__version__",
    "calibrate_classifier",
    "calibrate_regressor",
]
