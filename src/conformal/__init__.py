"""Conformal prediction for practitioners.

A minimal, framework-agnostic library for producing statistically guaranteed
uncertainty estimates from ML models.
"""

from importlib.metadata import version

from conformal.calibration import calibrate_classifier, calibrate_regressor

__version__ = version("conformal")

__all__ = ["__version__", "calibrate_classifier", "calibrate_regressor"]
