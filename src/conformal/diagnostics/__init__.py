"""Diagnostic tools for analysing conformal predictor behaviour.

This module provides functions that help practitioners understand how
calibration set size affects prediction quality.
"""

from conformal.diagnostics._common import DiagnosticConfig
from conformal.diagnostics.classifier import (
    ClassifierCalibrationPlan,
    ClassifierCoverageStability,
    classifier_calibration_plan,
    classifier_coverage_stability,
)
from conformal.diagnostics.regressor import (
    RegressorCalibrationPlan,
    RegressorCoverageStability,
    regressor_calibration_plan,
    regressor_coverage_stability,
)

__all__ = [
    "ClassifierCalibrationPlan",
    "ClassifierCoverageStability",
    "DiagnosticConfig",
    "RegressorCalibrationPlan",
    "RegressorCoverageStability",
    "classifier_calibration_plan",
    "classifier_coverage_stability",
    "regressor_calibration_plan",
    "regressor_coverage_stability",
]
