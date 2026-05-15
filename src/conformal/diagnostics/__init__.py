"""Diagnostic tools for analysing conformal predictor behaviour.

This module provides functions that help practitioners understand how
calibration set size affects prediction quality.
"""

from conformal.diagnostics._common import CalibrationPlan, CoverageStabilityResult, DiagnosticConfig
from conformal.diagnostics.classifier import classifier_calibration_plan, classifier_coverage_stability
from conformal.diagnostics.regressor import regressor_coverage_stability

__all__ = [
    "CalibrationPlan",
    "CoverageStabilityResult",
    "DiagnosticConfig",
    "classifier_calibration_plan",
    "classifier_coverage_stability",
    "regressor_coverage_stability",
]
