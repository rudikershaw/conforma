"""Diagnostic tools for analysing conformal predictor behaviour.

This module provides functions that help practitioners understand how
calibration set size affects prediction quality.
"""

from conformal.diagnostics._common import CoverageStabilityResult, DiagnosticConfig
from conformal.diagnostics.classifier import classifier_coverage_stability
from conformal.diagnostics.regressor import regressor_coverage_stability

__all__ = [
    "CoverageStabilityResult",
    "DiagnosticConfig",
    "classifier_coverage_stability",
    "regressor_coverage_stability",
]
