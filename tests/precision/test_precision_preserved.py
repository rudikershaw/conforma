"""Tests that all public functions preserve floating-point precision."""

import importlib
import inspect
import pkgutil

import numpy as np
import pytest

import conformal
from conformal.calibration import calibrate_classifier, calibrate_regressor
from conformal.classifier import ConformalClassifier
from conformal.core import compute_p_values, compute_quantile
from conformal.regressor import ConformalRegressor


def _make_compute_p_values_inputs(dtype_a, dtype_b):
    cal = np.array([0.1, 0.3, 0.5, 0.7], dtype=dtype_a)
    pred = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=dtype_b)
    return cal, pred


def _make_calibrate_classifier_inputs(dtype_a, _dtype_b):
    probs = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.2, 0.3, 0.5]], dtype=dtype_a)
    labels = np.array([0, 1, 2])
    return probs, labels


def _make_calibrate_regressor_inputs(dtype_a, dtype_b):
    preds = np.array([2.1, 5.3, 7.8], dtype=dtype_a)
    true_vals = np.array([2.0, 5.0, 8.0], dtype=dtype_b)
    return preds, true_vals


def _make_predict_p_values_inputs(dtype_a, dtype_b):
    cal_probs = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.2, 0.3, 0.5]], dtype=dtype_a)
    cal_labels = np.array([0, 1, 2])
    calibration = calibrate_classifier(cal_probs, cal_labels)
    wrapper = ConformalClassifier(predict_fn=lambda x: x, calibration=calibration)
    test_probs = np.array([[0.7, 0.2, 0.1]], dtype=dtype_b)
    return wrapper, test_probs


def _make_compute_quantile_inputs(dtype_a, _dtype_b):
    scores = np.array([0.1, 0.2, 0.3, 0.5], dtype=dtype_a)
    return scores, 0.5


def _make_regressor_predict_inputs(dtype_a, dtype_b):
    cal_preds = np.array([2.1, 5.3, 7.8, 3.2], dtype=dtype_a)
    cal_true = np.array([2.0, 5.0, 8.0, 3.5], dtype=dtype_b)
    calibration = calibrate_regressor(cal_preds, cal_true)
    wrapper = ConformalRegressor(predict_fn=lambda x: x, calibration=calibration)
    test_preds = np.array([4.0, 6.0], dtype=dtype_b)
    return wrapper, test_preds, 0.5


PRECISION_REGISTRY = (
    (compute_p_values, _make_compute_p_values_inputs),
    (compute_quantile, _make_compute_quantile_inputs),
    (calibrate_classifier, _make_calibrate_classifier_inputs),
    (calibrate_regressor, _make_calibrate_regressor_inputs),
    (ConformalClassifier.predict_p_values, _make_predict_p_values_inputs),
    (ConformalRegressor.predict, _make_regressor_predict_inputs),
)

PRECISION_EXEMPT: frozenset[str] = frozenset(
    {
        "ConformalClassifier.predict",
        "classifier_coverage_stability",
        "regressor_coverage_stability",
    }
)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64], ids=lambda d: d.__name__)
@pytest.mark.parametrize(
    "func,factory",
    PRECISION_REGISTRY,
    ids=lambda x: x.__name__ if callable(x) else "",
)
def test_precision_preserved(func, factory, dtype):
    """Test that uniform-precision inputs produce output with the same precision."""
    inputs = factory(dtype, dtype)
    result = func(*inputs)
    assert result.dtype == dtype, f"{func.__name__} returned {result.dtype}, expected {dtype}"


@pytest.mark.parametrize(
    "dtype_a,dtype_b",
    [(np.float16, np.float32), (np.float16, np.float64), (np.float32, np.float64)],
    ids=lambda x: x.__name__,
)
@pytest.mark.parametrize(
    "func,factory",
    PRECISION_REGISTRY,
    ids=lambda x: x.__name__ if callable(x) else "",
)
def test_mixed_precision_uses_result_type(func, factory, dtype_a, dtype_b):
    """Test that mixed-precision inputs upcast following np.result_type."""
    inputs = factory(dtype_a, dtype_b)
    result = func(*inputs)
    float_dtypes = [a for a in inputs if hasattr(a, "dtype") and np.issubdtype(a.dtype, np.floating)]
    expected = np.result_type(*float_dtypes)
    assert result.dtype == expected, (
        f"{func.__name__} returned {result.dtype}, expected {expected} "
        f"for inputs {dtype_a.__name__} + {dtype_b.__name__}"
    )


def test_precision_registry_is_complete():
    """Test that every public callable is registered for precision testing or explicitly exempt."""
    public_callables: set[str] = set()
    for module_info in pkgutil.walk_packages(conformal.__path__, prefix="conformal."):
        module = importlib.import_module(module_info.name)
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("_") and getattr(obj, "__module__", "") == module_info.name:
                public_callables.add(name)
        for class_name, cls in inspect.getmembers(module, inspect.isclass):
            if class_name.startswith("_") or getattr(cls, "__module__", "") != module_info.name:
                continue
            for method_name, _ in inspect.getmembers(cls, inspect.isfunction):
                if not method_name.startswith("_"):
                    public_callables.add(f"{class_name}.{method_name}")

    for name, obj in inspect.getmembers(conformal, inspect.isfunction):
        if not name.startswith("_") and getattr(obj, "__module__", "").startswith("conformal"):
            public_callables.add(name)

    public_callables -= PRECISION_EXEMPT
    registered: set[str] = set()
    for func, _ in PRECISION_REGISTRY:
        if "." in getattr(func, "__qualname__", ""):
            cls_name, method_name = func.__qualname__.rsplit(".", 1)
            registered.add(f"{cls_name}.{method_name}")
        else:
            registered.add(func.__name__)
    missing = public_callables - registered
    assert not missing, (
        f"Public callables missing from PRECISION_REGISTRY in test_precision_preserved.py: {missing}. "
        f"Add an entry with an input factory for each, or add to PRECISION_EXEMPT."
    )
