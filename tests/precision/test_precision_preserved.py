"""Tests that all public functions preserve floating-point precision."""

import importlib
import inspect
import pkgutil

import numpy as np
import pytest

import conformal
from conformal.calibration import calibrate_classifier, calibrate_regressor
from conformal.core import compute_p_values


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


PRECISION_REGISTRY = [
    (compute_p_values, _make_compute_p_values_inputs),
    (calibrate_classifier, _make_calibrate_classifier_inputs),
    (calibrate_regressor, _make_calibrate_regressor_inputs),
]

PRECISION_EXEMPT: frozenset[str] = frozenset()


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64], ids=lambda d: d.__name__)
@pytest.mark.parametrize(
    "func,factory",
    PRECISION_REGISTRY,
    ids=lambda x: x.__name__ if callable(x) else "",
)
def test_precision_preserved(func, factory, dtype):
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
    inputs = factory(dtype_a, dtype_b)
    result = func(*inputs)
    expected = np.result_type(*[a for a in inputs if np.issubdtype(a.dtype, np.floating)])
    assert result.dtype == expected, (
        f"{func.__name__} returned {result.dtype}, expected {expected} "
        f"for inputs {dtype_a.__name__} + {dtype_b.__name__}"
    )


def test_precision_registry_is_complete():
    public_functions: set[str] = set()
    for module_info in pkgutil.walk_packages(conformal.__path__, prefix="conformal."):
        module = importlib.import_module(module_info.name)
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("_") and getattr(obj, "__module__", "") == module_info.name:
                public_functions.add(name)

    for name, obj in inspect.getmembers(conformal, inspect.isfunction):
        if not name.startswith("_") and getattr(obj, "__module__", "").startswith("conformal"):
            public_functions.add(name)

    public_functions -= PRECISION_EXEMPT
    registered = {f.__name__ for f, _ in PRECISION_REGISTRY}
    missing = public_functions - registered
    assert not missing, (
        f"Public functions missing from PRECISION_REGISTRY in test_precision_preserved.py: {missing}. "
        f"Add an entry with an input factory for each."
    )
