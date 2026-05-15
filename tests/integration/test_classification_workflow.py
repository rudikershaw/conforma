"""Integration test: breast cancer classification with conformal prediction.

Uses scikit-learn's Breast Cancer Wisconsin dataset and a logistic regression
model to exercise the full conformal classification workflow end-to-end.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from conformal import ConformalClassifier, calibrate_classifier
from conformal.diagnostics import DiagnosticConfig, classifier_calibration_plan

MAX_SET_SIZE = 1.05


def test_breast_cancer_classification():
    # Load a real dataset
    inputs, labels = load_breast_cancer(return_X_y=True)

    # Split into train and a held-out pool for calibration + testing
    train_inputs, pool_inputs, train_labels, pool_labels = train_test_split(
        inputs, labels, test_size=0.5, random_state=42
    )

    # Train a model
    model = LogisticRegression(max_iter=10_000, random_state=42)
    model.fit(train_inputs, train_labels)

    # Use diagnostics to recommend a coverage level and calibration set size
    pool_probs = model.predict_proba(pool_inputs)
    config = DiagnosticConfig(rng=42)
    plan = classifier_calibration_plan(pool_probs, pool_labels, max_set_size=MAX_SET_SIZE, config=config)

    # Split the pool into calibration and test using the recommended size
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(pool_labels))
    cal_idx, test_idx = indices[: plan.cal_size], indices[plan.cal_size :]
    cal_inputs, cal_labels = pool_inputs[cal_idx], pool_labels[cal_idx]
    test_inputs, test_labels = pool_inputs[test_idx], pool_labels[test_idx]

    # Calibrate and wrap the model
    calibration = calibrate_classifier(model.predict_proba(cal_inputs), cal_labels)
    wrapper = ConformalClassifier(
        predict_fn=model.predict_proba,
        calibration=calibration,
    )

    # Get prediction sets at the recommended coverage
    prediction_sets = wrapper.predict(test_inputs, coverage=plan.coverage)

    # The coverage guarantee: the true class is included at least `plan.coverage` of the time
    correct_class_included = prediction_sets[np.arange(len(test_labels)), test_labels]
    empirical_coverage = correct_class_included.mean()
    assert empirical_coverage >= plan.coverage

    # And most predictions are singletons, as requested by max_set_size
    mean_set_size = prediction_sets.sum(axis=1).mean()
    assert mean_set_size <= MAX_SET_SIZE

    # P-values are valid probabilities
    p_values = wrapper.predict_p_values(test_inputs)
    assert np.all((p_values >= 0) & (p_values <= 1))
