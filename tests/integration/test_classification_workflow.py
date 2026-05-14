"""Integration test: breast cancer classification with conformal prediction.

Uses scikit-learn's Breast Cancer Wisconsin dataset and a logistic regression
model to exercise the full conformal classification workflow end-to-end.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from conformal import ConformalClassifier, calibrate_classifier
from conformal.diagnostics import DiagnosticConfig, classifier_coverage_stability

COVERAGE = 0.92


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

    # Use diagnostics to find the minimum calibration set size
    pool_probs = model.predict_proba(pool_inputs)
    config = DiagnosticConfig(rng=42)
    result = classifier_coverage_stability(pool_probs, pool_labels, coverage=COVERAGE, config=config)

    # Pick the smallest size where mean coverage meets the target
    viable = result.sizes[result.mean_coverage >= COVERAGE]
    cal_size = int(viable[0])

    # Split the pool into calibration and test using the recommended size
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(pool_labels))
    cal_idx, test_idx = indices[:cal_size], indices[cal_size:]
    cal_inputs, cal_labels = pool_inputs[cal_idx], pool_labels[cal_idx]
    test_inputs, test_labels = pool_inputs[test_idx], pool_labels[test_idx]

    # Calibrate and wrap the model
    calibration = calibrate_classifier(model.predict_proba(cal_inputs), cal_labels)
    wrapper = ConformalClassifier(
        predict_fn=model.predict_proba,
        calibration=calibration,
    )

    # Get prediction sets at 90% coverage
    prediction_sets = wrapper.predict(test_inputs, coverage=COVERAGE)

    # The coverage guarantee: the true class is included at least 92% of the time
    correct_class_included = prediction_sets[np.arange(len(test_labels)), test_labels]
    empirical_coverage = correct_class_included.mean()
    assert empirical_coverage >= COVERAGE

    # P-values are valid probabilities
    p_values = wrapper.predict_p_values(test_inputs)
    assert np.all((p_values >= 0) & (p_values <= 1))
