# Using conformal with a Classifier

The following code is an example of using conformal to calibrate a scikit-learn classifier (LogisticRegression) from the project's integration tests. You can find the code file in this project under [tests/integration/test_classification_workflow.py](tests/integration/test_classification_workflow.py). After reading through the example below, you can find a discussion of what each step achieves below the code example.

<!-- INSERT_CODE:tests/integration/test_classification_workflow.py -->
```py
"""Breast cancer classification with scikit-learn and conformal prediction."""
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

    # Split our data into training and a held-out reserve set for calibration + testing
    train_inputs, reserve_inputs, train_labels, reserve_labels = train_test_split(
        inputs, labels, test_size=0.5, random_state=42
    )

    # Train a model
    model = LogisticRegression(max_iter=10_000, random_state=42)
    model.fit(train_inputs, train_labels)

    # Use diagnostics to recommend a coverage level and calibration set size
    reserve_probs = model.predict_proba(reserve_inputs)
    config = DiagnosticConfig(rng=42)
    plan = classifier_calibration_plan(reserve_probs, reserve_labels, max_set_size=MAX_SET_SIZE, config=config)

    # Split the reserve set into calibration and test sets using the recommended size
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(reserve_labels))
    cal_idx, test_idx = indices[: plan.cal_size], indices[plan.cal_size :]
    cal_inputs, cal_labels = reserve_inputs[cal_idx], reserve_labels[cal_idx]
    test_inputs, test_labels = reserve_inputs[test_idx], reserve_labels[test_idx]

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
```
<!--CODE_END -->

First we pull in our dataset, split our data into training and a reserve sets, and then train our model on the training set. This is all achieved outside of the scope of the conformal library (in this case using scikit-learn).

Now, we need to choose the size of our calibration set. Typically calibration sets can be effective with a very small number of entries. Fewer than 100 examples are typically enough, depending on your coverage requirements. Rather than picking the size of our calibration set and coverage manually through trial and error, this examples uses the `classifier_calibration_plan` function to find the ideal values for us, given a maximum average set size. The maximum average set size tells us how many classes will be predicted (True) on average. Informally, you can think of a maximum set size of 1.05 (for example) as telling us that a classifier with 2 classes will predict both classes in only 1 in every 20 predictions on average. The `classifier_calibration_plan` will provide us with the highest possible coverage for our specified maximum set size, and tell us the smallest calibration set size that will be required to achieve these results.

Once we have a `plan` we use the results to break our reserve set into testing and calibration sets.

Next, we calibrate the model using the calibration set. The `calibrate_classifier` function takes the predicted probabilities and true labels from the calibration set and computes the necessary calibration data. This calibration data is then used to create a `ConformalClassifier` wrapper around our original model's `predict_proba` function.

With the calibrated wrapper, we can now make conformal predictions. The `predict` method returns prediction sets at the specified coverage level. Each prediction set is a boolean array indicating which classes are included in the prediction for that example. The coverage guarantee ensures that the true class is included in the prediction set at least as often as the specified coverage level. This means that the new conformal predictions may include multiple true values when the model is uncertain (which is how it maintains the coverage guarantee).

In the case where a prediction set contains multiple true values, you may still need to select a single class for your use-case. To enable this, the wrapper also provides a `predict_p_values` function, which will produce the p-values used to produce the conformal predictions. You may use these values to determine which single class the model is most sure of, and also to compare the model's confidence between classes.