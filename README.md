<!-- logo / banner placeholder -->

<div align="center">

# conformal

**Know when to trust your model's predictions.**

[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-orange?style=for-the-badge)](https://github.com/rudikershaw/conformal)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![NumPy only](https://img.shields.io/badge/deps-numpy_only-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![MIT License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

Conformal prediction is a technique for getting honest uncertainty estimates from trained ML models. `conformal` is a lightweight, framework-agnostic Python library that makes it easy to apply. Give it a held-out calibration set and it will wrap your model to produce prediction sets (for classifiers) or prediction intervals (for regressors) with real statistical guarantees. No changes to your model, no retraining, and only numpy as a dependency.

## Installation

<!-- TODO: update once published to PyPI -->

```bash
pip install conformal
```

## Quick start

```python
from conformal import calibrate_classifier, ConformalClassifier

# 1. Get your model's predicted probabilities on a held-out calibration set
cal_probs = model.predict_proba(X_cal)  # shape: (n_examples, n_classes)

# 2. Calibrate
calibration = calibrate_classifier(cal_probs, y_cal)

# 3. Wrap your model
wrapper = ConformalClassifier(
    predict_fn=model.predict_proba,
    calibration=calibration,
)

# 4. Predict with guaranteed coverage
prediction_sets = wrapper.predict(X_test, coverage=0.90)
# >>> array([[ True, False, False],
#            [ True,  True, False]])
```
Each row is a prediction set where `True` marks a plausible class. A single `True` means the model is confident. Multiple `True` values mean it is uncertain. The guarantee is that at least 90% of prediction sets will contain the correct class.

You can also get the underlying p-values for each class directly.

```python
p_values = wrapper.predict_p_values(X_test)
# >>> array([[0.92, 0.31, 0.04],
#            [0.87, 0.79, 0.11]])
```

Each value is the minimum coverage level at which that class would enter the prediction set. Higher means more plausible. This is useful when you want to rank classes by plausibility, or apply different coverage thresholds after the fact without re-running the model. The p-values themselves carry a guarantee: for the true class, the probability of its p-value falling below any threshold is at most that threshold.

## FAQ

**What is conformal prediction?**
A way to turn any model's outputs into prediction sets (for classifiers) or prediction intervals (for regressors) that come with a coverage guarantee: the true answer will be included at least X% of the time. It doesn't change the model. It just adds a calibration step after training.

**Does this work with my model?**
If your model can produce per-class scores (probabilities, logits, etc.) for classification or numeric predictions for regression, yes. It doesn't matter what framework you trained with.

**How much calibration data do I need?**
More calibration data means tighter prediction sets. A few hundred examples is a reasonable starting point, but even 50 can be useful. The library will raise an error if your calibration set is too small for your requested coverage level.

**What's the catch?**
The guarantee relies on one assumption: calibration data and test data are drawn from the same distribution. If the world changes after calibration (new user behaviour, different sensor, etc.), you should recalibrate.

**Is this really guaranteed? Where's the proof?**
Yes. The coverage guarantee is a proven theorem, not a heuristic. The original proof comes from Vovk, Gammerman, and Shafer in *Algorithmic Learning in a Random World* (2005). If you want something shorter, Shafer and Vovk also wrote *"A Tutorial on Conformal Prediction"* (2008) in the Journal of Machine Learning Research. Fair warning: neither is light reading.

## Documentation

<!-- TODO: add links once docs site is live -->

Documentation is coming soon.

