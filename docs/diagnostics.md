# Diagnostics API

The `conforma.diagnostics` module provides tools for understanding how calibration set size affects conformal prediction quality. It includes shared configuration and both classifier and regressor diagnostics.

## Key exports

- `DiagnosticConfig`
  - Controls the diagnostic sweep.
  - Parameters:
    - `sizes`: calibration set sizes to evaluate, or `None` to infer a default range.
    - `n_repetitions`: number of random subsamples per size. Defaults to `100`.
    - `rng`: random number generator or seed for reproducible results.

- `classifier_coverage_stability`
  - Measures how calibration set size affects classifier empirical coverage and prediction set size.
  - Returns `ClassifierCoverageStability` with:
    - `sizes`
    - `mean_coverage`
    - `std_coverage`
    - `mean_set_size`
    - `std_set_size`

- `classifier_calibration_plan`
  - Searches for the highest coverage that keeps mean prediction set size below `max_set_size`.
  - Returns `ClassifierCalibrationPlan` with:
    - recommended `coverage`
    - recommended `cal_size`
    - full search arrays for `mean_coverage`, `std_coverage`, `cal_sizes`, `mean_set_size`, and `std_set_size`

- `regressor_coverage_stability`
  - Measures how calibration set size affects regressor empirical coverage and interval width.
  - Returns `RegressorCoverageStability` with:
    - `sizes`
    - `mean_coverage`
    - `std_coverage`
    - `mean_interval_width`
    - `std_interval_width`

- `regressor_calibration_plan`
  - Searches for the highest coverage that keeps the mean prediction interval width below `max_interval_width`.
  - Returns `RegressorCalibrationPlan` with:
    - recommended `coverage`
    - recommended `cal_size`
    - full search arrays for `mean_coverage`, `std_coverage`, `cal_sizes`, `mean_interval_width`, and `std_interval_width`

## How it works

Both stability and calibration-plan diagnostics perform repeated subsampling of the provided calibration data. For each sampled calibration size they train a conformal calibration function and evaluate:

- empirical coverage on held-out test examples
- set size for classifiers / interval width for regressors

The calibration-plan helpers then choose the best empirical coverage that satisfies the user's budget constraint and return the smallest reliable calibration set size.

## Typical usage

1. Build a `DiagnosticConfig` for the sizes, repetitions, and RNG.
2. Use `classifier_coverage_stability` or `regressor_coverage_stability` to inspect how diagnostics change with calibration size.
3. Use `classifier_calibration_plan` or `regressor_calibration_plan` to get a recommended coverage and calibration size for production use.

## Notes

- For regressors, `max_interval_width` is expressed in the target variable's units.
- For classifiers, `max_set_size` is the maximum acceptable mean prediction set size.
- The returned plan objects include the full search results for validation and visualization.
