# Weekend-v2 implementation record

> Historical validation and candidate-count claims below are contaminated/
> unverified because their overnight inputs may contain benchmark-target
> contamination. They are retained for implementation history only and are
> excluded from current candidate/selection evidence. Weekend-v2 was not rerun;
> control with `anomaly_mode=off` remains recommended.

## Added

- `ml/weekend_v2_common.py`
  - candidate generation;
  - vectorized convex search;
  - horizon- and product-shrunk mixtures;
  - aggregate reconciliation;
  - leave-one-origin-out meta models;
  - constrained specialist-advantage gates;
  - origin-block bootstrap.
- `ml/run_weekend_v2_search.py`
  - NeuralNet-only screen/refine/confirmation funnel;
  - marginal-ensemble candidate promotion;
  - resumable subprocess isolation;
  - final recommendation and report generation.
- `ml/run_weekend_v2_final.py`
  - full-history training of only active winner members;
  - resumable member predictions;
  - final blend/gate application;
  - submission export.
- macOS launch, resume, status, and smoke scripts.
- weekend-v2 unit tests and documentation.

## Modified

- `ml/anomaly_detection.py`
  - `downweight`, `negative_only`, `hard_example`, and `signed` policies;
  - bounded upweighting;
  - policy metadata;
  - overflow-safe exponentiation.
- `ml/framework.py`
  - `anomaly_weight_policy` and `anomaly_max_weight` configuration.
- `ml/pipeline.py`
  - persistence of origin-known anomaly/autoencoder state in OOF;
  - final-forecast diagnostics for the same state.
- `ml/anomaly_search_common.py`
  - invalid `min_history > rolling_window` combinations prevented.
- `ml/offline_parquet.py`
  - byte-array/string support and adaptive timestamp units for artifact forensics in environments without PyArrow.

## Validation performed

- all modified Python files compile;
- all shell scripts pass `bash -n`;
- 20 targeted anomaly/search/integration tests pass;
- 141 non-integration repository tests pass, with one integration-marked test intentionally excluded in this constrained environment;
- seven dedicated weekend-v2 tests are included in those totals;
- smoke dry-run generates the expected candidate manifest;
- candidate generation from the uploaded overnight artifacts produces:
  - 45 default candidates;
  - 79 exhaustive candidates;
- actual overnight OOF was re-read and used to validate:
  - global, horizon, and product convex plans;
  - greedy diversity selection;
  - constrained statistical specialist gating;
  - vectorized origin bootstrap.

## Evidence boundary

Weekend-v2 itself has not been run. The included preflight metrics are derived only from the already-completed overnight confirmation OOF and are used to justify the new search design. They are not presented as the final weekend-v2 result.
