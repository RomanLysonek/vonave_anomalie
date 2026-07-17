# Implementation plan and completion state

## Phase 0 — preserve the control

- [x] Fork the current forecasting project into `vonave_anomalie`.
- [x] Keep `anomaly_mode="off"` as the default.
- [x] Preserve the original project documentation in `FORECAST_BASELINE.md`.
- [x] Ensure anomaly configuration participates in checkpoint identity.

## Phase 1 — causal anomaly score

- [x] Build a same-weekday weighted expectation from lags 7/14/21/28.
- [x] Respect availability censoring.
- [x] Calculate log-residual demand surprise.
- [x] Standardize against past-only product-local rolling median/IQR.
- [x] Add cross-product systemic severity.
- [x] Add causal-state regression tests.

## Phase 2 — DAVID-style statistical calibration

- [x] Implement POT/GPD upper-tail calibration.
- [x] Select candidate tail cutoffs by temporal validation exceedance-rate error and KS fit.
- [x] Add deterministic empirical fallback for small/unstable samples.
- [x] Persist calibration metadata.

## Phase 3 — forecasting actions

- [x] Implement bounded anomaly-derived sample weights.
- [x] Protect positive campaign/promotion spikes.
- [x] Protect systemic regime changes.
- [x] Normalize combined sample weights.
- [x] Add origin-known anomaly state features.
- [x] Exclude target anomaly diagnostics from model features.

## Phase 4 — pipeline integration

- [x] Fit anomaly profiles separately inside each direct CV fold.
- [x] Integrate features and/or weights through explicit modes.
- [x] Integrate the same contract into final direct training.
- [x] Expose runtime CLI controls.
- [x] Keep recursive forecasting unchanged pending evidence.

## Phase 5 — forecast-time confidence

- [x] Implement known-context novelty scoring.
- [x] Use training empirical percentiles for calibration.
- [x] Export row-level and day-level test-week risk.

## Phase 6 — literal DAVID analogue

- [x] Implement an optional multivariate 28-day demand-window autoencoder.
- [x] Fit normalization on the training block only.
- [x] Calibrate reconstruction error on a later block.
- [x] Preserve a temporal holdout.
- [x] Keep the component diagnostic rather than default.

## Phase 7 — experimental governance

- [x] Define a control plus five anomaly candidates.
- [x] Add a cheap LightGBM screen on the exact direct-panel contract.
- [x] Require development improvement and benchmark guard.
- [x] Emit the exact full NeuralNet confirmation command.
- [ ] Run actual-data anomaly audit in Roman's project environment.
- [ ] Run the screening grid.
- [ ] Run full NN confirmation for a passing candidate, if any.
- [ ] Compare optional autoencoder flags with top forecast-error dates.
- [ ] Freeze and present the measured conclusion.

## Acceptance criteria

The anomaly extension is promoted only when all conditions hold:

1. no leakage test failure;
2. no regression in original pipeline tests;
3. at least 0.2% relative development WAPE improvement;
4. no more than 2% relative recent-benchmark WAPE regression;
5. confirmation using the final NeuralNet configuration;
6. no evidence that improvement comes from suppressing explainable peak demand;
7. frozen final audit is evaluated once after candidate selection.
