# DAVID → `vonave_anomalie`: transfer audit and implementation rationale

## Executive decision

The sensible use of DAVID in this forecasting task is **not** to replace the forecaster with an anomaly-detection autoencoder. The useful transfer is a statistically calibrated, leakage-safe anomaly layer around the existing supervised pipeline:

1. derive causal demand surprise scores;
2. calibrate rare-tail thresholds with POT/GPD EVT;
3. summarize recent anomaly state at each forecast origin;
4. use anomaly severity as a bounded robust-training weight;
5. quantify forecast-time covariate shift from known test-week conditions;
6. retain a small multivariate autoencoder as a retrospective systemic diagnostic;
7. accept the extension only through predeclared walk-forward gates.

This gives an interview-grade story because it demonstrates architectural reuse, statistical restraint, leakage control, and negative-result tolerance.

## Sources studied

### DAVID framework

Relevant source concepts:

- `src/david/core/evaluation/evt.py`: GPD fitting, POT thresholds, false-alarm calibration, KS-based tail-fit checks.
- `src/david/metrics/models/autoencoder/`: multivariate metric-window autoencoding.
- `src/david/logs/models/autoencoder/`: reconstruction-based anomaly scoring for sequential data.
- DAVID inference modules: separation of learned score generation from threshold/action policy.

Irrelevant for this use case:

- Drain log parsing and template mining;
- sentence embedding of log messages;
- log-specific masking and sequence-token semantics;
- Airflow/provider abstractions and cloud deployment operators.

### DBAAS use case

The DBAAS project demonstrates the operational pattern around DAVID:

- preprocessing;
- autoencoder pretraining/fine-tuning;
- parameter search;
- validation-score thresholding;
- inference;
- filter/select action.

The transferable lesson is the staged contract, not the Oracle-specific implementation. In `vonave_anomalie`, the equivalent stages are score → calibrate → diagnose → ablate → accept/reject.

### Forecasting repository

The current `vonava_predikce` project already provides the mechanisms needed for a clean integration:

- leakage-safe walk-forward origins;
- direct `(ForecastOrigin × Horizon × ProductId)` panels;
- target-relative observed-history lookup rules;
- common-population WAPE evaluation;
- development, recent benchmark, and frozen final-audit separation;
- sample-weight support across the competitive supervised estimators;
- persisted diagnostics and strict checkpoint signatures.

That maturity makes a narrow extension preferable to a framework transplant.

## Baseline evidence that shaped the design

Checked-in direct-strategy metrics:

| Slice | NeuralNet WAPE | Ensemble WAPE | Interpretation |
|---|---:|---:|---|
| Development, conditional/common/global | 0.299796 | 0.267926 | ensemble has substantial average advantage |
| Recent benchmark, conditional/common/global | 0.273459 | 0.247078 | ensemble also wins recent average benchmark |
| Frozen final audit, conditional/common/global | 0.301079 | 0.299304 | ensemble advantage shrinks to 0.001775 absolute |
| Frozen final audit, test-aligned | **0.278328** | 0.279737 | NeuralNet remains canonical under the selected objective |
| Development top demand decile | 0.232008 | 0.206890 | high-demand days remain a distinct error regime |

The development top-demand-decile NeuralNet bias ratio is `-0.135327`, indicating material underprediction. Among its 100 largest development absolute errors:

- 89 occur in `holiday_event` strata;
- 59% are underpredictions.

Therefore an indiscriminate outlier filter would likely erase valuable examples of exactly the demand peaks the model needs to learn. This is the core reason for the event-protection and systemic-shock floors.

## Implemented detector 1: causal product-local surprise

For product `p` and day `t`, the expectation uses only observed available demand from prior same-weekday lags:

`b[p,t] = weighted_mean(y[p,t-7], y[p,t-14], y[p,t-21], y[p,t-28])`

with weights `4,3,2,1`. Missing values fall back to past-only rolling medians. The signed surprise is:

`r[p,t] = log1p(y[p,t]) - log1p(b[p,t])`.

Severity is the absolute deviation from a product-specific, past-only rolling median, divided by a robust rolling scale derived from the IQR. The score at day `t` cannot see the residual at any later day.

Why this is preferable to a default neural autoencoder here:

- only ~30 products and roughly two years of history are available;
- same-weekday structure is a strong, interpretable prior;
- the score is traceable to a concrete expected quantity;
- it remains usable inside every CV fold without expensive retraining;
- it provides substantially more observations for tail calibration than one score per day.

## Implemented detector 2: systemic cross-product shock

For each day, the systemic score is the 90th percentile of product-local severities. A quantile rather than a maximum prevents a single SKU from declaring a market-wide event, while still reacting when several products move unusually.

Systemic flags have their own EVT calibration and generate origin-state features:

- current systemic severity;
- current systemic flag;
- previous-28-day systemic anomaly rate.

Systemic rows retain at least the configured minimum weight because broad shocks may be regime changes, not bad labels.

## EVT/POT calibration

The implementation adapts DAVID's POT/GPD approach to the smaller retail score sample:

1. preserve score order and use a temporal tail holdout;
2. sweep candidate body/tail split quantiles;
3. fit a GPD to exceedances with location fixed at zero;
4. derive the threshold for target false-alarm probability `alpha`;
5. evaluate validation false-alarm error plus a KS fit penalty;
6. reject unstable, non-finite, or implausibly extrapolated fits;
7. use a deterministic empirical quantile when data are insufficient.

The fallback is a feature, not a failure. An EVT fit should not be forced when there are too few stable exceedances.

## Action policy: bounded robust weighting

An anomaly is not a statement that a row is wrong. It is a hypothesis that the row may deserve less influence.

For scores above the EVT threshold, a monotonically decaying weight is applied and clipped to `[min_weight, 1]`. It is then multiplied by any existing recency/sample weight and normalized to mean one.

Protection rules:

- positive surprises under known campaign, discount, or sale context are lifted to `known_event_min_weight`;
- all rows on a systemic anomaly day are lifted to `systemic_min_weight`;
- unavailable rows are not modified by this demand-weight layer;
- no row is dropped.

This policy is intentionally conservative because the existing error audit says peak demand is already underpredicted.

## Feature policy: origin-known state only

The following direct-panel features may be enabled:

- `anomaly_score_lag0`
- `anomaly_flag_lag0`
- `anomaly_rate_28`
- `days_since_anomaly`
- `systemic_anomaly_score_lag0`
- `systemic_anomaly_flag_lag0`
- `systemic_anomaly_rate_28`

They are joined to the last observed day at the forecast origin and replicated across the seven horizons. Target-date anomaly score and flag columns may exist as diagnostics in the training panel, but are absent from `direct_panel_feature_names` and therefore cannot enter a model.

## Forecast-time context risk

Future demand anomaly scores cannot be known before future demand exists. A second detector therefore scores only future-known context:

- product;
- price;
- web/app discount;
- web/app campaign subtype;
- promotion flag;
- availability;
- calendar cycles.

An Isolation Forest score is augmented with robust numeric rarity and categorical-frequency surprise, then calibrated to an empirical percentile against training rows.

This is a **forecast confidence diagnostic**, not a demand-anomaly prediction. A high percentile means the forecast is being made under sparse or novel covariate support and should be presented with caution.

## Optional multivariate autoencoder

The closest literal DAVID analogue is retained as `ml/systemic_autoencoder.py`:

- daily matrix of log-transformed demand across products;
- 28-day flattened windows;
- train-only normalization;
- small denoising MLP autoencoder;
- window reconstruction MSE;
- EVT threshold fitted on a later calibration block;
- untouched temporal holdout scores.

It is deliberately diagnostic rather than a default predictor because:

- it yields only one score per day/window, so the tail sample is limited;
- reconstruction error for the future week is unavailable at forecast time;
- a large LSTM/VAE would add variance and presentation burden without a justified data regime;
- systemic events are often business-relevant signal rather than contamination.

A useful analysis is to compare autoencoder flags with the forecast's top-error dates and known holiday/campaign events.

## Validation protocol

### Fast screen

`ml/run_anomaly_screening.py` evaluates six predeclared candidates with LightGBM on the same direct panel and sample-weight contract:

- control;
- soft/default weighting;
- features only;
- soft/default combined mode.

LightGBM is only a computational screen. It tests whether the transformed data contract contains incremental predictive value.

### Acceptance gate

A non-control candidate must:

- improve development conditional/common/global WAPE by at least 0.2% relative;
- keep recent-benchmark relative WAPE regression at or below 2%;
- then survive a full NeuralNet confirmation under the frozen selected C1–C4 configuration.

The final audit remains untouched until a candidate is selected. It must not become another tuning set.

### Interpretation of outcomes

- **Weighting wins:** likely label contamination/idiosyncratic spikes are harming estimation.
- **Features win:** instability has predictive persistence or regime meaning.
- **Both win:** robust fitting and regime state are complementary.
- **Control wins:** the anomaly layer is useful diagnostically but does not improve the forecast. This is still a rigorous result.

## Leakage audit

| Risk | Guard |
|---|---|
| Future residuals influence current score | rolling center/scale are shifted; same-weekday baseline uses positive lags only |
| CV threshold sees validation block | each fold builds the profile from raw rows strictly before its origin |
| Target anomaly becomes feature | target diagnostics are excluded from the feature schema |
| Test distribution tunes threshold | test demand is never used; test context risk is scored against training support only |
| Candidate checkpoints are mixed | anomaly configuration is part of the exact checkpoint signature |
| Final audit is repeatedly tuned | anomaly selection uses development + recent benchmark only |
| Event demand is erased | known-positive event and systemic minimum-weight floors |

## What should be claimed in the interview

Safe claims now:

- the architecture and tests are implemented;
- the original forecast is reproducible with anomaly mode off;
- the transfer is leakage-safe by construction and explicitly ablatable;
- the baseline diagnostics justify conservative handling of event anomalies;
- forecast-time context shift is separated from retrospective demand surprise.

Claims that require local execution first:

- the number and identity of actual anomalies;
- whether the test week is context-shifted;
- whether any anomaly candidate improves WAPE;
- whether the optional autoencoder flags align with model failures.

## Recommended execution order

1. `uv run pytest -q`
2. `uv run python ml/run_anomaly_audit.py`
3. inspect top anomaly rows against campaigns, availability, and top forecast errors;
4. `uv run python ml/run_anomaly_screening.py --profile screen`
5. run the emitted full NeuralNet confirmation command only if a candidate passes;
6. run the optional systemic autoencoder as an explanatory diagnostic;
7. report the result, including a control win if that is what the evidence says.
