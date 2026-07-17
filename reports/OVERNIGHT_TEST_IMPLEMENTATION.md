# Overnight anomaly experiment implementation

> Historical overnight/diagnostic/search outputs are scientifically
> contaminated and provenance-unverified because benchmark targets may have
> entered diagnostic inputs or next-week targets. They are audit-only and
> excluded from current evidence. No rerun was performed.

## Correction to the earlier test

The quick sandbox run tested one compact autoencoder and one lightweight anomaly-weighting policy. It was useful as a falsification check but too narrow to answer whether a carefully tuned anomaly representation can help the final NeuralNet.

The new implementation prepares that larger experiment without executing it in the sandbox.

## Implemented components

### Experiment-grade systemic autoencoder

`ml/systemic_autoencoder_v2.py` adds:

- chronological train/validation/calibration/holdout partitions;
- train-only median imputation and standardization;
- trailing training-window control;
- MLP, temporal convolution and GRU autoencoders;
- five input representations, including causal weekday residuals;
- MSE or Huber reconstruction loss;
- denoising noise, dropout, AdamW, cosine learning-rate decay and gradient clipping;
- temporal early stopping with best-checkpoint restoration;
- EVT or empirical thresholds;
- continuous calibration percentiles;
- bounded target weights with commercial-event protection;
- continuous origin features and fold/config caching.

### Forecast-pipeline integration

The direct pipeline now supports:

```text
anomaly_source = statistical | autoencoder | hybrid
```

Autoencoder origin state is available only when the origin day's sales are already observed. Autoencoder target-date state can affect historical training weights but is not a predictor. Hybrid weights are multiplied and normalized once.

### Search orchestration

`ml/run_overnight_anomaly_search.py` provides:

1. broad autoencoder diagnostic search;
2. exact direct-panel DynamicRidge screening;
3. MPS NeuralNet promotion;
4. multi-seed, wider-origin confirmation;
5. bootstrap uncertainty and explicit promotion gates;
6. atomic outputs, candidate subprocess isolation, checkpoints and resume;
7. a final winner JSON consumable by `ml/pipeline.py --anomaly-config`.

### Mac launch and monitoring

- `scripts/run_overnight_anomaly_search.sh`
- `scripts/run_weekend_anomaly_search.sh`
- `scripts/resume_anomaly_search.sh`
- `scripts/anomaly_search_status.sh`

The launcher verifies MPS, enables fallback for unsupported operations and prevents macOS sleep.

## Verification performed here

- all modified Python modules compile;
- search orchestration dry-runs through every stage;
- candidate generation is deterministic;
- winner JSON is loadable through the normal pipeline CLI;
- chronological V2 autoencoder training passes a synthetic CPU test;
- date-key feature and target-weight alignment tests pass;
- autoencoder/statistical/hybrid feature schemas are source-aware;
- existing anomaly, pipeline, checkpoint and compact-autoencoder tests remain green.

The actual overnight search was intentionally not run here. Its purpose is to use the user's Apple GPU and tolerate the long exact-panel evaluations that the sandbox could not sustain.
