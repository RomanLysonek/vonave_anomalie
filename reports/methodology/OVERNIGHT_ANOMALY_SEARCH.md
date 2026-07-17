# Overnight anomaly search methodology

## Purpose

The earlier 40-epoch autoencoder run was a diagnostic, not a serious model-selection experiment. It trained one architecture, one representation, one temporal split and one threshold policy. Its negative result says only that **that configuration** was not stable enough to promote.

The overnight suite tests the broader hypothesis:

> Can causal anomaly state, learned systemic reconstruction state, or conservative loss attenuation improve the existing direct NeuralNet forecast under walk-forward validation?

The control is always the unchanged forecasting model with `anomaly_mode=off`. An anomaly candidate is useful only if it improves the forecast; attractive reconstruction plots are not sufficient.

## One-command run

From the repository root:

```bash
uv sync
uv run pytest -q \
  tests/test_anomaly_detection.py \
  tests/test_anomaly_pipeline_integration.py \
  tests/test_systemic_autoencoder_v2.py \
  tests/test_overnight_anomaly_search.py

scripts/run_overnight_anomaly_search.sh
```

The launcher:

- verifies that PyTorch MPS is available;
- prevents macOS sleep with `caffeinate`;
- enables MPS CPU fallback for unsupported operators;
- uses atomic result files and fold checkpoints;
- resumes completed work automatically;
- writes the complete console stream to `outputs/overnight_anomaly_search/search.log`.

Check progress from another terminal:

```bash
scripts/anomaly_search_status.sh
```

Resume after interruption or reboot:

```bash
scripts/resume_anomaly_search.sh
```

## Search profiles

| Profile | AE diagnostics | Proxy origins | NN screening | Confirmation |
|---|---:|---:|---:|---:|
| `smoke` | 2 candidates, one cutoff/seed, 3 epoch cap | 1 dev + 1 benchmark | 1 seed, 2 epochs | 1 seed, 2 epochs |
| `overnight` | 36 candidates, 3 cutoffs × 2 seeds, 180 epoch cap | 6 dev + 8 benchmark | top 5, 1 seed, 25 epochs | top 2, 12 dev + 12 benchmark, 3 seeds, 45 epochs |
| `weekend` | 72 candidates, 4 cutoffs × 3 seeds, 240 epoch cap | 9 dev + 16 benchmark | top 8, 2 seeds, 35 epochs | top 3, 12 dev + 24 benchmark, 3 seeds, 60 epochs |
| `exhaustive` | 120 candidates, 5 cutoffs × 3 seeds, 320 epoch cap | all dev + 24 benchmark | top 12, 2 seeds, 45 epochs | top 4, all dev + 52 benchmark, 4 seeds, 75 epochs |

Run the wider profile with:

```bash
scripts/run_weekend_anomaly_search.sh
```

Or select explicitly:

```bash
PROFILE=exhaustive \
OUTPUT_DIR=outputs/exhaustive_anomaly_search \
  scripts/run_overnight_anomaly_search.sh
```

A graceful trial-boundary time budget is optional:

```bash
scripts/run_overnight_anomaly_search.sh --max-hours 12
```

When the budget is reached, the current subprocess finishes and the orchestrator stops before starting another trial. The same command resumes from the next missing result.

## Stage 1: autoencoder diagnostic search

The generated search covers:

### Representations

- `log_level`: standardized `log1p` quantity levels;
- `weekday_residual`: residuals against a causal weighted `t-7/t-14/t-21/t-28` expectation;
- `level_residual`: joint level and residual channels;
- `residual_availability`: residual plus product availability state;
- `level_residual_availability`: all three channels.

The residual representations specifically address the failure observed in the quick experiment: the autoencoder should not spend most of its capacity rediscovering long-term growth and level shifts.

### Architectures

- flattened denoising MLP autoencoder;
- temporal `Conv1d` autoencoder;
- GRU sequence autoencoder.

### Hyperparameters

The randomized but deterministic search varies:

- window length: 14, 28, 42, 56 or 84 days;
- hidden and latent dimensions;
- dropout and input noise;
- MSE versus Huber reconstruction loss;
- batch size, learning rate and weight decay;
- trailing training history: 365, 730, 1,095, 1,460 days or all history;
- calibration and holdout spans;
- EVT versus empirical thresholds;
- mean, 95th-percentile or hybrid reconstruction scoring;
- alert rate, attenuation strength, minimum weight and event-protection floor.

### Temporal validity

Every run uses disjoint chronological partitions:

```text
ignored older history | train | temporal validation | calibration | holdout
```

- imputation and scaling are fitted on training windows only;
- early stopping uses only temporal validation;
- thresholds and percentiles use only calibration;
- stability and predictive diagnostics use holdout;
- no future sales are available to the score at the forecast origin.

Candidates are not ranked by reconstruction loss alone. The diagnostic objective rewards:

- score stability across seeds;
- calibrated holdout alert rates;
- low monotonic time drift;
- an origin score that predicts next-week seasonal-baseline difficulty;
- concentration of difficult weeks in the highest score decile.

This stage is a filter, not proof of forecast improvement.

## Stage 2: exact direct-panel proxy

The strongest autoencoders are converted into two actions:

- `features`: continuous origin-known reconstruction state only;
- `both`: features plus bounded historical target attenuation.

They compete with a randomized statistical-anomaly search and then with statistical/autoencoder hybrids. Evaluation uses the exact production direct panel, target definition, availability filter, feature schema and sample-weight contract. DynamicRidge is used only as a cheaper ranking instrument.

Each fold trains its autoencoder on that fold's history. Fold/config profiles are cached under:

```text
outputs/overnight_anomaly_search/autoencoder_cache/
```

The cache key includes the fold end date, data extent and complete autoencoder configuration. A later fold or different configuration cannot silently reuse stale scores.

## Stage 3: MPS NeuralNet screening

The top proxy candidates are rerun with the actual submission NeuralNet on Apple GPU. This stage rejects proxy-specific improvements and measures the interaction with the nonlinear residual model.

Candidates and folds run in isolated Python processes. This prevents long MPS searches from accumulating model graphs and DataFrames in one interpreter.

## Stage 4: multi-seed confirmation

The final candidates receive wider development and recent-benchmark origins plus multiple NeuralNet seeds. Promotion requires all of the following:

- development WAPE improvement of at least 0.2% relative;
- recent-benchmark WAPE regression no worse than 2%;
- development top-demand-decile WAPE regression no worse than 3%;
- holiday/event WAPE regression no worse than 5%;
- at least 75% origin-bootstrap probability that the development improvement is positive.

If no candidate passes every gate, `control` wins. This is an intended outcome, not a failed experiment.

## Outputs

```text
outputs/overnight_anomaly_search/
├── preflight.json
├── manifest.json
├── candidates/
├── diagnostic/<candidate-id>/
├── proxy/<candidate-id>/
├── neural/<candidate-id>/
├── confirmation/<candidate-id>/
├── autoencoder_cache/
├── diagnostic_leaderboard.csv
├── proxy_leaderboard.csv
├── neural_leaderboard.csv
├── confirmation_leaderboard.csv
├── recommendation.json
├── winner_candidate.json
└── FINAL_REPORT.md
```

Every forecast trial directory contains:

- `result.json` or `failure.json`;
- `development_oof.parquet`;
- `benchmark_oof.parquet`;
- per-fold checkpoints;
- `trial.log`.

## Producing the final forecast

After confirmation, `recommendation.json` contains the exact final command. The winner can also be applied directly:

```bash
caffeinate -dimsu uv run python ml/pipeline.py \
  --forecast-strategy direct \
  --primary-strategy direct \
  --submission-model NeuralNet \
  --selection-metric WAPE \
  --selection-protocol test-aligned \
  --training-window-days all \
  --recency-half-life-days none \
  --baseline-variant weighted_4321 \
  --trend-features off \
  --c2-feature-groups price,campaign,lifecycle,market,event \
  --nn-loss mse \
  --nn-target-mode residual \
  --anomaly-config outputs/overnight_anomaly_search/winner_candidate.json \
  --nn-batch-size auto \
  --nn-training-backend auto \
  --resume
```

`--anomaly-config` accepts the candidate JSON and reproduces every statistical and autoencoder setting. Explicit anomaly CLI switches override values from the file.

## Leakage contract

- The evaluation origin is never included in autoencoder training beyond observations genuinely known by that origin.
- Autoencoder normalization is train-only and threshold calibration is chronologically later but still pre-evaluation.
- Reconstruction features are joined only to the observed origin date.
- Target-date reconstruction values may alter historical training loss only; they never enter target predictors.
- The actual forecast week cannot have a demand reconstruction score because its demand is unknown.
- Development origins drive tuning; recent origins are a guard; the frozen final audit remains outside this search.
