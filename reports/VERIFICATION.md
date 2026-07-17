# Verification record

> Scientific status: no anomaly truth labels exist; known events are
> explanatory proxies, not labels. The compact autoencoder evidence below is
> legacy and excluded because its preprocessing used future-derived medians.
> Fingerprinted V2 is canonical, no verified V2 artifact is published, and the
> current recommendation is the control with `anomaly_mode=off`. record

Date: 2026-07-15

## Completed checks

### Syntax/import surface

```bash
python -m py_compile \
  ml/anomaly_detection.py \
  ml/context_risk.py \
  ml/systemic_autoencoder.py \
  ml/run_anomaly_audit.py \
  ml/run_anomaly_screening.py \
  ml/run_systemic_autoencoder.py \
  ml/framework.py \
  ml/pipeline.py
```

Result: passed.

### New anomaly and checkpoint tests

```bash
python -m pytest -q \
  tests/test_anomaly_detection.py \
  tests/test_anomaly_pipeline_integration.py \
  tests/test_context_risk.py \
  tests/test_systemic_autoencoder.py \
  tests/test_fold_checkpoints.py
```

Result: `10 passed`.

### Core forecasting regression tests

```bash
python -m pytest -q \
  tests/test_pipeline.py \
  tests/test_direct_recursive_strategies.py \
  tests/test_fold_checkpoints.py
```

Result: `36 passed`.

### Ensemble/dashboard/static regression tests

```bash
python -m pytest -q \
  tests/test_c5_ensemble.py \
  tests/test_c6_dashboard.py \
  tests/test_webapp_strategy_sync.py
```

Result: `20 passed`.

### JavaScript dashboard smoke checks

```bash
node tests/webapp_smoke_test.js
```

Result: `8 JavaScript smoke checks passed`.

The groups above represent 64 distinct passing Python tests plus 8 JavaScript checks; `test_fold_checkpoints.py` appears in two commands.

## Environment limitations

- The execution container has Python 3.13 and no installed general-purpose Parquet engine, while the locked project runtime targets Python 3.14 with `pyarrow>=25.0.0`.
- A minimal fallback reader (`ml/offline_parquet.py`) was therefore added for the simple primitive/Snappy/data-page-v1 structure of the two bundled Parquet files. It was verified against the expected shapes and date ranges: train `(51,534, 11)`, test `(210, 8)`, training dates 2021-01-01 through 2026-01-11, and test dates 2026-01-12 through 2026-01-18.
- The exact direct LightGBM screening exceeded the per-fold execution window, and the exact direct-panel Ridge feature ablation exceeded the available memory/time envelope. These runs produced no valid comparison and are not reported as results.
- A lower-memory causal weighting ablation completed on four development and four recent-benchmark origins. It selected the control; both anomaly-weight policies slightly worsened aggregate WAPE.

## Actual-data execution added on 2026-07-15

```bash
PYTHONPATH=ml python ml/run_anomaly_audit.py \
  --output-dir outputs/anomaly_audit_real \
  --alpha 0.01

PYTHONPATH=ml OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python ml/run_systemic_autoencoder.py \
  --output-dir outputs/anomaly_autoencoder_real \
  --epochs 40

PYTHONPATH=ml python ml/run_lightweight_anomaly_ablation.py
```

Results:

- 485 local anomalies among 49,595 scored product-days;
- 35 systemic anomaly days;
- 0 of 210 test-context rows above the 99th-percentile shift threshold after fixing the feature contract;
- original-split compact autoencoder: 1,176 training, 362 calibration and 272 holdout windows, 64 flags, but a 33.3% validation exceedance rate against a 2% target; this legacy run is excluded because preprocessing used future-derived medians;
- recent-split autoencoder: 1,448 training, 181 calibration and 181 holdout windows, one calibration flag and zero holdout flags;
- anomaly weighting: no development gain and slight benchmark regression; selected policy `control`.

The full result and evidence boundary are documented in `reports/REAL_DATA_TEST_RESULTS.md`. No actual-data forecasting improvement is claimed.

## Post-fix targeted regression check

```bash
PYTHONPATH=ml pytest -q \
  tests/test_context_risk.py \
  tests/test_anomaly_detection.py \
  tests/test_systemic_autoencoder.py \
  tests/test_anomaly_pipeline_integration.py
```

Result: `8 passed in 10.33s`.
