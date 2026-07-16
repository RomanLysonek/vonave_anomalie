# Real-data test results

## Decision

**Do not promote anomaly weighting or the current autoencoder into the final forecasting model.**

The real-data experiments produced one useful diagnostic component but no validated forecast improvement:

- The causal statistical detector is well calibrated enough for retrospective diagnostics.
- The test week does not appear covariate-shifted after fixing two context-detector bugs.
- Downweighting anomalous seasonal lags made held-out WAPE slightly worse.
- The autoencoder's flags are highly dependent on the temporal split and primarily track long-run regime drift.
- Full direct LightGBM/NeuralNet confirmation was not completed because exact panel construction and model fitting exceeded the sandbox's memory/time envelope. Because the lightweight weighting test was already negative, no promotion claim is warranted.

## Data actually used

| Item | Value |
|---|---:|
| Original observed rows | 51,534 |
| Rows after daily reindexing | 51,563 |
| Products | 30 |
| Training period | 2021-01-01 to 2026-01-11 |
| Test rows | 210 |
| Test period | 2026-01-12 to 2026-01-18 |

The autoencoder was **not** trained on all 1,810 windows. Temporal holdouts were preserved by design:

- original split: 1,176 training / 362 calibration / 272 holdout windows;
- recent split: 1,448 training / 181 calibration / 181 holdout windows.

## 1. Causal statistical anomaly detector

| Result | Value |
|---|---:|
| Scored product-days | 49,595 |
| Local anomalies | 485 (0.941%) |
| Systemic anomaly days | 35 |
| Local threshold | 4.2773 |
| Local validation false-alarm rate | 0.494% (target 1.0%) |
| Systemic validation false-alarm rate | 1.385% (target 2.0%) |
| Flagged local anomalies on known events | 191 (39.4%) |
| Mean weight among flagged rows | 0.7327 |
| Mean weight across all rows | 0.9975 |

Interpretation: the EVT/POT layer is conservative rather than over-triggering. It is suitable for analysis and labeling. That does not imply that attenuating these rows improves forecasting.

## 2. Test-week context risk

After correcting the feature contract, the January 12–18, 2026 test week is not meaningfully out of distribution:

| Result | Value |
|---|---:|
| Rows scored | 210 |
| Context-shift flags at 99th percentile | 0 |
| Mean training-percentile risk | 0.4309 |
| Maximum training-percentile risk | 0.9882 |

Two implementation defects were found and fixed during real-data execution:

1. `ProductAvailable` was initially treated as a future context feature although it is absent from `test_data.parquet`.
2. Reindexed training campaign IDs such as `-1.0` were being treated as categories distinct from test IDs such as `-1`; categorical numeric codes are now canonicalized.

## 3. Forecast-impact test: anomaly weighting

This test held the forecast rule constant and changed only the influence of anomalous historical same-weekday observations. It used four development origins and four recent benchmark origins.

### Aggregate results

| Split | Policy | WAPE | MAE | Bias ratio | Rows |
|---|---|---:|---:|---:|---:|
| Development | Control | 0.456774 | 21.0493 | 0.0832 | 813 |
| Development | Soft weighting | 0.457194 | 21.0686 | 0.0859 | 813 |
| Development | Default weighting | 0.457998 | 21.1056 | 0.0881 | 813 |
| Benchmark | Control | 1.115914 | 31.8155 | 0.8550 | 840 |
| Benchmark | Soft weighting | 1.117833 | 31.8702 | 0.8573 | 840 |
| Benchmark | Default weighting | 1.119706 | 31.9236 | 0.8595 | 840 |

Relative to control:

- soft weighting was 0.0918% worse on development and 0.1719% worse on benchmark;
- default weighting was 0.2679% worse on development and 0.3398% worse on benchmark;
- neither candidate passed the predeclared 0.2% development-improvement gate;
- selected policy: **control**.

The effect was heterogeneous: default weighting helped the 2025-02-10 origin (`0.481748 -> 0.478210`) but hurt 2023-01-10 (`0.446484 -> 0.458078`). The aggregate result is negative.

## 4. Autoencoder training result

### Original temporal split

| Item | Value |
|---|---:|
| Window length | 28 days |
| Total windows scored | 1,810 |
| Training windows | 1,176 |
| Calibration windows | 362 |
| Holdout windows | 272 |
| Epochs | 40 |
| Final train loss | 0.300054 |
| Flagged windows | 64 |
| Validation false-alarm rate | 33.33% (target 2.0%) |
| Score/date correlation | 0.758 |

The threshold is not trustworthy: calibration false alarms reached 33.3%. Scores increase strongly with date, and the largest cluster covers Christmas/New Year 2025–2026. The network mostly learned the early-history regime and treated later distributional change as reconstruction failure.

### More recent training split

| Item | Value |
|---|---:|
| Training windows | 1,448 |
| Calibration windows | 181 |
| Holdout windows | 181 |
| Epochs | 30 |
| Final train loss | 0.318000 |
| Total flagged windows | 1 |
| Holdout flagged windows | 0 |
| Score/date correlation | 0.551 |

With 80% of history used for training, the 64 original flags collapse to one calibration flag and zero holdout flags. Therefore the binary anomaly labels are not temporally robust.

## What the autoencoder currently means

It is a **regime-drift diagnostic**, not a validated anomaly detector and not a useful forecast feature in its current form. To make it scientifically defensible, it would need at least one of:

- rolling or expanding retraining;
- per-window level/scale normalization;
- residual inputs after removing calendar, trend, campaign and price effects;
- explicit drift detection separate from isolated anomaly detection;
- calibration evaluated on several temporal folds rather than one split.

## Evidence boundary

The following results were not obtained and must not be claimed:

- no full direct LightGBM or NeuralNet anomaly-policy comparison;
- no completed representative anomaly-feature ablation;
- no evidence that the anomaly extension improves the final submission;
- no evidence that autoencoder flags identify data corruption.

The exact direct-panel runs exceeded the available sandbox memory/time envelope. The representative Ridge anomaly-feature run also exceeded five minutes before finishing a fold, so it was stopped without producing a result.

## Interview recommendation

Keep the statistical anomaly audit as an **EDA and model-risk layer**, but leave `anomaly_mode=off` for the submitted forecast.

The strong interview narrative is not “the autoencoder improved WAPE.” It is:

> I transferred a mature anomaly-detection hypothesis into the forecasting pipeline, enforced temporal leakage constraints, calibrated it statistically, tested it against held-out forecast origins, found that anomaly attenuation slightly degraded WAPE, and rejected the mechanism. The autoencoder additionally exposed long-run regime drift, which is useful diagnostics but not evidence of isolated anomalies.

That is a stronger data-science result than promoting an attractive but unvalidated neural component.

## Reproduction commands

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

The recent-split autoencoder used the same implementation with `train_fraction=0.80`, `calibration_fraction=0.10`, and 30 epochs.
