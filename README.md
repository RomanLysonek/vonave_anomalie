# vonave_anomalie

**DAVID-informed anomaly-aware extension of the `vonava_predikce` retail demand forecaster.**

This repository does not pretend that anomaly detection magically creates signal in a small forecasting dataset. It transfers the parts of DAVID that are statistically useful here—reconstruction error, EVT/POT tail calibration, temporal windows, and a clean separation between detection and action—into the existing leakage-safe direct forecasting pipeline.

The original forecasting implementation and its detailed documentation are preserved in [`FORECAST_BASELINE.md`](FORECAST_BASELINE.md). The design audit is in [`reports/DAVID_TRANSFER_AUDIT.md`](reports/DAVID_TRANSFER_AUDIT.md), and an interview-ready explanation is in [`reports/INTERVIEW_TALK_TRACK.md`](reports/INTERVIEW_TALK_TRACK.md).

## Current status

- The original submission behavior is unchanged by default: `anomaly_mode="off"`.
- The anomaly layer is integrated into direct walk-forward CV and final direct training.
- New anomaly code has synthetic leakage, alignment, threshold, protection, and integration tests.
- The real-data statistical audit, two real autoencoder trainings, and a low-memory held-out forecast-weight ablation have now been executed.
- The weighting ablation selected the control: soft weighting worsened development WAPE by 0.0918% and recent-benchmark WAPE by 0.1719%; default weighting worsened them by 0.2679% and 0.3398%.
- The original autoencoder split flagged 64 windows but failed temporal calibration; after moving the training boundary later, only one calibration window and zero holdout windows were flagged. The current autoencoder is therefore a regime-drift diagnostic, not a validated forecast component.
- The corrected test-context detector flagged 0 of 210 test rows at the 99th-percentile threshold.
- The earlier sandbox quick run could not complete exact direct LightGBM/NeuralNet confirmation and is treated only as a smoke test; the completed Apple-Silicon overnight search below supersedes it.
- The first Apple-Silicon overnight search is complete: no anomaly model beat the control robustly as a standalone NeuralNet, but its confirmed statistical specialist produced complementary OOF errors.
- A leave-one-origin-out blend of the three confirmed members improved development WAPE by 1.47% and recent-benchmark WAPE by 2.41%; this motivates weekend-v2.
- Weekend-v2 retains anomaly mode as a specialist generator, adds asymmetric/hard-example policies and anomaly-free regime experts, and selects cross-fitted mixtures or gates rather than requiring one specialist to replace the control.

The quick-run evidence is in [`reports/REAL_DATA_TEST_RESULTS.md`](reports/REAL_DATA_TEST_RESULTS.md). The serious local experiment is documented in [`docs/OVERNIGHT_ANOMALY_SEARCH.md`](docs/OVERNIGHT_ANOMALY_SEARCH.md).

## Anomaly Lab dashboard

The local webapp now contains a dedicated **Anomaly Lab** rather than only the inherited forecasting pages. It reads the research artifacts directly from `outputs/` and shows:

- product-level statistical anomalies against their causal seasonal expectation;
- local versus systemic flags and known-event protection;
- autoencoder reconstruction timelines, split calibration and temporal drift;
- test-week context novelty using future-known covariates only;
- the overnight candidate funnel and confirmed NeuralNet comparison;
- live weekend-v2 screening/refinement/confirmation progress;
- the final specialist/ensemble recommendation and generated command.

```bash
uv run python webapp/server.py
```

Open [`http://127.0.0.1:9001/anomalies`](http://127.0.0.1:9001/anomalies). The page polls the lightweight search-status endpoint once per minute, so a running weekend-v2 experiment becomes visible without restarting the server. Override the port with `VONAVE_ANOMALIE_PORT=<port>`.

## What was transferred from DAVID

| DAVID concept | Forecasting adaptation | Purpose |
|---|---|---|
| Reconstruction error | causal same-weekday residual and optional multivariate autoencoder reconstruction error | quantify unusual demand behavior |
| POT/GPD EVT calibration | temporally validated local and systemic upper-tail thresholds | replace arbitrary z-score cutoffs |
| Sliding windows | 28-day anomaly rate and days-since-anomaly state | expose recent demand instability at each forecast origin |
| Filter/select action layer | bounded sample-weight attenuation | reduce influence of likely corrupted/idiosyncratic labels without deleting history |
| Inference-time anomaly scoring | known-context Isolation Forest | report how unusual the future campaign/price/calendar context is before demand arrives |
| Model deployment separation | explicit `off`, `weight`, `features`, `both` modes | make every effect ablatable and reversible |

DAVID's log parsing, sentence embeddings, Drain templates, cloud providers, and Airflow orchestration were deliberately not copied: they solve a different problem and would make this interview repository look larger rather than smarter.

## Architecture

```text
historical product-day rows
        |
        +--> causal same-weekday expectation (t-7/14/21/28 only)
        |          |
        |          +--> log residual
        |                 |
        |                 +--> past-only rolling median/IQR severity
        |                           |
        |                           +--> POT/GPD EVT threshold
        |                                      |
        |                                      +--> local anomaly flag
        |
        +--> cross-product daily 90th percentile severity
        |          |
        |          +--> systemic POT/GPD threshold
        |
        +--> origin-known anomaly state features
        |
        +--> bounded target-row training weights
                    |
                    +--> existing NN / LightGBM / XGBoost / Ridge sample_weight contract

known test-week context (price, campaign, discount, calendar)
        |
        +--> Isolation Forest + robust rarity score
                    |
                    +--> forecast context-risk percentile
```

Two information paths are kept separate:

1. **Feature path:** only anomaly state known at the forecast origin enters the model.
2. **Robust-loss path:** target-date anomaly information may influence the training loss for already-observed historical labels, but never becomes a predictor.

## Why event spikes are protected

The checked-in baseline diagnostics show that the model's hard cases are not generic random outliers:

| Evaluation slice | NeuralNet WAPE | Ensemble WAPE |
|---|---:|---:|
| Development, conditional/common/global | 0.2998 | 0.2679 |
| Recent benchmark, conditional/common/global | 0.2735 | 0.2471 |
| Frozen final audit, conditional/common/global | 0.3011 | 0.2993 |
| Frozen final audit, test-aligned objective | **0.2783** | 0.2797 |
| Development top demand decile | 0.2320 | 0.2069 |

Among the 100 largest NeuralNet development errors, 89 are in `holiday_event` strata and 59% are underpredictions. Therefore:

- a positive residual under a known campaign/promotion receives a configurable minimum weight;
- a broad systemic shock also receives a minimum weight;
- no observed row is automatically deleted;
- every anomaly policy must beat the control on walk-forward WAPE before it can be recommended.

## Recommended next experiment: weekend-v2 specialist search

The first search answered the standalone-model question. Weekend-v2 asks the more useful question: **can anomaly-aware and recent-regime experts improve the canonical NeuralNet when combined only in the origins, products, or horizons where they add value?**

The prior confirmation OOF already supports this direction:

| Policy on prior confirmed OOF | Development improvement | Recent-benchmark improvement |
|---|---:|---:|
| Cross-fitted global convex blend | 1.47% | 2.41% |
| Horizon-shrunk blend | 1.36% | 2.99% |
| Product-shrunk blend | 0.86% | 3.01% |
| Bounded statistical specialist gate | 1.30% | 2.94% |

Weekend-v2 searches 45 initial NeuralNet candidates, including anomaly-free recency/loss specialists and four anomaly actions: symmetric downweighting, negative-only downweighting, hard-example upweighting, and signed positive-up/negative-down weighting. Candidate promotion is based on leave-one-origin-out marginal ensemble value. The final stage compares global, horizon, product, aggregate, residual-stacking, nonlinear-risk, and bounded specialist-gate policies.

```bash
uv sync
uv run pytest -q tests/test_weekend_v2.py
scripts/run_weekend_v2_smoke.sh
scripts/run_weekend_v2.sh
```

Monitor or resume with:

```bash
scripts/weekend_v2_status.sh
scripts/resume_weekend_v2.sh
```

See [`docs/WEEKEND_V2.md`](docs/WEEKEND_V2.md) for the complete protocol and [`reports/WEEKEND_V2_PREFLIGHT.md`](reports/WEEKEND_V2_PREFLIGHT.md) for the analysis of the uploaded overnight artifacts.

## Completed first experiment: Apple-GPU overnight search

The earlier single autoencoder run is not the final test. The repository now contains a staged search that varies the demand representation, architecture, temporal training span, calibration policy, anomaly action, and statistical/autoencoder hybrid. It then requires improvement on the exact forecast pipeline before promotion.

```bash
uv sync
uv run pytest -q \
  tests/test_anomaly_detection.py \
  tests/test_anomaly_pipeline_integration.py \
  tests/test_systemic_autoencoder_v2.py \
  tests/test_overnight_anomaly_search.py

scripts/run_overnight_anomaly_search.sh
```

Progress and resume commands:

```bash
scripts/anomaly_search_status.sh
scripts/resume_anomaly_search.sh
```

The default `overnight` profile evaluates 36 autoencoder configurations across multiple cutoffs and seeds, then promotes candidates through exact DynamicRidge and MPS NeuralNet stages. Wider `weekend` and `exhaustive` profiles are also available. Every candidate is isolated in its own process, autoencoder profiles are fold/config cached, and the final decision uses development WAPE, recent-benchmark protection, top-demand-decile and holiday/event guards, plus origin bootstrap uncertainty.

See [`docs/OVERNIGHT_ANOMALY_SEARCH.md`](docs/OVERNIGHT_ANOMALY_SEARCH.md) for the complete protocol, search dimensions, leakage contract, outputs, and final forecast command.

## Running the implementation

### 1. Install and test

```bash
uv sync
uv run pytest -q
```

### 2. Generate the retrospective anomaly and future-context audit

```bash
uv run python ml/run_anomaly_audit.py \
  --output-dir outputs/anomaly_audit \
  --alpha 0.01
```

Produces:

- `demand_anomaly_profile.csv`
- `test_context_risk.csv`
- `test_context_risk_daily.csv`
- `anomaly_metadata.json`

### 3. Screen anomaly policies on the exact direct-panel contract

```bash
uv run python ml/run_anomaly_screening.py --profile screen
```

The screen compares:

- `control`
- `weight_soft`
- `weight_default`
- `features_only`
- `both_soft`
- `both_default`

It uses LightGBM as a screening instrument, not as the final claim. Building the exact stacked direct panel is resource-intensive; use the lower-memory falsification script below before spending time on this grid. A candidate is accepted only when it:

- improves development WAPE by at least 0.2% relative; and
- regresses the recent pseudo-test benchmark by no more than 2% relative.

The generated `outputs/anomaly_screening/recommendation.json` includes the exact full NeuralNet confirmation command. If no candidate passes, the winner is `control`, which is a valid and useful result.

Completed lower-memory weighting ablation:

```bash
uv run python ml/run_lightweight_anomaly_ablation.py
```

The checked-in result selects `control`; see `outputs/lightweight_anomaly_ablation/result.json`.

### 4. Optional DAVID-like systemic autoencoder diagnostic

```bash
uv run python ml/run_systemic_autoencoder.py \
  --output-dir outputs/anomaly_autoencoder \
  --epochs 40
```

This detector is retrospective and diagnostic. It is not enabled as a default forecast feature because future reconstruction error cannot be known when the seven-day forecast is produced.

### 5. Run a selected anomaly mode directly

```bash
caffeinate -i uv run python ml/pipeline.py \
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
  --c34-config outputs/c34_screening/recommendation.json \
  --anomaly-mode both \
  --anomaly-evt-alpha 0.01 \
  --anomaly-weight-strength 0.50 \
  --anomaly-min-weight 0.40 \
  --nn-batch-size 512 \
  --nn-lr-scaling fixed \
  --reset-checkpoints
```

Use the settings emitted by the screening recommendation rather than assuming the example above is superior.

## Anomaly modes

| Mode | Origin features | Training weights | Intended interpretation |
|---|---:|---:|---|
| `off` | no | no | exact control / original project |
| `weight` | no | yes | robust fitting only |
| `features` | yes | no | regime-memory signal only |
| `both` | yes | yes | combined hypothesis |

The implementation is direct-first because the submitted model and strongest validation contract are direct multi-horizon. Recursive integration is intentionally deferred until direct ablation demonstrates value.

## Important files

```text
ml/anomaly_detection.py             causal local/systemic score, EVT, features, weights
ml/context_risk.py                  forecast-time known-context shift detector
ml/systemic_autoencoder.py          original compact reconstruction diagnostic
ml/systemic_autoencoder_v2.py       temporal MLP/Conv/GRU search implementation
ml/run_overnight_anomaly_search.py  first staged resumable search orchestrator
ml/weekend_v2_common.py              specialist candidates, blends, gates, bootstrap
ml/run_weekend_v2_search.py          NeuralNet-only weekend-v2 orchestrator
ml/run_weekend_v2_final.py           full-history winner training and submission
ml/run_autoencoder_diagnostic_trial.py isolated GPU diagnostic worker
ml/run_anomaly_forecast_trial.py    isolated exact-panel proxy/NN worker
ml/run_anomaly_audit.py             actual-data retrospective/context audit
ml/run_anomaly_screening.py         control-vs-candidate walk-forward screening
ml/run_systemic_autoencoder.py      optional reconstruction experiment
ml/run_lightweight_anomaly_ablation.py real-data low-memory weighting falsification
ml/offline_parquet.py               minimal fallback reader for the bundled flat Parquet files
ml/framework.py                     anomaly config and feature schema integration
ml/pipeline.py                      fold-local fitting and final-training integration
tests/test_anomaly_detection.py     causal/EVT/protection/alignment tests
tests/test_anomaly_pipeline_integration.py
tests/test_context_risk.py
tests/test_systemic_autoencoder.py
tests/test_systemic_autoencoder_v2.py
tests/test_overnight_anomaly_search.py
tests/test_weekend_v2.py
docs/OVERNIGHT_ANOMALY_SEARCH.md
docs/WEEKEND_V2.md
reports/WEEKEND_V2_PREFLIGHT.md
reports/DAVID_TRANSFER_AUDIT.md
reports/INTERVIEW_TALK_TRACK.md
reports/REAL_DATA_TEST_RESULTS.md
```

## Non-negotiable validity rules

- Each CV fold fits its anomaly profile only on that fold's training history.
- Target anomaly values never enter the predictor feature list.
- Known event spikes are not assumed to be noise.
- Sample weights are bounded and normalized to preserve effective learning-rate scale.
- Checkpoint signatures include the complete anomaly configuration, so control and anomaly runs cannot be mixed on resume.
- The frozen final audit is not used for anomaly policy selection.
- No improvement is reported until the actual screening and full NeuralNet confirmation have run.
