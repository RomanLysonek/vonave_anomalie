# Weekend-v2 methodology: specialist, ensemble, and regime search

> **Legacy evidence status: contaminated/unverified and excluded.** Historical
> overnight diagnostics, leaderboards, recommendations, and the non-nested
> preflight may contain benchmark-target contamination. Weekend-v2 must not seed
> candidates or selection from them. It was not rerun; control with
> `anomaly_mode=off` remains the recommendation.

## Purpose

The provenance-limited overnight report suggested two hypotheses:

1. no standalone anomaly candidate beat the canonical NeuralNet control;
2. a non-nested preflight blend appeared complementary, but its confidence interval crossed zero.

Weekend-v2 was not run. The proposed protocol asks whether anomaly-derived models are useful specialists inside a nested, cross-fitted mixture. The current recommendation remains the control with `anomaly_mode=off`.

The frozen final-audit origins remain excluded from all search and selection.

## Evidence motivating v2

The following historical numbers are retained for provenance review only and
are not current scientific or selection evidence. Using three legacy overnight
OOF members—control, `stat_019_both_rw90`, and `hybrid_01`—a leave-one-origin-out
global convex blend had reported:

| Evaluation | Control WAPE | Cross-fitted blend WAPE | Relative improvement |
|---|---:|---:|---:|
| Development | 0.302737 | 0.298280 | 1.472% |
| Recent benchmark | 0.276559 | 0.269894 | 2.410% |

The development-origin bootstrap estimated `P(improvement > 0) = 95.74%`. The full-development weights were approximately 52.7% control, 46.1% statistical specialist, and 1.2% hybrid specialist.

A constrained gate historically reported 1.302% development improvement and
2.943% recent-benchmark improvement. Because its sources are now classified
contaminated/unverified, this is not evidence for promotion or candidate
selection.

See [`../WEEKEND_V2_PREFLIGHT.md`](../WEEKEND_V2_PREFLIGHT.md).

## Why anomaly mode is retained

`anomaly_mode` is retained, but its interpretation is widened.

The original action assumed unusual observations were partly contaminated and should be attenuated. Weekend-v2 searches four actions:

| Policy | Positive unusual demand | Negative unusual demand | Hypothesis |
|---|---:|---:|---|
| `downweight` | downweight | downweight | both tails contain contamination |
| `negative_only` | preserve | downweight | negative anomalies are availability/censoring problems; positive spikes are demand signal |
| `hard_example` | upweight | upweight | anomalies are the commercially important hard cases the model must learn |
| `signed` | upweight | downweight | positive shocks are valuable; negative shocks are more likely censoring/noise |

The existing event-protection and systemic-shock floors remain active. We do not delete rows.

Weekend-v2 also includes anomaly-free recent-regime specialists. This gives the search permission to conclude that recency, loss, or target representation explains the complementarity better than anomaly detection.

## Search profile

The default `weekend-v2` profile starts with 45 candidates:

- 1 canonical control;
- 24 statistical anomaly candidates from fixed development hypotheses (legacy overnight winners are excluded);
- 10 anomaly-free regime specialists;
- 0 autoencoder specialists from historical diagnostics;
- 0 statistical/autoencoder hybrids seeded from historical diagnostics.

### Stage 1: NeuralNet screen

- 5 development origins;
- 8 recent-benchmark origins;
- 10 epochs;
- seed 42;
- all candidates use the actual direct NeuralNet path.

DynamicRidge is no longer the promotion proxy because the first search showed that its candidate ranking transferred poorly to the NeuralNet.

The refine set is selected by **greedy cross-fitted marginal ensemble value**. Standalone WAPE is considered, but a candidate can advance because it diversifies the current expert set.

### Stage 2: refine

- control plus up to 10 specialists;
- 10 development origins;
- 18 recent-benchmark origins;
- 30 epochs;
- seeds 42 and 123.

The confirmation set is again selected by leave-one-origin-out marginal ensemble value. At least one anomaly-derived candidate is retained for confirmation coverage unless every anomaly trial failed.

### Stage 3: confirmation

- control plus up to 5 specialists;
- all 12 development origins;
- 36 recent-benchmark origins;
- 55 epochs;
- seeds 42, 123, and 777.

### Stage 4: meta-policy search

Every final policy is evaluated by leave-one-origin-out development predictions and then applied to the recent benchmark after fitting on the full development set.

The policies are:

1. **Global convex blend** — one non-negative sum-to-one weight vector.
2. **Horizon-shrunk blend** — horizon-specific weights shrunk toward the global mixture.
3. **Product-shrunk blend** — SKU-specific weights shrunk strongly toward the global mixture.
4. **Aggregate reconciliation** — global mixture plus conservative horizon-level total-demand correction.
5. **Ridge residual stacker** — linear correction around the control.
6. **Nonlinear risk gate** — conservative nonlinear residual correction.
7. **Specialist advantage gates** — for every confirmed specialist, bounded 50% and 85% gates driven by predicted absolute-error advantage.

The convex search is vectorized; tens of thousands of Dirichlet draws are scored in matrix batches rather than through Python/pandas loops.

## Leakage contract

- Candidate anomaly/autoencoder profiles are fitted independently inside each walk-forward fold.
- Only origin-known state is persisted into OOF files.
- Candidate-specific anomaly state is namespaced by candidate id before meta-learning.
- Meta-policy predictions on development are leave-one-origin-out.
- The recent benchmark is used as a selection/safety regime, not to fit blend weights.
- The final-audit origins are not read by weekend-v2.
- Test-week actual demand is never used.

Final execution accepts only recommendation schema v4 with provenance schema v2.
Every member binds a safe search-root-relative `result.json` path, the complete
expected fingerprint, canonical result-body digest, candidate identity,
development/benchmark summary digests, and both OOF file fingerprints. The final
runner revalidates every source result and OOF, plan/weight membership, and any
authenticated pickle before loading training data. Missing, legacy, traversing,
or tampered sources fail closed.

## Promotion gates

A meta-policy is accepted only when all conditions hold:

- development WAPE improves by at least 0.2% relative;
- recent-benchmark WAPE does not regress by more than 1%;
- development top-decile WAPE does not regress by more than 2%;
- development holiday/event WAPE does not regress by more than 3%;
- origin-block bootstrap probability of positive improvement is at least 75%.

The highest selection score among accepted policies wins. If none passes, the winner remains the canonical control.

## Run

From the repository root:

```bash
uv sync
uv run pytest -q tests/test_weekend_v2.py
scripts/run_weekend_v2_smoke.sh
scripts/run_weekend_v2.sh
```

The full launcher requires:

```text
data/train_data.parquet
data/test_data.parquet
outputs/overnight_anomaly_search/recommendation.json
```

The launcher checks MPS, prevents sleep with `caffeinate`, streams a persistent log, isolates each candidate in a subprocess, and resumes completed trials.

## Monitor and resume

```bash
scripts/weekend_v2_status.sh
scripts/resume_weekend_v2.sh
```

A bounded session can be used without losing progress:

```bash
MAX_HOURS=16 scripts/run_weekend_v2.sh
```

Resume later with the ordinary resume command.

To retry failures:

```bash
RETRY_FAILED=1 scripts/resume_weekend_v2.sh
```

## Stage-specific execution

```bash
STAGE=screen scripts/run_weekend_v2.sh
STAGE=refine scripts/run_weekend_v2.sh
STAGE=confirmation scripts/run_weekend_v2.sh
STAGE=ensemble scripts/run_weekend_v2.sh
```

Later stages require completed earlier-stage artifacts.

## More aggressive profile

```bash
PROFILE=exhaustive-v2 \
OUTPUT_DIR=outputs/exhaustive_weekend_v2 \
scripts/run_weekend_v2.sh
```

The exhaustive profile widens candidate count, origins, epochs, seeds, and ensemble draws. It should be used only after the default v2 run verifies the machinery and produces a credible direction.

## Outputs

```text
outputs/weekend_v2_search/
  manifest.json
  candidate_pool.json
  refine_selection.json
  confirmation_selection.json
  screen/
  refine/
  confirmation/
  *_leaderboard.csv
  ensemble/
  ensemble_leaderboard.csv
  recommendation.json
  winner_plan.json
  FINAL_REPORT.md
```

The final command is stored in `recommendation.json`. It trains only the members required by the winning plan and writes:

```text
outputs/weekend_v2_search/final/submission.csv
outputs/weekend_v2_search/final/submission.parquet
outputs/weekend_v2_search/final/final_member_forecasts.parquet
```

## Interpreting the result

- **Anomaly standalone loses, blend wins:** anomaly mode is a useful specialist/regularizer, not the canonical estimator.
- **Signed or hard-example policy wins:** unusual positive demand was signal that the original downweighting action suppressed.
- **Regime expert dominates:** complementarity came primarily from recency or loss/target diversity, not anomaly detection.
- **Specialist gate wins:** anomaly value is conditional and should be activated only in identifiable regimes.
- **Control wins:** the additional machinery did not produce stable incremental forecasting value and should remain diagnostic only.
