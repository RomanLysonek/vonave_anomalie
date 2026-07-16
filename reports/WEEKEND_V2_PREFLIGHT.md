# Weekend-v2 preflight from the completed overnight artifacts

## Source

This analysis used the uploaded `outputs/overnight_anomaly_search` artifacts, including row-level development and recent-benchmark OOF predictions for:

- `cont-062b09b48747` — canonical NeuralNet control;
- `stat-322e3ec6f7ee` — `stat_019_both_rw90`;
- `hybr-747825677948` — `hybrid_01_stat-41cded1b1789_aeac-c8bed14e3b42`.

Only rows with known actual demand, finite predictions, and `ProductAvailable=True` were evaluated.

## Failed trial

The single failed overnight trial was `stat-53f9621c445f`. Its generated configuration had `anomaly_min_history=84` with `anomaly_rolling_window=60`, an invalid combination. It was not a promoted or confirmation candidate. Weekend-v2 constrains `min_history <= rolling_window` during candidate generation.

## Standalone confirmation

| Candidate | Development WAPE | Recent-benchmark WAPE | Interpretation |
|---|---:|---:|---|
| Control | **0.302737** | 0.276559 | robust global winner |
| Statistical specialist | 0.304499 | 0.271278 | 0.58% worse development, 1.91% better recent benchmark |
| Hybrid specialist | 0.322471 | **0.266121** | 6.52% worse development, 3.77% better recent benchmark |

Neither specialist is safe as a global replacement. Both contain recent-regime information that differs from the control.

## Cross-fitted mixture evidence

A leave-one-development-origin-out convex mixture was fitted without using the held-out origin. The full-development plan used for recent-benchmark evaluation was:

```text
control       0.526900
statistical   0.461240
hybrid        0.011860
```

| Evaluation | Control | Cross-fitted/full-dev mixture | Relative improvement |
|---|---:|---:|---:|
| Development WAPE | 0.302737 | **0.298280** | **1.472%** |
| Recent-benchmark WAPE | 0.276559 | **0.269894** | **2.410%** |
| Development top-decile WAPE | 0.238631 | **0.231255** | **3.091%** |
| Recent-benchmark top-decile WAPE | 0.217303 | **0.213917** | **1.558%** |

Development strata also improved:

- holiday/event: 0.328939 → 0.325633;
- regular: 0.282876 → 0.274297;
- winter/test-like: 0.256372 → 0.251977.

An origin-block bootstrap with 10,000 resamples gave:

```text
mean relative improvement: 1.451%
95% interval:             -0.242% to 2.846%
P(improvement > 0):       95.74%
```

The interval still touches zero because only 12 development origins exist, but the evidence is strong enough to justify a properly nested weekend search.

## Alternative structures on the same prior OOF

| Policy | Development improvement | Recent-benchmark improvement |
|---|---:|---:|
| Horizon-shrunk convex | 1.362% | 2.987% |
| Product-shrunk convex | 0.860% | 3.007% |
| Statistical specialist gate, max 85% | 1.302% | 2.943% |

The specialist gate also improved top-decile WAPE by 1.99% on development and 4.10% on the recent benchmark. This supports the hypothesis that anomaly mode has value **conditionally**, not necessarily as a universal model.

## Consequence for weekend-v2

The next search must:

1. retain anomaly modes as specialist generators;
2. add asymmetric and hard-example anomaly actions;
3. include non-anomaly recent-regime experts as competing explanations;
4. promote candidates by marginal ensemble value, not only standalone WAPE;
5. cross-fit every blend or gate by origin;
6. preserve top-decile and holiday/event safety gates;
7. leave the frozen final audit untouched.
