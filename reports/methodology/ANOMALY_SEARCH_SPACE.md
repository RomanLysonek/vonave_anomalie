# Anomaly search-space reference

The executable search space is defined in `ml/anomaly_search_common.py` so every generated candidate is deterministic from `--seed` and recorded in `manifest.json`.

## Statistical detector dimensions

- mode: features, weight, both;
- rolling robust history: 60–540 days;
- minimum history: 21–84 observations;
- robust scale floor: 0.05–0.40 log units;
- local tail alpha: 0.25%–5%;
- EVT tail start: 80%–95%;
- attenuation strength: 0.15–1.50;
- minimum row weight: 0.20–0.90;
- known-event minimum: 0.65–1.00;
- systemic-event minimum: 0.50–1.00.

## Autoencoder detector dimensions

See `reports/methodology/OVERNIGHT_ANOMALY_SEARCH.md` for representations and architectures. The autoencoder search is deliberately broader than the action search: several models may produce informative continuous state while their binary threshold is poor. Therefore `features` and `both` are evaluated independently after diagnostic ranking.

## Hybrid policy

The proxy stage combines the three strongest statistical candidates with the three strongest autoencoder candidates. Hybrid target weights multiply the two bounded weights and are normalized once, preserving mean training weight. Both origin-state feature families remain available to the forecast model.

## Reproducibility

Candidate IDs are SHA-256-derived from the full candidate payload. Fold checkpoints include all `Config` fields, including the complete autoencoder definition and cache location. Changing any semantic/training hyperparameter invalidates the checkpoint signature.
