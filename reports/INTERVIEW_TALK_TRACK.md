# Interview talk track: why DAVID belongs here—and where it does not

## 90-second version

My original solution is a leakage-safe seven-day direct demand forecaster. I also maintain DAVID, a substantially larger anomaly-detection framework, so I asked a deliberately difficult question: can any of its methodology improve this small retail task without turning the repository into architecture theatre?

I did not import DAVID wholesale. I transferred three ideas. First, reconstruction error became a causal demand-surprise score: each product is compared with its prior same weekdays, and severity is standardized only against its past. Second, DAVID's Peaks-Over-Threshold EVT calibration replaces an arbitrary outlier cutoff. Third, detection is separated from action. An anomaly may become an origin-known regime feature or a bounded training weight, but it is never blindly deleted.

That last distinction matters because my own error analysis shows that 89 of the neural network's 100 largest development errors occur around holiday or event periods, and most are underpredictions. Promotional spikes are often the signal I need to learn, not contamination. Therefore the implementation protects explainable positive campaign spikes and broad systemic events.

Every option is behind an `off`, `weight`, `features`, or `both` ablation. Each CV fold fits its detector only on pre-origin history, and target anomaly information cannot enter the feature schema. A fast LightGBM screen selects nothing unless development WAPE improves and the recent benchmark remains stable; only then is the full neural model rerun. A control win is an acceptable result. The contribution is not “I used an autoencoder”; it is that I translated an advanced anomaly system into a falsifiable, leakage-safe forecasting hypothesis.

## Five-minute technical walkthrough

### 1. Start from the decision boundary

The output is still a seven-day demand forecast. Future demand anomalies are unavailable at prediction time, so a retrospective anomaly score cannot simply be fed into the future rows. I split the problem into:

- historical demand surprise, used for robust training and origin state;
- future-known context shift, used as forecast confidence metadata.

### 2. Historical local score

For each product-day I build an expected quantity from available observations at lags 7, 14, 21, and 28, weighted toward the most recent same weekday. I calculate a log residual and standardize its absolute deviation using a past-only rolling median and IQR. This is interpretable, data-efficient, and causal.

### 3. Tail calibration

Instead of saying “anything above three sigma is anomalous,” I fit a Generalized Pareto Distribution to exceedances over candidate thresholds. The selected threshold balances target false-alarm rate on a later temporal block and tail goodness of fit. When the tail is too small, the method falls back to an empirical quantile rather than forcing an unstable EVT estimate.

### 4. Local versus systemic events

One unusual SKU is local. When the 90th percentile across products is also extreme, the day is systemic. This distinction controls the action: isolated severe rows may be downweighted more; systemic days retain a higher floor because they may indicate a genuine market regime.

### 5. Two downstream hypotheses

The feature hypothesis uses only state known at the origin: latest severity/flag, anomaly rate over 28 days, days since anomaly, and equivalent systemic state.

The robust-loss hypothesis uses a bounded weight for historical target rows. It never drops observations. Existing sample weights are multiplied and renormalized to keep optimizer scale comparable.

### 6. Business protection

Positive residuals with known campaign, discount, or promotion context receive a minimum weight. This is directly motivated by the baseline: top-demand days are underpredicted and holiday events dominate the largest errors. Treating them as noise would improve an anomaly statistic while damaging the business objective.

### 7. Leakage controls

For an origin on day `O`, the anomaly detector is fitted only on rows before `O`. The model sees anomaly state at `O`, not at target day `O+h`. Target anomaly columns are diagnostics/weights for already-observed training labels and are excluded from the feature list. The test week contributes no demand information.

### 8. Evaluation

I screen a predeclared set of anomaly policies with LightGBM on the same direct panel. A candidate needs at least 0.2% relative development WAPE improvement and at most 2% benchmark regression. Only a passing candidate earns an expensive neural confirmation. The frozen final audit remains untouched during selection.

### 9. Optional autoencoder

I implemented a small 28-day multivariate demand autoencoder because it is the closest literal DAVID transfer. It is intentionally retrospective: it can identify days when the joint 30-product demand shape is poorly reconstructed and compare them with forecast failures. I do not present it as a future feature because its reconstruction error cannot be observed before sales happen.

## Likely questions and exact answers

### “Why not simply remove anomalous observations?”

Because anomaly means improbable under a reference distribution, not incorrect. In retail, Black Friday, campaigns, stock recovery, and viral demand are improbable but economically real. I use bounded influence and protection rules, then let walk-forward WAPE decide.

### “Why EVT rather than a z-score?”

The score distribution is non-Gaussian and heterogeneous. EVT models the exceedance tail directly and lets me target a false-alarm probability. I still retain an empirical fallback because this dataset is small; refusing an unstable parametric fit is part of the method.

### “Is weighting by the target an information leak?”

Not when used only during fitting on already-observed historical labels. Robust regression routinely makes loss influence depend on residual or label behavior. The prohibited operation would be exposing target-derived anomaly values as predictors or calibrating the fold detector with its validation future. Both are explicitly blocked.

### “Why does the feature use lag zero?”

`lag0` refers to the latest observed state at the forecast origin, not the future target day. In the direct panel the same origin state is available to all seven horizons.

### “Why screen with LightGBM if the submission is a neural network?”

It is a computational triage stage on the exact same rows, features, and weights. It tests whether the proposed representation contains signal. It does not certify the neural result; a passing candidate must be confirmed with the submitted NeuralNet configuration.

### “What if the control wins?”

Then the honest conclusion is that DAVID-style scoring improves diagnostics and confidence reporting but not point forecast WAPE on this dataset. That is stronger than forcing a complex model into production because it was available.

### “What is the most useful output besides WAPE?”

A test-week context-risk percentile. It states whether the future price/campaign/calendar combinations are supported by training history. That is actionable uncertainty information even when anomaly weighting does not improve the point forecast.

### “Why direct-only initially?”

The direct strategy is the selected and audited submission contract. Adding the anomaly layer to both direct and recursive paths before establishing value would double the experimental surface and weaken attribution. Recursive integration is justified only after the direct hypothesis survives its gate.

## Demonstration sequence

1. Show the original model dashboard and the test-aligned final selection.
2. Show the baseline top-decile bias and top-error concentration in holiday events.
3. Open `ml/anomaly_detection.py` and explain score → EVT → action.
4. Point to the two separate joins: origin features and target-row weights.
5. Run or show `outputs/anomaly_audit/anomaly_metadata.json`.
6. Show `outputs/anomaly_screening/recommendation.json` and its acceptance gates.
7. End with the measured conclusion, including “control won” if applicable.
