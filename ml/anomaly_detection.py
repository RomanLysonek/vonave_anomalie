"""Leakage-safe demand anomaly layer inspired by DAVID.

The module deliberately separates three concerns:

1. scoring: causal same-weekday residuals and robust product-local severity;
2. calibration: Peaks-Over-Threshold / Generalized Pareto thresholds;
3. action: origin-state features and bounded loss weights.

The anomaly label is not treated as ground truth.  It is a model diagnostic and
an input to controlled forecast ablations.  Positive campaign/event spikes are
protected from aggressive down-weighting because they are often precisely the
observations a retail forecaster must learn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import genpareto, kstest


BASELINE_LAGS = (7, 14, 21, 28)
BASELINE_WEIGHTS = np.asarray((4.0, 3.0, 2.0, 1.0), dtype=float)
ANOMALY_ORIGIN_FEATURES = [
    "anomaly_score_lag0",
    "anomaly_flag_lag0",
    "anomaly_rate_28",
    "days_since_anomaly",
    "systemic_anomaly_score_lag0",
    "systemic_anomaly_flag_lag0",
    "systemic_anomaly_rate_28",
]


@dataclass(frozen=True)
class EVTCalibration:
    """Calibrated upper-tail threshold and fit diagnostics."""

    threshold: float
    method: str
    alpha: float
    tail_quantile: float
    u: float | None = None
    xi: float | None = None
    beta: float | None = None
    n_total: int = 0
    n_exceed: int = 0
    validation_exceedance_rate: float | None = None
    validation_exceedance_error: float | None = None
    ks_stat: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cfg(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def anomaly_weighting_enabled(cfg: Any) -> bool:
    return str(_cfg(cfg, "anomaly_mode", "off")).lower() in {"weight", "both"}


def anomaly_features_enabled(cfg: Any) -> bool:
    return str(_cfg(cfg, "anomaly_mode", "off")).lower() in {"features", "both"}


def _safe_quantity(frame: pd.DataFrame) -> pd.Series:
    if "Quantity" in frame.columns:
        return pd.to_numeric(frame["Quantity"], errors="coerce")
    app = pd.to_numeric(frame.get("QuantityApp", 0.0), errors="coerce").fillna(0.0)
    web = pd.to_numeric(frame.get("QuantityWeb", 0.0), errors="coerce").fillna(0.0)
    return app + web


def _known_event_mask(frame: pd.DataFrame) -> pd.Series:
    sale = frame.get("IsSaleOrPromo", pd.Series(False, index=frame.index))
    sale = sale.astype("boolean").fillna(False).astype(bool)
    cw = pd.to_numeric(
        frame.get("CampaignSubTypeWeb", pd.Series(-1, index=frame.index)),
        errors="coerce",
    ).fillna(-1)
    ca = pd.to_numeric(
        frame.get("CampaignSubTypeApp", pd.Series(-1, index=frame.index)),
        errors="coerce",
    ).fillna(-1)
    dw = pd.to_numeric(
        frame.get("DiscountValueWebRelative", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    da = pd.to_numeric(
        frame.get("DiscountValueAppRelative", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    return sale | cw.ne(-1) | ca.ne(-1) | dw.gt(0.0) | da.gt(0.0)


def _causal_same_weekday_baseline(work: pd.DataFrame) -> np.ndarray:
    """Weighted 7/14/21/28-day baseline with past-only fallback."""

    available = work["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
    qty = work["Quantity"].where(available)
    grouped = qty.groupby(work["ProductId"], sort=False)
    lag_matrix = np.column_stack([
        grouped.shift(lag).to_numpy(dtype=float) for lag in BASELINE_LAGS
    ])
    observed = np.isfinite(lag_matrix)
    numerator = np.nansum(lag_matrix * BASELINE_WEIGHTS, axis=1)
    denominator = (observed * BASELINE_WEIGHTS).sum(axis=1)
    baseline = np.divide(
        numerator,
        denominator,
        out=np.full(len(work), np.nan, dtype=float),
        where=denominator > 0.0,
    )

    fallback = grouped.transform(
        lambda series: series.shift(1).rolling(28, min_periods=4).median()
    ).to_numpy(dtype=float)
    fallback_short = grouped.transform(
        lambda series: series.shift(1).rolling(7, min_periods=1).median()
    ).to_numpy(dtype=float)
    baseline = np.where(np.isfinite(baseline), baseline, fallback)
    baseline = np.where(np.isfinite(baseline), baseline, fallback_short)
    return baseline


def _rolling_robust_score(
    residual: pd.Series,
    product: pd.Series,
    *,
    window: int,
    min_history: int,
    scale_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Past-only rolling median/IQR z-score, evaluated product by product."""

    center = np.full(len(residual), np.nan, dtype=float)
    scale = np.full(len(residual), np.nan, dtype=float)
    values = residual.to_numpy(dtype=float)

    for _, indices in product.groupby(product, sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        series = pd.Series(values[idx], index=idx, dtype=float)
        past = series.shift(1)
        med = past.rolling(window, min_periods=min_history).median()
        q25 = past.rolling(window, min_periods=min_history).quantile(0.25)
        q75 = past.rolling(window, min_periods=min_history).quantile(0.75)
        robust_sigma = (q75 - q25) / 1.349
        center[idx] = med.to_numpy(dtype=float)
        scale[idx] = robust_sigma.to_numpy(dtype=float)

    valid_scale = np.isfinite(scale)
    scale[valid_scale] = np.maximum(scale[valid_scale], float(scale_floor))
    score = np.divide(
        np.abs(values - center),
        scale,
        out=np.full(len(values), np.nan, dtype=float),
        where=np.isfinite(values) & np.isfinite(center) & np.isfinite(scale) & (scale > 0),
    )
    return score, center, scale


def _gpd_threshold(alpha: float, u: float, xi: float, beta: float, pu: float) -> float:
    if alpha >= pu:
        return float(u)
    tail_probability = alpha / pu
    if abs(xi) < 1e-7:
        return float(u - beta * np.log(tail_probability))
    return float(u + (beta / xi) * (tail_probability ** (-xi) - 1.0))


def calibrate_evt_threshold(
    scores: Iterable[float],
    *,
    alpha: float = 0.01,
    tail_quantile: float = 0.90,
    min_exceedances: int = 30,
    validation_fraction: float = 0.20,
) -> EVTCalibration:
    """Calibrate a DAVID-style POT/GPD threshold with temporal holdout checks.

    Candidate tail cut-offs are evaluated by validation exceedance-rate error
    plus a small KS goodness-of-fit penalty.  The empirical quantile is a
    deterministic fallback when the tail is too small or numerically unstable.
    """

    values = np.asarray(list(scores), dtype=float)
    values = values[np.isfinite(values) & (values >= 0.0)]
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.5 <= tail_quantile < 1.0:
        raise ValueError("tail_quantile must be in [0.5, 1)")
    if values.size == 0:
        return EVTCalibration(
            threshold=float("inf"),
            method="no_scores",
            alpha=alpha,
            tail_quantile=tail_quantile,
        )

    fallback = float(np.quantile(values, 1.0 - alpha))
    if values.size < max(2 * min_exceedances, 100):
        return EVTCalibration(
            threshold=fallback,
            method="empirical_quantile_small_sample",
            alpha=alpha,
            tail_quantile=tail_quantile,
            n_total=int(values.size),
        )

    split = int(round(values.size * (1.0 - validation_fraction)))
    split = min(max(split, min_exceedances * 2), values.size - min_exceedances)
    fit_values = values[:split]
    validation = values[split:]

    q_low = max(0.80, tail_quantile - 0.05)
    q_high = min(0.98, tail_quantile + 0.06)
    candidates = sorted(set(np.round(np.linspace(q_low, q_high, 7), 4)))
    best: tuple[float, EVTCalibration] | None = None

    for q in candidates:
        u = float(np.quantile(fit_values, q))
        exceedances = fit_values[fit_values > u] - u
        if exceedances.size < min_exceedances:
            continue
        try:
            xi, _, beta = genpareto.fit(exceedances, floc=0.0)
        except (ValueError, FloatingPointError, RuntimeError):
            continue
        if not np.isfinite(xi) or not np.isfinite(beta) or beta <= 0.0:
            continue
        pu = float(exceedances.size / fit_values.size)
        threshold = _gpd_threshold(alpha, u, float(xi), float(beta), pu)
        if not np.isfinite(threshold) or threshold < u:
            continue
        # A threshold orders of magnitude beyond the observed support is not
        # actionable for this small retail dataset.
        if threshold > max(float(np.max(fit_values)) * 5.0, u + 20.0):
            continue
        exceedance_rate = (
            float(np.mean(validation > threshold)) if validation.size else float("nan")
        )
        exceedance_error = (
            abs(exceedance_rate - alpha) if np.isfinite(exceedance_rate) else alpha
        )
        try:
            ks_stat = float(kstest(exceedances, "genpareto", args=(xi, 0.0, beta)).statistic)
        except (ValueError, FloatingPointError):
            ks_stat = 1.0
        objective = exceedance_error + 0.10 * ks_stat
        calibration = EVTCalibration(
            threshold=threshold,
            method="evt_pot_gpd",
            alpha=alpha,
            tail_quantile=float(q),
            u=u,
            xi=float(xi),
            beta=float(beta),
            n_total=int(fit_values.size),
            n_exceed=int(exceedances.size),
            validation_exceedance_rate=exceedance_rate,
            validation_exceedance_error=exceedance_error,
            ks_stat=ks_stat,
        )
        if best is None or objective < best[0]:
            best = (objective, calibration)

    if best is None:
        return EVTCalibration(
            threshold=fallback,
            method="empirical_quantile_fallback",
            alpha=alpha,
            tail_quantile=tail_quantile,
            n_total=int(values.size),
        )
    return best[1]


def _days_since_flag(flag: pd.Series, product: pd.Series) -> np.ndarray:
    result = np.full(len(flag), np.nan, dtype=float)
    for _, indices in product.groupby(product, sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        last: int | None = None
        for position, row_index in enumerate(idx):
            if bool(flag.iloc[row_index]):
                last = position
                result[row_index] = 0.0
            elif last is not None:
                result[row_index] = float(position - last)
    return result


def build_demand_anomaly_profile(
    raw_df: pd.DataFrame,
    cfg: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build row-level and systemic anomaly scores from observed history.

    Every local score at date *t* uses only residual-distribution information
    from dates before *t*.  EVT calibration sees only the supplied historical
    frame, so callers must pass the fold-specific training slice during CV.
    """

    required = {"ProductId", "DateKey", "ProductAvailable"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required anomaly columns: {sorted(missing)}")

    work = raw_df.copy()
    work["DateKey"] = pd.to_datetime(work["DateKey"])
    work["Quantity"] = _safe_quantity(work)
    work = work.sort_values(["ProductId", "DateKey"]).reset_index(drop=True)
    work["known_event"] = _known_event_mask(work)
    work["expected_quantity"] = _causal_same_weekday_baseline(work)

    available = work["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
    valid = (
        available
        & np.isfinite(work["Quantity"].to_numpy(dtype=float))
        & (work["Quantity"].to_numpy(dtype=float) >= 0.0)
        & np.isfinite(work["expected_quantity"].to_numpy(dtype=float))
        & (work["expected_quantity"].to_numpy(dtype=float) >= 0.0)
    )
    residual = np.full(len(work), np.nan, dtype=float)
    residual[valid] = (
        np.log1p(work.loc[valid, "Quantity"].to_numpy(dtype=float))
        - np.log1p(work.loc[valid, "expected_quantity"].to_numpy(dtype=float))
    )
    work["anomaly_signed_residual"] = residual

    score, center, scale = _rolling_robust_score(
        work["anomaly_signed_residual"],
        work["ProductId"],
        window=int(_cfg(cfg, "anomaly_rolling_window", 180)),
        min_history=int(_cfg(cfg, "anomaly_min_history", 28)),
        scale_floor=float(_cfg(cfg, "anomaly_scale_floor", 0.10)),
    )
    work["anomaly_residual_center"] = center
    work["anomaly_residual_scale"] = scale
    work["anomaly_score"] = score

    score_order = work.sort_values(["DateKey", "ProductId"])["anomaly_score"]
    local_evt = calibrate_evt_threshold(
        score_order,
        alpha=float(_cfg(cfg, "anomaly_evt_alpha", 0.01)),
        tail_quantile=float(_cfg(cfg, "anomaly_evt_tail_quantile", 0.90)),
        min_exceedances=int(_cfg(cfg, "anomaly_evt_min_exceedances", 30)),
    )
    work["anomaly_flag"] = (
        np.isfinite(work["anomaly_score"].to_numpy(dtype=float))
        & work["anomaly_score"].ge(local_evt.threshold)
    )

    # Cross-product shock: the 90th percentile is intentionally used instead
    # of a mean, so several products must move abnormally before a day is
    # treated as systemic, while one isolated SKU cannot dominate it.
    daily = (
        work.groupby("DateKey", sort=True)["anomaly_score"]
        .quantile(0.90)
        .rename("systemic_anomaly_score")
        .reset_index()
    )
    systemic_evt = calibrate_evt_threshold(
        daily["systemic_anomaly_score"],
        alpha=float(_cfg(cfg, "anomaly_systemic_evt_alpha", 0.02)),
        tail_quantile=float(_cfg(cfg, "anomaly_systemic_tail_quantile", 0.85)),
        min_exceedances=max(12, int(_cfg(cfg, "anomaly_evt_min_exceedances", 30)) // 2),
    )
    daily["systemic_anomaly_flag"] = (
        np.isfinite(daily["systemic_anomaly_score"].to_numpy(dtype=float))
        & daily["systemic_anomaly_score"].ge(systemic_evt.threshold)
    )
    daily["systemic_anomaly_rate_28"] = (
        daily["systemic_anomaly_flag"].astype(float).shift(1)
        .rolling(28, min_periods=1).mean()
    )
    work = work.merge(daily, on="DateKey", how="left", validate="many_to_one")

    work["anomaly_rate_28"] = work.groupby("ProductId", sort=False)[
        "anomaly_flag"
    ].transform(lambda s: s.astype(float).shift(1).rolling(28, min_periods=1).mean())
    work["days_since_anomaly"] = _days_since_flag(
        work["anomaly_flag"], work["ProductId"]
    )

    threshold = max(local_evt.threshold, np.finfo(float).eps)
    excess = np.maximum(work["anomaly_score"].to_numpy(dtype=float) - threshold, 0.0)
    severity = np.divide(
        excess,
        threshold,
        out=np.zeros(len(work), dtype=float),
        where=np.isfinite(excess),
    )
    strength = float(_cfg(cfg, "anomaly_weight_strength", 1.0))
    min_weight = float(_cfg(cfg, "anomaly_min_weight", 0.20))
    max_weight = float(_cfg(cfg, "anomaly_max_weight", 2.00))
    policy = str(_cfg(cfg, "anomaly_weight_policy", "downweight")).lower()
    signed_residual = work["anomaly_signed_residual"].to_numpy(dtype=float)
    if policy == "downweight":
        weight = np.exp(-strength * severity)
        weight = np.clip(weight, min_weight, 1.0)
    elif policy == "negative_only":
        weight = np.ones(len(work), dtype=float)
        negative = np.isfinite(signed_residual) & (signed_residual < 0.0)
        weight[negative] = np.clip(
            np.exp(-strength * severity[negative]), min_weight, 1.0
        )
    elif policy == "hard_example":
        exponent = np.minimum(strength * severity, np.log(max(max_weight, 1.0)))
        weight = np.exp(exponent)
        weight = np.clip(weight, 1.0, max_weight)
    elif policy == "signed":
        weight = np.ones(len(work), dtype=float)
        negative = np.isfinite(signed_residual) & (signed_residual < 0.0)
        positive = np.isfinite(signed_residual) & (signed_residual > 0.0)
        weight[negative] = np.clip(
            np.exp(-strength * severity[negative]), min_weight, 1.0
        )
        positive_exponent = np.minimum(
            strength * severity[positive], np.log(max(max_weight, 1.0))
        )
        weight[positive] = np.clip(
            np.exp(positive_exponent), 1.0, max_weight
        )
    else:
        raise ValueError(
            "anomaly_weight_policy must be one of: "
            "downweight, negative_only, hard_example, signed"
        )
    weight[~np.isfinite(work["anomaly_score"].to_numpy(dtype=float))] = 1.0

    if bool(_cfg(cfg, "anomaly_protect_known_events", True)):
        explained_positive = work["known_event"] & work["anomaly_signed_residual"].gt(0.0)
        event_floor = float(_cfg(cfg, "anomaly_known_event_min_weight", 0.65))
        weight[explained_positive.to_numpy(dtype=bool)] = np.maximum(
            weight[explained_positive.to_numpy(dtype=bool)], event_floor
        )
    # Broad market shocks may be real regime changes.  Keep them visible to
    # the learner instead of deleting them as if they were corrupted labels.
    systemic_floor = float(_cfg(cfg, "anomaly_systemic_min_weight", 0.50))
    systemic_mask = work["systemic_anomaly_flag"].fillna(False).to_numpy(dtype=bool)
    weight[systemic_mask] = np.maximum(weight[systemic_mask], systemic_floor)
    weight[~available.to_numpy(dtype=bool)] = 1.0
    work["anomaly_weight"] = weight

    profile_columns = [
        "ProductId", "DateKey", "Quantity", "expected_quantity",
        "anomaly_signed_residual", "anomaly_residual_center",
        "anomaly_residual_scale", "anomaly_score", "anomaly_flag",
        "anomaly_rate_28", "days_since_anomaly", "known_event",
        "systemic_anomaly_score", "systemic_anomaly_flag",
        "systemic_anomaly_rate_28", "anomaly_weight",
    ]
    profile = work[profile_columns].copy()
    metadata = {
        "local_evt": local_evt.to_dict(),
        "systemic_evt": systemic_evt.to_dict(),
        "n_rows": int(len(profile)),
        "n_scored": int(np.isfinite(profile["anomaly_score"]).sum()),
        "n_local_anomalies": int(profile["anomaly_flag"].sum()),
        "n_systemic_days": int(daily["systemic_anomaly_flag"].sum()),
        "mean_raw_weight": float(profile["anomaly_weight"].mean()),
        "weight_policy": policy,
        "weight_min": float(profile["anomaly_weight"].min()),
        "weight_max": float(profile["anomaly_weight"].max()),
    }
    return profile, metadata


def attach_anomaly_origin_features(
    feature_df: pd.DataFrame,
    profile: pd.DataFrame,
) -> pd.DataFrame:
    """Attach history-state values that are known at each forecast origin."""

    columns = [
        "ProductId", "DateKey", "anomaly_score", "anomaly_flag",
        "anomaly_rate_28", "days_since_anomaly", "systemic_anomaly_score",
        "systemic_anomaly_flag", "systemic_anomaly_rate_28",
    ]
    lookup = profile[columns].rename(columns={
        "anomaly_score": "anomaly_score_lag0",
        "anomaly_flag": "anomaly_flag_lag0",
        "systemic_anomaly_score": "systemic_anomaly_score_lag0",
        "systemic_anomaly_flag": "systemic_anomaly_flag_lag0",
    })
    result = feature_df.merge(
        lookup,
        on=["ProductId", "DateKey"],
        how="left",
        validate="one_to_one",
    )
    for column in ANOMALY_ORIGIN_FEATURES:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def apply_anomaly_weights_to_panel(
    panel: pd.DataFrame,
    profile: pd.DataFrame,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """Multiply existing sample weights by target-date anomaly weights.

    ``TargetDateKey`` is a supervised-row key, not a model feature.  Using its
    observed label to construct a robust-training weight is valid as long as
    the profile itself was built only from the fold's training history.
    """

    if "TargetDateKey" not in panel.columns:
        raise ValueError("Panel must contain TargetDateKey")
    result = panel.copy()
    lookup = profile[[
        "ProductId", "DateKey", "anomaly_weight", "anomaly_score", "anomaly_flag"
    ]].rename(columns={
        "DateKey": "TargetDateKey",
        "anomaly_weight": "anomaly_weight_raw",
        "anomaly_score": "target_anomaly_score",
        "anomaly_flag": "target_anomaly_flag",
    })
    result = result.merge(
        lookup,
        on=["ProductId", "TargetDateKey"],
        how="left",
        validate="many_to_one",
    )
    base = pd.to_numeric(
        result.get("sample_weight", pd.Series(1.0, index=result.index)),
        errors="coerce",
    ).fillna(1.0).to_numpy(dtype=float)
    anomaly = pd.to_numeric(result["anomaly_weight_raw"], errors="coerce").fillna(1.0)
    combined = base * anomaly.to_numpy(dtype=float)
    if normalize and combined.size:
        mean_weight = float(np.mean(combined))
        if np.isfinite(mean_weight) and mean_weight > 0.0:
            combined = combined / mean_weight
    result["sample_weight"] = combined
    return result
