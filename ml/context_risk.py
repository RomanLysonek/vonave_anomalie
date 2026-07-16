"""Forecast-time covariate-shift risk for the known test-week context.

Demand reconstruction cannot be evaluated before the future sales arrive.
This detector therefore answers a different, operationally useful question:
"How unusual are the campaign, discount, price and calendar conditions
under which this forecast is being made?" Availability is excluded because
it is not supplied for the test horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = [
    "PriceLocalVat",
    "DiscountValueWebRelative",
    "DiscountValueAppRelative",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]
CATEGORICAL_COLUMNS = [
    "ProductId",
    "CampaignSubTypeWeb",
    "CampaignSubTypeApp",
    "IsSaleOrPromo",
]


def _calendar(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    date = pd.to_datetime(result["DateKey"])
    dow = date.dt.dayofweek
    month = date.dt.month
    result["day_of_week_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    result["day_of_week_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    result["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    result["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    for column in NUMERIC_COLUMNS:
        if column not in result:
            result[column] = np.nan
    for column in CATEGORICAL_COLUMNS:
        if column not in result:
            result[column] = "missing"
        elif column == "IsSaleOrPromo":
            result[column] = (
                result[column].astype("boolean").astype("string").fillna("missing")
            )
        else:
            numeric = pd.to_numeric(result[column], errors="coerce").astype("Int64")
            result[column] = numeric.astype("string").fillna("missing")
    return result


@dataclass
class ContextRiskDetector:
    """Isolation-forest novelty score with empirical percentile calibration."""

    random_state: int = 42
    n_estimators: int = 300
    max_samples: int | float = 0.75

    def __post_init__(self) -> None:
        preprocessor = ColumnTransformer([
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_COLUMNS,
            ),
        ])
        detector = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("detector", detector),
        ])
        self._train_scores: np.ndarray | None = None
        self._numeric_median: pd.Series | None = None
        self._numeric_scale: pd.Series | None = None
        self._category_frequency: dict[str, pd.Series] = {}
        self._n_train = 0

    def _raw_score(self, features: pd.DataFrame) -> np.ndarray:
        if self._numeric_median is None or self._numeric_scale is None:
            raise RuntimeError("ContextRiskDetector must be fitted before score")
        isolation = -self.pipeline.decision_function(features)
        numeric = features[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
        z = (numeric - self._numeric_median) / self._numeric_scale
        numeric_rarity = np.log1p(np.nanmax(np.abs(z.to_numpy(dtype=float)), axis=1))
        numeric_rarity = np.nan_to_num(numeric_rarity, nan=0.0, posinf=10.0)

        categorical_rarity = np.zeros(len(features), dtype=float)
        floor = 1.0 / max(self._n_train + 1, 2)
        for column in CATEGORICAL_COLUMNS:
            frequency = self._category_frequency[column]
            probs = features[column].astype(str).map(frequency).fillna(floor).to_numpy(dtype=float)
            categorical_rarity = np.maximum(categorical_rarity, -np.log(np.clip(probs, floor, 1.0)))
        return np.asarray(isolation, dtype=float) + 0.10 * numeric_rarity + 0.05 * categorical_rarity

    def fit(self, frame: pd.DataFrame) -> "ContextRiskDetector":
        features = _calendar(frame)
        self.pipeline.fit(features)
        numeric = features[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
        self._numeric_median = numeric.median()
        q25 = numeric.quantile(0.25)
        q75 = numeric.quantile(0.75)
        scale = (q75 - q25) / 1.349
        fallback = numeric.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        self._numeric_scale = scale.where(scale > 1e-8, fallback).fillna(1.0)
        self._n_train = len(features)
        self._category_frequency = {
            column: features[column].astype(str).value_counts(normalize=True)
            for column in CATEGORICAL_COLUMNS
        }
        raw = self._raw_score(features)
        self._train_scores = np.sort(np.asarray(raw, dtype=float))
        return self

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._train_scores is None:
            raise RuntimeError("ContextRiskDetector must be fitted before score")
        features = _calendar(frame)
        raw = self._raw_score(features)
        # Right-sided empirical CDF: 1.0 is more unusual than every training row.
        rank = np.searchsorted(self._train_scores, raw, side="right")
        percentile = rank / max(len(self._train_scores), 1)
        result = frame[["ProductId", "DateKey"]].copy()
        result["context_risk_raw"] = raw
        result["context_risk_percentile"] = percentile
        result["context_shift_flag"] = percentile >= 0.99
        return result
