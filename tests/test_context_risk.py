from __future__ import annotations

import numpy as np
import pandas as pd

from context_risk import ContextRiskDetector


def _frame(n_days: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for product_id in range(1, 6):
        for date in dates:
            rows.append({
                "ProductId": product_id,
                "DateKey": date,
                "PriceLocalVat": 100.0 + product_id,
                "DiscountValueWebRelative": 0.0,
                "DiscountValueAppRelative": 0.0,
                "CampaignSubTypeWeb": -1,
                "CampaignSubTypeApp": -1,
                "IsSaleOrPromo": False,
                "ProductAvailable": True,
            })
    return pd.DataFrame(rows)


def test_unseen_extreme_context_scores_as_more_unusual():
    train = _frame()
    detector = ContextRiskDetector(n_estimators=80, max_samples=0.8).fit(train)
    ordinary = train.tail(10).copy()
    extreme = ordinary.copy()
    extreme["PriceLocalVat"] = 10000.0
    extreme["DiscountValueWebRelative"] = 15.0
    extreme["CampaignSubTypeWeb"] = 19
    extreme["IsSaleOrPromo"] = True
    ordinary_score = detector.score(ordinary)["context_risk_percentile"].mean()
    extreme_score = detector.score(extreme)["context_risk_percentile"].mean()
    assert extreme_score > ordinary_score
    assert 0.0 <= extreme_score <= 1.0
