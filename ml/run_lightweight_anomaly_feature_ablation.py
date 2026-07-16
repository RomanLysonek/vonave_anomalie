"""Low-memory direct residual ablation for origin-known anomaly features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from anomaly_detection import build_demand_anomaly_profile
from framework import Config, compute_baseline, compute_metrics, reindex_daily_calendar
from offline_parquet import read_parquet

DEV_ORIGINS = pd.to_datetime([
    "2023-01-10", "2024-06-20", "2024-11-29", "2025-02-10",
])
BASE_NUMERIC = [
    "log_baseline", "log_origin_qty", "log_roll7", "log_roll28",
    "horizon_scaled", "dow_sin", "dow_cos", "month_sin", "month_cos",
]
ANOMALY_NUMERIC = [
    "anomaly_score", "anomaly_rate_28", "log_days_since_anomaly",
    "systemic_anomaly_score", "systemic_anomaly_rate_28",
    "anomaly_flag", "systemic_anomaly_flag",
]
ANOMALY_STATE = [
    "anomaly_score", "anomaly_rate_28", "days_since_anomaly",
    "systemic_anomaly_score", "systemic_anomaly_rate_28",
    "anomaly_flag", "systemic_anomaly_flag",
]


def load_data() -> pd.DataFrame:
    train = read_parquet("data/train_data.parquet")
    train["Quantity"] = (train.QuantityApp + train.QuantityWeb).astype(float)
    return reindex_daily_calendar(train)


def state_frame(history: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    state = profile.copy().sort_values(["ProductId", "DateKey"])
    available = history[["ProductId", "DateKey", "ProductAvailable"]]
    state = state.merge(available, on=["ProductId", "DateKey"], how="left", validate="one_to_one")
    qty = state["Quantity"].where(state["ProductAvailable"].astype("boolean").fillna(False))
    state["origin_qty"] = qty
    state["roll7"] = qty.groupby(state["ProductId"]).transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    state["roll28"] = qty.groupby(state["ProductId"]).transform(
        lambda x: x.rolling(28, min_periods=1).mean()
    )
    keep = ["ProductId", "DateKey", "origin_qty", "roll7", "roll28", *ANOMALY_STATE]
    return state[keep].rename(columns={"DateKey": "OriginDateKey"})


def make_rows(history: pd.DataFrame, state: pd.DataFrame, fold_origin: pd.Timestamp,
              evaluation: bool) -> pd.DataFrame:
    pieces = []
    if evaluation:
        dates = pd.date_range(fold_origin + pd.Timedelta(days=1), periods=7)
        source = history[history.DateKey.isin(dates)].copy()
        for h, date in enumerate(dates, start=1):
            part = source[source.DateKey.eq(date)].copy()
            part["horizon"] = h
            part["OriginDateKey"] = fold_origin
            pieces.append(part)
    else:
        source = history[history.DateKey <= fold_origin].copy()
        # Preserve all history while avoiding the first month where the seasonal baseline is undefined.
        first_date = source.DateKey.min() + pd.Timedelta(days=35)
        source = source[source.DateKey >= first_date]
        for h in range(1, 8):
            part = source.copy()
            part["horizon"] = h
            part["OriginDateKey"] = part["DateKey"] - pd.Timedelta(days=h)
            part = part[part.OriginDateKey >= history.DateKey.min()]
            pieces.append(part)
    rows = pd.concat(pieces, ignore_index=True)
    rows = rows.merge(state, on=["ProductId", "OriginDateKey"], how="left", validate="many_to_one")
    target_keys = rows[["ProductId", "DateKey"]].copy()
    rows["baseline"] = compute_baseline(target_keys, history, "weighted_4321")
    available = rows["ProductAvailable"].astype("boolean").fillna(False).astype(bool)
    rows = rows[available & rows.Quantity.notna() & np.isfinite(rows.baseline)].copy()
    date = rows.DateKey
    rows["log_baseline"] = np.log1p(np.clip(rows.baseline, 0, None))
    rows["log_origin_qty"] = np.log1p(np.clip(pd.to_numeric(rows.origin_qty, errors="coerce"), 0, None))
    rows["log_roll7"] = np.log1p(np.clip(pd.to_numeric(rows.roll7, errors="coerce"), 0, None))
    rows["log_roll28"] = np.log1p(np.clip(pd.to_numeric(rows.roll28, errors="coerce"), 0, None))
    rows["horizon_scaled"] = rows.horizon / 7.0
    rows["dow_sin"] = np.sin(2*np.pi*date.dt.dayofweek/7)
    rows["dow_cos"] = np.cos(2*np.pi*date.dt.dayofweek/7)
    rows["month_sin"] = np.sin(2*np.pi*date.dt.month/12)
    rows["month_cos"] = np.cos(2*np.pi*date.dt.month/12)
    rows["log_days_since_anomaly"] = np.log1p(np.clip(
        pd.to_numeric(rows.days_since_anomaly, errors="coerce"), 0, 3650
    ))
    return rows


def design(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]):
    train_num = train[columns].apply(pd.to_numeric, errors="coerce")
    med = train_num.median().fillna(0.0)
    train_num = train_num.fillna(med)
    test_num = test[columns].apply(pd.to_numeric, errors="coerce").fillna(med)
    mean = train_num.mean()
    std = train_num.std(ddof=0).replace(0, 1).fillna(1)
    a = sparse.csr_matrix(((train_num-mean)/std).to_numpy(np.float32))
    b = sparse.csr_matrix(((test_num-mean)/std).to_numpy(np.float32))
    n_products = 30
    prod_a = sparse.csr_matrix((np.ones(len(train)), (np.arange(len(train)), train.ProductId.to_numpy(int)-1)), shape=(len(train), n_products))
    prod_b = sparse.csr_matrix((np.ones(len(test)), (np.arange(len(test)), test.ProductId.to_numpy(int)-1)), shape=(len(test), n_products))
    hor_a = sparse.csr_matrix((np.ones(len(train)), (np.arange(len(train)), train.horizon.to_numpy(int)-1)), shape=(len(train), 7))
    hor_b = sparse.csr_matrix((np.ones(len(test)), (np.arange(len(test)), test.horizon.to_numpy(int)-1)), shape=(len(test), 7))
    return sparse.hstack([a,prod_a,hor_a],format="csr"), sparse.hstack([b,prod_b,hor_b],format="csr")


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    X, Xt = design(train, test, columns)
    y = np.log1p(train.Quantity.to_numpy(float)) - np.log1p(train.baseline.to_numpy(float))
    model = Ridge(alpha=10.0, solver="lsqr", tol=1e-4)
    model.fit(X, y)
    residual = model.predict(Xt)
    return np.clip(np.expm1(residual + np.log1p(test.baseline.to_numpy(float))), 0, None)


def metric(rows: pd.DataFrame, pred: str) -> dict:
    return compute_metrics(rows.Quantity.to_numpy(float), rows[pred].to_numpy(float))


def main():
    out=Path('outputs/lightweight_anomaly_feature_ablation'); out.mkdir(parents=True,exist_ok=True)
    train=load_data(); max_date=train.DateKey.max()
    benchmark=pd.DatetimeIndex([max_date-pd.Timedelta(days=7*i) for i in range(1,5)])
    all_rows=[]; fold=[]
    for split,origins in [('development',DEV_ORIGINS),('benchmark',benchmark)]:
        for origin in origins:
            history=train[train.DateKey<=origin].copy()
            cfg=Config(); cfg.anomaly_mode='features'
            profile,_=build_demand_anomaly_profile(history,cfg)
            state=state_frame(history,profile)
            tr=make_rows(history,state,origin,False)
            # Evaluation actuals exist in the full dataset, while all predictors use history <= origin.
            eval_history=train[train.DateKey<=origin+pd.Timedelta(days=7)].copy()
            ev=make_rows(eval_history,state,origin,True)
            ev['pred_control_features']=fit_predict(tr,ev,BASE_NUMERIC)
            ev['pred_anomaly_features']=fit_predict(tr,ev,BASE_NUMERIC+ANOMALY_NUMERIC)
            ev['split']=split; ev['fold_origin']=origin
            for name in ['control_features','anomaly_features']:
                m=metric(ev,f'pred_{name}')
                fold.append({'split':split,'origin':str(origin.date()),'policy':name,**{k:m[k] for k in ['WAPE','MAE','BiasRatio','n']}})
            print(split,origin.date(),fold[-2]['WAPE'],fold[-1]['WAPE'])
            all_rows.append(ev[['ProductId','DateKey','Quantity','split','fold_origin','pred_control_features','pred_anomaly_features']])
    pred=pd.concat(all_rows,ignore_index=True)
    summary=[]
    for split in ['development','benchmark']:
        sub=pred[pred.split==split]
        for name in ['control_features','anomaly_features']:
            m=compute_metrics(sub.Quantity.to_numpy(float),sub[f'pred_{name}'].to_numpy(float))
            summary.append({'split':split,'policy':name,**{k:m[k] for k in ['WAPE','MAE','BiasRatio','n']}})
    s=pd.DataFrame(summary)
    c=s[s.policy=='control_features'].set_index('split'); a=s[s.policy=='anomaly_features'].set_index('split')
    dev=(c.loc['development','WAPE']-a.loc['development','WAPE'])/c.loc['development','WAPE']
    bench=(a.loc['benchmark','WAPE']-c.loc['benchmark','WAPE'])/c.loc['benchmark','WAPE']
    result={'test':'low-memory direct Ridge residual anomaly-feature ablation','summary':summary,'development_relative_improvement':float(dev),'benchmark_relative_change':float(bench),'passes_development_gate':bool(dev>=.002),'passes_benchmark_guard':bool(bench<=.02),'winner':'anomaly_features' if dev>=.002 and bench<=.02 else 'control_features'}
    pred.to_csv(out/'predictions.csv',index=False); pd.DataFrame(fold).to_csv(out/'fold_metrics.csv',index=False); s.to_csv(out/'summary.csv',index=False); (out/'result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
