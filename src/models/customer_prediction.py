"""
Customer purchase prediction.

Builds RFM (Recency, Frequency, Monetary) features per customer and trains a
gradient-boosted classifier to predict the probability that a customer will
purchase again within a horizon window. Also produces churn-risk segments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


@dataclass
class PurchaseModel:
    model: GradientBoostingClassifier
    features: list[str]
    auc: float


def build_rfm(sales: pd.DataFrame, horizon_days: int = 21) -> pd.DataFrame:
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["customer_id"] != "WALKIN"]  # anonymous walk-ins are not trackable
    cutoff = df["date"].max() - pd.Timedelta(days=horizon_days)

    hist = df[df["date"] <= cutoff]
    future = df[df["date"] > cutoff]

    snapshot = hist["date"].max()
    rfm = hist.groupby("customer_id").agg(
        recency=("date", lambda s: (snapshot - s.max()).days),
        frequency=("date", "nunique"),
        monetary=("revenue", "sum"),
        avg_basket=("revenue", "mean"),
        n_items=("quantity", "sum"),
    )
    rfm["purchased_next"] = rfm.index.isin(future["customer_id"].unique()).astype(int)
    return rfm.reset_index()


def train_purchase_model(sales: pd.DataFrame, horizon_days: int = 21) -> tuple[PurchaseModel, pd.DataFrame]:
    rfm = build_rfm(sales, horizon_days)
    features = ["recency", "frequency", "monetary", "avg_basket", "n_items"]
    X, y = rfm[features].fillna(0), rfm["purchased_next"]

    auc = float("nan")
    if y.nunique() > 1 and len(rfm) > 30:
        split = int(len(rfm) * 0.75)
        idx = rfm.sample(frac=1.0, random_state=42).index
        tr, te = idx[:split], idx[split:]
        m = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
        m.fit(X.loc[tr], y.loc[tr])
        try:
            auc = float(roc_auc_score(y.loc[te], m.predict_proba(X.loc[te])[:, 1]))
        except Exception:
            auc = float("nan")

    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
    model.fit(X, y)
    rfm["repurchase_proba"] = model.predict_proba(X)[:, 1].round(3)
    rfm["churn_risk"] = np.select(
        [rfm["repurchase_proba"] >= 0.6, rfm["repurchase_proba"] >= 0.3],
        ["low", "medium"],
        default="high",
    )
    return PurchaseModel(model=model, features=features, auc=auc), rfm


def predict_repurchase(model: PurchaseModel, rfm_row: dict) -> float:
    X = pd.DataFrame([{f: rfm_row.get(f, 0) for f in model.features}])
    return float(model.model.predict_proba(X)[0, 1])
