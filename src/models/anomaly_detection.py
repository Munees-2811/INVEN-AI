"""
Fraud & anomaly detection.

Uses an Isolation Forest over engineered transaction features (quantity, price
deviation from product norm, revenue, discount depth, time-of-week) to flag
suspicious transactions — bulk theft, price-override fraud, returns abuse — and
combines it with deterministic rules for clear-cut cases (negative quantity,
near-zero price).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _engineer(sales: pd.DataFrame) -> pd.DataFrame:
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    prod_price = df.groupby("product_id")["unit_price"].transform("median")
    prod_qty = df.groupby("product_id")["quantity"].transform("median")
    prod_qty_std = df.groupby("product_id")["quantity"].transform("std").replace(0, 1).fillna(1)

    feat = pd.DataFrame(
        {
            "quantity": df["quantity"].astype(float),
            "price_dev": (df["unit_price"] - prod_price) / prod_price.replace(0, np.nan),
            "qty_z": (df["quantity"] - prod_qty) / prod_qty_std,
            "revenue": df["revenue"].astype(float),
            "discount": (prod_price - df["unit_price"]).clip(lower=0) / prod_price.replace(0, np.nan),
            "dow": df["date"].dt.dayofweek,
        }
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    return feat


def detect_anomalies(sales: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """Return the sales frame annotated with anomaly flags and scores."""
    df = sales.copy().reset_index(drop=True)
    feat = _engineer(df)

    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso.fit(feat)
    raw_score = iso.score_samples(feat)            # higher = more normal
    df["anomaly_score"] = (-raw_score).round(4)    # higher = more anomalous
    ml_flag = iso.predict(feat) == -1

    # deterministic rule overrides
    rule_flag = (
        (df["quantity"] <= 0)
        | (df["unit_price"] <= df.groupby("product_id")["unit_price"].transform("median") * 0.2)
        | (df["quantity"] > df.groupby("product_id")["quantity"].transform("median") * 6 + 5)
    )

    df["is_anomaly"] = (ml_flag | rule_flag).astype(int)
    df["anomaly_reason"] = np.select(
        [
            df["quantity"] <= 0,
            df["unit_price"] <= df.groupby("product_id")["unit_price"].transform("median") * 0.2,
            df["quantity"] > df.groupby("product_id")["quantity"].transform("median") * 6 + 5,
            ml_flag,
        ],
        ["negative/zero quantity", "abnormally low price", "quantity spike", "ML outlier pattern"],
        default="",
    )
    return df


def anomaly_summary(annotated: pd.DataFrame) -> dict:
    flagged = annotated[annotated["is_anomaly"] == 1]
    return {
        "total_transactions": len(annotated),
        "flagged": int(len(flagged)),
        "flagged_pct": round(len(flagged) / max(len(annotated), 1) * 100, 2),
        "value_at_risk": round(float(flagged["revenue"].abs().sum()), 2),
        "top_reasons": flagged["anomaly_reason"].value_counts().head(5).to_dict(),
    }
