"""
Supplier performance analysis.

Scores every supplier on a weighted blend of on-time delivery, quality (defect
rate), lead-time reliability and price competitiveness — computed from actual
purchase-order history where available, falling back to the supplier master.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {
    "on_time": 0.35,
    "quality": 0.30,
    "lead_reliability": 0.20,
    "price": 0.15,
}


def supplier_scorecard(suppliers: pd.DataFrame, purchase_orders: pd.DataFrame | None = None) -> pd.DataFrame:
    df = suppliers.copy()

    if purchase_orders is not None and not purchase_orders.empty:
        agg = purchase_orders.groupby("supplier_id").agg(
            orders=("po_id", "count"),
            on_time_rate=("on_time", "mean"),
            avg_lead=("actual_lead_time", "mean"),
            lead_std=("actual_lead_time", "std"),
            defect_rate=("qty_defective", "sum"),
            total_qty=("qty_ordered", "sum"),
            total_value=("order_value", "sum"),
        )
        agg["defect_rate"] = (agg["defect_rate"] / agg["total_qty"]).fillna(0)
        df = df.merge(agg, on="supplier_id", how="left", suffixes=("", "_actual"))
        df["on_time_rate"] = df["on_time_rate_actual"].fillna(df["on_time_rate"])
        df["defect_rate"] = df["defect_rate_actual"].fillna(df["defect_rate"])
    else:
        df["orders"] = np.nan
        df["lead_std"] = df.get("lead_time_std", 1.0)
        df["total_value"] = np.nan

    # normalise sub-scores into 0..1 (higher is better)
    df["quality_score"] = 1 - df["defect_rate"].clip(0, 1)
    lead_std = df.get("lead_std", df.get("lead_time_std", 1.0)).fillna(1.0)
    df["lead_reliability"] = 1 / (1 + lead_std)
    df["price"] = df.get("price_competitiveness", 0.8).fillna(0.8)

    df["score"] = (
        WEIGHTS["on_time"] * df["on_time_rate"].fillna(0.8)
        + WEIGHTS["quality"] * df["quality_score"]
        + WEIGHTS["lead_reliability"] * df["lead_reliability"]
        + WEIGHTS["price"] * df["price"]
    )
    df["score_100"] = (df["score"] * 100).round(1)
    df["grade"] = np.select(
        [df["score_100"] >= 85, df["score_100"] >= 70, df["score_100"] >= 55],
        ["A", "B", "C"],
        default="D",
    )
    df["recommendation"] = np.where(
        df["score_100"] >= 70, "Preferred — keep / increase volume",
        np.where(df["score_100"] >= 55, "Acceptable — monitor", "At risk — review or replace"),
    )

    cols = [
        "supplier_id", "supplier_name", "score_100", "grade",
        "on_time_rate", "defect_rate", "avg_lead_time_days", "lead_reliability",
        "price", "orders", "total_value", "recommendation",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values("score_100", ascending=False).reset_index(drop=True)
