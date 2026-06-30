"""
AI sales trend analysis.

Decomposes revenue into trend / seasonality, computes growth momentum, ranks
fast- and slow-movers, and runs a simple ABC (Pareto) analysis to tell an MSME
owner where the money actually comes from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def revenue_timeseries(sales: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    ts = (
        df.set_index("date")
        .resample(freq)
        .agg(revenue=("revenue", "sum"), units=("quantity", "sum"), profit=("profit", "sum"))
        .reset_index()
    )
    ts["revenue_ma4"] = ts["revenue"].rolling(4, min_periods=1).mean()
    return ts


def growth_metrics(sales: pd.DataFrame, window_days: int = 30) -> dict:
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    end = df["date"].max()
    cur = df[df["date"] > end - pd.Timedelta(days=window_days)]
    prev = df[
        (df["date"] <= end - pd.Timedelta(days=window_days))
        & (df["date"] > end - pd.Timedelta(days=2 * window_days))
    ]
    cur_rev, prev_rev = cur["revenue"].sum(), prev["revenue"].sum()
    growth = ((cur_rev - prev_rev) / prev_rev * 100) if prev_rev > 0 else None
    return {
        "window_days": window_days,
        "current_revenue": round(float(cur_rev), 2),
        "previous_revenue": round(float(prev_rev), 2),
        "revenue_growth_pct": round(growth, 2) if growth is not None else None,
        "current_units": int(cur["quantity"].sum()),
        "current_profit": round(float(cur["profit"].sum()), 2),
        "avg_order_value": round(float(cur["revenue"].mean() or 0), 2),
    }


def product_momentum(sales: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Compare recent vs prior window per product to flag rising / falling SKUs."""
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    end = df["date"].max()
    cur = df[df["date"] > end - pd.Timedelta(days=window_days)]
    prev = df[
        (df["date"] <= end - pd.Timedelta(days=window_days))
        & (df["date"] > end - pd.Timedelta(days=2 * window_days))
    ]
    cur_u = cur.groupby("product_id")["quantity"].sum()
    prev_u = prev.groupby("product_id")["quantity"].sum()
    names = df.groupby("product_id")["product_name"].first()
    out = pd.DataFrame({"current_units": cur_u, "previous_units": prev_u}).fillna(0)
    out["product_name"] = names
    out["change_pct"] = np.where(
        out["previous_units"] > 0,
        (out["current_units"] - out["previous_units"]) / out["previous_units"] * 100,
        np.nan,
    )
    out["trend"] = np.select(
        [out["change_pct"] > 10, out["change_pct"] < -10],
        ["rising", "falling"],
        default="stable",
    )
    return out.reset_index().sort_values("change_pct", ascending=False)


def abc_analysis(sales: pd.DataFrame) -> pd.DataFrame:
    """Classic ABC / Pareto classification by revenue contribution."""
    rev = (
        sales.groupby(["product_id", "product_name"])["revenue"].sum().sort_values(ascending=False).reset_index()
    )
    total = rev["revenue"].sum()
    rev["revenue_share"] = rev["revenue"] / total * 100
    rev["cumulative_share"] = rev["revenue_share"].cumsum()
    rev["abc_class"] = np.select(
        [rev["cumulative_share"] <= 80, rev["cumulative_share"] <= 95],
        ["A", "B"],
        default="C",
    )
    return rev


def category_breakdown(sales: pd.DataFrame) -> pd.DataFrame:
    return (
        sales.groupby("category")
        .agg(revenue=("revenue", "sum"), units=("quantity", "sum"), profit=("profit", "sum"))
        .sort_values("revenue", ascending=False)
        .reset_index()
    )
