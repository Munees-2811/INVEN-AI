"""
Automated data preprocessing & feature engineering.

This is the first stage of the MLOps pipeline: it cleans raw transactions,
removes obviously corrupt rows, builds a continuous daily demand series per
product, and engineers calendar / lag / rolling features used by the
forecasting and other supervised models.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PreprocessReport:
    """Summary of what the cleaning stage did — surfaced in the MLOps UI."""

    rows_in: int = 0
    rows_out: int = 0
    dropped_negative: int = 0
    dropped_duplicates: int = 0
    imputed_prices: int = 0
    clipped_outliers: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "dropped_negative": self.dropped_negative,
            "dropped_duplicates": self.dropped_duplicates,
            "imputed_prices": self.imputed_prices,
            "clipped_outliers": self.clipped_outliers,
            "notes": self.notes,
        }


def clean_sales(sales: pd.DataFrame, clip_outliers: bool = True) -> tuple[pd.DataFrame, PreprocessReport]:
    """Clean raw sales transactions, returning the frame and an audit report."""
    rep = PreprocessReport(rows_in=len(sales))
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 1. drop exact duplicate transactions
    before = len(df)
    df = df.drop_duplicates()
    rep.dropped_duplicates = before - len(df)

    # 2. negative / zero quantities are data-entry errors (kept separately for fraud model)
    neg_mask = df["quantity"] <= 0
    rep.dropped_negative = int(neg_mask.sum())
    df = df[~neg_mask].copy()

    # 3. impute missing / non-positive prices with product median
    bad_price = df["unit_price"].isna() | (df["unit_price"] <= 0)
    rep.imputed_prices = int(bad_price.sum())
    if bad_price.any():
        med = df.groupby("product_id")["unit_price"].transform("median")
        df.loc[bad_price, "unit_price"] = med[bad_price]
        df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)

    # 4. clip extreme per-product quantity outliers (winsorise at 99.5th pct)
    if clip_outliers:
        clipped = 0
        caps = df.groupby("product_id")["quantity"].transform(lambda s: s.quantile(0.995))
        over = df["quantity"] > caps
        clipped = int(over.sum())
        df.loc[over, "quantity"] = caps[over].astype(int)
        rep.clipped_outliers = clipped

    rep.rows_out = len(df)
    rep.notes.append(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df.reset_index(drop=True), rep


def daily_demand(sales: pd.DataFrame, product_id: str | None = None) -> pd.DataFrame:
    """Aggregate transactions into a continuous daily demand series.

    Returns a frame with a complete date index (zero-filled) per product.
    """
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    if product_id is not None:
        df = df[df["product_id"] == product_id]

    grp = (
        df.groupby(["product_id", "date"])
        .agg(quantity=("quantity", "sum"), revenue=("revenue", "sum"))
        .reset_index()
    )

    # reindex each product onto a complete daily calendar
    out = []
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    for pid, g in grp.groupby("product_id"):
        g = g.set_index("date").reindex(full_range)
        g["product_id"] = pid
        g["quantity"] = g["quantity"].fillna(0.0)
        g["revenue"] = g["revenue"].fillna(0.0)
        g.index.name = "date"
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True) if out else grp


def make_features(series: pd.DataFrame) -> pd.DataFrame:
    """Engineer calendar + lag + rolling features for a single product's series.

    Expects columns: date, quantity. Returns a feature matrix ready for a
    supervised forecaster (target column = ``quantity``).
    """
    df = series.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    # calendar features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # cyclical encodings
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # lag features
    for lag in (1, 7, 14, 28):
        df[f"lag_{lag}"] = df["quantity"].shift(lag)

    # rolling statistics (shifted to avoid leakage)
    for w in (7, 14, 28):
        df[f"roll_mean_{w}"] = df["quantity"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df["quantity"].shift(1).rolling(w).std()

    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLUMNS = [
    "dayofweek", "day", "month", "weekofyear", "is_weekend",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_std_7", "roll_mean_14", "roll_std_14",
    "roll_mean_28", "roll_std_28",
]
