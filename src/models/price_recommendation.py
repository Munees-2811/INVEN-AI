"""
Dynamic price recommendations.

Estimates per-product price elasticity from historical (price, quantity)
observations via a log-log linear regression, then searches for the price that
maximises expected profit subject to sane guardrails around the current price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_elasticity(sales: pd.DataFrame, product_id: str) -> dict:
    df = sales[sales["product_id"] == product_id].copy()
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
    if df["unit_price"].nunique() < 3 or len(df) < 10:
        return {"elasticity": None, "r2": None, "n_points": len(df)}

    # aggregate to price points to reduce noise
    g = df.groupby("unit_price").agg(quantity=("quantity", "mean")).reset_index()
    logp = np.log(g["unit_price"].to_numpy())
    logq = np.log(g["quantity"].to_numpy())
    A = np.vstack([logp, np.ones_like(logp)]).T
    coef, *_ = np.linalg.lstsq(A, logq, rcond=None)
    slope = float(coef[0])
    pred = A @ coef
    ss_res = float(np.sum((logq - pred) ** 2))
    ss_tot = float(np.sum((logq - logq.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return {"elasticity": round(slope, 3), "intercept": float(coef[1]), "r2": round(r2, 3) if r2 else None,
            "n_points": len(df)}


def recommend_price(
    sales: pd.DataFrame,
    product_id: str,
    unit_cost: float,
    current_price: float,
    max_change: float = 0.20,
) -> dict:
    """Recommend a profit-maximising price within ±max_change of the current price."""
    est = estimate_elasticity(sales, product_id)
    elasticity = est.get("elasticity")

    if elasticity is None or elasticity >= 0:
        # not enough signal / non-economic elasticity → keep price, suggest margin check
        return {
            "product_id": product_id,
            "current_price": current_price,
            "recommended_price": current_price,
            "expected_profit_change_pct": 0.0,
            "elasticity": elasticity,
            "confidence": "low",
            "rationale": "Insufficient price variation to estimate elasticity; holding price.",
        }

    intercept = est["intercept"]
    lo, hi = current_price * (1 - max_change), current_price * (1 + max_change)
    grid = np.linspace(lo, hi, 60)

    def profit(price: float) -> float:
        demand = np.exp(intercept + elasticity * np.log(price))
        return (price - unit_cost) * demand

    cur_profit = profit(current_price)
    profits = np.array([profit(p) for p in grid])
    best_idx = int(np.argmax(profits))
    best_price = float(grid[best_idx])
    change = (profits[best_idx] - cur_profit) / cur_profit * 100 if cur_profit > 0 else 0.0

    confidence = "high" if (est.get("r2") or 0) > 0.5 else "medium"
    direction = "increase" if best_price > current_price else "decrease"
    return {
        "product_id": product_id,
        "current_price": round(current_price, 2),
        "recommended_price": round(best_price, 2),
        "expected_profit_change_pct": round(float(change), 2),
        "elasticity": elasticity,
        "confidence": confidence,
        "rationale": f"Elasticity {elasticity:.2f}: {direction} price toward profit-maximising point.",
    }


def price_recommendations_table(sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, p in products.iterrows():
        rec = recommend_price(
            sales, p["product_id"], float(p["unit_cost"]), float(p["unit_price"])
        )
        rows.append(
            {
                "product_id": p["product_id"],
                "product_name": p["product_name"],
                "category": p["category"],
                "current_price": rec["current_price"],
                "recommended_price": rec["recommended_price"],
                "price_change_pct": round((rec["recommended_price"] - rec["current_price"]) / rec["current_price"] * 100, 1)
                if rec["current_price"] else 0.0,
                "expected_profit_change_pct": rec["expected_profit_change_pct"],
                "elasticity": rec["elasticity"],
                "confidence": rec["confidence"],
            }
        )
    return pd.DataFrame(rows).sort_values("expected_profit_change_pct", ascending=False).reset_index(drop=True)
