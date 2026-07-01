"""
Dynamic price recommendations.

Primary engine: a pooled **XGBoost demand-response regressor** that learns how
daily unit demand reacts to the *relative* price (price ÷ product's typical
price), promotions and calendar, across the whole catalogue (so it has real
sample size instead of a handful of points per SKU). The recommender sweeps a
guard-railed price grid, predicts demand at each price with the model, and picks
the profit-maximising price.

Fallback: when the model is unavailable or a product is unseen, it falls back to
the transparent log-log **elasticity** regression on that product's
(price, quantity) history — keeping the feature usable fully offline and on
cold-start SKUs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models._boost import backend_label, boosted_regressor


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


# --------------------------------------------------------------------------- #
# Pooled XGBoost demand-response model
# --------------------------------------------------------------------------- #
_PRICE_FEATURES = ["price_rel", "unit_price", "on_promo", "dayofweek", "month",
                   "base_demand", "unit_cost"]


@dataclass
class PriceResponseModel:
    model: object = None                          # fitted regressor | None
    ctx: dict = field(default_factory=dict)       # product_id -> context features
    r2: float = float("nan")
    backend: str = backend_label()

    def demand_at(self, product_id: str, price: float) -> float | None:
        """Predicted daily units for a product at a candidate price."""
        if self.model is None or product_id not in self.ctx:
            return None
        c = self.ctx[product_id]
        med = c["median_price"] or price or 1.0
        row = pd.DataFrame([{
            "price_rel": price / med if med else 1.0,
            "unit_price": price,
            "on_promo": 0,
            "dayofweek": 2,          # representative mid-week day
            "month": c["month"],
            "base_demand": c["base_demand"],
            "unit_cost": c["unit_cost"],
        }])[_PRICE_FEATURES]
        return float(np.clip(self.model.predict(row)[0], 0, None))


def train_price_model(sales: pd.DataFrame, products: pd.DataFrame) -> PriceResponseModel:
    """Train the pooled demand-response regressor over the full catalogue."""
    df = sales.copy()
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
    if len(df) < 100:
        return PriceResponseModel()

    df["date"] = pd.to_datetime(df["date"])
    daily = (df.groupby(["product_id", "date"])
             .agg(quantity=("quantity", "sum"),
                  unit_price=("unit_price", "mean"),
                  on_promo=("on_promo", "max") if "on_promo" in df else ("quantity", "size"))
             .reset_index())
    med = df.groupby("product_id")["unit_price"].median()
    prod = products.set_index("product_id")
    daily["median_price"] = daily["product_id"].map(med)
    daily["price_rel"] = daily["unit_price"] / daily["median_price"].replace(0, np.nan)
    daily["dayofweek"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["base_demand"] = daily["product_id"].map(prod.get("base_daily_demand", pd.Series(dtype=float))).fillna(
        daily["quantity"])
    daily["unit_cost"] = daily["product_id"].map(prod.get("unit_cost", pd.Series(dtype=float))).fillna(0.0)
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna(subset=["price_rel"])
    if "on_promo" not in daily:
        daily["on_promo"] = 0
    daily["on_promo"] = daily["on_promo"].astype(int)

    X, y = daily[_PRICE_FEATURES], daily["quantity"]
    # Economic guardrail: demand must be NON-INCREASING in price (price_rel,
    # unit_price → -1) and non-decreasing in base_demand (+1). Without this a
    # flexible tree model can learn a flat/rising demand-vs-price curve and then
    # recommend "always raise price". Ignored automatically on the sklearn
    # fallback (which has no monotone support).
    # Order matches _PRICE_FEATURES: price_rel, unit_price, on_promo, dow, month, base_demand, unit_cost
    mono = (-1, -1, 0, 0, 0, 1, 0)
    model = boosted_regressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                              monotone_constraints=mono)
    model.fit(X, y)
    pred = np.clip(model.predict(X), 0, None)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else None

    last_month = int(daily["month"].iloc[-1]) if len(daily) else 1
    ctx = {}
    for pid, g in daily.groupby("product_id"):
        ctx[pid] = {
            "median_price": float(med.get(pid, g["unit_price"].median())),
            "base_demand": float(g["base_demand"].median()),
            "unit_cost": float(g["unit_cost"].median()),
            "month": last_month,
        }
    return PriceResponseModel(model=model, ctx=ctx, r2=round(r2, 3) if r2 is not None else float("nan"))


def _recommend_price_xgb(model: PriceResponseModel, product_id: str, unit_cost: float,
                         current_price: float, max_change: float, elasticity) -> dict | None:
    """Profit-maximising price using the XGBoost demand-response model."""
    lo, hi = current_price * (1 - max_change), current_price * (1 + max_change)
    grid = np.linspace(lo, hi, 60)
    demand = np.array([model.demand_at(product_id, float(p)) for p in grid], dtype=float)
    if np.isnan(demand).any() or demand.sum() <= 0:
        return None
    profits = (grid - unit_cost) * demand
    cur_demand = model.demand_at(product_id, current_price) or 0.0
    cur_profit = (current_price - unit_cost) * cur_demand
    best_idx = int(np.argmax(profits))
    best_price = float(grid[best_idx])
    change = (profits[best_idx] - cur_profit) / cur_profit * 100 if cur_profit > 0 else 0.0
    direction = "increase" if best_price > current_price else "decrease"
    confidence = "high" if (model.r2 == model.r2 and model.r2 > 0.5) else "medium"
    return {
        "product_id": product_id,
        "current_price": round(current_price, 2),
        "recommended_price": round(best_price, 2),
        "expected_profit_change_pct": round(float(change), 2),
        "elasticity": elasticity,
        "confidence": confidence,
        "method": "xgboost",
        "rationale": f"XGBoost demand-response: {direction} price toward profit-maximising point "
                     f"(model R²={model.r2}).",
    }


def recommend_price(
    sales: pd.DataFrame,
    product_id: str,
    unit_cost: float,
    current_price: float,
    max_change: float = 0.20,
    model: PriceResponseModel | None = None,
) -> dict:
    """Recommend a profit-maximising price within ±max_change of the current price.

    Uses the pooled XGBoost demand-response ``model`` when supplied; otherwise
    falls back to the per-product log-log elasticity estimate.
    """
    est = estimate_elasticity(sales, product_id)
    elasticity = est.get("elasticity")

    # Primary path: XGBoost demand-response model
    if model is not None:
        xgb_rec = _recommend_price_xgb(model, product_id, unit_cost, current_price,
                                       max_change, elasticity)
        if xgb_rec is not None:
            return xgb_rec

    if elasticity is None or elasticity >= 0:
        # not enough signal / non-economic elasticity → keep price, suggest margin check
        return {
            "product_id": product_id,
            "current_price": current_price,
            "recommended_price": current_price,
            "expected_profit_change_pct": 0.0,
            "elasticity": elasticity,
            "confidence": "low",
            "method": "hold",
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
        "method": "elasticity",
        "rationale": f"Elasticity {elasticity:.2f}: {direction} price toward profit-maximising point.",
    }


def price_recommendations_table(
    sales: pd.DataFrame,
    products: pd.DataFrame,
    model: PriceResponseModel | None = None,
) -> pd.DataFrame:
    """Whole-catalogue price recommendations.

    Trains the pooled XGBoost demand-response model once (unless one is passed
    in) and reuses it across every product — avoiding a redundant per-SKU fit.
    """
    if model is None:
        model = train_price_model(sales, products)
    rows = []
    for _, p in products.iterrows():
        rec = recommend_price(
            sales, p["product_id"], float(p["unit_cost"]), float(p["unit_price"]),
            model=model,
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
                "method": rec.get("method", "elasticity"),
            }
        )
    return pd.DataFrame(rows).sort_values("expected_profit_change_pct", ascending=False).reset_index(drop=True)
