"""
Smart reorder recommendations + overstock / understock detection.

Combines the demand forecast with classic inventory-control theory:
  * safety stock  = z(service level) * σ_demand * sqrt(lead time)
  * reorder point = demand during lead time + safety stock
  * EOQ-style order quantity from forecasted demand and review period

Then classifies every SKU as understocked / healthy / overstocked based on
days-of-cover versus configurable thresholds.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from config import (
    DEFAULT_LEAD_TIME_DAYS,
    DEFAULT_REVIEW_PERIOD_DAYS,
    DEFAULT_SERVICE_LEVEL,
    HOLDING_COST_RATE,
    OVERSTOCK_DAYS_THRESHOLD,
    UNDERSTOCK_DAYS_THRESHOLD,
)
from src.data.preprocessing import daily_demand
from src.utils.stats import z_for_service_level


def compute_reorder(
    daily_mean: float,
    daily_std: float,
    current_stock: float,
    unit_cost: float,
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> dict:
    """Return reorder point, safety stock and a recommended order quantity."""
    z = z_for_service_level(service_level)
    lead = max(lead_time_days, 1)

    safety_stock = z * daily_std * math.sqrt(lead)
    demand_during_lead = daily_mean * lead
    reorder_point = demand_during_lead + safety_stock

    # order-up-to level covers lead time + review period
    order_up_to = daily_mean * (lead + review_period_days) + safety_stock
    recommended_qty = max(0.0, order_up_to - current_stock)

    days_of_cover = (current_stock / daily_mean) if daily_mean > 0 else float("inf")
    needs_reorder = current_stock <= reorder_point

    holding_cost = current_stock * unit_cost * HOLDING_COST_RATE / 365 * max(days_of_cover if np.isfinite(days_of_cover) else 0, 0)

    return {
        "safety_stock": round(safety_stock, 1),
        "reorder_point": round(reorder_point, 1),
        "order_up_to_level": round(order_up_to, 1),
        "recommended_order_qty": int(math.ceil(recommended_qty)),
        "days_of_cover": round(days_of_cover, 1) if np.isfinite(days_of_cover) else None,
        "needs_reorder": bool(needs_reorder),
        "estimated_holding_cost": round(float(holding_cost), 2),
    }


def classify_stock(days_of_cover: float | None) -> str:
    if days_of_cover is None:
        return "no_demand"
    if days_of_cover < UNDERSTOCK_DAYS_THRESHOLD:
        return "understock"
    if days_of_cover > OVERSTOCK_DAYS_THRESHOLD:
        return "overstock"
    return "healthy"


def _recent_demand_stats(sales: pd.DataFrame, lookback_days: int = 56) -> pd.DataFrame:
    """Fast, vectorized recent daily-demand mean/std for every product.

    Used for the catalogue-wide reorder table where a demand *rate* (not a full
    forecast curve) is what inventory theory needs. The expensive gradient-boosted
    forecaster is reserved for the single-product Forecasting page.
    """
    series = daily_demand(sales)  # zero-filled daily series for all products
    series["date"] = pd.to_datetime(series["date"])
    cutoff = series["date"].max() - pd.Timedelta(days=lookback_days)
    recent = series[series["date"] > cutoff]
    # EWMA-weighted recency would be ideal; a trailing window mean/std is fast & robust.
    stats = recent.groupby("product_id")["quantity"].agg(
        daily_mean="mean", daily_std="std"
    ).reset_index()
    stats["daily_std"] = stats["daily_std"].fillna(0.0)
    return stats


def inventory_health(
    sales: pd.DataFrame,
    products: pd.DataFrame,
    suppliers: pd.DataFrame | None = None,
    horizon: int = 30,
) -> pd.DataFrame:
    """Full SKU-level reorder + health table across the catalogue (fast path)."""
    lead_lookup = {}
    if suppliers is not None:
        lead_lookup = suppliers.set_index("supplier_id")["avg_lead_time_days"].to_dict()

    stats = _recent_demand_stats(sales).set_index("product_id")

    rows = []
    for _, p in products.iterrows():
        s = stats.loc[p["product_id"]] if p["product_id"] in stats.index else None
        daily_mean = float(s["daily_mean"]) if s is not None else 0.0
        daily_std = max(float(s["daily_std"]) if s is not None else 0.0, 1e-6)
        lead = int(lead_lookup.get(p.get("supplier_id"), DEFAULT_LEAD_TIME_DAYS))

        ro = compute_reorder(
            daily_mean=daily_mean,
            daily_std=daily_std,
            current_stock=float(p["current_stock"]),
            unit_cost=float(p["unit_cost"]),
            lead_time_days=lead,
        )
        status = classify_stock(ro["days_of_cover"])
        rows.append(
            {
                "product_id": p["product_id"],
                "product_name": p["product_name"],
                "category": p["category"],
                "current_stock": int(p["current_stock"]),
                "forecast_daily_demand": round(daily_mean, 2),
                "lead_time_days": lead,
                "reorder_point": ro["reorder_point"],
                "safety_stock": ro["safety_stock"],
                "recommended_order_qty": ro["recommended_order_qty"],
                "days_of_cover": ro["days_of_cover"],
                "status": status,
                "needs_reorder": ro["needs_reorder"],
                "unit_cost": float(p["unit_cost"]),
                "tied_capital": round(float(p["current_stock"]) * float(p["unit_cost"]), 2),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["needs_reorder", "days_of_cover"], ascending=[False, True]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Trend-aware reorder plan + purchase-order drafting
# --------------------------------------------------------------------------- #
# How strongly demand momentum scales the order: half of the observed change,
# capped at ±50% momentum → factor bounded to [0.75, 1.25].
TREND_DAMPING = 0.5
TREND_CAP_PCT = 50.0


def reorder_plan(
    sales: pd.DataFrame,
    products: pd.DataFrame,
    suppliers: pd.DataFrame | None = None,
    window_days: int = 30,
) -> pd.DataFrame:
    """Reorder plan that adjusts quantities to the demand *trend*.

    Merges the rule-based reorder table (:func:`inventory_health`) with recent
    product momentum (:func:`sales_trends.product_momentum`): rising SKUs get an
    upsized suggested order, falling SKUs a downsized one, so replenishment
    follows where demand is heading rather than only where it has been. The
    adjustment is damped and capped to stay conservative.
    """
    from src.models.sales_trends import product_momentum

    health = inventory_health(sales, products, suppliers)
    mom = product_momentum(sales, window_days)[["product_id", "change_pct", "trend"]]
    plan = health.merge(mom, on="product_id", how="left")
    plan["trend"] = plan["trend"].fillna("stable")
    plan["change_pct"] = plan["change_pct"].fillna(0.0)

    factor = 1.0 + plan["change_pct"].clip(-TREND_CAP_PCT, TREND_CAP_PCT) / 100.0 * TREND_DAMPING
    plan["trend_factor"] = factor.round(2)
    plan["suggested_order_qty"] = (
        np.ceil(plan["recommended_order_qty"] * factor).clip(lower=0).astype(int)
    )
    return plan.sort_values(["needs_reorder", "days_of_cover"],
                            ascending=[False, True]).reset_index(drop=True)


def build_purchase_orders(
    plan: pd.DataFrame,
    products: pd.DataFrame,
    suppliers: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn a reorder plan into per-supplier draft purchase orders.

    Returns ``(lines, summary)``: one order line per SKU with a positive
    quantity, and a per-supplier roll-up with total value and a check against
    the supplier's minimum order value (so an MSME owner sees at a glance which
    drafts are ready to send and which need topping up).
    """
    qty_col = "suggested_order_qty" if "suggested_order_qty" in plan.columns else "recommended_order_qty"
    sel = plan[plan[qty_col] > 0].copy()
    if sel.empty:
        return (pd.DataFrame(columns=["supplier_id", "product_id", "product_name",
                                      "order_qty", "unit_cost", "line_value"]),
                pd.DataFrame(columns=["supplier_id", "supplier_name", "lines",
                                      "total_units", "order_value", "min_order_value",
                                      "meets_minimum"]))

    prod = products.set_index("product_id")
    sel["supplier_id"] = sel["product_id"].map(prod["supplier_id"])
    lines = sel[["supplier_id", "product_id", "product_name", qty_col, "unit_cost"]].rename(
        columns={qty_col: "order_qty"})
    lines["line_value"] = (lines["order_qty"] * lines["unit_cost"]).round(2)
    lines = lines.sort_values(["supplier_id", "line_value"],
                              ascending=[True, False]).reset_index(drop=True)

    summary = lines.groupby("supplier_id").agg(
        lines=("product_id", "count"),
        total_units=("order_qty", "sum"),
        order_value=("line_value", "sum"),
    ).reset_index()
    if suppliers is not None:
        sup = suppliers.set_index("supplier_id")
        summary["supplier_name"] = summary["supplier_id"].map(sup.get("supplier_name"))
        summary["min_order_value"] = summary["supplier_id"].map(
            sup.get("min_order_value", pd.Series(dtype=float))).fillna(0.0)
    else:
        summary["supplier_name"] = summary["supplier_id"]
        summary["min_order_value"] = 0.0
    summary["order_value"] = summary["order_value"].round(2)
    summary["meets_minimum"] = summary["order_value"] >= summary["min_order_value"]
    cols = ["supplier_id", "supplier_name", "lines", "total_units",
            "order_value", "min_order_value", "meets_minimum"]
    return lines, summary[cols].sort_values("order_value", ascending=False).reset_index(drop=True)
