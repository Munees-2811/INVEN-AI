"""
Inventory-risk meta-model + smart-reordering hybrid.

This module sits at the *top* of the inventory prediction stack and consumes the
outputs of the lower models — it trains no base features of its own beyond what
those models emit. Dependency flow:

    demand ─▶ stock_risk (stockout / understock / overstock probabilities)
                   │
                   ├─▶ inventory_risk (meta XGBoost classifier)  → composite risk tier
                   └─▶ reorder_hybrid (rule reorder point ⊕ stockout proba) → action

* ``train_inventory_risk`` — an XGBoost **meta-classifier** (stacking) that maps
  the three risk probabilities (+ demand volatility and lead time) to a single
  composite risk tier: low / medium / high. Because it only reads model outputs,
  it stays independent of raw feature plumbing (clean separation of concerns).

* ``reorder_hybrid`` — Smart Reordering is deliberately a **hybrid**: the
  deterministic reorder-point / EOQ rule (``inventory.compute_reorder``) decides
  *whether* to reorder and *how much*, while the stockout classifier's
  probability sets the *urgency/priority*. No separate reorder model is trained
  — that would duplicate the stockout classifier — so the two are composed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models._boost import backend_label, boosted_classifier
from src.models.stock_risk import StockRiskModels, build_risk_features, stock_risk_table

META_FEATURES = ["p_stockout", "p_understock", "p_overstock", "demand_cv", "lead_time_days"]
RISK_TIERS = ["low", "medium", "high"]


@dataclass
class InventoryRiskModel:
    model: object = None                 # fitted meta classifier | None (degenerate)
    classes: list = field(default_factory=list)
    fallback_tier: str = "low"
    features: list = field(default_factory=lambda: list(META_FEATURES))
    accuracy: float = float("nan")
    backend: str = backend_label()


def _composite_label(row: pd.Series) -> str:
    """Derive the ground-truth composite tier from base risk signals."""
    if row["p_stockout"] >= 0.5 or row["pred_stockout"] == 1:
        return "high"
    if row["pred_understock"] == 1 or row["pred_overstock"] == 1:
        return "medium"
    return "low"


def train_inventory_risk(
    sales: pd.DataFrame,
    products: pd.DataFrame,
    suppliers: pd.DataFrame | None = None,
    risk_bundle: StockRiskModels | None = None,
) -> tuple[InventoryRiskModel, pd.DataFrame]:
    """Train the composite inventory-risk meta-classifier from stock-risk outputs.

    Accepts an already-fitted ``risk_bundle`` to avoid retraining the base models
    (dependency reuse); otherwise trains them via ``train_stock_risk``.
    """
    feat = build_risk_features(sales, products, suppliers)
    if risk_bundle is None:
        from src.models.stock_risk import train_stock_risk
        risk_bundle, _ = train_stock_risk(sales, products, suppliers)

    risk_tbl = stock_risk_table(risk_bundle, feat)
    # join meta inputs: base probabilities + a couple of context features
    meta = risk_tbl.merge(
        feat[["product_id", "demand_cv", "lead_time_days"]], on="product_id", how="left"
    )
    meta["tier"] = meta.apply(_composite_label, axis=1)

    X = meta[META_FEATURES]
    y = meta["tier"]

    if y.nunique() < 2:
        model = InventoryRiskModel(model=None, classes=[y.iloc[0]] if len(y) else ["low"],
                                   fallback_tier=y.iloc[0] if len(y) else "low",
                                   accuracy=float("nan"))
    else:
        clf = boosted_classifier(n_estimators=200, max_depth=3, learning_rate=0.07)
        # XGBoost needs integer class codes for multiclass; map via categorical
        codes, uniques = pd.factorize(y)
        clf.fit(X, codes)
        acc = float((clf.predict(X) == codes).mean())
        model = InventoryRiskModel(model=clf, classes=list(uniques),
                                   accuracy=round(acc, 4))

    table = inventory_risk_table(model, meta)
    return model, table


def inventory_risk_table(model: InventoryRiskModel, meta: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU composite risk tier + score from the meta-model."""
    out = meta[["product_id", "product_name", "p_stockout", "p_understock",
                "p_overstock"]].copy()
    if model.model is None:
        out["risk_tier"] = model.fallback_tier
        out["risk_score"] = out["p_stockout"]
    else:
        pred = model.model.predict(meta[model.features])
        proba = model.model.predict_proba(meta[model.features])
        out["risk_tier"] = [model.classes[int(i)] for i in pred]
        # risk_score = probability mass on the 'high' tier when present, else stockout proba
        if "high" in model.classes:
            hi = model.classes.index("high")
            out["risk_score"] = np.round(proba[:, hi], 3)
        else:
            out["risk_score"] = out["p_stockout"]
    tier_rank = {"high": 0, "medium": 1, "low": 2}
    return out.sort_values("risk_tier", key=lambda s: s.map(tier_rank)).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Smart Reordering — hybrid (deterministic rule ⊕ stockout probability)
# --------------------------------------------------------------------------- #
def reorder_hybrid(health: pd.DataFrame, risk_table: pd.DataFrame) -> pd.DataFrame:
    """Combine the rule-based reorder decision with stockout probability.

    ``health`` comes from :func:`inventory.inventory_health` (the rule engine);
    ``risk_table`` from :func:`stock_risk.stock_risk_table`. The rule decides
    *whether/how much*; the classifier ranks *urgency*.
    """
    merged = health.merge(risk_table[["product_id", "p_stockout"]],
                          on="product_id", how="left")
    merged["p_stockout"] = merged["p_stockout"].fillna(0.0)

    # urgency blends the hard rule (needs_reorder) with the soft probability
    merged["reorder_urgency"] = (
        0.6 * merged["needs_reorder"].astype(float) + 0.4 * merged["p_stockout"]
    ).round(3)

    def _action(r):
        if r["needs_reorder"] and r["p_stockout"] >= 0.5:
            return "order_now_priority"
        if r["needs_reorder"]:
            return "order_now"
        if r["p_stockout"] >= 0.5:
            return "watch_high_risk"
        return "ok"

    merged["reorder_action"] = merged.apply(_action, axis=1)
    cols = ["product_id", "product_name", "current_stock", "days_of_cover",
            "reorder_point", "recommended_order_qty", "needs_reorder",
            "p_stockout", "reorder_urgency", "reorder_action"]
    cols = [c for c in cols if c in merged.columns]
    return merged[cols].sort_values("reorder_urgency", ascending=False).reset_index(drop=True)
