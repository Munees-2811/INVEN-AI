"""
Stock-risk classification: stockout, understock and overstock.

Three *separate but sibling* XGBoost classifiers answer three distinct business
questions from **one shared feature matrix** (so there is no duplicated feature
engineering):

    stockout   → will this SKU run out before replenishment arrives?
                 (days-of-cover < lead time)
    understock → is cover below the safety threshold?  (< UNDERSTOCK_DAYS)
    overstock  → is too much capital tied up?           (> OVERSTOCK_DAYS)

The deterministic inventory-theory rules (:func:`inventory.classify_stock`,
reorder point, days-of-cover) provide the training *labels*; the classifiers
learn to reproduce and generalise those thresholds from the underlying demand /
stock features, emitting a calibrated **probability** per risk (which the
reorder hybrid and the inventory-risk meta-model then consume). The rule layer
stays authoritative and untouched — this adds a probabilistic view on top.

Features (shared): current_stock, unit_cost, daily_mean, daily_std, demand_cv,
lead_time_days, days_of_cover, reorder_point, safety_stock, cover_ratio.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import (
    DEFAULT_LEAD_TIME_DAYS,
    OVERSTOCK_DAYS_THRESHOLD,
    UNDERSTOCK_DAYS_THRESHOLD,
)
from src.models._boost import backend_label, boosted_classifier
from src.models.inventory import _recent_demand_stats, compute_reorder

RISK_FEATURES = [
    "current_stock", "unit_cost", "daily_mean", "daily_std", "demand_cv",
    "lead_time_days", "days_of_cover", "reorder_point", "safety_stock", "cover_ratio",
]
RISKS = ("stockout", "understock", "overstock")


@dataclass
class StockRiskModels:
    """Bundle of the three fitted risk classifiers + their training metrics."""

    models: dict = field(default_factory=dict)      # risk -> fitted estimator | None
    fallback_label: dict = field(default_factory=dict)  # risk -> constant (single-class)
    features: list = field(default_factory=lambda: list(RISK_FEATURES))
    metrics: dict = field(default_factory=dict)
    backend: str = backend_label()

    def predict_proba(self, X: pd.DataFrame) -> dict:
        out = {}
        for risk in RISKS:
            mdl = self.models.get(risk)
            if mdl is not None:
                out[risk] = np.clip(mdl.predict_proba(X[self.features])[:, 1], 0, 1)
            else:  # single-class training set → constant probability
                out[risk] = np.full(len(X), float(self.fallback_label.get(risk, 0.0)))
        return out


def build_risk_features(
    sales: pd.DataFrame,
    products: pd.DataFrame,
    suppliers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the shared per-SKU feature/label matrix for stock-risk models."""
    lead_lookup = {}
    if suppliers is not None:
        lead_lookup = suppliers.set_index("supplier_id")["avg_lead_time_days"].to_dict()

    stats = _recent_demand_stats(sales).set_index("product_id")
    rows = []
    for _, p in products.iterrows():
        pid = p["product_id"]
        s = stats.loc[pid] if pid in stats.index else None
        daily_mean = float(s["daily_mean"]) if s is not None else 0.0
        daily_std = max(float(s["daily_std"]) if s is not None else 0.0, 1e-6)
        lead = int(lead_lookup.get(p.get("supplier_id"), DEFAULT_LEAD_TIME_DAYS))
        stock = float(p["current_stock"])

        ro = compute_reorder(daily_mean, daily_std, stock, float(p["unit_cost"]),
                             lead_time_days=lead)
        doc = ro["days_of_cover"]
        doc_val = float(doc) if doc is not None else 1e6  # no demand → effectively infinite cover
        rows.append({
            "product_id": pid,
            "product_name": p["product_name"],
            "current_stock": stock,
            "unit_cost": float(p["unit_cost"]),
            "daily_mean": daily_mean,
            "daily_std": daily_std,
            "demand_cv": daily_std / daily_mean if daily_mean > 0 else 0.0,
            "lead_time_days": lead,
            "days_of_cover": doc_val,
            "reorder_point": ro["reorder_point"],
            "safety_stock": ro["safety_stock"],
            "cover_ratio": stock / ro["reorder_point"] if ro["reorder_point"] > 0 else 2.0,
            # deterministic labels (ground truth from inventory theory)
            "y_stockout": int(daily_mean > 0 and doc_val < lead),
            "y_understock": int(daily_mean > 0 and doc_val < UNDERSTOCK_DAYS_THRESHOLD),
            "y_overstock": int(doc_val > OVERSTOCK_DAYS_THRESHOLD),
        })
    return pd.DataFrame(rows)


def _fit_one(feat: pd.DataFrame, target: str):
    """Fit one risk classifier; return (model|None, fallback_const, metric)."""
    y = feat[target]
    X = feat[RISK_FEATURES]
    if y.nunique() < 2:
        const = float(y.iloc[0]) if len(y) else 0.0
        return None, const, {"train_accuracy": 1.0, "positives": int(y.sum()), "single_class": True}
    model = boosted_classifier(n_estimators=200, max_depth=3, learning_rate=0.07)
    model.fit(X, y)
    acc = float((model.predict(X) == y).mean())
    return model, 0.0, {"train_accuracy": round(acc, 4), "positives": int(y.sum()),
                        "single_class": False}


def train_stock_risk(
    sales: pd.DataFrame,
    products: pd.DataFrame,
    suppliers: pd.DataFrame | None = None,
) -> tuple[StockRiskModels, pd.DataFrame]:
    """Train the three stock-risk classifiers and score the whole catalogue."""
    feat = build_risk_features(sales, products, suppliers)

    models, fallbacks, metrics = {}, {}, {}
    for risk in RISKS:
        mdl, const, met = _fit_one(feat, f"y_{risk}")
        models[risk] = mdl
        fallbacks[risk] = const
        metrics[risk] = met

    bundle = StockRiskModels(models=models, fallback_label=fallbacks, metrics=metrics)
    table = stock_risk_table(bundle, feat)
    return bundle, table


def stock_risk_table(bundle: StockRiskModels, feat: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU probabilities + predicted flags for each stock risk."""
    proba = bundle.predict_proba(feat)
    out = feat[["product_id", "product_name", "current_stock", "days_of_cover",
                "reorder_point"]].copy()
    for risk in RISKS:
        out[f"p_{risk}"] = np.round(proba[risk], 3)
        out[f"pred_{risk}"] = (proba[risk] >= 0.5).astype(int)
    # a single, human-friendly headline status per SKU (severity ordered)
    out["risk_flag"] = np.select(
        [out["pred_stockout"] == 1, out["pred_understock"] == 1, out["pred_overstock"] == 1],
        ["stockout", "understock", "overstock"],
        default="healthy",
    )
    return out.sort_values("p_stockout", ascending=False).reset_index(drop=True)
