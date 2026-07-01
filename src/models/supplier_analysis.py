"""
Supplier performance analysis.

Two complementary layers:

1. ``supplier_scorecard`` — a transparent, weighted blend of on-time delivery,
   quality (defect rate), lead-time reliability and price competitiveness.
   Deterministic and explainable — the number a buyer can defend.

2. ``train_supplier_model`` — a **predictive** XGBoost layer trained on
   purchase-order-level outcomes (one row per PO, so real sample size) to learn
   the probability that a *future* order from a supplier will arrive on time,
   from pre-delivery signals (order size, promised lead time, order value and
   the supplier's historical quality/reliability). Aggregated back to the
   supplier, this gives a forward-looking reliability score that complements the
   backward-looking scorecard.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models._boost import backend_label, boosted_classifier

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


# --------------------------------------------------------------------------- #
# Predictive layer — XGBoost on-time-delivery classifier (per purchase order)
# --------------------------------------------------------------------------- #
# Pre-delivery features only (no leakage from actual_lead_time / on_time).
PO_FEATURES = ["qty_ordered", "promised_lead_time", "order_value",
               "sup_defect_rate", "sup_lead_time_std", "sup_price_competitiveness"]


@dataclass
class SupplierModel:
    model: object = None                 # fitted classifier | None (degenerate)
    features: list = field(default_factory=lambda: list(PO_FEATURES))
    auc: float = float("nan")
    base_rate: float = float("nan")
    backend: str = backend_label()


def _po_training_frame(purchase_orders: pd.DataFrame, suppliers: pd.DataFrame) -> pd.DataFrame:
    sup = suppliers.set_index("supplier_id")
    df = purchase_orders.copy()
    df["sup_defect_rate"] = df["supplier_id"].map(sup["defect_rate"]).fillna(0.0)
    df["sup_lead_time_std"] = df["supplier_id"].map(sup.get("lead_time_std", pd.Series(dtype=float))).fillna(1.0)
    df["sup_price_competitiveness"] = df["supplier_id"].map(
        sup.get("price_competitiveness", pd.Series(dtype=float))).fillna(0.8)
    return df


def train_supplier_model(
    purchase_orders: pd.DataFrame | None,
    suppliers: pd.DataFrame,
) -> tuple[SupplierModel, pd.DataFrame]:
    """Train the PO-level on-time classifier and predict per-supplier reliability.

    Returns the model plus a per-supplier frame with predicted on-time
    probability. Falls back to the historical on-time rate when PO history is
    absent or single-class.
    """
    if purchase_orders is None or purchase_orders.empty or "on_time" not in purchase_orders.columns:
        rel = suppliers[["supplier_id", "supplier_name"]].copy() if "supplier_name" in suppliers \
            else suppliers[["supplier_id"]].copy()
        rel["predicted_on_time"] = suppliers.get("on_time_rate", pd.Series(0.8, index=suppliers.index)).values
        rel["reliability_source"] = "historical"
        return SupplierModel(), rel

    df = _po_training_frame(purchase_orders, suppliers)
    X, y = df[PO_FEATURES].fillna(0.0), df["on_time"].astype(int)
    base = float(y.mean())

    if y.nunique() < 2:
        model = SupplierModel(model=None, base_rate=base)
        proba = pd.Series(base, index=df.index)
    else:
        auc = float("nan")
        if len(df) > 40:
            idx = df.sample(frac=1.0, random_state=42).index
            split = int(len(idx) * 0.75)
            tr, te = idx[:split], idx[split:]
            m = boosted_classifier(n_estimators=200, max_depth=3)
            m.fit(X.loc[tr], y.loc[tr])
            try:
                auc = float(roc_auc_score(y.loc[te], m.predict_proba(X.loc[te])[:, 1]))
            except Exception:
                auc = float("nan")
        clf = boosted_classifier(n_estimators=200, max_depth=3)
        clf.fit(X, y)
        proba = pd.Series(clf.predict_proba(X)[:, 1], index=df.index)
        model = SupplierModel(model=clf, auc=auc, base_rate=base)

    df["_p_on_time"] = proba
    agg = df.groupby("supplier_id")["_p_on_time"].mean().reset_index()
    agg = agg.rename(columns={"_p_on_time": "predicted_on_time"})
    names = suppliers.set_index("supplier_id").get("supplier_name")
    if names is not None:
        agg["supplier_name"] = agg["supplier_id"].map(names)
    agg["predicted_on_time"] = agg["predicted_on_time"].round(3)
    agg["reliability_source"] = "xgboost" if model.model is not None else "base_rate"
    return model, agg.sort_values("predicted_on_time", ascending=False).reset_index(drop=True)
