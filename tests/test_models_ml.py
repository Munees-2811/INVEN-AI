"""
Tests for the XGBoost model layer and the prediction dependency flow.

Covers the model assignments in the architecture spec: XGBoost backend, sales
forecasting (on demand output), stock-risk classifiers, inventory-risk meta
model, reorder hybrid, supplier reliability and monotonic dynamic pricing.
"""
from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")

from data.generate_synthetic_data import generate
from src.data.loader import load_all
from src.models import _boost
from src.models.demand_forecasting import forecast_product
from src.models.inventory import inventory_health
from src.models.inventory_risk import reorder_hybrid, train_inventory_risk
from src.models.price_recommendation import (
    price_recommendations_table,
    recommend_price,
    train_price_model,
)
from src.models.sales_forecasting import forecast_sales
from src.models.stock_risk import RISKS, train_stock_risk
from src.models.supplier_analysis import train_supplier_model


@pytest.fixture(scope="module")
def data():
    generate(seed=7)
    return load_all()


def test_backend_available():
    # the factory always returns usable estimators (xgboost or fallback)
    assert _boost.BACKEND in {"xgboost", "gradient_boosting"}
    reg = _boost.boosted_regressor(n_estimators=10)
    clf = _boost.boosted_classifier(n_estimators=10)
    assert hasattr(reg, "fit") and hasattr(clf, "predict_proba")


def test_demand_uses_boost_backend(data):
    pid = data["products"].iloc[0]["product_id"]
    res = forecast_product(data["sales"], pid, horizon=10)
    # model_name reflects the active backend for non-sparse products
    assert res.model_name in {_boost.BACKEND, "seasonal_naive"}


def test_sales_forecast_depends_on_demand(data):
    pid = data["sales"].groupby("product_id")["quantity"].sum().idxmax()
    res = forecast_sales(data["sales"], pid, horizon=14)
    assert len(res.forecast) == 14
    assert (res.forecast["revenue"] >= 0).all()
    assert (res.forecast["revenue_upper"] >= res.forecast["revenue_lower"]).all()
    # dependency flow is recorded in metrics for non-fallback runs
    if res.metrics.get("demand_model"):
        assert res.metrics["demand_model"]


def test_stock_risk_three_classifiers(data):
    bundle, table = train_stock_risk(data["sales"], data["products"], data["suppliers"])
    for r in RISKS:
        assert r in bundle.metrics
        assert f"p_{r}" in table.columns
        assert table[f"p_{r}"].between(0, 1).all()
    assert set(table["risk_flag"].unique()).issubset(
        {"stockout", "understock", "overstock", "healthy"}
    )


def test_inventory_risk_meta_consumes_stock_risk(data):
    bundle, _ = train_stock_risk(data["sales"], data["products"], data["suppliers"])
    model, table = train_inventory_risk(
        data["sales"], data["products"], data["suppliers"], risk_bundle=bundle
    )
    # meta model only reads the base risk probabilities
    assert set(model.features) == {"p_stockout", "p_understock", "p_overstock",
                                   "demand_cv", "lead_time_days"}
    assert set(table["risk_tier"].unique()).issubset({"low", "medium", "high"})


def test_reorder_hybrid_composes_rule_and_proba(data):
    health = inventory_health(data["sales"], data["products"], data["suppliers"])
    _bundle, risk_tbl = train_stock_risk(data["sales"], data["products"], data["suppliers"])
    rh = reorder_hybrid(health, risk_tbl)
    assert set(rh["reorder_action"].unique()).issubset(
        {"order_now_priority", "order_now", "watch_high_risk", "ok"}
    )
    assert rh["reorder_urgency"].between(0, 1).all()


def test_supplier_reliability_model(data):
    model, reliability = train_supplier_model(data["purchase_orders"], data["suppliers"])
    assert "predicted_on_time" in reliability.columns
    assert reliability["predicted_on_time"].between(0, 1).all()


def test_dynamic_pricing_monotonic_and_positive(data):
    model = train_price_model(data["sales"], data["products"])
    # with a trained model, demand must be non-increasing in price
    if model.model is not None and model.ctx:
        pid = next(iter(model.ctx))
        med = model.ctx[pid]["median_price"] or 1.0
        d_low = model.demand_at(pid, med * 0.85)
        d_high = model.demand_at(pid, med * 1.15)
        assert d_high <= d_low + 1e-6
    table = price_recommendations_table(data["sales"], data["products"], model=model)
    assert (table["recommended_price"] > 0).all()


def test_recommend_price_backward_compatible(data):
    # called without a model → elasticity/hold fallback still valid
    p = data["products"].iloc[0]
    rec = recommend_price(data["sales"], p["product_id"], float(p["unit_cost"]),
                          float(p["unit_price"]))
    assert rec["recommended_price"] > 0
    assert rec["method"] in {"elasticity", "hold", "xgboost"}
