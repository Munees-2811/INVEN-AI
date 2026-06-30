"""
End-to-end smoke tests for the INVEN-AI data, models and MLOps pipeline.

Run:  pytest -q
These are fast, deterministic checks that the full stack wires together and that
each model produces structurally valid, sane output on the synthetic dataset.
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from data.generate_synthetic_data import generate
from src.data.loader import load_all
from src.data.preprocessing import clean_sales, daily_demand, make_features
from src.models.anomaly_detection import anomaly_summary, detect_anomalies
from src.models.categorization import suggest_category, train_categorizer
from src.models.customer_prediction import train_purchase_model
from src.models.demand_forecasting import forecast_product
from src.models.inventory import compute_reorder, inventory_health
from src.models.price_recommendation import recommend_price
from src.models.sales_trends import abc_analysis, growth_metrics
from src.models.supplier_analysis import supplier_scorecard
from src.utils.stats import z_for_service_level


@pytest.fixture(scope="module")
def data():
    generate(seed=7)
    return load_all()


def test_data_shapes(data):
    assert len(data["sales"]) > 1000
    assert {"date", "product_id", "quantity", "revenue"}.issubset(data["sales"].columns)
    assert len(data["products"]) > 20


def test_preprocessing_drops_bad_rows(data):
    clean, rep = clean_sales(data["sales"])
    assert (clean["quantity"] > 0).all()
    assert rep.rows_out <= rep.rows_in
    feats = make_features(daily_demand(clean, data["products"].iloc[0]["product_id"]))
    assert not feats.isna().any().any()


def test_forecast_structure(data):
    pid = data["products"].iloc[0]["product_id"]
    res = forecast_product(data["sales"], pid, horizon=14)
    assert len(res.forecast) == 14
    assert (res.forecast["yhat"] >= 0).all()
    assert (res.forecast["yhat_upper"] >= res.forecast["yhat_lower"]).all()


def test_reorder_math():
    ro = compute_reorder(daily_mean=10, daily_std=3, current_stock=20,
                         unit_cost=2.0, lead_time_days=7)
    assert ro["reorder_point"] > ro["safety_stock"] > 0
    assert ro["needs_reorder"] is True  # 20 < reorder point
    assert ro["recommended_order_qty"] >= 0


def test_z_score_monotonic():
    assert z_for_service_level(0.90) < z_for_service_level(0.95) < z_for_service_level(0.99)


def test_inventory_health(data):
    h = inventory_health(data["sales"], data["products"], data["suppliers"])
    assert set(h["status"].unique()).issubset({"understock", "healthy", "overstock", "no_demand"})
    assert (h["tied_capital"] >= 0).all()


def test_categorizer(data):
    model = train_categorizer(data["products"])
    pred = suggest_category(model, "Velo Cola 500ml")
    assert pred["predicted_category"] in model.classes
    assert 0 <= pred["confidence"] <= 1


def test_customer_model(data):
    model, rfm = train_purchase_model(data["sales"])
    assert "repurchase_proba" in rfm.columns
    assert rfm["repurchase_proba"].between(0, 1).all()


def test_anomaly_detection(data):
    annotated = detect_anomalies(data["sales"])
    summ = anomaly_summary(annotated)
    assert summ["flagged"] >= 0
    assert 0 <= summ["flagged_pct"] <= 100


def test_pricing(data):
    p = data["products"].iloc[0]
    rec = recommend_price(data["sales"], p["product_id"], float(p["unit_cost"]), float(p["unit_price"]))
    assert rec["recommended_price"] > 0


def test_suppliers(data):
    sc = supplier_scorecard(data["suppliers"], data["purchase_orders"])
    assert sc["score_100"].between(0, 100).all()


def test_trends(data):
    g = growth_metrics(data["sales"])
    assert "revenue_growth_pct" in g
    abc = abc_analysis(data["sales"])
    assert set(abc["abc_class"].unique()).issubset({"A", "B", "C"})


def test_full_mlops_pipeline(data):
    from src.mlops.pipeline import run_pipeline
    from src.mlops import registry

    run = run_pipeline(data)
    assert run.finished_at is not None
    assert len(run.stages) == 3
    # at least the forecaster should register a version
    assert registry.get_versions("demand_forecaster")
    assert "psi" in run.drift
