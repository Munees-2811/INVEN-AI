"""
Tests for the trend-aware reorder flow and the statistical MLOps layer.

Covers: reorder_plan trend adjustment, purchase-order drafting, the
Holt-Winters statistical forecaster, KS-test drift detection and the
champion/challenger demand stage.
"""
from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")

from data.generate_synthetic_data import generate
from src.data.loader import load_all
from src.data.preprocessing import clean_sales
from src.mlops.monitoring import detect_demand_drift
from src.models.inventory import (
    TREND_CAP_PCT,
    TREND_DAMPING,
    build_purchase_orders,
    reorder_plan,
)
from src.models.statistical_forecasting import forecast_product_stat


@pytest.fixture(scope="module")
def data():
    generate(seed=7)
    return load_all()


def test_reorder_plan_trend_adjustment(data):
    plan = reorder_plan(data["sales"], data["products"], data["suppliers"])
    assert {"trend", "change_pct", "trend_factor", "suggested_order_qty"}.issubset(plan.columns)
    assert set(plan["trend"].unique()).issubset({"rising", "falling", "stable"})
    # factor bounded by the damped, capped momentum
    lo = 1 - TREND_CAP_PCT / 100 * TREND_DAMPING
    hi = 1 + TREND_CAP_PCT / 100 * TREND_DAMPING
    assert plan["trend_factor"].between(lo, hi).all()
    assert (plan["suggested_order_qty"] >= 0).all()
    # direction: rising SKUs order at least the rule qty, falling at most
    rising = plan[(plan["trend"] == "rising") & (plan["recommended_order_qty"] > 0)]
    falling = plan[(plan["trend"] == "falling") & (plan["recommended_order_qty"] > 0)]
    if len(rising):
        assert (rising["suggested_order_qty"] >= rising["recommended_order_qty"]).all()
    if len(falling):
        assert (falling["suggested_order_qty"] <= falling["recommended_order_qty"]).all()


def test_build_purchase_orders(data):
    plan = reorder_plan(data["sales"], data["products"], data["suppliers"])
    lines, summary = build_purchase_orders(plan[plan["needs_reorder"]],
                                           data["products"], data["suppliers"])
    if lines.empty:
        pytest.skip("nothing to reorder in this dataset")
    assert (lines["order_qty"] > 0).all()
    assert (lines["line_value"] > 0).all()
    # summary reconciles with lines
    assert summary["order_value"].sum() == pytest.approx(lines["line_value"].sum(), rel=1e-6)
    assert {"supplier_name", "min_order_value", "meets_minimum"}.issubset(summary.columns)


def test_build_purchase_orders_empty_plan(data):
    plan = reorder_plan(data["sales"], data["products"], data["suppliers"])
    lines, summary = build_purchase_orders(plan[plan["suggested_order_qty"] < 0],
                                           data["products"], data["suppliers"])
    assert lines.empty and summary.empty


def test_statistical_forecaster_structure(data):
    pid = data["sales"].groupby("product_id")["quantity"].sum().idxmax()
    res = forecast_product_stat(data["sales"], pid, horizon=14)
    assert len(res.forecast) == 14
    assert (res.forecast["yhat"] >= 0).all()
    assert (res.forecast["yhat_upper"] >= res.forecast["yhat_lower"]).all()
    assert res.model_name in {"holt_winters", "seasonal_naive"}


def test_drift_has_statistical_test(data):
    drift = detect_demand_drift(data["sales"])
    assert {"psi", "ks_stat", "ks_pvalue", "status", "recommend_retrain"}.issubset(drift)
    assert 0.0 <= drift["ks_pvalue"] <= 1.0
    assert 0.0 <= drift["ks_stat"] <= 1.0


def test_champion_challenger_demand_stage(data):
    from src.mlops.pipeline import _train_demand

    clean, _ = clean_sales(data["sales"])
    stage = _train_demand(clean, sample_n=4)
    assert stage.status in {"passed", "failed", "skipped"}
    if stage.status != "skipped":
        assert stage.metrics["champion"] in {"xgboost", "holt_winters"}
        # champion's MAPE must be the better (lower) of the two evaluated
        both = [stage.metrics.get("avg_mape_xgboost"),
                stage.metrics.get("avg_mape_holt_winters")]
        both = [b for b in both if b is not None]
        assert stage.metrics["avg_mape"] == min(both)
