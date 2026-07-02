"""
Shared Streamlit state & cached data access.

Centralises expensive computations behind st.cache_data / st.cache_resource so
every page stays snappy and consistent. Cache keys are invalidated when the user
regenerates data or runs the MLOps pipeline.
"""
from __future__ import annotations

import streamlit as st

from src.data.loader import load_all
from src.models.anomaly_detection import anomaly_summary, detect_anomalies
from src.models.inventory import inventory_health, reorder_plan
from src.models.price_recommendation import train_price_model
from src.models.sales_trends import growth_metrics
from src.models.stock_risk import train_stock_risk
from src.models.supplier_analysis import supplier_scorecard, train_supplier_model


@st.cache_data(show_spinner=False)
def get_data(_version: int = 0) -> dict:
    return load_all()


@st.cache_data(show_spinner="Scoring inventory health…")
def get_health(_version: int = 0):
    data = get_data(_version)
    return inventory_health(data["sales"], data["products"], data["suppliers"])


@st.cache_data(show_spinner="Building trend-aware reorder plan…")
def get_reorder_plan(_version: int = 0):
    """inventory_health + demand momentum → trend-adjusted order quantities."""
    data = get_data(_version)
    return reorder_plan(data["sales"], data["products"], data["suppliers"])


@st.cache_data(show_spinner="Scanning for anomalies…")
def get_anomalies(_version: int = 0):
    data = get_data(_version)
    annotated = detect_anomalies(data["sales"])
    return annotated, anomaly_summary(annotated)


@st.cache_data(show_spinner="Scoring suppliers…")
def get_supplier_scores(_version: int = 0):
    data = get_data(_version)
    return supplier_scorecard(data["suppliers"], data["purchase_orders"])


@st.cache_data(show_spinner=False)
def get_growth(_version: int = 0):
    return growth_metrics(get_data(_version)["sales"])


@st.cache_resource(show_spinner="Training stock-risk classifiers…")
def get_stock_risk(_version: int = 0):
    """(StockRiskModels, per-SKU risk table) — stockout/understock/overstock."""
    data = get_data(_version)
    return train_stock_risk(data["sales"], data["products"], data["suppliers"])


@st.cache_resource(show_spinner="Training price-response model…")
def get_price_model(_version: int = 0):
    data = get_data(_version)
    return train_price_model(data["sales"], data["products"])


@st.cache_resource(show_spinner="Training supplier reliability model…")
def get_supplier_model(_version: int = 0):
    """(SupplierModel, per-supplier predicted on-time reliability)."""
    data = get_data(_version)
    return train_supplier_model(data["purchase_orders"], data["suppliers"])


def data_version() -> int:
    return st.session_state.get("data_version", 0)


def bump_version() -> None:
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
    st.cache_data.clear()
