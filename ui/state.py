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
from src.models.inventory import inventory_health
from src.models.sales_trends import growth_metrics
from src.models.supplier_analysis import supplier_scorecard


@st.cache_data(show_spinner=False)
def get_data(_version: int = 0) -> dict:
    return load_all()


@st.cache_data(show_spinner="Scoring inventory health…")
def get_health(_version: int = 0):
    data = get_data(_version)
    return inventory_health(data["sales"], data["products"], data["suppliers"])


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


def data_version() -> int:
    return st.session_state.get("data_version", 0)


def bump_version() -> None:
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
    st.cache_data.clear()
