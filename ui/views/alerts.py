"""Smart alerts feed page."""
from __future__ import annotations

import streamlit as st

from src.models.alerts import build_alerts
from ui.components import charts
from ui.state import (
    data_version,
    get_anomalies,
    get_growth,
    get_health,
    get_supplier_scores,
)


def render() -> None:
    st.header("🔔 Smart Alerts")
    v = data_version()
    health = get_health(v)
    growth = get_growth(v)
    try:
        _, an = get_anomalies(v)
    except Exception:
        an = None
    try:
        sup = get_supplier_scores(v)
    except Exception:
        sup = None

    alerts = build_alerts(health=health, anomaly_summary=an, supplier_scores=sup, growth=growth)

    sev_counts = {}
    for a in alerts:
        sev_counts[a["severity"]] = sev_counts.get(a["severity"], 0) + 1
    charts.kpi_row(
        [
            {"label": "Total alerts", "value": len(alerts)},
            {"label": "Critical", "value": sev_counts.get("critical", 0)},
            {"label": "High", "value": sev_counts.get("high", 0)},
            {"label": "Medium", "value": sev_counts.get("medium", 0)},
        ]
    )

    categories = sorted({a["category"] for a in alerts})
    chosen = st.multiselect("Filter by category", categories, default=categories)
    st.divider()

    shown = [a for a in alerts if a["category"] in chosen]
    if not shown:
        st.success("No alerts in the selected categories. ✅")
    for a in shown:
        charts.alert_card(a)
