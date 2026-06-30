"""Executive overview dashboard — KPIs, smart alerts, and headline charts."""
from __future__ import annotations

import streamlit as st

from src.models.alerts import build_alerts
from src.models.sales_trends import category_breakdown, revenue_timeseries
from ui.components import charts
from ui.state import (
    data_version,
    get_anomalies,
    get_data,
    get_growth,
    get_health,
    get_supplier_scores,
)


def render() -> None:
    st.header("📊 Business Overview")
    v = data_version()
    data = get_data(v)
    growth = get_growth(v)
    health = get_health(v)

    rev_growth = growth["revenue_growth_pct"]
    charts.kpi_row(
        [
            {"label": "Revenue (30d)", "value": f"${growth['current_revenue']:,.0f}",
             "delta": f"{rev_growth:+.1f}%" if rev_growth is not None else None},
            {"label": "Profit (30d)", "value": f"${growth['current_profit']:,.0f}"},
            {"label": "Units sold (30d)", "value": f"{growth['current_units']:,}"},
            {"label": "SKUs to reorder", "value": int(health["needs_reorder"].sum()),
             "help": "Products at or below their reorder point"},
            {"label": "Capital in stock", "value": f"${health['tied_capital'].sum():,.0f}"},
        ]
    )

    st.divider()
    left, right = st.columns([2, 1])

    with left:
        ts = revenue_timeseries(data["sales"], freq="W")
        fig = charts.line(ts, "date", "revenue", "Weekly Revenue Trend")
        fig.add_scatter(x=ts["date"], y=ts["revenue_ma4"], name="4-wk avg",
                        line=dict(color=charts.AMBER, dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

        cat = category_breakdown(data["sales"])
        st.plotly_chart(
            charts.bar(cat, "category", "revenue", "Revenue by Category", color=charts.PRIMARY),
            use_container_width=True,
        )

    with right:
        st.subheader("🔔 Smart Alerts")
        try:
            _, an_summary = get_anomalies(v)
        except Exception:
            an_summary = None
        try:
            sup = get_supplier_scores(v)
        except Exception:
            sup = None
        alerts = build_alerts(health=health, anomaly_summary=an_summary,
                              supplier_scores=sup, growth=growth)
        if not alerts:
            st.success("All clear — no urgent alerts.")
        for a in alerts[:8]:
            charts.alert_card(a)
        if len(alerts) > 8:
            st.caption(f"+{len(alerts) - 8} more alerts on the Smart Alerts page.")
