"""AI sales trend analysis page."""
from __future__ import annotations

import streamlit as st

from src.models.sales_trends import abc_analysis, product_momentum, revenue_timeseries
from ui.components import charts
from ui.state import data_version, get_data, get_growth


def render() -> None:
    st.header("📈 AI Sales Trend Analysis")
    v = data_version()
    data = get_data(v)
    growth = get_growth(v)

    g = growth["revenue_growth_pct"]
    charts.kpi_row(
        [
            {"label": "Revenue (30d)", "value": f"${growth['current_revenue']:,.0f}",
             "delta": f"{g:+.1f}%" if g is not None else None},
            {"label": "Avg order value", "value": f"${growth['avg_order_value']:,.2f}"},
            {"label": "Units (30d)", "value": f"{growth['current_units']:,}"},
            {"label": "Profit (30d)", "value": f"${growth['current_profit']:,.0f}"},
        ]
    )

    freq = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True)
    fmap = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
    ts = revenue_timeseries(data["sales"], freq=fmap[freq])
    fig = charts.line(ts, "date", "revenue", f"{freq} Revenue")
    fig.add_scatter(x=ts["date"], y=ts["revenue_ma4"], name="Trend (MA)",
                    line=dict(color=charts.AMBER, dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    mom = product_momentum(data["sales"])
    with c1:
        st.subheader("🚀 Rising products")
        st.dataframe(
            mom[mom["trend"] == "rising"].head(10)[["product_name", "current_units", "change_pct"]],
            use_container_width=True, hide_index=True,
        )
    with c2:
        st.subheader("📉 Falling products")
        st.dataframe(
            mom[mom["trend"] == "falling"].head(10)[["product_name", "current_units", "change_pct"]],
            use_container_width=True, hide_index=True,
        )

    st.divider()
    st.subheader("🔠 ABC (Pareto) Analysis")
    abc = abc_analysis(data["sales"])
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        dist = abc["abc_class"].value_counts().sort_index()
        st.plotly_chart(charts.donut(dist.index.tolist(), dist.values.tolist(), "SKU count by class"),
                        use_container_width=True)
    with cc2:
        st.dataframe(
            abc[["product_name", "revenue", "revenue_share", "cumulative_share", "abc_class"]].head(20),
            use_container_width=True, hide_index=True,
        )
    st.caption("Class A ≈ top 80% of revenue. Protect A-items from stockouts; rationalise C-items.")
