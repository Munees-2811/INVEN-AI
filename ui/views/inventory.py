"""Inventory page — reorder recommendations + over/understock detection."""
from __future__ import annotations

import streamlit as st

from ui.components import charts
from ui.state import data_version, get_health


def render() -> None:
    st.header("📦 Inventory & Reorder Intelligence")
    health = get_health(data_version())

    counts = health["status"].value_counts().to_dict()
    charts.kpi_row(
        [
            {"label": "Understocked", "value": counts.get("understock", 0)},
            {"label": "Healthy", "value": counts.get("healthy", 0)},
            {"label": "Overstocked", "value": counts.get("overstock", 0)},
            {"label": "Reorder now", "value": int(health["needs_reorder"].sum())},
            {"label": "Capital tied", "value": f"${health['tied_capital'].sum():,.0f}"},
        ]
    )
    st.divider()

    left, right = st.columns([1, 2])
    with left:
        pie = health["status"].value_counts()
        st.plotly_chart(
            charts.donut(pie.index.tolist(), pie.values.tolist(), "Stock Status Mix"),
            use_container_width=True,
        )
    with right:
        st.subheader("🛒 Recommended Purchase Orders")
        reorder = health[health["needs_reorder"]][
            ["product_name", "category", "current_stock", "days_of_cover",
             "recommended_order_qty", "lead_time_days"]
        ]
        if reorder.empty:
            st.success("Nothing needs reordering right now.")
        else:
            st.dataframe(reorder, use_container_width=True, hide_index=True)
            total_units = int(reorder["recommended_order_qty"].sum())
            st.caption(f"Total recommended order: **{total_units:,} units** across {len(reorder)} SKUs.")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["🔴 Understock", "🟠 Overstock", "📋 Full table"])
    with tab1:
        st.dataframe(
            health[health["status"] == "understock"][
                ["product_name", "category", "current_stock", "forecast_daily_demand",
                 "days_of_cover", "reorder_point", "recommended_order_qty"]
            ],
            use_container_width=True, hide_index=True,
        )
    with tab2:
        over = health[health["status"] == "overstock"].sort_values("tied_capital", ascending=False)
        st.dataframe(
            over[["product_name", "category", "current_stock", "days_of_cover", "tied_capital"]],
            use_container_width=True, hide_index=True,
        )
        st.caption(f"${over['tied_capital'].sum():,.0f} of working capital is locked in overstock.")
    with tab3:
        status_filter = st.multiselect("Filter status", health["status"].unique().tolist(),
                                       default=health["status"].unique().tolist())
        st.dataframe(
            health[health["status"].isin(status_filter)], use_container_width=True, hide_index=True
        )
