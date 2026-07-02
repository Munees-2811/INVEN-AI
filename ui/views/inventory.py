"""Inventory page — trend-aware reorder builder + over/understock detection."""
from __future__ import annotations

import streamlit as st

from src.models.inventory import build_purchase_orders
from src.models.inventory_risk import reorder_hybrid
from ui.components import charts
from ui.state import data_version, get_data, get_health, get_reorder_plan, get_stock_risk

_TREND_ICON = {"rising": "📈 rising", "falling": "📉 falling", "stable": "➡️ stable"}


def render() -> None:
    st.header("📦 Inventory & Reorder Intelligence")
    v = data_version()
    health = get_health(v)

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
        st.subheader("📊 Reorder demand trends")
        try:
            plan = get_reorder_plan(v)
            need = plan[plan["needs_reorder"]]
            tc = need["trend"].value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("📈 Rising", int(tc.get("rising", 0)),
                      help="Reorder SKUs with growing demand → order size scaled up")
            c2.metric("➡️ Stable", int(tc.get("stable", 0)))
            c3.metric("📉 Falling", int(tc.get("falling", 0)),
                      help="Reorder SKUs with shrinking demand → order size scaled down")
            base = int(need["recommended_order_qty"].sum())
            adj = int(need["suggested_order_qty"].sum())
            st.caption(f"Trend-adjusted plan: **{adj:,} units** "
                       f"(rule baseline {base:,}) across {len(need)} SKUs. Suggestions follow "
                       "each SKU's 30-day demand momentum, damped and capped at ±25%.")
        except Exception as exc:
            plan = None
            st.info(f"Trend analysis unavailable ({exc}).")

    st.divider()
    _po_builder(v, plan)

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔴 Understock", "🟠 Overstock", "📋 Full table", "🤖 ML risk & priority"]
    )
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
    with tab4:
        st.markdown("**XGBoost stock-risk probabilities** and a hybrid reorder "
                    "priority (rule-based reorder point ⊕ stockout probability).")
        try:
            _bundle, risk_tbl = get_stock_risk(v)
            priority = reorder_hybrid(health, risk_tbl)
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted stockouts", int((risk_tbl["pred_stockout"] == 1).sum()))
            c2.metric("Priority orders", int((priority["reorder_action"] == "order_now_priority").sum()))
            c3.metric("High-risk to watch", int((priority["reorder_action"] == "watch_high_risk").sum()))
            st.dataframe(
                priority[["product_name", "current_stock", "days_of_cover",
                          "recommended_order_qty", "p_stockout", "reorder_urgency",
                          "reorder_action"]].head(50),
                use_container_width=True, hide_index=True,
            )
            st.caption("Three sibling XGBoost classifiers (stockout / understock / overstock) share "
                       "one feature matrix; the reorder point stays the authoritative trigger while the "
                       "stockout probability sets urgency. No separate reorder model is trained.")
        except Exception as exc:  # never break the page if the model can't train
            st.info(f"ML risk scores unavailable for this dataset ({exc}).")


def _po_builder(v: int, plan) -> None:
    """Interactive purchase-order builder driven by the trend-aware plan."""
    st.subheader("🛒 Purchase-order builder")
    if plan is None:
        st.info("Reorder plan unavailable.")
        return
    candidates = plan[plan["suggested_order_qty"] > 0].copy()
    if candidates.empty:
        st.success("Nothing needs reordering right now.")
        return

    st.caption("Review the trend-adjusted quantities, tweak or untick lines, and export "
               "supplier-ready PO drafts.")
    try:
        editor = candidates[["product_id", "product_name", "trend", "current_stock",
                             "days_of_cover", "recommended_order_qty",
                             "suggested_order_qty"]].copy()
        editor["trend"] = editor["trend"].map(_TREND_ICON).fillna("➡️ stable")
        editor.insert(0, "order", candidates["needs_reorder"].astype(bool))
        edited = st.data_editor(
            editor,
            column_config={
                "order": st.column_config.CheckboxColumn("Order?", help="Include in the PO draft"),
                "suggested_order_qty": st.column_config.NumberColumn(
                    "Order qty", min_value=0, step=1,
                    help="Trend-adjusted suggestion — editable"),
                "recommended_order_qty": st.column_config.NumberColumn("Rule qty"),
            },
            disabled=["product_id", "product_name", "trend", "current_stock",
                      "days_of_cover", "recommended_order_qty"],
            hide_index=True, use_container_width=True, height=320,
        )

        chosen = edited[(edited["order"]) & (edited["suggested_order_qty"] > 0)]
        if chosen.empty:
            st.info("Tick at least one line to build purchase orders.")
            return

        data = get_data(v)
        sel = candidates.set_index("product_id").loc[chosen["product_id"]].reset_index()
        sel["suggested_order_qty"] = chosen["suggested_order_qty"].to_numpy().astype(int)
        lines, summary = build_purchase_orders(sel, data["products"], data["suppliers"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Suppliers", len(summary))
        c2.metric("Units", f"{int(lines['order_qty'].sum()):,}")
        c3.metric("Order value", f"${summary['order_value'].sum():,.0f}")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        below = summary[~summary["meets_minimum"]]
        if not below.empty:
            st.warning(f"{len(below)} draft(s) below the supplier's minimum order value — "
                       "top up those orders or bundle them with the next cycle.")
        with st.expander("PO line detail"):
            st.dataframe(lines, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download PO drafts (CSV)",
            lines.to_csv(index=False).encode(),
            file_name="purchase_order_drafts.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as exc:  # editor quirks must never take down the page
        st.info(f"PO builder unavailable ({exc}).")
