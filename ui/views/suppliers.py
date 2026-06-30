"""Supplier performance analysis page."""
from __future__ import annotations

import streamlit as st

from ui.components import charts
from ui.state import data_version, get_supplier_scores


def render() -> None:
    st.header("🚚 Supplier Performance Analysis")
    scores = get_supplier_scores(data_version())

    best = scores.iloc[0]
    worst = scores.iloc[-1]
    charts.kpi_row(
        [
            {"label": "Suppliers", "value": len(scores)},
            {"label": "Top supplier", "value": best["supplier_name"], "help": f"{best['score_100']}/100"},
            {"label": "Avg score", "value": f"{scores['score_100'].mean():.0f}/100"},
            {"label": "At risk (<60)", "value": int((scores["score_100"] < 60).sum())},
        ]
    )

    st.plotly_chart(
        charts.bar(scores.sort_values("score_100"), "supplier_name", "score_100",
                   "Supplier Score (0–100)", orientation="h",
                   color=charts.GREEN),
        use_container_width=True,
    )

    st.subheader("Scorecard")
    show = scores.copy()
    if "on_time_rate" in show:
        show["on_time_rate"] = (show["on_time_rate"] * 100).round(1)
    if "defect_rate" in show:
        show["defect_rate"] = (show["defect_rate"] * 100).round(2)
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption(
        "Weighted score: 35% on-time delivery · 30% quality · 20% lead-time reliability · 15% price. "
        "Computed from purchase-order history where available."
    )
