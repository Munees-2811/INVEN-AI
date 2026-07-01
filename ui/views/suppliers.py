"""Supplier performance analysis page."""
from __future__ import annotations

import streamlit as st

from ui.components import charts
from ui.state import data_version, get_supplier_model, get_supplier_scores


def render() -> None:
    st.header("🚚 Supplier Performance Analysis")
    v = data_version()
    scores = get_supplier_scores(v)

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

    # --- Predictive layer: XGBoost on-time reliability ------------------------
    st.divider()
    st.subheader("🔮 Predicted on-time reliability (XGBoost)")
    st.caption("An XGBoost classifier trained on purchase-order outcomes predicts the probability "
               "that a supplier's *next* order arrives on time, from pre-delivery signals "
               "(order size, promised lead time, value, historical quality).")
    try:
        model, reliability = get_supplier_model(v)
        auc_txt = f"{model.auc:.2f}" if model.auc == model.auc else "n/a"
        st.metric("Hold-out AUC", auc_txt,
                  help="ROC-AUC on held-out POs. Low values are expected on small PO sets.")
        st.dataframe(reliability, use_container_width=True, hide_index=True)
    except Exception as exc:  # never break the scorecard page
        st.info(f"Predictive reliability unavailable ({exc}).")
