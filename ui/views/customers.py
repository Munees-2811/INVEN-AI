"""Customer purchase prediction page."""
from __future__ import annotations

import streamlit as st

from src.models.customer_prediction import train_purchase_model
from ui.components import charts
from ui.state import data_version, get_data


@st.cache_resource(show_spinner="Training purchase model…")
def _model_and_scores(_v: int):
    data = get_data(_v)
    return train_purchase_model(data["sales"])


def render() -> None:
    st.header("🧑‍🤝‍🧑 Customer Purchase Prediction")
    model, rfm = _model_and_scores(data_version())

    charts.kpi_row(
        [
            {"label": "Customers scored", "value": len(rfm)},
            {"label": "Model AUC", "value": f"{model.auc:.3f}" if model.auc == model.auc else "n/a",
             "help": "Hold-out ROC-AUC for repurchase prediction"},
            {"label": "High churn risk", "value": int((rfm["churn_risk"] == "high").sum())},
            {"label": "Likely to return", "value": int((rfm["repurchase_proba"] >= 0.5).sum())},
        ]
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        dist = rfm["churn_risk"].value_counts()
        st.plotly_chart(charts.donut(dist.index.tolist(), dist.values.tolist(), "Churn risk segments"),
                        use_container_width=True)
    with c2:
        st.subheader("⚠️ Highest churn risk")
        st.dataframe(
            rfm.sort_values("repurchase_proba").head(10)[
                ["customer_id", "recency", "frequency", "monetary", "repurchase_proba", "churn_risk"]
            ],
            use_container_width=True, hide_index=True,
        )

    st.divider()
    st.subheader("All customers (RFM + repurchase probability)")
    st.dataframe(
        rfm.sort_values("repurchase_proba", ascending=False)[
            ["customer_id", "recency", "frequency", "monetary", "avg_basket",
             "repurchase_proba", "churn_risk"]
        ],
        use_container_width=True, hide_index=True,
    )
    st.caption("Gradient-boosted classifier on Recency/Frequency/Monetary features. "
               "Target customers with medium/high churn risk via offers.")
