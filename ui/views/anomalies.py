"""Fraud & anomaly detection page."""
from __future__ import annotations

import streamlit as st

from ui.components import charts
from ui.state import data_version, get_anomalies


def render() -> None:
    st.header("🛡️ Fraud & Anomaly Detection")
    annotated, summary = get_anomalies(data_version())

    charts.kpi_row(
        [
            {"label": "Transactions", "value": f"{summary['total_transactions']:,}"},
            {"label": "Flagged", "value": summary["flagged"]},
            {"label": "Flagged %", "value": f"{summary['flagged_pct']}%"},
            {"label": "Value at risk", "value": f"${summary['value_at_risk']:,.0f}"},
        ]
    )

    if summary["top_reasons"]:
        reasons = summary["top_reasons"]
        st.plotly_chart(
            charts.bar(
                __import__("pandas").DataFrame(
                    {"reason": list(reasons.keys()), "count": list(reasons.values())}
                ),
                "reason", "count", "Flag reasons", color=charts.RED,
            ),
            use_container_width=True,
        )

    st.subheader("🚩 Flagged transactions")
    flagged = annotated[annotated["is_anomaly"] == 1].sort_values("anomaly_score", ascending=False)
    st.dataframe(
        flagged[["date", "product_name", "quantity", "unit_price", "revenue",
                 "anomaly_score", "anomaly_reason"]].head(100),
        use_container_width=True, hide_index=True,
    )
    st.caption("Isolation Forest over transaction features (price deviation, quantity z-score, "
               "discount depth, timing) combined with deterministic fraud rules.")
