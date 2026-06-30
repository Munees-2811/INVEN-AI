"""AI-generated business reports page."""
from __future__ import annotations

from datetime import date

import streamlit as st

from src.ai import claude_client
from src.ai.reports import assemble_metrics, generate_report
from ui.state import (
    data_version,
    get_anomalies,
    get_data,
    get_health,
    get_supplier_scores,
)


def render() -> None:
    st.header("📝 AI Business Reports")
    if claude_client.available():
        st.caption("🟢 Reports written by Claude from your live metrics.")
    else:
        st.caption("🟡 Offline mode — structured template report (set `ANTHROPIC_API_KEY` for AI narrative).")

    v = data_version()
    data = get_data(v)
    health = get_health(v)
    try:
        _, an = get_anomalies(v)
    except Exception:
        an = None
    try:
        sup = get_supplier_scores(v)
    except Exception:
        sup = None

    if st.button("🪄 Generate weekly briefing", type="primary"):
        with st.spinner("Compiling metrics and writing the report…"):
            metrics = assemble_metrics(data, health, an, sup)
            report = generate_report(metrics)
        st.session_state["last_report"] = report

    report = st.session_state.get("last_report")
    if report:
        st.markdown(report)
        st.download_button("⬇️ Download report (Markdown)", report,
                           f"inven-ai-report-{date.today()}.md", "text/markdown")
    else:
        st.info("Click **Generate weekly briefing** to produce an executive report.")
