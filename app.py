"""
INVEN-AI — AI-powered Inventory Management System for MSMEs.

Streamlit entry point. Wires together the analytics, AI and MLOps modules behind
a clean multi-page navigation. Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make project root importable regardless of launch directory.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP_ICON, APP_TITLE  # noqa: E402
from src.ai import claude_client  # noqa: E402
from src.data.loader import ensure_data  # noqa: E402
from ui import state  # noqa: E402
from ui.views import (  # noqa: E402
    alerts,
    anomalies,
    categorization,
    chatbot,
    customers,
    forecasting,
    inventory,
    mlops,
    ocr,
    overview,
    pricing,
    reports,
    suppliers,
    trends,
    voice,
)

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide",
                   initial_sidebar_state="expanded")


def _sidebar() -> None:
    with st.sidebar:
        st.markdown(f"## {APP_ICON} INVEN-AI")
        st.caption("Smart Inventory for MSMEs")
        if claude_client.available():
            st.success("Claude AI: connected", icon="🟢")
        else:
            st.warning("Claude AI: offline (set ANTHROPIC_API_KEY)", icon="🟡")

        st.divider()
        if st.button("🔄 Regenerate demo data", use_container_width=True):
            from data.generate_synthetic_data import generate

            with st.spinner("Generating fresh synthetic dataset…"):
                generate()
                state.bump_version()
            st.toast("New dataset generated.")
            st.rerun()
        st.caption("Data is synthetic & generated locally on first run.")
        st.divider()
        st.caption("Stack: Streamlit · scikit-learn · Claude · file-based MLOps registry")


def main() -> None:
    ensure_data()
    _sidebar()

    pages = {
        "Home": [
            st.Page(overview.render, title="Overview", icon="📊", default=True),
            st.Page(alerts.render, title="Smart Alerts", icon="🔔"),
        ],
        "Forecasting & Inventory": [
            st.Page(forecasting.render, title="Demand Forecasting", icon="🔮"),
            st.Page(inventory.render, title="Reorder & Stock", icon="📦"),
            st.Page(pricing.render, title="Price Recommendations", icon="💲"),
        ],
        "Analytics": [
            st.Page(trends.render, title="Sales Trends", icon="📈"),
            st.Page(suppliers.render, title="Suppliers", icon="🚚"),
            st.Page(customers.render, title="Customers", icon="🧑‍🤝‍🧑"),
            st.Page(anomalies.render, title="Fraud & Anomalies", icon="🛡️"),
            st.Page(categorization.render, title="Categorization", icon="🏷️"),
        ],
        "AI Assistants": [
            st.Page(chatbot.render, title="Chatbot", icon="💬"),
            st.Page(voice.render, title="Voice Assistant", icon="🎙️"),
            st.Page(ocr.render, title="Invoice OCR", icon="🧾"),
            st.Page(reports.render, title="AI Reports", icon="📝"),
        ],
        "Operations": [
            st.Page(mlops.render, title="MLOps & Monitoring", icon="⚙️"),
        ],
    }

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
