"""Dynamic price recommendation page."""
from __future__ import annotations

import streamlit as st

from src.models.price_recommendation import price_recommendations_table, recommend_price
from ui.components import charts
from ui.state import data_version, get_data, get_price_model


@st.cache_data(show_spinner="Optimising prices with the demand-response model…")
def _table(_v: int):
    data = get_data(_v)
    model = get_price_model(_v)
    return price_recommendations_table(data["sales"], data["products"], model=model)


def render() -> None:
    st.header("💲 Dynamic Price Recommendations")
    v = data_version()
    data = get_data(v)
    price_model = get_price_model(v)
    table = _table(v)

    gainers = table[table["expected_profit_change_pct"] > 0]
    charts.kpi_row(
        [
            {"label": "Products analysed", "value": len(table)},
            {"label": "Repricing opportunities", "value": len(gainers)},
            {"label": "Avg profit uplift", "value": f"{gainers['expected_profit_change_pct'].mean():.1f}%"
             if len(gainers) else "0%"},
            {"label": "High-confidence", "value": int((table["confidence"] == "high").sum())},
        ]
    )

    st.subheader("Per-product price simulator")
    products = data["products"]
    label_map = {f"{r['product_name']} ({r['product_id']})": r for _, r in products.iterrows()}
    choice = st.selectbox("Product", list(label_map.keys()))
    p = label_map[choice]
    max_change = st.slider("Max price change", 0.05, 0.40, 0.20, step=0.05)
    rec = recommend_price(data["sales"], p["product_id"], float(p["unit_cost"]),
                          float(p["unit_price"]), max_change=max_change, model=price_model)
    c1, c2, c3 = st.columns(3)
    c1.metric("Current price", f"${rec['current_price']:.2f}")
    c2.metric("Recommended", f"${rec['recommended_price']:.2f}",
              f"{(rec['recommended_price']-rec['current_price'])/rec['current_price']*100:+.1f}%"
              if rec['current_price'] else None)
    c3.metric("Expected profit", f"{rec['expected_profit_change_pct']:+.1f}%")
    st.info(f"**{rec['confidence'].title()} confidence** · Elasticity: "
            f"{rec['elasticity'] if rec['elasticity'] is not None else 'n/a'} — {rec['rationale']}")

    st.divider()
    st.subheader("All recommendations")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("Prices optimised with a pooled **XGBoost demand-response regressor** "
               "(demand constrained to be non-increasing in price); the profit-maximising "
               "price is chosen within ±guardrail. Falls back to log-log elasticity for "
               "cold-start SKUs. The `method` column shows which engine priced each item.")
