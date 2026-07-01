"""AI demand forecasting page."""
from __future__ import annotations

import streamlit as st

from config import FORECAST_HORIZON_DAYS
from src.models.demand_forecasting import forecast_product
from src.models.sales_forecasting import forecast_sales
from ui.components import charts
from ui.state import data_version, get_data


def render() -> None:
    st.header("🔮 AI Demand Forecasting")
    data = get_data(data_version())
    products = data["products"]

    c1, c2 = st.columns([3, 1])
    with c1:
        label_map = {f"{r['product_name']} ({r['product_id']})": r["product_id"]
                     for _, r in products.iterrows()}
        choice = st.selectbox("Select a product", list(label_map.keys()))
        pid = label_map[choice]
    with c2:
        horizon = st.slider("Horizon (days)", 7, 90, FORECAST_HORIZON_DAYS, step=7)

    with st.spinner("Training forecaster and projecting demand…"):
        res = forecast_product(data["sales"], pid, horizon=horizon)

    m = res.metrics or {}
    charts.kpi_row(
        [
            {"label": "Model", "value": res.model_name.replace("_", " ").title()},
            {"label": f"Forecast total ({horizon}d)", "value": f"{res.forecast['yhat'].sum():,.0f}"},
            {"label": "Avg daily demand", "value": f"{res.forecast['yhat'].mean():.1f}"},
            {"label": "Backtest MAPE", "value": f"{m['mape']:.1f}%" if m.get("mape") is not None else "n/a",
             "help": "Mean abs. % error on a 28-day hold-out"},
            {"label": "Accuracy", "value": f"{max(0, 100 - m['mape']):.0f}%" if m.get("mape") is not None else "n/a"},
        ]
    )

    st.plotly_chart(
        charts.forecast_chart(res.history.tail(120), res.forecast, f"Forecast — {choice}"),
        use_container_width=True,
    )

    with st.expander("Forecast detail (table)"):
        st.dataframe(res.forecast, use_container_width=True, hide_index=True)
    st.caption(
        "XGBoost recursive forecaster on lag + calendar features, with an "
        "empirical 90% prediction interval from backtest residuals. Sparse products "
        "fall back to a seasonal-naive baseline."
    )

    # --- Sales (revenue) forecast — built on top of the demand forecast --------
    st.divider()
    st.subheader("💰 Sales (revenue) forecast")
    st.caption("A second XGBoost regressor maps forecasted **units → revenue** "
               "(absorbing price/promo/mix effects). It consumes the demand forecast above.")
    try:
        sres = forecast_sales(data["sales"], pid, horizon=horizon)
        sm = sres.metrics or {}
        charts.kpi_row(
            [
                {"label": "Model", "value": sres.model_name.replace("_", " ").title()},
                {"label": f"Revenue ({horizon}d)", "value": f"${sres.forecast['revenue'].sum():,.0f}"},
                {"label": "Avg daily revenue", "value": f"${sres.forecast['revenue'].mean():,.0f}"},
                {"label": "Fit R²", "value": f"{sm['r2']:.2f}" if sm.get("r2") is not None else "n/a",
                 "help": "In-sample R² of the units→revenue mapping"},
            ]
        )
        with st.expander("Revenue forecast detail (table)"):
            st.dataframe(sres.forecast, use_container_width=True, hide_index=True)
    except Exception as exc:  # keep the demand forecast usable regardless
        st.info(f"Revenue forecast unavailable for this product ({exc}).")
