"""Reusable Plotly chart + KPI helpers with a consistent theme."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

PRIMARY = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
GREY = "#94a3b8"

STATUS_COLORS = {
    "understock": RED,
    "overstock": AMBER,
    "healthy": GREEN,
    "no_demand": GREY,
}


def kpi_row(items: list[dict]) -> None:
    """Render a row of KPI metric cards. Each item: {label, value, delta?, help?}."""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.metric(it["label"], it["value"], it.get("delta"), help=it.get("help"))


def forecast_chart(history, forecast, title: str = "Demand Forecast") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=history["date"], y=history["quantity"], name="History",
                   line=dict(color=GREY, width=1.5))
    )
    if not forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=list(forecast["date"]) + list(forecast["date"][::-1]),
                y=list(forecast["yhat_upper"]) + list(forecast["yhat_lower"][::-1]),
                fill="toself", fillcolor="rgba(37,99,235,0.12)",
                line=dict(color="rgba(0,0,0,0)"), name="90% interval", hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(x=forecast["date"], y=forecast["yhat"], name="Forecast",
                       line=dict(color=PRIMARY, width=2.5))
        )
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.12), hovermode="x unified")
    return fig


def bar(df, x, y, title="", color=None, orientation="v") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[x] if orientation == "v" else df[y],
                         y=df[y] if orientation == "v" else df[x],
                         orientation=orientation,
                         marker_color=color or PRIMARY))
    fig.update_layout(title=title, height=360, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def line(df, x, y, title="", color=PRIMARY) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[x], y=df[y], line=dict(color=color, width=2.5)))
    fig.update_layout(title=title, height=340, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def donut(labels, values, title="") -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55))
    fig.update_layout(title=title, height=340, margin=dict(l=10, r=10, t=40, b=10))
    return fig


SEVERITY_BADGE = {
    "critical": ("🔴", "#fee2e2"),
    "high": ("🟠", "#ffedd5"),
    "medium": ("🟡", "#fef9c3"),
    "low": ("🔵", "#dbeafe"),
    "info": ("🟢", "#dcfce7"),
}


def alert_card(alert: dict) -> None:
    emoji, bg = SEVERITY_BADGE.get(alert["severity"], ("•", "#f1f5f9"))
    st.markdown(
        f"""<div style="background:{bg};padding:12px 16px;border-radius:10px;margin-bottom:8px;">
        <b>{emoji} {alert['title']}</b><br>
        <span style="color:#334155;">{alert['detail']}</span><br>
        <span style="color:#1e3a8a;"><b>→ {alert['action']}</b></span>
        </div>""",
        unsafe_allow_html=True,
    )
