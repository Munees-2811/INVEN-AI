"""MLOps dashboard — run the pipeline, inspect the registry, track performance."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.mlops import monitoring, registry
from src.mlops.pipeline import retrain_due, run_pipeline
from ui.components import charts
from ui.state import data_version, get_data


def render() -> None:
    st.header("⚙️ MLOps Pipeline & Monitoring")
    v = data_version()
    data = get_data(v)

    due = retrain_due()
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.subheader("Automated training pipeline")
        st.caption("Preprocess → Train → Validate (quality gate) → Version → Deploy → Monitor")
    with c2:
        st.metric("Retrain due?", "Yes" if due["due"] else "No",
                  help=f"Interval: {due['interval_days']}d · last: {due['last_trained'] or 'never'}")
    with c3:
        run = st.button("▶️ Run pipeline", type="primary", use_container_width=True)

    if run:
        with st.spinner("Running full MLOps pipeline…"):
            result = run_pipeline(data)
        st.session_state["last_pipeline"] = result.as_dict()
        st.success("Pipeline complete.")

    last = st.session_state.get("last_pipeline")
    if last:
        _render_run(last)

    st.divider()
    _render_registry()
    st.divider()
    _render_monitoring(data)


def _render_run(last: dict) -> None:
    st.subheader("🟢 Latest pipeline run")
    pr = last["preprocess_report"]
    charts.kpi_row(
        [
            {"label": "Rows in → out", "value": f"{pr['rows_in']:,} → {pr['rows_out']:,}"},
            {"label": "Dropped (neg)", "value": pr["dropped_negative"]},
            {"label": "Outliers clipped", "value": pr["clipped_outliers"]},
            {"label": "Drift PSI", "value": last["drift"].get("psi"),
             "help": last["drift"].get("status")},
        ]
    )
    stage_df = pd.DataFrame(last["stages"])
    if not stage_df.empty:
        def _fmt(r):
            icon = {"passed": "✅", "failed": "❌", "skipped": "⏭️"}.get(r["status"], "•")
            return f"{icon} {r['name']} — {r['status']} (v{r.get('version')}) · {r.get('metrics')}"
        for _, r in stage_df.iterrows():
            st.markdown(_fmt(r))
    if last["drift"].get("recommend_retrain"):
        st.warning(f"⚠️ Demand drift detected (PSI={last['drift']['psi']}). Retrain recommended.")


def _render_registry() -> None:
    st.subheader("📚 Model Registry")
    models = registry.list_models()
    if not models:
        st.info("No models registered yet — run the pipeline above.")
        return
    for name, entry in models.items():
        prod = entry.get("production")
        with st.expander(f"**{name}** · {len(entry['versions'])} version(s) · production: v{prod}"):
            rows = []
            for ver in entry["versions"]:
                rows.append(
                    {
                        "version": ver["version"],
                        "stage": ver.get("stage"),
                        "created_at": ver["created_at"][:19],
                        **{f"metric.{k}": v for k, v in (ver.get("metrics") or {}).items()
                           if isinstance(v, (int, float, str))},
                        "gate_passed": (ver.get("validation") or {}).get("passed"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_monitoring(data: dict) -> None:
    st.subheader("📈 Performance Tracking")
    perf = monitoring.performance_frame()
    if perf.empty:
        st.info("No performance history yet — run the pipeline a few times to see trends.")
    else:
        metric_opts = sorted(perf["metric"].unique())
        chosen = st.selectbox("Metric", metric_opts,
                              index=metric_opts.index("accuracy_pct") if "accuracy_pct" in metric_opts else 0)
        sub = perf[perf["metric"] == chosen]
        fig = charts.line(sub, "timestamp", "value", f"{chosen} over training runs")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌊 Data Drift Monitor")
    drift = monitoring.detect_demand_drift(data["sales"])
    c1, c2, c3 = st.columns(3)
    c1.metric("PSI", drift["psi"])
    c2.metric("Status", drift["status"])
    c3.metric("Retrain?", "Yes" if drift["recommend_retrain"] else "No")
    st.caption("Population Stability Index between the last 30 days and the prior 30 days of demand. "
               "PSI ≥ 0.2 indicates meaningful drift and triggers a retrain recommendation.")
