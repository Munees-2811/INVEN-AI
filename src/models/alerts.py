"""
Smart alerts engine.

Aggregates signals from every model into a single, prioritised, de-duplicated
alert feed an MSME owner can act on: stockouts, overstock capital lock-up,
reorder-now, supplier risk, anomalies and revenue drops.
"""
from __future__ import annotations

import pandas as pd

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def _alert(severity: str, category: str, title: str, detail: str, action: str) -> dict:
    return {
        "severity": severity,
        "category": category,
        "title": title,
        "detail": detail,
        "action": action,
    }


def build_alerts(
    health: pd.DataFrame | None = None,
    anomaly_summary: dict | None = None,
    supplier_scores: pd.DataFrame | None = None,
    growth: dict | None = None,
) -> list[dict]:
    alerts: list[dict] = []

    if health is not None and not health.empty:
        under = health[health["status"] == "understock"]
        for _, r in under.head(20).iterrows():
            sev = "critical" if (r["days_of_cover"] or 0) < 3 else "high"
            alerts.append(
                _alert(
                    sev, "stock",
                    f"Low stock: {r['product_name']}",
                    f"Only {r['current_stock']} units (~{r['days_of_cover']} days of cover).",
                    f"Reorder ~{int(r['recommended_order_qty'])} units now.",
                )
            )
        over = health[health["status"] == "overstock"].sort_values("tied_capital", ascending=False)
        for _, r in over.head(10).iterrows():
            alerts.append(
                _alert(
                    "medium", "stock",
                    f"Overstock: {r['product_name']}",
                    f"{r['current_stock']} units (~{r['days_of_cover']} days) — ${r['tied_capital']:.0f} tied up.",
                    "Run a promo or pause reordering.",
                )
            )

    if anomaly_summary and anomaly_summary.get("flagged", 0) > 0:
        alerts.append(
            _alert(
                "high", "fraud",
                f"{anomaly_summary['flagged']} suspicious transactions detected",
                f"~${anomaly_summary['value_at_risk']:.0f} of value at risk "
                f"({anomaly_summary['flagged_pct']}% of transactions).",
                "Review the Fraud & Anomaly page.",
            )
        )

    if supplier_scores is not None and not supplier_scores.empty:
        risky = supplier_scores[supplier_scores["score_100"] < 60]
        for _, r in risky.iterrows():
            alerts.append(
                _alert(
                    "medium", "supplier",
                    f"Supplier at risk: {r['supplier_name']}",
                    f"Performance score {r['score_100']}/100 (grade {r['grade']}).",
                    "Negotiate SLAs or source an alternative supplier.",
                )
            )

    if growth and growth.get("revenue_growth_pct") is not None:
        g = growth["revenue_growth_pct"]
        if g <= -10:
            alerts.append(
                _alert(
                    "high", "sales",
                    f"Revenue down {abs(g):.1f}% vs prior period",
                    f"${growth['current_revenue']:.0f} vs ${growth['previous_revenue']:.0f}.",
                    "Check trend analysis & run targeted promotions.",
                )
            )
        elif g >= 15:
            alerts.append(
                _alert(
                    "info", "sales",
                    f"Revenue up {g:.1f}% 🎉",
                    "Strong momentum — ensure top sellers stay in stock.",
                    "Review rising products and reorder proactively.",
                )
            )

    alerts.sort(key=lambda a: SEVERITY_ORDER.get(a["severity"], 9))
    return alerts
