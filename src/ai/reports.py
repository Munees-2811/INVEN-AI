"""
AI-generated business reports.

Assembles a quantitative briefing pack from every analytics module, then asks
Claude to write an executive narrative an MSME owner can read in two minutes.
Without an API key it produces a clean templated report from the same numbers.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd

from src.ai import claude_client
from src.models.sales_trends import abc_analysis, category_breakdown, growth_metrics, product_momentum

REPORT_SYSTEM = """You are a retail business analyst writing a concise weekly briefing
for a small-business owner. Use the supplied JSON metrics. Structure the report as:
1. Executive Summary (3-4 sentences)
2. Sales Performance
3. Inventory Health & Actions
4. Risks & Anomalies
5. Top 3 Recommended Actions (numbered, specific).
Be concrete, reference the numbers, and keep it under 400 words. Use markdown."""


def assemble_metrics(data: dict, health: pd.DataFrame | None, anomaly_summary: dict | None,
                     supplier_scores: pd.DataFrame | None) -> dict:
    sales = data["sales"]
    g = growth_metrics(sales)
    abc = abc_analysis(sales).head(5)[["product_name", "revenue", "abc_class"]].to_dict("records")
    mom = product_momentum(sales)
    rising = mom[mom["trend"] == "rising"].head(3)[["product_name", "change_pct"]].to_dict("records")
    falling = mom[mom["trend"] == "falling"].head(3)[["product_name", "change_pct"]].to_dict("records")

    metrics = {
        "report_date": str(date.today()),
        "growth": g,
        "top_products": abc,
        "rising_products": rising,
        "falling_products": falling,
        "category_revenue": category_breakdown(sales).head(6).to_dict("records"),
    }
    if health is not None and not health.empty:
        metrics["inventory"] = {
            "understock": int((health["status"] == "understock").sum()),
            "overstock": int((health["status"] == "overstock").sum()),
            "healthy": int((health["status"] == "healthy").sum()),
            "needs_reorder": int(health["needs_reorder"].sum()),
            "capital_tied": round(float(health["tied_capital"].sum()), 2),
            "top_reorders": health[health["needs_reorder"]].head(5)[
                ["product_name", "current_stock", "recommended_order_qty"]
            ].to_dict("records"),
        }
    if anomaly_summary:
        metrics["anomalies"] = anomaly_summary
    if supplier_scores is not None and not supplier_scores.empty:
        metrics["suppliers"] = {
            "best": supplier_scores.iloc[0][["supplier_name", "score_100"]].to_dict(),
            "worst": supplier_scores.iloc[-1][["supplier_name", "score_100"]].to_dict(),
        }
    return metrics


def generate_report(metrics: dict) -> str:
    if claude_client.available():
        try:
            prompt = f"Business metrics (JSON):\n{json.dumps(metrics, default=str, indent=2)}\n\nWrite the briefing."
            return claude_client.chat(prompt, system=REPORT_SYSTEM, max_tokens=1400, temperature=0.4)
        except Exception:
            pass
    return _template_report(metrics)


def _template_report(m: dict) -> str:
    g = m.get("growth", {})
    inv = m.get("inventory", {})
    an = m.get("anomalies", {})
    growth_txt = f"{g.get('revenue_growth_pct'):+.1f}%" if g.get("revenue_growth_pct") is not None else "n/a"
    lines = [
        f"# Weekly Business Briefing — {m.get('report_date')}",
        "",
        "## 1. Executive Summary",
        f"Revenue over the last {g.get('window_days', 30)} days was "
        f"**${g.get('current_revenue', 0):,.0f}** ({growth_txt} vs the prior period), "
        f"with **${g.get('current_profit', 0):,.0f}** profit. "
        f"{inv.get('needs_reorder', 0)} SKUs need reordering and "
        f"{inv.get('overstock', 0)} are overstocked.",
        "",
        "## 2. Sales Performance",
        "Top products: " + ", ".join(f"{p['product_name']} (${p['revenue']:,.0f})" for p in m.get("top_products", [])[:5]),
    ]
    if m.get("rising_products"):
        lines.append("Rising: " + ", ".join(f"{p['product_name']} ({p['change_pct']:+.0f}%)" for p in m["rising_products"]))
    if m.get("falling_products"):
        lines.append("Falling: " + ", ".join(f"{p['product_name']} ({p['change_pct']:+.0f}%)" for p in m["falling_products"]))

    lines += [
        "",
        "## 3. Inventory Health & Actions",
        f"- Understock: **{inv.get('understock', 0)}** · Overstock: **{inv.get('overstock', 0)}** · Healthy: **{inv.get('healthy', 0)}**",
        f"- Capital tied in stock: **${inv.get('capital_tied', 0):,.0f}**",
    ]
    for r in inv.get("top_reorders", [])[:5]:
        lines.append(f"  - Reorder **{r['product_name']}**: {r['current_stock']} on hand → order ~{r['recommended_order_qty']}")

    lines += [
        "",
        "## 4. Risks & Anomalies",
        f"- {an.get('flagged', 0)} flagged transactions, ~${an.get('value_at_risk', 0):,.0f} at risk." if an else "- No anomaly scan available.",
        "",
        "## 5. Top 3 Recommended Actions",
        "1. Reorder the understocked SKUs listed above before they stock out.",
        "2. Launch a clearance promo on overstocked items to free working capital.",
        "3. Review flagged transactions and underperforming suppliers.",
        "",
        "_Generated by INVEN-AI (offline template — set ANTHROPIC_API_KEY for an AI-written narrative)._",
    ]
    return "\n".join(lines)
