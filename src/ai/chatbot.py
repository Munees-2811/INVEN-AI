"""
AI inventory chatbot.

Answers natural-language questions about the business. It first computes a
compact, structured "business context" snapshot from live data and hands it to
Claude as grounding, so answers are specific and numeric rather than generic.
When the API is unavailable it falls back to a keyword router over the same
context so the assistant still responds usefully.
"""
from __future__ import annotations

import pandas as pd

from src.ai import claude_client
from src.models.sales_trends import abc_analysis, growth_metrics

SYSTEM_PROMPT = """You are INVEN-AI, an inventory & retail analytics assistant for a
small business (MSME) owner. You are given a JSON snapshot of the current
business state. Answer the user's question concisely and specifically, using the
numbers in the snapshot. Prefer bullet points and concrete recommendations.
If the snapshot lacks the data, say so briefly. Never invent figures."""


def build_context(data: dict, health: pd.DataFrame | None = None) -> dict:
    """Summarise the business into a small dict to ground the model."""
    sales = data["sales"]
    products = data["products"]
    g = growth_metrics(sales)
    abc = abc_analysis(sales)

    ctx = {
        "catalogue_size": int(len(products)),
        "date_range": [str(sales["date"].min().date()), str(sales["date"].max().date())],
        "revenue_last_30d": g["current_revenue"],
        "revenue_growth_pct": g["revenue_growth_pct"],
        "profit_last_30d": g["current_profit"],
        "top_products_by_revenue": abc.head(5)[["product_name", "revenue"]].to_dict("records"),
        "categories": data["sales"].groupby("category")["revenue"].sum().round(0).sort_values(ascending=False).to_dict(),
    }
    if health is not None and not health.empty:
        ctx["understock_count"] = int((health["status"] == "understock").sum())
        ctx["overstock_count"] = int((health["status"] == "overstock").sum())
        ctx["reorder_now"] = health[health["needs_reorder"]].head(8)[
            ["product_name", "current_stock", "recommended_order_qty"]
        ].to_dict("records")
        ctx["capital_tied_in_stock"] = round(float(health["tied_capital"].sum()), 2)
    return ctx


def answer(question: str, context: dict, history: list[dict] | None = None) -> str:
    import json

    grounding = f"Business snapshot (JSON):\n{json.dumps(context, default=str, indent=2)}\n\nUser question: {question}"
    if claude_client.available():
        try:
            return claude_client.chat(grounding, system=SYSTEM_PROMPT, history=history, max_tokens=900)
        except Exception as e:  # pragma: no cover - network dependent
            return f"(AI service error, showing rule-based answer)\n\n{_fallback(question, context)}"
    return _fallback(question, context)


def _fallback(question: str, ctx: dict) -> str:
    q = question.lower()
    if any(k in q for k in ("reorder", "restock", "low stock", "out of stock", "understock")):
        items = ctx.get("reorder_now", [])
        if not items:
            return "No products currently need reordering. ✅"
        lines = [f"- **{i['product_name']}**: {i['current_stock']} in stock → order ~{i['recommended_order_qty']}" for i in items]
        return "Products to reorder now:\n" + "\n".join(lines)
    if any(k in q for k in ("overstock", "excess", "tied", "capital")):
        return (
            f"You have **{ctx.get('overstock_count', 0)} overstocked SKUs**. "
            f"Approx **${ctx.get('capital_tied_in_stock', 0):,.0f}** is tied up in inventory. "
            "Consider promotions on slow movers."
        )
    if any(k in q for k in ("top", "best", "selling", "revenue", "sales")):
        tops = ctx.get("top_products_by_revenue", [])
        lines = [f"- {t['product_name']}: ${t['revenue']:,.0f}" for t in tops]
        g = ctx.get("revenue_growth_pct")
        gl = f"\nRevenue last 30d: ${ctx.get('revenue_last_30d', 0):,.0f}" + (f" ({g:+.1f}% vs prior)" if g is not None else "")
        return "Top products by revenue:\n" + "\n".join(lines) + gl
    if any(k in q for k in ("growth", "trend", "how are sales")):
        g = ctx.get("revenue_growth_pct")
        return f"Revenue last 30 days: ${ctx.get('revenue_last_30d', 0):,.0f}" + (f", {g:+.1f}% vs the prior period." if g is not None else ".")
    return (
        "I can help with reorders, overstock, top products, sales trends and supplier risk. "
        "Try: *'What should I reorder?'* or *'How are my sales trending?'*\n\n"
        "_(Tip: set ANTHROPIC_API_KEY to unlock full conversational AI.)_"
    )
