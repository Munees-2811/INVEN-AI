"""
Generate a realistic synthetic dataset for an MSME (e.g. a small retail / FMCG
shop). The data is engineered to contain trend, weekly + yearly seasonality,
promotions, price elasticity, supplier behaviour and a few injected anomalies so
that every downstream AI model has a meaningful signal to learn.

Run:  python -m data.generate_synthetic_data
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    CUSTOMERS_CSV,
    PRODUCTS_CSV,
    PURCHASE_ORDERS_CSV,
    SALES_CSV,
    SUPPLIERS_CSV,
)

RNG = np.random.default_rng(42)

CATEGORIES = {
    "Beverages": ["Cola 500ml", "Mineral Water 1L", "Orange Juice 1L", "Energy Drink", "Green Tea"],
    "Snacks": ["Potato Chips", "Salted Peanuts", "Chocolate Bar", "Biscuits Pack", "Popcorn"],
    "Dairy": ["Full Cream Milk 1L", "Greek Yogurt", "Cheddar Cheese 200g", "Butter 250g"],
    "Household": ["Dish Soap", "Laundry Detergent 2kg", "Paper Towels", "Trash Bags"],
    "Personal Care": ["Shampoo 400ml", "Toothpaste", "Hand Soap", "Face Wash"],
    "Grocery": ["Basmati Rice 5kg", "Wheat Flour 2kg", "Sunflower Oil 1L", "Sugar 1kg", "Tea Leaves 500g"],
}

SUPPLIER_NAMES = [
    "Apex Distributors", "BlueRiver Wholesale", "Sunrise Traders",
    "Metro Supply Co", "GreenLeaf Foods", "Prime Logistics",
]

# Brand prefixes create realistic SKU variants per item. This expands the
# catalogue (~70 SKUs) so the text categorizer has genuine in-category signal
# to learn from rather than a single example per label.
BRANDS = ["Velo", "Nuvo", "Daily", "Gold Leaf", "Prime", "Sunrise",
          "Everfresh", "Royal", "Nature's", "Urban"]


def _build_products() -> pd.DataFrame:
    rows = []
    pid = 1000
    for category, items in CATEGORIES.items():
        for name in items:
            n_variants = int(RNG.integers(4, 7))  # 4-6 brand variants per item
            for brand in RNG.choice(BRANDS, size=n_variants, replace=False):
                pid += 1
                unit_cost = round(float(RNG.uniform(0.5, 25.0)), 2)
                margin = float(RNG.uniform(0.20, 0.55))
                rows.append(
                    {
                        "product_id": f"P{pid}",
                        "product_name": f"{brand} {name}",
                        "category": category,
                        "unit_cost": unit_cost,
                        "unit_price": round(unit_cost * (1 + margin), 2),
                        "supplier_id": f"S{RNG.integers(1, len(SUPPLIER_NAMES) + 1):02d}",
                        "shelf_life_days": int(RNG.choice([30, 60, 90, 180, 365, 730])),
                        "current_stock": int(RNG.integers(20, 400)),
                        "reorder_point": 0,  # filled by the reorder model later
                        "base_daily_demand": round(float(RNG.uniform(2, 40)), 1),
                    }
                )
    return pd.DataFrame(rows)


def _build_suppliers() -> pd.DataFrame:
    rows = []
    for i, name in enumerate(SUPPLIER_NAMES, start=1):
        rows.append(
            {
                "supplier_id": f"S{i:02d}",
                "supplier_name": name,
                "avg_lead_time_days": int(RNG.integers(2, 14)),
                "lead_time_std": round(float(RNG.uniform(0.5, 4.0)), 2),
                "on_time_rate": round(float(RNG.uniform(0.70, 0.99)), 3),
                "defect_rate": round(float(RNG.uniform(0.0, 0.08)), 3),
                "price_competitiveness": round(float(RNG.uniform(0.6, 1.0)), 3),
                "min_order_value": int(RNG.choice([100, 250, 500, 1000])),
            }
        )
    return pd.DataFrame(rows)


def _build_customers(n: int = 800, days: int = 540) -> pd.DataFrame:
    """Customers carry a latent purchase ``propensity`` and a possible ``churn_day``.

    These latent variables drive who actually transacts (see _build_sales), which
    is what makes Recency/Frequency/Monetary features genuinely predictive of
    future repurchase instead of pure noise.
    """
    segments = ["Walk-in", "Regular", "Wholesale", "Corporate"]
    rows = []
    for i in range(1, n + 1):
        seg = RNG.choice(segments, p=[0.45, 0.35, 0.12, 0.08])
        loyalty = round(float(RNG.beta(2, 2)), 3)
        # higher loyalty => higher base propensity to transact
        propensity = round(float(0.2 + 0.8 * loyalty + RNG.normal(0, 0.05)), 4)
        # ~30% of customers churn at some point in the window
        churn_day = int(RNG.integers(int(days * 0.3), days)) if RNG.random() < 0.30 else days + 1
        rows.append(
            {
                "customer_id": f"C{i:04d}",
                "segment": seg,
                "avg_basket_value": round(float(RNG.uniform(5, 300)), 2),
                "visit_frequency_days": int(RNG.integers(2, 45)),
                "loyalty_score": loyalty,
                "propensity": max(propensity, 0.05),
                "churn_day": churn_day,
            }
        )
    return pd.DataFrame(rows)


def _seasonal_factor(date: pd.Timestamp, category: str) -> float:
    """Weekly + yearly seasonality, category dependent."""
    doy = date.dayofyear
    week = 1.0 + 0.25 * np.sin(2 * np.pi * date.dayofweek / 7.0)
    # weekend uplift for snacks/beverages
    if date.dayofweek >= 5 and category in ("Snacks", "Beverages"):
        week *= 1.3
    year = 1.0 + 0.20 * np.sin(2 * np.pi * doy / 365.0)
    # festive bump in Nov–Dec for grocery & household
    if date.month in (11, 12) and category in ("Grocery", "Household"):
        year *= 1.4
    return float(week * year)


def _build_sales(products: pd.DataFrame, customers: pd.DataFrame, days: int = 540) -> pd.DataFrame:
    end = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    dates = pd.date_range(start, end, freq="D")

    records = []
    cust_ids = np.array(customers["customer_id"].tolist())
    propensity = customers["propensity"].to_numpy(dtype=float)
    churn_day = customers["churn_day"].to_numpy(dtype=int)

    # Pre-compute, per day index, the pool of active customers and their
    # normalized sampling weights (propensity). Customers past their churn_day
    # drop out — this is the signal the repurchase model learns.
    n_days = len(dates)
    day_pools: list[np.ndarray] = []
    for t in range(n_days):
        # Churned customers drop out of the active pool. With a large customer
        # base and a fraction of sales identified, future-window presence becomes
        # a genuine (imperfect) signal the repurchase model can learn.
        w = propensity.copy()
        w[churn_day < t] = 0.0
        s = w.sum()
        day_pools.append(w / s if s > 0 else np.ones_like(w) / len(w))

    all_idx = np.arange(len(cust_ids))
    identified_rate = 0.45  # ~45% of sales tied to a known customer; rest walk-ins

    def _sample_customer(t: int) -> str:
        if RNG.random() > identified_rate:
            return "WALKIN"
        return cust_ids[RNG.choice(all_idx, p=day_pools[t])]

    for _, p in products.iterrows():
        base = p["base_daily_demand"]
        trend = float(RNG.uniform(-0.0003, 0.0010))  # gentle up/down drift per day
        price = p["unit_price"]
        for t, date in enumerate(dates):
            season = _seasonal_factor(date, p["category"])
            promo = 1.0
            on_promo = RNG.random() < 0.04
            if on_promo:
                promo = float(RNG.uniform(1.4, 2.2))  # demand spike during promo
            mu = base * season * (1 + trend * t) * promo
            qty = max(0, int(RNG.poisson(max(mu, 0.1))))
            if qty == 0:
                continue
            # price elasticity: occasional discount
            sale_price = round(price * (0.85 if on_promo else 1.0), 2)
            records.append(
                {
                    "date": date.date().isoformat(),
                    "product_id": p["product_id"],
                    "product_name": p["product_name"],
                    "category": p["category"],
                    "quantity": qty,
                    "unit_price": sale_price,
                    "revenue": round(qty * sale_price, 2),
                    "unit_cost": p["unit_cost"],
                    "profit": round(qty * (sale_price - p["unit_cost"]), 2),
                    "customer_id": _sample_customer(t),
                    "on_promo": int(on_promo),
                }
            )

    sales = pd.DataFrame(records)

    # Inject a handful of anomalies (data-entry errors / theft / fraud signals)
    anomaly_idx = RNG.choice(sales.index, size=max(8, len(sales) // 2500), replace=False)
    for idx in anomaly_idx:
        kind = RNG.choice(["qty_spike", "price_drop", "negative"])
        if kind == "qty_spike":
            sales.loc[idx, "quantity"] = int(sales.loc[idx, "quantity"] * RNG.uniform(8, 20))
        elif kind == "price_drop":
            sales.loc[idx, "unit_price"] = round(sales.loc[idx, "unit_price"] * 0.1, 2)
        else:
            sales.loc[idx, "quantity"] = -abs(int(sales.loc[idx, "quantity"]))
        sales.loc[idx, "revenue"] = round(sales.loc[idx, "quantity"] * sales.loc[idx, "unit_price"], 2)
        sales.loc[idx, "profit"] = round(
            sales.loc[idx, "quantity"] * (sales.loc[idx, "unit_price"] - sales.loc[idx, "unit_cost"]), 2
        )
    sales["is_injected_anomaly"] = 0
    sales.loc[anomaly_idx, "is_injected_anomaly"] = 1
    return sales


def _build_purchase_orders(products: pd.DataFrame, suppliers: pd.DataFrame, n: int = 600) -> pd.DataFrame:
    sup = suppliers.set_index("supplier_id")
    rows = []
    end = datetime.today()
    for i in range(n):
        p = products.sample(1, random_state=int(RNG.integers(0, 1_000_000))).iloc[0]
        sid = p["supplier_id"]
        s = sup.loc[sid]
        order_date = end - timedelta(days=int(RNG.integers(1, 500)))
        promised = int(s["avg_lead_time_days"])
        actual = max(1, int(RNG.normal(promised, s["lead_time_std"])))
        qty = int(RNG.integers(20, 500))
        defective = int(RNG.binomial(qty, s["defect_rate"]))
        rows.append(
            {
                "po_id": f"PO{10000 + i}",
                "order_date": order_date.date().isoformat(),
                "supplier_id": sid,
                "product_id": p["product_id"],
                "qty_ordered": qty,
                "qty_defective": defective,
                "promised_lead_time": promised,
                "actual_lead_time": actual,
                "on_time": int(actual <= promised),
                "order_value": round(qty * p["unit_cost"], 2),
            }
        )
    return pd.DataFrame(rows)


def generate(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate all tables, persist to CSV, and return them."""
    global RNG
    RNG = np.random.default_rng(seed)

    products = _build_products()
    suppliers = _build_suppliers()
    customers = _build_customers()
    sales = _build_sales(products, customers)
    pos = _build_purchase_orders(products, suppliers)

    products.to_csv(PRODUCTS_CSV, index=False)
    suppliers.to_csv(SUPPLIERS_CSV, index=False)
    customers.to_csv(CUSTOMERS_CSV, index=False)
    sales.to_csv(SALES_CSV, index=False)
    pos.to_csv(PURCHASE_ORDERS_CSV, index=False)

    return {
        "products": products,
        "suppliers": suppliers,
        "customers": customers,
        "sales": sales,
        "purchase_orders": pos,
    }


if __name__ == "__main__":
    tables = generate()
    for name, df in tables.items():
        print(f"{name:16s}: {len(df):>7,} rows -> saved")
    print("\nSynthetic dataset generated successfully.")
