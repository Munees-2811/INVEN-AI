"""
Data access layer. Loads the core tables, generating synthetic data on first run
so the app is never empty. All readers are defensive: missing files trigger a
regeneration rather than a crash.
"""
from __future__ import annotations

import pandas as pd

from config import (
    CUSTOMERS_CSV,
    PRODUCTS_CSV,
    PURCHASE_ORDERS_CSV,
    SALES_CSV,
    SUPPLIERS_CSV,
)


def ensure_data() -> None:
    """Generate the synthetic dataset if any core file is missing."""
    required = [SALES_CSV, PRODUCTS_CSV, SUPPLIERS_CSV, CUSTOMERS_CSV, PURCHASE_ORDERS_CSV]
    if not all(p.exists() for p in required):
        from data.generate_synthetic_data import generate

        generate()


def load_sales() -> pd.DataFrame:
    ensure_data()
    df = pd.read_csv(SALES_CSV, parse_dates=["date"])
    return df


def load_products() -> pd.DataFrame:
    ensure_data()
    return pd.read_csv(PRODUCTS_CSV)


def load_suppliers() -> pd.DataFrame:
    ensure_data()
    return pd.read_csv(SUPPLIERS_CSV)


def load_customers() -> pd.DataFrame:
    ensure_data()
    return pd.read_csv(CUSTOMERS_CSV)


def load_purchase_orders() -> pd.DataFrame:
    ensure_data()
    return pd.read_csv(PURCHASE_ORDERS_CSV, parse_dates=["order_date"])


def load_all() -> dict[str, pd.DataFrame]:
    return {
        "sales": load_sales(),
        "products": load_products(),
        "suppliers": load_suppliers(),
        "customers": load_customers(),
        "purchase_orders": load_purchase_orders(),
    }
