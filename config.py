"""
Central configuration for the INVEN-AI inventory management system.

Values can be overridden via environment variables (or Streamlit secrets),
which keeps secrets out of source control.
"""
from __future__ import annotations

import os
from pathlib import Path

# Load a local .env if present (optional dependency — silently skipped if absent).
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
REGISTRY_DIR = ROOT_DIR / "models_registry"
ARTIFACTS_DIR = REGISTRY_DIR / "artifacts"

for _d in (DATA_DIR, REGISTRY_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Primary data files (created by data/generate_synthetic_data.py)
SALES_CSV = DATA_DIR / "sales.csv"
PRODUCTS_CSV = DATA_DIR / "products.csv"
SUPPLIERS_CSV = DATA_DIR / "suppliers.csv"
CUSTOMERS_CSV = DATA_DIR / "customers.csv"
PURCHASE_ORDERS_CSV = DATA_DIR / "purchase_orders.csv"

# --------------------------------------------------------------------------- #
# Claude / Anthropic configuration
# --------------------------------------------------------------------------- #
def _get_secret(name: str, default: str = "") -> str:
    """Read a config value from env first, then Streamlit secrets if available."""
    val = os.environ.get(name)
    if val:
        return val
    try:  # streamlit may not be importable in pure-script contexts
        import streamlit as st

        if name in st.secrets:  # type: ignore[attr-defined]
            return str(st.secrets[name])
    except Exception:
        pass
    return default


ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY", "")

# Model ids — default to the latest, most capable Claude models.
CLAUDE_MODEL = _get_secret("CLAUDE_MODEL", "claude-opus-4-8")
CLAUDE_MODEL_FAST = _get_secret("CLAUDE_MODEL_FAST", "claude-haiku-4-5-20251001")
CLAUDE_VISION_MODEL = _get_secret("CLAUDE_VISION_MODEL", "claude-opus-4-8")

# --------------------------------------------------------------------------- #
# Business / domain defaults (MSME-friendly tunables)
# --------------------------------------------------------------------------- #
DEFAULT_SERVICE_LEVEL = 0.95          # target fill rate for safety stock
DEFAULT_LEAD_TIME_DAYS = 7            # supplier lead time when unknown
DEFAULT_REVIEW_PERIOD_DAYS = 7        # how often stock is reviewed
OVERSTOCK_DAYS_THRESHOLD = 60         # > N days of cover => overstock
UNDERSTOCK_DAYS_THRESHOLD = 7         # < N days of cover => understock
HOLDING_COST_RATE = 0.25             # annual holding cost as % of unit cost

# Forecasting
FORECAST_HORIZON_DAYS = 30
MIN_HISTORY_DAYS = 30

# MLOps
RETRAIN_INTERVAL_DAYS = 7
DRIFT_PSI_THRESHOLD = 0.2             # population stability index alert level

APP_TITLE = "INVEN-AI · Smart Inventory for MSMEs"
APP_ICON = "📦"
