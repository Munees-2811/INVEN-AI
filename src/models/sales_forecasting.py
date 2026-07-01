"""
Sales (revenue) forecasting — built *on top of* the demand forecast.

Separation of concerns / dependency flow:

    demand_forecasting.forecast_product  →  unit demand per day (the *what*)
    sales_forecasting.forecast_sales     →  revenue per day    (the *money*)

Rather than re-forecasting demand (which would be a redundant model), this
module **consumes the demand forecaster's output** as its primary input and
learns the mapping ``units → revenue`` with an XGBoost regressor. That mapping
is not just ``units × current_price``: it absorbs promotion effects, day-of-week
price/mix variation and basket effects that move revenue-per-unit around. Where
there is too little history to train, it falls back to demand × recent average
selling price.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import FORECAST_HORIZON_DAYS
from src.data.preprocessing import daily_demand
from src.models._boost import backend_label, boosted_regressor
from src.models.demand_forecasting import forecast_product

# Features fed to the units→revenue regressor. ``units`` is the demand signal
# (historical actuals at train time, forecasted units at inference) — this is the
# explicit dependency on the demand model.
_SALES_FEATURES = ["units", "dayofweek", "is_weekend", "month", "avg_price_7"]


@dataclass
class SalesForecastResult:
    product_id: str
    history: pd.DataFrame        # date, units, revenue
    forecast: pd.DataFrame       # date, units, revenue, revenue_lower, revenue_upper
    model_name: str
    metrics: dict


def _daily_revenue(sales: pd.DataFrame, product_id: str) -> pd.DataFrame:
    """Daily units + revenue for one product on a continuous calendar."""
    dd = daily_demand(sales, product_id=product_id)[["date", "quantity", "revenue"]]
    dd = dd.rename(columns={"quantity": "units"}).sort_values("date").reset_index(drop=True)
    return dd


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["dayofweek"] = out["date"].dt.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["month"] = out["date"].dt.month
    # recent realised selling price (revenue per unit), smoothed & lag-safe
    price = (out["revenue"] / out["units"].replace(0, np.nan))
    out["avg_price_7"] = price.shift(1).rolling(7, min_periods=1).mean().ffill().bfill().fillna(0.0)
    return out


def forecast_sales(
    sales: pd.DataFrame,
    product_id: str,
    horizon: int = FORECAST_HORIZON_DAYS,
) -> SalesForecastResult:
    """Forecast daily revenue for a product for ``horizon`` days ahead.

    Depends on :func:`demand_forecasting.forecast_product` for the unit-demand
    trajectory, then maps units → revenue via XGBoost.
    """
    # 1. dependency: get the unit-demand forecast
    demand = forecast_product(sales, product_id, horizon=horizon)
    fc_units = demand.forecast[["date", "yhat"]].rename(columns={"yhat": "units"}).copy()

    # 2. training frame: historical units + revenue
    hist = _daily_revenue(sales, product_id)
    feat = _build_features(hist)
    train = feat[feat["units"] > 0]

    avg_price = float((hist["revenue"].sum() / hist["units"].sum())) if hist["units"].sum() > 0 else 0.0

    # 3. too little signal → revenue = units × recent average price
    if len(train) < 20:
        fc = fc_units.copy()
        fc["revenue"] = (fc["units"] * avg_price).round(2)
        spread = float(train["revenue"].std() or 0.0)
        fc["revenue_lower"] = np.clip(fc["revenue"] - 1.64 * spread, 0, None).round(2)
        fc["revenue_upper"] = (fc["revenue"] + 1.64 * spread).round(2)
        return SalesForecastResult(product_id, hist, fc, "avg_price_fallback",
                                   {"r2": None, "rmse": None, "samples": len(train)})

    # 4. train XGBoost regressor: units (+ calendar/price) → revenue
    model = boosted_regressor(n_estimators=250, max_depth=3, learning_rate=0.05)
    X, y = train[_SALES_FEATURES], train["revenue"]
    model.fit(X, y)

    # in-sample fit quality (small data → report train R²/RMSE honestly)
    pred_in = np.clip(model.predict(X), 0, None)
    ss_res = float(np.sum((y - pred_in) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else None
    rmse = float(np.sqrt(np.mean((y - pred_in) ** 2)))
    resid_std = float(np.std(y - pred_in)) or 1.0

    # 5. inference: assemble features for the forecast horizon using forecast units
    fut = fc_units.copy()
    fut["date"] = pd.to_datetime(fut["date"])
    fut["dayofweek"] = fut["date"].dt.dayofweek
    fut["is_weekend"] = (fut["dayofweek"] >= 5).astype(int)
    fut["month"] = fut["date"].dt.month
    fut["avg_price_7"] = float(feat["avg_price_7"].iloc[-1]) if len(feat) else avg_price

    fut["revenue"] = np.clip(model.predict(fut[_SALES_FEATURES]), 0, None).round(2)
    fut["revenue_lower"] = np.clip(fut["revenue"] - 1.64 * resid_std, 0, None).round(2)
    fut["revenue_upper"] = (fut["revenue"] + 1.64 * resid_std).round(2)

    metrics = {
        "r2": round(r2, 3) if r2 is not None else None,
        "rmse": round(rmse, 2),
        "samples": int(len(train)),
        "demand_model": demand.model_name,
    }
    return SalesForecastResult(
        product_id, hist,
        fut[["date", "units", "revenue", "revenue_lower", "revenue_upper"]],
        backend_label(), metrics,
    )


def sales_forecast_summary(sales: pd.DataFrame, product_id: str,
                           horizon: int = FORECAST_HORIZON_DAYS) -> dict:
    """Compact revenue-forecast summary for dashboards / pipeline metrics."""
    res = forecast_sales(sales, product_id, horizon)
    return {
        "product_id": product_id,
        "model": res.model_name,
        "horizon_days": horizon,
        "forecast_revenue_total": round(float(res.forecast["revenue"].sum()), 2),
        "forecast_revenue_daily_mean": round(float(res.forecast["revenue"].mean() or 0.0), 2),
        "metrics": res.metrics,
    }
