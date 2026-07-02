"""
Statistical demand forecasting — Holt-Winters exponential smoothing.

The classical statistical counterpart to the XGBoost demand forecaster. The
MLOps pipeline backtests both on the same products (champion/challenger) and
promotes whichever wins on hold-out MAPE, so the registry always serves the
better of a proper statistical model and the ML model rather than assuming one.

Holt-Winters (additive trend + additive weekly seasonality) is the right
statistical baseline for MSME retail demand: it captures level, trend and the
strong day-of-week cycle with a handful of interpretable parameters and almost
no tuning. Falls back to the seasonal-naive heuristic for sparse series or if
the optimiser fails.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from config import FORECAST_HORIZON_DAYS, MIN_HISTORY_DAYS
from src.data.preprocessing import daily_demand
from src.models.demand_forecasting import ForecastResult, _seasonal_naive

MODEL_NAME = "holt_winters"
_SEASON = 7  # weekly cycle


def _fit_holt_winters(y: pd.Series):
    """Fit additive Holt-Winters; returns the fitted results object."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            y.astype(float),
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=_SEASON,
            initialization_method="estimated",
        )
        return model.fit(optimized=True)


def _backtest(series: pd.DataFrame, n_test: int = 28) -> tuple[dict, float]:
    """Hold out the last ``n_test`` days — same protocol as the XGBoost backtest."""
    y = series.sort_values("date")["quantity"].reset_index(drop=True)
    if len(y) <= n_test + 2 * _SEASON:
        return {"mae": None, "rmse": None, "mape": None, "samples": len(y)}, 1.0
    train, test = y.iloc[:-n_test], y.iloc[-n_test:].to_numpy()
    try:
        fit = _fit_holt_winters(train)
        pred = np.clip(fit.forecast(n_test), 0, None)
    except Exception:
        return {"mae": None, "rmse": None, "mape": None, "samples": len(y)}, 1.0
    mae = float(np.mean(np.abs(test - pred)))
    rmse = float(np.sqrt(np.mean((test - pred) ** 2)))
    nonzero = test > 0
    mape = float(np.mean(np.abs((test[nonzero] - pred[nonzero]) / test[nonzero])) * 100) \
        if nonzero.any() else None
    resid_std = float(np.std(test - pred)) or 1.0
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 2) if mape is not None else None,
        "samples": len(y),
    }, resid_std


def forecast_product_stat(
    sales: pd.DataFrame,
    product_id: str,
    horizon: int = FORECAST_HORIZON_DAYS,
) -> ForecastResult:
    """Holt-Winters forecast for one product — mirrors ``forecast_product``."""
    series = daily_demand(sales, product_id=product_id)[["date", "quantity"]]
    series = series.sort_values("date").reset_index(drop=True)

    if len(series) < max(MIN_HISTORY_DAYS, 3 * _SEASON):
        fc = _seasonal_naive(series, horizon)
        return ForecastResult(product_id, series, fc, "seasonal_naive",
                              {"mae": None, "rmse": None, "mape": None, "samples": len(series)})

    metrics, resid_std = _backtest(series)
    try:
        fit = _fit_holt_winters(series["quantity"])
        yhat = np.clip(fit.forecast(horizon), 0, None)
    except Exception:
        fc = _seasonal_naive(series, horizon)
        return ForecastResult(product_id, series, fc, "seasonal_naive", metrics)

    future = pd.date_range(series["date"].max() + pd.Timedelta(days=1),
                           periods=horizon, freq="D")
    fc = pd.DataFrame({
        "date": future,
        "yhat": np.round(np.asarray(yhat, dtype=float), 2),
    })
    fc["yhat_lower"] = np.clip(fc["yhat"] - 1.64 * resid_std, 0, None).round(2)
    fc["yhat_upper"] = (fc["yhat"] + 1.64 * resid_std).round(2)
    return ForecastResult(product_id, series, fc, MODEL_NAME, metrics)
