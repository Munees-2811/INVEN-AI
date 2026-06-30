"""
AI demand forecasting.

A gradient-boosted regressor trained on engineered lag/calendar features, used
recursively to roll forward an N-day forecast. Falls back to a seasonal-naive
baseline for products with too little history. Produces point forecasts plus an
empirical prediction interval derived from backtest residuals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import FORECAST_HORIZON_DAYS, MIN_HISTORY_DAYS
from src.data.preprocessing import FEATURE_COLUMNS, daily_demand, make_features


@dataclass
class ForecastResult:
    product_id: str
    history: pd.DataFrame             # date, quantity
    forecast: pd.DataFrame           # date, yhat, yhat_lower, yhat_upper
    model_name: str
    metrics: dict


def _seasonal_naive(series: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Repeat the trailing weekly pattern — robust baseline for sparse series."""
    s = series.sort_values("date")
    last_date = s["date"].max()
    recent = s["quantity"].tail(7).to_numpy()
    if len(recent) == 0:
        recent = np.array([0.0])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    yhat = np.array([recent[i % len(recent)] for i in range(horizon)], dtype=float)
    std = float(s["quantity"].tail(28).std() or 1.0)
    return pd.DataFrame(
        {
            "date": future_dates,
            "yhat": np.round(yhat, 2),
            "yhat_lower": np.clip(yhat - 1.64 * std, 0, None).round(2),
            "yhat_upper": (yhat + 1.64 * std).round(2),
        }
    )


def _backtest_metrics(feat: pd.DataFrame, model: GradientBoostingRegressor, n_test: int = 28) -> tuple[dict, float]:
    """Hold out the last ``n_test`` days to estimate error + residual spread."""
    if len(feat) <= n_test + 10:
        return {"mae": None, "rmse": None, "mape": None, "samples": len(feat)}, 1.0
    train, test = feat.iloc[:-n_test], feat.iloc[-n_test:]
    m = GradientBoostingRegressor(**model.get_params())
    m.fit(train[FEATURE_COLUMNS], train["quantity"])
    pred = np.clip(m.predict(test[FEATURE_COLUMNS]), 0, None)
    actual = test["quantity"].to_numpy()
    mae = float(mean_absolute_error(actual, pred))
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    nonzero = actual > 0
    mape = float(np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else None
    resid_std = float(np.std(actual - pred)) or 1.0
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 2) if mape is not None else None,
        "samples": len(feat),
    }, resid_std


def forecast_product(
    sales: pd.DataFrame,
    product_id: str,
    horizon: int = FORECAST_HORIZON_DAYS,
) -> ForecastResult:
    """Forecast a single product's daily demand for ``horizon`` days ahead."""
    series = daily_demand(sales, product_id=product_id)[["date", "quantity"]]
    series = series.sort_values("date").reset_index(drop=True)

    if len(series) < MIN_HISTORY_DAYS:
        fc = _seasonal_naive(series, horizon)
        return ForecastResult(product_id, series, fc, "seasonal_naive",
                              {"mae": None, "rmse": None, "mape": None, "samples": len(series)})

    feat = make_features(series)
    if len(feat) < 20:
        fc = _seasonal_naive(series, horizon)
        return ForecastResult(product_id, series, fc, "seasonal_naive",
                              {"mae": None, "rmse": None, "mape": None, "samples": len(series)})

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=42
    )
    metrics, resid_std = _backtest_metrics(feat, model)
    model.fit(feat[FEATURE_COLUMNS], feat["quantity"])

    # recursive multi-step forecast
    work = series.copy()
    preds = []
    for _ in range(horizon):
        f = make_features(work)
        if f.empty:
            break
        x = f[FEATURE_COLUMNS].iloc[[-1]]
        next_date = work["date"].max() + pd.Timedelta(days=1)
        yhat = float(np.clip(model.predict(x)[0], 0, None))
        preds.append((next_date, yhat))
        work = pd.concat(
            [work, pd.DataFrame({"date": [next_date], "quantity": [yhat]})], ignore_index=True
        )

    fc_df = pd.DataFrame(preds, columns=["date", "yhat"])
    fc_df["yhat"] = fc_df["yhat"].round(2)
    fc_df["yhat_lower"] = np.clip(fc_df["yhat"] - 1.64 * resid_std, 0, None).round(2)
    fc_df["yhat_upper"] = (fc_df["yhat"] + 1.64 * resid_std).round(2)

    return ForecastResult(product_id, series, fc_df, "gradient_boosting", metrics)


def forecast_summary(sales: pd.DataFrame, product_id: str, horizon: int = FORECAST_HORIZON_DAYS) -> dict:
    """Compact dict used by reorder logic and dashboards."""
    res = forecast_product(sales, product_id, horizon)
    daily_mean = float(res.forecast["yhat"].mean()) if not res.forecast.empty else 0.0
    return {
        "product_id": product_id,
        "model": res.model_name,
        "horizon_days": horizon,
        "forecast_total": round(float(res.forecast["yhat"].sum()), 1),
        "forecast_daily_mean": round(daily_mean, 2),
        "forecast_daily_std": round(float(res.forecast["yhat"].std() or 0.0), 2),
        "metrics": res.metrics,
    }
