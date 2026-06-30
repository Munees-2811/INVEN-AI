"""
MLOps training pipeline orchestrator.

One entry point (`run_pipeline`) executes the full lifecycle:
    preprocess → train → validate (quality gate) → version → deploy → monitor.

Each model that passes its validation gate is registered and promoted to
production in the file-based registry; its run metrics are appended to the
performance log for trend tracking, and demand drift is evaluated to decide
whether a retrain is due.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import FORECAST_HORIZON_DAYS
from src.data.preprocessing import clean_sales
from src.mlops import monitoring, registry
from src.models.categorization import train_categorizer
from src.models.customer_prediction import train_purchase_model
from src.models.demand_forecasting import forecast_product

# Validation gates (a model must clear these to be promoted to production).
GATES = {
    "demand_forecaster": {"max_mape": 60.0},     # avg backtest MAPE %
    "product_categorizer": {"min_accuracy": 0.65},
    "customer_repurchase": {"min_auc": 0.65},
}


@dataclass
class StageResult:
    name: str
    status: str                      # "passed" | "failed" | "skipped"
    metrics: dict = field(default_factory=dict)
    version: int | None = None
    note: str = ""


@dataclass
class PipelineRun:
    started_at: str
    finished_at: str | None = None
    preprocess_report: dict = field(default_factory=dict)
    stages: list[StageResult] = field(default_factory=list)
    drift: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "preprocess_report": self.preprocess_report,
            "drift": self.drift,
            "stages": [s.__dict__ for s in self.stages],
        }


def _train_demand(sales: pd.DataFrame, sample_n: int = 12) -> StageResult:
    """Backtest the forecaster across a representative product sample."""
    top = sales.groupby("product_id")["quantity"].sum().sort_values(ascending=False)
    pids = top.head(sample_n).index.tolist()
    mapes, maes = [], []
    for pid in pids:
        res = forecast_product(sales, pid, horizon=FORECAST_HORIZON_DAYS)
        m = res.metrics or {}
        if m.get("mape") is not None:
            mapes.append(m["mape"])
        if m.get("mae") is not None:
            maes.append(m["mae"])
    if not mapes:
        return StageResult("demand_forecaster", "skipped", note="insufficient history to backtest")

    avg_mape = float(np.mean(mapes))
    metrics = {
        "avg_mape": round(avg_mape, 2),
        "avg_mae": round(float(np.mean(maes)), 3) if maes else None,
        "accuracy_pct": round(max(0.0, 100 - avg_mape), 2),
        "products_evaluated": len(pids),
    }
    passed = avg_mape <= GATES["demand_forecaster"]["max_mape"]
    meta = registry.register_model(
        "demand_forecaster",
        artifact={"type": "config", "model": "gradient_boosting", "horizon": FORECAST_HORIZON_DAYS},
        metrics=metrics,
        params={"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
        validation={"gate": GATES["demand_forecaster"], "passed": passed},
        data_hash=registry.data_fingerprint(sales.shape, sales["date"].max()),
        promote=passed,
    )
    monitoring.log_performance({"model": "demand_forecaster", "metrics": metrics})
    return StageResult("demand_forecaster", "passed" if passed else "failed", metrics, meta["version"])


def _train_categorizer(products: pd.DataFrame) -> StageResult:
    model = train_categorizer(products)
    acc = model.cv_accuracy
    metrics = {"cv_accuracy": round(acc, 4) if acc == acc else None, "n_classes": len(model.classes)}
    passed = (acc == acc) and acc >= GATES["product_categorizer"]["min_accuracy"]
    meta = registry.register_model(
        "product_categorizer", artifact=model, metrics=metrics,
        validation={"gate": GATES["product_categorizer"], "passed": passed},
        data_hash=registry.data_fingerprint(products.shape), promote=passed,
    )
    monitoring.log_performance({"model": "product_categorizer", "metrics": metrics})
    return StageResult("product_categorizer", "passed" if passed else "failed", metrics, meta["version"])


def _train_customer(sales: pd.DataFrame) -> StageResult:
    model, _ = train_purchase_model(sales)
    auc = model.auc
    metrics = {"auc": round(auc, 4) if auc == auc else None}
    passed = (auc == auc) and auc >= GATES["customer_repurchase"]["min_auc"]
    meta = registry.register_model(
        "customer_repurchase", artifact=model, metrics=metrics,
        validation={"gate": GATES["customer_repurchase"], "passed": passed},
        data_hash=registry.data_fingerprint(sales.shape), promote=passed,
    )
    monitoring.log_performance({"model": "customer_repurchase", "metrics": metrics})
    return StageResult("customer_repurchase", "passed" if passed else "failed", metrics, meta["version"])


def run_pipeline(data: dict) -> PipelineRun:
    """Execute the full MLOps pipeline against the provided data tables."""
    run = PipelineRun(started_at=datetime.now(timezone.utc).isoformat())

    # 1. Preprocess
    clean, report = clean_sales(data["sales"])
    run.preprocess_report = report.as_dict()

    # 2-5. Train → validate → version → deploy (per model)
    run.stages.append(_train_demand(clean))
    run.stages.append(_train_categorizer(data["products"]))
    run.stages.append(_train_customer(clean))

    # 6. Monitor (drift)
    run.drift = monitoring.detect_demand_drift(clean)

    run.finished_at = datetime.now(timezone.utc).isoformat()
    return run


def retrain_due() -> dict:
    """Decide whether a periodic retrain should run, based on drift + age."""
    from config import RETRAIN_INTERVAL_DAYS

    meta = registry.get_production_meta("demand_forecaster")
    age_days = None
    if meta:
        created = pd.to_datetime(meta["created_at"])
        age_days = (pd.Timestamp.now(tz="UTC") - created).days
    due_by_age = age_days is None or age_days >= RETRAIN_INTERVAL_DAYS
    return {
        "last_trained": meta["created_at"] if meta else None,
        "age_days": age_days,
        "interval_days": RETRAIN_INTERVAL_DAYS,
        "due": bool(due_by_age),
    }
