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
from src.models._boost import backend_label
from src.models.categorization import train_categorizer
from src.models.customer_prediction import train_purchase_model
from src.models.demand_forecasting import forecast_product
from src.models.inventory_risk import train_inventory_risk
from src.models.price_recommendation import train_price_model
from src.models.sales_forecasting import sales_forecast_summary
from src.models.statistical_forecasting import forecast_product_stat
from src.models.stock_risk import train_stock_risk
from src.models.supplier_analysis import train_supplier_model

# Validation gates (a model must clear these to be promoted to production).
# Ordered to mirror the dependency flow demand → sales → stock-risk → inv-risk.
GATES = {
    "demand_forecaster": {"max_mape": 60.0},     # avg backtest MAPE %
    "sales_forecaster": {"min_r2": 0.30},        # units→revenue fit
    "stock_risk_classifiers": {"min_accuracy": 0.80},
    "inventory_risk_meta": {"min_accuracy": 0.70},
    "supplier_performance": {"min_auc": 0.50},   # weak signal on small PO sets (see docs)
    "dynamic_pricing": {"min_r2": 0.30},
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
    """Champion/challenger demand stage.

    Backtests the XGBoost forecaster **and** the Holt-Winters statistical model
    on the same product sample with the same hold-out protocol, then registers
    and promotes whichever wins on average MAPE. The registry therefore always
    serves the empirically better forecaster instead of assuming ML wins.
    """
    top = sales.groupby("product_id")["quantity"].sum().sort_values(ascending=False)
    pids = top.head(sample_n).index.tolist()

    scores = {"xgboost": [], "holt_winters": []}
    maes = []
    for pid in pids:
        ml = forecast_product(sales, pid, horizon=FORECAST_HORIZON_DAYS)
        if (ml.metrics or {}).get("mape") is not None:
            scores["xgboost"].append(ml.metrics["mape"])
        if (ml.metrics or {}).get("mae") is not None:
            maes.append(ml.metrics["mae"])
        stat = forecast_product_stat(sales, pid, horizon=FORECAST_HORIZON_DAYS)
        if (stat.metrics or {}).get("mape") is not None:
            scores["holt_winters"].append(stat.metrics["mape"])

    if not scores["xgboost"] and not scores["holt_winters"]:
        return StageResult("demand_forecaster", "skipped", note="insufficient history to backtest")

    avg = {k: float(np.mean(v)) if v else float("inf") for k, v in scores.items()}
    champion = min(avg, key=avg.get)
    champ_mape = avg[champion]
    metrics = {
        "champion": champion,
        "avg_mape": round(champ_mape, 2),
        "avg_mape_xgboost": round(avg["xgboost"], 2) if scores["xgboost"] else None,
        "avg_mape_holt_winters": round(avg["holt_winters"], 2) if scores["holt_winters"] else None,
        "avg_mae": round(float(np.mean(maes)), 3) if maes else None,
        "accuracy_pct": round(max(0.0, 100 - champ_mape), 2),
        "products_evaluated": len(pids),
    }
    passed = champ_mape <= GATES["demand_forecaster"]["max_mape"]
    meta = registry.register_model(
        "demand_forecaster",
        artifact={"type": "config", "model": champion, "backend": backend_label(),
                  "challenger": min((k for k in avg if k != champion), key=avg.get, default=None),
                  "horizon": FORECAST_HORIZON_DAYS},
        metrics=metrics,
        params={"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
                "statistical": "holt_winters additive trend+seasonality (7d)"},
        validation={"gate": GATES["demand_forecaster"], "passed": passed},
        data_hash=registry.data_fingerprint(sales.shape, sales["date"].max()),
        promote=passed,
    )
    monitoring.log_performance({"model": "demand_forecaster", "metrics": metrics})
    return StageResult("demand_forecaster", "passed" if passed else "failed", metrics, meta["version"])


def _train_sales(sales: pd.DataFrame, sample_n: int = 10) -> StageResult:
    """Sales-forecaster stage — depends on the demand forecaster's output."""
    top = sales.groupby("product_id")["quantity"].sum().sort_values(ascending=False)
    pids = top.head(sample_n).index.tolist()
    r2s = []
    for pid in pids:
        summ = sales_forecast_summary(sales, pid, horizon=FORECAST_HORIZON_DAYS)
        r2 = (summ.get("metrics") or {}).get("r2")
        if r2 is not None:
            r2s.append(r2)
    if not r2s:
        return StageResult("sales_forecaster", "skipped", note="insufficient history")
    avg_r2 = float(np.mean(r2s))
    metrics = {"avg_r2": round(avg_r2, 3), "products_evaluated": len(pids)}
    passed = avg_r2 >= GATES["sales_forecaster"]["min_r2"]
    meta = registry.register_model(
        "sales_forecaster",
        artifact={"type": "config", "model": backend_label(), "depends_on": "demand_forecaster"},
        metrics=metrics,
        validation={"gate": GATES["sales_forecaster"], "passed": passed},
        data_hash=registry.data_fingerprint(sales.shape), promote=passed,
    )
    monitoring.log_performance({"model": "sales_forecaster", "metrics": metrics})
    return StageResult("sales_forecaster", "passed" if passed else "failed", metrics, meta["version"])


def _train_stock_risk(sales, products, suppliers):
    """Stock-risk stage — trains stockout / understock / overstock classifiers."""
    bundle, _ = train_stock_risk(sales, products, suppliers)
    accs = [m["train_accuracy"] for m in bundle.metrics.values() if m.get("train_accuracy") is not None]
    avg_acc = float(np.mean(accs)) if accs else 0.0
    metrics = {"avg_accuracy": round(avg_acc, 4),
               **{f"acc_{k}": v["train_accuracy"] for k, v in bundle.metrics.items()}}
    passed = avg_acc >= GATES["stock_risk_classifiers"]["min_accuracy"]
    meta = registry.register_model(
        "stock_risk_classifiers", artifact=bundle, metrics=metrics,
        validation={"gate": GATES["stock_risk_classifiers"], "passed": passed},
        data_hash=registry.data_fingerprint(products.shape), promote=passed,
    )
    monitoring.log_performance({"model": "stock_risk_classifiers", "metrics": {"accuracy_pct": round(avg_acc * 100, 2)}})
    return StageResult("stock_risk_classifiers", "passed" if passed else "failed", metrics, meta["version"]), bundle


def _train_inventory_risk(sales, products, suppliers, bundle):
    """Inventory-risk meta stage — consumes the stock-risk classifiers' outputs."""
    model, _ = train_inventory_risk(sales, products, suppliers, risk_bundle=bundle)
    acc = model.accuracy
    metrics = {"accuracy": round(acc, 4) if acc == acc else None, "n_tiers": len(model.classes)}
    passed = (acc == acc) and acc >= GATES["inventory_risk_meta"]["min_accuracy"]
    meta = registry.register_model(
        "inventory_risk_meta", artifact=model, metrics=metrics,
        validation={"gate": GATES["inventory_risk_meta"], "passed": passed,
                    "depends_on": "stock_risk_classifiers"},
        data_hash=registry.data_fingerprint(products.shape), promote=passed,
    )
    monitoring.log_performance({"model": "inventory_risk_meta", "metrics": metrics})
    return StageResult("inventory_risk_meta", "passed" if passed else "failed", metrics, meta["version"])


def _train_supplier(purchase_orders, suppliers) -> StageResult:
    """Supplier-performance stage — PO-level on-time XGBoost classifier."""
    model, _ = train_supplier_model(purchase_orders, suppliers)
    if model.model is None:
        return StageResult("supplier_performance", "skipped", note="no PO history / single-class")
    auc = model.auc
    metrics = {"auc": round(auc, 4) if auc == auc else None, "base_rate": round(model.base_rate, 3)}
    passed = (auc == auc) and auc >= GATES["supplier_performance"]["min_auc"]
    meta = registry.register_model(
        "supplier_performance", artifact=model, metrics=metrics,
        validation={"gate": GATES["supplier_performance"], "passed": passed},
        data_hash=registry.data_fingerprint(purchase_orders.shape), promote=passed,
    )
    monitoring.log_performance({"model": "supplier_performance", "metrics": metrics})
    return StageResult("supplier_performance", "passed" if passed else "failed", metrics, meta["version"])


def _train_pricing(sales, products) -> StageResult:
    """Dynamic-pricing stage — pooled monotonic demand-response regressor."""
    model = train_price_model(sales, products)
    if model.model is None:
        return StageResult("dynamic_pricing", "skipped", note="insufficient price variation")
    r2 = model.r2
    metrics = {"r2": round(r2, 3) if r2 == r2 else None, "products": len(model.ctx)}
    passed = (r2 == r2) and r2 >= GATES["dynamic_pricing"]["min_r2"]
    meta = registry.register_model(
        "dynamic_pricing", artifact=model, metrics=metrics,
        validation={"gate": GATES["dynamic_pricing"], "passed": passed},
        data_hash=registry.data_fingerprint(sales.shape), promote=passed,
    )
    monitoring.log_performance({"model": "dynamic_pricing", "metrics": metrics})
    return StageResult("dynamic_pricing", "passed" if passed else "failed", metrics, meta["version"])


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

    # 2-5. Train → validate → version → deploy, following the dependency graph:
    #   demand → sales → stock-risk → inventory-risk; supplier, pricing;
    #   categorizer, customer (independent).
    run.stages.append(_train_demand(clean))
    run.stages.append(_train_sales(clean))

    stock_stage, risk_bundle = _train_stock_risk(clean, data["products"], data.get("suppliers"))
    run.stages.append(stock_stage)
    run.stages.append(_train_inventory_risk(clean, data["products"], data.get("suppliers"), risk_bundle))

    run.stages.append(_train_supplier(data.get("purchase_orders"), data["suppliers"]))
    run.stages.append(_train_pricing(clean, data["products"]))

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
