"""
Model & data monitoring.

Implements Population Stability Index (PSI) for drift detection between a
reference window and a recent window, plus a simple performance-tracking log
that records each training run's metrics over time so the dashboard can plot
accuracy trends and flag degradation.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import DRIFT_PSI_THRESHOLD, REGISTRY_DIR

PERF_LOG = REGISTRY_DIR / "performance_log.json"


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI between two distributions. >0.2 typically signals meaningful drift."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(reference, quantiles))
    if len(cuts) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=cuts)
    cur_counts, _ = np.histogram(current, bins=cuts)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_demand_drift(sales: pd.DataFrame, window_days: int = 30) -> dict:
    """Compare recent demand distribution against the prior window.

    Two complementary tests: PSI (magnitude of the shift, binned) and a
    two-sample Kolmogorov–Smirnov test (statistical significance of the shift).
    A retrain is recommended when the shift is material (PSI ≥ threshold) or
    clearly significant with a non-trivial effect (KS p < 0.01, stat > 0.1).
    """
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    end = df["date"].max()
    recent = df[df["date"] > end - pd.Timedelta(days=window_days)]["quantity"].to_numpy()
    reference = df[
        (df["date"] <= end - pd.Timedelta(days=window_days))
        & (df["date"] > end - pd.Timedelta(days=2 * window_days))
    ]["quantity"].to_numpy()
    psi = population_stability_index(reference, recent)

    ks_stat, ks_pvalue = 0.0, 1.0
    if len(reference) >= 20 and len(recent) >= 20:
        try:
            from scipy.stats import ks_2samp

            res = ks_2samp(reference, recent)
            ks_stat, ks_pvalue = float(res.statistic), float(res.pvalue)
        except Exception:  # scipy hiccup must not break monitoring
            pass

    ks_drift = ks_pvalue < 0.01 and ks_stat > 0.10
    status = "stable"
    if psi >= DRIFT_PSI_THRESHOLD or ks_drift:
        status = "drift_detected"
    elif psi >= DRIFT_PSI_THRESHOLD / 2 or ks_pvalue < 0.05:
        status = "minor_shift"
    return {
        "psi": round(psi, 4),
        "threshold": DRIFT_PSI_THRESHOLD,
        "ks_stat": round(ks_stat, 4),
        "ks_pvalue": round(ks_pvalue, 4),
        "status": status,
        "recommend_retrain": bool(psi >= DRIFT_PSI_THRESHOLD or ks_drift),
        "reference_n": int(len(reference)),
        "recent_n": int(len(recent)),
    }


def log_performance(run: dict) -> None:
    """Append a training-run record to the performance log."""
    history = load_performance()
    run = {**run, "timestamp": datetime.now(timezone.utc).isoformat()}
    history.append(run)
    PERF_LOG.write_text(json.dumps(history, indent=2, default=str))


def load_performance() -> list[dict]:
    if PERF_LOG.exists():
        try:
            return json.loads(PERF_LOG.read_text())
        except Exception:
            return []
    return []


def performance_frame() -> pd.DataFrame:
    rows = load_performance()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "model", "metric", "value"])
    flat = []
    for r in rows:
        ts = r.get("timestamp")
        model = r.get("model", "unknown")
        for k, v in (r.get("metrics") or {}).items():
            if isinstance(v, (int, float)):
                flat.append({"timestamp": ts, "model": model, "metric": k, "value": v})
    return pd.DataFrame(flat)
