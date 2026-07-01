"""
Central gradient-boosting backend for every supervised model in the system.

The whole platform standardises on **XGBoost** (per the model architecture
spec): regression tasks use an ``XGBRegressor`` and classification tasks use an
``XGBClassifier``. To keep the offline / zero-network promise intact, this
module transparently falls back to scikit-learn's gradient-boosting estimators
when XGBoost is not importable — the calling code never has to care which
backend is live.

Keeping the estimator choice in one place is a deliberate separation-of-concerns
decision: individual model modules describe *what* they predict and *which
features* they use, while this factory owns *how* the learner is configured.
Swapping or tuning the backend happens here and nowhere else.
"""
from __future__ import annotations

from typing import Any

try:  # primary backend
    from xgboost import XGBClassifier, XGBRegressor

    BACKEND = "xgboost"
    _HAS_XGB = True
except Exception:  # pragma: no cover - exercised only when xgboost is absent
    from sklearn.ensemble import (  # type: ignore
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )

    BACKEND = "gradient_boosting"
    _HAS_XGB = False


# Sane, MSME-scale defaults: small datasets, so shallow trees + mild subsampling
# to curb overfitting. Overridable per-call.
_REG_DEFAULTS: dict[str, Any] = dict(n_estimators=300, max_depth=4,
                                     learning_rate=0.05, subsample=0.9)
_CLF_DEFAULTS: dict[str, Any] = dict(n_estimators=250, max_depth=3,
                                     learning_rate=0.07, subsample=0.9)

# kwargs that only XGBoost understands — dropped on the sklearn fallback path.
_XGB_ONLY = ("colsample_bytree", "tree_method", "n_jobs", "eval_metric",
             "objective", "reg_lambda", "min_child_weight", "verbosity",
             "use_label_encoder", "monotone_constraints")


def _sklearn_safe(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k not in _XGB_ONLY}


def boosted_regressor(**overrides: Any):
    """Return a configured gradient-boosted regressor (XGBoost or fallback)."""
    params = {**_REG_DEFAULTS, **overrides}
    if _HAS_XGB:
        return XGBRegressor(
            random_state=42,
            n_jobs=2,
            objective="reg:squarederror",
            tree_method="hist",
            colsample_bytree=0.9,
            verbosity=0,
            **params,
        )
    return GradientBoostingRegressor(random_state=42, **_sklearn_safe(params))


def boosted_classifier(**overrides: Any):
    """Return a configured gradient-boosted classifier (XGBoost or fallback)."""
    params = {**_CLF_DEFAULTS, **overrides}
    if _HAS_XGB:
        return XGBClassifier(
            random_state=42,
            n_jobs=2,
            eval_metric="logloss",
            tree_method="hist",
            colsample_bytree=0.9,
            verbosity=0,
            **params,
        )
    return GradientBoostingClassifier(random_state=42, **_sklearn_safe(params))


def backend_label() -> str:
    """Human-readable id of the active backend, used in model metadata/UI."""
    return BACKEND
