# INVEN-AI — Model Architecture & Assignment

This document records the machine-learning model assigned to each feature, the
inputs each model is allowed to use, the dependency flow between predictions, and
a consistency review with improvement notes.

The platform standardises on **XGBoost** as the gradient-boosting backend for
every supervised task (regression and classification), routed through a single
factory — `src/models/_boost.py` — so the learner choice lives in exactly one
place. If XGBoost is ever unavailable the factory transparently falls back to
scikit-learn gradient boosting, preserving the offline / zero-network promise.

---

## 1. Feature → model map

| # | Feature | Model | Module | Key inputs (only) | Depends on |
|---|---------|-------|--------|-------------------|------------|
| 1 | Demand Forecasting | **XGBoost Regressor** (recursive) — champion/challenger vs **Holt-Winters** (`statistical_forecasting.py`) | `demand_forecasting.py` | lag + calendar features of a product's daily demand | — |
| 2 | Sales Forecasting | **XGBoost Regressor** on demand output | `sales_forecasting.py` | forecasted **units** (from #1) + calendar + recent selling price | #1 Demand |
| 3 | Smart Reordering | **Hybrid**: rule reorder-point ⊕ XGBoost stockout proba, **trend-adjusted** (`reorder_plan`) with supplier-grouped PO drafting (`build_purchase_orders`) | `inventory.py` + `inventory_risk.reorder_hybrid` | rule: demand rate, lead time, service level; trend: 30-day momentum; ML: stockout probability (#4) | #4 Stockout, sales momentum |
| 4a | Stockout Prediction | **XGBoost Classifier** | `stock_risk.py` | shared SKU risk features | #1 demand stats |
| 4b | Understock Prediction | **XGBoost Classifier** | `stock_risk.py` | shared SKU risk features | #1 demand stats |
| 4c | Overstock Prediction | **XGBoost Classifier** | `stock_risk.py` | shared SKU risk features | #1 demand stats |
| 5 | Supplier Performance | **XGBoost Classifier** (PO on-time) + scorecard | `supplier_analysis.py` | PO-level pre-delivery signals + supplier history | — |
| 6 | Customer Purchase Prediction | **XGBoost Classifier** | `customer_prediction.py` | RFM features | — |
| 7 | Dynamic Pricing | **XGBoost Regressor** (monotonic demand-response) | `price_recommendation.py` | relative price, promo, calendar, base demand, cost | — |
| 8 | Inventory Risk Prediction | **XGBoost Classifier (meta)** | `inventory_risk.py` | outputs of #4a/#4b/#4c + demand CV + lead time | #4 Stock-risk |

Product Categorization keeps its TF-IDF + Logistic-Regression text classifier —
XGBoost is not appropriate for short free-text product names, and the task
mapping does not call for it. This is a deliberate separation-of-concerns choice.

---

## 2. Dependency flow

```
                 ┌─────────────────────────┐
                 │  Demand Forecasting (1)  │  XGBoost Regressor
                 └────────────┬────────────┘
             ┌────────────────┼─────────────────────────┐
             ▼                ▼                          ▼
   ┌───────────────────┐  ┌───────────────────┐   demand rate / σ
   │ Sales Forecast (2)│  │ Stock-risk (4a/b/c)│  ┌──────────────────┐
   │ units → revenue   │  │ stockout/under/over│  │ Smart Reorder (3)│
   └───────────────────┘  └─────────┬─────────┘  │ rule reorder-pt  │
                                     │            │  ⊕ p(stockout)   │
                                     ▼            └──────────────────┘
                          ┌────────────────────┐        ▲
                          │ Inventory Risk (8)  │────────┘ (shares p_stockout)
                          │ meta-classifier     │
                          └────────────────────┘

   Independent chains:  Supplier Performance (5) · Customer Purchase (6) ·
                        Dynamic Pricing (7) · Categorization
```

The MLOps pipeline (`src/mlops/pipeline.py`) trains the models **in this
dependency order** and passes the fitted stock-risk bundle straight into the
inventory-risk stage, so the meta-model reuses (never re-derives) its inputs.

---

## 3. Separation of concerns & no-redundancy checks

- **One learner factory.** Every supervised model calls `boosted_regressor` /
  `boosted_classifier`; no module hard-codes an estimator. Backend swaps happen
  in one file.
- **One feature matrix for three risk heads.** Stockout, understock and
  overstock classifiers share a single `build_risk_features` matrix — three
  targets, not three feature pipelines.
- **Reordering reuses the stockout classifier.** The Smart-Reordering hybrid
  does **not** train its own "reorder now" model (which would duplicate the
  stockout classifier). It composes the deterministic reorder point with the
  existing stockout probability. This is the explicit "avoid redundant models"
  requirement.
- **Sales forecasting consumes demand, not raw history twice.** It takes the
  demand model's unit forecast as an input feature rather than re-forecasting
  demand.
- **Inventory-risk is pure stacking.** It reads only the three risk
  probabilities plus two context features — no access to raw plumbing, keeping
  the meta-model decoupled from base-feature changes.

---

## 4. Consistency validation (what was checked)

- ✅ Every model in the map is XGBoost-backed except Categorization (text, by
  design) and the deterministic reorder-point / scorecard rules that the hybrids
  intentionally retain.
- ✅ All existing public function signatures and return shapes are unchanged;
  new capability is additive. UI pages are guarded (`try/except`) so a model
  that cannot train never breaks a page.
- ✅ Full test suite: **22 passing** (13 original + 9 new covering backend,
  dependency flow, monotonic pricing and backward-compatibility).
- ✅ All 15 Streamlit pages render headlessly with no exceptions.
- ✅ MLOps pipeline runs end-to-end: 8 stages, each with a validation gate and a
  registered, versioned artifact; demand-drift monitoring intact.

---

## 5. Improvement notes (efficiency of the assignments)

These are honest observations where an assignment is weak or could be improved —
surfaced rather than hidden:

1. **Stock-risk & inventory-risk labels are near-deterministic.** Because the
   ground-truth labels come from inventory-theory thresholds on clean synthetic
   data, the classifiers reach ~1.0 train accuracy — they are effectively
   *learning the rule*. Their real value is (a) calibrated probabilities for
   ranking/urgency and (b) generalisation once noisy real data (partial counts,
   returns, shrinkage) is introduced. On tiny/clean data a rule engine would be
   just as accurate and cheaper; the ML layer earns its keep as data gets messy.

2. **Supplier on-time AUC is low on small PO sets** (~0.55–0.60 on the demo).
   With only a handful of suppliers and limited PO history there is little signal
   for a learner. Recommended improvements: pool across more history, add
   category/seasonality features, or fall back to the transparent scorecard (the
   code already does this when PO history is single-class/empty). The gate is set
   leniently (`min_auc = 0.50`) and this limitation is documented rather than
   masked.

3. **Dynamic pricing recommends boundary prices.** With the monotonic constraint
   the demand curve is now correctly non-increasing in price, but estimated
   demand is *inelastic* on the synthetic data, so the profit optimum sits at the
   ±guardrail edge for most SKUs (i.e. "raise to the cap"). In production:
   segment elasticity by category, widen/vary guardrails per SKU, and cross-check
   against the log-log elasticity model before applying. The guardrail (default
   ±20%) is the safety net that keeps this actionable today.

4. **Recursive demand forecasting compounds error.** Multi-step recursion feeds
   predictions back as features. A direct multi-horizon model or quantile
   objective would tighten long-horizon intervals; the current empirical interval
   from backtest residuals is a pragmatic stand-in. *Mitigation shipped:* the
   pipeline now backtests a **Holt-Winters** statistical challenger on the same
   hold-out and promotes whichever forecaster wins on MAPE, so a structural
   weakness in either approach is caught by the other.

5. **Drift detection is now two-tests.** PSI (magnitude) plus a two-sample
   Kolmogorov–Smirnov test (significance). A retrain fires on PSI ≥ 0.2 or a
   significant KS result with a non-trivial effect (p < 0.01 and stat > 0.1) —
   the effect-size guard prevents large-N samples from triggering retrains on
   microscopic shifts.

6. **Reordering follows the trend, not just the level.** `reorder_plan` scales
   the rule quantity by damped, capped 30-day momentum (±25% max), and
   `build_purchase_orders` groups drafts per supplier and checks them against
   the supplier's minimum order value before export.
