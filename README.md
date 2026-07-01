# 📦 INVEN-AI — AI-Powered Inventory Management for MSMEs

An end-to-end, AI-powered inventory management system designed for **Micro, Small &
Medium Enterprises (MSMEs)** — shops, distributors and small retailers — wrapped in a
clean, modern **Streamlit** interface and backed by a real **MLOps pipeline**.

It turns raw sales transactions into decisions: *what to reorder, what's overstocked,
which suppliers are slipping, which customers are about to churn, what to price things
at, and where fraud might be hiding* — and explains it all in plain language through an
AI chatbot, voice assistant and auto-written business reports.

> Works **fully offline** out of the box (synthetic demo data is generated on first run).
> Add an `ANTHROPIC_API_KEY` to unlock the live Claude-powered features.

---

## ✨ Features

| # | Capability | How it works |
|---|------------|--------------|
| 1 | **AI Demand & Sales Forecasting** | XGBoost recursive demand forecaster on lag/calendar features (backtested, 90% interval, seasonal-naive fallback) **plus** a second XGBoost regressor that maps forecasted units → revenue. |
| 2 | **Smart Reorder Recommendations** | Hybrid: rule-based safety stock / reorder point / order-up-to quantities, prioritised by an XGBoost stockout probability (rule ⊕ ML, no redundant model). |
| 3 | **Stockout / Overstock / Understock Prediction** | Three sibling XGBoost classifiers over one shared feature matrix, plus an XGBoost **inventory-risk meta-classifier** that stacks their outputs into a low/medium/high tier. |
| 4 | **AI Sales Trend Analysis** | Trend/seasonality view, rising vs falling movers, ABC (Pareto) revenue analysis. |
| 5 | **AI Chatbot** | Conversational assistant **grounded in a live business snapshot** (Claude), with a rule-based fallback. |
| 6 | **OCR Invoice Scanning** | Claude-vision invoice → structured line items (JSON), editable + CSV export; Tesseract fallback. |
| 7 | **AI Product Categorization** | TF-IDF (word + char n-grams) + Logistic Regression text classifier with cross-validated accuracy. |
| 8 | **Supplier Performance Analysis** | Weighted scorecard (on-time, quality, lead-time reliability, price) **plus** an XGBoost classifier that predicts next-order on-time reliability from purchase-order history. |
| 9 | **Customer Purchase Prediction** | RFM features + XGBoost repurchase/churn classifier (ROC-AUC reported). |
| 10 | **Dynamic Price Recommendations** | Pooled XGBoost demand-response regressor (demand constrained non-increasing in price) → profit-maximising price within guardrails; log-log elasticity fallback for cold-start SKUs. |
| 11 | **Fraud & Anomaly Detection** | Isolation Forest over transaction features + deterministic fraud rules; value-at-risk summary. |
| 12 | **Smart Alerts** | Prioritised, de-duplicated alert feed aggregating every model's signals. |
| 13 | **AI-Generated Business Reports** | Executive weekly briefing written by Claude from assembled metrics (templated offline). |
| 14 | **Voice Assistant** | Grounded Q&A with spoken (gTTS) playback and browser mic capture. |
| 15 | **MLOps Pipeline** | preprocess → train → validate (quality gates) → version → deploy → monitor, with drift detection & performance tracking. |

---

## 🏗️ Architecture

```
INVEN-AI/
├── app.py                     # Streamlit entry point (st.navigation multi-page)
├── config.py                  # Central config (env / secrets driven)
├── data/
│   └── generate_synthetic_data.py   # Realistic MSME dataset (trend, seasonality, churn, fraud)
├── src/
│   ├── data/                  # loader + automated preprocessing / feature engineering
│   ├── models/                # forecasting, inventory, trends, categorization,
│   │                          #   suppliers, customers, pricing, anomalies, alerts
│   ├── ai/                    # Claude client, chatbot, OCR, reports, voice
│   ├── mlops/                 # registry (versioning), pipeline, monitoring (drift/perf)
│   └── utils/                 # stats helpers
├── ui/
│   ├── state.py               # cached data access
│   ├── components/charts.py   # Plotly + KPI helpers
│   └── views/                 # one module per page
├── models_registry/           # file-based model registry + performance log (generated)
└── tests/                     # pytest smoke tests for the full stack
```

### MLOps lifecycle

```
            ┌─────────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐
 raw data → │ Preprocess  │ → │  Train   │ → │ Validate   │ → │ Version  │ → │  Deploy  │ → │  Monitor   │
            │ clean+feats │   │ 3 models │   │ qual. gate │   │ registry │   │ promote  │   │ drift/perf │
            └─────────────┘   └──────────┘   └────────────┘   └──────────┘   └──────────┘   └────────────┘
```

* **Validation gates** — a model is only promoted to *production* if it clears its gate
  (e.g. forecaster avg backtest MAPE ≤ 60%, categorizer accuracy ≥ 0.65, customer AUC ≥ 0.65).
* **Versioning** — every run writes `models_registry/artifacts/<model>/v<N>/` with the
  serialized artifact + `metadata.json` (metrics, params, data hash, stage).
* **Monitoring** — Population Stability Index (PSI) drift on demand + a performance log
  that powers the accuracy-over-time chart and a retrain recommendation.

---

## 🚀 Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. (optional) enable live AI features
cp .env.example .env        # then paste your ANTHROPIC_API_KEY

# 3. run — synthetic data is generated automatically on first launch
streamlit run app.py
```

Open <http://localhost:8501>. Use the sidebar to regenerate demo data, and the
**MLOps & Monitoring** page to run the training pipeline and watch models register.

### Run the tests

```bash
pytest -q
```

### Generate data manually (optional)

```bash
python -m data.generate_synthetic_data
```

---

## 🤖 AI configuration

The AI features call the **Claude API** via the official `anthropic` SDK and default to
the latest models:

| Setting | Default | Purpose |
|---------|---------|---------|
| `CLAUDE_MODEL` | `claude-opus-4-8` | chatbot, reports |
| `CLAUDE_MODEL_FAST` | `claude-haiku-4-5-20251001` | quick utility calls |
| `CLAUDE_VISION_MODEL` | `claude-opus-4-8` | invoice OCR |

Every AI feature **degrades gracefully**: without a key the chatbot uses a keyword
router, reports use a structured template, and OCR falls back to Tesseract — the app
stays 100% functional.

---

## 🧰 Development workflow (Claude Code ⟷ GitHub ⟷ PyCharm)

This project is built to fit the **Claude Code → GitHub → PyCharm** loop:

* **PyCharm** — open the folder, mark `INVEN-AI` as Sources Root, point the run config at
  `streamlit run app.py`. The package layout (`src/`, `ui/`) is import-clean from root.
* **Claude Code** — used to scaffold, refactor and extend modules; each model lives in its
  own file so changes stay localized.
* **GitHub** — feature branches per change; the `tests/` suite gives a fast CI gate.

---

## 📊 Tech stack

`Streamlit` · `pandas` / `numpy` · `XGBoost` · `scikit-learn` · `statsmodels` · `Plotly` ·
`Anthropic Claude` · `Pillow` / `pytesseract` · `gTTS` · file-based MLOps registry.

See [`docs/MODEL_ARCHITECTURE.md`](docs/MODEL_ARCHITECTURE.md) for the full
feature → model map, prediction dependency flow, and consistency review.

---

## ⚠️ Notes

* The included dataset is **synthetic** and generated locally — engineered with real
  trend, weekly/yearly seasonality, promotions, price elasticity, supplier behaviour,
  customer churn and injected anomalies so every model has genuine signal to learn.
* To use **your own data**, drop CSVs into `data/` matching the schemas in
  `data/generate_synthetic_data.py` (or adapt `src/data/loader.py`).
