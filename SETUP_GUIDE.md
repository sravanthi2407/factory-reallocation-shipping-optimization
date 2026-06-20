# 🚀 Nassau Candy — Local Setup Guide

Streamlit cannot be installed in the Claude cloud environment (no outbound pip access).
Run everything locally in 3 steps.

---

## Step 1 — Extract the project

```bash
tar -xzf nassau_candy_project.tar.gz
cd nassau_candy
```

---

## Step 2 — Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn matplotlib xgboost shap scipy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

> ✅ Python 3.9+ recommended. Works on Windows, Mac, Linux.

---

## Step 3 — Run the app

### Option A — Single-file app (simplest, best for Streamlit Cloud)
```bash
streamlit run app.py
```

### Option B — Modular multi-page app (best for GitHub portfolio)
```bash
streamlit run pages/1_Executive_Overview.py
```

The browser opens automatically at **http://localhost:8501**

---

## One-time model training (already done — skip if models/ folder exists)

```bash
python train_model.py
```

---

## Deploy to Streamlit Cloud (free)

1. Push the project folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Click **New App** → select your repo → set **Main file path** = `app.py`
4. Click **Deploy** — live URL in ~2 minutes

Make sure your repo includes:
- `data/Nassau_Candy_Distributor.csv`
- `models/best_model.pkl`
- `requirements.txt`

---

## Project structure

```
nassau_candy/
├── app.py                    ← SINGLE-FILE app (run this)
├── pages/                    ← Modular pages (alternative)
│   ├── 1_Executive_Overview.py
│   ├── 2_Factory_Network_Analysis.py
│   ├── 3_Lead_Time_Prediction.py
│   ├── 4_Factory_Optimization_Simulator.py
│   ├── 5_Recommendation_Dashboard.py
│   ├── 6_WhatIf_Scenario_Analysis.py
│   └── 7_Risk_Assessment.py
├── utils/
│   ├── shared.py             ← Cached loaders, helpers
│   └── styles.py             ← CSS dark theme
├── preprocessing.py
├── feature_engineering.py
├── train_model.py
├── optimization_engine.py
├── recommendation_engine.py
├── scenario_simulator.py
├── data/
│   └── Nassau_Candy_Distributor.csv
├── models/
│   ├── best_model.pkl
│   ├── model_results.csv
│   └── feature_importance.csv
├── requirements.txt
├── README.md
└── RESEARCH_PAPER.md
```

---

## Dashboard pages

| Page | What you see |
|------|-------------|
| 🏠 Executive Overview | KPI cards, revenue trends, factory summary |
| 🏭 Factory Network | Orders/revenue per factory, heatmaps, distance matrix |
| ⏱️ Lead Time Prediction | Model scorecard, feature importance, live predictor |
| ⚙️ Optimization Simulator | 75-scenario heatmap, per-product drill-down |
| 🎯 Recommendations | Top-20 table, score bars, download button |
| 🔮 What-If Analysis | Product × Region × Ship Mode simulator |
| ⚠️ Risk Assessment | Risk matrix, confidence analysis, mitigation playbook |
