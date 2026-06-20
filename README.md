# 🍬 Nassau Candy — Factory Reallocation & Shipping Optimization System

> **Decision Intelligence System** · Data Science Portfolio Project

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Project Overview

Nassau Candy Distributor currently assigns products to factories using **static, rules-based logic** — causing suboptimal shipping distances, high lead times, and margin erosion. This project builds a complete **Decision Intelligence System** that:

- **Predicts** shipping lead time using ML (R² = 0.717)
- **Simulates** all factory assignment alternatives for every product
- **Scores** each reallocation scenario using a composite formula
- **Recommends** the best factory per product with confidence and risk levels
- **Visualises** everything in a 7-page interactive Streamlit dashboard

---

## 🎯 Problem Statement

| Current State | Target State |
|---|---|
| Static factory assignments | Data-driven, optimised allocations |
| Unknown shipping performance | ML-predicted lead times |
| No scenario planning | What-If simulation engine |
| No risk visibility | Full risk matrix with mitigation playbook |

---

## 📊 Dataset

| Attribute | Value |
|---|---|
| File | `Nassau_Candy_Distributor.csv` |
| Rows | 10,194 orders |
| Columns | 18 |
| Date Range | Jan 2024 – Dec 2025 |
| Products | 15 unique Wonka-brand products |
| Factories | 5 manufacturing locations |
| Divisions | Chocolate (96.6%), Other (3.0%), Sugar (0.4%) |
| Customers | 5,044 unique |
| Revenue | $141,784 |
| Gross Profit | $93,443 (65.9% margin) |

### Columns
`Row ID · Order ID · Order Date · Ship Date · Ship Mode · Customer ID · Country/Region · City · State/Province · Postal Code · Division · Region · Product ID · Product Name · Sales · Units · Gross Profit · Cost`

---

## 🏭 Factory Network

| Factory | Latitude | Longitude | Products |
|---|---|---|---|
| Lot's O' Nuts | 32.88°N | 111.77°W | 3 Chocolate variants |
| Wicked Choccy's | 32.08°N | 81.09°W | 2 Chocolate variants |
| Sugar Shack | 48.12°N | 96.18°W | Laffy Taffy, SweeTARTS, Nerds, Fun Dip, Fizzy Lifting Drinks |
| Secret Factory | 41.45°N | 90.57°W | Gobstopper, Lickable Wallpaper, Wonka Gum |
| The Other Factory | 35.12°N | 89.97°W | Hair Toffee, Kazookles |

---

## 🏗️ Architecture

```
nassau_candy/
│
├── data/
│   └── Nassau_Candy_Distributor.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── model_results.csv
│   └── feature_importance.csv
│
├── pages/                          ← Modular Streamlit pages
│   ├── 1_Executive_Overview.py
│   ├── 2_Factory_Network_Analysis.py
│   ├── 3_Lead_Time_Prediction.py
│   ├── 4_Factory_Optimization_Simulator.py
│   ├── 5_Recommendation_Dashboard.py
│   ├── 6_WhatIf_Scenario_Analysis.py
│   └── 7_Risk_Assessment.py
│
├── utils/
│   ├── shared.py                   ← Cached data loaders, KPI helpers
│   └── styles.py                   ← CSS theme
│
├── preprocessing.py                ← Data cleaning, factory coords, haversine
├── feature_engineering.py          ← ML features, route stats, product summary
├── train_model.py                  ← 4-model ML pipeline, auto model selection
├── optimization_engine.py          ← All-factory simulation + scoring
├── recommendation_engine.py        ← Top-20 recommendations + exec summary
├── scenario_simulator.py           ← What-If scenario engine
├── app.py                          ← Single-file Streamlit app (deployable)
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### Phase 1 — Data Engineering
- Parsed `Order Date` / `Ship Date` (DD-MM-YYYY format)
- Re-engineered `Lead_Time` from `Ship Mode` (original ship dates corrupted to 2030)
- Mapped all 15 products to their 5 factories
- Derived state centroid coordinates for haversine distance computation

### Phase 2 — Feature Engineering
| Feature | Description |
|---|---|
| `Lead_Time` | Synthetic: Based on Ship Mode (Same Day=1d, First=2d, Second=4d, Standard=6d + noise) |
| `Distance_Miles` | Haversine: Factory coords → State centroid |
| `Profit_Margin` | Gross Profit / Sales |
| `Profit_Per_Unit` | Gross Profit / Units |
| `Route_Efficiency_Score` | 0–1 composite (distance + lead time normalised) |
| `Cost_Per_Mile` | Cost / Distance |

### Phase 3 — Machine Learning
**Target:** `Lead_Time`  
**Models Trained:**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Random Forest** ✅ | **1.065** | **0.771** | **0.717** |
| Linear Regression | 1.073 | 0.802 | 0.713 |
| Gradient Boosting | 1.080 | 0.786 | 0.709 |
| XGBoost | 1.068 | 0.774 | 0.716 |

**Top Feature:** `Ship_Mode_Code` — 94.5% importance

### Phase 4 — Reallocation Engine
For each of 15 products × 5 factories = **75 simulations**:
1. Predict lead time with candidate factory
2. Compute haversine distance to customer base
3. Apply composite scoring formula:

```
Score = 0.50 × Lead_Time_Improvement
      + 0.30 × Distance_Reduction  
      + 0.20 × Profit_Stability
```

### Phase 5 — Risk Classification
| Score | Risk Level | Action |
|---|---|---|
| ≥65 | 🟢 Low Risk | Proceed |
| 50–64 | 🟡 Moderate Risk | Pilot first |
| <50 | 🔴 High Risk | Defer |
| Any + Low Data | ⚪ Low Data Confidence | A/B test |

---

## 📈 Key Results

| KPI | Value |
|---|---|
| Products with positive LT improvement | 7 out of 15 |
| Avg lead time improvement (reallocation candidates) | 4.5% |
| Avg distance reduction | 14.5% |
| Estimated cost savings | $2,097 (based on distance proxy) |
| Best reallocation candidate | Fizzy Lifting Drinks → Secret Factory (Score: 64.9) |

---

## 🖥️ Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Executive Overview | KPI cards, revenue trends, division breakdown, factory summary |
| 🏭 Factory Network Analysis | Orders/revenue by factory, LT/distance boxplots, product heatmap |
| ⏱️ Lead Time Prediction | Model scorecard, feature importance, interactive predictor |
| ⚙️ Optimization Simulator | Full 75-simulation table, score heatmap, per-product drill-down |
| 🎯 Recommendation Dashboard | Top-20 ranked recommendations with download |
| 🔮 What-If Scenario Analysis | Product × Region × Ship Mode simulator with visual comparison |
| ⚠️ Risk Assessment | Risk distribution, data confidence, margin stability, risk matrix |

---

## 🚀 Installation & Running

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/nassau-candy-optimization.git
cd nassau-candy-optimization

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the ML model (run once)
python train_model.py

# 4a. Run the single-file app
streamlit run app.py

# 4b. Run the modular app (multi-page)
streamlit run pages/1_Executive_Overview.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11+ |
| Data | pandas, numpy |
| Machine Learning | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualisation | matplotlib, Streamlit |
| Distance Computation | Haversine (custom vectorised) |
| Serialisation | pickle |

---

## 💼 Business Impact

- **Operational:** Reduce average delivery time by ~4.5% for 7 product lines
- **Logistics:** Shorten shipping distance by ~14.5% through smarter factory assignment
- **Financial:** Estimated $2,097 in annual logistics cost savings (distance proxy)
- **Strategic:** Shift from static rules to data-driven factory allocation decisions

---

## 🔮 Future Enhancements

- [ ] Real factory production capacity constraints (capacity-aware optimisation)
- [ ] Actual shipping cost data ($/mile by carrier and mode)
- [ ] Multi-objective optimisation (Pareto frontier: cost vs. speed)
- [ ] Time-series demand forecasting per product × region
- [ ] Customer-level lat/lon for precise distance computation
- [ ] Live ERP data integration via API
- [ ] XGBoost SHAP waterfall plots per prediction

---

## 📄 Project Structure Summary

```
Entry Points:
  app.py              → Single deployable Streamlit app
  train_model.py      → Run once to train and persist ML models

Core Engines:
  preprocessing.py         → Clean + feature base
  feature_engineering.py   → Advanced features + ML matrix
  optimization_engine.py   → 75-scenario simulation
  recommendation_engine.py → Top-20 ranked output
  scenario_simulator.py    → What-If analysis

Dashboard:
  pages/1–7_*.py      → Modular Streamlit pages
  utils/shared.py     → Shared loaders + helpers
  utils/styles.py     → CSS dark theme
```

---

## 👤 Author

Built as an industry-level Data Science & Analytics portfolio project.  
Demonstrates: EDA · Feature Engineering · ML Pipeline · Optimisation · Decision Intelligence · Streamlit Dashboard

---

*Nassau Candy Distributor is a fictional business based on Roald Dahl's Wonka universe, used here purely for educational and portfolio purposes.*
