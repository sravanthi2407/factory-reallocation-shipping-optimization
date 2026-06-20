"""pages/3_Lead_Time_Prediction.py"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils.shared import (load_data, load_model, set_dark_style,
                           kpi_card, section, insight, COLORS, SHIP_COLORS)
from utils.styles import inject_css
from train_model import ML_FEATURES
from preprocessing import FACTORY_COORDS, PRODUCT_FACTORY
from feature_engineering import build_features

st.set_page_config(page_title="Lead Time Prediction · Nassau Candy", page_icon="⏱️", layout="wide")
inject_css()

df_raw, df_feat, prod_sum, route_stats = load_data()
model, model_name, feature_names       = load_model()
ALL_FACTORIES = list(FACTORY_COORDS.keys())

with st.sidebar:
    filter_ship = st.multiselect("Ship Mode", sorted(df_raw["Ship Mode"].unique()),
                                  default=sorted(df_raw["Ship Mode"].unique()))
df_f = df_feat[df_feat["Ship Mode"].isin(filter_ship)]

st.markdown("## ⏱️ Lead Time Prediction")
st.caption(f"ML Model: **{model_name}** · Target: Shipping Lead Time (days) · R² ≈ 0.717")
set_dark_style()

# Model scorecard
section("Model Performance Scorecard")
try:
    results_df = pd.read_csv("models/model_results.csv")
    c1,c2,c3,c4 = st.columns(4)
    best_row = results_df.iloc[0]
    c1.markdown(kpi_card(model_name, "Best Model"), unsafe_allow_html=True)
    c2.markdown(kpi_card(f"{best_row['RMSE']:.4f}", "RMSE (days)"), unsafe_allow_html=True)
    c3.markdown(kpi_card(f"{best_row['MAE']:.4f}",  "MAE (days)"),  unsafe_allow_html=True)
    c4.markdown(kpi_card(f"{best_row['R2']:.4f}",   "R² Score"),    unsafe_allow_html=True)

    section("All Models Comparison")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, metric, lower_better in zip(axes, ["RMSE","MAE","R2"], [True,True,False]):
        bars = ax.bar(results_df["Model"], results_df[metric],
                      color=[COLORS["primary"]]*len(results_df), alpha=0.85)
        best_idx = results_df[metric].idxmin() if lower_better else results_df[metric].idxmax()
        bars[best_idx].set_color(COLORS["success"])
        ax.set_title(f"{metric} ({'lower' if lower_better else 'higher'} = better)")
        ax.set_xticklabels(results_df["Model"], rotation=20, ha="right", fontsize=7)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
                    f"{b.get_height():.4f}", ha="center", fontsize=7, color=COLORS["text"])
        ax.grid(axis="y")
    fig.tight_layout(); st.pyplot(fig); plt.close()
except FileNotFoundError:
    st.warning("Run `python train_model.py` first to generate model results.")

# Feature importance
section("Feature Importance")
try:
    fi_df = pd.read_csv("models/feature_importance.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    top15 = fi_df.head(15)
    bar_c = [COLORS["success"]] + [COLORS["primary"]]*(len(top15)-1)
    ax.barh(top15["Feature"][::-1], top15["Importance"][::-1], color=bar_c[::-1], alpha=0.9)
    for b in ax.patches:
        ax.text(b.get_width()+0.001, b.get_y()+b.get_height()/2,
                f"{b.get_width():.4f}", va="center", fontsize=7.5, color=COLORS["text"])
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_xlabel("Importance Score"); ax.grid(axis="x")
    fig.tight_layout(); st.pyplot(fig); plt.close()
    insight("Ship_Mode_Code is the overwhelming predictor (94.5% importance). "
            "Distance and state location contribute secondary variance. "
            "This confirms ship mode selection is the key operational lever for lead time control.")
except FileNotFoundError:
    st.warning("Feature importance file not found.")

# Distribution by ship mode
section("Lead Time Distribution by Ship Mode")
fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
for ax, mode in zip(axes, ["Same Day","First Class","Second Class","Standard Class"]):
    vals = df_f[df_f["Ship Mode"]==mode]["Lead_Time"]
    if len(vals) == 0: ax.set_title(mode + "\n(no data)"); continue
    ax.hist(vals, bins=12, color=SHIP_COLORS.get(mode,"#7dd3fc"), alpha=0.85, edgecolor=COLORS["bg"])
    ax.axvline(vals.mean(), color=COLORS["warning"], linewidth=1.5, linestyle="--")
    ax.text(vals.mean()+0.05, ax.get_ylim()[1]*0.88,
            f"μ={vals.mean():.1f}", fontsize=8, color=COLORS["warning"])
    ax.set_title(mode, fontsize=9); ax.set_xlabel("Days"); ax.grid(axis="y")
fig.suptitle("Lead Time Distribution by Ship Mode", fontsize=11, color=COLORS["text"])
fig.tight_layout(); st.pyplot(fig); plt.close()

# Interactive predictor
section("🔢 Interactive Lead Time Predictor")
from preprocessing import haversine_scalar

col1, col2, col3 = st.columns(3)
with col1:
    pred_product  = st.selectbox("Product",   sorted(df_feat["Product Name"].unique()))
    pred_region   = st.selectbox("Region",    sorted(df_feat["Region"].unique()))
with col2:
    pred_shipmode = st.selectbox("Ship Mode", ["Same Day","First Class","Second Class","Standard Class"])
    pred_units    = st.slider("Units", 1, 14, 5)
with col3:
    pred_month   = st.slider("Order Month",   1, 12, 6)
    pred_quarter = st.slider("Order Quarter", 1,  4, 2)

factory = PRODUCT_FACTORY.get(pred_product, ALL_FACTORIES[0])
f_lat   = FACTORY_COORDS[factory]["lat"]
f_lon   = FACTORY_COORDS[factory]["lon"]
subset  = df_feat[df_feat["Product Name"]==pred_product]
slat    = subset["State_Lat"].mean() if len(subset)>0 else 39.5
slon    = subset["State_Lon"].mean() if len(subset)>0 else -98.35
dist_val = haversine_scalar(f_lat, f_lon, slat, slon)

ship_code    = {"Same Day":1,"First Class":2,"Second Class":3,"Standard Class":4}[pred_shipmode]
region_code  = ["Atlantic","Gulf","Interior","Pacific"].index(pred_region) \
               if pred_region in ["Atlantic","Gulf","Interior","Pacific"] else 0
division     = subset["Division"].iloc[0] if len(subset)>0 else "Chocolate"
div_code     = ["Chocolate","Other","Sugar"].index(division) if division in ["Chocolate","Other","Sugar"] else 0
factory_code = ALL_FACTORIES.index(factory)

feature_row = {
    "Ship_Mode_Code":   ship_code,
    "Region_Code":      region_code,
    "Division_Code":    div_code,
    "Factory_Code":     factory_code,
    "Distance_Miles":   dist_val,
    "Units":            pred_units,
    "Sales":            subset["Sales"].mean() if len(subset)>0 else 15.0,
    "Gross Profit":     subset["Gross Profit"].mean() if len(subset)>0 else 10.0,
    "Cost":             subset["Cost"].mean() if len(subset)>0 else 5.0,
    "Profit_Margin":    subset["Profit_Margin"].mean() if len(subset)>0 else 0.65,
    "Profit_Per_Unit":  subset["Profit_Per_Unit"].mean() if len(subset)>0 else 2.0,
    "Revenue_Per_Unit": subset["Revenue_Per_Unit"].mean() if len(subset)>0 else 3.0,
    "Order_Month":      pred_month,
    "Order_Quarter":    pred_quarter,
    "Order_DayOfWeek":  2,
    "Factory_Lat":      f_lat,
    "Factory_Lon":      f_lon,
    "State_Lat":        slat,
    "State_Lon":        slon,
}
X_pred  = pd.DataFrame([feature_row])[ML_FEATURES]
pred_lt = float(model.predict(X_pred)[0])

st.markdown(f"""
<div style="background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #7dd3fc;
            border-radius:12px;padding:28px;text-align:center;margin-top:16px;">
  <div style="font-size:3.8rem;font-weight:800;color:#7dd3fc;">{pred_lt:.1f}</div>
  <div style="font-size:1rem;color:#94a3b8;margin-top:6px;">Predicted Lead Time (days)</div>
  <div style="margin-top:14px;font-size:.85rem;color:#64748b;">
    Factory: <b style="color:#e2e8f0;">{factory}</b> &nbsp;·&nbsp;
    Distance: <b style="color:#e2e8f0;">{dist_val:,.0f} miles</b> &nbsp;·&nbsp;
    Model: <b style="color:#e2e8f0;">{model_name}</b>
  </div>
</div>
""", unsafe_allow_html=True)
