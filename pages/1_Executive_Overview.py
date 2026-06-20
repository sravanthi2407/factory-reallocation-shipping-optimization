"""
pages/1_Executive_Overview.py
Nassau Candy — Executive Overview (Modular Page)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils.shared import (
    load_data, load_recommendations, load_model,
    kpi_card, insight, section, set_dark_style, COLORS, SHIP_COLORS, FACTORY_COLORS
)

st.set_page_config(page_title="Executive Overview · Nassau Candy", page_icon="🏠", layout="wide")

from utils.styles import inject_css
inject_css()

df_raw, df_feat, prod_sum, route_stats = load_data()
sim_df, best_df, recs_df, exec_sum     = load_recommendations(df_raw)

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/candy.png", width=60)
    st.markdown("### Nassau Candy")
    filter_region   = st.multiselect("Region",   sorted(df_raw["Region"].unique()),
                                     default=sorted(df_raw["Region"].unique()))
    filter_division = st.multiselect("Division", sorted(df_raw["Division"].unique()),
                                     default=sorted(df_raw["Division"].unique()))
    filter_ship     = st.multiselect("Ship Mode",sorted(df_raw["Ship Mode"].unique()),
                                     default=sorted(df_raw["Ship Mode"].unique()))

mask = (
    df_feat["Region"].isin(filter_region) &
    df_feat["Division"].isin(filter_division) &
    df_feat["Ship Mode"].isin(filter_ship)
)
df_f = df_feat[mask]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🏠 Executive Overview")
st.caption("Nassau Candy Distributor · Factory Reallocation & Shipping Optimization System")

# ── KPI row 1 ──────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
for col, val, lbl in [
    (c1, f"{exec_sum['total_orders']:,}",         "Total Orders"),
    (c2, f"${exec_sum['total_revenue']:,.0f}",     "Total Revenue"),
    (c3, f"${exec_sum['total_profit']:,.0f}",      "Gross Profit"),
    (c4, f"{exec_sum['overall_margin_pct']:.1f}%", "Profit Margin"),
    (c5, f"{exec_sum['unique_customers']:,}",      "Unique Customers"),
    (c6, str(exec_sum['products_to_reallocate']),  "Products to Reallocate"),
]:
    col.markdown(kpi_card(val, lbl), unsafe_allow_html=True)

# ── KPI row 2 ──────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.markdown(kpi_card(f"{exec_sum['avg_lt_improvement_pct']:.1f}%", "Avg Lead Time Improvement", "Across candidates", True), unsafe_allow_html=True)
c2.markdown(kpi_card(f"{exec_sum['avg_dist_reduction_pct']:.1f}%", "Avg Distance Reduction", "Factory → Customer", True), unsafe_allow_html=True)
c3.markdown(kpi_card(f"${exec_sum['est_cost_saving_usd']:,.0f}", "Estimated Cost Savings", "Distance-based proxy", True), unsafe_allow_html=True)

# ── Revenue & Profit charts ────────────────────────────────────────────────────
section("Sales & Profit by Division")
set_dark_style()
col_l, col_r = st.columns(2)

with col_l:
    div_data = df_f.groupby("Division").agg(Sales=("Sales","sum"), Profit=("Gross Profit","sum")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    x, w = np.arange(len(div_data)), 0.35
    ax.bar(x-w/2, div_data["Sales"],  width=w, color=COLORS["primary"], label="Sales",  alpha=0.9)
    ax.bar(x+w/2, div_data["Profit"], width=w, color=COLORS["success"], label="Profit", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(div_data["Division"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    ax.set_title("Revenue & Profit by Division"); ax.legend(); ax.grid(axis="y")
    fig.tight_layout(); st.pyplot(fig); plt.close()

with col_r:
    ship_data = df_f.groupby("Ship Mode")["Sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(5, 3.8))
    wedge_colors = [SHIP_COLORS.get(s, "#64748b") for s in ship_data["Ship Mode"]]
    wedges, texts, autotexts = ax.pie(
        ship_data["Sales"], labels=ship_data["Ship Mode"], colors=wedge_colors,
        autopct="%1.1f%%", startangle=140, pctdistance=0.75,
        wedgeprops={"edgecolor": COLORS["bg"], "linewidth": 2},
    )
    for at in autotexts: at.set_fontsize(8); at.set_color(COLORS["bg"])
    for t  in texts:     t.set_color(COLORS["text"]); t.set_fontsize(8)
    ax.set_title("Sales by Ship Mode")
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ── Monthly trend ──────────────────────────────────────────────────────────────
section("Monthly Revenue & Profit Trend")
monthly = df_f.groupby(["Order_Year","Order_Month"]).agg(
    Sales=("Sales","sum"), Profit=("Gross Profit","sum")
).reset_index()
monthly["Period"] = monthly["Order_Year"].astype(str)+"-"+monthly["Order_Month"].astype(str).str.zfill(2)
monthly = monthly.sort_values("Period")

fig, ax = plt.subplots(figsize=(14, 3.8))
ax.fill_between(range(len(monthly)), monthly["Sales"],  alpha=0.15, color=COLORS["primary"])
ax.plot(range(len(monthly)), monthly["Sales"],  color=COLORS["primary"], linewidth=2, label="Sales")
ax.fill_between(range(len(monthly)), monthly["Profit"], alpha=0.15, color=COLORS["success"])
ax.plot(range(len(monthly)), monthly["Profit"], color=COLORS["success"], linewidth=2, linestyle="--", label="Profit")
ax.set_xticks(range(len(monthly))); ax.set_xticklabels(monthly["Period"], rotation=45, ha="right", fontsize=7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
ax.set_title("Monthly Sales & Profit Trend"); ax.legend(); ax.grid(axis="y")
fig.tight_layout(); st.pyplot(fig); plt.close()

insight("Chocolate division generates 96.6% of revenue. Standard Class shipping covers "
        "~62% of orders. Month-over-month performance is stable, signalling consistent "
        "demand — making factory optimisation low-risk from a revenue perspective.")

section("Factory Network Summary")
fac_perf = df_f.groupby("Factory").agg(
    Orders=("Row ID","count"), Revenue=("Sales","sum"), Profit=("Gross Profit","sum"),
    Avg_LT=("Lead_Time","mean"), Avg_Dist=("Distance_Miles","mean"), Margin=("Profit_Margin","mean"),
).reset_index()
for col_fmt, fmt in [("Revenue","${:,.0f}"),("Profit","${:,.0f}"),
                     ("Avg_LT","{:.1f}d"),("Avg_Dist","{:,.0f}mi"),("Margin","{:.1%}")]:
    fac_perf[col_fmt] = fac_perf[col_fmt].map(fmt.format)
st.dataframe(fac_perf, use_container_width=True, hide_index=True)
