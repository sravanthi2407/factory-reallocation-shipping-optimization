"""pages/4_Factory_Optimization_Simulator.py"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils.shared import (load_data, load_recommendations, set_dark_style,
                           kpi_card, section, insight, COLORS, FACTORY_COLORS)
from utils.styles import inject_css

st.set_page_config(page_title="Optimization Simulator · Nassau Candy", page_icon="⚙️", layout="wide")
inject_css()

df_raw, df_feat, prod_sum, route_stats = load_data()
sim_df, best_df, recs_df, exec_sum     = load_recommendations(df_raw)

st.markdown("## ⚙️ Factory Optimization Simulator")
st.caption("Simulate all factory assignments per product and compare reallocation scores.")
set_dark_style()

# Full simulation table
section("Full Simulation Table — All Products × All Factories")
score_col = "Reallocation_Score"
display_cols = [c for c in [
    "Product","Current_Factory","Candidate_Factory","Is_Current",
    "Predicted_Lead_Time","Lead_Time_Improvement_Pct",
    "Candidate_Distance","Distance_Reduction_Pct",
    "Reallocation_Score","Risk_Level","Data_Confidence"
] if c in sim_df.columns]
sim_display = sim_df[display_cols].copy()

def colour_score(val):
    if val >= 65:   return "background-color:#14532d; color:#4ade80"
    elif val >= 50: return "background-color:#713f12; color:#fbbf24"
    else:           return "background-color:#7f1d1d; color:#f87171"

try:
    styled = sim_display.style.map(colour_score, subset=[score_col])
except AttributeError:
    styled = sim_display.style.applymap(colour_score, subset=[score_col])
st.dataframe(styled, use_container_width=True, height=400, hide_index=True)

# Score heatmap
section("Reallocation Score Heatmap — Product × Factory")
piv_score = sim_df.pivot_table(
    index="Product", columns="Candidate_Factory", values="Reallocation_Score"
)
fig, ax = plt.subplots(figsize=(11, 7))
im = ax.imshow(piv_score.values, aspect="auto", cmap=plt.cm.RdYlGn, vmin=30, vmax=75)
ax.set_xticks(range(len(piv_score.columns)))
ax.set_xticklabels(piv_score.columns, fontsize=8, rotation=20, ha="right")
ax.set_yticks(range(len(piv_score.index)))
ax.set_yticklabels(piv_score.index, fontsize=7)
for i in range(len(piv_score.index)):
    for j in range(len(piv_score.columns)):
        v = piv_score.values[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                fontsize=7.5, color="white" if v < 50 else "black",
                fontweight="bold" if v >= 65 else "normal")
plt.colorbar(im, ax=ax, shrink=0.6, label="Reallocation Score (0-100)")
ax.set_title("Green = Best Candidate | Red = Poor Fit | Yellow = Current Factory")
fig.tight_layout(); st.pyplot(fig); plt.close()

# Per-product drill-down
section("Per-Product Factory Comparison")
selected_product = st.selectbox("Select Product", sorted(sim_df["Product"].unique()))
prod_sim = sim_df[sim_df["Product"]==selected_product].sort_values(
    "Reallocation_Score", ascending=False
).reset_index(drop=True)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, metric, title in zip(
    axes,
    ["Predicted_Lead_Time","Candidate_Distance","Reallocation_Score"],
    ["Predicted Lead Time (days)","Distance to Customers (miles)","Reallocation Score"]
):
    vals   = prod_sim[metric].values
    facs   = prod_sim["Candidate_Factory"].values
    is_cur = prod_sim["Is_Current"].values
    cmap_c = [COLORS["warning"] if ic else
               (COLORS["success"] if v==vals.max() and metric=="Reallocation_Score" else
                COLORS["danger"]  if v==vals.max() and metric!="Reallocation_Score" else
                COLORS["primary"])
              for v, ic in zip(vals, is_cur)]
    bars = ax.bar(range(len(facs)), vals, color=cmap_c, alpha=0.9)
    ax.set_xticks(range(len(facs)))
    ax.set_xticklabels(facs, rotation=25, ha="right", fontsize=7)
    ax.set_title(title, fontsize=9); ax.grid(axis="y")
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+vals.max()*0.02,
                f"{b.get_height():.1f}", ha="center", fontsize=7, color=COLORS["text"])

axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:,.0f}"))
fig.suptitle(f"{selected_product} — Factory Comparison", color=COLORS["text"])
st.markdown("🟡 = Current &nbsp;&nbsp; 🟢 = Best Score &nbsp;&nbsp; 🔵 = Other")
fig.tight_layout(); st.pyplot(fig); plt.close()

# Detailed table for product
st.markdown("**Detailed simulation for selected product:**")
st.dataframe(prod_sim, use_container_width=True, hide_index=True)

# Download
st.download_button(
    "⬇️ Download Full Simulation (CSV)", sim_df.to_csv(index=False),
    "nassau_candy_full_simulation.csv", "text/csv"
)
