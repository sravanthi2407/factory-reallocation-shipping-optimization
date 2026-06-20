"""pages/5_Recommendation_Dashboard.py"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils.shared import (load_data, load_recommendations, set_dark_style,
                           kpi_card, section, insight, COLORS)
from utils.styles import inject_css

st.set_page_config(page_title="Recommendations · Nassau Candy", page_icon="🎯", layout="wide")
inject_css()

df_raw, df_feat, prod_sum, route_stats = load_data()
sim_df, best_df, recs_df, exec_sum     = load_recommendations(df_raw)

st.markdown("## 🎯 Recommendation Dashboard")
st.caption("Top-20 factory reallocation recommendations ranked by composite score.")
set_dark_style()

# KPIs
c1,c2,c3,c4 = st.columns(4)
c1.markdown(kpi_card(len(recs_df), "Total Recommendations"), unsafe_allow_html=True)
if "LT_Improvement_%" in recs_df.columns:
    pos = recs_df[recs_df["LT_Improvement_%"] > 0]
    c2.markdown(kpi_card(len(pos), "Products with LT Gains", None, True), unsafe_allow_html=True)
    c3.markdown(kpi_card(f"{recs_df['LT_Improvement_%'].mean():.1f}%", "Avg LT Improvement", None, True), unsafe_allow_html=True)
if "Distance_Reduction_%" in recs_df.columns:
    c4.markdown(kpi_card(f"{recs_df['Distance_Reduction_%'].mean():.1f}%", "Avg Distance Reduction", None, True), unsafe_allow_html=True)

# Top-20 table
section("Top-20 Reallocation Recommendations")
RISK_EMOJI = lambda r: "🟢 " + r if "Low Risk" in r and "Low Data" not in r else \
                        "🟡 " + r if "Moderate" in r else "🔴 " + r

recs_display = recs_df.copy()
if "Risk_Level" in recs_display.columns:
    recs_display["Risk"] = recs_display["Risk_Level"].map(RISK_EMOJI)
st.dataframe(recs_display, use_container_width=True, hide_index=True, height=480)

# Score bar chart
section("Reallocation Score by Product")
score_col = "Reallocation_Score"
if score_col in recs_df.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    vals = recs_df[score_col].values
    bar_colors = [COLORS["success"] if v>=65 else COLORS["warning"] if v>=50 else COLORS["danger"]
                  for v in vals]
    bars = ax.barh(recs_df["Product"], vals, color=bar_colors, alpha=0.88)
    ax.axvline(50, color=COLORS["subtext"], linewidth=1, linestyle="--", label="Threshold (50)")
    ax.axvline(65, color=COLORS["success"], linewidth=1, linestyle="--", label="Good (65+)")
    for b in bars:
        ax.text(b.get_width()+0.4, b.get_y()+b.get_height()/2,
                f"{b.get_width():.1f}", va="center", fontsize=8, color=COLORS["text"])
    ax.set_xlabel("Reallocation Score (0–100)")
    ax.set_title("Score Formula: 0.50×LT Improvement + 0.30×Distance Reduction + 0.20×Profit Stability")
    ax.legend(); ax.grid(axis="x"); ax.set_xlim(0, 85)
    fig.tight_layout(); st.pyplot(fig); plt.close()

# LT improvement chart
if "LT_Improvement_%" in recs_df.columns:
    section("Lead Time Improvement % by Product")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    lt_vals = recs_df["LT_Improvement_%"].values
    bar_c   = [COLORS["success"] if v>0 else COLORS["danger"] for v in lt_vals]
    ax.barh(recs_df["Product"], lt_vals, color=bar_c, alpha=0.88)
    ax.axvline(0, color=COLORS["subtext"], linewidth=1)
    ax.set_xlabel("Lead Time Improvement (%) — Positive = Faster Delivery")
    ax.set_title("Lead Time Change vs. Current Factory")
    ax.grid(axis="x"); fig.tight_layout(); st.pyplot(fig); plt.close()

# Distance reduction
if "Distance_Reduction_%" in recs_df.columns:
    section("Distance Reduction % by Product")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    dist_vals = recs_df["Distance_Reduction_%"].values
    bar_c     = [COLORS["success"] if v>0 else COLORS["danger"] for v in dist_vals]
    ax.barh(recs_df["Product"], dist_vals, color=bar_c, alpha=0.88)
    ax.axvline(0, color=COLORS["subtext"], linewidth=1)
    ax.set_xlabel("Distance Reduction (%) — Positive = Shorter Routes")
    ax.set_title("Shipping Distance Change vs. Current Factory")
    ax.grid(axis="x"); fig.tight_layout(); st.pyplot(fig); plt.close()

insight("Fizzy Lifting Drinks and Laffy Taffy show >10% lead time improvement potential. "
        "Lot's O' Nuts Chocolate products benefit most from distance reduction via Secret Factory. "
        "Low Data confidence products (Sugar/Other divisions) should be validated before reallocation.")

st.download_button(
    "⬇️ Download Recommendations (CSV)",
    recs_df.to_csv(index=False),
    "nassau_candy_recommendations.csv",
    "text/csv"
)
