"""pages/7_Risk_Assessment.py"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils.shared import (load_data, load_recommendations, set_dark_style,
                           kpi_card, section, insight, COLORS, FACTORY_COLORS, RISK_COLORS)
from utils.styles import inject_css

st.set_page_config(page_title="Risk Assessment · Nassau Candy", page_icon="⚠️", layout="wide")
inject_css()

df_raw, df_feat, prod_sum, route_stats = load_data()
sim_df, best_df, recs_df, exec_sum     = load_recommendations(df_raw)

st.markdown("## ⚠️ Risk Assessment")
st.caption("Evaluate reallocation risks across products, factories, and regions.")
set_dark_style()

# Risk KPIs
risk_counts = sim_df["Risk_Level"].value_counts()
total_sims  = len(sim_df)
c1,c2,c3,c4 = st.columns(4)
c1.markdown(kpi_card(total_sims, "Total Simulations"), unsafe_allow_html=True)
c2.markdown(kpi_card(risk_counts.get("Low Risk",0), "Low Risk",
    f"{risk_counts.get('Low Risk',0)/total_sims*100:.1f}%", True), unsafe_allow_html=True)
c3.markdown(kpi_card(risk_counts.get("Moderate Risk",0), "Moderate Risk",
    f"{risk_counts.get('Moderate Risk',0)/total_sims*100:.1f}%", False), unsafe_allow_html=True)
c4.markdown(kpi_card(
    sum(v for k,v in risk_counts.items() if "High" in k),
    "High Risk", "Includes Low-Data flag", False
), unsafe_allow_html=True)

# Risk distribution
section("Risk Distribution")
col_l, col_r = st.columns(2)

with col_l:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    risk_labels = list(risk_counts.index)
    risk_vals   = list(risk_counts.values)
    colors_r    = [RISK_COLORS.get(r, COLORS["primary"]) for r in risk_labels]
    wedges, texts, autotexts = ax.pie(
        risk_vals, labels=risk_labels, colors=colors_r, autopct="%1.1f%%",
        startangle=120, pctdistance=0.78,
        wedgeprops={"edgecolor": COLORS["bg"], "linewidth": 2}
    )
    for at in autotexts: at.set_fontsize(8); at.set_color(COLORS["bg"])
    for t  in texts:     t.set_color(COLORS["text"]); t.set_fontsize(7.5)
    ax.set_title("Risk Level Distribution")
    fig.tight_layout(); st.pyplot(fig); plt.close()

with col_r:
    risk_prod = best_df[["Product","Risk_Level","Reallocation_Score","Data_Confidence"]].copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    bar_c = [RISK_COLORS.get(r, COLORS["primary"]) for r in risk_prod["Risk_Level"]]
    ax.barh(risk_prod["Product"], risk_prod["Reallocation_Score"],
            color=bar_c, alpha=0.88)
    ax.axvline(50, color=COLORS["subtext"], linewidth=1, linestyle="--", label="Threshold 50")
    ax.axvline(65, color=COLORS["success"], linewidth=1, linestyle="--", label="Good 65+")
    ax.set_xlabel("Reallocation Score"); ax.set_title("Risk per Product (best recommendation)")
    ax.legend(fontsize=7); ax.grid(axis="x"); ax.set_xlim(0, 85)
    fig.tight_layout(); st.pyplot(fig); plt.close()

# Data confidence
section("Data Confidence by Product")
conf = df_feat.groupby("Product Name").agg(
    Orders=("Row ID","count"),
    Confidence=("Data_Confidence","first")
).reset_index().sort_values("Orders", ascending=False)

fig, ax = plt.subplots(figsize=(12, 5))
bar_colors = [COLORS["success"] if c=="High" else COLORS["danger"] for c in conf["Confidence"]]
bars = ax.bar(range(len(conf)), conf["Orders"], color=bar_colors, alpha=0.85)
ax.set_xticks(range(len(conf)))
ax.set_xticklabels(conf["Product Name"], rotation=35, ha="right", fontsize=7.5)
ax.axhline(100, color=COLORS["warning"], linewidth=1.5, linestyle="--",
           label="Confidence Threshold (100 orders)")
ax.set_title("Orders per Product — Green = High Confidence (≥100), Red = Low")
ax.set_ylabel("Number of Orders"); ax.legend(); ax.grid(axis="y")
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+20,
            f"{b.get_height():,}", ha="center", fontsize=7, color=COLORS["text"])
fig.tight_layout(); st.pyplot(fig); plt.close()

# Margin stability
section("Profit Margin Stability by Factory")
fac_order = sorted(df_feat["Factory"].unique())
fig, ax = plt.subplots(figsize=(10, 4))
margin_data = [df_feat[df_feat["Factory"]==f]["Profit_Margin"].values for f in fac_order]
bp = ax.boxplot(margin_data, patch_artist=True, vert=True,
                medianprops={"color": COLORS["bg"], "linewidth": 2.5})
for patch, fac in zip(bp["boxes"], fac_order):
    patch.set_facecolor(FACTORY_COLORS.get(fac,"#64748b")); patch.set_alpha(0.85)
ax.set_xticks(range(1, len(fac_order)+1))
ax.set_xticklabels(fac_order, rotation=15, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0%}"))
ax.set_title("Profit Margin Distribution by Factory"); ax.set_ylabel("Profit Margin")
ax.grid(axis="y"); fig.tight_layout(); st.pyplot(fig); plt.close()

# Risk matrix
section("Risk Matrix — Score vs. Distance Reduction")
fig, ax = plt.subplots(figsize=(10, 5))
for _, row in best_df.iterrows():
    x    = row.get("Reallocation_Score", 50)
    y    = row.get("Distance_Reduction_Pct", 0)
    c    = RISK_COLORS.get(row.get("Risk_Level",""), COLORS["primary"])
    size = 120 if row.get("Data_Confidence","Low")=="High" else 60
    ax.scatter(x, y, c=c, s=size, alpha=0.85,
               edgecolors=COLORS["border"], linewidths=1)
    ax.annotate(row["Product"], (x, y), textcoords="offset points",
                xytext=(6,4), fontsize=6.5, color=COLORS["subtext"])
ax.axvline(50, color=COLORS["subtext"], linewidth=1, linestyle="--")
ax.axvline(65, color=COLORS["success"], linewidth=1, linestyle="--")
ax.axhline(0,  color=COLORS["subtext"], linewidth=1, linestyle="--")
ax.set_xlabel("Reallocation Score"); ax.set_ylabel("Distance Reduction %")
ax.set_title("Risk Matrix · Large dot = High Data Confidence")
ax.grid(True, alpha=0.3); fig.tight_layout(); st.pyplot(fig); plt.close()

insight("Products in the High Score + Positive Distance Reduction quadrant are prime reallocation "
        "candidates. Low confidence products (small dots) need additional data before committing to "
        "factory changes. All Chocolate products have solid data confidence.")

# Mitigation table
section("Risk Mitigation Playbook")
import pandas as pd
mitigation = pd.DataFrame([
    {"Risk Level":"🟢 Low Risk",     "Recommendation":"Proceed with reallocation",         "Timeline":"Q1 2026","Priority":"High"},
    {"Risk Level":"🟡 Moderate Risk","Recommendation":"Run 20% pilot before full rollout", "Timeline":"Q2 2026","Priority":"Medium"},
    {"Risk Level":"🔴 High Risk",    "Recommendation":"Defer; collect 6 months more data", "Timeline":"Q3 2026","Priority":"Low"},
    {"Risk Level":"⚪ Low Data",     "Recommendation":"A/B test with 2 factories",         "Timeline":"Q2–Q3", "Priority":"Research"},
])
st.dataframe(mitigation, use_container_width=True, hide_index=True)
