"""pages/2_Factory_Network_Analysis.py"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils.shared import (load_data, load_recommendations, set_dark_style,
                           kpi_card, section, insight, COLORS, FACTORY_COLORS)
from utils.styles import inject_css
from feature_engineering import compute_factory_distance_table

st.set_page_config(page_title="Factory Network · Nassau Candy", page_icon="🏭", layout="wide")
inject_css()

df_raw, df_feat, prod_sum, route_stats = load_data()
sim_df, best_df, recs_df, exec_sum     = load_recommendations(df_raw)

with st.sidebar:
    filter_region   = st.multiselect("Region",   sorted(df_raw["Region"].unique()),
                                     default=sorted(df_raw["Region"].unique()))
    filter_division = st.multiselect("Division", sorted(df_raw["Division"].unique()),
                                     default=sorted(df_raw["Division"].unique()))
mask = df_feat["Region"].isin(filter_region) & df_feat["Division"].isin(filter_division)
df_f = df_feat[mask]

st.markdown("## 🏭 Factory Network Analysis")
st.caption("Deep-dive into each factory's performance, product mix, and logistics footprint.")
set_dark_style()

# KPIs
fac_grp = df_f.groupby("Factory").agg(
    Orders=("Row ID","count"), Revenue=("Sales","sum"),
    Avg_LT=("Lead_Time","mean"), Avg_Dist=("Distance_Miles","mean"),
).reset_index()
cols = st.columns(len(fac_grp))
for i, (_, row) in enumerate(fac_grp.iterrows()):
    cols[i].markdown(kpi_card(
        f"{row['Orders']:,}", row["Factory"],
        f"${row['Revenue']:,.0f} · {row['Avg_LT']:.1f}d LT", True
    ), unsafe_allow_html=True)

section("Orders & Revenue by Factory")
col_l, col_r = st.columns(2)
fac_order = sorted(df_f["Factory"].unique())

with col_l:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [FACTORY_COLORS.get(f, "#64748b") for f in fac_grp["Factory"]]
    bars = ax.barh(fac_grp["Factory"], fac_grp["Orders"], color=colors, alpha=0.9)
    for b in bars:
        ax.text(b.get_width()+50, b.get_y()+b.get_height()/2,
                f"{b.get_width():,.0f}", va="center", fontsize=8, color=COLORS["text"])
    ax.set_title("Total Orders by Factory"); ax.grid(axis="x")
    ax.set_xlim(0, fac_grp["Orders"].max()*1.15)
    fig.tight_layout(); st.pyplot(fig); plt.close()

with col_r:
    rev_grp = df_f.groupby("Factory")["Sales"].sum().reset_index()
    colors  = [FACTORY_COLORS.get(f, "#64748b") for f in rev_grp["Factory"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(rev_grp["Factory"], rev_grp["Sales"], color=colors, alpha=0.9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    for b in bars:
        ax.text(b.get_width()+200, b.get_y()+b.get_height()/2,
                f"${b.get_width():,.0f}", va="center", fontsize=8, color=COLORS["text"])
    ax.set_title("Total Revenue by Factory"); ax.grid(axis="x")
    ax.set_xlim(0, rev_grp["Sales"].max()*1.15)
    fig.tight_layout(); st.pyplot(fig); plt.close()

section("Lead Time & Distance Distribution by Factory")
col_l, col_r = st.columns(2)
with col_l:
    fig, ax = plt.subplots(figsize=(6, 4))
    data_lt = [df_f[df_f["Factory"]==f]["Lead_Time"].values for f in fac_order]
    bp = ax.boxplot(data_lt, patch_artist=True, vert=False,
                    medianprops={"color": COLORS["bg"], "linewidth": 2})
    for patch, fac in zip(bp["boxes"], fac_order):
        patch.set_facecolor(FACTORY_COLORS.get(fac, "#64748b")); patch.set_alpha(0.85)
    ax.set_yticks(range(1, len(fac_order)+1)); ax.set_yticklabels(fac_order, fontsize=8)
    ax.set_xlabel("Lead Time (days)"); ax.set_title("Lead Time Distribution"); ax.grid(axis="x")
    fig.tight_layout(); st.pyplot(fig); plt.close()

with col_r:
    fig, ax = plt.subplots(figsize=(6, 4))
    data_dist = [df_f[df_f["Factory"]==f]["Distance_Miles"].values for f in fac_order]
    bp = ax.boxplot(data_dist, patch_artist=True, vert=False,
                    medianprops={"color": COLORS["bg"], "linewidth": 2})
    for patch, fac in zip(bp["boxes"], fac_order):
        patch.set_facecolor(FACTORY_COLORS.get(fac, "#64748b")); patch.set_alpha(0.85)
    ax.set_yticks(range(1, len(fac_order)+1)); ax.set_yticklabels(fac_order, fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:,.0f}mi"))
    ax.set_title("Shipping Distance Distribution"); ax.grid(axis="x")
    fig.tight_layout(); st.pyplot(fig); plt.close()

section("Product × Factory Order Heatmap")
piv = df_f.pivot_table(index="Product Name", columns="Factory",
                        values="Row ID", aggfunc="count", fill_value=0)
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(piv.values, aspect="auto", cmap=plt.cm.Blues)
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=8, rotation=20, ha="right")
ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index, fontsize=7)
for i in range(len(piv.index)):
    for j in range(len(piv.columns)):
        v = piv.values[i, j]
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=7,
                color="white" if v > piv.values.max()*0.5 else COLORS["text"])
plt.colorbar(im, ax=ax, shrink=0.6, label="Order Count")
ax.set_title("Orders per Product per Factory")
fig.tight_layout(); st.pyplot(fig); plt.close()

section("Factory → Region Revenue Flow")
flow = df_f.groupby(["Factory","Region"])["Sales"].sum().reset_index()
flow_piv = flow.pivot(index="Factory", columns="Region", values="Sales").fillna(0)
fig, ax = plt.subplots(figsize=(10, 4))
bottom = np.zeros(len(flow_piv.columns))
for fac, row_data in flow_piv.iterrows():
    ax.bar(flow_piv.columns, row_data.values, bottom=bottom,
           color=FACTORY_COLORS.get(fac, "#64748b"), alpha=0.85, label=fac)
    bottom += row_data.values
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
ax.set_title("Revenue Flow: Factory → Region"); ax.legend(fontsize=7, loc="upper right")
ax.grid(axis="y"); fig.tight_layout(); st.pyplot(fig); plt.close()

insight("Lot's O' Nuts handles ~94% of all orders (Chocolate dominance). "
        "Secret Factory covers the widest geographic spread. "
        "Sugar Shack's northern location creates long shipping distances for Southern/Pacific customers.")

section("Inter-Factory Distance Matrix (miles)")
dist_tbl = compute_factory_distance_table()
pivot_dist = dist_tbl.pivot(index="From_Factory", columns="To_Factory",
                             values="Distance_Miles").round(0).astype(int)
st.dataframe(pivot_dist, use_container_width=True)
st.caption("Used in the reallocation scoring engine to quantify relocation cost.")
