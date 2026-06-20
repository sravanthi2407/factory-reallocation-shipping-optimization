"""
app.py
Nassau Candy Distributor — Factory Reallocation & Shipping Optimization
Decision Intelligence System | 7-Page Streamlit Dashboard
"""

import os, sys, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config MUST be first Streamlit call ─────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy — Supply Chain Intelligence",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS theme ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  [data-testid="stAppViewContainer"] { background: #0e1117; }
  [data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #2a2f3e; }

  /* KPI Cards */
  .kpi-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #252b3b 100%);
    border: 1px solid #2e3650;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    text-align: center;
  }
  .kpi-value  { font-size: 2.2rem; font-weight: 700; color: #7dd3fc; line-height:1.1; }
  .kpi-label  { font-size: 0.82rem; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing:.05em; }
  .kpi-delta  { font-size: 0.78rem; margin-top: 6px; }
  .delta-pos  { color: #4ade80; }
  .delta-neg  { color: #f87171; }

  /* Section headers */
  .section-title {
    font-size: 1.3rem; font-weight: 700; color: #e2e8f0;
    border-left: 4px solid #7dd3fc; padding-left: 12px;
    margin: 28px 0 16px 0;
  }

  /* Tables */
  .dataframe thead tr th { background: #1e2533 !important; color: #7dd3fc !important; }
  .dataframe tbody tr:nth-child(even) { background: #1a1f2c !important; }
  .dataframe tbody tr:hover { background: #252b3e !important; }

  /* Risk badges */
  .risk-low      { background:#14532d; color:#4ade80; padding:3px 10px; border-radius:20px; font-size:.78rem; }
  .risk-moderate { background:#713f12; color:#fbbf24; padding:3px 10px; border-radius:20px; font-size:.78rem; }
  .risk-high     { background:#7f1d1d; color:#f87171; padding:3px 10px; border-radius:20px; font-size:.78rem; }

  /* Sidebar nav */
  .nav-header { color:#7dd3fc; font-size:.72rem; text-transform:uppercase; letter-spacing:.1em;
                padding:16px 0 6px 0; font-weight:600; }
  div[data-testid="stSidebarNav"] { display:none; }

  /* Insight box */
  .insight-box {
    background: #1e293b; border-left: 4px solid #7dd3fc;
    border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 12px 0;
    font-size: .88rem; color: #cbd5e1; line-height: 1.6;
  }

  /* Score bar */
  .score-fill {
    height: 8px; border-radius: 4px;
    background: linear-gradient(90deg, #ef4444, #f59e0b, #22c55e);
  }
</style>
""", unsafe_allow_html=True)

# ── Imports (project modules) ─────────────────────────────────────────────────
from preprocessing import (
    get_clean_data, FACTORY_COORDS, PRODUCT_FACTORY, ALL_FACTORIES
    if hasattr(__import__('preprocessing'), 'ALL_FACTORIES')
    else (lambda: None)()
)
# Re-define ALL_FACTORIES here for safety
ALL_FACTORIES = list(FACTORY_COORDS.keys())

from feature_engineering import (
    build_features, compute_product_summary,
    compute_route_stats, compute_factory_distance_table
)
from train_model import load_best_model, ML_FEATURES, ML_TARGET
from optimization_engine import run_reallocation_engine, get_best_factory_per_product
from recommendation_engine import generate_recommendations, get_executive_summary
from scenario_simulator import run_scenario

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

# ── Color palette (matplotlib-safe) ──────────────────────────────────────────
COLORS = {
    "primary":   "#7dd3fc",
    "secondary": "#a78bfa",
    "success":   "#4ade80",
    "warning":   "#fbbf24",
    "danger":    "#f87171",
    "bg":        "#0e1117",
    "card":      "#1a1f2e",
    "border":    "#2e3650",
    "text":      "#e2e8f0",
    "subtext":   "#94a3b8",
}
FACTORY_COLORS = {
    "Lot's O' Nuts":     "#7dd3fc",
    "Wicked Choccy's":  "#a78bfa",
    "Sugar Shack":       "#4ade80",
    "Secret Factory":    "#fbbf24",
    "The Other Factory": "#f87171",
}
SHIP_COLORS = {
    "Standard Class": "#94a3b8",
    "Second Class":   "#7dd3fc",
    "First Class":    "#a78bfa",
    "Same Day":       "#4ade80",
}

def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg"],
        "axes.facecolor":    COLORS["card"],
        "axes.edgecolor":    COLORS["border"],
        "axes.labelcolor":   COLORS["subtext"],
        "xtick.color":       COLORS["subtext"],
        "ytick.color":       COLORS["subtext"],
        "text.color":        COLORS["text"],
        "grid.color":        COLORS["border"],
        "grid.alpha":        0.5,
        "legend.facecolor":  COLORS["card"],
        "legend.edgecolor":  COLORS["border"],
        "font.family":       "sans-serif",
        "axes.titlecolor":   COLORS["text"],
        "axes.titlesize":    11,
        "axes.labelsize":    9,
    })


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df       = get_clean_data("data/Nassau_Candy_Distributor.csv")
    df_feat  = build_features(df)
    prod_sum = compute_product_summary(df_feat)
    route_st = compute_route_stats(df_feat)
    return df, df_feat, prod_sum, route_st

@st.cache_data(show_spinner="Generating recommendations…")
def load_recommendations(_df):
    sim_df   = run_reallocation_engine(_df)
    best     = get_best_factory_per_product(sim_df)
    recs     = generate_recommendations(_df, top_n=20)
    exec_sum = get_executive_summary(recs, _df)
    return sim_df, best, recs, exec_sum

@st.cache_resource(show_spinner="Loading ML model…")
def load_model():
    return load_best_model()

# ── Helper: KPI card HTML ─────────────────────────────────────────────────────
def kpi_card(value, label, delta=None, delta_positive=True, prefix="", suffix=""):
    delta_html = ""
    if delta is not None:
        cls   = "delta-pos" if delta_positive else "delta-neg"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    return f"""
    <div class="kpi-card">
      <div class="kpi-value">{prefix}{value}{suffix}</div>
      <div class="kpi-label">{label}</div>
      {delta_html}
    </div>"""

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
PAGES = [
    ("🏠", "Executive Overview"),
    ("🏭", "Factory Network Analysis"),
    ("⏱️", "Lead Time Prediction"),
    ("⚙️", "Factory Optimization Simulator"),
    ("🎯", "Recommendation Dashboard"),
    ("🔮", "What-If Scenario Analysis"),
    ("⚠️", "Risk Assessment"),
]

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 12px 0;'>
      <div style='font-size:2rem;'>🍬</div>
      <div style='font-size:1.05rem; font-weight:700; color:#e2e8f0; margin-top:4px;'>Nassau Candy</div>
      <div style='font-size:.75rem; color:#7dd3fc; letter-spacing:.08em; text-transform:uppercase;'>
        Supply Chain Intelligence
      </div>
    </div>
    <hr style='border-color:#2e3650; margin:0 0 8px 0;'>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-header">Navigation</div>', unsafe_allow_html=True)

    selected = st.radio(
        "nav",
        [f"{icon}  {name}" for icon, name in PAGES],
        label_visibility="collapsed",
    )
    page_name = selected.split("  ", 1)[1]

    st.markdown("<hr style='border-color:#2e3650; margin:20px 0 12px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="nav-header">Global Filters</div>', unsafe_allow_html=True)

    df_raw, df_feat, prod_sum, route_stats = load_data()

    filter_region = st.multiselect(
        "Region", sorted(df_raw["Region"].unique()),
        default=sorted(df_raw["Region"].unique())
    )
    filter_division = st.multiselect(
        "Division", sorted(df_raw["Division"].unique()),
        default=sorted(df_raw["Division"].unique())
    )
    filter_ship = st.multiselect(
        "Ship Mode", sorted(df_raw["Ship Mode"].unique()),
        default=sorted(df_raw["Ship Mode"].unique())
    )

    st.markdown("""
    <hr style='border-color:#2e3650; margin:20px 0 12px 0;'>
    <div style='font-size:.72rem; color:#64748b; text-align:center;'>
      Nassau Candy © 2025<br>Decision Intelligence v1.0
    </div>""", unsafe_allow_html=True)

# Apply global filters
mask = (
    df_feat["Region"].isin(filter_region) &
    df_feat["Division"].isin(filter_division) &
    df_feat["Ship Mode"].isin(filter_ship)
)
df_f = df_feat[mask].copy()

# Load ML model + recommendations
model, model_name, feature_names = load_model()
sim_df, best_df, recs_df, exec_sum = load_recommendations(df_raw)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page_name == "Executive Overview":
    st.markdown("## 🏠 Executive Overview")
    st.markdown(
        '<p style="color:#94a3b8; margin-top:-8px;">Nassau Candy Distributor · '
        'Factory Reallocation & Shipping Optimization System</p>',
        unsafe_allow_html=True
    )

    # ── Top KPI row ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, f"{exec_sum['total_orders']:,}", "Total Orders", None, True, "", ""),
        (c2, f"${exec_sum['total_revenue']:,.0f}", "Total Revenue", None, True, "", ""),
        (c3, f"${exec_sum['total_profit']:,.0f}", "Gross Profit", None, True, "", ""),
        (c4, f"{exec_sum['overall_margin_pct']:.1f}%", "Profit Margin", None, True, "", ""),
        (c5, f"{exec_sum['unique_customers']:,}", "Unique Customers", None, True, "", ""),
        (c6, f"{exec_sum['products_to_reallocate']}", "Products to Reallocate", None, True, "", ""),
    ]
    for col, val, lbl, delta, pos, pre, suf in kpis:
        col.markdown(kpi_card(val, lbl, delta, pos, pre, suf), unsafe_allow_html=True)

    # ── Optimization opportunity row ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi_card(
        f"{exec_sum['avg_lt_improvement_pct']:.1f}%",
        "Avg Lead Time Improvement",
        "Across reallocation candidates", True
    ), unsafe_allow_html=True)
    c2.markdown(kpi_card(
        f"{exec_sum['avg_dist_reduction_pct']:.1f}%",
        "Avg Distance Reduction",
        "Factory → Customer", True
    ), unsafe_allow_html=True)
    c3.markdown(kpi_card(
        f"${exec_sum['est_cost_saving_usd']:,.0f}",
        "Estimated Annual Savings",
        "Based on distance reduction", True
    ), unsafe_allow_html=True)

    # ── Charts row ───────────────────────────────────────────────────────────
    section("Sales & Profit by Division")
    col_l, col_r = st.columns(2)

    set_dark_style()

    with col_l:
        div_data = df_f.groupby("Division").agg(
            Sales=("Sales","sum"), Profit=("Gross Profit","sum")
        ).reset_index()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        x = np.arange(len(div_data))
        w = 0.35
        b1 = ax.bar(x - w/2, div_data["Sales"],  width=w,
                    color=COLORS["primary"],   label="Sales",  alpha=0.9)
        b2 = ax.bar(x + w/2, div_data["Profit"], width=w,
                    color=COLORS["success"],   label="Profit", alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(div_data["Division"], fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
        ax.set_title("Revenue & Profit by Division"); ax.legend(); ax.grid(axis="y")
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                    f"${bar.get_height():,.0f}", ha="center", va="bottom",
                    fontsize=7, color=COLORS["text"])
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col_r:
        ship_data = df_f.groupby("Ship Mode")["Sales"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(5, 3.8))
        wedge_colors = [SHIP_COLORS.get(s, "#64748b") for s in ship_data["Ship Mode"]]
        wedges, texts, autotexts = ax.pie(
            ship_data["Sales"],
            labels=ship_data["Ship Mode"],
            colors=wedge_colors,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.75,
            wedgeprops={"edgecolor": COLORS["bg"], "linewidth": 2},
        )
        for at in autotexts: at.set_fontsize(8); at.set_color(COLORS["bg"])
        for t  in texts:     t.set_color(COLORS["text"]); t.set_fontsize(8)
        ax.set_title("Sales by Ship Mode")
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Monthly revenue trend ─────────────────────────────────────────────────
    section("Monthly Revenue Trend (2024–2025)")
    monthly = df_f.groupby(["Order_Year","Order_Month"]).agg(
        Sales=("Sales","sum"), Profit=("Gross Profit","sum")
    ).reset_index()
    monthly["Period"] = monthly["Order_Year"].astype(str) + "-" + \
                        monthly["Order_Month"].astype(str).str.zfill(2)
    monthly = monthly.sort_values("Period")

    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.fill_between(range(len(monthly)), monthly["Sales"],
                    alpha=0.18, color=COLORS["primary"])
    ax.plot(range(len(monthly)), monthly["Sales"],
            color=COLORS["primary"], linewidth=2.2, label="Sales")
    ax.fill_between(range(len(monthly)), monthly["Profit"],
                    alpha=0.18, color=COLORS["success"])
    ax.plot(range(len(monthly)), monthly["Profit"],
            color=COLORS["success"], linewidth=2.2, label="Profit", linestyle="--")
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly["Period"], rotation=45, ha="right", fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    ax.set_title("Monthly Sales & Profit Trend"); ax.legend(); ax.grid(axis="y")
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    insight(
        "Chocolate division dominates with 96.6% of all orders. Standard Class is the "
        "most-used ship mode (~62%). Revenue is consistent month-over-month with no "
        "significant seasonal dip, indicating stable demand and supply."
    )

    # ── Factory map (text table fallback since no plotly/folium) ─────────────
    section("Factory Network Summary")
    fac_perf = df_f.groupby("Factory").agg(
        Orders   =("Row ID","count"),
        Revenue  =("Sales","sum"),
        Profit   =("Gross Profit","sum"),
        Avg_LT   =("Lead_Time","mean"),
        Avg_Dist =("Distance_Miles","mean"),
        Margin   =("Profit_Margin","mean"),
    ).reset_index()
    fac_perf["Revenue"] = fac_perf["Revenue"].map("${:,.0f}".format)
    fac_perf["Profit"]  = fac_perf["Profit"].map("${:,.0f}".format)
    fac_perf["Avg_LT"]  = fac_perf["Avg_LT"].map("{:.1f} days".format)
    fac_perf["Avg_Dist"]= fac_perf["Avg_Dist"].map("{:,.0f} mi".format)
    fac_perf["Margin"]  = fac_perf["Margin"].map("{:.1%}".format)
    st.dataframe(fac_perf, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FACTORY NETWORK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page_name == "Factory Network Analysis":
    st.markdown("## 🏭 Factory Network Analysis")
    st.caption("Deep-dive into each factory's performance, product mix, and logistics footprint.")

    set_dark_style()

    # ── Factory KPIs ─────────────────────────────────────────────────────────
    fac_grp = df_f.groupby("Factory").agg(
        Orders  =("Row ID","count"),
        Revenue =("Sales","sum"),
        Profit  =("Gross Profit","sum"),
        Avg_LT  =("Lead_Time","mean"),
        Avg_Dist=("Distance_Miles","mean"),
    ).reset_index()

    cols = st.columns(len(fac_grp))
    for i, (_, row) in enumerate(fac_grp.iterrows()):
        cols[i].markdown(kpi_card(
            f"{row['Orders']:,}", row["Factory"],
            f"${row['Revenue']:,.0f} rev · {row['Avg_LT']:.1f}d avg LT", True
        ), unsafe_allow_html=True)

    # ── Orders & Revenue per factory ─────────────────────────────────────────
    section("Orders & Revenue by Factory")
    col_l, col_r = st.columns(2)

    with col_l:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [FACTORY_COLORS.get(f, "#64748b") for f in fac_grp["Factory"]]
        bars = ax.barh(fac_grp["Factory"], fac_grp["Orders"], color=colors, alpha=0.9)
        for bar in bars:
            ax.text(bar.get_width()+50, bar.get_y()+bar.get_height()/2,
                    f"{bar.get_width():,.0f}", va="center", fontsize=8, color=COLORS["text"])
        ax.set_title("Total Orders by Factory"); ax.grid(axis="x")
        ax.set_xlim(0, fac_grp["Orders"].max() * 1.15)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(fac_grp["Factory"], fac_grp["Revenue"], color=colors, alpha=0.9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
        for bar in bars:
            ax.text(bar.get_width()+500, bar.get_y()+bar.get_height()/2,
                    f"${bar.get_width():,.0f}", va="center", fontsize=8, color=COLORS["text"])
        ax.set_title("Total Revenue by Factory"); ax.grid(axis="x")
        ax.set_xlim(0, fac_grp["Revenue"].max() * 1.15)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Lead time & Distance boxplots ─────────────────────────────────────────
    section("Lead Time & Distance Distribution by Factory")
    col_l, col_r = st.columns(2)

    with col_l:
        fig, ax = plt.subplots(figsize=(6, 4))
        fac_order = sorted(df_f["Factory"].unique())
        data_lt   = [df_f[df_f["Factory"]==f]["Lead_Time"].values for f in fac_order]
        bp = ax.boxplot(data_lt, patch_artist=True, vert=False,
                        medianprops={"color": COLORS["bg"], "linewidth": 2})
        for patch, fac in zip(bp["boxes"], fac_order):
            patch.set_facecolor(FACTORY_COLORS.get(fac, "#64748b"))
            patch.set_alpha(0.85)
        ax.set_yticks(range(1, len(fac_order)+1))
        ax.set_yticklabels(fac_order, fontsize=8)
        ax.set_xlabel("Lead Time (days)"); ax.set_title("Lead Time Distribution")
        ax.grid(axis="x"); fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        fig, ax = plt.subplots(figsize=(6, 4))
        data_dist = [df_f[df_f["Factory"]==f]["Distance_Miles"].values for f in fac_order]
        bp = ax.boxplot(data_dist, patch_artist=True, vert=False,
                        medianprops={"color": COLORS["bg"], "linewidth": 2})
        for patch, fac in zip(bp["boxes"], fac_order):
            patch.set_facecolor(FACTORY_COLORS.get(fac, "#64748b"))
            patch.set_alpha(0.85)
        ax.set_yticks(range(1, len(fac_order)+1))
        ax.set_yticklabels(fac_order, fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:,.0f}mi"))
        ax.set_xlabel("Distance (miles)"); ax.set_title("Shipping Distance Distribution")
        ax.grid(axis="x"); fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Product mix heatmap ───────────────────────────────────────────────────
    section("Product × Factory Order Heatmap")
    piv = df_f.pivot_table(
        index="Product Name", columns="Factory", values="Row ID",
        aggfunc="count", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(piv.values, aspect="auto",
                   cmap=plt.cm.Blues, interpolation="nearest")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=8, rotation=25, ha="right")
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index, fontsize=7)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    fontsize=7, color="white" if v > piv.values.max()*0.5 else COLORS["text"])
    plt.colorbar(im, ax=ax, shrink=0.6, label="Order Count")
    ax.set_title("Orders per Product per Factory")
    fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Region × Factory flow ────────────────────────────────────────────────
    section("Factory → Region Shipping Flow")
    flow = df_f.groupby(["Factory","Region"])["Sales"].sum().reset_index()
    flow_piv = flow.pivot(index="Factory", columns="Region", values="Sales").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(flow_piv.columns))
    region_colors = [COLORS["primary"], COLORS["secondary"], COLORS["success"], COLORS["warning"]]
    for i, (fac, row_data) in enumerate(flow_piv.iterrows()):
        vals = row_data.values
        ax.bar(flow_piv.columns, vals, bottom=bottom,
               color=FACTORY_COLORS.get(fac, "#64748b"), alpha=0.85, label=fac)
        bottom += vals
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    ax.set_title("Revenue Flow: Factory → Region"); ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y"); fig.tight_layout(); st.pyplot(fig); plt.close()

    insight(
        "Lot's O' Nuts handles ~94% of all orders (Chocolate division dominance). "
        "Secret Factory serves the most geographically diverse customer base with the "
        "shortest average distance. Sugar Shack ships from the farthest north, creating "
        "higher mileage for Southern and Pacific customers."
    )

    # ── Factory coordinates table ──────────────────────────────────────────────
    section("Factory Geographic Coordinates")
    dist_tbl = compute_factory_distance_table()
    pivot_dist = dist_tbl.pivot(index="From_Factory", columns="To_Factory",
                                values="Distance_Miles").round(0).astype(int)
    st.dataframe(pivot_dist, use_container_width=True)
    st.caption("Inter-factory distances (miles). Used in reallocation scoring.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LEAD TIME PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page_name == "Lead Time Prediction":
    st.markdown("## ⏱️ Lead Time Prediction")
    st.caption(f"ML Model: **{model_name}** · Target: Shipping Lead Time (days)")

    set_dark_style()

    # ── Model scorecard ───────────────────────────────────────────────────────
    section("Model Performance Scorecard")
    try:
        results_df = pd.read_csv("models/model_results.csv")
        c1, c2, c3, c4 = st.columns(4)
        best_row = results_df.iloc[0]
        c1.markdown(kpi_card(model_name.replace(" ","\n"), "Best Model", None, True), unsafe_allow_html=True)
        c2.markdown(kpi_card(f"{best_row['RMSE']:.4f}", "RMSE (days)", None, True), unsafe_allow_html=True)
        c3.markdown(kpi_card(f"{best_row['MAE']:.4f}",  "MAE (days)",  None, True), unsafe_allow_html=True)
        c4.markdown(kpi_card(f"{best_row['R2']:.4f}",   "R² Score",    None, True), unsafe_allow_html=True)

        section("All Model Comparison")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        metrics = ["RMSE", "MAE", "R2"]
        better  = ["lower", "lower", "higher"]
        for ax, metric, b in zip(axes, metrics, better):
            colors_bar = [COLORS["success"] if metric=="R2" else COLORS["danger"]
                          if (metric!="R2" and v==results_df[metric].max())
                          else COLORS["primary"]
                          for v in results_df[metric]]
            bars = ax.bar(results_df["Model"], results_df[metric],
                          color=[COLORS["primary"]]*len(results_df), alpha=0.85)
            # highlight best
            best_idx = results_df[metric].idxmin() if metric!="R2" else results_df[metric].idxmax()
            bars[best_idx].set_color(COLORS["success"])
            ax.set_title(f"{metric} ({b} is better)")
            ax.set_xticklabels(results_df["Model"], rotation=20, ha="right", fontsize=7)
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                        f"{bar.get_height():.4f}", ha="center", fontsize=7, color=COLORS["text"])
            ax.grid(axis="y")
        fig.tight_layout(); st.pyplot(fig); plt.close()
    except FileNotFoundError:
        st.warning("Run train_model.py first to generate model results.")

    # ── Feature importance ────────────────────────────────────────────────────
    section("Feature Importance")
    try:
        fi_df = pd.read_csv("models/feature_importance.csv")
        fig, ax = plt.subplots(figsize=(10, 5))
        top15 = fi_df.head(15)
        bar_c = [COLORS["primary"] if i>0 else COLORS["success"]
                 for i in range(len(top15))]
        bars = ax.barh(top15["Feature"][::-1], top15["Importance"][::-1],
                       color=bar_c[::-1], alpha=0.9)
        for bar in bars:
            ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                    f"{bar.get_width():.4f}", va="center", fontsize=7.5, color=COLORS["text"])
        ax.set_title(f"Feature Importance — {model_name}")
        ax.set_xlabel("Importance Score"); ax.grid(axis="x")
        fig.tight_layout(); st.pyplot(fig); plt.close()

        insight(
            "Ship_Mode_Code is the dominant predictor (94.5% importance), confirming that "
            "the shipping tier chosen at order placement is the primary driver of lead time. "
            "Distance, state location, and order timing contribute the remaining variance."
        )
    except FileNotFoundError:
        st.warning("Feature importance file not found. Run train_model.py first.")

    # ── Lead time distribution ────────────────────────────────────────────────
    section("Lead Time Distribution by Ship Mode")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, mode in zip(axes, ["Same Day","First Class","Second Class","Standard Class"]):
        vals = df_f[df_f["Ship Mode"]==mode]["Lead_Time"]
        ax.hist(vals, bins=12, color=SHIP_COLORS.get(mode, "#7dd3fc"), alpha=0.85, edgecolor=COLORS["bg"])
        ax.set_title(mode, fontsize=9)
        ax.set_xlabel("Days"); ax.set_ylabel("Count")
        ax.axvline(vals.mean(), color=COLORS["warning"], linewidth=1.5, linestyle="--")
        ax.text(vals.mean()+0.05, ax.get_ylim()[1]*0.88,
                f"μ={vals.mean():.1f}", fontsize=8, color=COLORS["warning"])
        ax.grid(axis="y")
    fig.suptitle("Lead Time Distribution by Ship Mode", fontsize=11, color=COLORS["text"])
    fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Interactive predictor ─────────────────────────────────────────────────
    section("🔢 Interactive Lead Time Predictor")
    st.markdown("Adjust inputs to get a predicted lead time from the ML model.")

    col1, col2, col3 = st.columns(3)
    with col1:
        pred_product  = st.selectbox("Product", sorted(df_f["Product Name"].unique()))
        pred_region   = st.selectbox("Region",  sorted(df_f["Region"].unique()))
    with col2:
        pred_shipmode = st.selectbox("Ship Mode", ["Same Day","First Class","Second Class","Standard Class"])
        pred_units    = st.slider("Units", 1, 14, 5)
    with col3:
        pred_month    = st.slider("Order Month", 1, 12, 6)
        pred_quarter  = st.slider("Order Quarter", 1, 4, 2)

    from feature_engineering import build_features as _bf
    from preprocessing import PRODUCT_FACTORY as _PF, STATE_CENTROIDS as _SC, haversine_scalar as _hav

    factory  = _PF.get(pred_product, ALL_FACTORIES[0])
    f_lat    = FACTORY_COORDS[factory]["lat"]
    f_lon    = FACTORY_COORDS[factory]["lon"]

    subset   = df_feat[df_feat["Product Name"] == pred_product]
    slat     = subset["State_Lat"].mean()
    slon     = subset["State_Lon"].mean()
    dist_val = _hav(f_lat, f_lon, slat, slon)

    region_code   = ["Atlantic","Gulf","Interior","Pacific"].index(pred_region) \
                    if pred_region in ["Atlantic","Gulf","Interior","Pacific"] else 0
    ship_code     = {"Same Day":1,"First Class":2,"Second Class":3,"Standard Class":4}[pred_shipmode]
    division      = subset["Division"].iloc[0] if len(subset)>0 else "Chocolate"
    div_code      = ["Chocolate","Other","Sugar"].index(division) if division in ["Chocolate","Other","Sugar"] else 0
    factory_code  = ALL_FACTORIES.index(factory)

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
    X_pred = pd.DataFrame([feature_row])[ML_FEATURES]
    pred_lt = float(model.predict(X_pred)[0])

    st.markdown(f"""
    <div style="background: linear-gradient(135deg,#1e293b,#0f172a); border:1px solid #7dd3fc;
                border-radius:12px; padding:24px; text-align:center; margin-top:16px;">
      <div style="font-size:3.5rem; font-weight:800; color:#7dd3fc;">{pred_lt:.1f}</div>
      <div style="font-size:1rem; color:#94a3b8; margin-top:4px;">Predicted Lead Time (days)</div>
      <div style="margin-top:12px; font-size:.85rem; color:#64748b;">
        Factory: <b style="color:#e2e8f0;">{factory}</b> &nbsp;·&nbsp;
        Distance: <b style="color:#e2e8f0;">{dist_val:,.0f} miles</b> &nbsp;·&nbsp;
        Model: <b style="color:#e2e8f0;">{model_name}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FACTORY OPTIMIZATION SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page_name == "Factory Optimization Simulator":
    st.markdown("## ⚙️ Factory Optimization Simulator")
    st.caption("Simulate all factory assignments per product and compare reallocation scores.")

    set_dark_style()

    # ── Full simulation table ─────────────────────────────────────────────────
    section("Full Simulation Results — All Products × All Factories")

    # Colour-coded score column
    def colour_score(val):
        if val >= 65:   return "background-color:#14532d; color:#4ade80"
        elif val >= 50: return "background-color:#713f12; color:#fbbf24"
        else:           return "background-color:#7f1d1d; color:#f87171"

    display_cols = [
        "Product","Current_Factory","Candidate_Factory","Is_Current",
        "Predicted_Lead_Time","Lead_Time_Improvement_Pct",
        "Candidate_Distance","Distance_Reduction_Pct",
        "Reallocation_Score","Risk_Level","Data_Confidence"
    ]
    sim_display = sim_df[[c for c in display_cols if c in sim_df.columns]].copy()
    sim_display["Reallocation_Score"] = sim_display["Reallocation_Score"].round(1)
    st.dataframe(
        sim_display.style.applymap(colour_score, subset=["Reallocation_Score"]),
        use_container_width=True, height=400
    )

    # ── Score heatmap ─────────────────────────────────────────────────────────
    section("Reallocation Score Heatmap — Product × Factory")
    piv_score = sim_df.pivot_table(
        index="Product", columns="Candidate_Factory",
        values="Reallocation_Score"
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.cm.RdYlGn
    im   = ax.imshow(piv_score.values, aspect="auto", cmap=cmap, vmin=30, vmax=75)
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
    ax.set_title("Reallocation Score: Green = Best Candidate, Red = Worst")
    fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Per-product deep dive ─────────────────────────────────────────────────
    section("Per-Product Factory Comparison")
    selected_product = st.selectbox(
        "Select Product to Analyse",
        sorted(sim_df["Product"].unique())
    )
    prod_sim = sim_df[sim_df["Product"]==selected_product].sort_values(
        "Reallocation_Score", ascending=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics  = ["Predicted_Lead_Time","Candidate_Distance","Reallocation_Score"]
    titles   = ["Predicted Lead Time (days)","Distance to Customers (miles)","Reallocation Score"]
    hi_is    = ["low","low","high"]

    for ax, metric, title, hi in zip(axes, metrics, titles, hi_is):
        vals   = prod_sim[metric].values
        facs   = prod_sim["Candidate_Factory"].values
        is_cur = prod_sim["Is_Current"].values
        colors_b = []
        for f, ic in zip(facs, is_cur):
            if ic:
                colors_b.append(COLORS["warning"])
            elif (hi=="high" and metric=="Reallocation_Score"):
                score = prod_sim[prod_sim["Candidate_Factory"]==f]["Reallocation_Score"].values[0]
                colors_b.append(COLORS["success"] if score==vals.max() else COLORS["primary"])
            else:
                colors_b.append(COLORS["primary"])

        bars = ax.bar(range(len(facs)), vals, color=colors_b, alpha=0.9)
        ax.set_xticks(range(len(facs)))
        ax.set_xticklabels(facs, rotation=25, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+vals.max()*0.02,
                    f"{bar.get_height():.1f}", ha="center", fontsize=7, color=COLORS["text"])

    axes[0].set_ylabel("Days")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:,.0f}"))
    fig.suptitle(f"{selected_product} — Factory Comparison", color=COLORS["text"])
    st.markdown("🟡 = Current Factory &nbsp; 🟢 = Best Alternative &nbsp; 🔵 = Other")
    fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Download full simulation ──────────────────────────────────────────────
    st.download_button(
        "⬇️ Download Full Simulation Table (CSV)",
        sim_df.to_csv(index=False),
        "nassau_candy_simulation.csv",
        "text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RECOMMENDATION DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page_name == "Recommendation Dashboard":
    st.markdown("## 🎯 Recommendation Dashboard")
    st.caption("Top-20 factory reallocation recommendations ranked by composite score.")

    set_dark_style()

    # ── Summary KPIs ─────────────────────────────────────────────────────────
    improvements = recs_df[recs_df.get("LT_Improvement_%", recs_df.columns[0]).replace("LT_Improvement_%","LT_Improvement_%") in recs_df.columns and
                           recs_df["LT_Improvement_%"] > 0] if "LT_Improvement_%" in recs_df.columns else recs_df

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card(len(recs_df), "Total Recommendations", None, True), unsafe_allow_html=True)
    if "LT_Improvement_%" in recs_df.columns:
        pos = recs_df[recs_df["LT_Improvement_%"] > 0]
        c2.markdown(kpi_card(len(pos), "Products with LT Gains", None, True), unsafe_allow_html=True)
        c3.markdown(kpi_card(
            f"{recs_df['LT_Improvement_%'].mean():.1f}%",
            "Avg LT Improvement", None, True
        ), unsafe_allow_html=True)
    if "Distance_Reduction_%" in recs_df.columns:
        c4.markdown(kpi_card(
            f"{recs_df['Distance_Reduction_%'].mean():.1f}%",
            "Avg Distance Reduction", None, True
        ), unsafe_allow_html=True)

    # ── Top recommendations table ─────────────────────────────────────────────
    section("Top-20 Reallocation Recommendations")

    def risk_badge(risk):
        if "Low Risk" in risk and "Low Data" not in risk:
            return "🟢 Low Risk"
        elif "Moderate" in risk:
            return "🟡 Moderate Risk"
        else:
            return "🔴 High Risk"

    recs_display = recs_df.copy()
    if "Risk_Level" in recs_display.columns:
        recs_display["Risk"] = recs_display["Risk_Level"].map(risk_badge)

    st.dataframe(recs_display, use_container_width=True, hide_index=True, height=460)

    # ── Score bar chart ───────────────────────────────────────────────────────
    section("Reallocation Score by Product")
    fig, ax = plt.subplots(figsize=(12, 5))
    score_col = "Reallocation_Score"
    if score_col not in recs_df.columns:
        score_col = recs_df.columns[-3]
    score_vals = recs_df[score_col].values
    bar_colors = [
        COLORS["success"] if v >= 65 else
        COLORS["warning"] if v >= 50 else
        COLORS["danger"]
        for v in score_vals
    ]
    bars = ax.barh(recs_df["Product"], score_vals, color=bar_colors, alpha=0.88)
    ax.axvline(50, color=COLORS["subtext"], linewidth=1, linestyle="--", label="Threshold (50)")
    ax.axvline(65, color=COLORS["success"], linewidth=1, linestyle="--", label="Good (65+)")
    for bar in bars:
        ax.text(bar.get_width()+0.4, bar.get_y()+bar.get_height()/2,
                f"{bar.get_width():.1f}", va="center", fontsize=8, color=COLORS["text"])
    ax.set_xlabel("Reallocation Score (0–100)")
    ax.set_title("Composite Reallocation Score: 0.50×LT + 0.30×Dist + 0.20×Profit")
    ax.legend(); ax.grid(axis="x"); ax.set_xlim(0, 85)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Lead time improvement chart ───────────────────────────────────────────
    if "LT_Improvement_%" in recs_df.columns:
        section("Lead Time Improvement % by Product")
        fig, ax = plt.subplots(figsize=(12, 4.5))
        lt_vals = recs_df["LT_Improvement_%"].values
        bar_c   = [COLORS["success"] if v>0 else COLORS["danger"] for v in lt_vals]
        ax.barh(recs_df["Product"], lt_vals, color=bar_c, alpha=0.88)
        ax.axvline(0, color=COLORS["subtext"], linewidth=1)
        ax.set_xlabel("Lead Time Improvement (%)")
        ax.set_title("Positive = faster delivery with recommended factory")
        ax.grid(axis="x"); fig.tight_layout(); st.pyplot(fig); plt.close()

    insight(
        "Fizzy Lifting Drinks and Laffy Taffy show the highest reallocation potential "
        "(LT improvement >10%). Lot's O' Nuts Chocolate products have strong distance "
        "reduction opportunity via Secret Factory. Products flagged High Risk (Low Data) "
        "should be validated with additional sales data before reallocation."
    )

    # ── Download ───────────────────────────────────────────────────────────────
    st.download_button(
        "⬇️ Download Recommendations (CSV)",
        recs_df.to_csv(index=False),
        "nassau_candy_recommendations.csv",
        "text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — WHAT-IF SCENARIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page_name == "What-If Scenario Analysis":
    st.markdown("## 🔮 What-If Scenario Analysis")
    st.caption("Simulate a specific product × region × ship mode to compare Current vs. Recommended factory.")

    set_dark_style()

    col1, col2, col3 = st.columns(3)
    with col1:
        sc_product = st.selectbox("Product", sorted(df_raw["Product Name"].unique()), key="sc_p")
    with col2:
        sc_region  = st.selectbox("Region",  sorted(df_raw["Region"].unique()), key="sc_r")
    with col3:
        sc_ship    = st.selectbox("Ship Mode",
                                  ["Standard Class","Second Class","First Class","Same Day"],
                                  key="sc_s")

    # Factory override
    current_factory = PRODUCT_FACTORY.get(sc_product, ALL_FACTORIES[0])
    rec_factory_opts = [f for f in ALL_FACTORIES if f != current_factory]
    sc_rec_fac = st.selectbox(
        "Recommended Factory (override or leave as auto)",
        ["Auto (from engine)"] + rec_factory_opts,
        key="sc_rf"
    )
    rec_fac_input = None if sc_rec_fac == "Auto (from engine)" else sc_rec_fac

    if st.button("▶ Run Scenario", type="primary", use_container_width=True):
        with st.spinner("Running simulation…"):
            result = run_scenario(
                product_name        = sc_product,
                region              = sc_region,
                ship_mode           = sc_ship,
                recommended_factory = rec_fac_input,
                df                  = df_raw,
            )

        section("Scenario Results")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"""
            <div class="kpi-card" style="border-color:#fbbf24;">
              <div style="font-size:.75rem; color:#fbbf24; text-transform:uppercase; letter-spacing:.1em; margin-bottom:12px;">
                📦 Current Assignment
              </div>
              <div style="font-size:1.6rem; font-weight:700; color:#fbbf24;">{result['current_factory']}</div>
              <hr style="border-color:#2e3650; margin:12px 0;">
              <table style="width:100%; font-size:.84rem; color:#cbd5e1;">
                <tr><td>Lead Time</td><td align="right"><b>{result['current_lead_time']:.1f} days</b></td></tr>
                <tr><td>Distance</td><td align="right"><b>{result['current_distance_miles']:,.0f} mi</b></td></tr>
                <tr><td>Profit Margin</td><td align="right"><b>{result['avg_profit_margin_pct']:.1f}%</b></td></tr>
              </table>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="kpi-card" style="border-color:#4ade80;">
              <div style="font-size:.75rem; color:#4ade80; text-transform:uppercase; letter-spacing:.1em; margin-bottom:12px;">
                🚀 Recommended Assignment
              </div>
              <div style="font-size:1.6rem; font-weight:700; color:#4ade80;">{result['recommended_factory']}</div>
              <hr style="border-color:#2e3650; margin:12px 0;">
              <table style="width:100%; font-size:.84rem; color:#cbd5e1;">
                <tr><td>Lead Time</td><td align="right"><b>{result['recommended_lead_time']:.1f} days</b></td></tr>
                <tr><td>Distance</td><td align="right"><b>{result['recommended_distance_miles']:,.0f} mi</b></td></tr>
                <tr><td>Est. Saving/Order</td><td align="right"><b>${result['est_saving_per_order_usd']:.4f}</b></td></tr>
              </table>
            </div>
            """, unsafe_allow_html=True)

        section("Impact Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi_card(
            f"{result['lead_time_saved_days']:.1f}d",
            "Lead Time Saved",
            f"{result['lead_time_improvement_pct']:.1f}%",
            result['lead_time_saved_days'] > 0
        ), unsafe_allow_html=True)
        c2.markdown(kpi_card(
            f"{result['distance_saved_miles']:,.0f}mi",
            "Distance Reduced",
            f"{result['distance_reduction_pct']:.1f}%",
            result['distance_saved_miles'] > 0
        ), unsafe_allow_html=True)
        c3.markdown(kpi_card(
            f"{result['efficiency_gain_pct']:.1f}%",
            "Efficiency Gain",
            None, True
        ), unsafe_allow_html=True)
        c4.markdown(kpi_card(
            result['reallocation_score'],
            "Reallocation Score",
            result['risk_level'], result['reallocation_score'] >= 50
        ), unsafe_allow_html=True)

        # ── Visual comparison bars ────────────────────────────────────────────
        section("Side-by-Side Comparison")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        labels = ["Current\nFactory", "Recommended\nFactory"]

        ax = axes[0]
        vals = [result["current_lead_time"], result["recommended_lead_time"]]
        bars = ax.bar(labels, vals,
                      color=[COLORS["warning"], COLORS["success"]], alpha=0.88)
        ax.set_title("Lead Time (days)"); ax.grid(axis="y")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    f"{bar.get_height():.1f}d", ha="center", fontsize=11,
                    fontweight="bold", color=COLORS["text"])

        ax = axes[1]
        vals = [result["current_distance_miles"], result["recommended_distance_miles"]]
        bars = ax.bar(labels, vals,
                      color=[COLORS["warning"], COLORS["success"]], alpha=0.88)
        ax.set_title("Distance to Customers (miles)"); ax.grid(axis="y")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:,.0f}"))
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                    f"{bar.get_height():,.0f}", ha="center", fontsize=10,
                    fontweight="bold", color=COLORS["text"])

        fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown(f"""
        <div class="insight-box">
          <b>Scenario Summary</b><br>
          Moving <b>{sc_product}</b> from <b>{result['current_factory']}</b> to
          <b>{result['recommended_factory']}</b> for <b>{sc_region}</b> orders via
          <b>{sc_ship}</b> reduces average delivery time by
          <b>{result['lead_time_saved_days']:.1f} days ({result['lead_time_improvement_pct']:.1f}%)</b>
          and cuts shipping distance by <b>{result['distance_saved_miles']:,.0f} miles</b>.
          Risk Level: <b>{result['risk_level']}</b>.
          Reallocation Score: <b>{result['reallocation_score']:.1f}/100</b>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Configure your scenario above and click **▶ Run Scenario** to simulate.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════
elif page_name == "Risk Assessment":
    st.markdown("## ⚠️ Risk Assessment")
    st.caption("Evaluate reallocation risks across products, factories, and regions.")

    set_dark_style()

    # ── Risk distribution KPIs ────────────────────────────────────────────────
    risk_counts = sim_df["Risk_Level"].value_counts()
    total_sims  = len(sim_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card(total_sims, "Total Simulations", None, True), unsafe_allow_html=True)
    c2.markdown(kpi_card(
        risk_counts.get("Low Risk", 0), "Low Risk Scenarios",
        f"{risk_counts.get('Low Risk',0)/total_sims*100:.1f}%", True
    ), unsafe_allow_html=True)
    c3.markdown(kpi_card(
        risk_counts.get("Moderate Risk", 0), "Moderate Risk",
        f"{risk_counts.get('Moderate Risk',0)/total_sims*100:.1f}%", False
    ), unsafe_allow_html=True)
    c4.markdown(kpi_card(
        sum(v for k,v in risk_counts.items() if "High" in k), "High Risk",
        "Includes Low-Data products", False
    ), unsafe_allow_html=True)

    # ── Risk distribution chart ───────────────────────────────────────────────
    section("Risk Level Distribution")
    col_l, col_r = st.columns(2)

    with col_l:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        risk_labels = risk_counts.index.tolist()
        risk_vals   = risk_counts.values.tolist()
        risk_color_map = {
            "Low Risk":             COLORS["success"],
            "Moderate Risk":        COLORS["warning"],
            "High Risk (Low Data)": COLORS["danger"],
            "High Risk":            "#dc2626",
        }
        colors_r = [risk_color_map.get(r, COLORS["primary"]) for r in risk_labels]
        wedges, texts, autotexts = ax.pie(
            risk_vals, labels=risk_labels, colors=colors_r,
            autopct="%1.1f%%", startangle=120,
            pctdistance=0.78,
            wedgeprops={"edgecolor": COLORS["bg"], "linewidth": 2}
        )
        for at in autotexts: at.set_fontsize(8); at.set_color(COLORS["bg"])
        for t  in texts:     t.set_color(COLORS["text"]); t.set_fontsize(7.5)
        ax.set_title("Risk Level Distribution")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        # Risk by product (best recommendation only)
        risk_prod = best_df[["Product","Risk_Level","Reallocation_Score","Data_Confidence"]].copy()
        fig, ax = plt.subplots(figsize=(6, 4))
        bar_c = [risk_color_map.get(r, COLORS["primary"]) for r in risk_prod["Risk_Level"]]
        ax.barh(risk_prod["Product"], risk_prod["Reallocation_Score"],
                color=bar_c, alpha=0.88)
        ax.axvline(50, color=COLORS["subtext"], linewidth=1, linestyle="--")
        ax.axvline(65, color=COLORS["success"], linewidth=1, linestyle="--")
        ax.set_xlabel("Reallocation Score")
        ax.set_title("Risk per Product (best recommendation)")
        ax.grid(axis="x"); ax.set_xlim(0, 85)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Data confidence analysis ──────────────────────────────────────────────
    section("Data Confidence Analysis")
    conf = df_f.groupby("Product Name").agg(
        Orders      = ("Row ID","count"),
        Avg_Margin  = ("Profit_Margin","mean"),
        Confidence  = ("Data_Confidence","first"),
    ).reset_index().sort_values("Orders", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    bar_colors = [COLORS["success"] if c=="High" else COLORS["danger"]
                  for c in conf["Confidence"]]
    bars = ax.bar(range(len(conf)), conf["Orders"], color=bar_colors, alpha=0.85)
    ax.set_xticks(range(len(conf)))
    ax.set_xticklabels(conf["Product Name"], rotation=35, ha="right", fontsize=7.5)
    ax.axhline(100, color=COLORS["warning"], linewidth=1.5, linestyle="--",
               label="Min. Confidence Threshold (100 orders)")
    ax.set_title("Order Count by Product (Green=High Confidence, Red=Low Confidence)")
    ax.set_ylabel("Number of Orders"); ax.legend(); ax.grid(axis="y")
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                f"{bar.get_height():,}", ha="center", fontsize=7, color=COLORS["text"])
    fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Margin stability across factories ─────────────────────────────────────
    section("Profit Margin Stability by Factory")
    fig, ax = plt.subplots(figsize=(10, 4))
    fac_order = sorted(df_f["Factory"].unique())
    margin_data = [df_f[df_f["Factory"]==f]["Profit_Margin"].values for f in fac_order]
    bp = ax.boxplot(margin_data, patch_artist=True, vert=True,
                    medianprops={"color": COLORS["bg"], "linewidth": 2.5})
    for patch, fac in zip(bp["boxes"], fac_order):
        patch.set_facecolor(FACTORY_COLORS.get(fac, "#64748b"))
        patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(fac_order)+1))
    ax.set_xticklabels(fac_order, rotation=15, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax.set_title("Profit Margin Distribution by Factory")
    ax.set_ylabel("Profit Margin"); ax.grid(axis="y")
    fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Risk matrix ───────────────────────────────────────────────────────────
    section("Risk Matrix — Reallocation Score vs Data Confidence")
    fig, ax = plt.subplots(figsize=(10, 5))
    for _, row in best_df.iterrows():
        x = row.get("Reallocation_Score", 50)
        y = row.get("Distance_Reduction_Pct", 0)
        c = risk_color_map.get(row.get("Risk_Level",""), COLORS["primary"])
        conf_val = row.get("Data_Confidence","High")
        size = 120 if conf_val == "High" else 60
        ax.scatter(x, y, c=c, s=size, alpha=0.85, edgecolors=COLORS["border"], linewidths=1)
        ax.annotate(row["Product"], (x, y),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=6.5, color=COLORS["subtext"])

    ax.axvline(50, color=COLORS["subtext"], linewidth=1, linestyle="--")
    ax.axvline(65, color=COLORS["success"], linewidth=1, linestyle="--")
    ax.axhline(0,  color=COLORS["subtext"], linewidth=1, linestyle="--")
    ax.set_xlabel("Reallocation Score (higher = better candidate)")
    ax.set_ylabel("Distance Reduction % (positive = shorter routes)")
    ax.set_title("Risk Matrix: Large dot = High Data Confidence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    insight(
        "Products with Low Data Confidence (Sugar, Other divisions — <100 orders) should "
        "be treated as High Risk regardless of score. Lot's O' Nuts Chocolate products "
        "carry Moderate Risk with solid data backing. No product currently sits in the "
        "Low Risk + High Confidence quadrant — the primary opportunity for leadership."
    )

    # ── Risk mitigation table ─────────────────────────────────────────────────
    section("Risk Mitigation Recommendations")
    risk_guide = pd.DataFrame([
        {"Risk Level": "🟢 Low Risk",     "Action": "Proceed with reallocation",    "Timeline": "Q1",     "Priority": "High"},
        {"Risk Level": "🟡 Moderate Risk","Action": "Pilot with 20% of orders first","Timeline": "Q2",    "Priority": "Medium"},
        {"Risk Level": "🔴 High Risk",    "Action": "Collect more data, defer decision","Timeline": "Q3+","Priority": "Low"},
        {"Risk Level": "⚪ Low Data",     "Action": "Run targeted A/B test",         "Timeline": "Q2–Q3", "Priority": "Research"},
    ])
    st.dataframe(risk_guide, use_container_width=True, hide_index=True)
