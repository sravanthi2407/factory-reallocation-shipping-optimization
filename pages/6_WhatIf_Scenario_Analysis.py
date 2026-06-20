"""pages/6_WhatIf_Scenario_Analysis.py"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils.shared import (load_data, set_dark_style, kpi_card, section, insight, COLORS)
from utils.styles import inject_css
from preprocessing import FACTORY_COORDS, PRODUCT_FACTORY
from scenario_simulator import run_scenario

st.set_page_config(page_title="What-If Scenarios · Nassau Candy", page_icon="🔮", layout="wide")
inject_css()

df_raw, df_feat, prod_sum, route_stats = load_data()
ALL_FACTORIES = list(FACTORY_COORDS.keys())

st.markdown("## 🔮 What-If Scenario Analysis")
st.caption("Simulate any product × region × ship mode combination. Compare Current vs. Recommended factory.")
set_dark_style()

col1, col2, col3 = st.columns(3)
with col1:
    sc_product = st.selectbox("Product",   sorted(df_raw["Product Name"].unique()))
with col2:
    sc_region  = st.selectbox("Region",    sorted(df_raw["Region"].unique()))
with col3:
    sc_ship    = st.selectbox("Ship Mode", ["Standard Class","Second Class","First Class","Same Day"])

current_factory = PRODUCT_FACTORY.get(sc_product, ALL_FACTORIES[0])
rec_factory_opts = [f for f in ALL_FACTORIES if f != current_factory]
sc_rec_fac = st.selectbox(
    "Recommended Factory (override or 'Auto')",
    ["Auto (from engine)"] + rec_factory_opts
)
rec_fac_input = None if sc_rec_fac == "Auto (from engine)" else sc_rec_fac

if st.button("▶ Run Scenario", type="primary", use_container_width=True):
    with st.spinner("Running simulation…"):
        result = run_scenario(sc_product, sc_region, sc_ship, rec_fac_input, df_raw)

    section("Scenario Results")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="kpi-card" style="border-color:#fbbf24;">
          <div style="font-size:.75rem;color:#fbbf24;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px;">
            📦 Current Assignment
          </div>
          <div style="font-size:1.6rem;font-weight:700;color:#fbbf24;">{result['current_factory']}</div>
          <hr style="border-color:#2e3650;margin:12px 0;">
          <table style="width:100%;font-size:.84rem;color:#cbd5e1;">
            <tr><td>Lead Time</td><td align="right"><b>{result['current_lead_time']:.1f} days</b></td></tr>
            <tr><td>Distance</td><td align="right"><b>{result['current_distance_miles']:,.0f} mi</b></td></tr>
            <tr><td>Profit Margin</td><td align="right"><b>{result['avg_profit_margin_pct']:.1f}%</b></td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card" style="border-color:#4ade80;">
          <div style="font-size:.75rem;color:#4ade80;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px;">
            🚀 Recommended Assignment
          </div>
          <div style="font-size:1.6rem;font-weight:700;color:#4ade80;">{result['recommended_factory']}</div>
          <hr style="border-color:#2e3650;margin:12px 0;">
          <table style="width:100%;font-size:.84rem;color:#cbd5e1;">
            <tr><td>Lead Time</td><td align="right"><b>{result['recommended_lead_time']:.1f} days</b></td></tr>
            <tr><td>Distance</td><td align="right"><b>{result['recommended_distance_miles']:,.0f} mi</b></td></tr>
            <tr><td>Est. Saving/Order</td><td align="right"><b>${result['est_saving_per_order_usd']:.4f}</b></td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    section("Impact Metrics")
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi_card(
        f"{result['lead_time_saved_days']:.1f}d", "Lead Time Saved",
        f"{result['lead_time_improvement_pct']:.1f}%", result['lead_time_saved_days']>0
    ), unsafe_allow_html=True)
    c2.markdown(kpi_card(
        f"{result['distance_saved_miles']:,.0f}mi", "Distance Reduced",
        f"{result['distance_reduction_pct']:.1f}%", result['distance_saved_miles']>0
    ), unsafe_allow_html=True)
    c3.markdown(kpi_card(
        f"{result['efficiency_gain_pct']:.1f}%", "Efficiency Gain", None, True
    ), unsafe_allow_html=True)
    c4.markdown(kpi_card(
        f"{result['reallocation_score']:.1f}", "Reallocation Score",
        result['risk_level'], result['reallocation_score']>=50
    ), unsafe_allow_html=True)

    section("Visual Comparison")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = ["Current Factory", "Recommended Factory"]
    clrs   = [COLORS["warning"], COLORS["success"]]

    for ax, (val_cur, val_rec), title in zip(
        axes,
        [(result["current_lead_time"], result["recommended_lead_time"]),
         (result["current_distance_miles"], result["recommended_distance_miles"])],
        ["Lead Time (days)", "Distance to Customers (miles)"]
    ):
        vals = [val_cur, val_rec]
        bars = ax.bar(labels, vals, color=clrs, alpha=0.88)
        ax.set_title(title); ax.grid(axis="y")
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.02,
                    f"{b.get_height():.1f}", ha="center", fontsize=11,
                    fontweight="bold", color=COLORS["text"])

    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:,.0f}"))
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown(f"""
    <div class="insight-box">
      <b>Summary:</b> Moving <b>{sc_product}</b> from <b>{result['current_factory']}</b>
      to <b>{result['recommended_factory']}</b> for <b>{sc_region}</b> customers via
      <b>{sc_ship}</b> saves <b>{result['lead_time_saved_days']:.1f} days
      ({result['lead_time_improvement_pct']:.1f}%)</b> in delivery time and cuts route
      distance by <b>{result['distance_saved_miles']:,.0f} miles</b>.
      Risk: <b>{result['risk_level']}</b> · Score: <b>{result['reallocation_score']:.1f}/100</b>
    </div>""", unsafe_allow_html=True)

    # Download scenario as CSV
    import pandas as pd
    st.download_button(
        "⬇️ Download Scenario Result",
        pd.DataFrame([result]).to_csv(index=False),
        "scenario_result.csv", "text/csv"
    )
else:
    st.info("⚙️ Configure your scenario above and click **▶ Run Scenario**.")
