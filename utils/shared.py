"""
utils/shared.py
Nassau Candy — Shared utilities for all Streamlit pages
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import matplotlib.pyplot as plt

# ── Colour Palette ─────────────────────────────────────────────────────────────
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

RISK_COLORS = {
    "Low Risk":             "#4ade80",
    "Moderate Risk":        "#fbbf24",
    "High Risk (Low Data)": "#f87171",
    "High Risk":            "#dc2626",
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


def insight(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ── Cached data loaders ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    from preprocessing       import get_clean_data
    from feature_engineering import build_features, compute_product_summary, compute_route_stats
    df       = get_clean_data("data/Nassau_Candy_Distributor.csv")
    df_feat  = build_features(df)
    prod_sum = compute_product_summary(df_feat)
    route_st = compute_route_stats(df_feat)
    return df, df_feat, prod_sum, route_st


@st.cache_data(show_spinner="Generating recommendations…")
def load_recommendations(_df):
    from optimization_engine   import run_reallocation_engine, get_best_factory_per_product
    from recommendation_engine import generate_recommendations, get_executive_summary
    sim_df   = run_reallocation_engine(_df)
    best     = get_best_factory_per_product(sim_df)
    recs     = generate_recommendations(_df, top_n=20)
    exec_sum = get_executive_summary(recs, _df)
    return sim_df, best, recs, exec_sum


@st.cache_resource(show_spinner="Loading ML model…")
def load_model():
    from train_model import load_best_model
    return load_best_model()
