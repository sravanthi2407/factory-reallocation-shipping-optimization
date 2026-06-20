"""
recommendation_engine.py
Nassau Candy Distributor — Top-20 Reallocation Recommendation Engine
"""

import pandas as pd
import numpy as np
from preprocessing import get_clean_data, FACTORY_COORDS
from feature_engineering import build_features, compute_product_summary
from optimization_engine import run_reallocation_engine, get_best_factory_per_product


def generate_recommendations(df: pd.DataFrame | None = None,
                               top_n: int = 20) -> pd.DataFrame:
    """
    End-to-end: load data → run engine → return polished top-N table.
    """
    if df is None:
        df = get_clean_data()

    sim_df = run_reallocation_engine(df)
    best   = get_best_factory_per_product(sim_df)

    # ── Format for display ────────────────────────────────────────────────────
    cols = [
        "Product",
        "Current_Factory",
        "Candidate_Factory",
        "Current_Lead_Time",
        "Predicted_Lead_Time",
        "Lead_Time_Improvement_Pct",
        "Current_Distance",
        "Candidate_Distance",
        "Distance_Reduction_Pct",
        "Profit_Stability_Pct",
        "Reallocation_Score",
        "Confidence_Score",
        "Risk_Level",
        "Data_Confidence",
    ]
    # Keep only columns that exist
    cols = [c for c in cols if c in best.columns]
    recs = best[cols].head(top_n).copy()

    # Human-readable rename
    recs = recs.rename(columns={
        "Candidate_Factory":           "Recommended_Factory",
        "Lead_Time_Improvement_Pct":   "LT_Improvement_%",
        "Distance_Reduction_Pct":      "Distance_Reduction_%",
        "Profit_Stability_Pct":        "Profit_Stability_%",
        "Current_Lead_Time":           "Current_LT_Days",
        "Predicted_Lead_Time":         "Recommended_LT_Days",
        "Current_Distance":            "Current_Dist_Miles",
        "Candidate_Distance":          "Recommended_Dist_Miles",
    })

    return recs


def get_recommendation_for_product(product_name: str,
                                    sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return full ranked simulation table for a single product.
    Sorted best → worst by Reallocation_Score.
    """
    prod_sim = sim_df[sim_df["Product"] == product_name].copy()
    return prod_sim.sort_values("Reallocation_Score", ascending=False).reset_index(drop=True)


def get_executive_summary(recs: pd.DataFrame, df: pd.DataFrame) -> dict:
    """
    High-level summary KPIs for the executive dashboard.
    """
    df_feat  = build_features(df)
    prod_sum = compute_product_summary(df_feat)

    # Products where reallocation is recommended (score > current)
    improvements = recs[recs["LT_Improvement_%"] > 0]
    
    avg_lt_saved   = improvements["LT_Improvement_%"].mean() if len(improvements) else 0
    avg_dist_saved  = improvements["Distance_Reduction_%"].mean() if len(improvements) else 0
    products_to_move = len(improvements)

    total_orders    = len(df)
    total_revenue   = df["Sales"].sum()
    total_profit    = df["Gross Profit"].sum()
    overall_margin  = total_profit / total_revenue * 100

    # Estimated annual savings (proxy: % distance reduction × total cost)
    est_cost_saving_pct = avg_dist_saved * 0.3  # rough 30¢/mile coefficient
    est_saving_usd = df["Cost"].sum() * est_cost_saving_pct / 100

    return {
        "total_orders":        total_orders,
        "total_revenue":       round(total_revenue, 2),
        "total_profit":        round(total_profit, 2),
        "overall_margin_pct":  round(overall_margin, 2),
        "unique_products":     df["Product Name"].nunique(),
        "unique_customers":    df["Customer ID"].nunique(),
        "products_to_reallocate": products_to_move,
        "avg_lt_improvement_pct": round(avg_lt_saved, 2),
        "avg_dist_reduction_pct": round(avg_dist_saved, 2),
        "est_cost_saving_usd": round(est_saving_usd, 2),
    }


if __name__ == "__main__":
    print("Generating Top-20 Reallocation Recommendations...\n")
    df   = get_clean_data()
    recs = generate_recommendations(df, top_n=20)
    print(recs.to_string(index=False))

    print("\n── Executive Summary ──")
    summary = get_executive_summary(recs, df)
    for k, v in summary.items():
        print(f"  {k:<35}: {v:,.2f}" if isinstance(v, float) else f"  {k:<35}: {v:,}")
