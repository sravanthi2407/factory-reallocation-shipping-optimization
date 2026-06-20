"""
optimization_engine.py
Nassau Candy Distributor — Factory Reallocation Simulation Engine

For every product, simulates assignment to ALL 5 factories and scores each
alternative using the composite reallocation score formula:
  Score = 0.50 × Lead_Time_Improvement
        + 0.30 × Distance_Reduction
        + 0.20 × Profit_Stability
"""

import numpy as np
import pandas as pd
from preprocessing import (
    FACTORY_COORDS, PRODUCT_FACTORY, STATE_CENTROIDS,
    LEAD_TIME_BASE, haversine_scalar
)
from feature_engineering import compute_product_summary, build_features, ML_FEATURES
from train_model import load_best_model

ALL_FACTORIES = list(FACTORY_COORDS.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def avg_distance_to_region(factory_name: str, region: str,
                            state_region_map: dict) -> float:
    """
    Average haversine distance from a factory to all states in a given region.
    """
    states = [s for s, r in state_region_map.items() if r == region]
    if not states:
        return 9999.0
    distances = []
    for state in states:
        if state in STATE_CENTROIDS:
            slat, slon = STATE_CENTROIDS[state]
            d = haversine_scalar(
                FACTORY_COORDS[factory_name]["lat"],
                FACTORY_COORDS[factory_name]["lon"],
                slat, slon
            )
            distances.append(d)
    return float(np.mean(distances)) if distances else 9999.0


def build_state_region_map(df: pd.DataFrame) -> dict:
    """Build {state: region} lookup from dataset."""
    return df.drop_duplicates("State/Province").set_index(
        "State/Province")["Region"].to_dict()


def simulate_factory_assignment(
    product_name: str,
    candidate_factory: str,
    product_df: pd.DataFrame,
    model,
    feature_names: list,
    state_region_map: dict,
) -> dict:
    """
    Simulate assigning `product_name` to `candidate_factory`.
    Returns a dict with predicted lead time and distance for the candidate.
    """
    f_lat = FACTORY_COORDS[candidate_factory]["lat"]
    f_lon = FACTORY_COORDS[candidate_factory]["lon"]

    # Build synthetic feature rows for each order of this product
    rows = product_df.copy()
    rows["Factory_Lat"] = f_lat
    rows["Factory_Lon"] = f_lon
    rows["Distance_Miles"] = haversine_scalar(
        f_lat, f_lon,
        rows["State_Lat"].values, rows["State_Lon"].values
    ) if len(rows) == 1 else np.array([
        haversine_scalar(f_lat, f_lon, r["State_Lat"], r["State_Lon"])
        for _, r in rows.iterrows()
    ])
    rows["Factory_Code"] = ALL_FACTORIES.index(candidate_factory)

    X = rows[feature_names]
    preds = model.predict(X)
    avg_dist = rows["Distance_Miles"].mean()

    return {
        "Factory": candidate_factory,
        "Predicted_Lead_Time": float(np.mean(preds)),
        "Avg_Distance_Miles": float(avg_dist),
    }


def reallocation_score(
    current_lt: float, cand_lt: float,
    current_dist: float, cand_dist: float,
    current_margin: float, cand_margin: float,
    w_lt: float = 0.50, w_dist: float = 0.30, w_profit: float = 0.20,
) -> tuple[float, float, float, float]:
    """
    Composite reallocation score (0–100).
    Returns (score, lt_improvement_pct, dist_reduction_pct, profit_stability_pct).
    """
    lt_imp   = (current_lt   - cand_lt)   / (current_lt   + 1e-9) * 100
    dist_red = (current_dist - cand_dist) / (current_dist + 1e-9) * 100
    # Profit stability: penalise margin drops, reward improvements
    prof_stab = min(100, max(-100, (cand_margin - current_margin) / (current_margin + 1e-9) * 100))

    # Normalise each component to [0, 100] contribution
    lt_score    = min(100, max(0, 50 + lt_imp))    # 0 = doubled LT, 100 = halved LT
    dist_score  = min(100, max(0, 50 + dist_red))
    profit_score = min(100, max(0, 50 + prof_stab))

    composite = w_lt * lt_score + w_dist * dist_score + w_profit * profit_score
    return composite, lt_imp, dist_red, prof_stab


def assign_risk_level(score: float, data_confidence: str) -> str:
    """Assign risk level based on score and data confidence."""
    if data_confidence == "Low":
        return "High Risk (Low Data)"
    if score >= 65:
        return "Low Risk"
    elif score >= 45:
        return "Moderate Risk"
    else:
        return "High Risk"


# ── Main Engine ───────────────────────────────────────────────────────────────

def run_reallocation_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core engine: for each product, simulate all factory assignments,
    score them, and return the full simulation table.
    """
    model, model_name, feature_names = load_best_model()
    df_feat = build_features(df)
    product_summary = compute_product_summary(df_feat)
    state_region_map = build_state_region_map(df_feat)

    all_simulations = []

    for _, prod_row in product_summary.iterrows():
        product      = prod_row["Product Name"]
        current_fac  = prod_row["Factory"]
        current_lt   = prod_row["Avg_Lead_Time"]
        current_dist = prod_row["Avg_Distance_Miles"]
        current_margin = prod_row["Avg_Profit_Margin"]
        confidence   = prod_row["Data_Confidence"]

        product_df = df_feat[df_feat["Product Name"] == product]

        for candidate in ALL_FACTORIES:
            sim = simulate_factory_assignment(
                product, candidate, product_df, model, feature_names, state_region_map
            )
            cand_lt   = sim["Predicted_Lead_Time"]
            cand_dist = sim["Avg_Distance_Miles"]
            # Assume profit margin is stable across factory (no factory-specific cost data)
            cand_margin = current_margin

            score, lt_imp, dist_red, prof_stab = reallocation_score(
                current_lt, cand_lt, current_dist, cand_dist,
                current_margin, cand_margin
            )
            risk = assign_risk_level(score, confidence)
            is_current = (candidate == current_fac)

            all_simulations.append({
                "Product":              product,
                "Current_Factory":      current_fac,
                "Candidate_Factory":    candidate,
                "Is_Current":           is_current,
                "Data_Confidence":      confidence,
                "Current_Lead_Time":    round(current_lt, 2),
                "Predicted_Lead_Time":  round(cand_lt, 2),
                "Lead_Time_Improvement_Pct": round(lt_imp, 2),
                "Current_Distance":     round(current_dist, 1),
                "Candidate_Distance":   round(cand_dist, 1),
                "Distance_Reduction_Pct": round(dist_red, 2),
                "Profit_Stability_Pct": round(prof_stab, 2),
                "Reallocation_Score":   round(score, 2),
                "Risk_Level":           risk,
            })

    sim_df = pd.DataFrame(all_simulations)
    return sim_df


def get_best_factory_per_product(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the full simulation table, pick the highest-scoring non-current factory
    for each product, plus compute improvement vs. current assignment.
    """
    # Current assignment scores
    current = sim_df[sim_df["Is_Current"]].copy()
    current = current.rename(columns={
        "Reallocation_Score": "Current_Score"
    })[["Product", "Current_Score"]]

    # Best alternative (exclude current)
    best_alt = (
        sim_df[~sim_df["Is_Current"]]
        .sort_values("Reallocation_Score", ascending=False)
        .groupby("Product")
        .first()
        .reset_index()
    )

    recs = best_alt.merge(current, on="Product")
    recs["Score_Improvement"] = (recs["Reallocation_Score"] - recs["Current_Score"]).round(2)

    # Confidence score = Reallocation_Score / 100 for display
    recs["Confidence_Score"] = (recs["Reallocation_Score"] / 100).round(3)

    return recs.sort_values("Reallocation_Score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from preprocessing import get_clean_data
    df = get_clean_data()
    print("Running reallocation engine...")
    sim_df = run_reallocation_engine(df)
    print(f"Simulations generated: {len(sim_df)} rows")

    best = get_best_factory_per_product(sim_df)
    print("\n── Top 10 Reallocation Recommendations ──")
    print(best[[
        "Product", "Current_Factory", "Candidate_Factory",
        "Lead_Time_Improvement_Pct", "Distance_Reduction_Pct",
        "Reallocation_Score", "Risk_Level"
    ]].head(10).to_string(index=False))
