"""
scenario_simulator.py
Nassau Candy Distributor — What-If Scenario Simulation Engine

Allows user to choose: Product × Region × Ship Mode
Compares Current Factory vs Recommended Factory with full KPI breakdown.
"""

import numpy as np
import pandas as pd
from preprocessing import (
    get_clean_data, FACTORY_COORDS, PRODUCT_FACTORY,
    STATE_CENTROIDS, LEAD_TIME_BASE, haversine_scalar
)
from feature_engineering import build_features, ML_FEATURES
from train_model import load_best_model
from optimization_engine import (
    run_reallocation_engine, get_best_factory_per_product,
    simulate_factory_assignment, reallocation_score
)

ALL_FACTORIES = list(FACTORY_COORDS.keys())


def build_scenario_input(
    product_name: str,
    region: str,
    ship_mode: str,
    factory_name: str,
    df_feat: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a representative feature row for a given scenario.
    Uses average values from matching orders where available,
    falling back to dataset-wide means otherwise.
    """
    # Filter orders matching product + region (best match)
    subset = df_feat[
        (df_feat["Product Name"] == product_name) &
        (df_feat["Region"] == region)
    ]
    if len(subset) == 0:
        subset = df_feat[df_feat["Product Name"] == product_name]
    if len(subset) == 0:
        subset = df_feat

    # Average state lat/lon for the region
    state_lat = subset["State_Lat"].mean()
    state_lon = subset["State_Lon"].mean()

    f_lat = FACTORY_COORDS[factory_name]["lat"]
    f_lon = FACTORY_COORDS[factory_name]["lon"]
    dist  = haversine_scalar(f_lat, f_lon, state_lat, state_lon)

    ship_mode_code = {"Same Day": 1, "First Class": 2,
                      "Second Class": 3, "Standard Class": 4}.get(ship_mode, 3)
    region_code    = ["Atlantic", "Gulf", "Interior", "Pacific"].index(region) \
                     if region in ["Atlantic", "Gulf", "Interior", "Pacific"] else 0
    division       = df_feat[df_feat["Product Name"] == product_name]["Division"].iloc[0] \
                     if len(subset) > 0 else "Chocolate"
    division_code  = ["Chocolate", "Other", "Sugar"].index(division) \
                     if division in ["Chocolate", "Other", "Sugar"] else 0
    factory_code   = ALL_FACTORIES.index(factory_name)

    row = {
        "Ship_Mode_Code":    ship_mode_code,
        "Region_Code":       region_code,
        "Division_Code":     division_code,
        "Factory_Code":      factory_code,
        "Distance_Miles":    dist,
        "Units":             subset["Units"].mean(),
        "Sales":             subset["Sales"].mean(),
        "Gross Profit":      subset["Gross Profit"].mean(),
        "Cost":              subset["Cost"].mean(),
        "Profit_Margin":     subset["Profit_Margin"].mean(),
        "Profit_Per_Unit":   subset["Profit_Per_Unit"].mean(),
        "Revenue_Per_Unit":  subset["Revenue_Per_Unit"].mean(),
        "Order_Month":       subset["Order_Month"].median(),
        "Order_Quarter":     subset["Order_Quarter"].median(),
        "Order_DayOfWeek":   subset["Order_DayOfWeek"].median(),
        "Factory_Lat":       f_lat,
        "Factory_Lon":       f_lon,
        "State_Lat":         state_lat,
        "State_Lon":         state_lon,
    }
    return pd.DataFrame([row])[ML_FEATURES]


def run_scenario(
    product_name: str,
    region: str,
    ship_mode: str,
    recommended_factory: str | None = None,
    df: pd.DataFrame | None = None,
) -> dict:
    """
    Run a single What-If scenario.
    Returns a dict with current vs. recommended factory comparison.
    """
    if df is None:
        df = get_clean_data()

    model, model_name, feature_names = load_best_model()
    df_feat = build_features(df)

    current_factory = PRODUCT_FACTORY.get(product_name, ALL_FACTORIES[0])

    # If no recommendation supplied, pick best from engine
    if recommended_factory is None or recommended_factory == current_factory:
        sim_df = run_reallocation_engine(df)
        best   = get_best_factory_per_product(sim_df)
        prod_best = best[best["Product"] == product_name]
        if len(prod_best) > 0:
            recommended_factory = prod_best.iloc[0]["Candidate_Factory"]
        else:
            recommended_factory = current_factory

    # ── Current factory prediction ────────────────────────────────────────────
    X_current = build_scenario_input(
        product_name, region, ship_mode, current_factory, df_feat
    )
    lt_current   = float(model.predict(X_current)[0])
    dist_current = float(X_current["Distance_Miles"].iloc[0])

    # ── Recommended factory prediction ────────────────────────────────────────
    X_rec = build_scenario_input(
        product_name, region, ship_mode, recommended_factory, df_feat
    )
    lt_rec   = float(model.predict(X_rec)[0])
    dist_rec = float(X_rec["Distance_Miles"].iloc[0])

    # ── Metrics ───────────────────────────────────────────────────────────────
    prod_data   = df_feat[df_feat["Product Name"] == product_name]
    avg_margin  = prod_data["Profit_Margin"].mean() if len(prod_data) > 0 else 0.65
    avg_revenue = prod_data["Sales"].mean() if len(prod_data) > 0 else 10.0

    lt_saved    = lt_current - lt_rec
    dist_saved  = dist_current - dist_rec
    lt_imp_pct  = (lt_saved / (lt_current + 1e-9)) * 100
    dist_pct    = (dist_saved / (dist_current + 1e-9)) * 100

    # Estimated cost saving per order (rough: $0.05 per mile saved)
    est_saving_per_order = max(0, dist_saved * 0.05)

    score, _, _, _ = reallocation_score(
        lt_current, lt_rec, dist_current, dist_rec, avg_margin, avg_margin
    )

    if score >= 65:
        risk = "Low Risk"
    elif score >= 45:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    efficiency_gain = max(0, lt_imp_pct * 0.5 + dist_pct * 0.3)

    return {
        "product":                product_name,
        "region":                 region,
        "ship_mode":              ship_mode,
        "current_factory":        current_factory,
        "recommended_factory":    recommended_factory,
        "current_lead_time":      round(lt_current, 2),
        "recommended_lead_time":  round(lt_rec, 2),
        "lead_time_saved_days":   round(lt_saved, 2),
        "lead_time_improvement_pct": round(lt_imp_pct, 2),
        "current_distance_miles": round(dist_current, 1),
        "recommended_distance_miles": round(dist_rec, 1),
        "distance_saved_miles":   round(dist_saved, 1),
        "distance_reduction_pct": round(dist_pct, 2),
        "est_saving_per_order_usd": round(est_saving_per_order, 4),
        "avg_profit_margin_pct":  round(avg_margin * 100, 2),
        "reallocation_score":     round(score, 2),
        "risk_level":             risk,
        "efficiency_gain_pct":    round(efficiency_gain, 2),
        "model_used":             model_name,
    }


def batch_scenarios(
    scenarios: list[dict],
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Run multiple scenarios. Each dict must have keys:
      product, region, ship_mode, (optional) recommended_factory
    """
    if df is None:
        df = get_clean_data()
    results = []
    for s in scenarios:
        res = run_scenario(
            product_name        = s["product"],
            region              = s["region"],
            ship_mode           = s["ship_mode"],
            recommended_factory = s.get("recommended_factory"),
            df                  = df,
        )
        results.append(res)
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Running sample scenario...")
    result = run_scenario(
        product_name = "Wonka Bar - Milk Chocolate",
        region       = "Pacific",
        ship_mode    = "Standard Class",
    )
    print("\n── Scenario Result ──")
    for k, v in result.items():
        print(f"  {k:<38}: {v}")
