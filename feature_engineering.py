"""
feature_engineering.py
Nassau Candy Distributor — Feature Engineering for ML & Optimization
"""

import pandas as pd
import numpy as np
from preprocessing import get_clean_data, FACTORY_COORDS, haversine_scalar

# ── Feature sets used by ML models ───────────────────────────────────────────
ML_FEATURES = [
    "Ship_Mode_Code",
    "Region_Code",
    "Division_Code",
    "Factory_Code",
    "Distance_Miles",
    "Units",
    "Sales",
    "Gross Profit",
    "Cost",
    "Profit_Margin",
    "Profit_Per_Unit",
    "Revenue_Per_Unit",
    "Order_Month",
    "Order_Quarter",
    "Order_DayOfWeek",
    "Factory_Lat",
    "Factory_Lon",
    "State_Lat",
    "State_Lon",
]
ML_TARGET = "Lead_Time"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extend the cleaned DataFrame with advanced features:
    - Route efficiency score
    - Factory regional concentration
    - Rolling product-level profit
    - Ship mode efficiency flag
    - Cost-to-distance ratio
    """
    df = df.copy()

    # ── Route Efficiency Score ─────────────────────────────────────────────
    # Normalise distance and lead time, combine into a 0-1 efficiency score
    dist_norm = (df["Distance_Miles"] - df["Distance_Miles"].min()) / \
                (df["Distance_Miles"].max() - df["Distance_Miles"].min() + 1e-9)
    lt_norm   = (df["Lead_Time"] - df["Lead_Time"].min()) / \
                (df["Lead_Time"].max() - df["Lead_Time"].min() + 1e-9)
    df["Route_Efficiency_Score"] = 1 - (0.6 * dist_norm + 0.4 * lt_norm)

    # ── Ship Mode Efficiency Flag ──────────────────────────────────────────
    # 1 = expedited (Same Day / First Class), 0 = economy
    df["Is_Expedited"] = df["Ship Mode"].isin(["Same Day", "First Class"]).astype(int)

    # ── Cost-to-Distance Ratio ─────────────────────────────────────────────
    df["Cost_Per_Mile"] = df["Cost"] / (df["Distance_Miles"] + 1e-9)

    # ── Profit per Mile ────────────────────────────────────────────────────
    df["Profit_Per_Mile"] = df["Gross Profit"] / (df["Distance_Miles"] + 1e-9)

    # ── Factory-Region Demand Concentration ───────────────────────────────
    factory_region = df.groupby(["Factory", "Region"])["Units"].sum().reset_index()
    factory_region.columns = ["Factory", "Region", "Factory_Region_Units"]
    df = df.merge(factory_region, on=["Factory", "Region"], how="left")

    # ── Product Monthly Revenue Rank ──────────────────────────────────────
    monthly_rev = df.groupby(["Product Name", "Order_Month"])["Sales"].sum().reset_index()
    monthly_rev.columns = ["Product Name", "Order_Month", "Product_Month_Revenue"]
    df = df.merge(monthly_rev, on=["Product Name", "Order_Month"], how="left")

    # ── Distance Tier ─────────────────────────────────────────────────────
    df["Distance_Tier"] = pd.cut(
        df["Distance_Miles"],
        bins=[0, 500, 1000, 1500, 2000, 9999],
        labels=["Very Short", "Short", "Medium", "Long", "Very Long"]
    )

    # ── Lead Time Category ────────────────────────────────────────────────
    df["Lead_Time_Category"] = df["Lead_Time"].map(
        lambda x: "Same Day" if x == 1 else
                  "Express"  if x == 2 else
                  "Standard" if x <= 4 else "Economy"
    )

    return df


def get_ml_ready(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (target) ready for sklearn."""
    df_feat = build_features(df)
    X = df_feat[ML_FEATURES].copy()
    y = df_feat[ML_TARGET].copy()
    # Drop any rows with NaN in features (should be none after clean)
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask]


def compute_factory_distance_table() -> pd.DataFrame:
    """
    Build a pairwise distance table: every factory vs every other factory.
    Used in the optimisation engine to quantify relocation distances.
    """
    factories = list(FACTORY_COORDS.keys())
    rows = []
    for f1 in factories:
        for f2 in factories:
            d = haversine_scalar(
                FACTORY_COORDS[f1]["lat"], FACTORY_COORDS[f1]["lon"],
                FACTORY_COORDS[f2]["lat"], FACTORY_COORDS[f2]["lon"]
            )
            rows.append({"From_Factory": f1, "To_Factory": f2, "Distance_Miles": round(d, 1)})
    return pd.DataFrame(rows)


def compute_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate product-level KPIs used in the recommendation engine.
    """
    summary = df.groupby("Product Name").agg(
        Factory           = ("Factory", "first"),
        Division          = ("Division", "first"),
        Total_Orders      = ("Row ID", "count"),
        Total_Sales       = ("Sales", "sum"),
        Total_Units       = ("Units", "sum"),
        Total_Profit      = ("Gross Profit", "sum"),
        Total_Cost        = ("Cost", "sum"),
        Avg_Lead_Time     = ("Lead_Time", "mean"),
        Avg_Distance_Miles= ("Distance_Miles", "mean"),
        Avg_Profit_Margin = ("Profit_Margin", "mean"),
        Avg_Profit_Per_Unit=("Profit_Per_Unit", "mean"),
        Data_Confidence   = ("Data_Confidence", "first"),
    ).reset_index()
    summary["Avg_Lead_Time"]      = summary["Avg_Lead_Time"].round(2)
    summary["Avg_Distance_Miles"] = summary["Avg_Distance_Miles"].round(1)
    summary["Avg_Profit_Margin"]  = summary["Avg_Profit_Margin"].round(4)
    return summary


def compute_route_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Factory × Region route-level aggregation for clustering input.
    """
    route = df.groupby(["Factory", "Region"]).agg(
        Total_Orders      = ("Row ID", "count"),
        Avg_Lead_Time     = ("Lead_Time", "mean"),
        Avg_Distance      = ("Distance_Miles", "mean"),
        Total_Sales       = ("Sales", "sum"),
        Total_Profit      = ("Gross Profit", "sum"),
        Avg_Profit_Margin = ("Profit_Margin", "mean"),
        Avg_Route_Efficiency = ("Route_Efficiency_Score", "mean"),
    ).reset_index()
    return route


if __name__ == "__main__":
    raw_clean = get_clean_data()
    df_feat   = build_features(raw_clean)
    X, y      = get_ml_ready(raw_clean)

    print(f"Feature matrix: {X.shape}  |  Target: {y.shape}")
    print(f"\nFeatures: {list(X.columns)}")
    print(f"\nLead Time distribution:\n{y.value_counts().sort_index()}")

    print("\n── Factory Distance Table ──")
    print(compute_factory_distance_table().to_string(index=False))

    print("\n── Product Summary ──")
    ps = compute_product_summary(df_feat)
    print(ps[["Product Name", "Factory", "Total_Orders", "Avg_Lead_Time",
              "Avg_Distance_Miles", "Avg_Profit_Margin"]].to_string(index=False))
