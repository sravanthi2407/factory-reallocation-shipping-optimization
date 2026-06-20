"""
preprocessing.py
Nassau Candy Distributor — Data Cleaning & Preparation
"""

import pandas as pd
import numpy as np
import os

# ── Factory Master Table ──────────────────────────────────────────────────────
FACTORY_COORDS = {
    "Lot's O' Nuts":      {"lat": 32.881893, "lon": -111.768036},
    "Wicked Choccy's":   {"lat": 32.076176, "lon": -81.088371},
    "Sugar Shack":        {"lat": 48.119140, "lon": -96.181150},
    "Secret Factory":     {"lat": 41.446333, "lon": -90.565487},
    "The Other Factory":  {"lat": 35.117500, "lon": -89.971107},
}

# Product → Factory mapping (canonical, typo-tolerant)
PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise":   "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":           "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":      "Lot's O' Nuts",   # original typo preserved
    "Wonka Bar - Milk Chocolate":          "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":   "Wicked Choccy's",
    "Laffy Taffy":                         "Sugar Shack",
    "SweeTARTS":                           "Sugar Shack",
    "Nerds":                               "Sugar Shack",
    "Fun Dip":                             "Sugar Shack",
    "Fizzy Lifting Drinks":                "Sugar Shack",
    "Everlasting Gobstopper":              "Secret Factory",
    "Lickable Wallpaper":                  "Secret Factory",
    "Wonka Gum":                           "Secret Factory",
    "Hair Toffee":                         "The Other Factory",
    "Kazookles":                           "The Other Factory",
}

# US/Canada state centroids (lat, lon) for haversine distance calculations
STATE_CENTROIDS = {
    "Alabama": (32.806671, -86.791130), "Alaska": (61.370716, -152.404419),
    "Arizona": (33.729759, -111.431221), "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564), "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371), "Delaware": (39.318523, -75.507141),
    "Florida": (27.766279, -81.686783), "Georgia": (33.040619, -83.643074),
    "Hawaii": (21.094318, -157.498337), "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137), "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526), "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067), "Louisiana": (31.169960, -91.867805),
    "Maine": (44.693947, -69.381927), "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106), "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192), "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368), "Montana": (46.921925, -110.454353),
    "Nebraska": (41.125370, -98.268082), "Nevada": (38.313515, -117.055374),
    "New Hampshire": (43.452492, -71.563896), "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482), "New York": (42.165726, -74.948051),
    "North Carolina": (35.630066, -79.806419), "North Dakota": (47.528912, -99.784012),
    "Ohio": (40.388783, -82.764915), "Oklahoma": (35.565342, -96.928917),
    "Oregon": (44.572021, -122.070938), "Pennsylvania": (40.590752, -77.209755),
    "Rhode Island": (41.680893, -71.511780), "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828), "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461), "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686), "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494), "West Virginia": (38.491226, -80.954453),
    "Wisconsin": (44.268543, -89.616508), "Wyoming": (42.755966, -107.302490),
    "District of Columbia": (38.897438, -77.026817),
    # Canada provinces
    "Ontario": (51.253775, -85.232212), "Quebec": (52.939916, -73.549136),
    "British Columbia": (53.726669, -127.647621), "Alberta": (55.001251, -114.998711),
    "Manitoba": (53.760860, -98.813873), "Saskatchewan": (52.939916, -106.450860),
    "Nova Scotia": (44.681988, -63.744311), "New Brunswick": (46.565422, -66.461914),
    "Newfoundland and Labrador": (53.135509, -57.660435),
    "Prince Edward Island": (46.510712, -63.416813),
    "Northwest Territories": (64.825516, -124.845905),
    "Nunavut": (70.299870, -83.107759), "Yukon": (64.282321, -135.000000),
}

# Synthetic lead time by Ship Mode (business-realistic, in days)
LEAD_TIME_BASE = {
    "Same Day":       1,
    "First Class":    2,
    "Second Class":   4,
    "Standard Class": 6,
}
LEAD_TIME_NOISE_STD = {
    "Same Day":       0.3,
    "First Class":    0.5,
    "Second Class":   0.8,
    "Standard Class": 1.2,
}


def load_raw(path: str = "data/Nassau_Candy_Distributor.csv") -> pd.DataFrame:
    """Load the raw CSV."""
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    - Parse dates (dayfirst=True)
    - Engineer synthetic Lead_Time from Ship Mode
    - Attach factory assignments + coordinates
    - Attach state centroid coordinates
    - Derive Profit_Margin, Profit_Per_Unit
    - Encode categoricals
    """
    df = df.copy()

    # ── 1. Date Parsing ───────────────────────────────────────────────────────
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=True)
    df["Order_Year"]  = df["Order Date"].dt.year
    df["Order_Month"] = df["Order Date"].dt.month
    df["Order_Quarter"] = df["Order Date"].dt.quarter
    df["Order_DayOfWeek"] = df["Order Date"].dt.dayofweek

    # ── 2. Synthetic Lead Time ────────────────────────────────────────────────
    # Ship dates in dataset are corrupted (extend to 2030).
    # We reconstruct a business-realistic lead time from Ship Mode.
    rng = np.random.default_rng(random_seed)
    lead_times = []
    for mode in df["Ship Mode"]:
        base  = LEAD_TIME_BASE.get(mode, 5)
        noise = rng.normal(0, LEAD_TIME_NOISE_STD.get(mode, 1.0))
        lt    = max(1, round(base + noise))
        lead_times.append(lt)
    df["Lead_Time"] = lead_times

    # ── 3. Factory Assignment ─────────────────────────────────────────────────
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY)
    df["Factory_Lat"] = df["Factory"].map(lambda f: FACTORY_COORDS[f]["lat"])
    df["Factory_Lon"] = df["Factory"].map(lambda f: FACTORY_COORDS[f]["lon"])

    # ── 4. State Centroid Coordinates ─────────────────────────────────────────
    df["State_Lat"] = df["State/Province"].map(
        lambda s: STATE_CENTROIDS.get(s, (39.5, -98.35))[0]
    )
    df["State_Lon"] = df["State/Province"].map(
        lambda s: STATE_CENTROIDS.get(s, (39.5, -98.35))[1]
    )

    # ── 5. Haversine Distance (factory → delivery state) ─────────────────────
    df["Distance_Miles"] = haversine_vectorized(
        df["Factory_Lat"].values, df["Factory_Lon"].values,
        df["State_Lat"].values,   df["State_Lon"].values
    )

    # ── 6. Profitability Features ─────────────────────────────────────────────
    df["Profit_Margin"]   = df["Gross Profit"] / df["Sales"]
    df["Profit_Per_Unit"] = df["Gross Profit"] / df["Units"]
    df["Revenue_Per_Unit"] = df["Sales"] / df["Units"]
    df["Cost_Per_Unit"]   = df["Cost"] / df["Units"]

    # ── 7. Encode Categoricals ────────────────────────────────────────────────
    df["Ship_Mode_Code"] = df["Ship Mode"].map({
        "Same Day": 1, "First Class": 2, "Second Class": 3, "Standard Class": 4
    })
    df["Region_Code"] = pd.Categorical(df["Region"]).codes
    df["Division_Code"] = pd.Categorical(df["Division"]).codes
    df["Factory_Code"] = pd.Categorical(df["Factory"]).codes

    # ── 8. Data Quality Flag ──────────────────────────────────────────────────
    # Sugar & Other divisions have very sparse data — flag for low confidence
    product_counts = df["Product Name"].value_counts()
    df["Data_Confidence"] = df["Product Name"].map(
        lambda p: "High" if product_counts.get(p, 0) >= 100 else "Low"
    )

    return df


def haversine_vectorized(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance in miles."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def haversine_scalar(lat1, lon1, lat2, lon2) -> float:
    """Scalar haversine distance in miles."""
    return float(haversine_vectorized(
        np.array([lat1]), np.array([lon1]),
        np.array([lat2]), np.array([lon2])
    )[0])


def get_clean_data(path: str = "data/Nassau_Candy_Distributor.csv") -> pd.DataFrame:
    """One-call convenience wrapper."""
    return clean(load_raw(path))


if __name__ == "__main__":
    df = get_clean_data()
    print(f"Clean dataset: {df.shape}")
    print(df[["Product Name", "Factory", "Lead_Time", "Distance_Miles",
              "Profit_Margin", "Data_Confidence"]].head(10).to_string())
    print("\nLead Time by Ship Mode:")
    print(df.groupby("Ship Mode")["Lead_Time"].describe().round(2))
