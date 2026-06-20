"""
train_model.py
Nassau Candy Distributor — ML Model Training Pipeline
Target: Lead_Time (synthetic, business-realistic)
Models: Linear Regression, Random Forest, Gradient Boosting, XGBoost
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model   import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. Skipping XGBRegressor.")

from preprocessing      import get_clean_data
from feature_engineering import get_ml_ready, ML_FEATURES, ML_TARGET

warnings.filterwarnings("ignore")

MODEL_DIR  = "models"
RESULTS_PATH = os.path.join(MODEL_DIR, "model_results.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH     = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_IMP_PATH = os.path.join(MODEL_DIR, "feature_importance.csv")

os.makedirs(MODEL_DIR, exist_ok=True)


def evaluate(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    print(f"  {name:<30}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2}


def get_models() -> dict:
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression())
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )
    return models


def train_and_evaluate(test_size: float = 0.2, random_state: int = 42):
    """Full training pipeline. Returns results DataFrame and best model."""
    print("=" * 60)
    print("  Nassau Candy — ML Training Pipeline")
    print("=" * 60)

    # ── Load & prepare ────────────────────────────────────────────────────────
    df = get_clean_data()
    X, y = get_ml_ready(df)
    print(f"\n  Dataset: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  Target : {ML_TARGET}  |  Range: {y.min()}–{y.max()} days")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # ── Train & evaluate ──────────────────────────────────────────────────────
    models  = get_models()
    results = []
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        res = evaluate(model, X_test, y_test, name)
        results.append(res)

    results_df = pd.DataFrame(results).sort_values("RMSE")
    best_name  = results_df.iloc[0]["Model"]
    best_model = trained[best_name]

    print(f"\n  ✅ Best Model: {best_name}  (RMSE={results_df.iloc[0]['RMSE']:.4f})")

    # ── Cross-validation on best model ───────────────────────────────────────
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="r2")
    print(f"  5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    results_df.to_csv(RESULTS_PATH, index=False)

    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "name": best_name, "features": ML_FEATURES}, f)
    print(f"  Saved: {BEST_MODEL_PATH}")

    # Scaler for raw input (used in scenario simulator)
    scaler = StandardScaler()
    scaler.fit(X_train)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # ── Feature Importance ────────────────────────────────────────────────────
    fi_df = extract_feature_importance(best_model, best_name, ML_FEATURES)
    if fi_df is not None:
        fi_df.to_csv(FEATURE_IMP_PATH, index=False)
        print(f"\n  Top 10 Features ({best_name}):")
        print(fi_df.head(10).to_string(index=False))

    return results_df, best_model, best_name


def extract_feature_importance(model, model_name: str, features: list) -> pd.DataFrame | None:
    """Extract feature importance where available."""
    if "Random Forest" in model_name or "Gradient Boosting" in model_name:
        imp = model.feature_importances_
    elif "XGBoost" in model_name:
        imp = model.feature_importances_
    elif "Linear Regression" in model_name:
        # Use Pipeline's inner model
        inner = model.named_steps.get("model", model)
        imp   = np.abs(inner.coef_)
    else:
        return None

    fi = pd.DataFrame({"Feature": features, "Importance": imp})
    fi = fi.sort_values("Importance", ascending=False).reset_index(drop=True)
    fi["Importance_Pct"] = (fi["Importance"] / fi["Importance"].sum() * 100).round(2)
    return fi


def load_best_model() -> tuple:
    """Load saved best model. Returns (model, model_name, features)."""
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run train_model.py first:\n  python train_model.py"
        )
    with open(BEST_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["name"], bundle["features"]


def predict_lead_time(model, feature_row: dict) -> float:
    """Predict lead time for a single feature row (dict keyed by ML_FEATURES)."""
    X = pd.DataFrame([feature_row])[ML_FEATURES]
    return float(model.predict(X)[0])


if __name__ == "__main__":
    results, best_model, best_name = train_and_evaluate()
    print("\n── All Model Results ──")
    print(results.to_string(index=False))
