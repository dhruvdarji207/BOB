"""
train_model.py
--------------
Loads CarPrice_Assignment.csv, preprocesses data, trains multiple regression
models, evaluates them, and saves the best pipeline + metrics to disk.

Run:
    python train_model.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "CarPrice_Assignment.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "car_price_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

os.makedirs(MODELS_DIR, exist_ok=True)


# ── 1. Load & clean data ───────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalise manufacturer name from CarName
    df["manufacturer"] = (
        df["CarName"]
        .str.split()
        .str[0]
        .str.lower()
        .str.strip()
    )

    # Fix known typos
    typo_map = {
        "maxda":     "mazda",
        "toyouta":   "toyota",
        "porcshce":  "porsche",
        "vokswagen": "volkswagen",
        "vw":        "volkswagen",
    }
    df["manufacturer"] = df["manufacturer"].replace(typo_map)

    # Drop non-predictive columns
    df.drop(columns=["car_ID", "CarName"], inplace=True)

    return df


# ── 2. Feature definitions ─────────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "symboling", "wheelbase", "carlength", "carwidth", "carheight",
    "curbweight", "enginesize", "boreratio", "stroke", "compressionratio",
    "horsepower", "peakrpm", "citympg", "highwaympg",
]

CATEGORICAL_FEATURES = [
    "manufacturer", "fueltype", "aspiration", "doornumber",
    "carbody", "drivewheel", "enginelocation", "enginetype",
    "cylindernumber", "fuelsystem",
]

TARGET = "price"


# ── 3. Build preprocessing pipeline ───────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline,  NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor


# ── 4. Define candidate models ─────────────────────────────────────────────
def get_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=10.0),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
        "Extra Trees":       ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=200, random_state=42),
    }


# ── 5. Evaluate a single pipeline ─────────────────────────────────────────
def evaluate_model(pipeline, X_train, X_test, y_train, y_test) -> dict:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    # 5-fold CV on training data
    cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")

    return {
        "mae":     round(float(mae),  2),
        "mse":     round(float(mse),  2),
        "rmse":    round(float(rmse), 2),
        "r2":      round(float(r2),   4),
        "cv_r2_mean": round(float(cv_r2.mean()), 4),
        "cv_r2_std":  round(float(cv_r2.std()),  4),
    }


# ── 6. Main training routine ───────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  Car Price Prediction - Model Training")
    print("=" * 60)

    # Load data
    print(f"\n[1] Loading dataset from: {DATA_PATH}")
    df = load_and_clean(DATA_PATH)
    print(f"    Records : {len(df)}")
    print(f"    Features: {len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)}")
    print(f"    Target  : {TARGET}")

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[2] Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    preprocessor = build_preprocessor()
    models = get_models()

    # Train & evaluate all models
    print("\n[3] Training and evaluating models ...\n")
    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipeline
        print(
            f"  {name:<26}  R²={metrics['r2']:.4f}  "
            f"MAE={metrics['mae']:>9,.2f}  RMSE={metrics['rmse']:>9,.2f}  "
            f"CV-R²={metrics['cv_r2_mean']:.4f}±{metrics['cv_r2_std']:.4f}"
        )

    # Select best model by test R²
    best_name = max(results, key=lambda n: results[n]["r2"])
    best_metrics = results[best_name]
    print(f"\n[4] Best model: {best_name}  (R²={best_metrics['r2']:.4f})")

    best_pipeline = fitted_pipelines[best_name]

    # Collect actual vs predicted for the test set (for frontend chart)
    y_pred_best = best_pipeline.predict(X_test)
    actual_vs_predicted = [
        {"actual": round(float(a), 2), "predicted": round(float(p), 2)}
        for a, p in zip(y_test.tolist(), y_pred_best.tolist())
    ]

    # Feature importance (if model supports it)
    feature_importance = []
    inner_model = best_pipeline.named_steps["model"]
    if hasattr(inner_model, "feature_importances_"):
        preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
        ohe = preprocessor_fitted.named_transformers_["cat"].named_steps["encoder"]
        cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        all_feature_names = NUMERICAL_FEATURES + cat_feature_names

        importances = inner_model.feature_importances_
        # Summarise by original feature (for OHE columns, sum their importances)
        importance_map = {}
        for fname, imp in zip(all_feature_names, importances):
            # OHE columns look like "manufacturer_toyota"
            original = fname.split("_")[0] if fname not in NUMERICAL_FEATURES else fname
            importance_map[original] = importance_map.get(original, 0) + float(imp)

        # Sort descending
        sorted_imp = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)
        feature_importance = [
            {"feature": k, "importance": round(v, 6)}
            for k, v in sorted_imp[:20]
        ]

    # Price distribution from full dataset
    df_full = load_and_clean(DATA_PATH)
    price_bins = pd.cut(df_full[TARGET], bins=10)
    price_dist = []
    for interval, count in price_bins.value_counts(sort=False).items():
        price_dist.append({
            "range": f"${interval.left:,.0f}–${interval.right:,.0f}",
            "count": int(count),
        })

    # Unique values for form dropdowns (from full dataset)
    dropdown_values = {}
    for col in CATEGORICAL_FEATURES:
        dropdown_values[col] = sorted(df_full[col].dropna().unique().tolist())

    # Numerical ranges
    numerical_ranges = {}
    for col in NUMERICAL_FEATURES:
        numerical_ranges[col] = {
            "min": round(float(df_full[col].min()), 2),
            "max": round(float(df_full[col].max()), 2),
            "mean": round(float(df_full[col].mean()), 2),
        }

    # Dataset stats
    dataset_info = {
        "records": int(len(df_full)),
        "features": len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES),
        "numerical_features": len(NUMERICAL_FEATURES),
        "categorical_features": len(CATEGORICAL_FEATURES),
        "target": TARGET,
        "price_min": round(float(df_full[TARGET].min()), 2),
        "price_max": round(float(df_full[TARGET].max()), 2),
        "price_mean": round(float(df_full[TARGET].mean()), 2),
        "price_median": round(float(df_full[TARGET].median()), 2),
    }

    # Assemble metrics JSON
    metrics_payload = {
        "best_model": best_name,
        "best_metrics": best_metrics,
        "all_models": results,
        "actual_vs_predicted": actual_vs_predicted,
        "feature_importance": feature_importance,
        "price_distribution": price_dist,
        "dropdown_values": dropdown_values,
        "numerical_ranges": numerical_ranges,
        "dataset_info": dataset_info,
    }

    # Save
    print(f"\n[5] Saving model  -> {MODEL_PATH}")
    joblib.dump(best_pipeline, MODEL_PATH)

    print(f"[5] Saving metrics -> {METRICS_PATH}")
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\n[OK] Training complete.")
    print(f"  Model : {best_name}")
    print(f"  R2    : {best_metrics['r2']}")
    print(f"  MAE   : ${best_metrics['mae']:,.2f}")
    print(f"  RMSE  : ${best_metrics['rmse']:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    train()
