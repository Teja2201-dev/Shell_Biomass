"""
Step 6 in Flow Chart: Improved / Hybrid Model for Groundnut Shell Biomass Estimation.
Builds an optimized Hybrid Ensemble combining Random Forest, XGBoost, and ExtraTrees Regressors.
Saves the trained model and artifacts to models/.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/processed/groundnut_biomass_dataset_2000_2017.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_biomass_model.pkl")

FEATURES = [
    "mean_temperature_C",
    "mean_relative_humidity_pct",
    "mean_root_zone_soil_wetness",
    "annual_precipitation_mm",
    "mean_ndvi",
    "mean_evi",
    "ndvi_std",
    "evi_std",
    "valid_observations"
]

TARGET = "estimated_shell_biomass_kg_ha"


def train_improved_hybrid_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("=" * 60)
    print("   STEP 6: TRAINING IMPROVED / HYBRID MODEL")
    print("=" * 60)
    print(f"Dataset Total Samples : {len(df)}")
    print(f"Training Samples       : {len(X_train)}")
    print(f"Testing Samples        : {len(X_test)}")
    print(f"Number of Features     : {len(FEATURES)}")

    # 1. Base Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )

    # 2. Base XGBoost
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    # 3. Variance-Reducing ExtraTrees Regressor
    et = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # 4. Improved Hybrid Model: Weighted Voting Ensemble
    hybrid_model = VotingRegressor(
        estimators=[
            ('random_forest', rf),
            ('xgboost', xgb),
            ('extra_trees', et)
        ],
        weights=[0.25, 0.15, 0.60]
    )

    print("\nFitting Hybrid Ensemble Model...")
    hybrid_model.fit(X_train, y_train)

    # Predictions
    train_preds = hybrid_model.predict(X_train)
    test_preds = hybrid_model.predict(X_test)

    # Metrics
    train_mae = mean_absolute_error(y_train, train_preds)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_r2 = r2_score(y_train, train_preds)

    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)
    mape = np.mean(np.abs((y_test - test_preds) / (y_test + 1e-6))) * 100

    print("\n" + "-" * 55)
    print("        HYBRID MODEL PERFORMANCE METRICS")
    print("-" * 55)
    print(f"{'Metric':<20} | {'Train Split':<14} | {'Test Split':<14}")
    print("-" * 55)
    print(f"{'MAE (kg/ha)':<20} | {train_mae:<14.4f} | {test_mae:<14.4f}")
    print(f"{'RMSE (kg/ha)':<20} | {train_rmse:<14.4f} | {test_rmse:<14.4f}")
    print(f"{'R-squared (R²)':<20} | {train_r2:<14.4f} | {test_r2:<14.4f}")
    print(f"{'MAPE (%)':<20} | {'-':<14} | {mape:<14.2f}%")
    print("-" * 55)

    # Also fit on full dataset for live deployment
    print("\nFitting deployment model on full dataset...")
    deploy_model = VotingRegressor(
        estimators=[
            ('random_forest', RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=-1)),
            ('xgboost', XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1)),
            ('extra_trees', ExtraTreesRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1))
        ],
        weights=[0.25, 0.15, 0.60]
    )
    deploy_model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(deploy_model, f)

    print(f"Hybrid model saved successfully to: {MODEL_PATH}")
    print("=" * 60)
    return hybrid_model, test_preds, (test_mae, test_rmse, test_r2)


if __name__ == "__main__":
    train_improved_hybrid_model()
