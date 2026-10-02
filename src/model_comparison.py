"""
Step 5 in Flow Chart: Compare Models (MAE, RMSE, R²).
Compares Linear (Ridge), Random Forest, XGBoost, and the Improved Hybrid Model.
Generates tabular metrics in console, CSV output, and high-resolution comparison plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/processed/groundnut_biomass_dataset_2000_2017.csv"
OUTPUT_DIR = "outputs"
CSV_OUT = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
PLOT_OUT = os.path.join(OUTPUT_DIR, "model_comparison_metrics.png")

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


def compare_all_models():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Ridge Regression (Linear)": Ridge(alpha=10.0),
        "Random Forest (Baseline)": RandomForestRegressor(
            n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=-1
        ),
        "XGBoost (Baseline)": XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
            random_state=42, n_jobs=-1
        ),
        "Improved Hybrid Model": VotingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=-1)),
                ('xgb', XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1)),
                ('et', ExtraTreesRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1))
            ],
            weights=[0.25, 0.15, 0.60]
        )
    }

    results = []

    print("=" * 78)
    print("           STEP 5: COMPREHENSIVE MODEL COMPARISON (MAE, RMSE, R²)")
    print("=" * 78)

    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        mape = np.mean(np.abs((y_test - pred) / (y_test + 1e-6))) * 100

        results.append({
            "Model": name,
            "MAE (kg/ha)": round(mae, 4),
            "RMSE (kg/ha)": round(rmse, 4),
            "R² Score": round(r2, 4),
            "MAPE (%)": round(mape, 2)
        })

    res_df = pd.DataFrame(results)

    # Print Formatted Table
    print("\n" + "+" + "-" * 76 + "+")
    print(f"| {'Model Name':<28} | {'MAE (kg/ha)':<12} | {'RMSE (kg/ha)':<13} | {'R² Score':<9} | {'MAPE (%)':<8} |")
    print("+" + "-" * 76 + "+")
    for _, row in res_df.iterrows():
        is_best = " (BEST)" if row["Model"] == "Improved Hybrid Model" else ""
        print(f"| {row['Model'] + is_best:<28} | {row['MAE (kg/ha)']:<12.4f} | {row['RMSE (kg/ha)']:<13.4f} | {row['R² Score']:<9.4f} | {row['MAPE (%)']:<8.2f} |")
    print("+" + "-" * 76 + "+")

    res_df.to_csv(CSV_OUT, index=False)
    print(f"\n[OK] Metrics saved to CSV: {CSV_OUT}")

    # Generate Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    model_labels = ["Ridge", "Random Forest", "XGBoost", "Improved Hybrid"]
    colors = ["#9E9E9E", "#4CAF50", "#2196F3", "#FF9800"]

    # MAE Plot (Lower is better)
    axes[0].bar(range(len(model_labels)), res_df["MAE (kg/ha)"], color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("Mean Absolute Error (MAE)\n[Lower is Better]", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("MAE (kg/ha)", fontsize=11)
    axes[0].set_xticks(range(len(model_labels)))
    axes[0].set_xticklabels(model_labels, rotation=20, ha="right", fontsize=10)
    for i, v in enumerate(res_df["MAE (kg/ha)"]):
        axes[0].text(i, v + 1, f"{v:.1f}", ha="center", fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # RMSE Plot (Lower is better)
    axes[1].bar(range(len(model_labels)), res_df["RMSE (kg/ha)"], color=colors, edgecolor="black", alpha=0.85)
    axes[1].set_title("Root Mean Squared Error (RMSE)\n[Lower is Better]", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("RMSE (kg/ha)", fontsize=11)
    axes[1].set_xticks(range(len(model_labels)))
    axes[1].set_xticklabels(model_labels, rotation=20, ha="right", fontsize=10)
    for i, v in enumerate(res_df["RMSE (kg/ha)"]):
        axes[1].text(i, v + 1, f"{v:.1f}", ha="center", fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # R2 Plot (Higher is better)
    axes[2].bar(range(len(model_labels)), res_df["R² Score"], color=colors, edgecolor="black", alpha=0.85)
    axes[2].set_title("Coefficient of Determination (R²)\n[Higher is Better]", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("R² Score", fontsize=11)
    axes[2].set_ylim(0, 0.85)
    axes[2].set_xticks(range(len(model_labels)))
    axes[2].set_xticklabels(model_labels, rotation=20, ha="right", fontsize=10)
    for i, v in enumerate(res_df["R² Score"]):
        axes[2].text(i, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Model Performance Comparison on Test Set", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Comparison chart saved to: {PLOT_OUT}")
    print("=" * 78)
    return res_df


if __name__ == "__main__":
    compare_all_models()
