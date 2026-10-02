"""
Step 8 in Flow Chart: Explain Prediction (XAI) using SHAP.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/processed/groundnut_biomass_dataset_2000_2017.csv"
OUTPUT_DIR = "outputs"
GLOBAL_PLOT = os.path.join(OUTPUT_DIR, "xai_global_feature_importance.png")
LOCAL_PLOT = os.path.join(OUTPUT_DIR, "xai_local_district_explanation.png")
CSV_OUT = os.path.join(OUTPUT_DIR, "shap_detailed_analysis.csv")

FEATURES = [
    "mean_temperature_C", "mean_relative_humidity_pct", "mean_root_zone_soil_wetness",
    "annual_precipitation_mm", "mean_ndvi", "mean_evi", "ndvi_std", "evi_std", "valid_observations"
]

FEATURE_LABELS = {
    "mean_temperature_C": "Mean Temp (C)",
    "mean_relative_humidity_pct": "Relative Humidity (%)",
    "mean_root_zone_soil_wetness": "Root Zone Soil Wetness",
    "annual_precipitation_mm": "Annual Precipitation (mm)",
    "mean_ndvi": "Mean NDVI (Vegetation)",
    "mean_evi": "Mean EVI (Enhanced Veg)",
    "ndvi_std": "NDVI Std Dev",
    "evi_std": "EVI Std Dev",
    "valid_observations": "Valid Satellite Obs"
}

TARGET = "estimated_shell_biomass_kg_ha"


def run_xai_analysis(district_to_explain="Ananthapur", year_to_explain=2017):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    model = RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=2, n_jobs=-1)
    model.fit(X, y)

    print("=" * 80)
    print("         STEP 8: EXPLAINABLE AI (XAI) ANALYSIS VIA SHAP")
    print("=" * 80)
    print(f"Dataset Rows  : {len(X)}")
    print(f"Explaining    : Global Feature Contributions + Local District Case Study")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_val = float(base_val[0])

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_imp_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS[f] for f in FEATURES],
        "Feature_Key": FEATURES,
        "Mean_Abs_SHAP_Impact": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP_Impact", ascending=False)

    global_imp_df.to_csv(CSV_OUT, index=False)

    print("\n--- GLOBAL FEATURE IMPORTANCE (SHAP Impact on Shell Biomass) ---")
    print("+" + "-" * 55 + "+")
    print(f"| {'Feature Name':<32} | {'Mean |SHAP| (kg/ha)':<18} |")
    print("+" + "-" * 55 + "+")
    for _, r in global_imp_df.iterrows():
        print(f"| {r['Feature']:<32} | {r['Mean_Abs_SHAP_Impact']:<18.3f} |")
    print("+" + "-" * 55 + "+")

    plt.figure(figsize=(9, 5.5))
    y_pos = np.arange(len(global_imp_df))
    plt.barh(y_pos, global_imp_df["Mean_Abs_SHAP_Impact"], color="#2E7D32", edgecolor="black", alpha=0.85)
    plt.yticks(y_pos, global_imp_df["Feature"], fontsize=10)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |SHAP Value| (Impact on Biomass Yield in kg/ha)", fontsize=11)
    plt.title("XAI Global Feature Importance (TreeSHAP Analysis)", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(GLOBAL_PLOT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Global SHAP plot saved to: {GLOBAL_PLOT}")

    sample_mask = (df["Dist Name"] == district_to_explain) & (df["year"] == year_to_explain)
    if not sample_mask.any():
        sample_idx = 0
        actual_dist = df.iloc[0]["Dist Name"]
        actual_yr = df.iloc[0]["year"]
    else:
        sample_idx = df[sample_mask].index[0]
        actual_dist = district_to_explain
        actual_yr = year_to_explain

    sample_x = X.iloc[sample_idx]
    sample_shap = shap_values[sample_idx]
    actual_pred = model.predict(pd.DataFrame([sample_x], columns=FEATURES))[0]

    local_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS[f] for f in FEATURES],
        "Input_Value": sample_x.values,
        "SHAP_Contribution": sample_shap
    }).sort_values("SHAP_Contribution", key=abs, ascending=False)

    print(f"\n--- LOCAL XAI EXPLANATION: {actual_dist} ({actual_yr}) ---")
    print(f"Base Expected Biomass : {base_val:.2f} kg/ha")
    print(f"Model Predicted Yield : {actual_pred:.2f} kg/ha")
    print(f"Net Explained Shift   : {actual_pred - base_val:+.2f} kg/ha\n")

    print("+" + "-" * 68 + "+")
    print(f"| {'Feature':<28} | {'Input Value':<14} | {'SHAP Impact (kg/ha)':<20} |")
    print("+" + "-" * 68 + "+")
    for _, row in local_df.iterrows():
        impact_str = f"{row['SHAP_Contribution']:+.2f}"
        print(f"| {row['Feature']:<28} | {row['Input_Value']:<14.2f} | {impact_str:<20} |")
    print("+" + "-" * 68 + "+")

    plt.figure(figsize=(10, 6))
    bar_colors = ["#4CAF50" if v >= 0 else "#E53935" for v in local_df["SHAP_Contribution"]]
    y_pos = np.arange(len(local_df))
    plt.barh(y_pos, local_df["SHAP_Contribution"], color=bar_colors, edgecolor="black", alpha=0.85)
    plt.yticks(y_pos, [f"{f} = {val:.1f}" for f, val in zip(local_df["Feature"], local_df["Input_Value"])], fontsize=10)
    plt.axvline(0, color="black", linestyle="--", linewidth=1.2)
    plt.gca().invert_yaxis()
    plt.xlabel("SHAP Value (Contribution to Prediction in kg/ha)", fontsize=11)
    plt.title(f"XAI Local Explanation: {actual_dist} ({actual_yr})\n[Green = Increased Biomass, Red = Decreased Biomass]",
              fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(LOCAL_PLOT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Local XAI plot saved to: {LOCAL_PLOT}")
    print(f"[OK] Detailed CSV saved to: {CSV_OUT}")
    print("=" * 80)
    return global_imp_df, local_df


if __name__ == "__main__":
    run_xai_analysis()
