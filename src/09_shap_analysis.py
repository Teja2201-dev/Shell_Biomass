import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/processed/groundnut_biomass_dataset_2000_2017.csv"

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

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df[TARGET]

print("Training Random Forest...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

print("Calculating SHAP...")

# Explain only a small sample for speed
X_sample = X.sample(
    n=min(50, len(X)),
    random_state=42
)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

importance = np.abs(shap_values).mean(axis=0)

result = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": importance
}).sort_values(
    "mean_abs_shap",
    ascending=False
)

print("\nSHAP FEATURE IMPORTANCE")
print(result.to_string(index=False))

result.to_csv(
    "outputs/shap_feature_importance.csv",
    index=False
)

plt.figure(figsize=(8, 6))
shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.tight_layout()
plt.savefig(
    "outputs/shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSHAP ANALYSIS COMPLETE")
print("Saved: outputs/shap_feature_importance.csv")
print("Saved: outputs/shap_summary_plot.png")