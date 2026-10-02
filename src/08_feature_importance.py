
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


INPUT = "data/processed/groundnut_biomass_dataset_2000_2017.csv"

TARGET = "estimated_shell_biomass_kg_ha"

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


df = pd.read_csv(INPUT)

X = df[FEATURES]
y = df[TARGET]


# Train Random Forest on complete dataset
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1
)

rf.fit(X, y)


# Feature importance
importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)


print("\n==============================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("==============================")

print(importance.to_string(index=False))


# Save results
importance.to_csv(
    "outputs/random_forest_feature_importance.csv",
    index=False
)


# Plot
plt.figure(figsize=(9, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "outputs/random_forest_feature_importance.png",
    dpi=300
)

plt.show()