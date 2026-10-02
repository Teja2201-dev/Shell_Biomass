import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

DATA_PATH = "data/processed/groundnut_biomass_dataset_2000_2017.csv"
TARGET = "estimated_shell_biomass_kg_ha"

df = pd.read_csv(DATA_PATH)

feature_sets = {
    "Weather Only": [
        "mean_temperature_C",
        "mean_relative_humidity_pct",
        "mean_root_zone_soil_wetness",
        "annual_precipitation_mm"
    ],

    "Weather + NDVI/EVI": [
        "mean_temperature_C",
        "mean_relative_humidity_pct",
        "mean_root_zone_soil_wetness",
        "annual_precipitation_mm",
        "mean_ndvi",
        "mean_evi"
    ],

    "Weather + Vegetation Variability": [
        "mean_temperature_C",
        "mean_relative_humidity_pct",
        "mean_root_zone_soil_wetness",
        "annual_precipitation_mm",
        "mean_ndvi",
        "mean_evi",
        "ndvi_std",
        "evi_std"
    ],

    "All Features": [
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
}

results = []

for name, features in feature_sets.items():

    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Feature Set": name,
        "Features": len(features),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

result_df = pd.DataFrame(results)

print("\n==============================")
print("ABLATION STUDY RESULTS")
print("==============================")
print(result_df.to_string(index=False))

result_df.to_csv(
    "outputs/ablation_study_results.csv",
    index=False
)

print("\nABLATION STUDY COMPLETE")
print("Saved: outputs/ablation_study_results.csv")