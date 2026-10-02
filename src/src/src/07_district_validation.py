import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


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

districts = sorted(df["Dist Name"].unique())

rf_actual = []
rf_predicted = []

xgb_actual = []
xgb_predicted = []


print("\n==============================")
print("DISTRICT-WISE VALIDATION")
print("==============================")

print("Districts:", len(districts))


for test_district in districts:

    train = df[df["Dist Name"] != test_district]
    test = df[df["Dist Name"] == test_district]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]


    # -----------------------------
    # Random Forest
    # -----------------------------

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    rf_prediction = rf.predict(X_test)

    rf_actual.extend(y_test.tolist())
    rf_predicted.extend(rf_prediction.tolist())


    # -----------------------------
    # XGBoost
    # -----------------------------

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

    xgb.fit(X_train, y_train)

    xgb_prediction = xgb.predict(X_test)

    xgb_actual.extend(y_test.tolist())
    xgb_predicted.extend(xgb_prediction.tolist())


    print(
        f"Test district: {test_district:20s} "
        f"Rows: {len(test)}"
    )


# -----------------------------
# Overall metrics
# -----------------------------

rf_mae = mean_absolute_error(
    rf_actual,
    rf_predicted
)

rf_rmse = np.sqrt(
    mean_squared_error(
        rf_actual,
        rf_predicted
    )
)

rf_r2 = r2_score(
    rf_actual,
    rf_predicted
)


xgb_mae = mean_absolute_error(
    xgb_actual,
    xgb_predicted
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        xgb_actual,
        xgb_predicted
    )
)

xgb_r2 = r2_score(
    xgb_actual,
    xgb_predicted
)


print("\n==============================")
print("FINAL DISTRICT-WISE RESULTS")
print("==============================")


print("\nRandom Forest")
print("------------------------------")
print(f"MAE  : {rf_mae:.4f}")
print(f"RMSE : {rf_rmse:.4f}")
print(f"R²   : {rf_r2:.4f}")


print("\nXGBoost")
print("------------------------------")
print(f"MAE  : {xgb_mae:.4f}")
print(f"RMSE : {xgb_rmse:.4f}")
print(f"R²   : {xgb_r2:.4f}")