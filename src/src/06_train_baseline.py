import pandas as pd

from sklearn.model_selection import train_test_split
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


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv(INPUT)

X = df[FEATURES]
y = df[TARGET]


# -----------------------------
# Train / Test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# -----------------------------
# Random Forest
# -----------------------------

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)


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

xgb_pred = xgb.predict(X_test)


# -----------------------------
# Evaluation
# -----------------------------

def evaluate(name, y_true, prediction):

    mae = mean_absolute_error(
        y_true,
        prediction
    )

    rmse = mean_squared_error(
        y_true,
        prediction
    ) ** 0.5

    r2 = r2_score(
        y_true,
        prediction
    )

    print(f"\n{name}")
    print("-" * 30)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


print("\n==============================")
print("BASELINE MODEL RESULTS")
print("==============================")

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))
print("Features     :", len(FEATURES))

evaluate(
    "Random Forest",
    y_test,
    rf_pred
)

evaluate(
    "XGBoost",
    y_test,
    xgb_pred
)