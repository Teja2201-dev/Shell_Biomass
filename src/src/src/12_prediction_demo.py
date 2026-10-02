import pandas as pd
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

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

print("\nGROUNDNUT SHELL BIOMASS PREDICTION")
print("-----------------------------------")

temperature = float(input("Mean Temperature (C): "))
humidity = float(input("Relative Humidity (%): "))
soil = float(input("Root Zone Soil Wetness: "))
rainfall = float(input("Annual Precipitation (mm): "))
ndvi = float(input("Mean NDVI: "))
evi = float(input("Mean EVI: "))
ndvi_std = float(input("NDVI Std: "))
evi_std = float(input("EVI Std: "))
valid_obs = float(input("Valid Observations: "))

input_data = pd.DataFrame([[
    temperature,
    humidity,
    soil,
    rainfall,
    ndvi,
    evi,
    ndvi_std,
    evi_std,
    valid_obs
]], columns=FEATURES)

prediction = model.predict(input_data)[0]

print("\n===================================")
print(f"Estimated Shell Biomass: {prediction:.2f} kg/ha")
print("===================================")