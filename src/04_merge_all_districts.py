import pandas as pd
import os

ICRISAT = "data/raw/icrisat_andhra_pradesh_groundnut_2000_2017.csv"
NASA = "data/processed/nasa_all_districts_2000_2017.csv"
MODIS = "data/processed/modis_all_districts_2000_2017.csv"

OUTPUT = "data/processed/final_multisource_dataset_2000_2017.csv"


# -----------------------------
# Read datasets
# -----------------------------

icrisat = pd.read_csv(ICRISAT)
nasa = pd.read_csv(NASA)
modis = pd.read_csv(MODIS)


# -----------------------------
# Standardize year
# -----------------------------

icrisat["year"] = icrisat["Year"].astype(int)
nasa["year"] = nasa["year"].astype(int)
modis["year"] = modis["year"].astype(int)


# -----------------------------
# Standardize district name
# -----------------------------

icrisat["Dist Name"] = icrisat["Dist Name"].astype(str)
nasa["Dist Name"] = nasa["Dist Name"].astype(str)
modis["Dist Name"] = modis["Dist Name"].astype(str)


# -----------------------------
# Select ICRISAT columns
# -----------------------------

icrisat = icrisat[
    [
        "Dist Code",
        "year",
        "State Code",
        "State Name",
        "Dist Name",
        "GROUNDNUT AREA (1000 ha)",
        "GROUNDNUT PRODUCTION (1000 tons)",
        "GROUNDNUT YIELD (Kg per ha)"
    ]
]


# -----------------------------
# Merge ICRISAT + NASA
# -----------------------------

df = pd.merge(
    icrisat,
    nasa,
    on=["Dist Name", "year"],
    how="inner"
)


# -----------------------------
# Merge MODIS
# -----------------------------

df = pd.merge(
    df,
    modis,
    on=["Dist Name", "year"],
    how="inner"
)


# -----------------------------
# Rename columns
# -----------------------------

df = df.rename(
    columns={
        "GROUNDNUT AREA (1000 ha)": "Groundnut Area",
        "GROUNDNUT PRODUCTION (1000 tons)": "Groundnut Production",
        "GROUNDNUT YIELD (Kg per ha)": "Groundnut Yield"
    }
)


# -----------------------------
# Sort
# -----------------------------

df = df.sort_values(
    ["Dist Name", "year"]
).reset_index(drop=True)


# -----------------------------
# Save
# -----------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)


# -----------------------------
# Check
# -----------------------------

print("\n==============================")
print("FINAL MULTISOURCE DATASET")
print("==============================")

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Districts:", df["Dist Name"].nunique())
print("Years:", df["year"].min(), "-", df["year"].max())

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicates:")
print(
    df.duplicated(
        ["Dist Name", "year"]
    ).sum()
)

print("\nOutput:")
print(OUTPUT)