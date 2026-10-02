import pandas as pd

INPUT = "data/processed/final_multisource_dataset_2000_2017.csv"
OUTPUT = "data/processed/groundnut_biomass_dataset_2000_2017.csv"

df = pd.read_csv(INPUT)

# Literature-based derived shell biomass target
# 20% of groundnut pod yield
df["estimated_shell_biomass_kg_ha"] = (
    df["Groundnut Yield"] * 0.20
)

df.to_csv(OUTPUT, index=False)

print("\n==============================")
print("BIOMASS TARGET CREATED")
print("==============================")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nTarget:")
print(df["estimated_shell_biomass_kg_ha"].describe())

print("\nOutput:")
print(OUTPUT)