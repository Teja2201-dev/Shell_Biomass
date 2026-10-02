import pandas as pd
import requests
import time
from pathlib import Path

INPUT = Path("data/raw/icrisat_andhra_pradesh_groundnut_2000_2017.csv")
OUTPUT = Path("data/processed/nasa_all_districts_2000_2017.csv")

COORDS = {
    "Srikakulam": (18.30, 83.90),
    "Visakhapatnam": (17.69, 83.22),
    "East Godavari": (16.98, 82.24),
    "West Godavari": (16.71, 81.10),
    "Krishna": (16.18, 81.13),
    "Guntur": (16.31, 80.44),
    "S.P.S. Nellore": (14.44, 79.99),
    "Kurnool": (15.83, 78.04),
    "Ananthapur": (14.68, 77.60),
    "Kadapa YSR": (14.47, 78.82),
    "Chittoor": (13.22, 79.10),
}

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

rows = []

for district, (lat, lon) in COORDS.items():

    print("Downloading:", district)

    params = {
        "parameters": "T2M,RH2M,GWETROOT,PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": "20000101",
        "end": "20171231",
        "format": "JSON",
    }

    response = requests.get(URL, params=params, timeout=120)
    response.raise_for_status()

    data = response.json()["properties"]["parameter"]

    for year in range(2000, 2018):

        values = {
            "T2M": [],
            "RH2M": [],
            "GWETROOT": [],
            "PRECTOTCORR": [],
        }

        for date, temp in data["T2M"].items():

            if int(date[:4]) == year:

                values["T2M"].append(temp)
                values["RH2M"].append(data["RH2M"][date])
                values["GWETROOT"].append(data["GWETROOT"][date])
                values["PRECTOTCORR"].append(data["PRECTOTCORR"][date])

        rows.append({
            "Dist Name": district,
            "year": year,
            "mean_temperature_C": sum(values["T2M"]) / len(values["T2M"]),
            "mean_relative_humidity_pct": sum(values["RH2M"]) / len(values["RH2M"]),
            "mean_root_zone_soil_wetness": sum(values["GWETROOT"]) / len(values["GWETROOT"]),
            "annual_precipitation_mm": sum(values["PRECTOTCORR"]),
        })

    time.sleep(1)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)

print("\nDONE!")
print("Rows:", len(df))
print("Output:", OUTPUT)