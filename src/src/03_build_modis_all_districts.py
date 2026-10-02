import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

API = "https://modis.ornl.gov/rst/api/v1"
PRODUCT = "MOD13Q1"

START_YEAR = 2000
END_YEAR = 2017

OUTPUT = Path("data/processed/modis_all_districts_2000_2017.csv")

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

NDVI_BAND = "250m_16_days_NDVI"
EVI_BAND = "250m_16_days_EVI"

session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "User-Agent": "groundnut-biomass-project/1.0"
})


def get_dates(lat, lon):
    r = session.get(
        f"{API}/{PRODUCT}/dates",
        params={"latitude": lat, "longitude": lon},
        timeout=60
    )
    r.raise_for_status()

    payload = r.json()
    dates = payload.get("dates", payload) if isinstance(payload, dict) else payload

    result = []

    for item in dates:
        if isinstance(item, dict):
            modis_date = item.get("modis_date")
            calendar_date = item.get("calendar_date")
        else:
            modis_date = item
            calendar_date = None

        if not modis_date:
            continue

        year = int(modis_date[1:5])

        if START_YEAR <= year <= END_YEAR:
            if not calendar_date:
                doy = int(modis_date[5:])
                d = datetime(year, 1, 1) + timedelta(days=doy - 1)
                calendar_date = d.strftime("%Y-%m-%d")

            result.append((modis_date, calendar_date))

    return result


def get_subset(lat, lon, start_date, end_date):
    r = session.get(
        f"{API}/{PRODUCT}/subset",
        params={
            "latitude": lat,
            "longitude": lon,
            "startDate": start_date,
            "endDate": end_date,
            "kmAboveBelow": 0,
            "kmLeftRight": 0,
        },
        timeout=90
    )
    r.raise_for_status()
    return r.json().get("subset", [])


observations = []

for district, (lat, lon) in COORDS.items():

    print(f"\nDownloading: {district}")

    dates = get_dates(lat, lon)
    print("Available dates:", len(dates))

    for i in range(0, len(dates), 10):

        batch = dates[i:i + 10]

        try:
            subset = get_subset(
                lat,
                lon,
                batch[0][0],
                batch[-1][0]
            )

            values = {}

            for item in subset:
                modis_date = item.get("modis_date")
                band = item.get("band")
                data = item.get("data", [])

                if not modis_date or not data:
                    continue

                values.setdefault(modis_date, {})

                if band == NDVI_BAND:
                    values[modis_date]["ndvi"] = data[0]

                elif band == EVI_BAND:
                    values[modis_date]["evi"] = data[0]

            for modis_date, calendar_date in batch:

                v = values.get(modis_date, {})

                ndvi_raw = v.get("ndvi")
                evi_raw = v.get("evi")

                if ndvi_raw is None or evi_raw is None:
                    continue

                # MOD13Q1 scale factor = 0.0001
                ndvi = float(ndvi_raw) * 0.0001
                evi = float(evi_raw) * 0.0001

                # Remove fill/out-of-range values
                if not (-1.0 <= ndvi <= 1.0):
                    continue

                if not (-1.0 <= evi <= 1.0):
                    continue

                observations.append({
                    "Dist Name": district,
                    "date": calendar_date,
                    "year": int(calendar_date[:4]),
                    "ndvi": ndvi,
                    "evi": evi
                })

        except Exception as e:
            print("Batch failed:", e)

        if i % 50 == 0:
            print(
                "Progress:",
                min(i + 10, len(dates)),
                "/",
                len(dates)
            )

        time.sleep(0.2)


print("\nCreating yearly features...")

df = pd.DataFrame(observations)

if df.empty:
    raise RuntimeError("No MODIS observations obtained.")

yearly = (
    df.groupby(["Dist Name", "year"])
      .agg(
          mean_ndvi=("ndvi", "mean"),
          mean_evi=("evi", "mean"),
          ndvi_std=("ndvi", "std"),
          evi_std=("evi", "std"),
          valid_observations=("ndvi", "count")
      )
      .reset_index()
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
yearly.to_csv(OUTPUT, index=False)

print("\n==============================")
print("MODIS COMPLETE")
print("==============================")
print("Observations:", len(df))
print("Yearly rows:", len(yearly))
print("Districts:", yearly["Dist Name"].nunique())
print("Missing values:", yearly.isna().sum().sum())
print("Output:", OUTPUT)