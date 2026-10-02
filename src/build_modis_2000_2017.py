"""Build yearly MOD13Q1 Collection 6.1 features for the study location.

The script uses the NASA ORNL DAAC TESViS location-subset REST service instead
of downloading complete MODIS granules. It stores the filtered point
observations used for aggregation and does not modify the local 2020 test
files or their summary.
"""

import csv
import json
import re
from datetime import date, timedelta
from pathlib import Path
from time import sleep

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OBSERVATIONS_PATH = OUTPUT_DIR / "modis_observations_2000_2017.csv"
YEARLY_PATH = OUTPUT_DIR / "modis_yearly_features_2000_2017.csv"

API_BASE_URL = "https://modis.ornl.gov/rst/api/v1"
PRODUCT = "MOD13Q1"
COLLECTION = "6.1"
API_PRODUCT = PRODUCT
LATITUDE = 14.47
LONGITUDE = 78.82
START_YEAR = 2000
END_YEAR = 2017
MAX_DATES_PER_REQUEST = 10
REQUEST_PAUSE_SECONDS = 0.5
USER_AGENT = "groundnut-biomass-project/1.0"
REQUEST_TIMEOUT_SECONDS = 30

NDVI_BAND = "250m_16_days_NDVI"
EVI_BAND = "250m_16_days_EVI"
QUALITY_BAND = "250m_16_days_VI_Quality"
SCALE_FACTOR = 0.0001
MODIS_FILL_VALUE = 65535
API_FILL_VALUE = -3000
QUALITY_FILL_VALUE = -1
MODIS_DATE_PATTERN = re.compile(r"^A(\d{4})(\d{3})$")

# No crop-season calendar is assumed. Pass a season explicitly in code or add
# a validated crop-calendar source before deriving a seasonal feature.
SEASON_START_MONTH: int | None = None
SEASON_END_MONTH: int | None = None


def create_session() -> requests.Session:
    """Create a retrying HTTP session for transient API failures."""
    retry_policy = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry_policy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    return session


def get_response(
    session: requests.Session,
    endpoint: str,
    params: dict[str, str] | None = None,
) -> requests.Response:
    """Fetch one API response and raise an actionable HTTP error."""
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as error:
        status = getattr(error.response, "status_code", None)
        raise RuntimeError(
            f"NASA ORNL request failed ({status or 'no HTTP status'}): {error}"
        ) from error
    return response


def get_json(
    session: requests.Session,
    endpoint: str,
    params: dict[str, str] | None = None,
) -> dict:
    """Fetch and decode one JSON response from the public NASA ORNL API."""
    response = get_response(session, endpoint, params)
    try:
        return response.json()
    except ValueError as error:
        raise RuntimeError(
            f"NASA ORNL returned non-JSON content for {response.url}"
        ) from error


def parse_modis_date(value: str) -> date:
    """Convert an AyyyyDDD MODIS date to a calendar date."""
    match = MODIS_DATE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid MODIS date: {value}")
    year, day_of_year = (int(part) for part in match.groups())
    return date(year, 1, 1) + timedelta(days=day_of_year - 1)


def get_dates(session: requests.Session) -> list[dict[str, str]]:
    """Get the available MOD13Q1 dates for the study coordinate."""
    payload = get_json(
        session,
        f"{API_PRODUCT}/dates",
        {"latitude": str(LATITUDE), "longitude": str(LONGITUDE)},
    )
    dates = [
        item
        for item in payload["dates"]
        if START_YEAR <= parse_modis_date(item["modis_date"]).year <= END_YEAR
    ]
    return sorted(dates, key=lambda item: item["calendar_date"])


def get_band_values(
    session: requests.Session, band: str, modis_dates: list[str]
) -> dict[str, int]:
    """Fetch one band in batches and return raw values keyed by MODIS date."""
    values: dict[str, int] = {}
    for start in range(0, len(modis_dates), MAX_DATES_PER_REQUEST):
        batch = modis_dates[start : start + MAX_DATES_PER_REQUEST]
        payload = get_json(
            session,
            f"{API_PRODUCT}/subset",
            {
                "latitude": str(LATITUDE),
                "longitude": str(LONGITUDE),
                "band": band,
                "startDate": batch[0],
                "endDate": batch[-1],
                "kmAboveBelow": "0",
                "kmLeftRight": "0",
            },
        )
        for item in payload["subset"]:
            data = item["data"]
            if len(data) != 1:
                raise ValueError(
                    f"Expected one point value for {band} {item['modis_date']}"
                )
            values[item["modis_date"]] = int(data[0])
        print(
            f"Retrieved {band}: {start + len(batch)}/{len(modis_dates)} dates"
        )
        sleep(REQUEST_PAUSE_SECONDS)
    if set(values) != set(modis_dates):
        missing = sorted(set(modis_dates) - set(values))
        raise ValueError(f"Missing {band} dates from API response: {missing}")
    return values


def is_valid_observation(ndvi: int, evi: int, quality: int) -> bool:
    """Apply the documented MODIS VI Quality and fill-value filters."""
    if ndvi in (MODIS_FILL_VALUE, API_FILL_VALUE):
        return False
    if evi in (MODIS_FILL_VALUE, API_FILL_VALUE):
        return False
    if quality == QUALITY_FILL_VALUE:
        return False
    modland_qa = quality & 0b11
    vi_usefulness = (quality >> 2) & 0b1111
    return modland_qa in (0, 1) and vi_usefulness <= 5


def test_one_date(
    session: requests.Session, item: dict[str, str]
) -> None:
    """Verify one historical 2000 date before starting the full retrieval."""
    modis_date = item["modis_date"]
    responses = {}
    for band in (NDVI_BAND, EVI_BAND, QUALITY_BAND):
        responses[band] = get_response(
            session,
            f"{API_PRODUCT}/subset",
            {
                "latitude": str(LATITUDE),
                "longitude": str(LONGITUDE),
                "band": band,
                "startDate": modis_date,
                "endDate": modis_date,
                "kmAboveBelow": "0",
                "kmLeftRight": "0",
            },
        )
    values = {}
    for band, response in responses.items():
        subset = response.json()["subset"]
        if len(subset) != 1 or len(subset[0]["data"]) != 1:
            raise ValueError(f"Unexpected one-date response for {band}")
        values[band] = subset[0]["data"][0]
    print(
        "One-date API test: "
        f"HTTP statuses={[response.status_code for response in responses.values()]}, "
        f"MODIS date={modis_date}, NDVI={values[NDVI_BAND]}, "
        f"EVI={values[EVI_BAND]}, VI Quality={values[QUALITY_BAND]}"
    )


def build_observations(
    session: requests.Session, dates: list[dict[str, str]]
) -> list[dict[str, object]]:
    """Fetch, filter, and return the observations used for aggregation."""
    modis_dates = [item["modis_date"] for item in dates]
    ndvi_values = get_band_values(session, NDVI_BAND, modis_dates)
    evi_values = get_band_values(session, EVI_BAND, modis_dates)
    quality_values = get_band_values(session, QUALITY_BAND, modis_dates)
    observations = []
    for item in dates:
        modis_date = item["modis_date"]
        ndvi = ndvi_values[modis_date]
        evi = evi_values[modis_date]
        quality = quality_values[modis_date]
        if not is_valid_observation(ndvi, evi, quality):
            continue
        observations.append(
            {
                "date": item["calendar_date"],
                "year": parse_modis_date(modis_date).year,
                "modis_date": modis_date,
                "ndvi": ndvi * SCALE_FACTOR,
                "evi": evi * SCALE_FACTOR,
                "vi_quality": quality,
                "modland_qa": quality & 0b11,
                "vi_usefulness": (quality >> 2) & 0b1111,
            }
        )
    return observations


def write_observations(observations: list[dict[str, object]]) -> None:
    """Write the filtered point observations used for aggregation."""
    with OBSERVATIONS_PATH.open("w", newline="", encoding="utf-8") as output:
        fieldnames = [
            "date",
            "year",
            "modis_date",
            "ndvi",
            "evi",
            "vi_quality",
            "modland_qa",
            "vi_usefulness",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(observations)


def mean(values: list[float]) -> float:
    """Return the arithmetic mean of a non-empty list."""
    return sum(values) / len(values)


def standard_deviation(values: list[float], average: float) -> float:
    """Return the population standard deviation for one year's observations."""
    return (sum((value - average) ** 2 for value in values) / len(values)) ** 0.5


def build_yearly_features(observations: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate filtered observations into one row per study year."""
    by_year: dict[int, list[dict[str, object]]] = {}
    for observation in observations:
        by_year.setdefault(int(observation["year"]), []).append(observation)
    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        year_observations = by_year.get(year, [])
        if not year_observations:
            rows.append(
                {
                    "year": year,
                    "mean_ndvi": "",
                    "mean_evi": "",
                    "ndvi_std": "",
                    "evi_std": "",
                    "valid_observations": 0,
                }
            )
            continue
        ndvi = [float(item["ndvi"]) for item in year_observations]
        evi = [float(item["evi"]) for item in year_observations]
        ndvi_mean = mean(ndvi)
        evi_mean = mean(evi)
        rows.append(
            {
                "year": year,
                "mean_ndvi": f"{ndvi_mean:.6f}",
                "mean_evi": f"{evi_mean:.6f}",
                "ndvi_std": f"{standard_deviation(ndvi, ndvi_mean):.6f}",
                "evi_std": f"{standard_deviation(evi, evi_mean):.6f}",
                "valid_observations": len(year_observations),
            }
        )
    return rows


def write_yearly_features(rows: list[dict[str, object]]) -> None:
    """Write the yearly feature table."""
    with YEARLY_PATH.open("w", newline="", encoding="utf-8") as output:
        fieldnames = [
            "year",
            "mean_ndvi",
            "mean_evi",
            "ndvi_std",
            "evi_std",
            "valid_observations",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Fetch the study-period observations and write both output tables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = create_session()
    dates = get_dates(session)
    historical_dates = [item for item in dates if parse_modis_date(item["modis_date"]).year == 2000]
    if not historical_dates:
        raise ValueError("The API returned no MOD13Q1 dates for the year 2000")
    test_one_date(session, historical_dates[0])
    observations = build_observations(session, dates)
    if not observations:
        raise ValueError("No valid MODIS observations were returned")
    observations.sort(key=lambda item: item["date"])
    write_observations(observations)
    write_yearly_features(build_yearly_features(observations))
    print(f"Observations obtained: {len(observations)}")
    print(f"First observation date: {observations[0]['date']}")
    print(f"Last observation date: {observations[-1]['date']}")
    print(f"Yearly rows: {END_YEAR - START_YEAR + 1}")
    print(f"Observation output: {OBSERVATIONS_PATH}")
    print(f"Yearly output: {YEARLY_PATH}")


if __name__ == "__main__":
    main()