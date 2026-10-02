"""Calculate quality-filtered mean NDVI and EVI from a MODIS HDF4 product."""

import csv
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from pyhdf.SD import SD, SDC


ROOT_DIR = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT_DIR / "data" / "processed" / "modis_summary.csv"
MODIS_DIR = ROOT_DIR / "data" / "raw"
NDVI_DATASET = "250m 16 days NDVI"
EVI_DATASET = "250m 16 days EVI"
VI_QUALITY_DATASET = "250m 16 days VI Quality"
NDVI_SCALE_FACTOR = 0.0001
EVI_SCALE_FACTOR = 0.0001
NDVI_FILL_VALUE = 65535
EVI_FILL_VALUE = 65535
MODIS_DATE_PATTERN = re.compile(r"\.A(\d{4})(\d{3})\.")


def extract_summary(path: Path) -> dict[str, object]:
    """Extract the filtered NDVI and EVI summary from one MODIS HDF file."""
    hdf_file = SD(str(path), SDC.READ)
    try:
        ndvi = hdf_file.select(NDVI_DATASET).get()
        evi = hdf_file.select(EVI_DATASET).get()
        vi_quality = hdf_file.select(VI_QUALITY_DATASET).get()
    finally:
        hdf_file.end()

    if ndvi.shape != evi.shape or ndvi.shape != vi_quality.shape:
        raise ValueError(
            f"Dataset shapes do not match in {path.name}: "
            f"NDVI {ndvi.shape}, EVI {evi.shape}, "
            f"VI Quality {vi_quality.shape}"
        )

    modland_qa = vi_quality & 0b11
    vi_usefulness = (vi_quality >> 2) & 0b1111
    valid_pixels = (
        (ndvi != NDVI_FILL_VALUE)
        & (evi != EVI_FILL_VALUE)
        & np.isin(modland_qa, (0, 1))
        & (vi_usefulness <= 5)
    )
    scaled_ndvi = ndvi[valid_pixels].astype(np.float64) * NDVI_SCALE_FACTOR
    scaled_evi = evi[valid_pixels].astype(np.float64) * EVI_SCALE_FACTOR
    if scaled_ndvi.size == 0:
        raise ValueError(f"No valid pixels remain after filtering in {path.name}")
    return {
        "date": extract_date(path),
        "mean_ndvi": float(np.mean(scaled_ndvi)),
        "mean_evi": float(np.mean(scaled_evi)),
        "valid_pixels": int(valid_pixels.sum()),
    }


def extract_date(path: Path) -> date:
    """Extract and convert the MODIS AyyyyDDD date from a filename."""
    match = MODIS_DATE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not find AyyyyDDD date in {path.name}")
    year, day_of_year = (int(value) for value in match.groups())
    return date(year, 1, 1) + timedelta(days=day_of_year - 1)


def main() -> None:
    """Process all MOD13Q1 HDF files and write date-sorted summaries."""
    summaries = [
        extract_summary(path) for path in sorted(MODIS_DIR.glob("MOD13Q1*.hdf"))
    ]
    if not summaries:
        raise FileNotFoundError(f"No MOD13Q1 HDF files found in {MODIS_DIR}")
    summaries.sort(key=lambda summary: summary["date"])

    for summary in summaries:
        print(f"Date: {summary['date']}")
        print(f"Valid pixels: {summary['valid_pixels']}")
        print(f"Mean NDVI: {summary['mean_ndvi']:.6f}")
        print(f"Mean EVI: {summary['mean_evi']:.6f}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["date", "mean_ndvi", "mean_evi", "valid_pixels"],
        )
        writer.writeheader()
        writer.writerows(
            {
                "date": summary["date"].isoformat(),
                "mean_ndvi": f"{summary['mean_ndvi']:.6f}",
                "mean_evi": f"{summary['mean_evi']:.6f}",
                "valid_pixels": summary["valid_pixels"],
            }
            for summary in summaries
        )


if __name__ == "__main__":
    main()