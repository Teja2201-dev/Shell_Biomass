"""Merge the downloaded ICRISAT Groundnut area, production, and yield data."""

import csv
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "icrisat_groundnut_kadapa_2000_2017.csv"

KEY_COLUMNS = ["Dist Code", "Year", "State Code", "State Name", "Dist Name"]
OUTPUT_COLUMNS = KEY_COLUMNS + [
    "Groundnut Area",
    "Groundnut Production",
    "Groundnut Yield",
]

INPUTS = {
    "Groundnut Area": (
        RAW_DIR / "ICRISAT-District Level Data (1).csv",
        "GROUNDNUT AREA (1000 ha)",
    ),
    "Groundnut Production": (
        RAW_DIR / "ICRISAT-District Level Data (2).csv",
        "GROUNDNUT PRODUCTION (1000 tons)",
    ),
    "Groundnut Yield": (
        RAW_DIR / "ICRISAT-District Level Data.csv",
        "GROUNDNUT YIELD (Kg per ha)",
    ),
}


def read_measure_file(path: Path, source_column: str) -> dict[tuple[str, ...], dict[str, str]]:
    """Read one measure file and index its rows by the shared ICRISAT keys."""
    with path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        expected_columns = KEY_COLUMNS + [source_column]
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"Unexpected columns in {path.name}: {reader.fieldnames!r}"
            )

        indexed_rows = {}
        for row in reader:
            key = tuple(row[column] for column in KEY_COLUMNS)
            if key in indexed_rows:
                raise ValueError(f"Duplicate key in {path.name}: {key!r}")
            if any(row[column] == "" for column in expected_columns):
                raise ValueError(f"Missing value in {path.name}: {key!r}")
            indexed_rows[key] = {
                column: row[column] for column in KEY_COLUMNS
            }
            indexed_rows[key][source_column] = row[source_column]
        return indexed_rows


def merge_data() -> list[dict[str, str]]:
    """Merge all three measures and return rows in year order."""
    measure_data = {
        output_column: read_measure_file(path, source_column)
        for output_column, (path, source_column) in INPUTS.items()
    }
    key_sets = [set(rows) for rows in measure_data.values()]
    if not all(keys == key_sets[0] for keys in key_sets[1:]):
        raise ValueError("Input files do not contain the same shared keys")

    merged_rows = []
    for key in sorted(key_sets[0], key=lambda item: int(item[1])):
        row = {column: key[index] for index, column in enumerate(KEY_COLUMNS)}
        for output_column, (_, source_column) in INPUTS.items():
            row[output_column] = measure_data[output_column][key][source_column]
        merged_rows.append(row)
    return merged_rows


def main() -> None:
    """Merge the raw files and write the processed dataset."""
    rows = merge_data()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()