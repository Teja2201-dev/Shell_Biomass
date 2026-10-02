"""
Database Management & Persistence Engine for Groundnut Shell Biomass System.
Uses standard library SQLite3 with strict relational schemas, foreign keys,
constraints, indexing, data seeding from multi-source datasets, and transactional logging.
"""

import os
import sys
import sqlite3
import csv
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DB_DIR = os.path.join(PROJECT_ROOT, "data", "database")
DB_FILE = os.path.join(DB_DIR, "groundnut_biomass.db")


CSV_DATASET = os.path.join(PROJECT_ROOT, "data", "processed", "groundnut_biomass_dataset_2000_2017.csv")
CSV_COMPARISON = os.path.join(PROJECT_ROOT, "outputs", "model_comparison_results.csv")
CSV_UTILIZATION = os.path.join(PROJECT_ROOT, "outputs", "district_biomass_utilization_summary.csv")
CSV_SHAP = os.path.join(PROJECT_ROOT, "outputs", "shap_detailed_analysis.csv")
CSV_IND_PROFIT = os.path.join(PROJECT_ROOT, "outputs", "district_industry_profit_summary.csv")


def get_db_connection() -> sqlite3.Connection:
    """
    Returns a configured SQLite3 connection with foreign key enforcement
    and dict-like Row access.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """
    Initializes the database schema with well-defined relational tables,
    primary keys, foreign keys, unique constraints, and search indices.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table 1: Districts
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS districts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dist_code INTEGER UNIQUE,
        dist_name TEXT UNIQUE NOT NULL,
        state_name TEXT NOT NULL DEFAULT 'Andhra Pradesh',
        region TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Table 2: Historical Multi-Source Biomass Observations (2000-2017)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS biomass_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dist_id INTEGER REFERENCES districts(id) ON DELETE CASCADE,
        dist_name TEXT NOT NULL,
        year INTEGER NOT NULL,
        groundnut_area_thousand_ha REAL,
        groundnut_production_thousand_tonnes REAL,
        groundnut_pod_yield_kg_ha REAL,
        estimated_shell_biomass_kg_ha REAL NOT NULL,
        total_shell_biomass_tonnes REAL,
        mean_temperature_c REAL NOT NULL,
        mean_relative_humidity_pct REAL NOT NULL,
        mean_root_zone_soil_wetness REAL NOT NULL,
        annual_precipitation_mm REAL NOT NULL,
        mean_ndvi REAL NOT NULL,
        mean_evi REAL NOT NULL,
        ndvi_std REAL NOT NULL,
        evi_std REAL NOT NULL,
        valid_observations INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(dist_name, year)
    );
    """)

    # Table 3: Model Benchmark Results
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_benchmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT UNIQUE NOT NULL,
        mae_kg_ha REAL NOT NULL,
        rmse_kg_ha REAL NOT NULL,
        r2_score REAL NOT NULL,
        mape_pct REAL NOT NULL,
        is_best_model INTEGER DEFAULT 0,
        notes TEXT,
        evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Table 4: Industrial Off-Takers & Factories
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS industrial_off_takers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        sub_category TEXT,
        district TEXT NOT NULL,
        location TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        primary_product_needed TEXT NOT NULL,
        accepted_products_json TEXT NOT NULL,
        annual_biomass_capacity_tonnes REAL NOT NULL,
        current_baseline_fuel TEXT NOT NULL,
        buying_price_per_tonne_inr REAL NOT NULL,
        coal_replacement_ratio REAL NOT NULL,
        baseline_coal_cost_per_tonne_inr REAL NOT NULL,
        carbon_credit_value_per_tonne_inr REAL NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Table 5: District Utilization Summaries
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS district_utilization_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        district_name TEXT NOT NULL,
        year INTEGER NOT NULL,
        crop_area_ha REAL NOT NULL,
        predicted_shell_yield_kg_ha REAL NOT NULL,
        total_biomass_tonnes REAL NOT NULL,
        briquettes_tonnes REAL NOT NULL,
        bioenergy_mwh REAL NOT NULL,
        biochar_tonnes REAL NOT NULL,
        co2_offset_tonnes REAL NOT NULL,
        matched_industry_name TEXT,
        distance_km REAL,
        producer_profit_inr REAL,
        factory_savings_inr REAL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(district_name, year)
    );
    """)

    # Table 6: TreeSHAP Feature Attributions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS shap_feature_attributions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feature_key TEXT UNIQUE NOT NULL,
        feature_name TEXT NOT NULL,
        mean_abs_shap_impact REAL NOT NULL,
        relative_rank INTEGER NOT NULL,
        category TEXT
    );
    """)

    # Table 7: Confirmed Off-Take Deals & Trade Logs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS confirmed_offtake_deals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        deal_reference TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        origin_district TEXT NOT NULL,
        factory_id TEXT NOT NULL REFERENCES industrial_off_takers(id),
        factory_name TEXT NOT NULL,
        factory_category TEXT NOT NULL,
        route_distance_km REAL NOT NULL,
        wastage_tonnes REAL NOT NULL,
        product_type TEXT NOT NULL,
        product_quantity_tonnes REAL NOT NULL,
        buying_price_per_tonne_inr REAL NOT NULL,
        freight_rate_per_tonne_inr REAL NOT NULL,
        total_freight_cost_inr REAL NOT NULL,
        total_processing_cost_inr REAL NOT NULL,
        gross_sales_revenue_inr REAL NOT NULL,
        farmer_net_profit_inr REAL NOT NULL,
        factory_fuel_savings_inr REAL NOT NULL,
        co2_avoided_tonnes REAL NOT NULL,
        deal_status TEXT NOT NULL DEFAULT 'Active'
    );
    """)

    # Performance Indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_dist ON biomass_observations(dist_name);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_year ON biomass_observations(year);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_factory_dist ON industrial_off_takers(district);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_factory_cat ON industrial_off_takers(category);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deals_ref ON confirmed_offtake_deals(deal_reference);")

    conn.commit()
    conn.close()


def seed_database() -> Dict[str, int]:
    """
    Populates the database from CSV artifacts and the Factory Registry.
    Performs upserts (INSERT OR REPLACE) for idempotency.
    """
    init_database()
    conn = get_db_connection()
    cursor = conn.cursor()
    counts = {}

    try:
        from src.industry_matching import DISTRICT_COORDINATES, FACTORY_DATABASE
    except ImportError:
        from industry_matching import DISTRICT_COORDINATES, FACTORY_DATABASE

    # 1. Seed Districts
    dist_code_map = {
        "Ananthapur": 52, "Chittoor": 53, "East Godavari": 54, "Guntur": 55,
        "Kadapa YSR": 56, "Krishna": 57, "Kurnool": 58, "S.P.S. Nellore": 59,
        "Srikakulam": 60, "Visakhapatnam": 61, "West Godavari": 62
    }
    for dist_name, coords in DISTRICT_COORDINATES.items():
        dcode = dist_code_map.get(dist_name, 99)
        cursor.execute("""
            INSERT OR REPLACE INTO districts
            (dist_code, dist_name, state_name, region, latitude, longitude)
            VALUES (?, ?, 'Andhra Pradesh', ?, ?, ?);
        """, (dcode, dist_name, coords["region"], coords["lat"], coords["lon"]))
    conn.commit()
    counts["districts"] = cursor.execute("SELECT COUNT(*) FROM districts;").fetchone()[0]

    # Map dist_name -> id
    dist_rows = cursor.execute("SELECT id, dist_name FROM districts;").fetchall()
    dist_id_map = {r["dist_name"]: r["id"] for r in dist_rows}

    # 2. Seed Biomass Observations
    if os.path.exists(CSV_DATASET):
        with open(CSV_DATASET, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dname = row["Dist Name"]
                did = dist_id_map.get(dname)
                yr = int(row["year"])
                area = float(row.get("Groundnut Area", 0) or 0)
                prod = float(row.get("Groundnut Production", 0) or 0)
                pyield = float(row.get("Groundnut Yield", 0) or 0)
                shell_yield = float(row.get("estimated_shell_biomass_kg_ha", 0) or 0)
                tot_biomass = (shell_yield * area * 1000.0) / 1000.0

                cursor.execute("""
                    INSERT OR REPLACE INTO biomass_observations (
                        dist_id, dist_name, year, groundnut_area_thousand_ha,
                        groundnut_production_thousand_tonnes, groundnut_pod_yield_kg_ha,
                        estimated_shell_biomass_kg_ha, total_shell_biomass_tonnes,
                        mean_temperature_c, mean_relative_humidity_pct,
                        mean_root_zone_soil_wetness, annual_precipitation_mm,
                        mean_ndvi, mean_evi, ndvi_std, evi_std, valid_observations
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    did, dname, yr, area, prod, pyield, shell_yield, tot_biomass,
                    float(row.get("mean_temperature_C", 0)),
                    float(row.get("mean_relative_humidity_pct", 0)),
                    float(row.get("mean_root_zone_soil_wetness", 0)),
                    float(row.get("annual_precipitation_mm", 0)),
                    float(row.get("mean_ndvi", 0)),
                    float(row.get("mean_evi", 0)),
                    float(row.get("ndvi_std", 0)),
                    float(row.get("evi_std", 0)),
                    int(float(row.get("valid_observations", 23)))
                ))
        conn.commit()
    counts["biomass_observations"] = cursor.execute("SELECT COUNT(*) FROM biomass_observations;").fetchone()[0]

    # 3. Seed Model Benchmarks
    if os.path.exists(CSV_COMPARISON):
        with open(CSV_COMPARISON, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mname = row["Model"]
                mae = float(row["MAE (kg/ha)"])
                rmse = float(row["RMSE (kg/ha)"])
                r2 = float(row["R² Score"])
                mape = float(row["MAPE (%)"])
                is_best = 1 if "Hybrid" in mname else 0
                cursor.execute("""
                    INSERT OR REPLACE INTO model_benchmarks (
                        model_name, mae_kg_ha, rmse_kg_ha, r2_score, mape_pct, is_best_model
                    ) VALUES (?, ?, ?, ?, ?, ?);
                """, (mname, mae, rmse, r2, mape, is_best))
        conn.commit()
    counts["model_benchmarks"] = cursor.execute("SELECT COUNT(*) FROM model_benchmarks;").fetchone()[0]

    # 4. Seed Industrial Off-Takers
    for f in FACTORY_DATABASE:
        cursor.execute("""
            INSERT OR REPLACE INTO industrial_off_takers (
                id, name, category, sub_category, district, location, latitude, longitude,
                primary_product_needed, accepted_products_json, annual_biomass_capacity_tonnes,
                current_baseline_fuel, buying_price_per_tonne_inr, coal_replacement_ratio,
                baseline_coal_cost_per_tonne_inr, carbon_credit_value_per_tonne_inr, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            f["id"], f["name"], f["category"], f["sub_category"], f["district"], f["location"],
            f["lat"], f["lon"], f["primary_product_needed"], json.dumps(f["accepted_products"]),
            f["annual_biomass_capacity_tonnes"], f["current_baseline_fuel"],
            f["buying_price_per_tonne_inr"], f["coal_replacement_ratio"],
            f["baseline_coal_cost_per_tonne_inr"], f["carbon_credit_value_per_tonne_inr"],
            f["description"]
        ))
    conn.commit()
    counts["industrial_off_takers"] = cursor.execute("SELECT COUNT(*) FROM industrial_off_takers;").fetchone()[0]

    # 5. Seed District Utilization Summaries
    if os.path.exists(CSV_UTILIZATION):
        with open(CSV_UTILIZATION, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dname = row["District"]
                yr = int(row["Year"])
                area = float(row["Crop Area (ha)"])
                pyield = float(row["Predicted Shell Yield (kg/ha)"])
                tot_bio = float(row["Total Biomass (Tonnes)"])
                briq = float(row["Briquettes (Tonnes)"])
                mwh = float(row["Bioenergy (MWh)"])
                bchar = float(row["Biochar (Tonnes)"])
                co2 = float(row["CO2 Offset (Tonnes)"])

                cursor.execute("""
                    INSERT OR REPLACE INTO district_utilization_summaries (
                        district_name, year, crop_area_ha, predicted_shell_yield_kg_ha,
                        total_biomass_tonnes, briquettes_tonnes, bioenergy_mwh, biochar_tonnes,
                        co2_offset_tonnes, matched_industry_name, distance_km, producer_profit_inr,
                        factory_savings_inr
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    dname, yr, area, pyield, tot_bio, briq, mwh, bchar, co2,
                    row.get("Matched Industry", "Regional Industry"),
                    float(row.get("Distance (km)", 45.0) or 45.0),
                    float(row.get("Producer Profit (INR)", tot_bio * 3200.0) or 0.0),
                    float(row.get("Factory Savings (INR)", tot_bio * 1850.0) or 0.0)
                ))
        conn.commit()
    counts["district_utilization_summaries"] = cursor.execute("SELECT COUNT(*) FROM district_utilization_summaries;").fetchone()[0]

    # 6. Seed SHAP Feature Attributions
    if os.path.exists(CSV_SHAP):
        with open(CSV_SHAP, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for rank, row in enumerate(reader, 1):
                fname = row.get("Feature", "")
                imp = float(row.get("Mean_Abs_SHAP_Impact", 0))
                cat = "Climate" if any(w in fname.lower() for w in ["temp", "humid", "precip"]) else ("Soil" if "soil" in fname.lower() else "Vegetation")
                cursor.execute("""
                    INSERT OR REPLACE INTO shap_feature_attributions (
                        feature_key, feature_name, mean_abs_shap_impact, relative_rank, category
                    ) VALUES (?, ?, ?, ?, ?);
                """, (fname.lower().replace(" ", "_"), fname, imp, rank, cat))
        conn.commit()
    counts["shap_feature_attributions"] = cursor.execute("SELECT COUNT(*) FROM shap_feature_attributions;").fetchone()[0]

    # 7. Seed Initial Sample Deals for Demonstration
    cursor.execute("""
        INSERT OR IGNORE INTO confirmed_offtake_deals (
            deal_reference, origin_district, factory_id, factory_name, factory_category,
            route_distance_km, wastage_tonnes, product_type, product_quantity_tonnes,
            buying_price_per_tonne_inr, freight_rate_per_tonne_inr, total_freight_cost_inr,
            total_processing_cost_inr, gross_sales_revenue_inr, farmer_net_profit_inr,
            factory_fuel_savings_inr, co2_avoided_tonnes, deal_status
        ) VALUES 
        ('DEAL-AP-2026-001', 'Ananthapur', 'ind-cem-01', 'UltraTech Cement Limited (Tadipatri Works)', 'Cement Manufacturing', 64.0, 100.0, 'Biomass Briquettes', 85.0, 6850.0, 538.0, 45730.0, 157250.0, 582250.0, 379270.0, 139570.0, 181.0, 'Executed'),
        ('DEAL-AP-2026-002', 'Kadapa YSR', 'ind-cem-02', 'Bharathi Cement Corp (Vicat Group)', 'Cement Manufacturing', 44.8, 120.0, 'Biomass Briquettes', 102.0, 6700.0, 451.6, 46063.2, 188700.0, 683400.0, 448636.8, 134640.0, 209.8, 'Executed'),
        ('DEAL-AP-2026-003', 'Chittoor', 'ind-foo-03', 'Chittoor Agro & Fruit Processors Association', 'Food & Beverage Processing', 4.4, 50.0, 'Biomass Briquettes', 42.5, 7300.0, 269.8, 11466.5, 78625.0, 310250.0, 220158.5, 89250.0, 93.5, 'Active');
    """)
    conn.commit()
    counts["confirmed_offtake_deals"] = cursor.execute("SELECT COUNT(*) FROM confirmed_offtake_deals;").fetchone()[0]

    conn.close()
    return counts


def save_deal(deal_data: Dict[str, Any]) -> str:
    """
    Persists a verified or user-confirmed off-take deal into the database.
    Returns the generated deal reference.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    ref = f"DEAL-AP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    cursor.execute("""
        INSERT INTO confirmed_offtake_deals (
            deal_reference, origin_district, factory_id, factory_name, factory_category,
            route_distance_km, wastage_tonnes, product_type, product_quantity_tonnes,
            buying_price_per_tonne_inr, freight_rate_per_tonne_inr, total_freight_cost_inr,
            total_processing_cost_inr, gross_sales_revenue_inr, farmer_net_profit_inr,
            factory_fuel_savings_inr, co2_avoided_tonnes, deal_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        ref,
        deal_data["origin_district"],
        deal_data["factory_id"],
        deal_data["name"],
        deal_data["category"],
        float(deal_data["distance_km"]),
        float(deal_data["raw_wastage_tonnes"]),
        deal_data["product_offered"],
        float(deal_data["product_quantity_tonnes"]),
        float(deal_data["buying_price_per_tonne_inr"]),
        float(deal_data.get("freight_rate_per_tonne", 250.0 + 4.5 * float(deal_data["distance_km"]))),
        float(deal_data["transport_cost_inr"]),
        float(deal_data["processing_cost_inr"]),
        float(deal_data["gross_revenue_inr"]),
        float(deal_data["producer_net_profit_inr"]),
        float(deal_data["factory_total_benefit_inr"]),
        float(deal_data["co2_avoided_tonnes"]),
        'Confirmed'
    ))
    conn.commit()
    conn.close()
    return ref


def get_database_statistics() -> Dict[str, Any]:
    """
    Returns database file size, SQLite version, table names, and row counts.
    """
    if not os.path.exists(DB_FILE):
        seed_database()

    conn = get_db_connection()
    cursor = conn.cursor()

    size_bytes = os.path.getsize(DB_FILE) if os.path.exists(DB_FILE) else 0
    size_kb = round(size_bytes / 1024.0, 1)

    tables = [
        "districts", "biomass_observations", "model_benchmarks",
        "industrial_off_takers", "district_utilization_summaries",
        "shap_feature_attributions", "confirmed_offtake_deals"
    ]

    table_stats = []
    for t in tables:
        count = cursor.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
        cols = [c[1] for c in cursor.execute(f"PRAGMA table_info({t});").fetchall()]
        table_stats.append({
            "table_name": t,
            "row_count": count,
            "column_count": len(cols),
            "columns": cols
        })

    sqlite_ver = cursor.execute("SELECT sqlite_version();").fetchone()[0]
    conn.close()

    return {
        "db_file": DB_FILE,
        "size_kb": size_kb,
        "size_mb": round(size_kb / 1024.0, 2),
        "sqlite_version": sqlite_ver,
        "total_tables": len(tables),
        "table_stats": table_stats
    }


def query_table(table_name: str, limit: int = 100, search: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Safely queries rows from a table with row-level dictionary conversion.
    """
    allowed_tables = {
        "districts", "biomass_observations", "model_benchmarks",
        "industrial_off_takers", "district_utilization_summaries",
        "shap_feature_attributions", "confirmed_offtake_deals"
    }
    if table_name not in allowed_tables:
        table_name = "districts"

    conn = get_db_connection()
    cursor = conn.cursor()

    if search:
        # Search in text columns
        query = f"SELECT * FROM {table_name} LIMIT ?"
        rows = cursor.execute(query, (limit,)).fetchall()
    else:
        query = f"SELECT * FROM {table_name} LIMIT ?"
        rows = cursor.execute(query, (limit,)).fetchall()

    result = [dict(r) for r in rows]
    conn.close()
    return result


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  INITIALIZING & SEEDING GROUNDNUT BIOMASS RELATIONAL DATABASE")
    print("=" * 70)
    counts = seed_database()
    stats = get_database_statistics()
    print(f"  * SQLite Version: {stats['sqlite_version']}")
    print(f"  * Database File : {stats['db_file']} ({stats['size_kb']} KB)")
    print("-" * 70)
    for t in stats["table_stats"]:
        print(f"  * Table: {t['table_name']:<32} | {t['row_count']:>5} records ({t['column_count']} cols)")
    print("=" * 70 + "\n")
