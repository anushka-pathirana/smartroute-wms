"""
Data Loading and Structure - SQLite review copy of the raw dataset.

Builds a single SQLite file from the public Mendeley CSV files so the data can be reviewed
independently of the Python pipeline (e.g. in DB Browser for SQLite or any SQL client).

Principles
* The CSV files in data/raw/ remain the source of truth; this database is a generated,
  read-only copy and is rebuilt from scratch on every run.
* Faithful copy: every row and every column of each CSV is kept. The only changes are
  snake_case column names, trimmed whitespace around text values and numeric column types.
  The original column names are recorded in the table `_columns`.
* Two convenience additions are clearly marked: `customer_order.creation_datetime` (ISO
  8601 copy of `creation_date`, so SQL date functions work) and the long-format table
  `storage_policy_slot` (one row per occupied slot of the four storage-policy files).

Run:  python scripts/build_review_db.py
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
import re
import sqlite3
from datetime import datetime

import pandas as pd

from smartroute import config

REVIEW_DB_FILE = config.PROJECT_ROOT / "data" / "review" / "smartroute_review.sqlite"
DATASET_CITATION = ("de Assis, de Paula Ferreira and Ouhimmou (2025), Order Picking Dataset from a Warehouse "
                    "of a Footwear Manufacturing Company, Mendeley Data, "
                    "https://data.mendeley.com/datasets/pf2w725pw3/1 (CC BY 4.0)")

# =============================================================================
# DATA LOADING AND STRUCTURE
# =============================================================================


# --- Core Data Structures -----------------------------------------------------------
def to_snake_case(name):
    """Data Loading - Core Data Structures: 'quantityToPick (units)' -> 'quantity_to_pick_units'."""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip())
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return f"level_{name}" if name.isdigit() else name


def _tidy(df):
    """Rename columns to snake_case and trim text values; returns (frame, column map)."""
    column_map = {}
    for column in df.columns:
        column_map[to_snake_case(str(column))] = str(column)
    df.columns = list(column_map)
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].str.strip()
    return df, column_map


def _read_csv(path, sep):
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df, column_map = _tidy(df)
    for column in df.columns:          # numeric where every non-empty value is numeric
        values = df[column].replace("", pd.NA)
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().sum() == values.notna().sum() and values.notna().any():
            df[column] = numeric.astype("Int64") if (numeric.dropna() % 1 == 0).all() else numeric
        else:
            df[column] = values
    return df, column_map


# --- Data Sources -----------------------------------------------------------------
RAW_TABLES = {   # table name -> (raw file, separator, description)
    "picking_wave": (config.PICKING_WAVE_FILE, ";", "One row per unit to pick, per picking wave"),
    "customer_order": (config.CUSTOMER_ORDER_FILE, ";", "Customer order lines with creation date and wave"),
    "product": (config.PRODUCT_FILE, ";", "Product master with ABC demand class and sector"),
    "storage_location": (config.STORAGE_LOCATION_FILE, ",", "Rack locations with CAD coordinates x, y and level z"),
    "storage_policy_random": (config.STORAGE_POLICY_FILES["Random"], ",", "Random storage: up to 18 'reference;size' slots per location"),
    "storage_policy_class_based": (config.STORAGE_POLICY_FILES["ClassBased"], ";", "Class-based (ABC) storage slots per location"),
    "storage_policy_dedicated": (config.STORAGE_POLICY_FILES["Dedicated"], ";", "Dedicated (XYZ) storage slots per location"),
    "storage_policy_hybrid": (config.STORAGE_POLICY_FILES["Hybrid"], ";", "Hybrid storage slots per location"),
}


def _support_points():
    """The navigation file is a two-column CSV ('points_specified;labels') with CR line breaks.

    Each row is '(x, y, z);LABEL'; the published columns are kept and x, y, z are added as numbers.
    """
    raw = config.SUPPORT_POINTS_FILE.read_text(encoding="utf-8-sig")
    lines = [line.strip() for line in re.split(r"\r\n|\r|\n", raw) if line.strip()]
    rows = []
    for line in lines[1:]:                                  # lines[0] is the header
        points, label = (part.strip() for part in line.split(";", 1))
        x, y, z = (float(v) for v in re.findall(r"[-\d.]+", points))
        rows.append({"points_specified": points, "labels": label, "x": x, "y": y, "z": z})
    return pd.DataFrame(rows)


def _policy_slots(policy, df):
    """Long format: one row per occupied slot (location, level, reference, size)."""
    location_col = "original_location" if "original_location" in df.columns else "location"
    level_cols = [c for c in df.columns if c.startswith("level_") or c.startswith("col_")]
    long = df.melt(id_vars=[location_col], value_vars=level_cols, var_name="slot", value_name="value")
    long = long[long["value"].notna() & (long["value"] != "")]
    parts = long["value"].str.split(";", n=1, expand=True)
    return pd.DataFrame({
        "storage_policy": policy,
        "location": long[location_col].values,
        "slot": long["slot"].str.extract(r"(\d+)")[0].astype(int).values,
        "reference": parts[0].str.strip().values,
        "size_us": pd.to_numeric(parts[1], errors="coerce").values,
    })


VIEWS = {
    # Same grouping as preprocessing.clean_picking_waves(): unit rows -> order lines
    "v_order_line": """
        SELECT w.wave_number, w.reference, w.size_us, w.locations AS location, w.operator,
               SUM(w.quantity_to_pick_units) AS units,
               CASE WHEN s.original_location IS NULL THEN 0 ELSE 1 END AS is_rack_location
        FROM picking_wave w
        LEFT JOIN storage_location s ON s.original_location = w.locations
        GROUP BY w.wave_number, w.reference, w.size_us, w.locations, w.operator""",
    "v_wave_summary": """
        SELECT w.wave_number,
               COUNT(*)                      AS pick_rows,
               SUM(w.quantity_to_pick_units) AS units,
               COUNT(DISTINCT w.reference)   AS references_,
               COUNT(DISTINCT w.locations)   AS distinct_locations,
               MIN(w.operator)               AS operator,
               (SELECT MIN(c.creation_datetime) FROM customer_order c WHERE c.wave_number = w.wave_number) AS first_order_created
        FROM picking_wave w
        GROUP BY w.wave_number""",
    "v_pick_with_location": """
        SELECT w.*, s.x, s.y, s.z, p.abccod, p.sector,
               CASE WHEN s.original_location IS NULL THEN 0 ELSE 1 END AS is_rack_location
        FROM picking_wave w
        LEFT JOIN storage_location s ON s.original_location = w.locations
        LEFT JOIN product p          ON p.reference = w.reference""",
    "v_location_usage": """
        SELECT w.locations AS location, COUNT(*) AS pick_rows, COUNT(DISTINCT w.wave_number) AS waves,
               s.x, s.y, s.z
        FROM picking_wave w
        LEFT JOIN storage_location s ON s.original_location = w.locations
        GROUP BY w.locations""",
}

INDEXES = [
    ("picking_wave", "wave_number"), ("picking_wave", "reference"), ("picking_wave", "locations"),
    ("customer_order", "wave_number"), ("customer_order", "reference"),
    ("product", "reference"), ("storage_location", "original_location"),
    ("storage_policy_slot", "location"), ("storage_policy_slot", "reference"),
]


def build_review_db(output_path=REVIEW_DB_FILE):
    """Data Loading - build the SQLite review copy; returns {table: row count}."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    counts, columns, tables, slots = {}, [], [], []
    with sqlite3.connect(output_path) as con:
        for table, (path, sep, description) in RAW_TABLES.items():
            df, column_map = _read_csv(path, sep)
            if table == "customer_order":
                df["creation_datetime"] = pd.to_datetime(df["creation_date"], format="%d/%m/%Y %H:%M") \
                    .dt.strftime("%Y-%m-%d %H:%M")
                column_map["creation_datetime"] = "(added) ISO 8601 copy of creationDate"
            if table.startswith("storage_policy_"):
                slots.append(_policy_slots(table.removeprefix("storage_policy_"), df))
            df.to_sql(table, con, index=False)
            counts[table] = len(df)
            tables.append({"table_name": table, "source_file": path.name, "rows": len(df), "description": description})
            columns += [{"table_name": table, "column_name": k, "original_name": v} for k, v in column_map.items()]

        support = _support_points()
        support.to_sql("support_points", con, index=False)
        counts["support_points"] = len(support)
        tables.append({"table_name": "support_points", "source_file": config.SUPPORT_POINTS_FILE.name,
                       "rows": len(support), "description": "Navigation support points (aisle ends on the cross aisles)"})

        slot_df = pd.concat(slots, ignore_index=True)
        slot_df.to_sql("storage_policy_slot", con, index=False)
        counts["storage_policy_slot"] = len(slot_df)
        tables.append({"table_name": "storage_policy_slot", "source_file": "(derived) the four storage-policy files",
                       "rows": len(slot_df), "description": "Long format: one row per occupied slot per storage policy"})

        pd.DataFrame(tables).to_sql("_tables", con, index=False)
        pd.DataFrame(columns).to_sql("_columns", con, index=False)
        pd.DataFrame([{"key": "dataset", "value": DATASET_CITATION},
                      {"key": "source_folder", "value": "data/raw/"},
                      {"key": "built_at", "value": datetime.now().isoformat(timespec="seconds")},
                      {"key": "note", "value": "Generated read-only copy; the CSV files are the source of truth."}]
                     ).to_sql("_about", con, index=False)
        for table, column in INDEXES:
            con.execute(f"CREATE INDEX idx_{table}_{column} ON {table} ({column})")
        for name, sql in VIEWS.items():
            con.execute(f"CREATE VIEW {name} AS {sql}")
    print(f"[Review DB] {output_path} - " + ", ".join(f"{k}: {v:,}" for k, v in counts.items()))
    return counts
