"""
Data Loading and Structure.

Data Sources        : the public Mendeley "Order Picking Dataset from a Warehouse of a Footwear
                      Manufacturing Company" (de Assis, de Paula Ferreira and Ouhimmou, 2025).
Core Data Structures: every raw file is converted into a tidy pandas DataFrame with clean,
                      snake_case column names and stripped string identifiers.
Initial Data Inspection: inspect_data() prints shape, dtypes, null and duplicate counts.
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
import re

import pandas as pd

from smartroute import config

# =============================================================================
# DATA LOADING AND STRUCTURE
# =============================================================================


# --- Data Sources -----------------------------------------------------------------
def load_picking_waves(path=config.PICKING_WAVE_FILE):
    """Data Loading - Data Sources: picking-wave lines (one row per unit to pick)."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = ["wave_number", "reference", "size_us", "quantity", "location", "operator"]
    df["reference"] = df["reference"].astype(str).str.strip()
    df["location"] = df["location"].astype(str).str.strip()
    df["operator"] = df["operator"].astype(str).str.strip()
    return df


def load_customer_orders(path=config.CUSTOMER_ORDER_FILE):
    """Data Loading - Data Sources: customer order lines with creation timestamp and wave id."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = ["customer_code", "order_number", "order_to_collect", "reference", "size_us",
                  "quantity", "creation_date", "wave_number", "operator"]
    df["customer_code"] = df["customer_code"].astype(str).str.strip()
    df["reference"] = df["reference"].astype(str).str.strip()
    df["creation_date"] = pd.to_datetime(df["creation_date"], format="%d/%m/%Y %H:%M")
    return df


def load_products(path=config.PRODUCT_FILE):
    """Data Loading - Data Sources: product master with ABC demand class."""
    df = pd.read_csv(path, sep=";")
    df.columns = ["reference", "abc_class", "sector"]
    for column in df.columns:
        df[column] = df[column].astype(str).str.strip()
    return df


def load_storage_locations(path=config.STORAGE_LOCATION_FILE):
    """Data Loading - Data Sources: storage locations with Cartesian x, y and level z."""
    df = pd.read_csv(path)
    df = df.rename(columns={"originalLocation": "location"})
    df["location"] = df["location"].str.strip()
    df["level"] = df["location"].str[5].astype(int)
    return df[["location", "x", "y", "z", "level"]]


def load_support_points(path=config.SUPPORT_POINTS_FILE):
    """Data Loading - Data Sources: navigation support points (aisle ends on the cross aisles)."""
    raw = path.read_text(encoding="utf-8-sig")
    matches = re.findall(r"\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\);([A-Z]+-\d+)", raw)
    data = [{"label": label, "x": float(x), "y": float(y), "z": float(z)}
            for x, y, z, label in matches]
    return pd.DataFrame(data)


def load_storage_policy(policy_name, path=None):
    """Data Loading - Data Sources: one storage-policy file in long format.

    Each row of the raw file is a location with up to 18 'reference;size' slots. The result
    has one row per occupied slot: location, reference, size_us, storage_policy.
    """
    path = path or config.STORAGE_POLICY_FILES[policy_name]
    if policy_name == "Random":
        df = pd.read_csv(path)
        location_column = "originalLocation"
    else:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        location_column = "Location"
    slot_columns = [c for c in df.columns if c not in (location_column, "ABCCOD", "XYZCOD")]
    long_df = df.melt(id_vars=[location_column], value_vars=slot_columns, value_name="slot")
    long_df = long_df.dropna(subset=["slot"])
    parts = long_df["slot"].astype(str).str.split(";", expand=True)
    stock_df = pd.DataFrame({
        "location": long_df[location_column].astype(str).str.strip().values,
        "reference": parts[0].str.strip().values,
        "size_us": pd.to_numeric(parts[1], errors="coerce").values,
    })
    stock_df["storage_policy"] = policy_name
    return stock_df


def load_data():
    """Data Loading - Core Data Structures: load every source into a dict of DataFrames."""
    data = {
        "picking_waves": load_picking_waves(),
        "customer_orders": load_customer_orders(),
        "products": load_products(),
        "storage_locations": load_storage_locations(),
        "support_points": load_support_points(),
        "storage_policies": {name: load_storage_policy(name) for name in config.STORAGE_POLICIES},
    }
    print(f"[Data Sources] loaded {len(data) - 1} tables + {len(data['storage_policies'])} storage policies")
    return data


# --- Initial Data Inspection ------------------------------------------------------
def inspect_data(data):
    """Data Loading - Initial Data Inspection: shape, nulls and duplicates for every table."""
    rows = []
    tables = {k: v for k, v in data.items() if isinstance(v, pd.DataFrame)}
    tables.update({f"storage_{k}": v for k, v in data["storage_policies"].items()})
    for name, df in tables.items():
        rows.append({"table": name, "rows": len(df), "columns": df.shape[1],
                     "null_cells": int(df.isnull().sum().sum()),
                     "duplicate_rows": int(df.duplicated().sum())})
    inspection_df = pd.DataFrame(rows)
    print("[Initial Data Inspection]")
    print(inspection_df.to_string(index=False))
    return inspection_df
