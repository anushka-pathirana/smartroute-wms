"""
Warehouse 3D viewer - data builder.

Reads the SQLite review copy of the raw dataset (../review/smartroute_review.sqlite) and writes
warehouse_data.js, which index.html loads. Everything comes from the published data files:
rack locations and levels (storage_location), aisle-end points (support_points), picks
(picking_wave grouped into order lines), products (product) and the four storage-policy plans.

Run:  python build_viewer.py          (also run automatically by "Open Warehouse 3D.cmd")
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
CODE_ROOT = TOOL_DIR.parents[1]                       # smartroute-wms/
REVIEW_DB_FILE = CODE_ROOT / "data" / "review" / "smartroute_review.sqlite"
OUTPUT_FILE = TOOL_DIR / "warehouse_data.js"
UNIT_TO_METRES = 3.66 / 48.0                          # same scale as the SmartRoute pipeline
# location row: [code, x, y, z, rack, bay, slot, order_lines, units, waves, top3 [[ref_id, units]], abc_of_top_ref,
#                all picked ref_ids]
DEPOT_LABEL = "CC-08"                                 # I/O point (teal square in the authors' drawings)
POLICIES = [("random", "Random"), ("class_based", "Class-based"), ("dedicated", "Dedicated"), ("hybrid", "Hybrid")]
POLICY_CLASS_COLUMN = {"random": None, "class_based": "abccod", "dedicated": "xyzcod", "hybrid": "xyzcod"}

# =============================================================================
# DATA LOADING AND STRUCTURE
# =============================================================================


def ensure_review_db():
    """Build the SQLite review copy first if it does not exist yet."""
    if REVIEW_DB_FILE.exists():
        return
    print("[Warehouse 3D] review database not found - building it from data/raw ...")
    sys.path.insert(0, str(CODE_ROOT / "src"))
    from smartroute.review_db import build_review_db
    build_review_db(REVIEW_DB_FILE)


def build_viewer_data(con):
    """Data Loading - collect everything the viewer needs into one JSON-serialisable dict."""
    q = lambda sql, *args: con.execute(sql, args).fetchall()  # noqa: E731

    # --- master data -------------------------------------------------------------
    products = q("SELECT reference, abccod FROM product ORDER BY reference")
    ref_index = {ref: i for i, (ref, _) in enumerate(products)}
    product_abc = dict(products)

    def ref_id(ref):
        if ref not in ref_index:                       # references only seen in storage plans
            ref_index[ref] = len(ref_index)
            products.append((ref, None))
        return ref_index[ref]

    locations = q("SELECT original_location, x, y, z FROM storage_location ORDER BY original_location")
    loc_index = {row[0]: i for i, row in enumerate(locations)}
    support = q("SELECT labels, x, y FROM support_points")

    # --- transactions: order lines per location -------------------------------------
    lines = q("SELECT wave_number, reference, location, units FROM v_order_line WHERE is_rack_location = 1")
    picks, units, waves_at, refs_at = Counter(), Counter(), defaultdict(set), defaultdict(Counter)
    wave_locs = defaultdict(set)
    for wave, ref, loc, n in lines:
        i = loc_index[loc]
        picks[i] += 1
        units[i] += n
        waves_at[i].add(wave)
        refs_at[i][ref] += n
        wave_locs[wave].add(i)
    non_rack = Counter(w for (w,) in q("SELECT wave_number FROM v_order_line WHERE is_rack_location = 0"))

    loc_rows = []
    for i, (code, x, y, z) in enumerate(locations):
        rack, bay, slot = code.split("-")
        top = refs_at[i].most_common(3)
        main_abc = product_abc.get(top[0][0]) if top else None
        loc_rows.append([code, x, y, z, rack, bay, slot, picks[i], units[i], len(waves_at[i]),
                         [[ref_id(r), c] for r, c in top], main_abc, sorted(ref_id(r) for r in refs_at[i])])

    # --- storage-policy plans ---------------------------------------------------------
    policies = {}
    for key, label in POLICIES:
        cls_col = POLICY_CLASS_COLUMN[key]
        table = f"storage_policy_{key}"
        loc_col = "original_location" if key == "random" else "location"
        classes = dict(q(f"SELECT {loc_col}, {cls_col} FROM {table}")) if cls_col else {}
        slots = defaultdict(list)
        for loc, slot, ref, size in q("SELECT location, slot, reference, size_us FROM storage_policy_slot "
                                      "WHERE storage_policy = ? ORDER BY location, slot", key):
            if loc in loc_index:
                slots[loc_index[loc]].append([ref_id(ref), size])
        policies[key] = {"label": label, "classColumn": cls_col.upper() if cls_col else None,
                         "classes": {loc_index[k]: v for k, v in classes.items() if k in loc_index},
                         "slots": slots}

    # --- waves --------------------------------------------------------------------------
    wave_meta = {w: (t, op) for w, t, op in q(
        "SELECT wave_number, first_order_created, operator FROM v_wave_summary")}
    waves = []
    for wave in sorted(wave_locs):
        created, operator = wave_meta.get(wave, (None, None))
        waves.append([wave, created, operator, sorted(wave_locs[wave]), non_rack.get(wave, 0)])

    period = q("SELECT MIN(creation_datetime), MAX(creation_datetime) FROM customer_order")[0]
    return {
        "meta": {"builtAt": datetime.now().isoformat(timespec="seconds"), "unitToMetres": UNIT_TO_METRES,
                 "depot": DEPOT_LABEL, "period": period, "source": "data/review/smartroute_review.sqlite",
                 "numLocations": len(locations), "numWaves": len(waves),
                 "numOrderLines": sum(picks.values()), "numProducts": len(product_abc)},
        "products": [[r, a] for r, a in products],
        "locations": loc_rows,
        "support": [[label, x, y] for label, x, y in support],
        "policies": policies,
        "waves": waves,
    }


def main():
    ensure_review_db()
    with sqlite3.connect(REVIEW_DB_FILE) as con:
        data = build_viewer_data(con)
    payload = json.dumps(data, separators=(",", ":"))
    OUTPUT_FILE.write_text("// Generated by build_viewer.py - do not edit.\nwindow.WAREHOUSE = " + payload + ";\n",
                           encoding="utf-8")
    m = data["meta"]
    print(f"[Warehouse 3D] {OUTPUT_FILE.name}: {m['numLocations']:,} locations, {m['numWaves']:,} waves, "
          f"{m['numOrderLines']:,} order lines ({OUTPUT_FILE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
