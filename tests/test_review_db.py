"""Testing and Validation - the SQLite review copy is a faithful copy of the raw CSV files."""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartroute import config  # noqa: E402
from smartroute.review_db import RAW_TABLES, build_review_db  # noqa: E402


@pytest.fixture(scope="module")
def review_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("review") / "review.sqlite"
    counts = build_review_db(path)
    with sqlite3.connect(path) as con:
        yield con, counts


@pytest.mark.parametrize("table", list(RAW_TABLES))
def test_row_and_column_counts_match_csv(review_db, table):
    con, _ = review_db
    path, sep, _ = RAW_TABLES[table]
    csv_df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    db_df = pd.read_sql(f"SELECT * FROM {table}", con)
    extra = {"creation_datetime"} if table == "customer_order" else set()
    assert len(db_df) == len(csv_df)
    assert len(db_df.columns) - len(extra) == len(csv_df.columns)


def test_review_db_reproduces_pipeline_counts(review_db):
    con, _ = review_db
    waves = con.execute("SELECT COUNT(DISTINCT wave_number) FROM picking_wave").fetchone()[0]
    lines, non_rack = con.execute("SELECT COUNT(*), AVG(1 - is_rack_location) FROM v_order_line").fetchone()
    support = con.execute("SELECT COUNT(*) FROM support_points").fetchone()[0]
    assert waves == 9707
    assert lines == 112991
    assert round(100 * non_rack, 1) == 5.2
    assert support == 44 and config.SUPPORT_POINTS_FILE.exists()
