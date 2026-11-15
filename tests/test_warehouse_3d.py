"""Testing and Validation - the Warehouse 3D viewer data builder."""
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartroute.review_db import build_review_db  # noqa: E402

spec = importlib.util.spec_from_file_location("build_viewer", ROOT / "data" / "warehouse_3d" / "build_viewer.py")
build_viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_viewer)


@pytest.fixture(scope="module")
def viewer_data(tmp_path_factory):
    path = tmp_path_factory.mktemp("viewer") / "review.sqlite"
    build_review_db(path)
    with sqlite3.connect(path) as con:
        return build_viewer.build_viewer_data(con)


def test_viewer_contains_every_rack_location_and_depot(viewer_data):
    assert len(viewer_data["locations"]) == 2292
    assert {loc[3] for loc in viewer_data["locations"]} == {1, 2, 3, 4}
    labels = {s[0] for s in viewer_data["support"]}
    assert viewer_data["meta"]["depot"] in labels and len(labels) == 44


def test_viewer_pick_counts_match_order_lines(viewer_data):
    assert sum(loc[7] for loc in viewer_data["locations"]) == 107076
    assert viewer_data["meta"]["numWaves"] == 9143
    num_locations = len(viewer_data["locations"])
    assert all(0 <= i < num_locations for w in viewer_data["waves"] for i in w[3])


def test_viewer_has_all_four_storage_plans(viewer_data):
    assert set(viewer_data["policies"]) == {"random", "class_based", "dedicated", "hybrid"}
    assert all(viewer_data["policies"][k]["slots"] for k in viewer_data["policies"])
