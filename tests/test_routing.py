"""Testing and Validation - functional tests for the layout, routing, features and split."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartroute import config, data_loading, preprocessing  # noqa: E402
from smartroute.features import FEATURE_COLUMNS, compute_order_features  # noqa: E402
from smartroute.warehouse.layout import PickPoint, build_layout, path_length  # noqa: E402
from smartroute.warehouse.routing import route_heuristic, route_optimal  # noqa: E402


@pytest.fixture(scope="module")
def layout():
    return build_layout(data_loading.load_support_points())


@pytest.fixture(scope="module")
def storage_locations():
    return data_loading.load_storage_locations()


def random_pick_list(layout, storage_locations, size, seed):
    sample = storage_locations.sample(n=size, random_state=seed)
    return [layout.to_pick_point(x, y) for x, y in zip(sample["x"], sample["y"])]


def test_layout_matches_dataset(layout):
    assert len(layout.aisle_ys) == 17
    assert len(layout.right_aisle_ys) == 10
    assert layout.depot_xy == config.DEPOT_XY


def test_distance_is_symmetric_and_metric(layout, storage_locations):
    points = random_pick_list(layout, storage_locations, 12, seed=1)
    matrix = layout.distance_matrix(points)
    assert np.allclose(matrix, matrix.T)
    assert (np.diag(matrix) == 0).all()
    for i in range(len(points)):
        for j in range(len(points)):
            assert matrix[i, j] <= (matrix[i, :] + matrix[:, j]).min() + 1e-6


def test_same_aisle_distance(layout):
    p, q = PickPoint("L", 61.0, 100.0), PickPoint("L", 61.0, 300.0)
    assert layout.distance(p, q) == pytest.approx(200.0)


def test_cross_block_distance_uses_front_aisle(layout):
    p, q = PickPoint("L", 841.0, 300.0), PickPoint("R", 841.0, 500.0)
    assert layout.distance(p, q) == pytest.approx(103.0 + 97.0)


@pytest.mark.parametrize("policy", config.ROUTING_POLICIES)
@pytest.mark.parametrize("seed", range(15))
def test_heuristic_routes_are_feasible_and_not_better_than_optimal(layout, storage_locations, policy, seed):
    points = random_pick_list(layout, storage_locations, 3 + seed, seed)
    route = route_heuristic(layout, points, policy)
    optimal = route_optimal(layout, points)
    waypoints = np.asarray(route.waypoints)
    steps = np.diff(waypoints, axis=0)
    assert ((steps[:, 0] == 0) | (steps[:, 1] == 0)).all(), "moves must follow aisles"
    assert tuple(waypoints[0]) == tuple(waypoints[-1]) == layout.depot.xy
    assert route.distance_units >= optimal.distance_units - 1e-6
    assert optimal.is_optimal_proven
    for p in points:
        on_route = any(min(a[0], b[0]) - 1e-6 <= p.x <= max(a[0], b[0]) + 1e-6
                       and min(a[1], b[1]) - 1e-6 <= p.aisle_y <= max(a[1], b[1]) + 1e-6
                       for a, b in zip(waypoints[:-1], waypoints[1:]))
        assert on_route, f"{policy} misses pick {p}"


def test_optimal_waypoints_length_matches_tour(layout, storage_locations):
    points = random_pick_list(layout, storage_locations, 14, seed=7)
    optimal = route_optimal(layout, points)
    assert path_length(optimal.waypoints) == pytest.approx(optimal.distance_units)


def test_single_pick_all_policies_equal(layout):
    points = [PickPoint("L", 1021.0, 250.0)]
    distances = {p: route_heuristic(layout, points, p).distance_units for p in config.ROUTING_POLICIES}
    assert len(set(np.round(list(distances.values()), 6))) == 1
    assert distances["SShape"] == pytest.approx(route_optimal(layout, points).distance_units)


def test_features_complete_and_finite(layout, storage_locations):
    points = random_pick_list(layout, storage_locations, 10, seed=3)
    features = compute_order_features(layout, points)
    assert set(FEATURE_COLUMNS) == set(features)
    assert all(np.isfinite(v) for v in features.values())
    assert not any(c.startswith("dist_") for c in FEATURE_COLUMNS), "no route distance may leak into X"


def test_split_has_no_wave_or_configuration_leakage():
    df = pd.DataFrame({"storage_policy": ["A", "A", "B", "B"] * 25, "wave_number": np.repeat(np.arange(50), 2),
                       "wave_rank": np.repeat(np.arange(50), 2)})
    cutoff = preprocessing.chronological_cutoff(df)
    train_df, test_df = preprocessing.split_data(df, "B", cutoff)
    assert set(test_df["storage_policy"]) == {"B"} and "B" not in set(train_df["storage_policy"])
    assert not set(train_df["wave_number"]) & set(test_df["wave_number"])
    assert train_df["wave_rank"].max() < test_df["wave_rank"].min()
