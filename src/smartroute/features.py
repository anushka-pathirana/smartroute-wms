"""
Preprocessing and Feature Engineering - order-level features for routing-policy selection.

Every feature is computed from the pick list and the static layout only (number of picks,
aisles visited, how far apart the picks are, how deep into the aisles they go, block usage,
product-class mix). No feature uses a route distance, so the label cannot leak into X.
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
from collections import defaultdict

import numpy as np
import pandas as pd

from smartroute import config
from smartroute.warehouse.routing import unique_points

FEATURE_GROUPS = {
    "Order size": ["num_lines", "num_units", "num_picks", "picks_per_aisle_mean", "picks_per_aisle_max"],
    "Aisle coverage": ["num_aisles", "num_aisles_left", "num_aisles_right", "num_blocks",
                       "share_right_block", "odd_aisles_left", "odd_aisles_right",
                       "aisle_span_m", "aisle_density"],
    "Aisle depth": ["depth_mean_frac", "depth_max_frac", "depth_std_frac", "deepest_frac_mean",
                    "deep_aisle_share", "largest_gap_frac_mean"],
    "Dispersion": ["depot_distance_mean_m", "depot_distance_max_m", "nn_distance_mean_m",
                   "x_std_m", "y_std_m", "share_below_depot"],
    "Product mix": ["share_a", "share_b", "share_c"],
}
FEATURE_COLUMNS = [f for group in FEATURE_GROUPS.values() for f in group]

# =============================================================================
# PREPROCESSING AND FEATURE ENGINEERING
# =============================================================================


# --- Feature Engineering ------------------------------------------------------------
def compute_order_features(layout, points, num_lines=None, num_units=None, abc_shares=(0.0, 0.0, 0.0)):
    """Preprocessing - Feature Engineering: features of one pick list."""
    unit = config.UNIT_TO_METRES
    picks = unique_points(points)
    num_picks = len(picks)
    aisles = defaultdict(list)
    for p in picks:
        aisles[(p.block, p.aisle_y)].append(layout.depth(p) / layout.aisle_length[p.block])
    num_aisles = len(aisles)
    left = [k for k in aisles if k[0] == "L"]
    right = [k for k in aisles if k[0] == "R"]
    depths = np.array([d for v in aisles.values() for d in v]) if num_picks else np.zeros(1)
    deepest = np.array([max(v) for v in aisles.values()]) if num_aisles else np.zeros(1)
    largest_gaps = []
    for v in aisles.values():
        positions = np.concatenate([[0.0], np.sort(v), [1.0]])
        largest_gaps.append(np.diff(positions).max())
    aisle_ys = sorted({k[1] for k in aisles})
    span = (aisle_ys[-1] - aisle_ys[0]) if aisle_ys else 0.0
    aisles_in_span = sum(1 for y in layout.aisle_ys if aisle_ys and aisle_ys[0] <= y <= aisle_ys[-1])
    if num_picks:
        depot_d = np.array([layout.distance(layout.depot, p) for p in picks])
        if num_picks > 1:
            matrix = layout.distance_matrix(picks)
            np.fill_diagonal(matrix, np.inf)
            nn_mean = matrix.min(axis=1).mean()
        else:
            nn_mean = 0.0
        xs = np.array([p.x for p in picks])
        ys = np.array([p.aisle_y for p in picks])
    else:
        depot_d, nn_mean, xs, ys = np.zeros(1), 0.0, np.zeros(1), np.zeros(1)
    counts = np.array([len(v) for v in aisles.values()]) if num_aisles else np.zeros(1)
    return {
        "num_lines": num_lines if num_lines is not None else len(points),
        "num_units": num_units if num_units is not None else len(points),
        "num_picks": num_picks,
        "picks_per_aisle_mean": counts.mean(),
        "picks_per_aisle_max": counts.max(),
        "num_aisles": num_aisles,
        "num_aisles_left": len(left),
        "num_aisles_right": len(right),
        "num_blocks": int(bool(left)) + int(bool(right)),
        "share_right_block": len(right) / num_aisles if num_aisles else 0.0,
        "odd_aisles_left": len(left) % 2,
        "odd_aisles_right": len(right) % 2,
        "aisle_span_m": span * unit,
        "aisle_density": num_aisles / aisles_in_span if aisles_in_span else 0.0,
        "depth_mean_frac": depths.mean(),
        "depth_max_frac": depths.max(),
        "depth_std_frac": depths.std(),
        "deepest_frac_mean": deepest.mean(),
        "deep_aisle_share": (deepest > 0.5).mean(),
        "largest_gap_frac_mean": float(np.mean(largest_gaps)) if largest_gaps else 0.0,
        "depot_distance_mean_m": depot_d.mean() * unit,
        "depot_distance_max_m": depot_d.max() * unit,
        "nn_distance_mean_m": nn_mean * unit,
        "x_std_m": xs.std() * unit,
        "y_std_m": ys.std() * unit,
        "share_below_depot": float((ys < layout.depot_xy[1]).mean()),
        "share_a": abc_shares[0], "share_b": abc_shares[1], "share_c": abc_shares[2],
    }


def engineer_features(instances_df, layout):
    """Preprocessing - Feature Engineering: feature matrix for every routing instance."""
    rows = [compute_order_features(layout, r.points, r.num_lines, r.num_units, (r.share_a, r.share_b, r.share_c))
            for r in instances_df.itertuples()]
    features_df = pd.DataFrame(rows, index=instances_df.index)[FEATURE_COLUMNS]
    print(f"[Feature Engineering] {features_df.shape[1]} features for {len(features_df):,} instances")
    return features_df
