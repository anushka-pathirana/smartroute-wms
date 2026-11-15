"""
Preprocessing and Feature Engineering - cleaning, routing-instance construction,
label generation and leakage-safe data splitting.

Routing instance  : one picking wave (pick list) served under one storage configuration.
Storage configs   : AsRecorded (locations logged by the WMS) + the four storage-policy files
                    published with the dataset (Random, ClassBased, Dedicated, Hybrid).
Label             : best_policy = argmin over {SShape, Return, LargestGap} of route distance;
                    ties are broken in favour of the simpler, industry-default policy order.
Splitting         : test = held-out storage configuration AND the most recent 20 % of waves,
                    so a wave never appears in both training and test data and the model is
                    always tested on a storage configuration it has not seen.
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from smartroute import config
from smartroute.warehouse.routing import route_heuristic, route_optimal

ALL_STORAGE_CONFIGS = ["AsRecorded"] + config.STORAGE_POLICIES

# =============================================================================
# PREPROCESSING AND FEATURE ENGINEERING
# =============================================================================


# --- Data Cleaning (rule-based, does not learn from data: safe before split) ----------
def clean_picking_waves(picking_waves_df, storage_locations_df):
    """Preprocessing - Data Cleaning: aggregate unit rows into order lines and flag
    non-rack locations (staging/consolidation zones such as RC-01, ZN-COR, EX-xx)."""
    rack_locations = set(storage_locations_df["location"])
    lines_df = (picking_waves_df
                .groupby(["wave_number", "reference", "size_us", "location", "operator"], as_index=False)
                .agg(units=("quantity", "sum")))
    lines_df["is_rack_location"] = lines_df["location"].isin(rack_locations)
    print(f"[Data Cleaning] {len(picking_waves_df):,} unit rows -> {len(lines_df):,} order lines; "
          f"{(~lines_df['is_rack_location']).mean():.1%} lines at non-rack locations")
    return lines_df


def build_wave_calendar(customer_orders_df, lines_df):
    """Preprocessing - Data Cleaning: wave release time = first order creation in the wave."""
    calendar_df = (customer_orders_df.groupby("wave_number", as_index=False)
                   .agg(wave_time=("creation_date", "min"), num_orders=("order_number", "nunique")))
    waves_df = pd.DataFrame({"wave_number": sorted(lines_df["wave_number"].unique())})
    waves_df = waves_df.merge(calendar_df, on="wave_number", how="left")
    # waves without matching orders keep their position in the WMS wave sequence
    waves_df["wave_time"] = waves_df["wave_time"].ffill().bfill()
    waves_df = waves_df.sort_values(["wave_time", "wave_number"]).reset_index(drop=True)
    waves_df["wave_rank"] = np.arange(len(waves_df))
    return waves_df


def _allocate_slots(lines_df, stock_df, storage_locations_df, policy_name):
    """Assign each order line to one storage slot holding the item under a storage policy.

    Stock allocation in a WMS is normally driven by lot/FIFO rules rather than travel
    distance, so the slot is drawn uniformly (seeded) among picking-level slots that hold
    the same reference and size; reference-only and any-level matches are fall-backs.
    """
    rng = np.random.default_rng(config.RANDOM_STATE + config.STORAGE_POLICIES.index(policy_name))
    levels = storage_locations_df.set_index("location")["level"]
    stock_df = stock_df[stock_df["location"].isin(levels.index)].copy()
    stock_df["level"] = stock_df["location"].map(levels)
    picking_stock = stock_df[stock_df["level"].isin(config.PICKING_LEVELS)]

    by_ref_size = picking_stock.groupby(["reference", "size_us"])["location"].apply(list).to_dict()
    by_ref = picking_stock.groupby("reference")["location"].apply(list).to_dict()
    by_ref_any = stock_df.groupby("reference")["location"].apply(list).to_dict()

    allocated, match_type = [], []
    for reference, size_us in zip(lines_df["reference"], lines_df["size_us"]):
        for candidates, kind in ((by_ref_size.get((reference, float(size_us))), "ref_size"),
                                 (by_ref.get(reference), "ref"),
                                 (by_ref_any.get(reference), "ref_any_level")):
            if candidates:
                allocated.append(candidates[rng.integers(len(candidates))])
                match_type.append(kind)
                break
        else:
            allocated.append(None)
            match_type.append("unmatched")
    result = lines_df.copy()
    result["location"] = allocated
    result["match_type"] = match_type
    return result


def build_scenario_lines(lines_df, storage_policies, storage_locations_df):
    """Preprocessing - build order lines with locations for every storage configuration."""
    frames = []
    recorded = lines_df[lines_df["is_rack_location"]].copy()
    recorded["match_type"] = "recorded"
    recorded["storage_policy"] = "AsRecorded"
    frames.append(recorded)
    base = lines_df.drop(columns=["location", "is_rack_location"]).drop_duplicates(
        ["wave_number", "reference", "size_us"])
    for policy_name, stock_df in storage_policies.items():
        allocated = _allocate_slots(base, stock_df, storage_locations_df, policy_name)
        allocated["storage_policy"] = policy_name
        frames.append(allocated[allocated["location"].notna()])
        print(f"[Data Cleaning] {policy_name}: "
              f"{allocated['match_type'].value_counts(normalize=True).round(3).to_dict()}")
    scenario_lines_df = pd.concat(frames, ignore_index=True)
    return scenario_lines_df.drop(columns=["is_rack_location"], errors="ignore")


def build_instances(scenario_lines_df, storage_locations_df, products_df, waves_df, layout):
    """Preprocessing - one routing instance per (storage_policy, wave)."""
    coords = storage_locations_df.set_index("location")[["x", "y"]]
    abc = products_df.set_index("reference")["abc_class"]
    df = scenario_lines_df.join(coords, on="location")
    df["abc_class"] = df["reference"].map(abc).fillna("C")
    rows = []
    for (policy, wave), group in df.groupby(["storage_policy", "wave_number"], sort=False):
        points = [layout.to_pick_point(x, y) for x, y in zip(group["x"], group["y"])]
        rows.append({
            "storage_policy": policy, "wave_number": wave,
            "operator": group["operator"].iloc[0],
            "locations": group["location"].tolist(),
            "points": points,
            "num_lines": len(group), "num_units": int(group["units"].sum()),
            "share_a": (group["abc_class"] == "A").mean(),
            "share_b": (group["abc_class"] == "B").mean(),
            "share_c": (group["abc_class"] == "C").mean(),
        })
    instances_df = pd.DataFrame(rows).merge(waves_df[["wave_number", "wave_time", "wave_rank"]],
                                            on="wave_number", how="left")
    print(f"[Core Data Structures] {len(instances_df):,} routing instances "
          f"({instances_df['wave_number'].nunique():,} waves x {instances_df['storage_policy'].nunique()} storage configs)")
    return instances_df


# --- Label generation (routing-policy labels and optimal benchmark) ----------------
def _evaluate_instance(layout, points):
    record = {}
    for policy in config.ROUTING_POLICIES:
        result = route_heuristic(layout, points, policy)
        record[f"dist_{policy}"] = result.distance_m
        record[f"time_{policy}"] = result.compute_time_s
    optimal = route_optimal(layout, points, with_waypoints=False)
    record["dist_Optimal"] = optimal.distance_m
    record["time_Optimal"] = optimal.compute_time_s
    record["optimal_proven"] = optimal.is_optimal_proven
    return record


def generate_labels(instances_df, layout, num_jobs=config.NUM_JOBS):
    """Preprocessing - route every instance with every policy and derive the label.

    The label only depends on the instance itself (no information from other instances),
    so it is computed before the split without leakage.
    """
    records = Parallel(n_jobs=num_jobs, batch_size=64)(
        delayed(_evaluate_instance)(layout, pts) for pts in instances_df["points"])
    labelled_df = pd.concat([instances_df.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    distances = labelled_df[[f"dist_{p}" for p in config.ROUTING_POLICIES]].to_numpy()
    rounded = np.round(distances, 6)
    labelled_df[config.TARGET] = [config.ROUTING_POLICIES[i] for i in rounded.argmin(axis=1)]
    labelled_df["is_tie"] = (rounded.max(axis=1) - rounded.min(axis=1)) < 1e-6
    labelled_df["cost_spread_m"] = distances.max(axis=1) - distances.min(axis=1)
    print(f"[Label Generation] label distribution: "
          f"{labelled_df[config.TARGET].value_counts(normalize=True).round(3).to_dict()}; "
          f"ties: {labelled_df['is_tie'].mean():.1%}; optimum proven: {labelled_df['optimal_proven'].mean():.1%}")
    return labelled_df


# --- Data Splitting -----------------------------------------------------------------
def chronological_cutoff(instances_df, test_fraction=config.TEST_WAVE_FRACTION):
    """Wave rank that separates the training period from the (later) test period."""
    return int(np.quantile(instances_df["wave_rank"].unique(), 1 - test_fraction))


def split_data(df, held_out_policy, cutoff_rank):
    """Preprocessing - Data Splitting: leave-one-storage-configuration-out + time split.

    train = other storage configurations, waves released before the cut-off
    test  = the held-out storage configuration, waves released after the cut-off
    """
    is_train = (df["storage_policy"] != held_out_policy) & (df["wave_rank"] < cutoff_rank)
    is_test = (df["storage_policy"] == held_out_policy) & (df["wave_rank"] >= cutoff_rank)
    train_df, test_df = df[is_train].copy(), df[is_test].copy()
    assert not set(train_df["wave_number"]) & set(test_df["wave_number"]), "wave leakage"
    return train_df, test_df
