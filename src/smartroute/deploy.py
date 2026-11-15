"""
Deployment & MLOps.

Model Packaging : save_model() / load_model() persist the fitted pipelines with joblib, and a
                  JSON model card records data version, split, hyperparameters and metrics.
Model Serving   : RouteRecommender is the inference service used by the WMS user interface:
                  pick list -> features -> ML policy recommendation -> picking route, with the
                  baseline heuristics and the exact optimum available for comparison.
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
import datetime as dt
import json
import time

import joblib
import numpy as np
import pandas as pd

from smartroute import config
from smartroute.features import FEATURE_COLUMNS, compute_order_features
from smartroute.models import MODEL_NAMES
from smartroute.warehouse.routing import route_heuristic, route_optimal

# =============================================================================
# DEPLOYMENT & MLOPS
# =============================================================================


# --- Model Packaging ------------------------------------------------------------------
def save_model(model, path):
    """Deployment - Model Packaging: persist a fitted pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[Model Packaging] saved {path.name}")


def load_model(path):
    """Deployment - Model Packaging: load a fitted pipeline."""
    return joblib.load(path)


def write_model_card(card):
    """Deployment - MLOps & Lifecycle: write a versioned model card."""
    card = {"created": dt.datetime.now().isoformat(timespec="seconds"), **card}
    config.MODEL_CARD_FILE.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    print(f"[MLOps & Lifecycle] model card written to {config.MODEL_CARD_FILE.name}")


# --- Model Serving --------------------------------------------------------------------
def pick_sequence(route, points, locations):
    """Order the pick locations by when the route first passes them (picker instructions)."""
    waypoints = np.asarray(route.waypoints, dtype=float)
    cumulative = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(waypoints, axis=0)).sum(axis=1))])
    stops = []
    for point, location in zip(points, locations):
        for k, (a, b) in enumerate(zip(waypoints[:-1], waypoints[1:])):
            if (min(a[0], b[0]) - 1e-6 <= point.x <= max(a[0], b[0]) + 1e-6
                    and min(a[1], b[1]) - 1e-6 <= point.aisle_y <= max(a[1], b[1]) + 1e-6):
                travelled = cumulative[k] + abs(point.x - a[0]) + abs(point.aisle_y - a[1])
                stops.append((travelled, location))
                break
    stops.sort()
    return pd.DataFrame({"stop": range(1, len(stops) + 1), "location": [s[1] for s in stops],
                         "distance_from_depot_m": [round(s[0] * config.UNIT_TO_METRES, 1) for s in stops]})


class RouteRecommender:
    """Inference service: recommend a routing policy and build the route for a pick list."""

    def __init__(self, layout, storage_locations_df, products_df, algo=None):
        self.layout = layout
        self.coords = storage_locations_df.set_index("location")[["x", "y"]]
        self.abc = products_df.set_index("reference")["abc_class"]
        card = json.loads(config.MODEL_CARD_FILE.read_text()) if config.MODEL_CARD_FILE.exists() else {}
        self.selected_algo = algo or card.get("selected_model", "rf")
        self.models = {}
        for key, path in (("rf", config.RF_MODEL_FILE), ("xgb", config.XGB_MODEL_FILE)):
            if path.exists():
                self.models[key] = load_model(path)

    def points_for(self, locations):
        known = [loc for loc in locations if loc in self.coords.index]
        return [self.layout.to_pick_point(*self.coords.loc[loc].to_numpy()) for loc in known], known

    def features_for(self, locations, references=None, num_units=None):
        points, known = self.points_for(locations)
        refs = references or []
        classes = [self.abc.get(r, "C") for r in refs] or ["C"]
        shares = tuple(float(np.mean([c == k for c in classes])) for k in "ABC")
        features = compute_order_features(self.layout, points, num_lines=len(known),
                                          num_units=num_units or len(known), abc_shares=shares)
        return pd.DataFrame([features])[FEATURE_COLUMNS], points

    def recommend(self, locations, references=None, num_units=None, algo=None, include_optimal=True):
        """Model Serving: full decision-support output for one pick list."""
        algo = algo or self.selected_algo
        start = time.perf_counter()
        X, points = self.features_for(locations, references, num_units)
        model = self.models[algo]
        y_proba = model.predict_proba(X)[0]
        policy = config.ROUTING_POLICIES[int(np.argmax(y_proba))]
        decision_time_s = time.perf_counter() - start
        routes = {p: route_heuristic(self.layout, points, p) for p in config.ROUTING_POLICIES}
        if include_optimal:
            routes["Optimal"] = route_optimal(self.layout, points)
        return {
            "algo": algo, "model_name": MODEL_NAMES[algo],
            "recommended_policy": policy,
            "probabilities": dict(zip(config.ROUTING_POLICIES, y_proba)),
            "decision_time_ms": 1000 * decision_time_s,
            "features": X.iloc[0].to_dict(),
            "routes": routes, "points": points,
        }
