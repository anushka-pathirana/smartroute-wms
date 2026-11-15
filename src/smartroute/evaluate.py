"""
Evaluation - run once per outer fold on the held-out test set with the final models.

Classification Metrics : accuracy, macro precision / recall / F1 and the confusion matrix
                         (on decisive test waves, i.e. waves where the policies differ).
Route Metrics          : travel distance, estimated picking time, gap to the exact optimum,
                         improvement over always-S-shape, regret and computation time.
Statistical tests      : Wilcoxon signed-rank (paired, per wave) and Friedman tests.
Model Explainability   : permutation importance (regret-based) and native tree importances.
Model Comparison & Selection: comparison table and choice of the deployed model.
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
import time

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score)

from smartroute import config
from smartroute.features import FEATURE_COLUMNS
from smartroute.models import MODEL_NAMES, encode_labels
from smartroute.validation import DISTANCE_COLUMNS, RegretScorer

FIXED_METHODS = {"SShape": "S-shape (always)", "Return": "Return (always)", "LargestGap": "Largest Gap (always)"}

# =============================================================================
# EVALUATION
# =============================================================================


def estimate_picking_time_min(distance_m, num_lines):
    """Picking time = set-up + travel / speed + lines x time per line (minutes)."""
    seconds = (config.SETUP_TIME_S + distance_m / config.WALK_SPEED_MPS
               + num_lines * config.PICK_TIME_PER_LINE_S)
    return seconds / 60.0


# --- Classification Metrics -----------------------------------------------------------
def evaluate_classification(y_true, y_pred):
    """Evaluation - Classification Metrics on integer-encoded policies."""
    labels = list(range(len(config.ROUTING_POLICIES)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
    }, confusion_matrix(y_true, y_pred, labels=labels)


def predict_policies(model, test_df):
    """Predict a policy per test wave and time the single-wave inference latency."""
    X_test = test_df[FEATURE_COLUMNS]
    y_pred = model.predict(X_test).astype(int)
    sample = X_test.iloc[: min(200, len(X_test))]
    start = time.perf_counter()
    for i in range(len(sample)):
        model.predict(sample.iloc[[i]])
    latency_s = (time.perf_counter() - start) / max(len(sample), 1)
    return y_pred, latency_s


# --- Route Metrics -------------------------------------------------------------------
def build_route_results(test_df, predictions):
    """Per-wave distances of every method on the test fold (long format)."""
    table = test_df[DISTANCE_COLUMNS].to_numpy()
    per_wave = pd.DataFrame({
        "storage_policy": test_df["storage_policy"].values,
        "wave_number": test_df["wave_number"].values,
        "num_lines": test_df["num_lines"].values,
        "num_picks": test_df["num_picks"].values,
        "best_policy": test_df[config.TARGET].values,
        "is_tie": test_df["is_tie"].values,
        "Optimal": test_df["dist_Optimal"].values,
        "PerfectSelection": table.min(axis=1),
    })
    for policy in config.ROUTING_POLICIES:
        per_wave[policy] = test_df[f"dist_{policy}"].values
    for algo, y_pred in predictions.items():
        per_wave[algo.upper()] = table[np.arange(len(table)), y_pred]
        per_wave[f"pred_{algo}"] = [config.ROUTING_POLICIES[i] for i in y_pred]
    return per_wave


def summarise_methods(per_wave_df, timings):
    """Evaluation - comparison table across fixed policies, ML selectors, perfect policy selection and optimum."""
    methods = list(config.ROUTING_POLICIES) + [c for c in ("RF", "XGB") if c in per_wave_df] + ["PerfectSelection", "Optimal"]
    labels = {**FIXED_METHODS, "RF": "RF-selected policy", "XGB": "XGBoost-selected policy",
              "PerfectSelection": "Perfect policy selection", "Optimal": "Optimal route (exact TSP)"}
    optimal = per_wave_df["Optimal"].to_numpy()
    s_shape = per_wave_df["SShape"].to_numpy()
    valid = optimal > 0
    rows = []
    for method in methods:
        d = per_wave_df[method].to_numpy()
        times = estimate_picking_time_min(d, per_wave_df["num_lines"].to_numpy())
        rows.append({
            "method": labels[method], "key": method,
            "mean_distance_m": d.mean(),
            "median_distance_m": np.median(d),
            "total_distance_km": d.sum() / 1000,
            "mean_picking_time_min": times.mean(),
            "gap_to_optimal_pct": 100 * ((d[valid] - optimal[valid]) / optimal[valid]).mean(),
            "improvement_vs_sshape_pct": 100 * (s_shape.sum() - d.sum()) / s_shape.sum(),
            "mean_regret_m": (d - per_wave_df["PerfectSelection"].to_numpy()).mean(),
            "optimal_hit_rate_pct": 100 * np.isclose(d, optimal, atol=1e-6).mean(),
            "compute_time_ms": 1000 * timings.get(method, np.nan),
        })
    return pd.DataFrame(rows)


# --- Statistical tests ---------------------------------------------------------------
def run_significance_tests(per_wave_df):
    """Evaluation - paired Wilcoxon signed-rank tests and a Friedman test."""
    rows = []
    pairs = [("RF", "SShape"), ("XGB", "SShape"), ("RF", "XGB"), ("RF", "PerfectSelection"), ("RF", "Optimal")]
    for a, b in pairs:
        diff = per_wave_df[a] - per_wave_df[b]
        non_zero = diff[np.abs(diff) > 1e-9]
        if len(non_zero) < 10:
            continue
        w_stat, p_value = stats.wilcoxon(non_zero)
        ranks = stats.rankdata(np.abs(non_zero))
        r_plus = ranks[non_zero > 0].sum()
        r_minus = ranks[non_zero < 0].sum()
        rank_biserial = (r_plus - r_minus) / ranks.sum()
        rows.append({"comparison": f"{a} vs {b}", "n_non_tied": len(non_zero),
                     "mean_diff_m": diff.mean(), "median_diff_m": diff.median(),
                     "w_stat": w_stat, "p_value": p_value, "rank_biserial": rank_biserial})
    methods = ["SShape", "Return", "LargestGap", "RF", "XGB"]
    f_stat, f_p = stats.friedmanchisquare(*[per_wave_df[m] for m in methods])
    rows.append({"comparison": "Friedman: " + ", ".join(methods), "n_non_tied": len(per_wave_df),
                 "mean_diff_m": np.nan, "median_diff_m": np.nan, "w_stat": f_stat, "p_value": f_p,
                 "rank_biserial": np.nan})
    tests_df = pd.DataFrame(rows)
    print("[Statistical tests]\n" + tests_df.round(4).to_string(index=False))
    return tests_df


# --- Model Explainability --------------------------------------------------------------
def explain_model(model, test_df, algo, num_repeats=5):
    """Evaluation - Model Explainability: permutation importance on regret + native importance."""
    X_test, y_test = test_df[FEATURE_COLUMNS], encode_labels(test_df[config.TARGET])
    result = permutation_importance(model, X_test, y_test, scoring=RegretScorer(test_df),
                                    n_repeats=num_repeats, random_state=config.RANDOM_STATE,
                                    n_jobs=config.NUM_JOBS)
    native = model.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "model": MODEL_NAMES[algo],
                                  "permutation_regret_increase_m": result.importances_mean,
                                  "permutation_std": result.importances_std,
                                  "native_importance": native})
    return importance_df.sort_values("permutation_regret_increase_m", ascending=False)


# --- Model Comparison & Selection ----------------------------------------------------------
def select_final_model(summary_df):
    """Evaluation - Model Comparison & Selection: lowest pooled mean distance wins."""
    ml = summary_df[summary_df["key"].isin(["RF", "XGB"])]
    best = ml.sort_values("mean_distance_m").iloc[0]
    print(f"[Model Comparison & Selection] selected {best['method']} "
          f"({best['mean_distance_m']:.2f} m, gap to optimal {best['gap_to_optimal_pct']:.2f} %)")
    return best["key"].lower()
