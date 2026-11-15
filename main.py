"""
SmartRoute ML - Machine Learning-Based WMS for intelligent order-picking route optimisation.

Runs the AI Engineering stages in order:
Problem Definition & Environment Setup -> Data Loading and Structure -> Exploratory Data
Analysis -> Preprocessing and Feature Engineering -> Model Training -> Testing and
Validation -> Evaluation -> Deployment & MLOps.

Usage:
    python main.py              # full pipeline (uses cached routing instances if present)
    python main.py --rebuild    # recompute routing instances, labels and features
    python main.py --quick      # fewer tuning iterations (smoke test)
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from smartroute import config  # noqa: E402
from smartroute import data_loading, eda, evaluate, features, preprocessing, validation, visualization  # noqa: E402
from smartroute.deploy import save_model, write_model_card  # noqa: E402
from smartroute.features import FEATURE_COLUMNS, compute_order_features  # noqa: E402
from smartroute.models import MODEL_NAMES, build_model, compute_sample_weights, encode_labels, train_model  # noqa: E402
from smartroute.warehouse.layout import build_layout, layout_summary  # noqa: E402
from smartroute.warehouse.routing import route_heuristic, route_optimal  # noqa: E402

ALGOS = ["rf", "xgb"]


def save_table(df, name):
    df.to_csv(config.TABLES_DIR / f"{name}.csv", index=False)


def tuned_params(model, algo):
    """Hyperparameters of a tuned pipeline restricted to the searched keys."""
    search_space = config.RF_PARAM_DIST if algo == "rf" else config.XGB_PARAM_DIST
    return {k: v for k, v in model.get_params().items() if k in search_space}


def make_example_route(layout, storage_locations_df, instances_df, per_wave_df, selected):
    """Evaluation - illustrative route comparison for one as-recorded test wave.

    The wave is the median-saving case among test waves (12-18 locations) where the selected model
    correctly departs from S-shape, so the example shows a typical rather than the best case.
    """
    pred = f"pred_{selected}"
    candidates = per_wave_df[(per_wave_df["storage_policy"] == "AsRecorded") & (per_wave_df["num_picks"].between(12, 18))
                             & (per_wave_df[pred] == per_wave_df["best_policy"]) & (per_wave_df[pred] != "SShape")]
    candidates = candidates.assign(saving=candidates["SShape"] - candidates[selected.upper()]).sort_values("saving")
    chosen = candidates.iloc[len(candidates) // 2]
    instance = instances_df[(instances_df["storage_policy"] == "AsRecorded")
                            & (instances_df["wave_number"] == chosen["wave_number"])].iloc[0]
    routes = {p: route_heuristic(layout, instance["points"], p) for p in config.ROUTING_POLICIES}
    routes["Optimal"] = route_optimal(layout, instance["points"])
    recommended = chosen[pred]
    fig = visualization.plot_route_comparison(layout, storage_locations_df, routes, instance["points"], recommended)
    fig.suptitle(f"Wave {int(chosen['wave_number'])} ({int(instance['num_picks'])} pick locations) - ML recommends "
                 f"{visualization.METHOD_LABELS[recommended]}", fontweight="bold", y=1.01)
    visualization.save_figure(fig, "example_route_comparison")
    return {"wave_number": int(chosen["wave_number"]), "recommended": recommended,
            "candidates": len(candidates), "median_saving_m": round(float(chosen["saving"]), 2),
            "distances_m": {k: round(v.distance_m, 2) for k, v in routes.items()}}


def measure_computation_scaling(layout, instances_df, model, rng, max_batch=4, samples_per_size=2):
    """Evaluation - computational efficiency versus pick-list size (incl. batched waves)."""
    recorded = instances_df[instances_df["storage_policy"] == "AsRecorded"]
    rows = []
    waves = recorded.sample(n=min(400, len(recorded)), random_state=config.RANDOM_STATE)
    pick_lists = [list(p) for p in waves["points"]]
    for batch in range(2, max_batch + 1):        # batch several waves into one larger pick list
        for _ in range(samples_per_size * 6):
            chosen = recorded.sample(n=batch, random_state=int(rng.integers(1_000_000)))
            pick_lists.append([p for pts in chosen["points"] for p in pts])
    for points in pick_lists:
        n = len({(p.block, p.aisle_y, p.x) for p in points})
        if n == 0:
            continue
        for policy in config.ROUTING_POLICIES:
            rows.append({"method": policy, "num_picks": n, "time_ms": 1000 * route_heuristic(layout, points, policy).compute_time_s})
        start = time.perf_counter()
        X = pd.DataFrame([compute_order_features(layout, points)])[FEATURE_COLUMNS]
        model.predict(X)
        rows.append({"method": "ML", "num_picks": n, "time_ms": 1000 * (time.perf_counter() - start)})
        optimal = route_optimal(layout, points, time_limit=config.OPTIMAL_TIME_LIMIT_S, with_waypoints=False)
        rows.append({"method": "Optimal", "num_picks": n, "time_ms": 1000 * optimal.compute_time_s,
                     "proven": optimal.is_optimal_proven})
    scaling_df = pd.DataFrame(rows)
    scaling_df["size_band"] = pd.cut(scaling_df["num_picks"], [0, 5, 10, 15, 20, 25, 40, 60, 120],
                                     labels=["1-5", "6-10", "11-15", "16-20", "21-25", "26-40", "41-60", "61+"])
    return scaling_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        config.NUM_SEARCH_ITERATIONS = 3
    config.ensure_directories()
    t_start = time.time()
    results = {}

    # =========================================================================
    # DATA LOADING AND STRUCTURE
    # =========================================================================
    data = data_loading.load_data()
    save_table(data_loading.inspect_data(data), "data_inspection")
    layout = build_layout(data["support_points"])
    save_table(layout_summary(layout), "layout_summary")

    # =========================================================================
    # PREPROCESSING AND FEATURE ENGINEERING (instance construction & labels)
    # =========================================================================
    lines_df = preprocessing.clean_picking_waves(data["picking_waves"], data["storage_locations"])
    if config.INSTANCES_FILE.exists() and not args.rebuild:
        instances_df = pd.read_pickle(config.INSTANCES_FILE)
        print(f"[Core Data Structures] loaded cached instances: {len(instances_df):,}")
    else:
        waves_df = preprocessing.build_wave_calendar(data["customer_orders"], lines_df)
        scenario_lines_df = preprocessing.build_scenario_lines(lines_df, data["storage_policies"], data["storage_locations"])
        instances_df = preprocessing.build_instances(scenario_lines_df, data["storage_locations"], data["products"],
                                                     waves_df, layout)
        instances_df = preprocessing.generate_labels(instances_df, layout)
        instances_df = pd.concat([instances_df, features.engineer_features(instances_df, layout)], axis=1)
        instances_df.to_pickle(config.INSTANCES_FILE)
    instances_df = instances_df.loc[:, ~instances_df.columns.duplicated()]
    instances_df[["storage_policy", "wave_number", "wave_time", "num_lines"] + FEATURE_COLUMNS
                 + [f"dist_{p}" for p in config.ROUTING_POLICIES] + ["dist_Optimal", config.TARGET, "is_tie"]
                 ].to_csv(config.FEATURES_FILE, index=False)

    # =========================================================================
    # EXPLORATORY DATA ANALYSIS (EDA)
    # =========================================================================
    if config.RUN_EDA:
        eda_results = eda.run_eda(data, lines_df, instances_df, layout)
        save_table(eda_results["wave_stats"], "eda_wave_statistics")
        results["label_test"] = eda_results["label_test"]
        save_table(pd.crosstab(instances_df["storage_policy"], instances_df[config.TARGET]).reset_index(),
                   "eda_label_counts")

    # --- Data Splitting -----------------------------------------------------------
    cutoff_rank = preprocessing.chronological_cutoff(instances_df)
    cutoff_time = instances_df.loc[instances_df["wave_rank"] == cutoff_rank, "wave_time"].min()
    results["split"] = {"cutoff_rank": cutoff_rank, "cutoff_time": str(cutoff_time),
                        "test_wave_fraction": config.TEST_WAVE_FRACTION}
    folds = {}
    for held_out in preprocessing.ALL_STORAGE_CONFIGS:
        train_df, test_df = preprocessing.split_data(instances_df, held_out, cutoff_rank)
        folds[held_out] = (train_df[~train_df["is_tie"]], test_df)
    save_table(pd.DataFrame([{"held_out": k, "train_instances": len(v[0]), "test_instances": len(v[1]),
                              "test_decisive": int((~v[1]["is_tie"]).sum())} for k, v in folds.items()]), "split_sizes")
    print(f"[Data Splitting] cut-off {cutoff_time}; folds: "
          + ", ".join(f"{k}: {len(v[0]):,}/{len(v[1]):,}" for k, v in folds.items()))

    # =========================================================================
    # MODEL TRAINING + TESTING AND VALIDATION (nested: tuning inside each outer fold)
    # =========================================================================
    primary_train, primary_test = folds["AsRecorded"]
    save_table(validation.compare_weighting(primary_train), "validation_weighting_ablation")
    fitted, tuning_rows = {}, []
    for held_out, (train_df, _) in folds.items():
        for algo in ALGOS:
            best_model, best_params, best_score, cv_results = validation.tune_model(algo, train_df)
            fitted[(held_out, algo)] = best_model
            tuning_rows.append({"held_out": held_out, "algo": algo, "cv_regret_m": best_score,
                                **{k.replace("model__", ""): v for k, v in best_params.items()}})
            if held_out == "AsRecorded":
                save_table(cv_results[["params", "mean_test_score", "std_test_score", "mean_train_score"]]
                           .assign(algo=algo), f"tuning_results_{algo}")
    save_table(pd.DataFrame(tuning_rows), "tuning_best_params")

    curves = {algo: validation.diagnose_learning_curve(algo, tuned_params(fitted[("AsRecorded", algo)], algo),
                                                       primary_train)
              for algo in ALGOS}
    for algo, curve_df in curves.items():
        save_table(curve_df.assign(algo=algo), f"learning_curve_{algo}")
    visualization.save_figure(visualization.plot_learning_curves(curves), "validation_learning_curves")

    # =========================================================================
    # EVALUATION (each outer test fold used once)
    # =========================================================================
    per_wave_frames, fold_summaries, class_rows, cms = [], [], [], {a: 0 for a in ALGOS}
    feature_time_s = []
    for held_out, (train_df, test_df) in folds.items():
        predictions, timings = {}, {p: test_df[f"time_{p}"].mean() for p in config.ROUTING_POLICIES}
        timings["Optimal"] = test_df["time_Optimal"].mean()
        timings["PerfectSelection"] = sum(timings[p] for p in config.ROUTING_POLICIES)
        sample = test_df.sample(n=min(150, len(test_df)), random_state=config.RANDOM_STATE)
        start = time.perf_counter()
        for points in sample["points"]:
            compute_order_features(layout, points)
        feature_time_s.append((time.perf_counter() - start) / len(sample))
        for algo in ALGOS:
            model = fitted[(held_out, algo)]
            y_pred, latency = evaluate.predict_policies(model, test_df)
            predictions[algo] = y_pred
            timings[algo.upper()] = feature_time_s[-1] + latency + np.mean(
                [test_df[f"time_{p}"].mean() for p in config.ROUTING_POLICIES])
            decisive = ~test_df["is_tie"].to_numpy()
            y_true = encode_labels(test_df[config.TARGET]).to_numpy()
            metrics, cm = evaluate.evaluate_classification(y_true[decisive], y_pred[decisive])
            train_X, train_y = train_df[FEATURE_COLUMNS], encode_labels(train_df[config.TARGET])
            metrics["train_accuracy"] = (model.predict(train_X) == train_y).mean()
            class_rows.append({"held_out": held_out, "algo": algo, "model": MODEL_NAMES[algo], **metrics})
            cms[algo] = cms[algo] + cm
        per_wave_df = evaluate.build_route_results(test_df, predictions)
        per_wave_frames.append(per_wave_df)
        fold_summary = evaluate.summarise_methods(per_wave_df, timings)
        fold_summaries.append(fold_summary.assign(held_out=held_out))

    per_wave_all = pd.concat(per_wave_frames, ignore_index=True)
    per_wave_all.to_csv(config.TABLES_DIR / "evaluation_per_wave.csv", index=False)
    fold_summary_df = pd.concat(fold_summaries, ignore_index=True)
    save_table(fold_summary_df, "evaluation_fold_summary")
    pooled_timings = fold_summary_df.groupby("key")["compute_time_ms"].mean() / 1000
    pooled_summary = evaluate.summarise_methods(per_wave_all, pooled_timings.to_dict())
    save_table(pooled_summary, "evaluation_method_comparison")
    print("[Evaluation] pooled comparison\n" + pooled_summary.round(2).to_string(index=False))
    classification_df = pd.DataFrame(class_rows)
    save_table(classification_df, "evaluation_classification_metrics")
    pooled_class = classification_df.groupby("model")[["accuracy", "precision_macro", "recall_macro", "f1_macro",
                                                        "f1_weighted", "train_accuracy"]].mean().reset_index()
    save_table(pooled_class, "evaluation_classification_pooled")
    for algo in ALGOS:
        save_table(pd.DataFrame(cms[algo], index=config.ROUTING_POLICIES, columns=config.ROUTING_POLICIES)
                   .reset_index(), f"confusion_matrix_{algo}")
        visualization.save_figure(visualization.plot_confusion_matrix(
            cms[algo], f"{MODEL_NAMES[algo]} (all test folds)"), f"evaluation_confusion_{algo}")

    tests_df = evaluate.run_significance_tests(per_wave_all)
    save_table(tests_df, "evaluation_significance_tests")
    by_size = per_wave_all.assign(size_band=pd.cut(per_wave_all["num_picks"], [0, 5, 10, 15, 20, 30],
                                                   labels=["1-5", "6-10", "11-15", "16-20", "21-25"]))
    size_table = by_size.groupby("size_band", observed=True)[["SShape", "Return", "LargestGap", "RF", "XGB",
                                                              "PerfectSelection", "Optimal"]].mean().reset_index()
    size_table["waves"] = by_size.groupby("size_band", observed=True).size().values
    save_table(size_table, "evaluation_by_pick_list_size")

    visualization.save_figure(visualization.plot_method_comparison(pooled_summary), "evaluation_method_comparison")
    visualization.save_figure(visualization.plot_gap_distribution(per_wave_all), "evaluation_gap_distribution")
    visualization.save_figure(visualization.plot_fold_improvement(fold_summary_df), "evaluation_fold_improvement")

    # --- Model Explainability ---------------------------------------------------------
    importance_df = pd.concat([evaluate.explain_model(fitted[("AsRecorded", algo)], primary_test.reset_index(drop=True), algo)
                               for algo in ALGOS], ignore_index=True)
    save_table(importance_df, "evaluation_feature_importance")
    visualization.save_figure(visualization.plot_feature_importance(importance_df), "evaluation_feature_importance")

    # --- Model Comparison & Selection -----------------------------------------------------
    selected = evaluate.select_final_model(pooled_summary)

    # --- Computational efficiency ----------------------------------------------------------
    rng = np.random.default_rng(config.RANDOM_STATE)
    scaling_df = measure_computation_scaling(layout, instances_df, fitted[("AsRecorded", selected)], rng)
    save_table(scaling_df, "evaluation_computation_scaling")
    scaling_table = scaling_df.groupby(["size_band", "method"], observed=True)["time_ms"].agg(["median", "mean", "max", "count"]).reset_index()
    save_table(scaling_table, "evaluation_computation_scaling_summary")
    visualization.save_figure(visualization.plot_compute_scaling(scaling_df), "evaluation_computation_scaling")

    # =========================================================================
    # DEPLOYMENT & MLOPS
    # =========================================================================
    # --- Model Packaging: production models refitted on all decisive data ------------------
    production_df = instances_df[~instances_df["is_tie"]]
    for algo, path in (("rf", config.RF_MODEL_FILE), ("xgb", config.XGB_MODEL_FILE)):
        model = build_model(algo, **tuned_params(fitted[("AsRecorded", algo)], algo))
        train_model(model, production_df[FEATURE_COLUMNS], encode_labels(production_df[config.TARGET]),
                    sample_weight=compute_sample_weights(production_df))
        save_model(model, path)

    # --- Example routes for the report and the UI --------------------------------------------
    results["example_wave"] = make_example_route(layout, data["storage_locations"], instances_df, per_wave_all, selected)

    # --- MLOps & Lifecycle: model card + results summary -------------------------------------
    row = pooled_summary.set_index("key")
    results.update({
        "selected_model": selected,
        "num_instances": len(instances_df), "num_waves": int(instances_df["wave_number"].nunique()),
        "tie_share": float(instances_df["is_tie"].mean()),
        "label_share": instances_df[config.TARGET].value_counts(normalize=True).to_dict(),
        "pooled": row[["mean_distance_m", "gap_to_optimal_pct", "improvement_vs_sshape_pct", "mean_regret_m",
                       "compute_time_ms", "mean_picking_time_min"]].round(4).to_dict(orient="index"),
        "classification": pooled_class.round(4).to_dict(orient="records"),
        "runtime_minutes": round((time.time() - t_start) / 60, 1),
    })
    (config.REPORTS_DIR / "results_summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    write_model_card({
        "selected_model": selected, "models": {k: str(v.name) for k, v in
                                               (("rf", config.RF_MODEL_FILE), ("xgb", config.XGB_MODEL_FILE))},
        "task": "multiclass routing-policy selection", "classes": config.ROUTING_POLICIES,
        "features": FEATURE_COLUMNS, "training_rows": len(production_df),
        "dataset": "Mendeley pf2w725pw3 v1 (de Assis, de Paula Ferreira and Ouhimmou, 2025)",
        "validation": "leave-one-storage-configuration-out x chronological 80/20 wave split; GroupKFold(wave) tuning",
        "test_metrics": results["pooled"],
    })
    print(f"[Done] pipeline finished in {results['runtime_minutes']} min")


if __name__ == "__main__":
    main()
