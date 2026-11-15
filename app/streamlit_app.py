"""
Deployment & MLOps - Model Serving: SmartRoute ML Warehouse Management System (Streamlit).

WMS modules: User Management (role-based navigation), Order Management, Inventory
Management, ML Route Optimisation and Visualisation/Reporting.

Run:  streamlit run app/streamlit_app.py
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartroute import config, data_loading  # noqa: E402
from smartroute.deploy import RouteRecommender, pick_sequence  # noqa: E402
from smartroute.evaluate import estimate_picking_time_min  # noqa: E402
from smartroute.visualization import METHOD_COLORS, METHOD_LABELS, plot_layout, plot_route  # noqa: E402
from smartroute.warehouse.layout import build_layout  # noqa: E402

st.set_page_config(page_title="SmartRoute ML - WMS", page_icon="📦", layout="wide")

ROLES = {
    "Warehouse Manager": ["Dashboard", "Order Management", "Inventory Management", "Route Optimisation", "Model Performance"],
    "Supervisor": ["Dashboard", "Order Management", "Route Optimisation", "Model Performance"],
    "Order Picker": ["Route Optimisation"],
}


# =============================================================================
# DATA LOADING AND STRUCTURE (cached for the session)
# =============================================================================
@st.cache_resource(show_spinner="Loading warehouse data and models ...")
def load_context():
    data = data_loading.load_data()
    layout = build_layout(data["support_points"])
    instances_df = pd.read_pickle(config.INSTANCES_FILE)
    instances_df = instances_df.loc[:, ~instances_df.columns.duplicated()]
    recommender = RouteRecommender(layout, data["storage_locations"], data["products"])
    return data, layout, instances_df, recommender


def load_results():
    path = config.REPORTS_DIR / "results_summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def read_table(name):
    path = config.TABLES_DIR / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


data, layout, instances_df, recommender = load_context()
results = load_results()

# --- User Management -----------------------------------------------------------------
st.sidebar.title("📦 SmartRoute ML")
st.sidebar.caption("ML-based WMS for order-picking route optimisation")
role = st.sidebar.selectbox("Signed in as", list(ROLES))
page = st.sidebar.radio("Module", ROLES[role])
st.sidebar.divider()
st.sidebar.caption(f"Selected model: **{recommender.selected_algo.upper()}** · "
                   f"{len(instances_df):,} routing instances · dataset: Mendeley pf2w725pw3")


# =============================================================================
# DASHBOARD
# =============================================================================
if page == "Dashboard":
    st.title("Warehouse dashboard")
    recorded = instances_df[instances_df["storage_policy"] == "AsRecorded"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Picking waves", f"{recorded['wave_number'].nunique():,}")
    c2.metric("Order lines", f"{int(recorded['num_lines'].sum()):,}")
    c3.metric("Storage locations", f"{len(data['storage_locations']):,}")
    c4.metric("Operators", f"{data['picking_waves']['operator'].nunique()}")
    pooled = results.get("pooled", {})
    if pooled:
        st.subheader("Routing performance on unseen storage configurations (test folds)")
        k1, k2, k3, k4 = st.columns(4)
        sel = results.get("selected_model", "rf").upper()
        k1.metric("Mean distance - always S-shape", f"{pooled['SShape']['mean_distance_m']:.1f} m")
        k2.metric(f"Mean distance - {sel}-selected", f"{pooled[sel]['mean_distance_m']:.1f} m",
                  f"{-pooled[sel]['improvement_vs_sshape_pct']:.2f} % vs S-shape", delta_color="inverse")
        k3.metric("Gap to optimal (ML)", f"{pooled[sel]['gap_to_optimal_pct']:.1f} %",
                  f"{pooled[sel]['gap_to_optimal_pct'] - pooled['SShape']['gap_to_optimal_pct']:.1f} pp vs S-shape",
                  delta_color="inverse")
        k4.metric("ML decision time", f"{pooled[sel]['compute_time_ms']:.1f} ms")
    left, right = st.columns([1, 1])
    fig_path = config.FIGURES_DIR / "evaluation_method_comparison.png"
    if fig_path.exists():
        left.image(str(fig_path), caption="Mean distance per wave by method")
    heat = config.FIGURES_DIR / "eda_layout_pick_frequency.png"
    if heat.exists():
        right.image(str(heat), caption="Pick frequency on the layout", width=360)

# =============================================================================
# ORDER MANAGEMENT
# =============================================================================
elif page == "Order Management":
    st.title("Order management")
    orders = data["customer_orders"]
    recorded = instances_df[instances_df["storage_policy"] == "AsRecorded"][
        ["wave_number", "wave_time", "operator", "num_lines", "num_units", "num_picks", "num_aisles"]]
    c1, c2 = st.columns(2)
    operator = c1.selectbox("Operator", ["All"] + sorted(recorded["operator"].unique()))
    dates = c2.date_input("Wave release date range", (recorded["wave_time"].min().date(), recorded["wave_time"].max().date()))
    view = recorded if operator == "All" else recorded[recorded["operator"] == operator]
    if isinstance(dates, tuple) and len(dates) == 2:
        view = view[(view["wave_time"].dt.date >= dates[0]) & (view["wave_time"].dt.date <= dates[1])]
    st.caption(f"{len(view):,} picking waves")
    st.dataframe(view.sort_values("wave_time", ascending=False), width="stretch", height=300, hide_index=True)
    wave = st.selectbox("Inspect wave", view["wave_number"].head(500))
    st.subheader(f"Customer orders in wave {wave}")
    st.dataframe(orders[orders["wave_number"] == wave], width="stretch", hide_index=True)
    st.subheader("Pick lines")
    st.dataframe(data["picking_waves"][data["picking_waves"]["wave_number"] == wave]
                 .groupby(["reference", "size_us", "location"], as_index=False)["quantity"].sum(),
                 width="stretch", hide_index=True)

# =============================================================================
# INVENTORY MANAGEMENT
# =============================================================================
elif page == "Inventory Management":
    st.title("Inventory management")
    policy = st.selectbox("Storage configuration", config.STORAGE_POLICIES)
    stock = data["storage_policies"][policy]
    products = data["products"]
    c1, c2 = st.columns([1, 2])
    reference = c1.selectbox("Product reference", sorted(products["reference"]))
    holding = stock[stock["reference"] == reference].merge(data["storage_locations"], on="location")
    c1.metric("Slots holding product", len(holding))
    c1.metric("ABC class", products.set_index("reference").loc[reference, "abc_class"])
    c1.dataframe(holding.groupby(["location", "size_us"], as_index=False).size().rename(columns={"size": "slots"}),
                 hide_index=True, height=320)
    fig, ax = plt.subplots(figsize=(5.5, 9))
    plot_layout(layout, data["storage_locations"], ax=ax)
    u = config.UNIT_TO_METRES
    ax.scatter(holding["x"] * u, holding["y"] * u, s=40, color=METHOD_COLORS["Return"], zorder=6, label=reference)
    ax.set_title(f"{reference} under {policy} storage")
    ax.legend(loc="lower left")
    c2.pyplot(fig, width="content")

# =============================================================================
# ROUTE OPTIMISATION (core ML module)
# =============================================================================
elif page == "Route Optimisation":
    st.title("ML route optimisation")
    c1, c2, c3 = st.columns(3)
    storage_policy = c1.selectbox("Storage configuration", ["AsRecorded"] + config.STORAGE_POLICIES)
    subset = instances_df[instances_df["storage_policy"] == storage_policy]
    mode = c2.radio("Pick list source", ["Existing wave", "Custom locations"], horizontal=True)
    algo = c3.selectbox("Model", list(recommender.models), format_func=lambda a: {"rf": "Random Forest", "xgb": "XGBoost"}[a],
                        index=list(recommender.models).index(recommender.selected_algo))
    if mode == "Existing wave":
        example = results.get("example_wave", {}).get("wave_number")
        default = int(example) if example in set(subset["wave_number"]) else int(
            subset.loc[subset["num_picks"].between(12, 16), "wave_number"].iloc[0])
        wave = st.number_input("Wave number", value=default, step=1)
        row = subset[subset["wave_number"] == wave]
        if row.empty:
            st.warning("Wave not found for this storage configuration.")
            st.stop()
        row = row.iloc[0]
        locations = row["locations"]
        lines = data["picking_waves"][data["picking_waves"]["wave_number"] == wave]
        if storage_policy == "AsRecorded":
            lines = lines[lines["location"].isin(recommender.coords.index)]
            line_keys = ["reference", "size_us", "location"]
        else:
            line_keys = ["reference", "size_us"]
        references = lines.drop_duplicates(line_keys)["reference"].tolist()
        num_units = int(row["num_units"])
    else:
        sample = ", ".join(data["storage_locations"].sample(8, random_state=1)["location"])
        text = st.text_area("Storage location codes (comma separated)", sample)
        locations = [s.strip() for s in text.split(",") if s.strip()]
        unknown = [loc for loc in locations if loc not in recommender.coords.index]
        if unknown:
            st.warning(f"Ignored {len(unknown)} unknown location code(s): {', '.join(unknown)}")
        if len(unknown) == len(locations):
            st.error("Enter at least one valid storage location code.")
            st.stop()
        references, num_units = None, None
    show_optimal = st.toggle("Compute exact optimal benchmark", value=True)
    output = recommender.recommend(locations, references, num_units, algo=algo, include_optimal=show_optimal)
    routes, points = output["routes"], output["points"]
    policy = output["recommended_policy"]
    chosen = routes[policy]

    st.success(f"**{output['model_name']} recommends {METHOD_LABELS[policy]}** for this pick list "
               f"({len(points)} stops, decision in {output['decision_time_ms']:.1f} ms).")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recommended route", f"{chosen.distance_m:.1f} m")
    m2.metric("vs always S-shape", f"{routes['SShape'].distance_m - chosen.distance_m:.1f} m saved")
    if "Optimal" in routes:
        gap = 100 * (chosen.distance_m - routes["Optimal"].distance_m) / max(routes["Optimal"].distance_m, 1e-9)
        m3.metric("Gap to optimal", f"{gap:.1f} %")
    m4.metric("Est. picking time", f"{estimate_picking_time_min(chosen.distance_m, len(locations)):.1f} min")

    tab_route, tab_compare, tab_why = st.tabs(["Picker route", "Compare methods", "Why this policy?"])
    with tab_route:
        left, right = st.columns([3, 2])
        fig, ax = plt.subplots(figsize=(6, 10))
        plot_route(layout, data["storage_locations"], chosen, points, ax=ax)
        left.pyplot(fig, width="content")
        known = [loc for loc in locations if loc in recommender.coords.index]
        right.subheader("Picking instructions")
        right.dataframe(pick_sequence(chosen, points, known), hide_index=True, height=520)
    with tab_compare:
        rows = []
        for key, route in routes.items():
            rows.append({"method": METHOD_LABELS[key] + ("  ◀ ML" if key == policy else ""),
                         "distance_m": round(route.distance_m, 1),
                         "picking_time_min": round(estimate_picking_time_min(route.distance_m, len(locations)), 2),
                         "gap_to_optimal_pct": round(100 * (route.distance_m - routes["Optimal"].distance_m)
                                                     / max(routes["Optimal"].distance_m, 1e-9), 1) if "Optimal" in routes else None,
                         "compute_time_ms": round(1000 * route.compute_time_s, 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        keys = list(routes)
        fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 8.5))
        for ax, key in zip(axes, keys):
            plot_route(layout, data["storage_locations"], routes[key], points, ax=ax,
                       title=f"{METHOD_LABELS[key]}{' [ML]' if key == policy else ''}\n{routes[key].distance_m:.1f} m")
        fig.tight_layout()
        st.pyplot(fig)
    with tab_why:
        proba = pd.DataFrame({"policy": [METHOD_LABELS[p] for p in output["probabilities"]],
                              "probability": list(output["probabilities"].values())})
        st.bar_chart(proba, x="policy", y="probability", color="#6250d6")
        st.caption("Order features used by the model (computed from the pick list only):")
        st.dataframe(pd.DataFrame([output["features"]]).T.rename(columns={0: "value"}).round(3), width="stretch")

# =============================================================================
# MODEL PERFORMANCE (Visualisation / Reporting)
# =============================================================================
elif page == "Model Performance":
    st.title("Model performance and evaluation")
    st.caption("Leave-one-storage-configuration-out evaluation on the most recent 20 % of waves.")
    st.subheader("Method comparison (pooled test folds)")
    st.dataframe(read_table("evaluation_method_comparison").drop(columns=["key"]).round(2), hide_index=True,
                 width="stretch")
    st.subheader("Classification metrics")
    st.dataframe(read_table("evaluation_classification_metrics").round(3), hide_index=True, width="stretch")
    figures = [("evaluation_fold_improvement", "Improvement per held-out storage configuration"),
               ("evaluation_gap_distribution", "Gap to optimal"),
               ("evaluation_confusion_rf", "Random Forest confusion matrix"),
               ("evaluation_confusion_xgb", "XGBoost confusion matrix"),
               ("evaluation_feature_importance", "Permutation importance"),
               ("evaluation_computation_scaling", "Computation time")]
    cols = st.columns(2)
    for i, (name, caption) in enumerate(figures):
        path = config.FIGURES_DIR / f"{name}.png"
        if path.exists():
            cols[i % 2].image(str(path), caption=caption)
    st.subheader("Statistical significance")
    st.dataframe(read_table("evaluation_significance_tests"), hide_index=True, width="stretch")
