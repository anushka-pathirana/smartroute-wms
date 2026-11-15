"""
Exploratory Data Analysis / Evaluation - shared plotting helpers.

Colours follow the entity, never its rank: each routing method keeps the same colour in
every figure of the report and the user interface.
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from smartroute import config

mpl.use("Agg")
METHOD_COLORS = {"SShape": "#2a78d6", "Return": "#eb6834", "LargestGap": "#1baf7a",
                 "RF": "#6250d6", "XGB": "#e87ba4", "PerfectSelection": "#eda100", "Optimal": "#52514e"}
METHOD_LABELS = {"SShape": "S-shape", "Return": "Return", "LargestGap": "Largest Gap",
                 "RF": "RF-selected", "XGB": "XGBoost-selected", "PerfectSelection": "Perfect policy selection",
                 "Optimal": "Optimal (exact)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQUENTIAL_CMAP = sns.light_palette("#2a78d6", as_cmap=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False, "figure.dpi": 100,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save_figure(fig, name):
    path = config.FIGURES_DIR / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --- Warehouse layout and routes -----------------------------------------------------
def plot_layout(layout, storage_locations_df, ax=None, values=None, value_label=None):
    """Draw racks, aisle centre lines, cross aisles and the depot (metres)."""
    u = config.UNIT_TO_METRES
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 11))
    racks = storage_locations_df.drop_duplicates(["x", "y"])
    if values is not None:
        sc = ax.scatter(racks["x"] * u, racks["y"] * u, c=values.reindex(racks.index).fillna(0),
                        cmap=SEQUENTIAL_CMAP, s=10, marker="s", linewidths=0, zorder=2)
        plt.colorbar(sc, ax=ax, shrink=0.5, label=value_label)
    else:
        ax.scatter(racks["x"] * u, racks["y"] * u, s=6, marker="s", color="#c3c2b7", linewidths=0, zorder=2)
    for y in layout.aisle_ys:
        ax.plot([layout.back_x["L"] * u, layout.front_x * u], [y * u, y * u], ls=":", lw=0.6, color=MUTED, zorder=1)
    for y in layout.right_aisle_ys:
        ax.plot([layout.front_x * u, layout.back_x["R"] * u], [y * u, y * u], ls=":", lw=0.6, color=MUTED, zorder=1)
    ys = np.array(layout.aisle_ys) * u
    for x, (lo, hi) in ((layout.back_x["L"], (ys.min(), ys.max())), (layout.front_x, (ys.min(), ys.max())),
                        (layout.back_x["R"], (min(layout.right_aisle_ys) * u, ys.max()))):
        ax.plot([x * u, x * u], [lo, hi], lw=1.2, color=MUTED, zorder=1)
    dx, dy = layout.depot_xy
    ax.scatter([dx * u], [dy * u], marker="s", s=90, color=INK, zorder=5)
    ax.annotate("Depot (I/O)", (dx * u, dy * u), xytext=(6, -12), textcoords="offset points", fontsize=8, color=INK)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(False)
    return ax


def plot_route(layout, storage_locations_df, route, points, ax=None, color=None, title=None):
    """Overlay one route (waypoints) and its pick points on the layout."""
    u = config.UNIT_TO_METRES
    ax = plot_layout(layout, storage_locations_df, ax=ax)
    color = color or METHOD_COLORS.get(route.policy, INK)
    wp = np.asarray(route.waypoints, dtype=float) * u
    if len(wp) > 1:
        ax.plot(wp[:, 0], wp[:, 1], color=color, lw=2, zorder=4, solid_capstyle="round")
        for (x0, y0), (x1, y1) in zip(wp[:-1], wp[1:]):
            if np.hypot(x1 - x0, y1 - y0) > 4:
                ax.annotate("", xy=((x0 + x1) / 2 + (x1 - x0) * 0.01, (y0 + y1) / 2 + (y1 - y0) * 0.01),
                            xytext=((x0 + x1) / 2, (y0 + y1) / 2),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2), zorder=4)
    if points:
        ax.scatter([p.x * u for p in points], [p.aisle_y * u for p in points], s=36, color="white",
                   edgecolors=INK, linewidths=1.2, zorder=6)
    ax.set_title(title or f"{METHOD_LABELS.get(route.policy, route.policy)}: {route.distance_m:.1f} m")
    return ax


def plot_route_comparison(layout, storage_locations_df, routes, points, recommended=None):
    """Side-by-side routes for every method on one wave."""
    keys = [k for k in ("SShape", "Return", "LargestGap", "Optimal") if k in routes]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 9))
    for ax, key in zip(np.atleast_1d(axes), keys):
        tag = "  [ML choice]" if key == recommended else ""
        plot_route(layout, storage_locations_df, routes[key], points, ax=ax,
                   title=f"{METHOD_LABELS[key]}{tag}\n{routes[key].distance_m:.1f} m")
    fig.tight_layout()
    return fig


# --- Evaluation charts ---------------------------------------------------------------
def plot_method_comparison(summary_df, title="Mean travel distance per wave on held-out test data"):
    order = ["SShape", "Return", "LargestGap", "RF", "XGB", "PerfectSelection", "Optimal"]
    df = summary_df.set_index("key").loc[[k for k in order if k in set(summary_df["key"])]]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([METHOD_LABELS[k] for k in df.index], df["mean_distance_m"],
                  color=[METHOD_COLORS[k] for k in df.index], width=0.62, edgecolor="white", linewidth=2)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.annotate(f"{row['mean_distance_m']:.1f} m\n+{row['gap_to_optimal_pct']:.1f}% vs opt",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom",
                    fontsize=8, color=INK, xytext=(0, 3), textcoords="offset points")
    ax.set_ylabel("Mean distance per wave (m)")
    ax.set_ylim(0, df["mean_distance_m"].max() * 1.18)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="x", visible=False)
    return fig


def plot_gap_distribution(per_wave_df):
    keys = ["SShape", "Return", "LargestGap", "RF", "XGB", "PerfectSelection"]
    valid = per_wave_df["Optimal"] > 0
    data = pd.DataFrame({METHOD_LABELS[k]: 100 * (per_wave_df.loc[valid, k] - per_wave_df.loc[valid, "Optimal"])
                         / per_wave_df.loc[valid, "Optimal"] for k in keys})
    long_df = data.melt(var_name="method", value_name="gap")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=long_df, x="method", y="gap", hue="method", legend=False, ax=ax, showfliers=False,
                palette={METHOD_LABELS[k]: METHOD_COLORS[k] for k in keys}, width=0.55, linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("Gap to optimal route (%)")
    ax.set_title("Per-wave gap to the exact optimum (outliers hidden)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return fig


def plot_fold_improvement(fold_summary_df):
    pivot = fold_summary_df.pivot(index="held_out", columns="key", values="improvement_vs_sshape_pct")
    keys = [k for k in ("RF", "XGB", "PerfectSelection") if k in pivot.columns]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(keys)
    x = np.arange(len(pivot))
    for i, key in enumerate(keys):
        bars = ax.bar(x + i * width - 0.4 + width / 2, pivot[key], width=width * 0.92,
                      color=METHOD_COLORS[key], label=METHOD_LABELS[key])
        for bar in bars:
            ax.annotate(f"{bar.get_height():.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=7, color=INK, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, pivot.index)
    ax.set_ylabel("Distance reduction vs always S-shape (%)")
    ax.set_xlabel("Held-out storage configuration (test fold)")
    ax.set_title("Generalisation to unseen storage configurations")
    ax.legend(loc="upper left", ncol=3)
    ax.grid(axis="x", visible=False)
    return fig


def plot_confusion_matrix(cm, title):
    labels = [METHOD_LABELS[p] for p in config.ROUTING_POLICIES]
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(cm_norm, annot=np.array([[f"{v:.2f}\n({n:,})" for v, n in zip(rv, rn)] for rv, rn in zip(cm_norm, cm)]),
                fmt="", cmap=SEQUENTIAL_CMAP, cbar=False, xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=2, linecolor="white", annot_kws={"fontsize": 8})
    ax.set_xlabel("Predicted best policy")
    ax.set_ylabel("True best policy")
    ax.set_title(title)
    return fig


def plot_feature_importance(importance_df, top_n=12):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=False)
    for ax, (model_name, df) in zip(axes, importance_df.groupby("model")):
        df = df.sort_values("permutation_regret_increase_m", ascending=True).tail(top_n)
        key = "RF" if model_name == "Random Forest" else "XGB"
        ax.barh(df["feature"], df["permutation_regret_increase_m"], xerr=df["permutation_std"],
                color=METHOD_COLORS[key], height=0.6, error_kw={"ecolor": MUTED, "lw": 0.8})
        ax.set_title(f"{model_name}")
        ax.set_xlabel("Increase in mean regret when permuted (m)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Permutation feature importance on the held-out As-Recorded test fold", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_learning_curves(curves):
    fig, ax = plt.subplots(figsize=(7, 4))
    for algo, curve_df in curves.items():
        key = algo.upper()
        ax.plot(curve_df["train_size"], curve_df["train_accuracy"], color=METHOD_COLORS[key], lw=2, ls="--",
                label=f"{METHOD_LABELS[key].split('-')[0]} train")
        ax.plot(curve_df["train_size"], curve_df["validation_accuracy"], color=METHOD_COLORS[key], lw=2,
                marker="o", ms=5, label=f"{METHOD_LABELS[key].split('-')[0]} validation")
        ax.fill_between(curve_df["train_size"], curve_df["validation_accuracy"] - curve_df["validation_std"],
                        curve_df["validation_accuracy"] + curve_df["validation_std"], color=METHOD_COLORS[key], alpha=0.12)
    ax.set_xlabel("Training waves")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning curves (GroupKFold by wave)")
    ax.legend(ncol=2)
    return fig


def plot_compute_scaling(scaling_df):
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ("SShape", "Return", "LargestGap", "ML", "Optimal"):
        if key not in scaling_df["method"].unique():
            continue
        df = scaling_df[scaling_df["method"] == key].groupby("num_picks")["time_ms"].median()
        color = METHOD_COLORS.get(key, METHOD_COLORS["RF"])
        label = "ML decision (features + prediction)" if key == "ML" else METHOD_LABELS[key]
        ax.plot(df.index, df.values, marker="o", ms=4, lw=2, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Distinct pick locations in the pick list")
    ax.set_ylabel("Median computation time (ms, log scale)")
    ax.set_title("Computation time versus pick-list size")
    ax.legend()
    return fig
