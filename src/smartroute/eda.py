"""
Exploratory Data Analysis (EDA).

Statistical Analysis: wave-size statistics, skewness/kurtosis, label distribution per
                      storage configuration, chi-square test of label vs configuration,
                      IQR outliers and feature correlations.
Data Visualization  : layout map, wave-size distributions, label mix, distance by policy,
                      correlation heatmap.
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.stats import chi2_contingency, kurtosis, skew

from smartroute import config
from smartroute.features import FEATURE_COLUMNS
from smartroute.visualization import (INK, METHOD_COLORS, METHOD_LABELS, MUTED, SEQUENTIAL_CMAP,
                                      plot_layout, save_figure)

# =============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================


# --- Statistical Analysis -------------------------------------------------------------
def describe_waves(lines_df, instances_df):
    """EDA - Statistical Analysis: central tendency, dispersion and shape of wave sizes."""
    recorded = instances_df[instances_df["storage_policy"] == "AsRecorded"]
    rows = []
    for column, label in (("num_lines", "Order lines per wave"), ("num_units", "Units per wave"),
                          ("num_picks", "Distinct pick locations per wave"), ("num_aisles", "Aisles visited per wave")):
        s = recorded[column]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        rows.append({"variable": label, "mean": s.mean(), "median": s.median(), "std": s.std(),
                     "min": s.min(), "max": s.max(), "skewness": skew(s), "kurtosis": kurtosis(s),
                     "iqr_outliers": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())})
    stats_df = pd.DataFrame(rows)
    print("[Statistical Analysis]\n" + stats_df.round(2).to_string(index=False))
    return stats_df


def test_label_independence(instances_df):
    """EDA - Statistical Analysis: chi-square test, best policy vs storage configuration."""
    table = pd.crosstab(instances_df["storage_policy"], instances_df[config.TARGET])
    chi2, p_value, dof, _ = chi2_contingency(table)
    cramers_v = np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))
    print(f"[Statistical Analysis] chi2={chi2:.1f}, dof={dof}, p={p_value:.2e}, Cramer's V={cramers_v:.3f}")
    return {"chi2": chi2, "dof": dof, "p_value": p_value, "cramers_v": cramers_v}


# --- Data Visualization ---------------------------------------------------------------
def run_eda(data, lines_df, instances_df, layout):
    """EDA - produce statistics tables and figures; returns a dict of results."""
    results = {"wave_stats": describe_waves(lines_df, instances_df),
               "label_test": test_label_independence(instances_df)}
    storage_locations_df = data["storage_locations"]

    # pick frequency heat map on the layout (as recorded)
    pick_counts = lines_df[lines_df["is_rack_location"]].merge(storage_locations_df, on="location")
    by_xy = pick_counts.groupby(["x", "y"]).size()
    racks = storage_locations_df.drop_duplicates(["x", "y"]).set_index(["x", "y"])
    values = pd.Series(by_xy.reindex(racks.index).fillna(0).to_numpy(),
                       index=storage_locations_df.drop_duplicates(["x", "y"]).index)
    fig, ax = plt.subplots(figsize=(6.5, 11))
    plot_layout(layout, storage_locations_df, ax=ax, values=values, value_label="Order lines picked")
    ax.set_title("Warehouse layout and pick frequency (as recorded)")
    save_figure(fig, "eda_layout_pick_frequency")

    # wave-size distributions
    recorded = instances_df[instances_df["storage_policy"] == "AsRecorded"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, column, label in zip(axes, ("num_lines", "num_picks", "num_aisles"),
                                 ("Order lines per wave", "Distinct pick locations", "Aisles visited")):
        ax.hist(recorded[column], bins=range(int(recorded[column].max()) + 2), color=METHOD_COLORS["SShape"],
                edgecolor="white", linewidth=1)
        ax.axvline(recorded[column].median(), color=INK, lw=1.2, ls="--")
        ax.set_title(label)
        ax.set_ylabel("Waves")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Pick-list structure of the 9,707 picking waves", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "eda_wave_size_distributions")

    # waves released per month
    monthly = recorded.set_index("wave_time").resample("MS").size()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(monthly.index.strftime("%b %Y"), monthly.values, color=METHOD_COLORS["SShape"], width=0.6)
    ax.set_ylabel("Waves released")
    ax.set_title("Picking waves per month (2023)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="x", visible=False)
    save_figure(fig, "eda_waves_per_month")

    # label distribution by storage configuration (100% stacked)
    share = pd.crosstab(instances_df["storage_policy"], instances_df[config.TARGET], normalize="index")
    share = share[config.ROUTING_POLICIES].loc[["AsRecorded"] + config.STORAGE_POLICIES]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    left = np.zeros(len(share))
    for policy in config.ROUTING_POLICIES:
        ax.barh(share.index, share[policy], left=left, color=METHOD_COLORS[policy],
                label=METHOD_LABELS[policy], edgecolor="white", linewidth=2, height=0.62)
        for i, v in enumerate(share[policy]):
            if v > 0.06:
                ax.text(left[i] + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
        left += share[policy].to_numpy()
    ax.set_xlabel("Share of waves for which the policy is shortest")
    ax.set_title("Best routing policy (label) by storage configuration")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(False)
    save_figure(fig, "eda_label_distribution")

    # distance by routing policy and storage configuration
    long_df = instances_df.melt(id_vars=["storage_policy"],
                                value_vars=[f"dist_{p}" for p in config.ROUTING_POLICIES] + ["dist_Optimal"],
                                var_name="method", value_name="distance_m")
    long_df["method"] = long_df["method"].str.replace("dist_", "").map(METHOD_LABELS)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=long_df, x="storage_policy", y="distance_m", hue="method", ax=ax, errorbar=None,
                palette={METHOD_LABELS[k]: METHOD_COLORS[k] for k in ("SShape", "Return", "LargestGap", "Optimal")},
                order=["AsRecorded"] + config.STORAGE_POLICIES, edgecolor="white", linewidth=1.5)
    ax.set_xlabel("Storage configuration")
    ax.set_ylabel("Mean route distance per wave (m)")
    ax.set_title("Route distance by routing method and storage configuration (all waves)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="x", visible=False)
    save_figure(fig, "eda_distance_by_policy")

    # feature correlation heat map
    correlation_matrix = instances_df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(correlation_matrix, cmap=sns.diverging_palette(250, 20, as_cmap=True), center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.6, "label": "Pearson r"}, ax=ax)
    ax.set_title("Feature correlation matrix")
    save_figure(fig, "eda_feature_correlation")
    results["correlation_matrix"] = correlation_matrix

    # how much does the choice matter? distribution of cost spread
    fig, ax = plt.subplots(figsize=(7, 3.4))
    decisive = instances_df.loc[~instances_df["is_tie"], "cost_spread_m"]
    ax.hist(decisive, bins=60, color=METHOD_COLORS["RF"], edgecolor="white", linewidth=0.8)
    ax.axvline(decisive.median(), color=INK, ls="--", lw=1.2)
    ax.annotate(f"median {decisive.median():.1f} m", (decisive.median(), ax.get_ylim()[1] * 0.9),
                xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("Worst minus best heuristic distance (m)")
    ax.set_ylabel("Waves")
    ax.set_title("What is at stake in the policy choice (non-tied waves)")
    ax.grid(axis="x", visible=False)
    save_figure(fig, "eda_cost_spread")
    print(f"[Data Visualization] EDA figures written to {config.FIGURES_DIR}")
    return results
