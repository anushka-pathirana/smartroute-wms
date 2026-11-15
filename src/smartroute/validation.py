"""
Testing and Validation - grouped cross-validation, hyperparameter optimisation and
model diagnostics, all performed on training data only.

Validation Techniques     : GroupKFold with the wave number as group, so the same wave served
                            under different storage configurations never sits in both the
                            fitting and the validation fold.
Hyperparameter Optimization: RandomizedSearchCV scored by negative mean regret (metres), i.e.
                            the operational objective rather than accuracy.
Model Diagnostics         : learning curves and train-versus-validation gaps (overfitting).
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score, learning_curve

from smartroute import config
from smartroute.features import FEATURE_COLUMNS
from smartroute.models import build_model, compute_sample_weights, encode_labels

DISTANCE_COLUMNS = [f"dist_{p}" for p in config.ROUTING_POLICIES]


class RegretScorer:
    """Scorer: negative mean extra distance (m) of the predicted policy over the best policy.

    The distance table is looked up through the DataFrame index of X, which scikit-learn
    preserves when it slices folds.
    """

    def __init__(self, distances_df):
        self.distances = distances_df[DISTANCE_COLUMNS]

    def __call__(self, estimator, X, y):
        y_pred = estimator.predict(X).astype(int)
        table = self.distances.loc[X.index].to_numpy()
        chosen = table[np.arange(len(table)), y_pred]
        return -float((chosen - table.min(axis=1)).mean())


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================


# --- Validation Techniques ----------------------------------------------------------
def build_cv(train_df, n_splits=config.CV_FOLDS):
    """Testing and Validation - GroupKFold grouped by wave."""
    kf = GroupKFold(n_splits=n_splits)
    return list(kf.split(train_df, groups=train_df["wave_number"]))


def cross_validate_model(algo, train_df, params=None, weighting="spread"):
    """Testing and Validation - grouped CV regret of one configuration."""
    X_train, y_train = train_df[FEATURE_COLUMNS], encode_labels(train_df[config.TARGET])
    model = build_model(algo, **(params or {}))
    weights = _weights(train_df, weighting)
    cv_scores = cross_val_score(model, X_train, y_train, cv=build_cv(train_df),
                                scoring=RegretScorer(train_df), n_jobs=config.NUM_JOBS,
                                params={"model__sample_weight": weights} if weights is not None else None)
    return cv_scores


def _weights(train_df, weighting):
    if weighting == "spread":
        return compute_sample_weights(train_df)
    if weighting == "balanced":
        counts = train_df[config.TARGET].value_counts()
        return train_df[config.TARGET].map(len(train_df) / (len(counts) * counts)).to_numpy()
    return None


def compare_weighting(train_df, algo="rf"):
    """Testing and Validation - ablation of the class/sample weighting strategy."""
    rows = []
    for weighting in ("none", "balanced", "spread"):
        cv_scores = cross_validate_model(algo, train_df, weighting=weighting)
        rows.append({"weighting": weighting, "cv_regret_m": -cv_scores.mean(), "cv_regret_std": cv_scores.std()})
    weighting_df = pd.DataFrame(rows)
    print("[Validation Techniques] weighting ablation\n" + weighting_df.round(3).to_string(index=False))
    return weighting_df


# --- Hyperparameter Optimization -----------------------------------------------------
def tune_model(algo, train_df, num_iterations=None):
    """Testing and Validation - Hyperparameter Optimization with RandomizedSearchCV."""
    num_iterations = num_iterations or config.NUM_SEARCH_ITERATIONS
    X_train, y_train = train_df[FEATURE_COLUMNS], encode_labels(train_df[config.TARGET])
    param_dist = config.RF_PARAM_DIST if algo == "rf" else config.XGB_PARAM_DIST
    random_search = RandomizedSearchCV(
        build_model(algo), param_dist, n_iter=num_iterations, scoring=RegretScorer(train_df),
        cv=build_cv(train_df), n_jobs=config.NUM_JOBS, random_state=config.RANDOM_STATE,
        refit=True, return_train_score=True)
    random_search.fit(X_train, y_train, model__sample_weight=compute_sample_weights(train_df))
    best_params = random_search.best_params_
    best_score = -random_search.best_score_
    print(f"[Hyperparameter Optimization] {algo}: CV regret {best_score:.2f} m with {best_params}")
    return random_search.best_estimator_, best_params, best_score, pd.DataFrame(random_search.cv_results_)


# --- Model Diagnostics ---------------------------------------------------------------
def diagnose_learning_curve(algo, params, train_df):
    """Testing and Validation - Model Diagnostics: learning curve (accuracy) on grouped CV."""
    X_train, y_train = train_df[FEATURE_COLUMNS], encode_labels(train_df[config.TARGET])
    train_sizes, train_scores, val_scores = learning_curve(
        build_model(algo, **params), X_train, y_train, cv=build_cv(train_df),
        train_sizes=np.linspace(0.1, 1.0, 6), scoring="accuracy", n_jobs=config.NUM_JOBS, shuffle=True,
        random_state=config.RANDOM_STATE)
    curve_df = pd.DataFrame({"train_size": train_sizes,
                             "train_accuracy": train_scores.mean(axis=1),
                             "validation_accuracy": val_scores.mean(axis=1),
                             "validation_std": val_scores.std(axis=1)})
    print(f"[Model Diagnostics] {algo}: final train/val accuracy "
          f"{curve_df['train_accuracy'].iloc[-1]:.3f}/{curve_df['validation_accuracy'].iloc[-1]:.3f}")
    return curve_df
