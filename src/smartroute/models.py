"""
Model Training - model families for routing-policy selection.

Ensemble Learning:
* Random Forest (Breiman, 2001) - bagged decision trees, robust on tabular data.
* XGBoost (Chen and Guestrin, 2016) - gradient-boosted trees, the comparison model.

Tree ensembles are invariant to monotone feature scaling, so the preprocessing pipeline
contains no scaler; the Pipeline wrapper keeps the deployment interface identical for both.
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from smartroute import config

MODEL_NAMES = {"rf": "Random Forest", "xgb": "XGBoost"}

# =============================================================================
# MODEL TRAINING
# =============================================================================


# --- Ensemble Learning ------------------------------------------------------------
def build_model(algo, **params):
    """Model Training - Ensemble Learning: return a Pipeline for 'rf' or 'xgb'."""
    if algo == "rf":
        model = RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=1)
    elif algo == "xgb":
        model = XGBClassifier(n_estimators=300, objective="multi:softprob",
                              num_class=len(config.ROUTING_POLICIES), tree_method="hist",
                              eval_metric="mlogloss", random_state=config.RANDOM_STATE, n_jobs=1)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    pipeline = Pipeline([("model", model)])
    if params:
        pipeline.set_params(**params)
    return pipeline


def encode_labels(y):
    """Map policy names to integer classes in config.ROUTING_POLICIES order."""
    mapping = {policy: i for i, policy in enumerate(config.ROUTING_POLICIES)}
    return y.map(mapping).astype(int)


def decode_labels(y_pred):
    return [config.ROUTING_POLICIES[int(i)] for i in y_pred]


# --- Training Mechanics -------------------------------------------------------------
def compute_sample_weights(train_df):
    """Model Training - Training Mechanics: cost-sensitive weights.

    Each training wave is weighted by the spread between its worst and best policy
    distance, so the loss emphasises orders where choosing the wrong policy costs most.
    """
    spread = train_df["cost_spread_m"]
    return (spread / spread.mean()).to_numpy()


def train_model(model, X_train, y_train, sample_weight=None):
    """Model Training - Training Mechanics: fit the pipeline."""
    model.fit(X_train, y_train, model__sample_weight=sample_weight)
    return model
