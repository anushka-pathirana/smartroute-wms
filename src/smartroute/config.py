"""
Problem Definition & Environment Setup.

Problem Framing
Objective      : For every picking wave (pick list) in a multi-block warehouse, recommend the
                 routing policy (S-shape, Return or Largest Gap) that yields the shortest
                 picker travel distance, and integrate that recommendation into a WMS workflow.
Problem type   : Multiclass classification (algorithm / routing-policy selection).
Target         : best_policy - the heuristic with the minimum route distance for the wave.
Success metric : Mean route distance of the ML-selected policy on unseen storage policies
                 and future waves is lower than the best single fixed policy (always S-shape),
                 with a smaller gap to the exact optimal (TSP) route; macro-F1 and regret are
                 reported as supporting classification metrics.

Environment & Tooling
All constants (paths, seeds, hyperparameters, warehouse geometry, time model) are defined once
here and imported by every other stage.
"""

# =============================================================================
# PROBLEM DEFINITION & ENVIRONMENT SETUP
# =============================================================================

# --- Environment & Tooling ------------------------------------------------------
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Paths ------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

PICKING_WAVE_FILE = RAW_DATA_DIR / "dataset_Picking_Wave.csv"
CUSTOMER_ORDER_FILE = RAW_DATA_DIR / "dataset_Customer_Order.csv"
PRODUCT_FILE = RAW_DATA_DIR / "dataset_Product.csv"
STORAGE_LOCATION_FILE = RAW_DATA_DIR / "dataset_Storage_Location.csv"
SUPPORT_POINTS_FILE = RAW_DATA_DIR / "dataset_Support_Points_Navigation.csv"
STORAGE_POLICY_FILES = {
    "Random": RAW_DATA_DIR / "dataset_Random_Storage.csv",
    "ClassBased": RAW_DATA_DIR / "dataset_Class_Based_Storage.csv",
    "Dedicated": RAW_DATA_DIR / "dataset_Dedicated_Storage.csv",
    "Hybrid": RAW_DATA_DIR / "dataset_Hybrid_Storage.csv",
}

INSTANCES_FILE = PROCESSED_DATA_DIR / "footwear_routing_instances.pkl"
FEATURES_FILE = PROCESSED_DATA_DIR / "footwear_routing_features.csv"
RF_MODEL_FILE = MODELS_DIR / "footwear_routing_policy_rf.joblib"
XGB_MODEL_FILE = MODELS_DIR / "footwear_routing_policy_xgb.joblib"
MODEL_CARD_FILE = MODELS_DIR / "model_card.json"

# --- Reproducibility ------------------------------------------------------------
RANDOM_STATE = 42
NUM_JOBS = -1
RUN_EDA = True

# --- Warehouse geometry (derived from the dataset, see warehouse/layout.py) -------
# Storage coordinates are expressed in CAD layout units. Three picking compartments
# (16 units each) make one 3.66 m shelf (de Assis, de Paula Ferreira and Ouhimmou, 2025),
# so 48 units = 3.66 m.
UNIT_TO_METRES = 3.66 / 48.0
FRONT_CROSS_AISLE_X = 403.0          # central cross aisle (CC support points)
LEFT_BACK_CROSS_AISLE_X = 66.0       # left cross aisle (LC support points)
RIGHT_BACK_CROSS_AISLE_X = 686.0     # right cross aisle (RC support points)
DEPOT_XY = (403.0, 631.0)            # I/O point at support point CC-08
PICKING_LEVELS = (1, 2)              # levels 1-2 are manual picking, 3-4 replenishment

# --- Routing problem --------------------------------------------------------------
STORAGE_POLICIES = list(STORAGE_POLICY_FILES.keys())
ROUTING_POLICIES = ["SShape", "Return", "LargestGap"]
POLICY_LABELS = {"SShape": "S-shape", "Return": "Return", "LargestGap": "Largest Gap",
                 "Optimal": "Optimal (exact TSP)"}
TARGET = "best_policy"
OPTIMAL_TIME_LIMIT_S = 20.0

# --- Picking-time model (assumptions, used to translate distance into time) ------
WALK_SPEED_MPS = 1.0                 # walking speed with trolley
PICK_TIME_PER_LINE_S = 10.0          # search + extract + confirm per order line
SETUP_TIME_S = 60.0                  # trolley set-up and pick-list hand-over per wave

# --- Data splitting ---------------------------------------------------------------
TEST_WAVE_FRACTION = 0.20            # most recent waves are reserved for testing
CV_FOLDS = 3

# --- Hyperparameter search spaces -------------------------------------------------
NUM_SEARCH_ITERATIONS = 12
RF_PARAM_DIST = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [None, 8, 12, 16, 24],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__max_features": ["sqrt", "log2", 0.5],
}
XGB_PARAM_DIST = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [3, 4, 6, 8],
    "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__min_child_weight": [1, 3, 5],
}


def ensure_directories():
    """Environment & Tooling: create every output directory used by the pipeline."""
    for directory in (PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
