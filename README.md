# SmartRoute ML — ML-Based WMS for Intelligent Order-Picking Route Optimisation

MSc Artificial Intelligence dissertation artefact — Anushka Pathirana.

For every picking wave (pick list), a **Random Forest** or **XGBoost** classifier predicts which
routing policy — **S-shape**, **Return** or **Largest Gap** — gives the shortest picker route in a
real two-block footwear warehouse. The recommended route is compared with the three fixed
heuristics, perfect policy selection (the best of the three heuristics per wave), and the
**exact optimal route** (TSP solved to proven optimality). The models are served through a Streamlit WMS prototype.

Dataset: *Order Picking Dataset from a Warehouse of a Footwear Manufacturing Company*,
Mendeley Data, <https://data.mendeley.com/datasets/pf2w725pw3/1> (CC BY 4.0), stored in `data/raw/`.

## Quick start

**Windows (one click):** double-click `Launch SmartRoute WMS.cmd`. It installs missing packages,
trains the models if `models/` is empty, and opens the WMS in the browser at <http://localhost:8501>.

**Command line** (run from this folder):

```bash
pip install -r requirements.txt
python main.py                      # full pipeline (about 25 min on 12 cores; cached instances make reruns faster)
python main.py --rebuild            # recompute routing instances, labels and features
python -m pytest tests -q           # 65 tests (53 routing/functional + 9 review-database + 3 viewer checks)
streamlit run app/streamlit_app.py  # WMS user interface
```

## Reviewing the raw data without Python (SQLite)

`data/review/SmartRoute Dataset User Guide.docx` explains every file (master data vs transactions, columns,
examples, data quirks). `data/review/smartroute_review.sqlite` is a read-only SQLite copy of the raw CSV files, for reviewing
the data independently of the pipeline. Open it in any SQLite viewer, e.g. the free
[DB Browser for SQLite](https://sqlitebrowser.org/) (*Open Database* → *Browse Data* / *Execute SQL*).

* **The CSV files in `data/raw/` stay the source of truth.** The pipeline never reads the database.
  Rebuild it after any change with `python scripts/build_review_db.py` (about 10 s).
* It is a faithful copy: every row and column is kept. Column names are snake_case
  (`_columns` maps them to the original names), text is trimmed and numbers are typed.
* Added for convenience: `customer_order.creation_datetime` (ISO copy of `creation_date`),
  `storage_policy_slot` (the four storage-policy files in long format) and four views.

| Object | Content |
|---|---|
| `picking_wave`, `customer_order`, `product`, `storage_location`, `support_points` | Raw files, one table each |
| `storage_policy_random / _class_based / _dedicated / _hybrid` | Storage-policy files as published (18 slot columns) |
| `storage_policy_slot` | One row per occupied slot: policy, location, slot, reference, size |
| `v_order_line` | Unit rows grouped into order lines, flagged rack / non-rack (same rule as the pipeline) |
| `v_wave_summary` | Per wave: rows, units, products, distinct locations, operator, first order time |
| `v_pick_with_location` | Every pick row with its x, y, z coordinates and ABC class |
| `v_location_usage` | Pick frequency per location |
| `_tables`, `_columns`, `_about` | Row counts, column-name map, dataset citation and build time |

Example queries that reproduce numbers used in the dissertation:

```sql
-- number of picking waves (9,707)
SELECT COUNT(DISTINCT wave_number) FROM picking_wave;

-- order lines and share at non-rack locations (112,991 lines, 5.2 %)
SELECT COUNT(*), ROUND(100.0 * AVG(1 - is_rack_location), 1) FROM v_order_line;

-- the non-rack locations (staging and consolidation zones)
SELECT location, COUNT(*) FROM v_order_line WHERE is_rack_location = 0 GROUP BY location ORDER BY 2 DESC;

-- distribution of wave size (distinct locations per wave)
SELECT distinct_locations, COUNT(*) AS waves FROM v_wave_summary GROUP BY 1 ORDER BY 1;

-- picks by product ABC class
SELECT abccod, COUNT(*) FROM v_pick_with_location GROUP BY abccod;

-- busiest rack locations with coordinates
SELECT * FROM v_location_usage WHERE x IS NOT NULL ORDER BY pick_rows DESC LIMIT 20;
```

## Warehouse 3D viewer

`data/warehouse_3d/` is an independent, offline 3D view of the warehouse built from the review
database: rack locations by level, aisles, the I/O point, pick frequency, the four storage plans,
product search, picking waves and the authors' SVG floor plans. Double-click
`data/warehouse_3d/Open Warehouse 3D.cmd`; see `data/warehouse_3d/README.md`.

## Folder structure

```
smartroute-wms/
├── Launch SmartRoute WMS.cmd     # one-click launcher for the WMS user interface
├── main.py                       # runs every AI Engineering stage in order
├── src/smartroute/
│   ├── config.py                 # Problem Definition & Environment Setup: constants, paths, seeds, search spaces
│   ├── data_loading.py           # Data Loading and Structure: load_data(), inspect_data()
│   ├── eda.py                    # Exploratory Data Analysis: run_eda()
│   ├── preprocessing.py          # Preprocessing: cleaning, routing instances, labels, split_data()
│   ├── features.py               # Feature Engineering: 29 order-level features
│   ├── models.py                 # Model Training: build_model() (RF, XGBoost), sample weights
│   ├── validation.py             # Testing and Validation: GroupKFold, tune_model(), learning curves
│   ├── evaluate.py               # Evaluation: metrics, significance tests, explainability, model selection
│   ├── deploy.py                 # Deployment & MLOps: save/load, model card, RouteRecommender service
│   ├── visualization.py          # shared plotting (fixed colour per method)
│   ├── review_db.py              # SQLite review copy of the raw data
│   └── warehouse/
│       ├── layout.py             # two-block aisle network, shortest-path distances
│       └── routing.py            # S-shape, Return, Largest Gap, exact optimal (Held-Karp / MILP)
├── app/streamlit_app.py          # Deployment - Model Serving: WMS user interface
├── tests/                        # test_routing.py (functional), test_review_db.py (review copy)
├── data/
│   ├── raw/                      # public Mendeley dataset (read-only, source of truth)
│   ├── processed/                # routing instances & feature table (generated)
│   ├── review/                   # smartroute_review.sqlite (SQLite copy of raw data) + SmartRoute Dataset User Guide.docx
│   └── warehouse_3d/             # independent offline 3D viewer of layout, locations and storage (Open Warehouse 3D.cmd)
├── scripts/build_review_db.py    # rebuilds the SQLite review copy
├── models/                       # footwear_routing_policy_{rf,xgb}.joblib + model_card.json (generated)
├── reports/                      # figures/, tables/, results_summary.json (generated)
└── requirements.txt
```

## AI Engineering stages → implementation

| Stage | Where |
|---|---|
| Problem Definition & Environment Setup | `config.py` (problem framing docstring, constants) |
| Data Loading and Structure | `data_loading.py` |
| Exploratory Data Analysis | `eda.py` (behind `RUN_EDA`) |
| Preprocessing and Feature Engineering | `preprocessing.py`, `features.py`, `warehouse/layout.py` |
| Model Training | `models.py`, baselines in `warehouse/routing.py` |
| Testing and Validation | `validation.py`, `tests/` |
| Evaluation | `evaluate.py`, optimal benchmark in `warehouse/routing.py` |
| Deployment & MLOps | `deploy.py`, `app/streamlit_app.py`, `models/model_card.json` |

## Machine-learning task (one-page summary)

* **Unit of analysis**: one picking wave served under one storage configuration
  (as recorded + Random, Class-based, Dedicated, Hybrid storage) → 47,971 routing instances.
* **Inputs (X)**: 29 features computed from the pick list only — order size, aisles visited,
  block usage, aisle depth, largest-gap structure, dispersion, distance to depot, ABC mix.
* **Label (y)**: the routing policy with the shortest route (S-shape / Return / Largest Gap).
* **Models**: Random Forest and XGBoost, cost-sensitive sample weights (worst − best distance).
* **Validation**: test = held-out storage configuration **and** the most recent 20 % of waves
  (5 outer folds); hyper-parameters tuned inside each fold with GroupKFold by wave, scored by regret (m).
* **Benchmarks**: always-S-shape, always-Return, always-Largest-Gap, perfect policy selection, exact optimum.
