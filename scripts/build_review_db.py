"""Build data/review/smartroute_review.sqlite from the raw CSV files in data/raw/.

Run from the smartroute-wms folder:  python scripts/build_review_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartroute.review_db import build_review_db  # noqa: E402

if __name__ == "__main__":
    build_review_db()
