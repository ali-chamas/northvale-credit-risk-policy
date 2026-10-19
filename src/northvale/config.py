"""Paths, scope, split windows and constants shared by every step."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"
RAW_ROWS = 2_260_668  # loans in the file (the 33 summary lines without an issue date are not loans)

INTERIM = ROOT / "data" / "interim"
SCOPE_TEXT = INTERIM / "scope_text.parquet"  # 36m 2012-2015, all columns as text
LOANS = INTERIM / "loans_36m_2012_2015.parquet"  # same rows, typed, with target and split
MODELS = INTERIM / "models"
SCORED = INTERIM / "scored.parquet"  # valid + test rows with every model's PD

SQL = ROOT / "sql"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
METRICS = REPORTS / "metrics.json"
DOCS = ROOT / "docs"
APP_DATA = ROOT / "app" / "data"

SEED = 42
N_JOBS = 4  # two other projects share this 16 GB machine

TERM = "36 months"
ISSUE_START, ISSUE_END = "2012-01-01", "2015-12-01"

# Out-of-time split on issue month. No random split anywhere.
SPLITS = {
    "train": ("2012-01-01", "2014-12-01"),
    "valid": ("2015-01-01", "2015-06-01"),
    "test": ("2015-07-01", "2015-12-01"),
}

BAD_STATUSES = ("Charged Off", "Default")
GOOD_STATUSES = ("Fully Paid",)

CURRENCY = "$"  # Lending Club amounts are US dollars
