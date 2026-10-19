"""Default-rate tables for exploratory analysis (notebook 03). Training window only."""

from __future__ import annotations

import pandas as pd

BANDS = {
    "fico": ([660, 680, 700, 720, 740, 760, 851], ["660-679", "680-699", "700-719", "720-739", "740-759", "760+"]),
    "dti": ([0, 10, 15, 20, 25, 30, 40], ["<10", "10-15", "15-20", "20-25", "25-30", "30+"]),
    "loan_to_income": ([0, 0.1, 0.2, 0.3, 0.4, 10], ["<10%", "10-20%", "20-30%", "30-40%", "40%+"]),
    "acc_open_past_24mths": ([0, 2, 4, 6, 8, 10, 100], ["0-1", "2-3", "4-5", "6-7", "8-9", "10+"]),
    "annual_inc": ([0, 40e3, 60e3, 80e3, 100e3, 150e3, 1e9], ["<40k", "40-60k", "60-80k", "80-100k", "100-150k", "150k+"]),
    "emp_length_years": ([-1, 0.5, 2.5, 5.5, 9.5, 10.5], ["<1", "1-2", "3-5", "6-9", "10+"]),
}


def rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Loans, share and default rate by band (numeric) or category, with a row for missing values."""
    tr = df[df["split"] == "train"]
    if col in BANDS:
        edges, labels = BANDS[col]
        key = pd.cut(tr[col], edges, labels=labels, right=False).astype("string").fillna("missing")
    else:
        key = tr[col].astype("string").fillna("missing")
    t = tr.groupby(key, sort=False)["target"].agg(n_loans="size", default_rate="mean")
    t["share"] = t["n_loans"] / t["n_loans"].sum()
    if col not in BANDS:
        t = t.sort_values("default_rate")
    else:
        order = list(labels) + ["missing"]
        t = t.reindex([o for o in order if o in t.index])
    t.index.name = col
    return t.reset_index()
