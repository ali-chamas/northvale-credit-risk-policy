"""Scope, censoring and the vintage default-rate story (notebook 01)."""

from __future__ import annotations

import duckdb
import pandas as pd

from . import config as C

TERMINAL = ("Fully Paid", "Charged Off", "Default")


def _q(sql: str) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET threads = 1")  # multi-threaded float sums differ in the last digit between runs
    df = con.execute(sql).df()
    con.close()
    return df


def censoring_table() -> pd.DataFrame:
    """Share of loans still running at the snapshot, by issue year and term (all loans in the file)."""
    st = pd.read_csv(C.TABLES / "status_by_vintage_all_terms.csv")
    st["policy_flag"] = st["loan_status"].str.startswith("Does not meet")
    st["status"] = st["loan_status"].str.replace("Does not meet the credit policy. Status:", "", regex=False)
    st["terminal"] = st["status"].isin(TERMINAL)
    g = st.groupby(["issue_year", "term"]).apply(
        lambda d: pd.Series({
            "n_loans": d["n_loans"].sum(),
            "n_non_terminal": d.loc[~d["terminal"], "n_loans"].sum(),
        }), include_groups=False,
    ).reset_index()
    g["non_terminal_share"] = g["n_non_terminal"] / g["n_loans"]
    return g


def scope_flow() -> pd.DataFrame:
    """Row counts from raw file to modelling sample, with the reason for each exclusion."""
    L = C.LOANS
    df = _q(f"""
        SELECT year(issue_d) AS issue_year,
               count(*) AS scope_36m,
               count(*) FILTER (WHERE application_type <> 'Individual') AS joint_excluded,
               count(*) FILTER (WHERE application_type = 'Individual' AND target IS NULL) AS non_terminal_excluded,
               count(*) FILTER (WHERE in_sample) AS sample
        FROM '{L}' GROUP BY 1 ORDER BY 1
    """)
    return df


def vintage_table(freq: str = "quarter") -> pd.DataFrame:
    """Default rate by issue period (36-month individual loans, resolved only)."""
    period = "year(issue_d)" if freq == "year" else "year(issue_d) || '-Q' || quarter(issue_d)"
    return _q(f"""
        SELECT {period} AS period, min(split) AS split, count(*) AS n_loans, sum(target) AS n_default,
               avg(target) AS default_rate, sum(loan_amnt) AS amount, avg(int_rate) / 100 AS avg_rate
        FROM '{C.LOANS}' WHERE in_sample GROUP BY 1 ORDER BY 1
    """)


def grade_year_table() -> pd.DataFrame:
    return _q(f"""
        SELECT grade, year(issue_d) AS issue_year, count(*) AS n_loans, avg(target) AS default_rate,
               avg(int_rate) / 100 AS avg_rate
        FROM '{C.LOANS}' WHERE in_sample GROUP BY ALL ORDER BY ALL
    """)


def mix_vs_rate(gy: pd.DataFrame, y0: int, y1: int) -> dict:
    """Split the change in default rate between two vintages into grade mix and within-grade rate.

    change = sum_g (w1 - w0) * r0   (mix: same grade rates, new grade mix)
           + sum_g w1 * (r1 - r0)   (rate: new mix, change in each grade's default rate)
    The two parts add up exactly to the total change.
    """
    a = gy[gy["issue_year"] == y0].set_index("grade")
    b = gy[gy["issue_year"] == y1].set_index("grade")
    w0 = a["n_loans"] / a["n_loans"].sum()
    w1 = b["n_loans"] / b["n_loans"].sum()
    r0, r1 = a["default_rate"], b["default_rate"]
    total0, total1 = float((w0 * r0).sum()), float((w1 * r1).sum())
    mix = float(((w1 - w0) * r0).sum())
    rate = float((w1 * (r1 - r0)).sum())
    return {
        "year_from": y0, "year_to": y1,
        "default_rate_from": total0, "default_rate_to": total1, "change": total1 - total0,
        "mix_effect": mix, "rate_effect": rate,
        "share_ab_from": float(w0.loc[["A", "B"]].sum()), "share_ab_to": float(w1.loc[["A", "B"]].sum()),
    }


def run() -> dict:
    cens = censoring_table()
    cens.to_csv(C.TABLES / "censoring_by_vintage.csv", index=False)
    flow = scope_flow()
    flow.to_csv(C.TABLES / "scope_flow.csv", index=False)
    vq = vintage_table("quarter")
    vy = vintage_table("year")
    vq.to_csv(C.TABLES / "default_rate_by_quarter.csv", index=False)
    vy.to_csv(C.TABLES / "default_rate_by_year.csv", index=False)
    gy = grade_year_table()
    gy.to_csv(C.TABLES / "default_rate_by_grade_year.csv", index=False)

    def cs(year, term):
        r = cens[(cens["issue_year"] == year) & (cens["term"] == term)].iloc[0]
        return float(r["non_terminal_share"])

    in_scope = cens[(cens["term"] == C.TERM) & cens["issue_year"].between(2012, 2015)]
    splits = {}
    sample = _q(f"""SELECT split, count(*) n, avg(target) dr, sum(loan_amnt) amount, min(issue_d) AS d_start, max(issue_d) AS d_end
                   FROM '{C.LOANS}' WHERE in_sample GROUP BY 1""")
    for _, r in sample.iterrows():
        splits[r["split"]] = {"n": int(r["n"]), "default_rate": float(r["dr"]), "amount": float(r["amount"]),
                              "start": str(r["d_start"])[:7], "end": str(r["d_end"])[:7]}

    dec = mix_vs_rate(gy, 2013, 2015)
    g13 = gy[gy["issue_year"] == 2013].set_index("grade")
    g15 = gy[gy["issue_year"] == 2015].set_index("grade")
    by_grade = {
        g: {"dr_2013": float(g13.loc[g, "default_rate"]), "dr_2015": float(g15.loc[g, "default_rate"]),
            "rate_2013": float(g13.loc[g, "avg_rate"]), "rate_2015": float(g15.loc[g, "avg_rate"]),
            "n_2013": int(g13.loc[g, "n_loans"]), "n_2015": int(g15.loc[g, "n_loans"])}
        for g in "ABCDEFG"
    }
    qmin = vq.loc[vq["default_rate"].idxmin()]
    qmax = vq.loc[vq["default_rate"].idxmax()]
    return {
        "data": {
            "raw_rows": C.RAW_ROWS,
            "raw_columns": 151,
            "scope_rows": int(flow["scope_36m"].sum()),
            "joint_excluded": int(flow["joint_excluded"].sum()),
            "non_terminal_excluded": int(flow["non_terminal_excluded"].sum()),
            "sample_rows": int(flow["sample"].sum()),
            "sample_amount": float(sum(s["amount"] for s in splits.values())),
            "sample_default_rate": float(_q(f"SELECT avg(target) FROM '{C.LOANS}' WHERE in_sample").iloc[0, 0]),
            "fico_min": float(_q(f"SELECT min(fico_range_low) FROM '{C.LOANS}' WHERE in_sample").iloc[0, 0]),
            "splits": splits,
        },
        "censoring": {
            "scope_non_terminal_share": float(in_scope["n_non_terminal"].sum() / in_scope["n_loans"].sum()),
            "share_36m_2015": cs(2015, C.TERM),
            "share_36m_2016": cs(2016, C.TERM),
            "share_36m_2017": cs(2017, C.TERM),
            "share_60m_2015": cs(2015, "60 months"),
            "share_60m_2014": cs(2014, "60 months"),
        },
        "vintage": {
            "by_year": {str(int(r["period"])): float(r["default_rate"]) for _, r in vy.iterrows()},
            "n_by_year": {str(int(r["period"])): int(r["n_loans"]) for _, r in vy.iterrows()},
            "min_quarter": str(qmin["period"]), "min_quarter_rate": float(qmin["default_rate"]),
            "max_quarter": str(qmax["period"]), "max_quarter_rate": float(qmax["default_rate"]),
            "decomposition_2013_2015": dec,
            "by_grade": by_grade,
        },
    }
