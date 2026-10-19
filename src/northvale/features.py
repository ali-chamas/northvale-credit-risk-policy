"""Load the modelling sample and build application-time features.

Only fields a lender would hold when it decides on an application become model
features. The decision for every raw column, with its reason, is in
`audit.py` and is written to docs/feature_audit.csv.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from . import config as C

# Raw application-time columns used as-is.
RAW_NUMERIC = [
    "loan_amnt", "annual_inc", "dti", "inq_last_6mths", "delinq_2yrs", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mths_since_last_delinq", "mths_since_last_record",
    "mths_since_last_major_derog", "collections_12_mths_ex_med", "acc_now_delinq", "tot_coll_amt",
    "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_bc_dlq", "mths_since_recent_inq",
    "mths_since_recent_revol_delinq", "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl",
    "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
    "num_rev_tl_bal_gt_0", "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit",
]
# Derived from application-time columns (see engineer()).
DERIVED_NUMERIC = ["fico", "credit_history_months", "emp_length_years", "loan_to_income", "revol_bal_to_income"]
CATEGORICAL = ["home_ownership", "verification_status", "purpose"]

FEATURES = DERIVED_NUMERIC + RAW_NUMERIC + CATEGORICAL
NUMERIC = DERIVED_NUMERIC + RAW_NUMERIC

# Incumbent pricing (benchmark and P&L only, never a model feature).
PRICING = ["grade", "sub_grade", "int_rate", "installment"]
# Cash fields for the P&L layer (post-origination: outcome measurement only).
CASH = ["funded_amnt", "total_pymnt", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
        "recoveries", "collection_recovery_fee"]
# Post-origination fields used only in the leakage demonstration.
LEAKY_DEMO = ["last_fico_range_high"]

KEYS = ["id", "issue_d", "split", "target", "loan_status"]

_SOURCE = [
    "fico_range_low", "fico_range_high", "earliest_cr_line", "emp_length",
]

# Plain-English names, used on charts, in reason codes and in the app.
LABELS = {
    "fico": "FICO score at application",
    "credit_history_months": "Length of credit history (months)",
    "emp_length_years": "Years in current job",
    "loan_to_income": "Loan amount / annual income",
    "revol_bal_to_income": "Revolving balance / annual income",
    "loan_amnt": "Loan amount requested",
    "annual_inc": "Annual income",
    "dti": "Debt-to-income ratio (%)",
    "inq_last_6mths": "Credit inquiries, last 6 months",
    "delinq_2yrs": "Delinquencies, last 2 years",
    "open_acc": "Open credit lines",
    "pub_rec": "Derogatory public records",
    "revol_bal": "Revolving balance",
    "revol_util": "Revolving utilisation (%)",
    "total_acc": "Total credit lines ever",
    "mths_since_last_delinq": "Months since last delinquency",
    "mths_since_last_record": "Months since last public record",
    "mths_since_last_major_derog": "Months since last 90+ day rating",
    "collections_12_mths_ex_med": "Collections, last 12 months",
    "acc_now_delinq": "Accounts now delinquent",
    "tot_coll_amt": "Total collection amounts owed",
    "tot_cur_bal": "Total current balance, all accounts",
    "total_rev_hi_lim": "Total revolving credit limit",
    "acc_open_past_24mths": "Accounts opened, last 24 months",
    "avg_cur_bal": "Average balance per account",
    "bc_open_to_buy": "Open-to-buy on bankcards",
    "bc_util": "Bankcard utilisation (%)",
    "chargeoff_within_12_mths": "Charge-offs, last 12 months",
    "delinq_amnt": "Past-due amount owed",
    "mo_sin_old_il_acct": "Months since oldest instalment account",
    "mo_sin_old_rev_tl_op": "Months since oldest revolving account",
    "mo_sin_rcnt_rev_tl_op": "Months since newest revolving account",
    "mo_sin_rcnt_tl": "Months since newest account",
    "mort_acc": "Mortgage accounts",
    "mths_since_recent_bc": "Months since newest bankcard",
    "mths_since_recent_bc_dlq": "Months since last bankcard delinquency",
    "mths_since_recent_inq": "Months since last inquiry",
    "mths_since_recent_revol_delinq": "Months since last revolving delinquency",
    "num_accts_ever_120_pd": "Accounts ever 120+ days past due",
    "num_actv_bc_tl": "Active bankcard accounts",
    "num_actv_rev_tl": "Active revolving accounts",
    "num_bc_sats": "Satisfactory bankcard accounts",
    "num_bc_tl": "Bankcard accounts",
    "num_il_tl": "Instalment accounts",
    "num_op_rev_tl": "Open revolving accounts",
    "num_rev_accts": "Revolving accounts",
    "num_rev_tl_bal_gt_0": "Revolving accounts with a balance",
    "num_sats": "Satisfactory accounts",
    "num_tl_120dpd_2m": "Accounts 120+ days past due, last 2 months",
    "num_tl_30dpd": "Accounts 30 days past due, last 2 months",
    "num_tl_90g_dpd_24m": "Accounts 90+ days past due, last 24 months",
    "num_tl_op_past_12m": "Accounts opened, last 12 months",
    "pct_tl_nvr_dlq": "Share of accounts never delinquent (%)",
    "percent_bc_gt_75": "Share of bankcards above 75% of limit (%)",
    "pub_rec_bankruptcies": "Public record bankruptcies",
    "tax_liens": "Tax liens",
    "tot_hi_cred_lim": "Total credit limit",
    "total_bal_ex_mort": "Total balance excluding mortgage",
    "total_bc_limit": "Total bankcard limit",
    "total_il_high_credit_limit": "Total instalment credit limit",
    "home_ownership": "Home ownership",
    "verification_status": "Income verification",
    "purpose": "Loan purpose",
}


def load(columns: list[str] | None = None, where: str = "in_sample") -> pd.DataFrame:
    """Read the scoped parquet with DuckDB, projecting only the needed columns."""
    cols = columns or (KEYS + _SOURCE + [c for c in RAW_NUMERIC] + CATEGORICAL + PRICING + CASH + LEAKY_DEMO)
    cols = list(dict.fromkeys(cols))
    sel = ", ".join(f'"{c}"' for c in cols)
    con = duckdb.connect()
    df = con.execute(f"SELECT {sel} FROM '{C.LOANS}' WHERE {where} ORDER BY issue_d, id").df()
    con.close()
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derived application-time features. Float32 to keep memory low."""
    out = df.copy()
    out["fico"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
    issue = pd.to_datetime(out["issue_d"])
    ecl = pd.to_datetime(out["earliest_cr_line"])
    out["credit_history_months"] = (issue.dt.year - ecl.dt.year) * 12 + (issue.dt.month - ecl.dt.month)
    emp = out["emp_length"].astype("string")
    years = emp.str.extract(r"(\d+)")[0].astype("float")
    years = years.where(~emp.str.startswith("<", na=False), 0.0)
    out["emp_length_years"] = years  # missing stays missing (no employment length given)
    out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"]
    out["revol_bal_to_income"] = out["revol_bal"] / out["annual_inc"]
    out["home_ownership"] = out["home_ownership"].where(out["home_ownership"].isin(["MORTGAGE", "RENT", "OWN"]), "OTHER")
    rare = {"educational", "renewable_energy", "wedding", "house"}
    out["purpose"] = out["purpose"].where(~out["purpose"].isin(rare), "other")
    for c in NUMERIC:
        out[c] = out[c].astype("float32")
    for c in CATEGORICAL:
        out[c] = out[c].astype("string").fillna("missing")
    out["sub_grade_rank"] = out["sub_grade"].map(subgrade_rank).astype("int16")
    out["issue_d"] = pd.to_datetime(out["issue_d"])
    return out


def subgrade_rank(sg: str) -> int:
    """A1 -> 1 ... G5 -> 35. Higher rank = riskier according to Lending Club."""
    return (ord(sg[0]) - ord("A")) * 5 + int(sg[1])


def sample() -> pd.DataFrame:
    return engineer(load())


def split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[df["split"] == name]


def lgbm_matrix(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Feature matrix for LightGBM: numeric as float32, categoricals as pandas category."""
    feats = features or FEATURES
    X = df[feats].copy()
    for c in CATEGORICAL:
        if c in X:
            X[c] = pd.Categorical(X[c], categories=sorted(CATEGORY_LEVELS[c]))
    return X


CATEGORY_LEVELS = {
    "home_ownership": ["MORTGAGE", "RENT", "OWN", "OTHER"],
    "verification_status": ["Not Verified", "Source Verified", "Verified"],
    "purpose": ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
                "medical", "small_business", "car", "moving", "vacation"],
}


def amount_weighted(values: pd.Series, weights: pd.Series) -> float:
    w = weights.astype(float)
    return float(np.sum(values.astype(float) * w) / np.sum(w))
