"""Feature audit: a documented decision for every one of the 151 raw columns (notebook 02).

Output: docs/feature_audit.csv with, per column, whether it is available when
the lender decides, its category, keep/drop, the reason, missing rates in the
train and test windows, and a single-feature AUC screen on the training window.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from . import features as F

POST_ORIGINATION = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low", "pymnt_plan",
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status", "deferral_term", "hardship_amount",
    "hardship_start_date", "hardship_end_date", "payment_plan_start_date", "hardship_length", "hardship_dpd",
    "hardship_loan_status", "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount",
    "hardship_last_payment_amount", "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage", "settlement_term",
]
LATE_2015 = [
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il", "il_util",
    "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util", "inq_fi", "total_cu_tl", "inq_last_12m",
]
JOINT = [
    "annual_inc_joint", "dti_joint", "verification_status_joint", "revol_bal_joint", "sec_app_fico_range_low",
    "sec_app_fico_range_high", "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", "sec_app_mort_acc",
    "sec_app_open_acc", "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med", "sec_app_mths_since_last_major_derog",
]

# Explicit decisions. Anything not listed here and not in the groups above is an
# application-time field kept as a candidate feature (features.RAW_NUMERIC / CATEGORICAL).
EXPLICIT = {
    "id": ("identifier", "yes", "drop", "Loan identifier; no information about the borrower."),
    "member_id": ("identifier", "yes", "drop", "Borrower identifier; empty in this release."),
    "url": ("identifier", "yes", "drop", "Listing URL; contains only the loan id."),
    "desc": ("free_text", "yes", "drop", "Free-text description written by the borrower; common in 2012 and almost never filled by 2015, so the model would learn a period effect."),
    "title": ("free_text", "yes", "drop", "Free-text loan title typed by the borrower; duplicates purpose."),
    "emp_title": ("free_text", "yes", "drop", "Free-text job title with very high cardinality and spelling variants; reducing it to groups would take manual coding and can proxy for protected characteristics."),
    "loan_status": ("target", "no", "target", "Outcome: Charged Off or Default = 1, Fully Paid = 0; other statuses excluded."),
    "issue_d": ("split_key", "yes", "split only", "Issue month; defines vintages and the out-of-time split. Not a feature."),
    "term": ("scope", "yes", "scope filter", "Constant after scoping to 36-month loans."),
    "application_type": ("scope", "yes", "scope filter", "Only individual applications kept; joint applications start in October 2015 and do not exist in training."),
    "policy_code": ("scope", "yes", "drop", "Constant (1) for every loan in scope."),
    "grade": ("lc_pricing", "yes", "benchmark only", "Lending Club's own risk grade: the incumbent model. Used as the benchmark, not as a feature."),
    "sub_grade": ("lc_pricing", "yes", "benchmark only", "Lending Club sub-grade (A1 to G5): the incumbent ranker the models are compared with."),
    "int_rate": ("lc_pricing", "yes", "P&L only", "Price set from the sub-grade; a model using it would copy Lending Club's model. Used in the P&L layer."),
    "installment": ("lc_pricing", "yes", "drop", "Monthly payment = f(amount, int_rate, term), so it encodes the incumbent price."),
    "funded_amnt": ("lc_marketplace", "yes", "P&L only", "Amount Lending Club funded; equal to loan_amnt for almost every loan. Used as exposure in the P&L."),
    "funded_amnt_inv": ("lc_marketplace", "no", "drop", "Amount investors funded; set after the listing, a marketplace outcome."),
    "initial_list_status": ("lc_marketplace", "yes", "drop", "Whole vs fractional listing: Lending Club's allocation between investor pools, not a borrower attribute."),
    "disbursement_method": ("lc_marketplace", "yes", "drop", "Cash vs direct payment to creditors; the same value (Cash) for every loan in scope."),
    "zip_code": ("fairness", "yes", "drop", "3-digit zip; geography can proxy for race and ethnicity (ECOA / Reg B disparate impact)."),
    "addr_state": ("fairness", "yes", "drop", "State of residence; same proxy risk as zip code, and pricing rules differ by state."),
    "fico_range_low": ("application_derived", "yes", "transformed", "Combined with fico_range_high into one FICO midpoint (the range is 4 points wide for almost every loan)."),
    "fico_range_high": ("application_derived", "yes", "transformed", "Combined with fico_range_low into one FICO midpoint."),
    "earliest_cr_line": ("application_derived", "yes", "transformed", "Converted to credit history length in months at the issue date."),
    "emp_length": ("application_derived", "yes", "transformed", "Converted to years (0 to 10+); missing kept as missing."),
}

FLAG_AUC = 0.70  # a single application field stronger than this would be suspicious (FICO alone is about 0.60)


def rule(col: str) -> tuple[str, str, str, str]:
    if col in EXPLICIT:
        return EXPLICIT[col]
    if col in POST_ORIGINATION:
        return ("post_origination", "no", "drop", "Recorded after the loan is issued (payments, recoveries, hardship, settlement or refreshed credit data). Leaks the outcome.")
    if col in LATE_2015:
        return ("late_2015_field", "yes", "drop", "Bureau field added in late 2015: empty for every training loan, so the model cannot learn it.")
    if col in JOINT:
        return ("joint_applicant", "yes", "drop", "Joint or secondary applicant field; joint applications start in October 2015 and are out of scope.")
    if col in F.RAW_NUMERIC or col in F.CATEGORICAL:
        return ("application", "yes", "keep", "Known at application from the form or the credit bureau pull.")
    raise KeyError(f"No audit rule for column {col}")


def _binned_auc(x: pd.Series, y: np.ndarray) -> float:
    """Single-feature AUC after binning (10 quantile bins + a missing bin, or categories).

    The bin's default rate is the score, so the AUC does not depend on the
    direction of the relationship and missing values count as information.
    """
    if pd.api.types.is_numeric_dtype(x) or pd.api.types.is_datetime64_any_dtype(x):
        xv = pd.to_numeric(x, errors="coerce") if not pd.api.types.is_datetime64_any_dtype(x) else x.astype("int64").where(x.notna())
        if xv.nunique(dropna=True) <= 1 and xv.notna().all():
            return 0.5
        try:
            b = pd.qcut(xv, 10, duplicates="drop").astype("string")
        except ValueError:
            b = xv.astype("string")
        b = b.fillna("missing")
    else:
        s = x.astype("string").fillna("missing")
        counts = s.value_counts()
        keep = counts[counts >= 100].index
        b = s.where(s.isin(keep), "other")
    if b.nunique() <= 1:
        return 0.5
    rate = pd.Series(y).groupby(b.values).transform("mean").values
    return float(roc_auc_score(y, rate))


def _months_since_issue(df: pd.DataFrame, col: str) -> pd.Series:
    d = pd.to_datetime(df[col])
    i = pd.to_datetime(df["issue_d"])
    return (d.dt.year - i.dt.year) * 12 + (d.dt.month - i.dt.month)


def screen_and_missing() -> pd.DataFrame:
    """Missing rates (train / test) and the single-feature AUC screen for every raw column."""
    con = duckdb.connect()
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{C.LOANS}'").fetchall()]
    raw = [c for c in cols if c not in ("target", "split", "in_sample")]
    y = con.execute(f"SELECT target FROM '{C.LOANS}' WHERE in_sample AND split = 'train' ORDER BY issue_d, id").df()["target"].to_numpy()
    rows = []
    for i in range(0, len(raw), 20):
        batch = raw[i : i + 20]
        sel = ", ".join(f'"{c}"' for c in dict.fromkeys(batch + ["issue_d"]))
        tr = con.execute(f"SELECT {sel} FROM '{C.LOANS}' WHERE in_sample AND split = 'train' ORDER BY issue_d, id").df()
        miss = con.execute(
            "SELECT split, " + ", ".join(f'avg(("{c}" IS NULL)::INT) AS "{c}"' for c in batch)
            + f" FROM '{C.LOANS}' WHERE in_sample GROUP BY split"
        ).df().set_index("split")
        for c in batch:
            cat = rule(c)[0]
            if cat in ("identifier", "target", "split_key", "free_text") or c in ("term", "application_type", "policy_code"):
                auc = np.nan
            else:
                x = tr[c]
                if pd.api.types.is_datetime64_any_dtype(x) or str(x.dtype).startswith("date") or c.endswith("_d") or c.endswith("_date") or c == "earliest_cr_line":
                    x = _months_since_issue(tr, c).astype("float") if c != "earliest_cr_line" else -_months_since_issue(tr, c).astype("float")
                auc = _binned_auc(x, y)
            rows.append({
                "column": c,
                "missing_train": float(miss.loc["train", c]),
                "missing_test": float(miss.loc["test", c]),
                "single_feature_auc_train": auc,
            })
    con.close()
    return pd.DataFrame(rows)


def derived_screen(df: pd.DataFrame) -> pd.DataFrame:
    tr = df[df["split"] == "train"]
    y = tr["target"].to_numpy()
    rows = [{"column": c, "single_feature_auc_train": _binned_auc(tr[c], y)} for c in F.DERIVED_NUMERIC]
    rows.append({"column": "sub_grade (LC benchmark)", "single_feature_auc_train": _binned_auc(tr["sub_grade"], y)})
    return pd.DataFrame(rows)


def build_audit() -> pd.DataFrame:
    scr = screen_and_missing()
    recs = []
    for _, r in scr.iterrows():
        cat, avail, decision, reason = rule(r["column"])
        auc = r["single_feature_auc_train"]
        flag = ""
        if pd.notna(auc) and auc >= FLAG_AUC:
            flag = "suspicious: stronger than any application field" if cat == "post_origination" else "strong: check"
        recs.append({
            "column": r["column"],
            "available_at_application": avail,
            "category": cat,
            "decision": decision,
            "reason": reason,
            "missing_train": round(r["missing_train"], 4),
            "missing_test": round(r["missing_test"], 4),
            "single_feature_auc_train": round(auc, 4) if pd.notna(auc) else "",
            "screen_flag": flag,
        })
    audit = pd.DataFrame(recs)
    if len(audit) != 151:
        raise ValueError(f"Audit should cover 151 raw columns, got {len(audit)}")
    return audit


def run(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    audit = build_audit()
    audit.to_csv(C.DOCS / "feature_audit.csv", index=False)
    der = derived_screen(df)
    der.to_csv(C.TABLES / "single_feature_auc_derived.csv", index=False)

    a = audit.copy()
    a["auc"] = pd.to_numeric(a["single_feature_auc_train"], errors="coerce")
    app = a[a["category"] == "application"]
    post = a[a["category"] == "post_origination"]
    flagged = a[a["screen_flag"] != ""].sort_values("auc", ascending=False)
    top_app = pd.concat([app[["column", "auc"]], der.rename(columns={"single_feature_auc_train": "auc"})])
    top_app = top_app[~top_app["column"].str.contains("sub_grade")].sort_values("auc", ascending=False)
    m = {
        "n_columns": int(len(a)),
        "by_category": a["category"].value_counts().to_dict(),
        "by_decision": a["decision"].value_counts().to_dict(),
        "n_candidate_features": len(F.FEATURES),
        "n_post_origination": int(len(post)),
        "flag_threshold": FLAG_AUC,
        "n_flagged": int(len(flagged)),
        "flagged": flagged["column"].tolist(),
        "flagged_all_post_origination_or_pricing": bool(flagged["category"].isin(["post_origination", "lc_pricing", "lc_marketplace"]).all()),
        "max_post_auc": float(post["auc"].max()),
        "max_post_column": str(post.loc[post["auc"].idxmax(), "column"]),
        "max_app_auc": float(top_app["auc"].iloc[0]),
        "max_app_column": str(top_app["column"].iloc[0]),
        "subgrade_auc_train": float(der.loc[der["column"].str.contains("sub_grade"), "single_feature_auc_train"].iloc[0]),
        "int_rate_auc_train": float(a.loc[a["column"] == "int_rate", "auc"].iloc[0]),
        "zip_auc_train": float(a.loc[a["column"] == "zip_code", "auc"].iloc[0]),
        "last_fico_auc_train": float(a.loc[a["column"] == "last_fico_range_high", "auc"].iloc[0]),
        "total_rec_prncp_auc_train": float(a.loc[a["column"] == "total_rec_prncp", "auc"].iloc[0]),
    }
    return m, top_app


def leakage_demo(df: pd.DataFrame, params: dict, baseline_auc: float) -> tuple[dict, pd.DataFrame]:
    """Retrain the chosen LightGBM with one post-origination field added and compare test AUC."""
    from . import models as M  # local import: models imports features, avoid a cycle at import time

    tr, va, te = (F.split(df, s) for s in ("train", "valid", "test"))
    rows = [{"label": "Application\nfields only", "field": "", "test_auc": baseline_auc}]
    for col, lab in (("last_fico_range_high", "+ last FICO\n(refreshed later)"), ("total_rec_prncp", "+ principal\nrepaid to date")):
        feats = F.FEATURES + [col]
        Xtr, Xva, Xte = (F.lgbm_matrix(d, feats) for d in (tr, va, te))
        bst = M.fit_lgbm(Xtr, tr["target"].to_numpy(), Xva, va["target"].to_numpy(), params)
        auc = roc_auc_score(te["target"], bst.predict(Xte, num_iteration=bst.best_iteration))
        rows.append({"label": lab, "field": col, "test_auc": float(auc)})
        print(f"  leakage demo + {col}: test AUC {auc:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES / "leakage_demo.csv", index=False)
    return {
        "baseline_auc": baseline_auc,
        "auc_with_last_fico": float(out.loc[1, "test_auc"]),
        "auc_with_total_rec_prncp": float(out.loc[2, "test_auc"]),
        "jump_last_fico": float(out.loc[1, "test_auc"] - baseline_auc),
        "jump_total_rec_prncp": float(out.loc[2, "test_auc"] - baseline_auc),
    }, out


def write_data_dictionary(scorecard_chars: list[str]) -> None:
    """docs/data_dictionary.md: every model feature with its source, coverage and strength (generated)."""
    a = pd.read_csv(C.DOCS / "feature_audit.csv").set_index("column")
    der = pd.read_csv(C.TABLES / "single_feature_auc_derived.csv").set_index("column")["single_feature_auc_train"]
    shap_rank = pd.read_csv(C.TABLES / "shap_importance.csv")["feature"].tolist()
    source = {"fico": "fico_range_low, fico_range_high", "credit_history_months": "earliest_cr_line, issue_d",
              "emp_length_years": "emp_length", "loan_to_income": "loan_amnt, annual_inc",
              "revol_bal_to_income": "revol_bal, annual_inc"}
    lines = [
        "<!-- Generated by `make analysis` (audit.write_data_dictionary). Do not edit by hand. -->",
        "# Data dictionary: model features",
        "",
        f"The {len(F.FEATURES)} application-time features used by LightGBM. The scorecard uses the {len(scorecard_chars)} marked in the "
        "Scorecard column. Missing shares are for the training (2012-2014) and test (Jul-Dec 2015) windows; single-field AUC is "
        "measured on the training window after binning (0.5 = no information). The decision for every one of the 151 raw "
        "columns is in `feature_audit.csv`.",
        "",
        "| Feature | Meaning | Source column(s) | Type | Missing, train | Missing, test | Single-field AUC | Scorecard | SHAP rank |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f in F.FEATURES:
        src = source.get(f, f)
        if f in a.index:
            mt, ms, auc = a.loc[f, "missing_train"], a.loc[f, "missing_test"], a.loc[f, "single_feature_auc_train"]
        else:
            first = src.split(",")[0]
            mt, ms, auc = a.loc[first, "missing_train"], a.loc[first, "missing_test"], der.get(f, np.nan)
        kind = "categorical" if f in F.CATEGORICAL else "numeric"
        rank = shap_rank.index(f) + 1 if f in shap_rank else ""
        auc_s = f"{float(auc):.3f}" if pd.notna(auc) and auc != "" else ""
        lines.append(f"| `{f}` | {F.LABELS[f]} | {src} | {kind} | {float(mt) * 100:.1f}% | {float(ms) * 100:.1f}% | {auc_s} | "
                     f"{'yes' if f in scorecard_chars else ''} | {rank} |")
    lines += ["", "Derived features: `fico` is the midpoint of the 4-point FICO range; `credit_history_months` counts months "
              "from the earliest credit line to the issue month; `emp_length_years` maps '< 1 year' to 0 and '10+ years' to 10; "
              "home ownership values other than MORTGAGE, RENT and OWN are grouped as OTHER; loan purposes with fewer than "
              "2,500 loans are grouped into 'other'.", ""]
    (C.DOCS / "data_dictionary.md").write_text("\n".join(lines))
