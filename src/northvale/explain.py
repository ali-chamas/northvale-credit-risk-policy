"""SHAP explanations for the LightGBM model and plain-English reason codes (notebook 05).

SHAP values are in log-odds of the uncalibrated model. Platt recalibration is
monotonic, so the order and sign of the reasons are unchanged.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from . import config as C
from . import features as F

N_SHAP = 20_000  # test-window applicants explained (random, seed 42)
N_APP = 200  # applicants exported to the app

# Short, plain-English reason templates. {v} = applicant value, {m} = median applicant.
def _money(x):
    return f"${x:,.0f}"


def _pct(x):
    return f"{x:.0f}%"


def _n(x):
    return f"{x:.0f}"


REASONS = {
    "fico": ("FICO score of {v}", "median applicant {m}", _n),
    "dti": ("Debt-to-income ratio of {v}", "median {m}", _pct),
    "annual_inc": ("Annual income of {v}", "median {m}", _money),
    "loan_to_income": ("Loan equal to {v} of annual income", "median {m}", lambda x: f"{x * 100:.0f}%"),
    "loan_amnt": ("Loan amount of {v}", "median {m}", _money),
    "acc_open_past_24mths": ("{v} credit accounts opened in the last 24 months", "median {m}", _n),
    "num_tl_op_past_12m": ("{v} credit accounts opened in the last 12 months", "median {m}", _n),
    "mths_since_recent_inq": ("Last credit inquiry {v} months ago", "median {m}", _n),
    "inq_last_6mths": ("{v} credit inquiries in the last 6 months", "median {m}", _n),
    "bc_open_to_buy": ("Unused bankcard credit of {v}", "median {m}", _money),
    "tot_hi_cred_lim": ("Total credit limit of {v}", "median {m}", _money),
    "total_bc_limit": ("Total bankcard limit of {v}", "median {m}", _money),
    "total_rev_hi_lim": ("Total revolving limit of {v}", "median {m}", _money),
    "avg_cur_bal": ("Average balance per account of {v}", "median {m}", _money),
    "tot_cur_bal": ("Total balances of {v}", "median {m}", _money),
    "mort_acc": ("{v} mortgage accounts", "median {m}", _n),
    "revol_util": ("Revolving credit utilisation of {v}", "median {m}", _pct),
    "bc_util": ("Bankcard utilisation of {v}", "median {m}", _pct),
    "percent_bc_gt_75": ("{v} of bankcards above 75% of their limit", "median {m}", _pct),
    "credit_history_months": ("Credit history of {v} months", "median {m}", _n),
    "mo_sin_old_rev_tl_op": ("Oldest revolving account opened {v} months ago", "median {m}", _n),
    "mo_sin_old_il_acct": ("Oldest instalment account opened {v} months ago", "median {m}", _n),
    "mo_sin_rcnt_tl": ("Newest account opened {v} months ago", "median {m}", _n),
    "mo_sin_rcnt_rev_tl_op": ("Newest revolving account opened {v} months ago", "median {m}", _n),
    "mths_since_recent_bc": ("Newest bankcard opened {v} months ago", "median {m}", _n),
    "emp_length_years": ("{v} years in current job", "median {m}", _n),
    "revol_bal": ("Revolving balance of {v}", "median {m}", _money),
    "revol_bal_to_income": ("Revolving balance equal to {v} of income", "median {m}", lambda x: f"{x * 100:.0f}%"),
    "total_bal_ex_mort": ("Non-mortgage balances of {v}", "median {m}", _money),
    "total_il_high_credit_limit": ("Instalment credit limit of {v}", "median {m}", _money),
    "num_actv_rev_tl": ("{v} active revolving accounts", "median {m}", _n),
    "num_rev_tl_bal_gt_0": ("{v} revolving accounts carrying a balance", "median {m}", _n),
    "open_acc": ("{v} open credit lines", "median {m}", _n),
    "total_acc": ("{v} credit lines in total", "median {m}", _n),
    "num_il_tl": ("{v} instalment accounts", "median {m}", _n),
    "num_bc_tl": ("{v} bankcard accounts", "median {m}", _n),
    "num_bc_sats": ("{v} bankcards in good standing", "median {m}", _n),
    "num_sats": ("{v} accounts in good standing", "median {m}", _n),
    "num_rev_accts": ("{v} revolving accounts", "median {m}", _n),
    "num_op_rev_tl": ("{v} open revolving accounts", "median {m}", _n),
    "num_actv_bc_tl": ("{v} active bankcards", "median {m}", _n),
    "pct_tl_nvr_dlq": ("{v} of accounts never delinquent", "median {m}", _pct),
    "delinq_2yrs": ("{v} delinquencies in the last 2 years", "median {m}", _n),
    "pub_rec": ("{v} derogatory public records", "median {m}", _n),
    "pub_rec_bankruptcies": ("{v} bankruptcies on record", "median {m}", _n),
    "tax_liens": ("{v} tax liens", "median {m}", _n),
    "collections_12_mths_ex_med": ("{v} collections in the last 12 months", "median {m}", _n),
    "chargeoff_within_12_mths": ("{v} charge-offs in the last 12 months", "median {m}", _n),
    "acc_now_delinq": ("{v} accounts now delinquent", "median {m}", _n),
    "delinq_amnt": ("Past-due amount of {v}", "median {m}", _money),
    "tot_coll_amt": ("Collection amounts owed of {v}", "median {m}", _money),
    "num_accts_ever_120_pd": ("{v} accounts ever 120+ days past due", "median {m}", _n),
    "num_tl_120dpd_2m": ("{v} accounts 120+ days past due now", "median {m}", _n),
    "num_tl_30dpd": ("{v} accounts 30 days past due now", "median {m}", _n),
    "num_tl_90g_dpd_24m": ("{v} accounts 90+ days late in the last 24 months", "median {m}", _n),
    "mths_since_last_delinq": ("Last delinquency {v} months ago", "median {m}", _n),
    "mths_since_last_record": ("Last public record {v} months ago", "median {m}", _n),
    "mths_since_last_major_derog": ("Last 90+ day late rating {v} months ago", "median {m}", _n),
    "mths_since_recent_bc_dlq": ("Last bankcard delinquency {v} months ago", "median {m}", _n),
    "mths_since_recent_revol_delinq": ("Last revolving delinquency {v} months ago", "median {m}", _n),
}
MIN_REASON = 0.03  # ignore contributions below this (log-odds) when writing reason codes
CAT_REASONS = {
    "purpose": "Loan purpose: {v}",
    "home_ownership": "Home ownership: {v}",
    "verification_status": "Income verification: {v}",
}


def reason_text(feature: str, value, median) -> str:
    if feature in CAT_REASONS:
        return CAT_REASONS[feature].format(v=str(value).replace("_", " ").lower())
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{F.LABELS.get(feature, feature)}: not reported"
    if feature in REASONS:
        main, ref, f = REASONS[feature]
        return f"{main.format(v=f(value))} ({ref.format(m=f(median))})"
    return f"{F.LABELS.get(feature, feature)}: {value:,.1f} (median {median:,.1f})"


def compute(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """SHAP values for a random sample of the test window."""
    te = F.split(df, "test")
    smp = te.sample(n=min(N_SHAP, len(te)), random_state=C.SEED)
    X = F.lgbm_matrix(smp)
    bst = lgb.Booster(model_file=str(C.MODELS / "lgbm.txt"))
    exp = shap.TreeExplainer(bst)
    sv = exp.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    base = exp.expected_value
    base = float(base[1] if np.ndim(base) else base)
    return smp, pd.DataFrame(sv, columns=X.columns, index=smp.index), base


def run(df: pd.DataFrame, pd_cutoff: float) -> dict:
    smp, sv, base = compute(df)
    imp = sv.abs().mean().sort_values(ascending=False)
    imp_tbl = pd.DataFrame({"feature": imp.index, "label": [F.LABELS.get(c, c) for c in imp.index], "mean_abs_shap": imp.values})
    imp_tbl.to_csv(C.TABLES / "shap_importance.csv", index=False)
    sv.to_parquet(C.INTERIM / "shap_values.parquet")
    smp[["id"] + F.FEATURES].to_parquet(C.INTERIM / "shap_sample.parquet")

    tr = F.split(df, "train")
    medians = {c: float(tr[c].median()) for c in F.NUMERIC}
    cal = json.loads((C.MODELS / "calibration.json").read_text())["lgbm"]
    raw = 1 / (1 + np.exp(-(base + sv.sum(axis=1))))
    logit_raw = np.log(raw / (1 - raw))
    pd_cal = 1 / (1 + np.exp(-(cal[0] + cal[1] * logit_raw)))

    def explain_row(i):
        contrib = sv.loc[i].sort_values(key=np.abs, ascending=False)
        risk_up = sv.loc[i][sv.loc[i] >= MIN_REASON].sort_values(ascending=False)
        risk_down = sv.loc[i][sv.loc[i] <= -MIN_REASON].sort_values()
        row = smp.loc[i]
        return {
            "top": [{"feature": f, "label": F.LABELS.get(f, f), "shap": float(contrib[f]),
                     "value": (None if pd.isna(row[f]) else (str(row[f]) if f in F.CATEGORICAL else float(row[f])))}
                    for f in contrib.index[:8]],
            "other_shap": float(contrib.iloc[8:].sum()),
            "reasons": [reason_text(f, row[f], medians.get(f)) for f in risk_up.index[:3]],
            "strengths": [reason_text(f, row[f], medians.get(f)) for f in risk_down.index[:2]],
        }

    # Two applicants for the waterfalls: a typical approval and a typical decline
    pdv = pd.Series(pd_cal, index=smp.index)

    def typical(lo: float, hi: float):
        """Most typical applicant in a PD percentile band: feature ranks closest to the band's median."""
        idx = pdv[(pdv >= pdv.quantile(lo)) & (pdv <= pdv.quantile(hi))].index
        ranks = smp.loc[idx, F.NUMERIC].rank(pct=True)
        return (ranks - 0.5).abs().mean(axis=1).idxmin()

    approved_i = typical(0.08, 0.12)
    declined_i = typical(0.96, 0.98)
    examples = {}
    for name, i in (("approved", approved_i), ("declined", declined_i)):
        r = smp.loc[i]
        e = explain_row(i)
        examples[name] = {
            "pd": float(pdv[i]), "grade": str(r["sub_grade"]), "int_rate": float(r["int_rate"]) / 100,
            "loan_amnt": float(r["loan_amnt"]), "purpose": str(r["purpose"]),
            "decision": "approve" if pdv[i] <= pd_cutoff else "decline",
            "base_logodds": base, "final_logodds": float(base + sv.loc[i].sum()), **e,
        }

    # ~200 applicants for the app: 120 approvals and 80 declines under the recommended cutoff
    rng = np.random.default_rng(C.SEED)
    appr = pdv[pdv <= pd_cutoff].index.to_numpy()
    decl = pdv[pdv > pd_cutoff].index.to_numpy()
    pick = list(rng.choice(appr, size=min(120, len(appr)), replace=False)) + list(rng.choice(decl, size=min(80, len(decl)), replace=False))
    apps = []
    for k, i in enumerate(sorted(pick, key=lambda j: pdv[j])):
        r = smp.loc[i]
        apps.append({
            "applicant": f"Applicant {k + 1:03d}", "pd": float(pdv[i]), "decision": "approve" if pdv[i] <= pd_cutoff else "decline",
            "sub_grade": str(r["sub_grade"]), "int_rate": float(r["int_rate"]) / 100, "loan_amnt": float(r["loan_amnt"]),
            "purpose": str(r["purpose"]).replace("_", " "), "issue_month": r["issue_d"].strftime("%Y-%m"),
            "outcome": "defaulted" if r["target"] == 1 else "paid in full",
            "base_logodds": base, **explain_row(i),
        })
    C.APP_DATA.mkdir(parents=True, exist_ok=True)
    (C.APP_DATA / "reason_codes.json").write_text(json.dumps({"pd_cutoff": pd_cutoff, "applicants": apps}, indent=1))

    top = imp_tbl.head(5)
    return {
        "n_explained": int(len(smp)),
        "base_logodds": base,
        "top_features": top["feature"].tolist(),
        "top_labels": top["label"].tolist(),
        "top_share_of_total": float(imp.head(5).sum() / imp.sum()),
        "n_features_used": int((imp > 0).sum()),
        "examples": examples,
        "n_app_applicants": len(apps),
        "medians": {k: medians[k] for k in ("fico", "dti", "annual_inc", "loan_to_income")},
    }, smp, sv
