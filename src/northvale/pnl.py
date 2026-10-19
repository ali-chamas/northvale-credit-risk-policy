"""Business layer: cash checks, P&L assumptions, policy curves and the recommendation (notebook 06).

Conventions
- Cash received by the lender = total_pymnt - collection_recovery_fee (the fee goes
  to the collection agency). No time value of money.
- Realized P&L per loan = cash received - funded principal - costs.
- Expected P&L per loan = principal x [(1 - PD) x G(rate band) - PD x L - c]
  with G = net interest earned on loans that pay off, as a share of principal, by
  rate band; L = net loss on defaulted loans as a share of principal; both measured
  on the training window only. c = cost of funds x principal-years outstanding + opex.
- Cutoffs are chosen on the validation window and evaluated once on the test window.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from . import features as F

RATE_EDGES = [0, 8, 11, 14, 17, 20, 100]
RATE_LABELS = ["<8%", "8-11%", "11-14%", "14-17%", "17-20%", "20%+"]

# Editable cost assumptions (not in the data). Low / base / high.
COST_OF_FUNDS = {"low": 0.03, "base": 0.04, "high": 0.06}  # annual rate on principal outstanding
OPEX = {"low": 0.005, "base": 0.01, "high": 0.02}  # servicing and collections, share of principal over the loan's life

MIN_VOLUME = {"conservative": None, "balanced": 0.90, "growth": 0.95}
GRADE_RULES = {"Decline G": ["G"], "Decline F-G": ["F", "G"], "Decline E-G": ["E", "F", "G"], "Decline D-G": ["D", "E", "F", "G"]}
APPROVAL_GRID = np.round(np.arange(0.50, 1.0001, 0.005), 4)


def band(rate_pct: pd.Series) -> pd.Series:
    return pd.cut(rate_pct, RATE_EDGES, labels=RATE_LABELS, right=False).astype(str)


def net_cash(d: pd.DataFrame) -> pd.Series:
    return d["total_pymnt"] - d["collection_recovery_fee"]


# ---------------------------------------------------------------- checks and calibration

def cash_checks(df: pd.DataFrame) -> dict:
    parts = df["total_rec_prncp"] + df["total_rec_int"] + df["total_rec_late_fee"] + df["recoveries"]
    resid = (df["total_pymnt"] - parts).abs()
    good, bad = df[df["target"] == 0], df[df["target"] == 1]
    return {
        "identity_max_abs_residual": float(resid.max()),
        "identity_share_within_1_cent": float((resid <= 0.011).mean()),
        "good_principal_repaid_share": float(good["total_rec_prncp"].sum() / good["funded_amnt"].sum()),
        "bad_principal_repaid_share": float(bad["total_rec_prncp"].sum() / bad["funded_amnt"].sum()),
        "bad_interest_share": float(bad["total_rec_int"].sum() / bad["funded_amnt"].sum()),
        "bad_recoveries_share": float(bad["recoveries"].sum() / bad["funded_amnt"].sum()),
        "bad_collection_fee_share": float(bad["collection_recovery_fee"].sum() / bad["funded_amnt"].sum()),
        "bad_with_recovery_share": float((bad["recoveries"] > 0).mean()),
        "funded_equals_requested_share": float((df["funded_amnt"] == df["loan_amnt"]).mean()),
    }


def calibrate(train: pd.DataFrame) -> dict:
    """G by rate band, L and principal-years, all from the training window (amount-weighted)."""
    t = train.assign(band=band(train["int_rate"]), cash=net_cash(train))
    good, bad = t[t["target"] == 0], t[t["target"] == 1]
    g = good.groupby("band").apply(lambda d: (d["cash"] - d["funded_amnt"]).sum() / d["funded_amnt"].sum(),
                                   include_groups=False)
    g_by_year = good.groupby([good["issue_d"].dt.year, "band"]).apply(
        lambda d: (d["cash"] - d["funded_amnt"]).sum() / d["funded_amnt"].sum(), include_groups=False).unstack()
    lgd = float((bad["funded_amnt"] - bad["cash"]).sum() / bad["funded_amnt"].sum())
    l_by_year = bad.groupby(bad["issue_d"].dt.year).apply(
        lambda d: (d["funded_amnt"] - d["cash"]).sum() / d["funded_amnt"].sum(), include_groups=False)
    l_by_band = bad.groupby("band").apply(
        lambda d: (d["funded_amnt"] - d["cash"]).sum() / d["funded_amnt"].sum(), include_groups=False)
    # Interest received / annual rate = dollar-years of principal outstanding.
    py = float((t["total_rec_int"] / (t["int_rate"] / 100)).sum() / t["funded_amnt"].sum())
    return {
        "G": {b: float(g[b]) for b in RATE_LABELS},
        "G_low": {b: float(g_by_year[b].min()) for b in RATE_LABELS},
        "G_high": {b: float(g_by_year[b].max()) for b in RATE_LABELS},
        "G_all": float((good["cash"] - good["funded_amnt"]).sum() / good["funded_amnt"].sum()),
        "L": lgd,
        "L_low": float(l_by_year.min()),
        "L_high": float(l_by_year.max()),
        "L_by_band": {b: float(l_by_band[b]) for b in RATE_LABELS},
        "principal_years": py,
        "n_good": int(len(good)),
        "n_bad": int(len(bad)),
        "band_default_rate": {b: float(t.loc[t["band"] == b, "target"].mean()) for b in RATE_LABELS},
        "band_n": {b: int((t["band"] == b).sum()) for b in RATE_LABELS},
    }


def realized_params(d: pd.DataFrame) -> dict:
    """G and L as they turned out in a later window (drift check against the training calibration)."""
    t = d.assign(band=band(d["int_rate"]), cash=net_cash(d))
    good, bad = t[t["target"] == 0], t[t["target"] == 1]
    g = good.groupby("band").apply(lambda x: (x["cash"] - x["funded_amnt"]).sum() / x["funded_amnt"].sum(),
                                   include_groups=False)
    return {"G": {b: float(g[b]) for b in RATE_LABELS},
            "L": float((bad["funded_amnt"] - bad["cash"]).sum() / bad["funded_amnt"].sum()),
            "principal_years": float((t["total_rec_int"] / (t["int_rate"] / 100)).sum() / t["funded_amnt"].sum())}


def assumptions(cal: dict, scenario: str = "base") -> dict:
    """One full set of P&L assumptions. 'low'/'high' = favourable / adverse for the lender."""
    fav = scenario == "low"
    adv = scenario == "high"
    G = cal["G_high"] if fav else cal["G_low"] if adv else cal["G"]
    L = cal["L_low"] if fav else cal["L_high"] if adv else cal["L"]
    cof = COST_OF_FUNDS["low" if fav else "high" if adv else "base"]
    opex = OPEX["low" if fav else "high" if adv else "base"]
    return {"G": G, "L": L, "cost_of_funds": cof, "opex": opex, "principal_years": cal["principal_years"],
            "cost_rate": cof * cal["principal_years"] + opex}


# ---------------------------------------------------------------- P&L per loan

def expected_pnl(d: pd.DataFrame, pd_col: str, a: dict) -> np.ndarray:
    g = band(d["int_rate"]).map(a["G"]).to_numpy()
    p = d[pd_col].to_numpy()
    return d["funded_amnt"].to_numpy() * ((1 - p) * g - p * a["L"] - a["cost_rate"])


def realized_pnl(d: pd.DataFrame, a: dict) -> np.ndarray:
    return (net_cash(d) - d["funded_amnt"] - a["cost_rate"] * d["funded_amnt"]).to_numpy()


def book(d: pd.DataFrame, approve: np.ndarray, a: dict, pd_col: str = "pd_lgbm") -> dict:
    """Outcome of approving the rows where `approve` is True."""
    s = d[approve]
    amt = s["funded_amnt"]
    bad = s["target"] == 1
    loss = (s.loc[bad, "funded_amnt"] - net_cash(s.loc[bad])).sum()
    return {
        "n": int(len(s)),
        "approval_rate": float(len(s) / len(d)),
        "volume_share": float(amt.sum() / d["funded_amnt"].sum()),
        "amount": float(amt.sum()),
        "default_rate": float(bad.mean()),
        "n_defaults": int(bad.sum()),
        "net_credit_loss": float(loss),
        "realized_profit": float(realized_pnl(s, a).sum()),
        "expected_profit": float(expected_pnl(s, pd_col, a).sum()),
    }


def curve(d: pd.DataFrame, pd_col: str, a: dict, grid=APPROVAL_GRID) -> pd.DataFrame:
    """Approve the safest x% by PD, for x in the grid."""
    order = np.argsort(d[pd_col].to_numpy(), kind="stable")
    n = len(d)
    exp = expected_pnl(d, pd_col, a)[order]
    real = realized_pnl(d, a)[order]
    y = d["target"].to_numpy()[order]
    amt = d["funded_amnt"].to_numpy()[order]
    pdv = d[pd_col].to_numpy()[order]
    ce, cr, cy, ca = np.cumsum(exp), np.cumsum(real), np.cumsum(y), np.cumsum(amt)
    rows = []
    for x in grid:
        k = max(int(round(x * n)), 1)
        rows.append({"approval_rate": k / n, "pd_cutoff": float(pdv[k - 1]), "default_rate": cy[k - 1] / k,
                     "expected_profit": ce[k - 1], "realized_profit": cr[k - 1], "volume_share": ca[k - 1] / ca[-1]})
    return pd.DataFrame(rows)


def choose_cutoffs(valid: pd.DataFrame, pd_col: str, a: dict) -> dict:
    """Pick PD cutoffs on the validation window by expected profit (no outcomes needed)."""
    cv = curve(valid, pd_col, a)
    out = {}
    for name, floor in MIN_VOLUME.items():
        c = cv if floor is None else cv[cv["approval_rate"] >= floor - 1e-9]
        r = c.loc[c["expected_profit"].idxmax()]
        out[name] = {"pd_cutoff": float(r["pd_cutoff"]), "valid_approval_rate": float(r["approval_rate"]),
                     "min_volume": floor}
    return out


# ---------------------------------------------------------------- run

def run(df: pd.DataFrame) -> dict:
    train = F.split(df, "train")
    scored = pd.read_parquet(C.SCORED)
    valid, test = scored[scored["split"] == "valid"].copy(), scored[scored["split"] == "test"].copy()
    cal = calibrate(train)
    a = assumptions(cal, "base")

    checks = cash_checks(df)
    # 1. Does the expected P&L reproduce the realized P&L on the test book?
    sq = book(test, np.ones(len(test), bool), a)
    sq_valid = book(valid, np.ones(len(valid), bool), a)

    # 2. Policy curves on test for each model (approve the safest x%)
    curves = {}
    for m in ("lgbm", "scorecard", "grade"):
        cv = curve(test, f"pd_{m}", a)
        cv.to_csv(C.TABLES / f"policy_curve_{m}_test.csv", index=False)
        curves[m] = cv

    # 3. Grade rules (the outdated policy family) vs the model at the same approval rate
    grade_rows = {}
    for name, declined in GRADE_RULES.items():
        approve = ~test["grade"].isin(declined).to_numpy()
        g = book(test, approve, a)
        k = int(approve.sum())
        same = np.zeros(len(test), bool)
        same[np.argsort(test["pd_lgbm"].to_numpy(), kind="stable")[:k]] = True
        mdl = book(test, same, a)
        grade_rows[name] = {"grade_rule": g, "lgbm_same_rate": mdl}

    # 4. Cutoffs chosen on validation, applied to test
    cuts = choose_cutoffs(valid, "pd_lgbm", a)
    options = {}
    for name, c in cuts.items():
        approve = test["pd_lgbm"].to_numpy() <= c["pd_cutoff"]
        b = book(test, approve, a)
        b.update(c)
        b["default_rate_change"] = b["default_rate"] - sq["default_rate"]
        b["default_rate_rel_change"] = b["default_rate"] / sq["default_rate"] - 1
        b["profit_change"] = b["realized_profit"] - sq["realized_profit"]
        b["expected_profit_change"] = b["expected_profit"] - sq["expected_profit"]
        b["loss_avoided"] = sq["net_credit_loss"] - b["net_credit_loss"]
        b["defaults_avoided"] = sq["n_defaults"] - b["n_defaults"]
        b["volume_lost"] = sq["amount"] - b["amount"]
        declined = test[~approve]
        b["declined_n"] = int(len(declined))
        b["declined_default_rate"] = float(declined["target"].mean())
        b["declined_realized_profit"] = float(realized_pnl(declined, a).sum())
        b["good_loans_declined"] = int((declined["target"] == 0).sum())
        b["good_profit_forgone"] = float(realized_pnl(declined[declined["target"] == 0], a).sum())
        options[name] = b
    # Scorecard at the balanced approval rate (simpler model, same volume)
    sc_cut = choose_cutoffs(valid, "pd_scorecard", a)["balanced"]
    sc_bal = book(test, test["pd_scorecard"].to_numpy() <= sc_cut["pd_cutoff"], a, "pd_scorecard")
    sc_bal["profit_change"] = sc_bal["realized_profit"] - sq["realized_profit"]

    rec = options["balanced"]
    # Alternative rule: rank by expected margin per dollar (uses the price), same number approved.
    margin = expected_pnl(test, "pd_lgbm", a) / test["funded_amnt"].to_numpy()
    by_margin = np.zeros(len(test), bool)
    by_margin[np.argsort(-margin, kind="stable")[: rec["n"]]] = True
    pa = book(test, by_margin, a)
    price_aware = {"realized_profit": pa["realized_profit"], "default_rate": pa["default_rate"],
                   "approval_rate": pa["approval_rate"],
                   "gap_vs_pd_cutoff": pa["realized_profit"] - rec["realized_profit"]}
    months = 6
    annual = 12 / months
    # Scenario sensitivity: same cutoff (chosen under base assumptions), different P&L assumptions
    scen = {}
    for s in ("low", "base", "high"):
        aa = assumptions(cal, s)
        approve = test["pd_lgbm"].to_numpy() <= rec["pd_cutoff"]
        base_all = book(test, np.ones(len(test), bool), aa)
        pol = book(test, approve, aa)
        scen[s] = {
            "G_mid": aa["G"]["11-14%"], "L": aa["L"], "cost_of_funds": aa["cost_of_funds"], "opex": aa["opex"],
            "cost_rate": aa["cost_rate"],
            "status_quo_expected_profit": base_all["expected_profit"], "policy_expected_profit": pol["expected_profit"],
            "expected_profit_change": pol["expected_profit"] - base_all["expected_profit"],
            "realized_profit_change": pol["realized_profit"] - base_all["realized_profit"],
            "status_quo_realized_profit": base_all["realized_profit"],
            "annual_realized_profit_change": annual * (pol["realized_profit"] - base_all["realized_profit"]),
            "annual_expected_profit_change": annual * (pol["expected_profit"] - base_all["expected_profit"]),
        }
    pd.DataFrame(scen).T.to_csv(C.TABLES / "scenario_sensitivity.csv")

    # Band table for the README
    band_tbl = pd.DataFrame({
        "rate_band": RATE_LABELS,
        "G_base": [cal["G"][b] for b in RATE_LABELS],
        "G_low": [cal["G_low"][b] for b in RATE_LABELS],
        "G_high": [cal["G_high"][b] for b in RATE_LABELS],
        "L_by_band": [cal["L_by_band"][b] for b in RATE_LABELS],
        "train_default_rate": [cal["band_default_rate"][b] for b in RATE_LABELS],
        "n_train": [cal["band_n"][b] for b in RATE_LABELS],
    })
    band_tbl["breakeven_pd"] = (band_tbl["G_base"] - a["cost_rate"]) / (band_tbl["G_base"] + cal["L"])
    band_tbl.to_csv(C.TABLES / "pnl_assumptions_by_band.csv", index=False)

    comp = [{"policy": "Approve all (status quo)", **sq}]
    for name, r in grade_rows.items():
        comp.append({"policy": name, **r["grade_rule"]})
        comp.append({"policy": f"LightGBM at same approval as {name}", **r["lgbm_same_rate"]})
    for name, r in options.items():
        comp.append({"policy": f"LightGBM {name}", **{k: r[k] for k in sq}})
    comp.append({"policy": "Scorecard balanced", **{k: sc_bal[k] for k in sq}})
    pd.DataFrame(comp).to_csv(C.TABLES / "policy_comparison_test.csv", index=False)

    # Defaults for the app
    C.APP_DATA.mkdir(parents=True, exist_ok=True)
    app_assump = {
        "G_by_band": cal["G"], "rate_bands": RATE_LABELS, "rate_edges": RATE_EDGES,
        "L": cal["L"], "principal_years": cal["principal_years"],
        "cost_of_funds": COST_OF_FUNDS, "opex": OPEX, "L_range": [cal["L_low"], cal["L_high"]],
        "cutoffs": {k: v["pd_cutoff"] for k, v in cuts.items()},
        "grade_rules": GRADE_RULES, "months_in_test": months,
    }
    (C.APP_DATA / "assumptions.json").write_text(json.dumps(app_assump, indent=2))
    app = test[["issue_d", "grade", "sub_grade", "int_rate", "funded_amnt", "target", "pd_lgbm", "pd_scorecard", "pd_grade"]].copy()
    app["net_cash"] = net_cash(test).astype("float32")
    app["issue_d"] = app["issue_d"].dt.strftime("%Y-%m")
    for c in ("int_rate", "pd_lgbm", "pd_scorecard", "pd_grade"):
        app[c] = app[c].astype("float32")
    app["funded_amnt"] = app["funded_amnt"].astype("float32")
    app["target"] = app["target"].astype("int8")
    app.reset_index(drop=True).to_parquet(C.APP_DATA / "scored_test.parquet", index=False)

    test_amount = float(test["funded_amnt"].sum())
    return {
        "cash": checks,
        "assumptions": {
            "G": cal["G"], "G_low": cal["G_low"], "G_high": cal["G_high"], "G_all": cal["G_all"],
            "L": cal["L"], "L_low": cal["L_low"], "L_high": cal["L_high"], "L_by_band": cal["L_by_band"],
            "principal_years": cal["principal_years"],
            "cost_of_funds": COST_OF_FUNDS, "opex": OPEX,
            "cost_rate_base": a["cost_rate"],
            "breakeven_pd": {b: float(v) for b, v in zip(RATE_LABELS, band_tbl["breakeven_pd"], strict=True)},
            "band_default_rate": cal["band_default_rate"],
        },
        "status_quo": sq,
        "status_quo_valid": sq_valid,
        "expected_vs_realized_gap": (sq["expected_profit"] - sq["realized_profit"]) / test_amount,
        "drift": {"test": realized_params(test), "valid": realized_params(valid)},
        "grade_rules": grade_rows,
        "options": options,
        "scorecard_balanced": sc_bal,
        "price_aware": price_aware,
        "recommended": "balanced",
        "scenarios": scen,
        "annualisation": {
            "months_in_test": months, "factor": annual,
            "annual_loans": annual * len(test), "annual_amount": annual * test_amount,
        },
        "annual": {
            "profit_change": annual * rec["profit_change"],
            "loss_avoided": annual * rec["loss_avoided"],
            "defaults_avoided": annual * rec["defaults_avoided"],
            "volume_lost": annual * rec["volume_lost"],
            "expected_profit_change": annual * rec["expected_profit_change"],
        },
        "test_amount": test_amount,
        "test_n": int(len(test)),
    }
