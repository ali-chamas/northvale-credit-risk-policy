"""`make analysis`: every number in reports/metrics.json, every table and every figure.

Order: scope -> audit -> models -> leakage demo -> P&L and policy -> SHAP -> monitoring.
Runs in about five minutes on a laptop (LightGBM uses 4 threads).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from . import audit, eda, explain, models, pnl, scope
from . import config as C
from . import features as F
from . import figures as G
from . import metrics as M

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _step(name: str, t0: float) -> float:
    t = time.time()
    print(f"analysis: {name} done ({t - t0:.0f}s)")
    return t


def main() -> None:
    t = t_start = time.time()
    for d in (C.FIGURES, C.TABLES, C.APP_DATA, C.MODELS):
        d.mkdir(parents=True, exist_ok=True)
    M.reset()
    df = F.sample()

    # 1. Scope, censoring, vintages
    s = scope.run()
    for k in ("data", "censoring", "vintage"):
        M.update(k, s[k])
    G.censoring(scope.censoring_table())
    G.censoring(scope.censoring_table(), name="censoring_by_vintage_wide", size="finding")
    G.default_by_quarter(scope.vintage_table("quarter"))
    gy = scope.grade_year_table()
    G.grade_drift(gy, "default_rate", "Default rate by grade", "36-month loans, 2013 vs 2015 vintage", "grade_drift_default")
    G.grade_drift(gy, "avg_rate", "Average interest rate by grade", "36-month loans, 2013 vs 2015 vintage", "grade_drift_price")
    G.eda_drivers({c: eda.rate_table(df, c) for c in ("fico", "dti", "loan_to_income", "acc_open_past_24mths")})
    t = _step("scope", t)

    # 2. Feature audit + single-feature AUC screen
    a, _ = audit.run(df)
    M.update("audit", a)
    G.auc_screen(pd.read_csv(C.DOCS / "feature_audit.csv"), pd.read_csv(C.TABLES / "single_feature_auc_derived.csv"), audit.FLAG_AUC)
    t = _step("audit", t)

    # 3. Models
    res = models.run(df)
    scored = pd.read_parquet(C.SCORED)
    test = scored[scored["split"] == "test"]
    G.capture_curve(test)
    _, wg = G.within_grade(test)
    res["within_grade"] = {r["grade"]: {"safest": r["safest"], "riskiest": r["riskiest"], "all": r["all"]} for _, r in wg.iterrows()}
    ratios = wg["riskiest"] / wg["safest"]
    res["within_grade_ratio_min"], res["within_grade_ratio_max"] = float(ratios.min()), float(ratios.max())
    deciles = {m: pd.read_csv(C.TABLES / f"deciles_{m}_test.csv") for m in ("lgbm", "scorecard")}
    G.calibration(deciles)
    G.decile_bars(deciles["lgbm"])
    G.model_bars(res)
    # capture at 10% declined, per model
    y = test["target"].to_numpy()
    cap = {}
    for m, score in (("lgbm", test["pd_lgbm"]), ("scorecard", test["pd_scorecard"]),
                     ("grade", test["sub_grade_rank"] + test["pd_grade"] * 1e-6)):
        order = np.argsort(-score.to_numpy(), kind="stable")
        cap[m] = float(y[order][: int(0.10 * len(y))].sum() / y.sum())
    res["capture_at_10pct"] = cap
    M.update("models", res)
    t = _step("models", t)

    # 4. Leakage demonstration
    leak, rows = audit.leakage_demo(df, res["lgbm"]["best_params"], res["lgbm"]["auc"])
    M.update("leakage", leak)
    G.leakage_demo(rows)
    t = _step("leakage demo", t)

    # 5. P&L and policy
    p = pnl.run(df)
    M.update("pnl", p)
    curves = {m: pd.read_csv(C.TABLES / f"policy_curve_{m}_test.csv") for m in ("lgbm", "scorecard", "grade")}
    G.policy_profit(curves, p["options"], p["status_quo"], p["grade_rules"])
    G.policy_default(curves, p["options"])
    G.expected_vs_realized(curves["lgbm"])
    bands = pd.read_csv(C.TABLES / "pnl_assumptions_by_band.csv")
    tb = test.assign(band=pnl.band(test["int_rate"])).groupby("band")["target"].mean().reindex(pnl.RATE_LABELS)
    G.breakeven(bands, pd.DataFrame({"default_rate": tb.values}))
    G.scenarios(pd.read_csv(C.TABLES / "scenario_sensitivity.csv", index_col=0))
    t = _step("P&L and policy", t)

    # 6. SHAP and reason codes
    rec_cut = p["options"]["balanced"]["pd_cutoff"]
    sh, smp, sv = explain.run(df, rec_cut)
    M.update("shap", sh)
    imp = pd.read_csv(C.TABLES / "shap_importance.csv")
    G.shap_bar(imp)
    G.shap_beeswarm(smp, sv, imp)
    num_top = [f for f in imp["feature"] if f in F.NUMERIC][:3]
    G.shap_dependence(smp, sv, num_top)
    G.waterfall(sh["examples"]["approved"], "Why this applicant scores as low risk", "shap_waterfall_approved")
    G.waterfall(sh["examples"]["declined"], "Why this applicant scores as high risk", "shap_waterfall_declined")
    t = _step("SHAP", t)

    # 7. Monitoring baseline: PSI of the score and of the top features, train vs test
    tr, te = F.split(df, "train"), F.split(df, "test")
    feat_psi = {f: models.psi(tr[f].dropna(), te[f].dropna()) for f in imp["feature"].head(10) if f in F.NUMERIC}
    M.update("monitoring", {
        "score_psi_lgbm": res["lgbm"]["psi_train_test"],
        "score_psi_scorecard": res["scorecard"]["psi_train_test"],
        "feature_psi": feat_psi,
        "max_feature_psi": max(feat_psi.values()),
        "max_feature_psi_name": max(feat_psi, key=feat_psi.get),
        "psi_alert": 0.25, "psi_watch": 0.10,
    })
    pd.Series(feat_psi, name="psi_train_vs_test").to_csv(C.TABLES / "feature_psi.csv")
    audit.write_data_dictionary(res["scorecard"]["characteristics"])
    _step("monitoring and data dictionary", t)
    print(f"analysis: finished in {time.time() - t_start:.0f}s -> {C.METRICS.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
