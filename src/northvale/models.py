"""Incumbent benchmark, WoE logistic scorecard and LightGBM (notebooks 04 and 05).

Every model is fitted on the training window (2012-2014), tuned and recalibrated
on the validation window (2015 H1) and evaluated once on the test window
(2015 H2). Lending Club's grade, sub-grade and interest rate are never features.
"""

from __future__ import annotations

import json
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from . import config as C
from . import features as F

# Scorecard scaling: 600 points at 50:1 good:bad odds, 20 points to double the odds.
BASE_POINTS, BASE_ODDS, PDO = 600, 50, 20
FACTOR = PDO / np.log(2)
OFFSET = BASE_POINTS - FACTOR * np.log(BASE_ODDS)

IV_MIN = 0.02  # "weak" predictive power threshold in scorecard practice
CORR_MAX = 0.70  # drop the weaker of two characteristics with |r| above this (on WoE)
MIN_GAIN = 0.0005  # stepwise: validation AUC gain needed to add a characteristic
MAX_CHARS = 15  # keep the scorecard readable

LGB_BASE = {
    "objective": "binary",
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "max_bin": 255,
    "verbose": -1,
    "seed": C.SEED,
    "num_threads": C.N_JOBS,
    "deterministic": True,
    "force_row_wise": True,
}
LGB_GRID = [
    {"num_leaves": nl, "min_data_in_leaf": mc}
    for nl in (15, 31, 63)
    for mc in (200, 1000)
]


# ---------------------------------------------------------------- metrics

def ks_stat(y, score) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index of `actual` against `expected`, deciles of `expected`."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def evaluate(y, pd_hat, score=None) -> dict:
    """Discrimination on the score, calibration on the PD."""
    s = pd_hat if score is None else score
    auc = roc_auc_score(y, s)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_stat(y, s),
        "brier": float(brier_score_loss(y, pd_hat)),
        "mean_pd": float(np.mean(pd_hat)),
        "observed": float(np.mean(y)),
    }


def decile_table(y, pd_hat, n: int = 10) -> pd.DataFrame:
    """Default rate and mean PD by decile of predicted risk (1 = safest)."""
    d = pd.DataFrame({"y": y, "pd": pd_hat})
    d["decile"] = pd.qcut(d["pd"].rank(method="first"), n, labels=range(1, n + 1)).astype(int)
    return d.groupby("decile").agg(n=("y", "size"), default_rate=("y", "mean"), mean_pd=("pd", "mean")).reset_index()


def brier_skill(brier: float, base_rate: float) -> float:
    return 1 - brier / (base_rate * (1 - base_rate))


# ---------------------------------------------------------------- calibration

def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Platt:
    """Logistic recalibration of a score on the validation window: PD = sigmoid(a + b * logit(p))."""

    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
        self.a, self.b = float(self.lr.intercept_[0]), float(self.lr.coef_[0, 0])
        return self

    def predict(self, p):
        return 1 / (1 + np.exp(-(self.a + self.b * logit(p))))


# ---------------------------------------------------------------- incumbent

def grade_pd_map(train: pd.DataFrame) -> pd.Series:
    """Sub-grade -> training-window default rate (the incumbent's implied PD)."""
    return train.groupby("sub_grade")["target"].mean()


# ---------------------------------------------------------------- scorecard

def fit_binning(train: pd.DataFrame) -> BinningProcess:
    numeric_params = {c: {"monotonic_trend": "auto_asc_desc"} for c in F.NUMERIC}
    bp = BinningProcess(
        variable_names=F.FEATURES,
        categorical_variables=F.CATEGORICAL,
        max_n_prebins=20,
        min_prebin_size=0.02,
        min_bin_size=0.03,
        binning_fit_params=numeric_params,
        n_jobs=C.N_JOBS,
    )
    bp.fit(train[F.FEATURES], train["target"].to_numpy())
    return bp


def _lr(X, y):
    return LogisticRegression(C=1.0, max_iter=2000).fit(X, y)


def select_characteristics(bp: BinningProcess, Xw_tr: pd.DataFrame, y_tr, Xw_va: pd.DataFrame, y_va) -> tuple[list[str], pd.DataFrame]:
    """IV filter -> correlation filter -> stepwise in IV order on validation AUC -> sign check."""
    summ = bp.summary().set_index("name")
    iv = summ["iv"].astype(float).sort_values(ascending=False)
    log = pd.DataFrame({"iv": iv})
    log["step"] = ""
    cands = [c for c in iv.index if iv[c] >= IV_MIN]
    log.loc[iv.index[iv < IV_MIN], "step"] = f"dropped: IV < {IV_MIN}"
    corr = Xw_tr[cands].corr().abs()
    kept: list[str] = []
    for c in cands:  # in IV order, so the stronger of a correlated pair survives
        clash = [k for k in kept if corr.loc[c, k] > CORR_MAX]
        if clash:
            log.loc[c, "step"] = f"dropped: |r| > {CORR_MAX} with {clash[0]}"
        else:
            kept.append(c)
    chosen: list[str] = []
    best = 0.5
    for c in kept:
        if len(chosen) >= MAX_CHARS:
            log.loc[c, "step"] = f"not added: scorecard capped at {MAX_CHARS}"
            continue
        trial = chosen + [c]
        m = _lr(Xw_tr[trial], y_tr)
        coefs = m.coef_[0]
        auc = roc_auc_score(y_va, m.predict_proba(Xw_va[trial])[:, 1])
        if auc - best >= MIN_GAIN and np.all(coefs < 0):
            chosen, best = trial, auc
            log.loc[c, "step"] = f"added (validation AUC {auc:.4f})"
        elif not np.all(coefs < 0):
            log.loc[c, "step"] = "not added: would flip a coefficient sign"
        else:
            log.loc[c, "step"] = f"not added: AUC gain < {MIN_GAIN}"
    log = log.reset_index().rename(columns={"index": "characteristic", "name": "characteristic"})
    return chosen, log


def points_table(bp: BinningProcess, model: LogisticRegression, chars: list[str]) -> pd.DataFrame:
    """Points per bin: score = OFFSET + FACTOR * ln(good:bad odds), intercept spread evenly."""
    b0 = float(model.intercept_[0])
    n = len(chars)
    rows = []
    for c, bj in zip(chars, model.coef_[0], strict=True):
        bt = bp.get_binned_variable(c).binning_table.build()
        bt = bt[bt.index != "Totals"]
        for _, r in bt.iterrows():
            if r["Count"] == 0:
                continue
            woe = float(r["WoE"])
            pts = -FACTOR * bj * woe + (OFFSET - FACTOR * b0) / n
            rows.append({
                "characteristic": c, "label": F.LABELS.get(c, c), "bin": str(r["Bin"]), "count": int(r["Count"]),
                "event_rate": float(r["Event rate"]), "woe": woe, "coefficient": float(bj), "points": round(pts),
            })
    return pd.DataFrame(rows)


def score_points(bp: BinningProcess, model: LogisticRegression, chars: list[str], X: pd.DataFrame) -> np.ndarray:
    Xw = bp.transform(X[F.FEATURES], metric="woe")[chars]
    logodds_bad = model.intercept_[0] + Xw.to_numpy() @ model.coef_[0]
    return OFFSET - FACTOR * logodds_bad


# ---------------------------------------------------------------- LightGBM

def fit_lgbm(Xtr, ytr, Xva, yva, params: dict) -> lgb.Booster:
    dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=False)
    p = {**LGB_BASE, **params, "metric": "auc"}
    return lgb.train(
        p, dtr, num_boost_round=3000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def search_lgbm(Xtr, ytr, Xva, yva) -> tuple[lgb.Booster, dict, pd.DataFrame]:
    rows, best = [], None
    for g in LGB_GRID:
        t0 = time.time()
        bst = fit_lgbm(Xtr, ytr, Xva, yva, g)
        auc = roc_auc_score(yva, bst.predict(Xva, num_iteration=bst.best_iteration))
        rows.append({**g, "best_iteration": bst.best_iteration, "valid_auc": auc})
        print(f"  lgbm {g} -> valid AUC {auc:.4f} at {bst.best_iteration} trees ({time.time() - t0:.0f}s)")
        if best is None or auc > best[1]:
            best = (bst, auc, g)
    return best[0], best[2], pd.DataFrame(rows)


# ---------------------------------------------------------------- run

def run(df: pd.DataFrame) -> dict:
    C.MODELS.mkdir(parents=True, exist_ok=True)
    tr, va, te = (F.split(df, s) for s in ("train", "valid", "test"))
    ytr, yva, yte = (d["target"].to_numpy() for d in (tr, va, te))

    # 1. Incumbent: Lending Club sub-grade
    gmap = grade_pd_map(tr)
    raw_grade = {k: d["sub_grade"].map(gmap).fillna(gmap.max()).to_numpy() for k, d in (("valid", va), ("test", te))}
    cal_grade = Platt().fit(raw_grade["valid"], yva)
    pd_grade_te = cal_grade.predict(raw_grade["test"])
    res = {"grade": evaluate(yte, pd_grade_te, score=te["sub_grade_rank"].to_numpy())}

    # 2. Scorecard
    t0 = time.time()
    bp = fit_binning(tr)
    Xw_tr = bp.transform(tr[F.FEATURES], metric="woe")
    Xw_va = bp.transform(va[F.FEATURES], metric="woe")
    chars, sel_log = select_characteristics(bp, Xw_tr, ytr, Xw_va, yva)
    sc = _lr(Xw_tr[chars], ytr)
    pts = points_table(bp, sc, chars)
    pts.to_csv(C.TABLES / "scorecard_points.csv", index=False)
    sel_log.to_csv(C.TABLES / "scorecard_selection.csv", index=False)
    s_tr, s_va, s_te = (score_points(bp, sc, chars, d) for d in (tr, va, te))
    raw_sc = {k: 1 / (1 + np.exp((s - OFFSET) / FACTOR)) for k, s in (("train", s_tr), ("valid", s_va), ("test", s_te))}
    cal_sc = Platt().fit(raw_sc["valid"], yva)
    pd_sc_te = cal_sc.predict(raw_sc["test"])
    res["scorecard"] = evaluate(yte, pd_sc_te, score=-s_te)
    res["scorecard"]["brier_uncalibrated"] = float(brier_score_loss(yte, raw_sc["test"]))
    res["scorecard"]["mean_pd_uncalibrated"] = float(raw_sc["test"].mean())
    res["scorecard"]["psi_train_test"] = psi(s_tr, s_te)
    res["scorecard"]["n_characteristics"] = len(chars)
    res["scorecard"]["characteristics"] = chars
    res["scorecard"]["n_iv_pass"] = int((bp.summary()["iv"].astype(float) >= IV_MIN).sum())
    print(f"models: scorecard {len(chars)} characteristics, test AUC {res['scorecard']['auc']:.4f} ({time.time() - t0:.0f}s)")
    with open(C.MODELS / "scorecard.pkl", "wb") as f:
        pickle.dump({"binning": bp, "model": sc, "chars": chars, "platt": (cal_sc.a, cal_sc.b)}, f)

    # 3. LightGBM
    t0 = time.time()
    Xtr, Xva, Xte = (F.lgbm_matrix(d) for d in (tr, va, te))
    bst, best_params, grid = search_lgbm(Xtr, ytr, Xva, yva)
    grid.to_csv(C.TABLES / "lgbm_search.csv", index=False)
    bst.save_model(str(C.MODELS / "lgbm.txt"), num_iteration=bst.best_iteration)
    raw_lgb = {k: bst.predict(X, num_iteration=bst.best_iteration) for k, X in (("train", Xtr), ("valid", Xva), ("test", Xte))}
    cal_lgb = Platt().fit(raw_lgb["valid"], yva)
    pd_lgb_te = cal_lgb.predict(raw_lgb["test"])
    res["lgbm"] = evaluate(yte, pd_lgb_te)
    res["lgbm"]["brier_uncalibrated"] = float(brier_score_loss(yte, raw_lgb["test"]))
    res["lgbm"]["mean_pd_uncalibrated"] = float(raw_lgb["test"].mean())
    res["lgbm"]["psi_train_test"] = psi(raw_lgb["train"], raw_lgb["test"])
    res["lgbm"]["best_params"] = best_params
    res["lgbm"]["best_iteration"] = int(bst.best_iteration)
    res["lgbm"]["valid_auc"] = float(grid["valid_auc"].max())
    res["lgbm"]["valid_auc_range"] = float(grid["valid_auc"].max() - grid["valid_auc"].min())
    res["lgbm"]["n_features"] = len(F.FEATURES)
    res["lgbm"]["train_auc"] = float(roc_auc_score(ytr, raw_lgb["train"]))
    res["grade"]["psi_train_test"] = psi(tr["sub_grade_rank"], te["sub_grade_rank"])
    res["scorecard"]["train_auc"] = float(roc_auc_score(ytr, -s_tr))
    with open(C.MODELS / "calibration.json", "w") as f:
        json.dump({"lgbm": [cal_lgb.a, cal_lgb.b], "scorecard": [cal_sc.a, cal_sc.b], "grade": [cal_grade.a, cal_grade.b]}, f)

    # Does the model add information beyond the grade? Within-grade AUC and a stacked check.
    within = {}
    for name, s in (("lgbm", pd_lgb_te), ("scorecard", pd_sc_te)):
        aucs, ws = [], []
        for idx in te.groupby("grade").indices.values():
            yy = yte[idx]
            if yy.min() == yy.max() or len(idx) < 500:
                continue
            aucs.append(roc_auc_score(yy, s[idx]))
            ws.append(len(idx))
        within[name] = float(np.average(aucs, weights=ws))
    stack = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.c_[logit(cal_lgb.predict(raw_lgb["valid"])), va["sub_grade_rank"]], yva)
    pd_stack = stack.predict_proba(np.c_[logit(pd_lgb_te), te["sub_grade_rank"]])[:, 1]
    res["incremental"] = {
        "within_grade_auc_lgbm": within["lgbm"],
        "within_grade_auc_scorecard": within["scorecard"],
        "auc_lgbm_plus_grade": float(roc_auc_score(yte, pd_stack)),
        "auc_lift_lgbm_vs_grade": res["lgbm"]["auc"] - res["grade"]["auc"],
        "auc_lift_scorecard_vs_grade": res["scorecard"]["auc"] - res["grade"]["auc"],
        "auc_lift_lgbm_vs_scorecard": res["lgbm"]["auc"] - res["scorecard"]["auc"],
        "corr_lgbm_grade": float(pd.Series(pd_lgb_te).corr(te["sub_grade_rank"].reset_index(drop=True), method="spearman")),
    }

    # Decile tables and calibration (test)
    for name, p in (("lgbm", pd_lgb_te), ("scorecard", pd_sc_te), ("grade", pd_grade_te)):
        decile_table(yte, p).to_csv(C.TABLES / f"deciles_{name}_test.csv", index=False)
    dt = decile_table(yte, pd_lgb_te)
    res["lgbm"]["decile_1_rate"] = float(dt["default_rate"].iloc[0])
    res["lgbm"]["decile_10_rate"] = float(dt["default_rate"].iloc[-1])
    res["lgbm"]["decile_ratio"] = float(dt["default_rate"].iloc[-1] / dt["default_rate"].iloc[0])
    res["lgbm"]["max_decile_calibration_gap"] = float((dt["mean_pd"] - dt["default_rate"]).abs().max())
    for k in ("grade", "scorecard", "lgbm"):
        res[k]["brier_skill"] = brier_skill(res[k]["brier"], res[k]["observed"])

    # Scored validation + test rows for SHAP, policy and the app
    keep = ["id", "issue_d", "split", "target", "loan_status", "grade", "sub_grade", "sub_grade_rank", "int_rate",
            "loan_amnt", *F.CASH]
    out = []
    for _split, d, rg, rs, rl, s in (("valid", va, raw_grade["valid"], raw_sc["valid"], raw_lgb["valid"], s_va),
                                   ("test", te, raw_grade["test"], raw_sc["test"], raw_lgb["test"], s_te)):
        o = d[keep].copy()
        o["pd_grade"] = cal_grade.predict(rg)
        o["pd_scorecard"] = cal_sc.predict(rs)
        o["pd_lgbm"] = cal_lgb.predict(rl)
        o["lgbm_raw"] = rl
        o["score_points"] = s
        out.append(o)
    pd.concat(out).to_parquet(C.SCORED, index=False)

    res["scaling"] = {"base_points": BASE_POINTS, "base_odds": BASE_ODDS, "pdo": PDO,
                      "factor": float(FACTOR), "offset": float(OFFSET)}
    res["selection"] = {"iv_min": IV_MIN, "corr_max": CORR_MAX, "min_gain": MIN_GAIN, "max_chars": MAX_CHARS}
    return res
