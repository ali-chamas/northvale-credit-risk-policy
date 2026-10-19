"""Every chart in the README, notebooks and deck. All go through viz.py.

Chart titles say what is measured; slide titles carry the conclusion.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from . import config as C
from . import viz

ACC = viz.use_style("northvale")
SECOND = viz.SERIES[1]  # orange: the second series when two are compared
THIRD = viz.SERIES[2]


def _money_m(v, _=None):
    return f"${v / 1e6:,.0f}M"


def _src(fig, text="Source: Lending Club accepted loans, 36-month, issued 2012-2015"):
    viz.note(fig, text)


# ---------------------------------------------------------------- 01 scope

def censoring(cens: pd.DataFrame, name="censoring_by_vintage", size="half"):
    fig, ax = viz.figure(size)
    d = cens[cens["issue_year"].between(2010, 2018)]
    ax.axvspan(2011.5, 2015.5, color=viz.GRID, alpha=0.6, lw=0)
    ax.text(2013.5, 0.97, "Scope:\n36-month,\n2012-2015", ha="center", va="top", fontsize=9.5, color=viz.INK_2,
            transform=ax.get_xaxis_transform())
    for term, color, lab in ((C.TERM, ACC, "36-month"), ("60 months", viz.MUTED, "60-month")):
        s = d[d["term"] == term].sort_values("issue_year")
        ax.plot(s["issue_year"], s["non_terminal_share"], color=color, marker="o", ms=5, label=lab)
        pt = s[s["issue_year"] == 2016].iloc[0]
        off = (-10, 8) if term != C.TERM else (10, -4)
        ax.annotate(lab, (pt["issue_year"], pt["non_terminal_share"]), xytext=off, textcoords="offset points",
                    ha="right" if term != C.TERM else "left", va="center", fontsize=10, color=viz.INK_2)
    viz.pct_axis(ax)
    ax.set_ylim(0, 1)
    ax.set_xlim(2009.6, 2019.2)
    ax.set_xticks(range(2010, 2019))
    viz.titles(ax, "Loans still running at the data snapshot", "Share of loans not yet Fully Paid or Charged Off, by issue year")
    _src(fig, "Source: Lending Club accepted loans 2007-2018Q4, all terms")
    return viz.save(fig, name, C.FIGURES)


def default_by_quarter(vq: pd.DataFrame, name="default_rate_by_quarter", size="finding"):
    fig, ax = viz.figure(size)
    x = np.arange(len(vq))
    lo, hi = vq["default_rate"].idxmin(), vq["default_rate"].idxmax()
    colors = [ACC if s != "train" else viz.CONTEXT for s in vq["split"]]
    bars = ax.bar(x, vq["default_rate"], color=colors, width=0.7)
    viz.label_bars(ax, [bars[lo], bars[hi], bars[len(vq) - 1]], fmt=lambda v: f"{v * 100:.1f}%")
    ax.set_xticks(x, [p.replace("-", "\n") for p in vq["period"]], fontsize=9)
    ax.yaxis.set_major_locator(MultipleLocator(0.04))
    viz.pct_axis(ax)
    ax.set_ylim(0, vq["default_rate"].max() * 1.22)
    ymax = ax.get_ylim()[1]
    for lab, a, b in (("Training window", 0, 11), ("Validation", 12, 13), ("Test", 14, 15)):
        ax.annotate("", (a - 0.4, ymax * 0.95), (b + 0.4, ymax * 0.95),
                    arrowprops={"arrowstyle": "-", "color": viz.AXIS, "lw": 1})
        ax.text((a + b) / 2, ymax * 0.965, lab, ha="center", va="bottom", fontsize=9.5, color=viz.INK_2)
    viz.titles(ax, "Default rate by issue quarter", "36-month individual loans, resolved outcomes only")
    _src(fig)
    return viz.save(fig, name, C.FIGURES)


def grade_drift(gy: pd.DataFrame, value: str, title: str, subtitle: str, name: str, size="half"):
    fig, ax = viz.figure(size)
    grades = list("ABCDEF")
    a = gy[gy["issue_year"] == 2013].set_index("grade").loc[grades, value]
    b = gy[gy["issue_year"] == 2015].set_index("grade").loc[grades, value]
    x = np.arange(len(grades))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, a, width=w, color=viz.CONTEXT, label="2013 vintage")
    bars = ax.bar(x + w / 2 + 0.01, b, width=w, color=ACC, label="2015 vintage")
    old = ax.containers[0]
    viz.label_bars(ax, [old[2], bars[2], old[4], bars[4]], fmt=lambda v: f"{v * 100:.1f}%", size=9.5)
    ax.set_xticks(x, grades)
    ax.set_xlabel("Lending Club grade")
    viz.pct_axis(ax)
    ax.set_ylim(0, max(a.max(), b.max()) * 1.15)
    ax.legend(loc="upper left")
    viz.titles(ax, title, subtitle)
    _src(fig, "Source: Lending Club, 36-month loans; grade G omitted (15 loans in 2013)")
    return viz.save(fig, name, C.FIGURES)


# ---------------------------------------------------------------- 02 audit

CAT_LABEL = {
    "post_origination": "Post-origination (dropped)",
    "lc_pricing": "Lending Club pricing (benchmark only)",
    "application": "Application field (kept)",
    "application_derived": "Application field (kept)",
}


def auc_screen(audit: pd.DataFrame, derived: pd.DataFrame, flag: float, name="leakage_auc_screen", size="finding", top=16):
    a = audit.copy()
    a["auc"] = pd.to_numeric(a["single_feature_auc_train"], errors="coerce")
    a = a[a["category"].isin(["post_origination", "lc_pricing", "application"])]
    a = a[a["column"] != "grade"]  # sub_grade shown instead
    d = derived.rename(columns={"single_feature_auc_train": "auc"})
    d = d[~d["column"].str.contains("sub_grade")].assign(category="application_derived")
    s = pd.concat([a[["column", "category", "auc"]], d[["column", "category", "auc"]]]).dropna()
    s = s.sort_values("auc", ascending=False).head(top).iloc[::-1]
    colors = {"post_origination": ACC, "lc_pricing": SECOND, "application": viz.CONTEXT, "application_derived": viz.CONTEXT}
    fig, ax = viz.figure(size)
    bars = ax.barh(s["column"], s["auc"] - 0.5, left=0.5, color=[colors[c] for c in s["category"]], height=0.68)
    for b, v in zip(bars, s["auc"], strict=True):
        ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=9, color=viz.INK_2)
    ax.axvline(flag, color=viz.INK_2, lw=1)
    n_app = int((s["category"] != "post_origination").sum())
    ax.text(flag + 0.004, n_app - 1, f"Flag at {flag:.2f}", fontsize=9.5, color=viz.INK_2, va="center")
    viz.hbar_style(ax)
    ax.set_xlim(0.5, 0.95)
    ax.tick_params(axis="y", labelsize=9.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in ("post_origination", "lc_pricing", "application")]
    ax.legend(handles, [CAT_LABEL[k] for k in ("post_origination", "lc_pricing", "application")], loc="lower right")
    viz.titles(ax, "Single-field AUC on the training window (top 16 columns)", "Each column alone, binned; 0.5 = no information")
    _src(fig, "Source: Lending Club, 36-month loans issued 2012-2014 (training window)")
    return viz.save(fig, name, C.FIGURES)


def leakage_demo(rows: pd.DataFrame, name="leakage_demo", size="half"):
    fig, ax = viz.figure(size)
    colors = [viz.CONTEXT if i == 0 else ACC for i in range(len(rows))]
    bars = ax.bar(rows["label"], rows["test_auc"], color=colors, width=0.6)
    viz.label_bars(ax, bars, fmt=lambda v: f"{v:.3f}")
    ax.set_ylim(0.5, 1.08)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis="x", labelsize=9.5)
    viz.titles(ax, "Test AUC of the same LightGBM model", "Application fields only vs adding one post-origination field")
    _src(fig, "Test window: 36-month loans issued Jul-Dec 2015")
    return viz.save(fig, name, C.FIGURES)


# ---------------------------------------------------------------- 04-05 models

MODEL_NAMES = {"lgbm": "LightGBM", "scorecard": "Scorecard", "grade": "LC sub-grade"}
MODEL_COLORS = {"lgbm": ACC, "scorecard": THIRD, "grade": SECOND}


def capture_curve(test: pd.DataFrame, name="default_capture_curve", size="half", max_x=0.5):
    fig, ax = viz.figure(size)
    y = test["target"].to_numpy()
    n = len(y)
    xs = np.linspace(0, max_x, 101)
    for m in ("grade", "scorecard", "lgbm"):
        score = test["sub_grade_rank"].to_numpy() + test["pd_grade"].to_numpy() * 1e-6 if m == "grade" else test[f"pd_{m}"].to_numpy()
        order = np.argsort(-score, kind="stable")
        cum = np.cumsum(y[order]) / y.sum()
        ys = [cum[max(int(x * n) - 1, 0)] if x > 0 else 0 for x in xs]
        ax.plot(xs, ys, color=MODEL_COLORS[m], label=MODEL_NAMES[m], lw=2.2 if m == "lgbm" else 1.8)
    ax.plot([0, max_x], [0, max_x], color=viz.AXIS, lw=1, label="Random")
    viz.pct_axis(ax)
    viz.pct_axis(ax, axis="x")
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Share of applicants declined, riskiest first")
    ax.legend(loc="lower right")
    viz.titles(ax, "Share of defaults avoided", "Test window, Jul-Dec 2015")
    _src(fig, "Test window: 161,997 resolved 36-month loans issued Jul-Dec 2015")
    return viz.save(fig, name, C.FIGURES)


def within_grade(test: pd.DataFrame, name="within_grade_spread", size="finding"):
    rows = []
    for g in "ABCDE":
        d = test[test["grade"] == g]
        q = pd.qcut(d["pd_lgbm"].rank(method="first"), 5, labels=False)
        rows.append({"grade": g, "safest": d.loc[q == 0, "target"].mean(), "riskiest": d.loc[q == 4, "target"].mean(),
                     "all": d["target"].mean()})
    t = pd.DataFrame(rows).iloc[::-1]
    fig, ax = viz.figure(size)
    yk = np.arange(len(t))
    ax.hlines(yk, t["safest"], t["riskiest"], color=viz.CONTEXT, lw=3)
    ax.scatter(t["all"], yk, marker="|", s=260, color=viz.INK_2, label="Grade average", zorder=3)
    ax.scatter(t["safest"], yk, s=90, color=ACC, label="Safest 20% by LightGBM", zorder=4, edgecolors="white", linewidths=1.5)
    ax.scatter(t["riskiest"], yk, s=90, color=SECOND, label="Riskiest 20% by LightGBM", zorder=4, edgecolors="white", linewidths=1.5)
    for k, (_, r) in enumerate(t.iterrows()):
        ax.annotate(f"{r['safest'] * 100:.1f}%", (r["safest"], k), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9.5, color=viz.INK_2)
        ax.annotate(f"{r['riskiest'] * 100:.1f}%", (r["riskiest"], k), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9.5, color=viz.INK_2)
    ax.set_yticks(yk, [f"Grade {g}" for g in t["grade"]])
    viz.hbar_style(ax)
    viz.pct_axis(ax, axis="x")
    ax.set_xlim(0, t["riskiest"].max() * 1.12)
    ax.set_ylim(-0.6, len(t) - 0.3)
    ax.legend(loc="upper right", ncols=1)
    viz.titles(ax, "Default rate inside each Lending Club grade", "Test window; applicants split into fifths by LightGBM PD within their grade")
    _src(fig, "Test window: 36-month loans issued Jul-Dec 2015; grades F-G omitted (1% of loans)")
    return viz.save(fig, name, C.FIGURES), pd.DataFrame(rows)


def calibration(deciles: dict, name="calibration_test", size="half"):
    fig, ax = viz.figure(size)
    hi = max(d["default_rate"].max() for d in deciles.values()) * 1.08
    ax.plot([0, hi], [0, hi], color=viz.AXIS, lw=1, label="Perfect calibration")
    for m, d in deciles.items():
        ax.plot(d["mean_pd"], d["default_rate"], marker="o", ms=6, color=MODEL_COLORS[m], label=MODEL_NAMES[m])
    viz.pct_axis(ax)
    viz.pct_axis(ax, axis="x")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Mean predicted PD (decile)")
    ax.grid(axis="x", visible=True)
    ax.legend(loc="upper left")
    viz.titles(ax, "Observed default rate vs predicted PD", "Test window, by decile; PDs recalibrated on validation")
    _src(fig, "Test window: 36-month loans issued Jul-Dec 2015")
    return viz.save(fig, name, C.FIGURES)


def decile_bars(d: pd.DataFrame, name="default_by_decile_lgbm", size="half"):
    fig, ax = viz.figure(size)
    colors = [viz.CONTEXT] * 9 + [ACC]
    bars = ax.bar(d["decile"].astype(str), d["default_rate"], color=colors, width=0.7)
    viz.label_bars(ax, [bars[0], bars[-1]], fmt=lambda v: f"{v * 100:.1f}%")
    viz.pct_axis(ax)
    ax.set_ylim(0, d["default_rate"].max() * 1.15)
    ax.set_xlabel("LightGBM risk decile (1 = safest)")
    viz.titles(ax, "Default rate by LightGBM risk decile", "Test window, Jul-Dec 2015")
    _src(fig, "Test window: 36-month loans issued Jul-Dec 2015")
    return viz.save(fig, name, C.FIGURES)


def model_bars(res: dict, name="model_comparison", size="half"):
    fig, ax = viz.figure(size)
    ms = ["grade", "scorecard", "lgbm"]
    x = np.arange(2)
    w = 0.26
    for k, m in enumerate(ms):
        vals = [res[m]["auc"], res[m]["ks"]]
        bars = ax.bar(x + (k - 1) * (w + 0.02), vals, width=w, color=MODEL_COLORS[m], label=MODEL_NAMES[m])
        viz.label_bars(ax, bars, fmt=lambda v: f"{v:.3f}", size=9)
    ax.set_xticks(x, ["AUC", "KS"])
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper right")
    viz.titles(ax, "Discrimination on the test window", "Higher is better; same 161,997 loans for every model")
    _src(fig, "Test window: 36-month loans issued Jul-Dec 2015")
    return viz.save(fig, name, C.FIGURES)


# ---------------------------------------------------------------- 05 SHAP

def _lab(f):
    from .features import LABELS
    return LABELS.get(f, f)


def shap_bar(imp: pd.DataFrame, name="shap_importance", size="half", top=12):
    t = imp.head(top).iloc[::-1]
    fig, ax = viz.figure(size)
    colors = [viz.CONTEXT] * (len(t) - 3) + [ACC] * 3
    ax.barh(t["label"], t["mean_abs_shap"], color=colors, height=0.66)
    viz.hbar_style(ax)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.set_xlabel("Mean |SHAP| (log-odds of default)")
    viz.titles(ax, "What moves the LightGBM score", f"Top {top} features, {20000:,} test applicants")
    _src(fig, "Test window sample, seed 42")
    return viz.save(fig, name, C.FIGURES)


def shap_beeswarm(smp: pd.DataFrame, sv: pd.DataFrame, imp: pd.DataFrame, name="shap_beeswarm", size="finding", top=10, n=5000):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", viz.SEQ_BLUE[:1] + viz.SEQ_BLUE[3:])
    feats = imp["feature"].head(top).tolist()
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(len(sv), size=min(n, len(sv)), replace=False)
    fig, ax = viz.figure(size)
    for k, f in enumerate(feats[::-1]):
        s = sv[f].to_numpy()[idx]
        v = smp[f].iloc[idx]
        # vertical jitter by local density so the swarm shows the distribution
        bins = np.round(s / 0.02)
        _, inv, cnt = np.unique(bins, return_inverse=True, return_counts=True)
        jitter = (rng.random(len(s)) - 0.5) * np.minimum(cnt[inv] / cnt.max(), 1) * 0.8
        if pd.api.types.is_numeric_dtype(v):
            r = v.rank(pct=True).to_numpy()
            col = cmap(np.nan_to_num(r, nan=0.0))
            col[np.isnan(r)] = (0.75, 0.75, 0.75, 1)
        else:
            col = [viz.MUTED] * len(s)
        ax.scatter(s, k + jitter, c=col, s=4, alpha=0.8, linewidths=0, rasterized=True)
    ax.axvline(0, color=viz.AXIS, lw=1)
    ax.set_yticks(range(len(feats)), [_lab(f) for f in feats[::-1]], fontsize=9.5)
    viz.hbar_style(ax)
    ax.set_xlabel("SHAP value: effect on log-odds of default (right = riskier)")
    sm = plt.cm.ScalarMappable(cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01, aspect=25)
    cb.set_ticks([0, 1], labels=["low", "high"])
    cb.outline.set_visible(False)
    cb.set_label("Feature value (percentile)", color=viz.INK_2, fontsize=9.5)
    cb.ax.tick_params(labelsize=9.5)
    viz.titles(ax, "How each feature value pushes the default score", f"{len(idx):,} test applicants, top {top} features; grey = categorical or missing")
    _src(fig, "Test window sample, seed 42")
    return viz.save(fig, name, C.FIGURES)


def shap_dependence(smp: pd.DataFrame, sv: pd.DataFrame, feats: list[str], name="shap_dependence", size="full"):
    import matplotlib.pyplot as plt2
    fig, axes = plt2.subplots(1, len(feats), figsize=viz.SIZES[size], layout="constrained", sharey=True)
    for ax, f in zip(axes, feats, strict=True):
        x = smp[f].astype(float)
        s = sv[f]
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        keep = x.between(lo, hi)
        ax.scatter(x[keep], s[keep], s=3, color=viz.CONTEXT, alpha=0.5, linewidths=0, rasterized=True)
        b = pd.qcut(x[keep], 20, duplicates="drop")
        mean = pd.DataFrame({"x": x[keep], "s": s[keep], "b": b}).groupby("b", observed=True).mean()
        ax.plot(mean["x"], mean["s"], color=ACC, lw=2.2)
        ax.axhline(0, color=viz.AXIS, lw=1)
        ax.set_xlabel(_lab(f))
        ax.set_title(_lab(f), fontsize=11.5, loc="left", pad=8)
        if f in ("annual_inc", "bc_open_to_buy", "tot_hi_cred_lim", "loan_amnt", "avg_cur_bal", "total_bc_limit"):
            viz.thousands_axis(ax, axis="x", prefix="$")
        if f == "loan_to_income":
            viz.pct_axis(ax, axis="x")
    axes[0].set_ylabel("SHAP value (log-odds)")
    fig.suptitle("Effect of a feature's value on the default score (line = average)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=viz.INK)
    viz.note(fig, "Test window sample of 20,000 applicants; 1st-99th percentile of each feature shown")
    return viz.save(fig, name, C.FIGURES)


def waterfall(ex: dict, title: str, name: str, size="half", k=7):
    from .fmt import pct
    top = ex["top"][:k]
    other = sum(t["shap"] for t in ex["top"][k:]) + ex["other_shap"]
    labels = [_short_value(t) for t in top] + ["All other features"]
    vals = [t["shap"] for t in top] + [other]
    fig, ax = viz.figure(size)
    start = ex["base_logodds"]
    cur = start
    neg, pos = viz.DIVERGING[0], viz.DIVERGING[-1]
    for i, v in enumerate(vals):
        ax.barh(i, v, left=cur, color=pos if v > 0 else neg, height=0.62)
        ax.text(max(cur, cur + v) + 0.02, i, f"{v:+.2f}", va="center", fontsize=9.5, color=viz.INK_2)
        cur += v
    ax.axvline(start, color=viz.AXIS, lw=1)
    ax.set_yticks(range(len(labels)), labels, fontsize=9.5)
    ax.invert_yaxis()
    viz.hbar_style(ax)
    lo = min(start, cur, *(start + np.cumsum(vals))) - 0.15
    hi = max(start, cur, *(start + np.cumsum(vals))) + 0.35
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Log-odds of default (red raises risk, blue lowers it)")
    # Title at figure level: long y labels would push an axes-level title off the canvas.
    fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.84))
    fig.text(0.01, 0.975, title, fontsize=13, fontweight="bold", color=viz.INK, ha="left", va="top")
    fig.text(0.01, 0.905, f"Sub-grade {ex['grade']}, PD {pct(ex['pd'])}: {ex['decision']} at the recommended cutoff",
             fontsize=10.5, color=viz.MUTED, ha="left", va="top")
    fig.text(0.01, 0.01, "Starting point = average applicant; test window, Jul-Dec 2015", fontsize=8.5, color=viz.MUTED, ha="left", va="bottom")
    return viz.save(fig, name, C.FIGURES)


SHORT = {
    "revol_bal_to_income": "Revolving balance / income", "mo_sin_old_rev_tl_op": "Oldest revolving account (months)",
    "credit_history_months": "Credit history (months)", "percent_bc_gt_75": "Bankcards above 75% of limit (%)",
    "mths_since_recent_bc": "Newest bankcard (months ago)", "mths_since_recent_inq": "Last inquiry (months ago)",
    "num_tl_op_past_12m": "Accounts opened, last 12m", "acc_open_past_24mths": "Accounts opened, last 24m",
    "total_rev_hi_lim": "Total revolving limit", "loan_to_income": "Loan / annual income",
    "total_il_high_credit_limit": "Instalment credit limit", "mo_sin_rcnt_tl": "Newest account (months ago)",
    "mo_sin_old_il_acct": "Oldest instalment account (months)", "mo_sin_rcnt_rev_tl_op": "Newest revolving account (months)",
}


def _short_value(t: dict) -> str:
    v, f = t["value"], t["feature"]
    lab = SHORT.get(f, _lab(f))
    if v is None:
        return f"{lab}: missing"
    if isinstance(v, str):
        return f"{lab}: {v.replace('_', ' ').lower()}"
    if f in ("annual_inc", "bc_open_to_buy", "tot_hi_cred_lim", "loan_amnt", "avg_cur_bal", "revol_bal", "total_bc_limit",
             "tot_cur_bal", "total_rev_hi_lim", "total_bal_ex_mort", "total_il_high_credit_limit"):
        return f"{lab}: ${v:,.0f}"
    if f in ("loan_to_income", "revol_bal_to_income"):
        return f"{lab}: {v * 100:.0f}%"
    return f"{lab}: {v:,.0f}" if abs(v) >= 10 or float(v).is_integer() else f"{lab}: {v:.1f}"


# ---------------------------------------------------------------- 06 policy

def policy_profit(curves: dict, options: dict, sq: dict, grade_pts: dict, name="policy_profit_curve", size="finding"):
    fig, ax = viz.figure(size)
    for m in ("grade", "scorecard", "lgbm"):
        c = curves[m]
        c = c[c["approval_rate"] >= 0.6]
        ax.plot(c["approval_rate"], c["realized_profit"], color=MODEL_COLORS[m], lw=2.4 if m == "lgbm" else 1.8,
                label=f"{MODEL_NAMES[m]} cutoff")
    gx = [g["grade_rule"]["approval_rate"] for g in grade_pts.values()]
    gy = [g["grade_rule"]["realized_profit"] for g in grade_pts.values()]
    ax.scatter(gx, gy, marker="D", s=46, color=SECOND, zorder=4, edgecolors="white", linewidths=1, label="Grade rules (decline D-G, E-G, ...)")
    ax.scatter([1.0], [sq["realized_profit"]], s=70, color=viz.INK_2, zorder=5, label="Approve all (status quo)")
    for nm, o in options.items():
        ax.scatter([o["approval_rate"]], [o["realized_profit"]], s=110 if nm == "balanced" else 60, color=ACC, zorder=6,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(nm.capitalize(), (o["approval_rate"], o["realized_profit"]), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=10, color=viz.INK, fontweight="bold" if nm == "balanced" else "normal")
    ax.axhline(0, color=viz.AXIS, lw=1)
    viz.pct_axis(ax, axis="x")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_money_m))
    ax.set_xlabel("Approval rate (share of applicants approved)")
    ax.legend(loc="lower left", fontsize=9.5)
    viz.titles(ax, "Realized profit on the test book by approval rate", "Cash received minus principal and costs, Jul-Dec 2015 loans; base cost assumptions")
    _src(fig, "Test window: 161,997 loans, $2.1B; cutoffs chosen on Jan-Jun 2015, never on test")
    return viz.save(fig, name, C.FIGURES)


def policy_default(curves: dict, options: dict, name="policy_default_curve", size="half"):
    fig, ax = viz.figure(size)
    for m in ("grade", "scorecard", "lgbm"):
        c = curves[m]
        c = c[c["approval_rate"] >= 0.6]
        ax.plot(c["approval_rate"], c["default_rate"], color=MODEL_COLORS[m], lw=2.2 if m == "lgbm" else 1.8, label=MODEL_NAMES[m])
    o = options["balanced"]
    ax.scatter([o["approval_rate"]], [o["default_rate"]], s=90, color=ACC, zorder=5, edgecolors="white", linewidths=1.5)
    ax.annotate(f"Balanced: {o['default_rate'] * 100:.1f}%", (o["approval_rate"], o["default_rate"]), xytext=(-8, 12),
                textcoords="offset points", ha="right", fontsize=10, color=viz.INK)
    viz.pct_axis(ax)
    viz.pct_axis(ax, axis="x")
    ax.set_xlabel("Approval rate")
    ax.legend(loc="upper left")
    viz.titles(ax, "Default rate among approved loans", "Test window, by approval rate")
    _src(fig, "Test window: 36-month loans issued Jul-Dec 2015")
    return viz.save(fig, name, C.FIGURES)


def expected_vs_realized(curve: pd.DataFrame, name="expected_vs_realized", size="half"):
    fig, ax = viz.figure(size)
    c = curve[curve["approval_rate"] >= 0.6]
    ax.plot(c["approval_rate"], c["expected_profit"], color=viz.MUTED, lw=1.8, label="Expected (PD x assumptions)")
    ax.plot(c["approval_rate"], c["realized_profit"], color=ACC, lw=2.2, label="Realized (actual cash)")
    ax.axhline(0, color=viz.AXIS, lw=1)
    viz.pct_axis(ax, axis="x")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_money_m))
    ax.set_xlabel("Approval rate (LightGBM cutoff)")
    ax.legend(loc="lower left")
    viz.titles(ax, "Expected vs realized profit, LightGBM cutoffs", "Test window; base assumptions calibrated on 2012-2014")
    _src(fig, "Test window: 36-month loans issued Jul-Dec 2015")
    return viz.save(fig, name, C.FIGURES)


def breakeven(bands: pd.DataFrame, test_bands: pd.DataFrame, name="breakeven_by_band", size="half"):
    fig, ax = viz.figure(size)
    x = np.arange(len(bands))
    ax.bar(x, bands["breakeven_pd"], color=viz.CONTEXT, width=0.6, label="Break-even PD (price covers risk below it)")
    ax.plot(x, test_bands["default_rate"], color=ACC, marker="o", ms=7, lw=0, label="Observed default rate, test")
    for i, b in enumerate(bands["breakeven_pd"]):
        ax.text(i, 0.008, f"{b * 100:.0f}%", ha="center", va="bottom", fontsize=9.5, color=viz.INK)
    ax.set_xticks(x, bands["rate_band"])
    ax.set_xlabel("Interest rate band")
    viz.pct_axis(ax)
    ax.set_ylim(0, max(bands["breakeven_pd"].max(), test_bands["default_rate"].max()) * 1.2)
    ax.legend(loc="upper left", fontsize=9.5)
    viz.titles(ax, "Break-even PD vs observed default rate", "Base assumptions; break-even = (G - costs) / (G + L)")
    _src(fig, "G and L measured on 2012-2014; observed rates on Jul-Dec 2015")
    return viz.save(fig, name, C.FIGURES)


def scenarios(scen: pd.DataFrame, name="scenario_impact", size="half"):
    fig, ax = viz.figure(size)
    labels = {"low": "Favourable", "base": "Base", "high": "Adverse"}
    t = scen.loc[["low", "base", "high"]]
    colors = [viz.CONTEXT, ACC, viz.CONTEXT]
    bars = ax.bar([labels[i] for i in t.index], t["annual_realized_profit_change"], color=colors, width=0.55)
    viz.label_bars(ax, bars, fmt=lambda v: f"+${v / 1e6:,.1f}M")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_money_m))
    ax.set_ylim(0, t["annual_realized_profit_change"].max() * 1.2)
    viz.titles(ax, "Annual profit gain of the balanced policy vs approve-all", "Realized test-window gain x2, by cost and loss scenario")
    _src(fig, "Same cutoff in every scenario; scenarios change costs, G and L")
    return viz.save(fig, name, C.FIGURES)
