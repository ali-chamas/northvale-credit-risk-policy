"""Northvale Credit: approval policy simulator for a credit manager.

Reads only the small files in app/data/ written by `make analysis`:
- scored_test.parquet  test-window loans (Jul-Dec 2015) with each model's PD and the cash outcome
- assumptions.json     P&L assumptions calibrated on 2012-2014, cutoffs chosen on Jan-Jun 2015
- reason_codes.json    SHAP explanations for 200 sample applicants

No LightGBM or SHAP at runtime. The P&L formulas mirror src/northvale/pnl.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data"
ACCENT, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
RISK_UP, RISK_DOWN = "#c22d2c", "#1c5cab"
MODELS = {"LightGBM (recommended)": "pd_lgbm", "Logistic scorecard": "pd_scorecard", "Lending Club sub-grade": "pd_grade"}

st.set_page_config(page_title="Northvale Credit · Policy simulator", layout="wide")


@st.cache_data
def load():
    df = pd.read_parquet(DATA / "scored_test.parquet")
    a = json.loads((DATA / "assumptions.json").read_text())
    rc = json.loads((DATA / "reason_codes.json").read_text())
    edges = a["rate_edges"]
    df["band"] = pd.cut(df["int_rate"], edges, labels=a["rate_bands"], right=False).astype(str)
    return df, a, rc


def money(v: float) -> str:
    sign = "-" if v < 0 else ""
    v = abs(v)
    return f"{sign}${v / 1e6:,.1f}M" if v >= 1e6 else f"{sign}${v:,.0f}"


def md_escape(text: str) -> str:
    return text.replace("$", "\\$")


def pnl_columns(df: pd.DataFrame, pd_col: str, a: dict, L: float, g_mult: float, cost_rate: float):
    g = df["band"].map(a["G_by_band"]).to_numpy() * g_mult
    p = df[pd_col].to_numpy()
    amt = df["funded_amnt"].to_numpy()
    expected = amt * ((1 - p) * g - p * L - cost_rate)
    realized = df["net_cash"].to_numpy() - amt - cost_rate * amt
    return expected, realized


def summarize(df, approve, expected, realized):
    s = df[approve]
    bad = s["target"] == 1
    return {
        "Approval rate": approve.mean(),
        "Volume ($)": s["funded_amnt"].sum(),
        "Default rate": bad.mean() if len(s) else np.nan,
        "Net credit loss ($)": (s.loc[bad, "funded_amnt"] - s.loc[bad, "net_cash"]).sum(),
        "Expected profit ($)": expected[approve].sum(),
        "Realized profit ($)": realized[approve].sum(),
    }


def curve(df, pd_col, expected, realized):
    order = np.argsort(df[pd_col].to_numpy(), kind="stable")
    n = len(df)
    ce, cr = np.cumsum(expected[order]), np.cumsum(realized[order])
    cy = np.cumsum(df["target"].to_numpy()[order])
    rows = []
    for x in np.arange(0.50, 1.0001, 0.01):
        k = max(int(round(x * n)), 1)
        rows.append({"Approval rate": k / n, "Expected profit": ce[k - 1], "Realized profit": cr[k - 1],
                     "Default rate": cy[k - 1] / k})
    return pd.DataFrame(rows)


df, A, RC = load()

st.title("Northvale Credit: approval policy simulator")
st.info(
    "**Case study: fictional company.** Northvale Credit is a fictional lender invented for this project. "
    "The simulator uses public Lending Club loan data (36-month loans issued July to December 2015, all resolved) "
    "and was built for learning purposes. No real client, revenue or employer is represented. "
    "Profit figures are projections under the assumptions in the sidebar.",
)

# ---------------------------------------------------------------- controls
with st.sidebar:
    st.header("Policy")
    model_label = st.selectbox("Risk model", list(MODELS), index=0)
    pd_col = MODELS[model_label]
    mode = st.radio("Set the cutoff by", ["Target approval rate", "Maximum PD"], horizontal=True)
    if mode == "Target approval rate":
        target = st.slider("Approve the safest share of applicants", 0.50, 1.00, 0.91, 0.01, format="%.2f")
        pd_cut = float(np.quantile(df[pd_col], target)) if target < 1 else 1.0
    else:
        default_cut = A["cutoffs"]["balanced"] if pd_col == "pd_lgbm" else 0.28
        pd_cut = st.slider("Decline applicants with PD above", 0.05, 0.60, round(default_cut, 3), 0.005, format="%.3f")
    grade_rule = st.selectbox("Compare with grade rule", list(A["grade_rules"]), index=2)

    st.header("Assumptions")
    L = st.slider("Loss on a defaulted loan (% of principal)", 0.25, 0.50, round(A["L"], 3), 0.005, format="%.3f",
                  help="Calibrated on 2012-2014 defaults: principal not recovered net of interest received and recoveries.")
    g_mult = st.slider("Interest earned on loans that pay off (x calibrated)", 0.80, 1.20, 1.00, 0.01,
                       help="Scales G, the net interest earned per $ on fully paid loans, measured by rate band on 2012-2014.")
    cof = st.slider("Cost of funds (annual)", 0.00, 0.10, A["cost_of_funds"]["base"], 0.0025, format="%.4f")
    opex = st.slider("Operating cost (% of principal, loan life)", 0.000, 0.040, A["opex"]["base"], 0.0025, format="%.4f")
    cost_rate = cof * A["principal_years"] + opex
    st.caption(f"Cost per $ lent = cost of funds x {A['principal_years']:.2f} principal-years + opex = {cost_rate * 100:.2f}%")

expected, realized = pnl_columns(df, pd_col, A, L, g_mult, cost_rate)
approve = df[pd_col].to_numpy() <= pd_cut
sq = summarize(df, np.ones(len(df), bool), expected, realized)
gr = summarize(df, ~df["grade"].isin(A["grade_rules"][grade_rule]).to_numpy(), expected, realized)
pol = summarize(df, approve, expected, realized)

tab_policy, tab_app, tab_about = st.tabs(["Policy impact", "Applicant explanations", "Method and limits"])

# ---------------------------------------------------------------- policy tab
with tab_policy:
    st.subheader(f"Selected policy: decline when {model_label.split(' (')[0]} PD > {pd_cut * 100:.1f}%")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Approval rate", f"{pol['Approval rate'] * 100:.1f}%", f"{(pol['Approval rate'] - 1) * 100:+.1f} pp vs approve-all")
    c2.metric("Default rate, approved loans", f"{pol['Default rate'] * 100:.1f}%",
              f"{(pol['Default rate'] - sq['Default rate']) * 100:+.1f} pp", delta_color="inverse")
    c3.metric("Realized profit, Jul-Dec 2015 book", money(pol["Realized profit ($)"]),
              f"{money(pol['Realized profit ($)'] - sq['Realized profit ($)'])} vs approve-all")
    c4.metric("Expected profit (assumptions)", money(pol["Expected profit ($)"]),
              f"{money(pol['Expected profit ($)'] - sq['Expected profit ($)'])} vs approve-all")
    st.caption(
        f"Test book: {len(df):,} loans, {money(sq['Volume ($)'])} issued July to December 2015. "
        "Realized = cash actually received minus principal and costs (no time value). "
        "Expected = the selected model's PD x assumptions. Multiply by 2 for an annual view."
    )

    table = pd.DataFrame({"Approve all (status quo)": sq, grade_rule: gr, "Selected policy": pol}).T
    fmt = table.copy().astype(object)
    for c in table.columns:
        if "rate" in c.lower():
            fmt[c] = table[c].map(lambda v: f"{v * 100:.1f}%")
        else:
            fmt[c] = table[c].map(money)
    st.dataframe(fmt, width="stretch")

    cv = curve(df, pd_col, expected, realized)
    long = cv.melt("Approval rate", ["Realized profit", "Expected profit"], var_name="Measure", value_name="Profit")
    point = pd.DataFrame({"Approval rate": [pol["Approval rate"]], "Profit": [pol["Realized profit ($)"]]})
    x = alt.X("Approval rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0.5, 1.0]))
    lines = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=x, y=alt.Y("Profit:Q", axis=alt.Axis(format="$,.2s"), title="Profit on the test book"),
        color=alt.Color("Measure:N", scale=alt.Scale(domain=["Realized profit", "Expected profit"], range=[ACCENT, GREY]),
                        legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("Approval rate:Q", format=".1%"), "Measure:N", alt.Tooltip("Profit:Q", format="$,.0f")],
    )
    dot = alt.Chart(point).mark_point(size=140, filled=True, color=ACCENT, stroke="white", strokeWidth=2).encode(x=x, y="Profit:Q")
    dr = alt.Chart(cv).mark_line(strokeWidth=2.5, color=ACCENT).encode(
        x=x, y=alt.Y("Default rate:Q", axis=alt.Axis(format="%"), title="Default rate among approved"),
        tooltip=[alt.Tooltip("Approval rate:Q", format=".1%"), alt.Tooltip("Default rate:Q", format=".1%")],
    )
    dr_dot = alt.Chart(pd.DataFrame({"Approval rate": [pol["Approval rate"]], "Default rate": [pol["Default rate"]]})).mark_point(
        size=140, filled=True, color=ACCENT, stroke="white", strokeWidth=2).encode(x=x, y="Default rate:Q")
    col_l, col_r = st.columns(2)
    col_l.markdown("**Profit by approval rate** (dot = selected policy)")
    col_l.altair_chart((lines + dot).properties(height=320), width="stretch")
    col_r.markdown("**Default rate among approved loans**")
    col_r.altair_chart((dr + dr_dot).properties(height=320), width="stretch")

# ---------------------------------------------------------------- applicant tab
with tab_app:
    apps = RC["applicants"]
    n_ok = sum(a["decision"] == "approve" for a in apps)
    st.markdown(
        f"Explanations come from SHAP values of the LightGBM model for {len(apps)} test-window applicants "
        f"({n_ok} approved and {len(apps) - n_ok} declined at the recommended cutoff). Reasons are the features that "
        "raised the applicant's score the most compared with the average applicant: the basis for an adverse-action notice."
    )
    labels = [f"{x['applicant']} · {x['sub_grade']} · PD {x['pd'] * 100:.1f}%" for x in apps]
    i = st.selectbox("Applicant", range(len(apps)), format_func=lambda k: labels[k], index=len(apps) - 5)
    x = apps[i]
    # The explanations are for LightGBM, so the decision follows the sidebar cutoff only when LightGBM is selected.
    if pd_col == "pd_lgbm":
        decision, where = ("Approve" if x["pd"] <= pd_cut else "Decline"), "Decision at current cutoff"
    else:
        decision, where = x["decision"].capitalize(), "Decision at recommended cutoff"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LightGBM PD", f"{x['pd'] * 100:.1f}%")
    c2.metric(where, decision)
    c3.metric("Lending Club sub-grade / rate", f"{x['sub_grade']} / {x['int_rate'] * 100:.1f}%")
    c4.metric("Loan amount", f"${x['loan_amnt']:,.0f}")
    st.caption(f"Purpose: {x['purpose']}. Issued {x['issue_month']}. Actual outcome (not known at decision time): {x['outcome']}.")

    left, right = st.columns([3, 2])
    bars = pd.DataFrame(
        [{"Feature": f"{t['label']}: {t['display']}", "SHAP": t["shap"]} for t in x["top"]]
        + [{"Feature": "All other features", "SHAP": x["other_shap"]}]
    )
    bars["Effect"] = np.where(bars["SHAP"] > 0, "Raises risk", "Lowers risk")
    chart = alt.Chart(bars).mark_bar().encode(
        x=alt.X("SHAP:Q", title="Contribution to log-odds of default"),
        y=alt.Y("Feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color("Effect:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[RISK_UP, RISK_DOWN]),
                        legend=alt.Legend(orient="bottom")),
        tooltip=["Feature:N", alt.Tooltip("SHAP:Q", format="+.3f")],
    ).properties(height=330)
    left.altair_chart(chart, width="stretch")
    right.markdown("**Main reasons the score is above average**")
    # Escape "$" so two amounts on one line are not read as a LaTeX formula
    right.markdown("\n".join(f"{k + 1}. {md_escape(r)}" for k, r in enumerate(x["reasons"])) or "None above the 0.03 threshold.")
    right.markdown("**Factors that lowered the score**")
    right.markdown("\n".join(f"- {md_escape(r)}" for r in x["strengths"]) or "None above the 0.03 threshold.")

# ---------------------------------------------------------------- about tab
with tab_about:
    st.markdown(
        f"""
**Question.** Northvale's credit committee asked for an approval policy that cuts expected losses without
killing loan volume. This simulator lets a credit manager move the cutoff and the P&L assumptions and see the
trade-off on a book of loans the models never saw during training or tuning.

**Data and split.** 36-month Lending Club loans issued 2012 to 2015, all resolved. Models trained on 2012-2014,
tuned and recalibrated on January-June 2015, evaluated on July-December 2015 (this book).

**P&L.** Expected profit per loan = principal x [(1 - PD) x G(rate band) - PD x L - costs]. G and L are measured
on 2012-2014 loans: G is net interest earned per $ on loans that pay off, L is the net loss per $ on loans that
default ({A['L'] * 100:.1f}% by default). Costs = cost of funds x {A['principal_years']:.2f} principal-years + opex.
Realized profit uses the cash each loan actually returned. No time value of money.

**Limits.** Lending Club only shows approved loans, so a cutoff can only tighten: the effect of approving
applicants Lending Club rejected is unknown (no reject inference). Prices are held fixed. 2015 credit conditions.
Lending Club is a marketplace; Northvale is a fictional balance-sheet lender.
"""
    )
