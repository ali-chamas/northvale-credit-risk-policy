"""Data assumptions and metric sanity checks. Skipped when the data has not been built."""

import duckdb
import pandas as pd

from northvale import config as C


def q(sql):
    con = duckdb.connect()
    out = con.execute(sql).df()
    con.close()
    return out


# ---------------------------------------------------------------- data assumptions

def test_scope_is_36_month_2012_2015(loans_path):
    r = q(f"SELECT min(issue_d) a, max(issue_d) b, count(DISTINCT term) t, any_value(term) term FROM '{loans_path}'").iloc[0]
    assert str(r["a"])[:7] == "2012-01" and str(r["b"])[:7] == "2015-12"
    assert r["t"] == 1 and r["term"] == "36 months"


def test_target_only_on_terminal_statuses(loans_path):
    s = q(f"SELECT loan_status, avg(target) t, count(target) n_t, count(*) n FROM '{loans_path}' GROUP BY 1").set_index("loan_status")
    assert s.loc["Charged Off", "t"] == 1 and s.loc["Fully Paid", "t"] == 0
    for st in ("Current", "In Grace Period", "Late (31-120 days)"):
        if st in s.index:
            assert s.loc[st, "n_t"] == 0


def test_out_of_time_split_has_no_overlap(loans_path):
    s = q(f"SELECT split, min(issue_d) a, max(issue_d) b FROM '{loans_path}' WHERE in_sample GROUP BY 1").set_index("split")
    assert s.loc["train", "b"] < s.loc["valid", "a"]
    assert s.loc["valid", "b"] < s.loc["test", "a"]


def test_no_joint_applications_in_sample(loans_path):
    n = q(f"SELECT count(*) n FROM '{loans_path}' WHERE in_sample AND application_type <> 'Individual'").iloc[0, 0]
    assert n == 0


def test_cash_identity_holds(loans_path):
    r = q(f"""SELECT max(abs(total_pymnt - (total_rec_prncp + total_rec_int + total_rec_late_fee + recoveries))) r
              FROM '{loans_path}' WHERE in_sample""").iloc[0, 0]
    assert r < 0.02


def test_scored_pds_are_probabilities():
    if not C.SCORED.exists():
        import pytest
        pytest.skip("scored parquet not built")
    s = pd.read_parquet(C.SCORED)
    for c in ("pd_lgbm", "pd_scorecard", "pd_grade"):
        assert s[c].between(0, 1).all()


# ---------------------------------------------------------------- metric sanity

def test_sample_counts_add_up(m):
    d = m["data"]
    assert d["scope_rows"] == d["sample_rows"] + d["joint_excluded"] + d["non_terminal_excluded"]
    assert sum(s["n"] for s in d["splits"].values()) == d["sample_rows"]


def test_scope_is_resolved(m):
    assert m["censoring"]["scope_non_terminal_share"] < 0.001
    assert m["censoring"]["share_36m_2016"] > 0.2  # why 2016+ is out of scope


def test_audit_covers_every_column(m):
    a = pd.read_csv(C.DOCS / "feature_audit.csv")
    assert len(a) == m["data"]["raw_columns"] == a["column"].nunique()
    kept = a[a["decision"] == "keep"]
    assert (kept["available_at_application"] == "yes").all()
    assert not (a[a["category"] == "post_origination"]["decision"] == "keep").any()
    assert m["audit"]["flagged_all_post_origination_or_pricing"]


def test_model_metrics_are_plausible(m):
    for k in ("grade", "scorecard", "lgbm"):
        assert 0.6 < m["models"][k]["auc"] < 0.8
        assert 0 < m["models"][k]["brier"] < m["models"][k]["observed"]
    # recalibrated PDs match the observed test default rate within 1 pp
    assert abs(m["models"]["lgbm"]["mean_pd"] - m["models"]["lgbm"]["observed"]) < 0.01
    assert m["models"]["lgbm"]["psi_train_test"] < 0.1


def test_leakage_demo_shows_the_jump(m):
    assert m["leakage"]["auc_with_last_fico"] > m["leakage"]["baseline_auc"] + 0.1
    assert m["leakage"]["auc_with_total_rec_prncp"] > 0.95


def test_recommended_policy_respects_volume_floor(m):
    rec = m["pnl"]["options"][m["pnl"]["recommended"]]
    assert rec["valid_approval_rate"] >= rec["min_volume"] - 1e-9
    assert abs(rec["approval_rate"] - rec["min_volume"]) < 0.03  # test approval close to the target


def test_policy_accounting_is_consistent(m):
    p = m["pnl"]
    rec = p["options"]["balanced"]
    assert rec["n"] + rec["declined_n"] == p["status_quo"]["n"]
    assert abs(rec["profit_change"] + rec["declined_realized_profit"]) < 1
    assert abs(p["annual"]["profit_change"] - 2 * rec["profit_change"]) < 1


def test_app_files_are_small_and_complete():
    files = {"scored_test.parquet", "assumptions.json", "reason_codes.json"}
    present = {f.name for f in C.APP_DATA.glob("*")}
    if not files <= present:
        import pytest
        pytest.skip("app data not built")
    assert sum((C.APP_DATA / f).stat().st_size for f in files) < 20e6
    s = pd.read_parquet(C.APP_DATA / "scored_test.parquet")
    assert {"pd_lgbm", "pd_scorecard", "pd_grade", "net_cash", "funded_amnt", "target", "grade", "int_rate"} <= set(s.columns)
