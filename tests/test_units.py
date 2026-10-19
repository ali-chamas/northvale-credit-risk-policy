"""Unit tests that need no data: formulas and rules behave as documented."""

import numpy as np
import pandas as pd
import pytest

from northvale import audit, explain, fmt, models, pnl, scope
from northvale import features as F


def test_psi_is_zero_for_identical_and_positive_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20_000)
    assert models.psi(a, a) == pytest.approx(0, abs=1e-9)
    assert models.psi(a, a + 0.5) > 0.1


def test_ks_perfect_and_random():
    y = np.r_[np.zeros(500), np.ones(500)]
    assert models.ks_stat(y, y) == pytest.approx(1.0)
    assert models.ks_stat(y, np.zeros(1000)) == pytest.approx(0.0)


def test_scorecard_scaling_hits_anchor():
    # 600 points at 50:1 good:bad odds, +20 points doubles the odds
    pd_at_600 = 1 / 51
    score = models.OFFSET + models.FACTOR * np.log((1 - pd_at_600) / pd_at_600)
    assert score == pytest.approx(600)
    assert models.OFFSET + models.FACTOR * np.log(100) == pytest.approx(620)


def test_platt_preserves_order():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.6, 5000)
    y = (rng.uniform(size=5000) < p * 1.2).astype(int)
    q = models.Platt().fit(p, y).predict(p)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)


def test_subgrade_rank():
    assert F.subgrade_rank("A1") == 1
    assert F.subgrade_rank("C3") == 13
    assert F.subgrade_rank("G5") == 35


def test_mix_vs_rate_adds_up():
    gy = pd.DataFrame({
        "grade": list("AB") * 2, "issue_year": [2013, 2013, 2015, 2015],
        "n_loans": [100, 100, 300, 100], "default_rate": [0.05, 0.15, 0.06, 0.20],
    })
    d = scope.mix_vs_rate(gy, 2013, 2015)
    assert d["mix_effect"] + d["rate_effect"] == pytest.approx(d["change"])


def test_rate_bands():
    b = pnl.band(pd.Series([5.0, 8.0, 10.99, 11.0, 19.99, 20.0, 28.0]))
    assert list(b) == ["<8%", "8-11%", "8-11%", "11-14%", "17-20%", "20%+", "20%+"]


def test_expected_pnl_formula():
    d = pd.DataFrame({"int_rate": [12.0], "funded_amnt": [10_000.0], "pd": [0.1]})
    a = {"G": {"11-14%": 0.2}, "L": 0.4, "cost_rate": 0.05}
    # 10,000 x (0.9 x 0.2 - 0.1 x 0.4 - 0.05) = 900
    assert pnl.expected_pnl(d, "pd", a)[0] == pytest.approx(900)


def test_breakeven_pd_zero_expected_profit():
    G, L, c = 0.17, 0.36, 0.06
    p = (G - c) / (G + L)
    assert (1 - p) * G - p * L - c == pytest.approx(0)


def test_no_leaky_or_pricing_field_is_a_feature():
    banned = set(audit.POST_ORIGINATION) | set(audit.LATE_2015) | set(audit.JOINT) | set(F.PRICING)
    banned |= {"funded_amnt", "funded_amnt_inv", "zip_code", "addr_state", "initial_list_status"}
    assert not banned & set(F.FEATURES)


def test_every_kept_feature_has_an_audit_rule_and_label():
    for c in F.RAW_NUMERIC + F.CATEGORICAL:
        assert audit.rule(c)[2] == "keep"
    for c in F.FEATURES:
        assert c in F.LABELS
    for c in F.NUMERIC:
        assert c in explain.REASONS


def test_unknown_column_has_no_silent_default():
    with pytest.raises(KeyError):
        audit.rule("brand_new_column")


def test_reason_text_is_plain_english():
    assert explain.reason_text("dti", 31.2, 17.6) == "Debt-to-income ratio of 31% (median 18%)"
    assert explain.reason_text("purpose", "small_business", None) == "Loan purpose: small business"


def test_fmt_refuses_missing_values():
    with pytest.raises(ValueError):
        fmt.pct(float("nan"))
    assert fmt.pct(0.1234) == "12.3%"
    assert fmt.pp(-0.0192) == "-1.9 pp"
