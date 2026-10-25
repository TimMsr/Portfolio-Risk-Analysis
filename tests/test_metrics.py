"""Metrics on small examples that can be checked by hand."""

import numpy as np
import pandas as pd
import pytest

from src import metrics
from src.metrics import PERIODS_PER_YEAR


def series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_total_return():
    # 1.10 x 0.90 x 1.05 = 1.0395
    assert metrics.total_return(series([0.10, -0.10, 0.05])) == pytest.approx(0.0395)


def test_annualised_return_one_year_is_total_return():
    r = series([0.0] * (PERIODS_PER_YEAR - 1) + [0.08])
    assert metrics.annualised_return(r) == pytest.approx(0.08)


def test_volatility():
    # Returns +1%, -1%, +1%, -1%: mean 0, sample variance = 4 x 0.0001 / 3
    r = series([0.01, -0.01, 0.01, -0.01])
    expected = np.sqrt(4 * 0.0001 / 3) * np.sqrt(PERIODS_PER_YEAR)
    assert metrics.volatility(r) == pytest.approx(expected)


def test_max_drawdown_with_dates():
    # Growth path: 1.10, 1.21, 0.968 (-20% from 1.21), 1.0648, 1.278 (new high)
    r = series([0.10, 0.10, -0.20, 0.10, 0.20])
    dd = metrics.max_drawdown(r)
    assert dd["depth"] == pytest.approx(-0.20)
    assert dd["peak"] == r.index[1]
    assert dd["trough"] == r.index[2]
    assert dd["recovery"] == r.index[4]   # 1.278 >= 1.21


def test_max_drawdown_not_recovered():
    r = series([0.10, -0.50, 0.10])
    dd = metrics.max_drawdown(r)
    assert dd["depth"] == pytest.approx(-0.50)
    assert dd["recovery"] is None


def test_drawdown_series_is_never_positive():
    r = series([0.05, -0.02, 0.03, -0.10, 0.01])
    assert (metrics.drawdown_series(r) <= 0).all()


def test_sharpe():
    # Excess returns: 0.002, 0.000, 0.004 -> mean 0.002, sample stdev 0.002
    # Sharpe = 0.002 / 0.002 x sqrt(261) = sqrt(261)
    r = series([0.003, 0.001, 0.005])
    rf = series([0.001, 0.001, 0.001])
    assert metrics.sharpe_ratio(r, rf) == pytest.approx(np.sqrt(PERIODS_PER_YEAR))


def test_beta_alpha_exact_line():
    # Portfolio excess return = 0.0001 + 1.5 x benchmark excess return, exactly.
    x = series([0.01, -0.02, 0.015, 0.0, -0.005])
    rf = series([0.0] * 5)
    y = 0.0001 + 1.5 * x
    reg = metrics.beta_alpha(y, x, rf)
    assert reg["beta"] == pytest.approx(1.5)
    assert reg["alpha_annual"] == pytest.approx(0.0001 * PERIODS_PER_YEAR)
    assert reg["r2"] == pytest.approx(1.0)


def test_beta_hand_computed():
    # x = [1, -1, 2, 0] %, y = [2, -1, 3, 0] %
    # mean x = 0.5, mean y = 1.0
    # cov = [(0.5)(1) + (-1.5)(-2) + (1.5)(2) + (-0.5)(-1)] / 3 = 7/3
    # var x = [0.25 + 2.25 + 2.25 + 0.25] / 3 = 5/3   -> beta = 7/5 = 1.4
    # alpha (daily, in %) = 1.0 - 1.4 x 0.5 = 0.3
    x = series([0.01, -0.01, 0.02, 0.0])
    y = series([0.02, -0.01, 0.03, 0.0])
    reg = metrics.beta_alpha(y, x, series([0.0] * 4))
    assert reg["beta"] == pytest.approx(1.4)
    assert reg["alpha_annual"] == pytest.approx(0.003 * PERIODS_PER_YEAR)


def test_tracking_error_zero_when_identical():
    r = series([0.01, -0.02, 0.03])
    assert metrics.tracking_error(r, r) == pytest.approx(0.0)


def test_daily_risk_free_includes_weekend():
    # Flat 3.6% p.a. ACT/360 = 0.01% per calendar day.
    cal = pd.bdate_range("2024-01-08", "2024-01-12").append(pd.DatetimeIndex(["2024-01-15"]))
    estr = pd.Series(3.6, index=pd.date_range("2024-01-01", "2024-01-15"))
    rf = metrics.daily_risk_free(estr, cal)
    assert rf.loc["2024-01-09"] == pytest.approx(0.0001)
    # Friday -> Monday covers 3 calendar days.
    assert rf.loc["2024-01-15"] == pytest.approx(1.0001 ** 3 - 1)


def test_attribution_adds_up():
    # Two days, everything in the ETF sleeve, no costs.
    idx = pd.bdate_range("2024-01-01", periods=2)
    pnl = pd.DataFrame({"etf": [5.0, -2.1], "stock": 0.0, "cash": 0.0, "costs": 0.0}, index=idx)
    weights = pd.DataFrame({"etf": [1.0, 1.0], "stock": 0.0, "cash": 0.0}, index=idx)
    capital = pd.Series([100.0, 105.0], index=idx)
    rb = pd.Series([0.03, 0.01], index=idx)
    out = metrics.attribution(pnl, weights, capital, rb)
    parts = out.drop("Total gap to benchmark").sum()
    assert parts == pytest.approx(out["Total gap to benchmark"])
    # Portfolio: 1.05 x 0.98 = 1.029; benchmark 1.03 x 1.01 = 1.0403
    assert out["Total gap to benchmark"] == pytest.approx(1.029 - 1.0403)


def test_to_weekly_compounds_within_week():
    # Mon..Fri of one week: 1.01 x 1.02 = 1.0302, rest zero
    r = series([0.01, 0.02, 0.0, 0.0, 0.0], start="2024-01-01")
    weekly = metrics.to_weekly(r)
    assert len(weekly) == 1
    assert weekly.iloc[0] == pytest.approx(0.0302)
