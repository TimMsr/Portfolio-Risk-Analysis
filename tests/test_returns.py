"""Time-weighted return on small examples that can be checked by hand."""

import pandas as pd
import pytest

from src.returns import cumulative, daily_twr

DAYS = pd.bdate_range("2024-01-01", periods=4)


def test_twr_ignores_deposit():
    # Day 1: deposit 100, it closes at 110            -> +10%
    # Day 2: deposit another 110 at the start, so capital is 220; close at 242 -> +10%
    # Day 3: no flow, value falls to 217.8            -> -10%
    # Day 4: withdraw 17.8 at the start (capital 200), close at 200 -> 0%
    values = pd.Series([110.0, 242.0, 217.8, 200.0], index=DAYS)
    flows = pd.Series([100.0, 110.0, 0.0, -17.8], index=DAYS)
    r = daily_twr(values, flows)
    assert r.tolist() == pytest.approx([0.10, 0.10, -0.10, 0.0])
    # Chained: 1.1 x 1.1 x 0.9 x 1.0 - 1 = 8.9%, whatever the deposits were.
    assert cumulative(r).iloc[-1] - 1 == pytest.approx(0.089)


def test_twr_same_growth_regardless_of_deposit_size():
    # Two investors with identical daily returns but very different deposits
    # must get the same TWR.
    small = daily_twr(pd.Series([105.0, 110.25], index=DAYS[:2]),
                      pd.Series([100.0, 0.0], index=DAYS[:2]))
    big = daily_twr(pd.Series([105.0, 1110.25], index=DAYS[:2]),
                    pd.Series([100.0, 952.381], index=DAYS[:2]))
    assert small.tolist() == pytest.approx([0.05, 0.05])
    assert big.tolist() == pytest.approx([0.05, 0.05], rel=1e-5)


def test_twr_with_income_paid_out():
    # A sleeve worth 100 pays a dividend of 2 (moves to cash) and closes at 99.
    # Return = (99 + 2) / 100 - 1 = +1%.
    values = pd.Series([100.0, 99.0], index=DAYS[:2])
    flows = pd.Series([100.0, 0.0], index=DAYS[:2])
    income = pd.Series([0.0, 2.0], index=DAYS[:2])
    r = daily_twr(values, flows, income)
    assert r.iloc[1] == pytest.approx(0.01)


def test_twr_nan_before_anything_is_invested():
    values = pd.Series([0.0, 0.0, 50.0, 55.0], index=DAYS)
    flows = pd.Series([0.0, 0.0, 50.0, 0.0], index=DAYS)
    r = daily_twr(values, flows)
    assert r.iloc[:2].isna().all()
    assert r.iloc[2:].tolist() == pytest.approx([0.0, 0.10])
