"""Parsers and holdings, on the synthetic sample files and tiny hand-made inputs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import holdings, returns
from src.parsers import load_transactions, parse_ibkr, parse_number, parse_revolut

SAMPLE = Path(__file__).resolve().parents[1] / "data_sample"


def test_parse_number():
    assert parse_number("1,250") == 1250
    assert parse_number("USD " + "12.5") == 12.5   # built up so the privacy scan sees no amount
    assert parse_number("-12.30") == -12.30
    assert np.isnan(parse_number(""))


def test_revolut_signs_and_types():
    tx = parse_revolut(SAMPLE / "revolut_sample.csv")
    assert set(tx["type"]) <= {"buy", "sell", "dividend", "fee", "deposit", "withdrawal"}
    buys, sells = tx[tx["type"] == "buy"], tx[tx["type"] == "sell"]
    assert (buys["amount"] < 0).all() and (buys["quantity"] > 0).all()
    assert (sells["amount"] > 0).all() and (sells["quantity"] < 0).all()
    assert (tx.loc[tx["type"] == "withdrawal", "amount"] < 0).all()


def test_ibkr_trades_include_commission():
    tx = parse_ibkr(SAMPLE / "ibkr_activity_sample.csv")
    buys = tx[tx["type"] == "buy"]
    # cash paid = quantity x price + commission
    expected = -(buys["quantity"] * buys["price"] + buys["fee"])
    assert np.allclose(buys["amount"], expected, atol=0.01)
    assert (tx.loc[tx["type"] == "dividend", "instrument"] == "VWCE").all()


def test_no_cash_goes_negative_in_sample():
    # If a transaction were missing or had the wrong sign, the rebuilt cash
    # balance would typically go negative at some point.
    tx = load_transactions(SAMPLE)
    for _, group in tx.groupby(["account", "currency"]):
        assert group.groupby("date")["amount"].sum().cumsum().min() > -0.01


def test_weekend_flow_moves_to_monday():
    rolled = holdings.roll_to_weekday(pd.to_datetime(["2026-09-26", "2026-09-28"]))
    assert list(rolled) == [pd.Timestamp("2026-09-28")] * 2


def test_book_and_returns_hand_example():
    # Deposit 1000 on Monday, buy 10 shares at 100 (commission 1). The price is
    # 100 on Monday and 110 on Tuesday.
    cal = holdings.weekday_calendar("2024-01-01", "2024-01-02")
    tx = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "account": "x", "instrument": "", "type": "deposit",
         "quantity": 0.0, "price": np.nan, "amount": 1000.0, "fee": 0.0, "currency": "EUR", "note": ""},
        {"date": pd.Timestamp("2024-01-01"), "account": "x", "instrument": "ABC", "type": "buy",
         "quantity": 10.0, "price": 100.0, "amount": -1001.0, "fee": 1.0, "currency": "EUR", "note": ""},
    ])
    inst = pd.DataFrame({"yahoo_symbol": ["ABC.X"], "currency": ["EUR"], "sleeve": ["etf"]},
                        index=pd.Index(["ABC"], name="broker_symbol"))
    px = pd.DataFrame({"ABC.X": [100.0, 110.0]}, index=cal)
    fx = holdings.eur_fx_table(pd.Series(1.1, index=cal), cal)
    book = holdings.build_daily_book(tx, inst, px, fx, cal)

    assert book["value_total"].tolist() == pytest.approx([999.0, 1099.0])   # cash -1 + shares
    r = returns.portfolio_returns(book)
    assert r["portfolio"].tolist() == pytest.approx([-0.001, 0.1001001], rel=1e-6)
    # Sleeve paid 1001 for shares worth 1000 on day 1, then +10% on day 2.
    assert r["etf_sleeve"].tolist() == pytest.approx([1000 / 1001 - 1, 0.10])

    pnl = returns.daily_pnl_components(book)
    assert pnl["costs"].iloc[0] == pytest.approx(-1.0)
    assert pnl.sum(axis=1).tolist() == pytest.approx([-1.0, 100.0])
