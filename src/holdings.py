"""Rebuild daily holdings, cash and cash flows from the transaction table.

The analysis runs on a weekday calendar (Monday-Friday). A transaction that
happens on a weekend (e.g. a bank transfer settling on a Sunday) is moved to
the next weekday, so that every flow lands on a day for which we have prices.

All money amounts returned here are in EUR. They stay inside the program and
are never written to public outputs; only percentages derived from them are.
"""

import numpy as np
import pandas as pd


def weekday_calendar(start, end):
    """All Monday-Friday dates from start to end (inclusive)."""
    return pd.bdate_range(start, end)


def roll_to_weekday(dates):
    """Move Saturday/Sunday dates to the following Monday; weekdays unchanged."""
    return pd.DatetimeIndex(dates).map(pd.offsets.BDay().rollforward)


def _daily_sum(tx, column, by, calendar):
    """Sum `column` per (weekday, `by`) and lay it out as a date x `by` table."""
    df = tx.assign(day=roll_to_weekday(tx["date"]))
    table = df.pivot_table(index="day", columns=by, values=column, aggfunc="sum", fill_value=0.0)
    return table.reindex(calendar, fill_value=0.0)


def daily_positions(tx, calendar):
    """Number of shares held at the end of each day, per instrument."""
    trades = tx[tx["type"].isin(["buy", "sell"])]
    return _daily_sum(trades, "quantity", "instrument", calendar).cumsum()


def daily_cash(tx, calendar):
    """Cash balance at the end of each day, per currency (all accounts together)."""
    return _daily_sum(tx, "amount", "currency", calendar).cumsum()


def eur_fx_table(usd_per_eur, calendar):
    """Factor that converts one unit of each currency into EUR, per day.

    EUR -> 1.0; USD -> 1 / (USD per EUR). Days without an ECB fixing (TARGET
    holidays) use the previous fixing.
    """
    usd = usd_per_eur.reindex(usd_per_eur.index.union(calendar)).ffill().reindex(calendar)
    return pd.DataFrame({"EUR": 1.0, "USD": 1.0 / usd}, index=calendar)


def in_eur(table_by_currency, fx):
    """Convert a (date x currency) table into a single EUR series."""
    unknown = set(table_by_currency.columns) - set(fx.columns)
    if unknown:
        raise ValueError(f"no FX rate for currencies {sorted(unknown)}")
    return (table_by_currency * fx[table_by_currency.columns]).sum(axis=1)


def build_daily_book(tx, instruments, prices_eur, fx, calendar):
    """Everything the return calculation needs, per day and in EUR.

    Parameters
    ----------
    tx           normalised transactions (see parsers.py)
    instruments  table indexed by broker symbol with columns yahoo_symbol, sleeve
    prices_eur   (date x yahoo symbol) prices in EUR on `calendar`
    fx           output of eur_fx_table
    calendar     weekday DatetimeIndex

    Returns a DataFrame with one row per day and these columns:
      value_etf, value_stock     market value of each sleeve at the close
      cash                       cash in all accounts and currencies
      value_total                value_etf + value_stock + cash
      external_flow              deposits minus withdrawals (money from outside)
      net_buys_etf/_stock        money spent on buying minus received from selling
                                 (commissions included, so costs stay in the sleeve)
      income_etf/_stock          dividends received (net of withholding tax)
      commission_etf/_stock      trading commissions paid
      other_costs                custody/other fees and tax not tied to a holding
      interest                   interest received on cash
    """
    unknown = sorted(set(tx.loc[tx["instrument"] != "", "instrument"]) - set(instruments.index))
    if unknown:
        raise ValueError(f"instruments missing from instruments.csv: {unknown}")

    tx = tx.copy()
    # EUR value of each transaction, using the ECB rate of the (rolled) day.
    day = roll_to_weekday(tx["date"])
    rate = [fx.at[d, c] for d, c in zip(day, tx["currency"])]
    tx["amount_eur"] = tx["amount"] * rate
    tx["fee_eur"] = tx["fee"] * rate
    tx["sleeve"] = tx["instrument"].map(instruments["sleeve"]).fillna("")

    def per_day(rows, column):
        if rows.empty:
            return pd.Series(0.0, index=calendar)
        return _daily_sum(rows.assign(_one="all"), column, "_one", calendar)["all"]

    book = pd.DataFrame(index=calendar)

    # Market value of each sleeve: shares held x price in EUR.
    positions = daily_positions(tx, calendar)
    for sleeve in ("etf", "stock"):
        cols = [s for s in positions.columns if instruments.at[s, "sleeve"] == sleeve]
        yahoo = [instruments.at[s, "yahoo_symbol"] for s in cols]
        held = positions[cols].to_numpy()
        px = prices_eur.reindex(columns=yahoo).to_numpy()
        # A zero position is worth zero even on days without a price (NaN x 0 = NaN).
        value = np.where(held != 0, held * px, 0.0)
        if np.isnan(value).any():
            raise ValueError(f"{sleeve} sleeve: missing price on a day a position is held")
        book[f"value_{sleeve}"] = value.sum(axis=1) if cols else 0.0

    book["cash"] = in_eur(daily_cash(tx, calendar), fx)
    book["value_total"] = book["value_etf"] + book["value_stock"] + book["cash"]

    is_type = tx["type"].isin
    book["external_flow"] = per_day(tx[is_type(["deposit", "withdrawal"])], "amount_eur")
    for sleeve in ("etf", "stock"):
        mine = tx["sleeve"] == sleeve
        # amount is negative for a buy, so net money put INTO the sleeve is -amount.
        book[f"net_buys_{sleeve}"] = -per_day(tx[mine & is_type(["buy", "sell"])], "amount_eur")
        book[f"income_{sleeve}"] = per_day(tx[mine & is_type(["dividend", "tax"])], "amount_eur")
        book[f"commission_{sleeve}"] = per_day(tx[mine & is_type(["buy", "sell"])], "fee_eur")
    # Fees and taxes that do not belong to a holding (e.g. custody fee, tax on interest).
    book["other_costs"] = -per_day(tx[(tx["sleeve"] == "") & is_type(["fee", "tax"])], "amount_eur")
    book["interest"] = per_day(tx[is_type(["interest"])], "amount_eur")
    return book
