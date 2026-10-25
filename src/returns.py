"""Time-weighted returns that remove the effect of deposits and withdrawals.

Why time-weighted?
------------------
If you deposit money just before the market rises, your euro gain grows, but
that is a timing-of-deposits effect, not investment skill. The time-weighted
return (TWR) measures how one euro invested at the start would have grown, so
it can be compared fairly with a benchmark.

Daily formula used here
-----------------------
Money that arrives during day t (a deposit, or a purchase for a sleeve) is
assumed to be available from the START of that day:

    r_t = (V_t + I_t) / (V_{t-1} + F_t) - 1

    V_t       value at the close of day t
    V_{t-1}   value at the close of the previous day
    F_t       net money added during day t (deposits minus withdrawals)
    I_t       income paid OUT of the thing being measured during day t
              (dividends, for a sleeve; zero for the whole portfolio,
              because dividends stay in the account as cash)

The cumulative TWR chains the daily returns: (1 + r_1)(1 + r_2)... - 1.

Start-of-day is a convention. It matters only on days with flows, and it is the
natural choice here because deposits were usually invested on the same day.
"""

import numpy as np
import pandas as pd


def daily_twr(values, flows, income=None):
    """Daily time-weighted returns for a value series with external flows.

    Days where nothing was invested (start-of-day capital is zero or negative)
    return NaN: there is no return to measure. Callers decide how to treat them.
    """
    income = 0.0 if income is None else income
    capital = values.shift(1, fill_value=0.0) + flows  # start-of-day capital
    r = (values + income) / capital - 1
    # Tiny capital (e.g. a few cents left after selling) gives meaningless returns.
    return r.where(capital > 1e-6 * values.abs().max())


def cumulative(returns):
    """Growth of 1 unit: (1 + r_1)(1 + r_2)... ; NaN days count as zero return."""
    return (1 + returns.fillna(0.0)).cumprod()


def rebase(returns, start_value=100.0):
    """Cumulative return series starting at `start_value` (100 by default).

    The first point is the value at the close of the first day, so a return
    on day one is already included. Prepend a 100 if you need the pre-start level.
    """
    return start_value * cumulative(returns)


def portfolio_returns(book):
    """Daily TWR of the whole portfolio and of each sleeve.

    `book` is the output of holdings.build_daily_book. Returns a DataFrame with
    columns portfolio, etf_sleeve, stock_sleeve.
    """
    return pd.DataFrame({
        "portfolio": daily_twr(book["value_total"], book["external_flow"]),
        "etf_sleeve": daily_twr(book["value_etf"], book["net_buys_etf"], book["income_etf"]),
        "stock_sleeve": daily_twr(book["value_stock"], book["net_buys_stock"], book["income_stock"]),
    })


def daily_pnl_components(book):
    """Split each day's portfolio profit into its sources (in EUR, internal only).

    The components add up exactly to V_t - V_{t-1} - F_t, the profit of the
    whole portfolio on day t:

      etf, stock   change in sleeve value + dividends - money put in,
                   with commissions added back (so this is the GROSS result)
      costs        minus (commissions + other fees + tax not tied to a holding)
      cash         interest received + currency gains/losses on cash balances
    """
    pnl = pd.DataFrame(index=book.index)
    for sleeve in ("etf", "stock"):
        v = book[f"value_{sleeve}"]
        net = v - v.shift(1, fill_value=0.0) + book[f"income_{sleeve}"] - book[f"net_buys_{sleeve}"]
        pnl[sleeve] = net + book[f"commission_{sleeve}"]
    pnl["costs"] = -(book["commission_etf"] + book["commission_stock"] + book["other_costs"])
    total = book["value_total"] - book["value_total"].shift(1, fill_value=0.0) - book["external_flow"]
    # Whatever is left is the cash result: interest, FX on cash balances.
    pnl["cash"] = total - pnl[["etf", "stock", "costs"]].sum(axis=1)
    return pnl


def start_of_day_weights(book):
    """Share of start-of-day capital in each sleeve and in cash (sums to 1)."""
    capital = book["value_total"].shift(1, fill_value=0.0) + book["external_flow"]
    w = pd.DataFrame(index=book.index)
    w["etf"] = (book["value_etf"].shift(1, fill_value=0.0) + book["net_buys_etf"]) / capital
    w["stock"] = (book["value_stock"].shift(1, fill_value=0.0) + book["net_buys_stock"]) / capital
    w["cash"] = 1.0 - w["etf"] - w["stock"]
    return w.where(capital > 0).replace([np.inf, -np.inf], np.nan)
