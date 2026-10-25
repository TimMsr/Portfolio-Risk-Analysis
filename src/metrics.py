"""Performance and risk metrics on daily returns.

Conventions (used consistently everywhere)
------------------------------------------
- Returns are simple daily returns on a Monday-Friday calendar.
- PERIODS_PER_YEAR = 261 weekdays (52.2 weeks x 5). Exchange holidays stay in
  the calendar with a zero return, so annualising with 261 keeps the yearly
  variance right.
- Standard deviations use the sample formula (divide by n - 1).
- The risk-free rate is €STR, turned into a daily return for each day of our
  calendar (see `daily_risk_free`).
"""

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 261


def daily_risk_free(estr_percent, calendar):
    """€STR (percent per year, ACT/360) as a simple return for each calendar day.

    €STR is an overnight rate that accrues on every calendar day, including
    weekends, at the last published rate. We build an accrual index over all
    calendar days and take its growth between consecutive weekdays, so Monday's
    risk-free return includes Saturday and Sunday.
    """
    every_day = pd.date_range(calendar[0] - pd.Timedelta(days=7), calendar[-1], freq="D")
    rate = estr_percent.reindex(every_day.union(estr_percent.index)).ffill().reindex(every_day)
    index = (1 + rate.fillna(0) / 100 / 360).cumprod()
    return index.reindex(calendar).pct_change().fillna(0.0)


def total_return(r):
    """Total return: how much 1 unit grew over the whole period, minus 1.

        TR = (1 + r_1)(1 + r_2)...(1 + r_n) - 1
    """
    return float((1 + r.dropna()).prod() - 1)


def annualised_return(r):
    """Compound annual growth rate implied by the total return.

        AR = (1 + TR) ^ (PERIODS_PER_YEAR / n) - 1,  n = number of daily returns

    For periods shorter than a year this extrapolates and should be read with care.
    """
    n = r.dropna().size
    return float((1 + total_return(r)) ** (PERIODS_PER_YEAR / n) - 1)


def volatility(r):
    """Annualised volatility: how much returns swing around their average.

        vol = stdev(daily returns) x sqrt(PERIODS_PER_YEAR)

    The square root comes from variances of independent days adding up.
    """
    return float(r.dropna().std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def drawdown_series(r):
    """Drop from the highest value reached so far, at every date (0 or negative).

        W_t  = growth of 1 unit up to day t
        DD_t = W_t / max(W_1..W_t) - 1
    """
    wealth = (1 + r.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r):
    """Largest peak-to-trough loss, with the dates involved.

    Returns a dict: depth (negative number), peak date, trough date and
    recovery date (first date back at the old peak; None if not yet recovered).
    """
    dd = drawdown_series(r)
    trough = dd.idxmin()
    wealth = (1 + r.fillna(0.0)).cumprod()
    peak = wealth.loc[:trough].idxmax()
    after = wealth.loc[trough:]
    recovered = after[after >= wealth.loc[peak]]
    return {"depth": float(dd.min()), "peak": peak, "trough": trough,
            "recovery": recovered.index[0] if len(recovered) else None}


def sharpe_ratio(r, rf):
    """Excess return per unit of risk (annualised).

        Sharpe = mean(r - rf) x PERIODS_PER_YEAR / (stdev(r - rf) x sqrt(PERIODS_PER_YEAR))

    r and rf are daily returns on the same dates. A Sharpe of 0.5 means that
    for every 1% of annual volatility the portfolio earned 0.5% a year above cash.
    """
    excess = (r - rf).dropna()
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def to_weekly(r):
    """Compound daily returns into Friday-to-Friday weekly returns.

    Used for a second beta estimate: US stocks close hours after the European
    benchmark, so part of a day's market move shows up in the stock's return only
    on the next day. Over a week that timing mismatch mostly washes out.
    """
    return (1 + r.fillna(0.0)).resample("W-FRI").prod() - 1


def beta_alpha(r, rb, rf, periods_per_year=PERIODS_PER_YEAR):
    """Regression of portfolio excess returns on benchmark excess returns (CAPM).

        (r - rf) = alpha + beta x (rb - rf) + error     (ordinary least squares)

    beta  = cov(x, y) / var(x): how much the portfolio moves when the benchmark
            moves 1%. Beta 1.1 means typically 1.1% for a 1% benchmark move.
    alpha = mean(y) - beta x mean(x): the average daily return not explained by
            the benchmark; reported annualised as alpha x periods_per_year.
    r2    = share of the portfolio's daily variance explained by the benchmark.
    """
    df = pd.DataFrame({"y": r - rf, "x": rb - rf}).dropna()
    beta = df["x"].cov(df["y"]) / df["x"].var()
    alpha_daily = df["y"].mean() - beta * df["x"].mean()
    r2 = df["x"].corr(df["y"]) ** 2
    return {"beta": float(beta), "alpha_annual": float(alpha_daily * periods_per_year),
            "r2": float(r2)}


def tracking_error(r, rb):
    """Annualised volatility of the daily return difference to the benchmark.

        TE = stdev(r - rb) x sqrt(PERIODS_PER_YEAR)

    Low TE means the portfolio behaves like the benchmark day to day.
    """
    return float((r - rb).dropna().std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def summary(r, rb, rf):
    """All headline metrics for one return series, as a dict of plain numbers."""
    dd = max_drawdown(r)
    reg = beta_alpha(r, rb, rf)
    weekly = beta_alpha(to_weekly(r), to_weekly(rb), to_weekly(rf), periods_per_year=52)
    # Returns over less than a year are not annualised (it would extrapolate).
    long_enough = r.dropna().size >= PERIODS_PER_YEAR
    return {
        "total_return": total_return(r),
        "annualised_return": annualised_return(r) if long_enough else np.nan,
        "volatility": volatility(r),
        "max_drawdown": dd["depth"],
        "max_drawdown_peak": dd["peak"].date(),
        "max_drawdown_trough": dd["trough"].date(),
        "max_drawdown_recovery": dd["recovery"].date() if dd["recovery"] is not None else None,
        "sharpe": sharpe_ratio(r, rf),
        "beta": reg["beta"],
        "alpha_annual": reg["alpha_annual"],
        "r2": reg["r2"],
        "beta_weekly": weekly["beta"],
        "r2_weekly": weekly["r2"],
        "tracking_error": tracking_error(r, rb),
    }


def attribution(pnl, weights, capital, rb):
    """Split the gap between portfolio and benchmark into its sources.

    Each day, the portfolio return is the sum of contributions c_k = pnl_k / capital,
    and the start-of-day weights w_k (ETF sleeve, stock sleeve, cash) sum to 1. So:

        r_t - rb_t = sum over k of (c_k - w_k x rb_t)        (exact, every day)

    - ETF sleeve:   what the ETFs earned (before commissions) minus what the
                    same money would have earned in the benchmark.
    - Stock sleeve: the same for the individual stocks.
    - Cash:         interest and FX on cash, minus the benchmark return the
                    uninvested cash missed (cash drag).
    - Costs:        commissions, fees and tax not tied to a holding (benchmark
                    weight zero, so the whole cost is a drag).

    Daily contributions are added up over the period. Because returns compound,
    that sum differs from the actual gap of the cumulative returns; the
    difference is shown as "compounding effect" so the table adds up exactly.
    All values are fractions (multiply by 100 for percentage points).
    """
    valid = capital > 0  # days before the first deposit have nothing to attribute
    pnl, weights, capital, rb = pnl[valid], weights[valid], capital[valid], rb[valid]
    rows = {}
    for k in ("etf", "stock", "cash"):
        rows[k] = (pnl[k] / capital - weights[k].fillna(0) * rb).sum()
    rows["costs"] = (pnl["costs"] / capital).sum()
    r = pnl.sum(axis=1) / capital
    actual_gap = (1 + r).prod() - (1 + rb).prod()
    rows["compounding"] = actual_gap - sum(rows.values())
    labels = {"etf": "Index ETF sleeve", "stock": "Individual stock sleeve",
              "cash": "Cash (interest, FX, cash drag)", "costs": "Costs (commissions, fees)",
              "compounding": "Compounding effect"}
    out = pd.Series({labels[k]: float(v) for k, v in rows.items()})
    out["Total gap to benchmark"] = float(actual_gap)
    return out
