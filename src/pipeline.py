"""Run the whole analysis: transactions -> holdings -> returns -> metrics -> charts.

    python -m src.pipeline                 # synthetic sample data (default)
    python -m src.pipeline --private       # real data in data_private/

Outputs from sample data go to output_sample/ (safe to publish after
check_public_outputs.py passes). Outputs from real data go to
data_private/output/ and are never committed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import holdings, metrics, plots, prices, returns
from src.parsers import load_instruments, load_transactions

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = "IWDA.AS"   # iShares Core MSCI World UCITS ETF (Acc), Euronext Amsterdam, EUR
ROLLING_WINDOW = 130    # 6 months of weekdays (261 / 2)
SAMPLE_END = "2026-10-22"  # sample runs stop here so committed outputs are reproducible
NAMES = {"portfolio": "Portfolio", "etf_sleeve": "Index ETF sleeve",
         "stock_sleeve": "Individual stock sleeve"}


def paths(private):
    if private:
        return ROOT / "data_private", ROOT / "data_private" / "output", ROOT / "data_private" / "cache"
    return ROOT / "data_sample", ROOT / "output_sample", ROOT / ".cache"


def price_table(tx, instruments, close_eur, usd_per_eur, calendar, positions, notes):
    """Market prices in EUR per broker symbol, with gaps filled and flagged.

    If no market price exists on a day an instrument is held (symbol not found,
    or held before its price history starts), the last trade price is used
    instead. This is an ESTIMATE and is reported in `notes`.
    """
    out = pd.DataFrame(index=calendar)
    trades = tx[tx["type"].isin(["buy", "sell"])].copy()
    trades["day"] = holdings.roll_to_weekday(trades["date"])
    for sym in positions.columns:
        yahoo = instruments.at[sym, "yahoo_symbol"]
        market = close_eur[yahoo] if yahoo in close_eur else pd.Series(np.nan, index=calendar)
        held = positions[sym] != 0
        missing = held & market.isna()
        if missing.any():
            t = trades[trades["instrument"] == sym].groupby("day")["price"].last()
            if instruments.at[sym, "currency"] == "USD":
                t = t / usd_per_eur.reindex(t.index)
            fallback = t.reindex(calendar).ffill()
            market = market.fillna(fallback.where(missing))
            notes.append(f"ESTIMATE: '{sym}' valued at last trade price on {int(missing.sum())} "
                         "day(s) without a market price")
        out[sym] = market
    return out


def check_trade_prices(tx, instruments, close_local, notes, tolerance=0.10):
    """Compare each trade price with the market close of that day.

    A large gap usually means a wrong Yahoo symbol or currency in instruments.csv.
    """
    trades = tx[tx["type"].isin(["buy", "sell"])]
    for _, t in trades.iterrows():
        yahoo = instruments.at[t["instrument"], "yahoo_symbol"]
        if yahoo not in close_local:
            continue
        series = close_local[yahoo].dropna()
        near = series.loc[:t["date"]]
        if near.empty:
            continue
        gap = t["price"] / near.iloc[-1] - 1
        if abs(gap) > tolerance:
            notes.append(f"CHECK: trade in '{t['instrument']}' on {t['date'].date()} is "
                         f"{gap:+.0%} away from the market close - check instruments.csv")


def run(private=False, accounts=None, start=None, end=None, refresh=False, make_charts=True,
        return_book=False):
    """Run the full analysis and return a dict of results.

    The results contain no money amounts, unless `return_book=True`, which adds
    the daily EUR book under "book" for local validation only (never save it).
    """
    data_dir, out_dir, cache_dir = paths(private)
    notes = []

    tx = load_transactions(data_dir)
    if accounts:
        tx = tx[tx["account"].isin(accounts)]
    instruments = load_instruments(data_dir)
    notes += [f"ESTIMATE: {n}" for n in tx.loc[tx["note"] != "", "note"].unique()]

    first_day = pd.Timestamp(start) if start else tx["date"].min()
    if start:
        # A later start date is only valid if nothing was held before it.
        if (tx["date"] < first_day).any():
            raise ValueError("transactions exist before `start`; earlier holdings would be missed")
    if end is None and not private:
        end = SAMPLE_END
    last_day = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    fetch_from = first_day - pd.Timedelta(days=10)

    # --- Market data -------------------------------------------------------
    used = sorted(set(tx.loc[tx["instrument"] != "", "instrument"]))
    yahoo = sorted({instruments.at[s, "yahoo_symbol"] for s in used})
    close, _, splits, problems = prices.fetch_prices(yahoo, fetch_from, last_day, cache_dir, refresh)
    _, bench_adj, _, bench_problems = prices.fetch_prices([BENCHMARK], fetch_from, last_day,
                                                          cache_dir, refresh)
    notes += problems + bench_problems
    # yfinance prices are split-adjusted but broker quantities are not, so a
    # split while the position was held would distort its value.
    for sym in used:
        first_trade = tx.loc[tx["instrument"] == sym, "date"].min()
        for split_day in splits.get(instruments.at[sym, "yahoo_symbol"], []):
            if split_day > first_trade:
                notes.append(f"PROBLEM: '{sym}' split on {split_day.date()} while held; "
                             "quantities before that date are not adjusted")
    if bench_adj.empty:
        raise RuntimeError(f"benchmark {BENCHMARK} could not be downloaded")
    usd_per_eur = prices.fetch_usd_per_eur(fetch_from, last_day, cache_dir, refresh)
    estr = prices.fetch_estr(fetch_from, last_day, cache_dir, refresh)

    # The analysis ends on the last day with a benchmark close.
    last_day = min(last_day, bench_adj.index.max())
    calendar = holdings.weekday_calendar(holdings.roll_to_weekday([first_day])[0], last_day)

    check_trade_prices(tx, instruments, close, notes)
    close_cal, gap_notes = prices.align(close, calendar)
    notes += gap_notes
    currencies = {instruments.at[s, "yahoo_symbol"]: instruments.at[s, "currency"] for s in used}
    usd_cal = usd_per_eur.reindex(usd_per_eur.index.union(calendar)).ffill().reindex(calendar)
    close_eur = prices.to_eur(close_cal, currencies, usd_cal)

    positions = holdings.daily_positions(tx, calendar)
    by_symbol = price_table(tx, instruments, close_eur, usd_cal, calendar, positions, notes)
    # build_daily_book looks prices up by Yahoo symbol.
    prices_eur = by_symbol.rename(columns=lambda s: instruments.at[s, "yahoo_symbol"])
    fx = holdings.eur_fx_table(usd_per_eur, calendar)
    book = holdings.build_daily_book(tx, instruments, prices_eur, fx, calendar)

    # --- Returns -------------------------------------------------------------
    r = returns.portfolio_returns(book)
    bench = prices.align(bench_adj, calendar)[0][BENCHMARK].pct_change()
    rf = metrics.daily_risk_free(estr, calendar)

    # Each series is measured from its first invested day.
    table, windows = {}, {}
    for col in r.columns:
        series = r[col]
        if series.notna().sum() < 20:
            notes.append(f"{NAMES[col]}: fewer than 20 days of history - metrics skipped")
            continue
        first = series.first_valid_index()
        s = series.loc[first:].fillna(0.0)
        b, f = bench.loc[first:].fillna(0.0), rf.loc[first:]
        table[NAMES[col]] = metrics.summary(s, b, f)
        if first not in windows.values():  # one benchmark row per distinct start date
            table[f"Benchmark (from {first.date()})"] = metrics.summary(b, b, f)
        windows[col] = first
    summary = pd.DataFrame(table).T

    # --- Attribution over the whole portfolio period --------------------------
    start_p = windows["portfolio"]
    capital = (book["value_total"].shift(1, fill_value=0.0) + book["external_flow"]).loc[start_p:]
    pnl = returns.daily_pnl_components(book).loc[start_p:]
    weights = returns.start_of_day_weights(book).loc[start_p:]
    attribution = metrics.attribution(pnl, weights, capital, bench.loc[start_p:].fillna(0.0))

    # --- Series for charts (indices and percentages only) ---------------------
    main = pd.DataFrame({"Portfolio": r["portfolio"], "Benchmark": bench}).loc[start_p:].fillna(0.0)
    rebased = returns.rebase(main)
    drawdowns = main.apply(metrics.drawdown_series)
    rolling_vol = main.rolling(ROLLING_WINDOW).std() * np.sqrt(metrics.PERIODS_PER_YEAR)
    sleeves = None
    if "stock_sleeve" in windows and "etf_sleeve" in windows:
        since = max(windows["stock_sleeve"], windows["etf_sleeve"])
        sleeves = returns.rebase(pd.DataFrame({
            "Index ETF sleeve": r["etf_sleeve"], "Individual stock sleeve": r["stock_sleeve"],
            "Benchmark": bench}).loc[since:].fillna(0.0))

    results = {"summary": summary, "attribution": attribution, "rebased": rebased,
               "drawdowns": drawdowns, "rolling_vol": rolling_vol, "sleeves": sleeves,
               "daily_returns": r.assign(benchmark=bench, risk_free=rf),
               "notes": notes, "out_dir": out_dir, "period": (start_p.date(), calendar[-1].date())}
    if return_book:
        results["book"] = book
    if make_charts:
        write_outputs(results)
    return results


def format_summary(summary):
    """Metrics table as text: percentages for returns/risk, plain numbers for ratios."""
    pct = ["total_return", "annualised_return", "volatility", "max_drawdown",
           "alpha_annual", "tracking_error"]
    out = summary.copy().astype(object)
    for col in pct:
        out[col] = summary[col].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2%}")
    for col in ("sharpe", "beta", "r2", "beta_weekly", "r2_weekly"):
        out[col] = summary[col].map(lambda v: f"{v:.2f}")
    return out


def write_outputs(results):
    out = Path(results["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    plots.plot_rebased(results["rebased"], out / "cumulative_return.png")
    plots.plot_drawdown(results["drawdowns"], out / "drawdown.png")
    plots.plot_rolling_vol(results["rolling_vol"].dropna(how="all"), out / "rolling_volatility.png")
    if results["sleeves"] is not None:
        plots.plot_rebased(results["sleeves"], out / "sleeves_vs_benchmark.png",
                           title="Sleeves vs benchmark, since both sleeves exist (start = 100)")

    start, end = results["period"]
    lines = [f"# Results ({start} to {end})", "",
             "All returns are time-weighted, in EUR, unrealised, pre-tax, based on market values.", "",
             "## Metrics", "", format_summary(results["summary"]).T.to_markdown(), "",
             "## Gap to benchmark, by source (percentage points)", "",
             (results["attribution"] * 100).round(2).to_frame("pp").to_markdown(), "",
             "## Data quality notes", ""]
    lines += [f"- {n}" for n in results["notes"]] or ["- none"]
    (out / "results.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private", action="store_true", help="use data_private/ instead of data_sample/")
    parser.add_argument("--refresh", action="store_true", help="re-download market data")
    args = parser.parse_args()
    res = run(private=args.private, refresh=args.refresh)
    print((Path(res["out_dir"]) / "results.md").read_text())


if __name__ == "__main__":
    main()
