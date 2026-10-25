"""Download and cache market data: prices (yfinance), FX and €STR (ECB).

Data sources
------------
- Prices: Yahoo Finance via the `yfinance` package. For holdings we use the
  daily "Close" (adjusted for splits but NOT for dividends), because dividends
  are already recorded as cash in the broker data; using a dividend-adjusted
  price would count them twice. For the benchmark we use "Adj Close" (total
  return, dividends reinvested).
- FX: ECB euro reference rate, US dollar per euro (series EXR.D.USD.EUR.SP00.A).
  A USD price is converted with:  price_EUR = price_USD / (USD per EUR).
- Risk-free rate: euro short-term rate €STR (series EST.B.EU000A2X2A25.WT),
  in percent per year, ACT/360 convention.

Every download is cached as a CSV in `cache_dir` so that results are
reproducible and the analysis runs offline after the first download.
"""

import warnings
from pathlib import Path

import pandas as pd
import requests

ECB_API = "https://data-api.ecb.europa.eu/service/data"
ECB_USD_SERIES = "EXR/D.USD.EUR.SP00.A"
ECB_ESTR_SERIES = "EST/B.EU000A2X2A25.WT"


# ---------------------------------------------------------------------------
# Cache helper
# ---------------------------------------------------------------------------
def _cached(cache_dir, name, start, end, download, refresh=False):
    """Return a cached Series/DataFrame if it covers [start, end], else download it.

    `download(start, end)` must return a DataFrame indexed by date.

    Next to each cache file we store the start date that was requested. A
    series that only begins later (e.g. a fund listed after that date) then
    still counts as complete and is not downloaded again on every run.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name.replace('/', '_')}.csv"
    requested = path.with_suffix(".start")
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        first = pd.Timestamp(requested.read_text()) if requested.exists() else df.index.min()
        # Allow a few days of slack for weekends/holidays.
        covers = (len(df) > 0 and first <= pd.Timestamp(start) + pd.Timedelta(days=7)
                  and df.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=4))
        if covers:
            return df
    df = download(start, end)
    if len(df) > 0:
        df.to_csv(path)
        requested.write_text(str(pd.Timestamp(start).date()))
    return df


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------
def _yahoo_download(symbol):
    def download(start, end):
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # yfinance emits pandas deprecation noise
            hist = yf.Ticker(symbol).history(start=start, end=pd.Timestamp(end) + pd.Timedelta(days=1),
                                             auto_adjust=False, actions=True)
        if hist.empty:
            return pd.DataFrame(columns=["close", "adj_close", "split"])
        hist.index = hist.index.tz_localize(None).normalize()
        return pd.DataFrame({"close": hist["Close"], "adj_close": hist["Adj Close"],
                             "split": hist.get("Stock Splits", 0.0)})
    return download


def fetch_prices(symbols, start, end, cache_dir, refresh=False):
    """Daily prices for each Yahoo symbol.

    Returns (close, adj_close, splits, problems): close/adj_close are DataFrames
    (date x symbol), splits maps symbol -> list of split dates in the period,
    and `problems` lists symbols that could not be found.
    """
    close, adj, splits, problems = {}, {}, {}, []
    for sym in symbols:
        df = _cached(cache_dir, f"yahoo_{sym}", start, end, _yahoo_download(sym), refresh)
        if df.empty:
            problems.append(f"no price data found for Yahoo symbol '{sym}'")
            continue
        close[sym], adj[sym] = df["close"], df["adj_close"]
        splits[sym] = list(df.index[df["split"].fillna(0) != 0])
    return pd.DataFrame(close), pd.DataFrame(adj), splits, problems


# ---------------------------------------------------------------------------
# ECB
# ---------------------------------------------------------------------------
def _ecb_download(series):
    def download(start, end):
        resp = requests.get(f"{ECB_API}/{series}",
                            params={"format": "csvdata", "startPeriod": str(pd.Timestamp(start).date()),
                                    "endPeriod": str(pd.Timestamp(end).date())},
                            timeout=60)
        resp.raise_for_status()
        from io import StringIO
        raw = pd.read_csv(StringIO(resp.text))
        return pd.DataFrame({"value": raw["OBS_VALUE"].astype(float).values},
                            index=pd.to_datetime(raw["TIME_PERIOD"]))
    return download


def fetch_usd_per_eur(start, end, cache_dir, refresh=False):
    """ECB reference rate: how many US dollars one euro buys, per business day."""
    return _cached(cache_dir, "ecb_usd_per_eur", start, end,
                   _ecb_download(ECB_USD_SERIES), refresh)["value"]


def fetch_estr(start, end, cache_dir, refresh=False):
    """€STR in percent per year, per TARGET business day."""
    return _cached(cache_dir, "ecb_estr", start, end,
                   _ecb_download(ECB_ESTR_SERIES), refresh)["value"]


# ---------------------------------------------------------------------------
# Alignment and currency conversion
# ---------------------------------------------------------------------------
def align(df, calendar, max_gap_days=5):
    """Put a price table on our weekday calendar, forward-filling missing days.

    Markets close on different holidays, so on a day where an exchange was
    closed we carry the last known price forward. Returns (aligned, notes),
    where notes report long gaps (more than `max_gap_days` weekdays filled in a row)
    and series that start after the calendar does.
    """
    df = df.sort_index()
    aligned = df.reindex(df.index.union(calendar)).ffill().reindex(calendar)
    notes = []
    for col in df.columns:
        observed = df[col].dropna().reindex(calendar).notna()
        filled_run = (~observed).astype(int).groupby(observed.cumsum()).cumsum()
        # Ignore the run before the first observation (instrument not yet listed/held).
        after_first = observed.cumsum() > 0
        longest = filled_run[after_first].max() if after_first.any() else 0
        if longest > max_gap_days:
            notes.append(f"'{col}': up to {int(longest)} consecutive weekdays forward-filled")
    return aligned, notes


def to_eur(prices, currencies, usd_per_eur):
    """Convert a (date x symbol) price table to EUR.

    `currencies` maps symbol -> 'EUR' or 'USD'. Other currencies are rejected
    rather than silently mis-converted.
    """
    out = prices.copy()
    for sym in prices.columns:
        ccy = currencies[sym]
        if ccy == "USD":
            out[sym] = prices[sym] / usd_per_eur
        elif ccy != "EUR":
            raise ValueError(f"currency {ccy} for '{sym}' not supported (only EUR and USD)")
    return out
