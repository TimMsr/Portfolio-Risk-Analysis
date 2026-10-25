"""Generate the synthetic sample data in data_sample/.

    python data_sample/generate_sample.py

Creates two files in exactly the same format as the real broker exports:
  - ibkr_activity_sample.csv     (Interactive Brokers Activity Statement)
  - revolut_sample.csv           (Revolut trading-account statement)

The account, deposits and trades are invented. Trade prices are taken from
real market closes on the trade date (via yfinance), so the sample produces
realistic-looking results. The random generator is seeded, so the output is
the same every time (given the same market data).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import prices  # noqa: E402

HERE = Path(__file__).resolve().parent
START, END = pd.Timestamp("2024-01-15"), pd.Timestamp("2026-10-22")
CACHE = ROOT / ".cache"

# Sample instruments: broker symbol -> (Yahoo symbol, currency, sleeve, ISIN, name, exchange)
SAMPLE = {
    "VWCE": ("VWCE.DE", "EUR", "etf", "IE00BK5BQT80", "VANGUARD FTSE ALL-WORLD UCITS", "IBIS"),
    "SXR8": ("SXR8.DE", "EUR", "etf", "IE00B5BMR087", "ISHARES CORE S&P 500", "IBIS"),
    "EXSA": ("EXSA.DE", "EUR", "etf", "DE0002635307", "ISHARES STOXX EUROPE 600", "IBIS"),
    "AAPL": ("AAPL", "USD", "stock", "US0378331005", "APPLE INC", "NASDAQ"),
    "MSFT": ("MSFT", "USD", "stock", "US5949181045", "MICROSOFT CORP", "NASDAQ"),
    "NVDA": ("NVDA", "USD", "stock", "US67066G1040", "NVIDIA CORP", "NASDAQ"),
}


def market_data():
    symbols = [v[0] for v in SAMPLE.values()]
    close, _, _, problems = prices.fetch_prices(symbols, START - pd.Timedelta(days=10), END, CACHE)
    if problems:
        print("\n".join(problems))
    fx = prices.fetch_usd_per_eur(START - pd.Timedelta(days=10), END, CACHE)
    return close, fx


def price_on(close, broker_symbol, day):
    """Market close on `day` (or the last close before it), rounded like a broker fill."""
    series = close[SAMPLE[broker_symbol][0]].dropna().loc[:day]
    return round(float(series.iloc[-1]), 2)


def fx_on(fx, day):
    return round(float(fx.loc[:day].iloc[-1]), 4)


def revolut_rows(close, fx, rng):
    """Monthly deposits into an ETF, plus USD deposits used to buy US stocks."""
    rows = []

    def add(ts, ticker, kind, qty, price, total, ccy, rate):
        rows.append({"Date": ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "Ticker": ticker, "Type": kind,
                     "Quantity": qty, "Price per share": f"{ccy} {price}" if price != "" else "",
                     "Total Amount": f"{ccy} {total}", "Currency": ccy, "FX Rate": f"{rate:.4f}"})

    for month in pd.date_range(START, END, freq="MS"):
        day = month + pd.offsets.BDay(rng.integers(3, 15))
        if day > END:
            break
        amount = int(rng.choice([500, 750, 1000, 1500]))
        add(day + pd.Timedelta(hours=9), "", "CASH TOP-UP", "", "", amount, "EUR", 1.0)
        px = price_on(close, "EXSA", day)
        qty = round(amount / px, 8)
        add(day + pd.Timedelta(hours=10), "EXSA", "BUY - MARKET", qty, px, amount, "EUR", 1.0)
        if month.month in (3, 6, 9, 12):
            # Quarterly distribution, roughly 0.6% of the position's value.
            add(day + pd.Timedelta(days=12, hours=15), "EXSA", "DIVIDEND", "", "",
                round(amount * 0.006 * (month.year - 2023), 2), "EUR", 1.0)

    # US stocks bought with USD top-ups; one partial sale to exercise the parser.
    for ticker, date, qty in [("AAPL", "2024-03-12", 10), ("MSFT", "2024-09-10", 5),
                              ("NVDA", "2025-02-11", 20), ("AAPL", "2025-06-10", 5),
                              ("NVDA", "2026-01-14", 10)]:
        day = pd.Timestamp(date)
        px = price_on(close, ticker, day)
        total = round(qty * px, 2)
        rate = fx_on(fx, day)
        add(day + pd.Timedelta(hours=14), "", "CASH TOP-UP", "", "", total, "USD", rate)
        add(day + pd.Timedelta(hours=15), ticker, "BUY - LIMIT", qty, px, total, "USD", rate)
    day = pd.Timestamp("2026-05-12")
    px = price_on(close, "MSFT", day)
    add(day + pd.Timedelta(hours=15), "MSFT", "SELL - LIMIT", 2, px, round(2 * px, 2), "USD", fx_on(fx, day))

    add(pd.Timestamp("2024-04-02 08:00"), "", "CUSTODY FEE", "", "", -0.10, "EUR", 1.0)
    add(pd.Timestamp("2025-07-21 15:00"), "", "CASH WITHDRAWAL", "", "", -20, "EUR", 1.0)
    df = pd.DataFrame(rows).sort_values("Date", kind="stable")
    return df


def ibkr_rows(close, rng):
    """A larger account: quarterly deposits invested in two ETFs, with commissions."""
    trades, flows, dividends = [], [], []
    for q in pd.date_range("2024-02-01", END, freq="QS-FEB"):
        day = q + pd.offsets.BDay(rng.integers(1, 10))
        amount = int(rng.choice([3000, 5000, 8000]))
        flows.append((day, amount))
        for sym, share in [("VWCE", 0.7), ("SXR8", 0.3)]:
            trade_day = day + pd.offsets.BDay(1)
            px = price_on(close, sym, trade_day)
            qty = int(amount * share // px)
            if qty == 0:
                continue
            proceeds = -round(qty * px, 2)
            commission = -round(max(1.25, 0.0005 * qty * px), 6)
            trades.append((sym, trade_day, qty, px, proceeds, commission))
    held_vwce = 0
    for q in pd.date_range("2024-03-25", END, freq="QS-MAR"):
        held_vwce = sum(t[2] for t in trades if t[0] == "VWCE" and t[1] < q)
        if held_vwce:
            dividends.append((q + pd.offsets.BDay(2), round(held_vwce * 0.25, 2)))
    return trades, flows, dividends


def write_ibkr(path, trades, flows, dividends):
    lines = [
        "Statement,Header,Field Name,Field Value",
        "Statement,Data,BrokerName,Interactive Brokers Ireland Limited",
        "Statement,Data,Title,Activity Statement",
        f'Statement,Data,Period,"January 1, 2024 - {END.strftime("%B %-d, %Y")}"',
        "Account Information,Header,Field Name,Field Value",
        "Account Information,Data,Name,Sample Investor",
        "Account Information,Data,Account,SAMPLE-ACCOUNT",
        "Account Information,Data,Base Currency,EUR",
        "Mark-to-Market Performance Summary,Header,Asset Category,Symbol,Prior Quantity,"
        "Current Quantity,Prior Price,Current Price,Mark-to-Market P/L Position,"
        "Mark-to-Market P/L Transaction,Mark-to-Market P/L Commissions,Mark-to-Market P/L Other,"
        "Mark-to-Market P/L Total,Code",
    ]
    for sym in ("VWCE", "SXR8"):
        qty = sum(t[2] for t in trades if t[0] == sym)
        lines.append(f"Mark-to-Market Performance Summary,Data,Stocks,{sym},0,{qty},--,0,0,0,0,0,0,")
    lines += [
        "Cash Report,Header,Currency Summary,Currency,Total,Securities,Futures,",
        "Cash Report,Data,Starting Cash,Base Currency Summary,0,0,0,",
        "Cash Report,Data,Starting Cash,EUR,0,0,0,",
        "Trades,Header,DataDiscriminator,Asset Category,Currency,Symbol,Date/Time,Quantity,"
        "T. Price,C. Price,Proceeds,Comm/Fee,Basis,Realized P/L,MTM P/L,Code",
    ]
    for sym, day, qty, px, proceeds, comm in sorted(trades, key=lambda t: t[1]):
        qty_text = f'"{qty:,}"' if qty >= 1000 else str(qty)
        lines.append(f'Trades,Data,Order,Stocks,EUR,{sym},"{day.date()}, 05:30:00",{qty_text},'
                     f"{px},{px},{proceeds},{comm},{-proceeds - comm},0,0,O")
    lines.append("Deposits & Withdrawals,Header,Currency,Settle Date,Description,Amount")
    for day, amount in flows:
        lines.append(f"Deposits & Withdrawals,Data,EUR,{day.date()},Electronic Fund Transfer,{amount}")
    lines.append(f"Deposits & Withdrawals,Data,Total,,,{sum(a for _, a in flows)}")
    lines.append("Dividends,Header,Currency,Date,Description,Amount")
    for day, amount in dividends:
        lines.append(f"Dividends,Data,EUR,{day.date()},VWCE(IE00BK5BQT80) Cash Dividend "
                     f"EUR 0.25 per Share (Ordinary Dividend),{amount}")
    lines.append(f"Dividends,Data,Total,,,{round(sum(a for _, a in dividends), 2)}")
    lines += [
        "Withholding Tax,Header,Currency,Date,Description,Amount,Code",
        "Withholding Tax,Data,EUR,2025-10-06,Withholding @ 20% on Credit Interest for Sep-2025,-0.84,",
        "Withholding Tax,Data,Total,,,-0.84,",
        "Interest,Header,Currency,Date,Description,Amount",
        "Interest,Data,EUR,2025-10-03,EUR Credit Interest for Sep-2025,4.20",
        "Interest,Data,Total,,,4.20",
        "Financial Instrument Information,Header,Asset Category,Symbol,Description,Conid,"
        "Security ID,Underlying,Listing Exch,Multiplier,Type,Code",
    ]
    for sym in ("VWCE", "SXR8"):
        _, _, _, isin, name, exch = SAMPLE[sym]
        lines.append(f"Financial Instrument Information,Data,Stocks,{sym},{name},0,{isin},{sym},{exch},1,ETF,")
    path.write_text("\n".join(lines) + "\n")


def main():
    rng = np.random.default_rng(42)
    close, fx = market_data()
    revolut_rows(close, fx, rng).to_csv(HERE / "revolut_sample.csv", index=False)
    write_ibkr(HERE / "ibkr_activity_sample.csv", *ibkr_rows(close, rng))
    pd.DataFrame([{"broker_symbol": k, "yahoo_symbol": v[0], "currency": v[1], "sleeve": v[2]}
                  for k, v in SAMPLE.items()]).to_csv(HERE / "instruments.csv", index=False)
    print("Sample data written to", HERE)


if __name__ == "__main__":
    main()
