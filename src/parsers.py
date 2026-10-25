"""Read broker exports and normalise them into one transaction table.

Every parser returns a DataFrame with these columns (the "normalised" format):

    date        calendar date of the transaction (pandas Timestamp, no time)
    account     "ibkr" or "revolut"
    instrument  broker ticker symbol, empty string for pure cash movements
    type        buy | sell | dividend | interest | fee | tax | deposit | withdrawal
    quantity    shares bought (+) or sold (-); 0 for non-trades
    price       trade price per share in `currency`; NaN for non-trades
    amount      signed effect on cash in `currency` (money in = +, money out = -)
    fee         commission included in `amount`, as a positive number (cost)
    currency    currency of `price` and `amount`
    note        free text; non-empty when a value is estimated

Sign convention: `amount` is always the change in the account's cash. A buy
therefore has a negative amount (cash leaves), a deposit a positive one.
"""

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["date", "account", "instrument", "type", "quantity", "price",
           "amount", "fee", "currency", "note"]

# ---------------------------------------------------------------------------
# Revolut trading-account statement: column and type mapping
# ---------------------------------------------------------------------------
# Header of the export:
#   Date,Ticker,Type,Quantity,Price per share,Total Amount,Currency,FX Rate
REVOLUT_COLUMNS = {
    "Date": "date",                # ISO timestamp in UTC, e.g. 2020-05-04T09:30:00.000Z
    "Ticker": "instrument",        # empty for cash movements
    "Type": "raw_type",            # see REVOLUT_TYPES
    "Quantity": "quantity",        # always positive in the export
    "Price per share": "price",    # currency code, a space, then the number
    "Total Amount": "total",       # same format as the price; negative for withdrawals/fees
    "Currency": "currency",
    "FX Rate": "fx_rate",          # not used: we convert with ECB rates instead
}
REVOLUT_TYPES = {
    "CASH TOP-UP": "deposit",
    "CASH WITHDRAWAL": "withdrawal",
    "BUY - MARKET": "buy",
    "BUY - LIMIT": "buy",
    "BUY - STOP": "buy",
    "SELL - MARKET": "sell",
    "SELL - LIMIT": "sell",
    "SELL - STOP": "sell",
    "DIVIDEND": "dividend",
    "CUSTODY FEE": "fee",
}

# ---------------------------------------------------------------------------
# Interactive Brokers Activity Statement (CSV): section and column mapping
# ---------------------------------------------------------------------------
# The file is a stack of tables. Column 1 is the section name, column 2 is
# "Header" (column names follow) or "Data" (values follow).
IBKR_SECTIONS = {
    "trades": "Trades",                        # DataDiscriminator, Asset Category, Currency, Symbol,
                                               # Date/Time, Quantity, T. Price, Proceeds, Comm/Fee, ...
    "cash_flows": "Deposits & Withdrawals",    # Currency, Settle Date, Description, Amount
    "dividends": "Dividends",                  # Currency, Date, Description, Amount
    "withholding": "Withholding Tax",          # Currency, Date, Description, Amount, Code
    "interest": "Interest",                    # Currency, Date, Description, Amount
    "fees": "Other Fees",                      # Currency, Date, Description, Amount
    "cash_report": "Cash Report",              # Currency Summary, Currency, Total, ...
    "mtm": "Mark-to-Market Performance Summary",  # Asset Category, Symbol, Prior Quantity, ...
    "statement": "Statement",                  # Field Name, Field Value (contains "Period")
}


def parse_number(text):
    """Turn '1,250', '-12.30', a number prefixed by a currency code, or '' into a float (NaN if empty)."""
    if text is None:
        return np.nan
    cleaned = re.sub(r"[A-Za-z€$£\s,]", "", str(text))
    return float(cleaned) if cleaned not in ("", "-", "--") else np.nan


# ---------------------------------------------------------------------------
# Revolut
# ---------------------------------------------------------------------------
def parse_revolut(path):
    """Parse a Revolut trading-account statement CSV into the normalised format."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(REVOLUT_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"{Path(path).name}: missing Revolut columns {sorted(missing)}")
    df = raw.rename(columns=REVOLUT_COLUMNS)

    unknown = sorted(set(df["raw_type"]) - set(REVOLUT_TYPES))
    if unknown:
        raise ValueError(f"{Path(path).name}: unknown Revolut transaction types {unknown}. "
                         "Add them to REVOLUT_TYPES.")

    out = pd.DataFrame({
        # Timestamps are UTC; we only need the calendar day.
        "date": pd.to_datetime(df["date"].str[:10]),
        "account": "revolut",
        "instrument": df["instrument"].str.strip(),
        "type": df["raw_type"].map(REVOLUT_TYPES),
        "quantity": df["quantity"].map(parse_number).fillna(0.0),
        "price": df["price"].map(parse_number),
        "total": df["total"].map(parse_number),
        "currency": df["currency"].str.strip(),
    })

    # Cash effect. The export shows trade totals as positive numbers and
    # withdrawals/fees already negative, so we set the sign from the type.
    sign = out["type"].map({"buy": -1, "sell": 1}).fillna(1.0)
    out["amount"] = np.where(out["type"].isin(["buy", "sell"]),
                             sign * out["total"].abs(), out["total"])
    out.loc[out["type"] == "sell", "quantity"] *= -1

    # Revolut does not report commissions separately. We estimate them as
    # total paid minus quantity x price, when that difference is positive.
    # Small negative differences come from the displayed price being rounded.
    is_trade = out["type"].isin(["buy", "sell"])
    gross = (out["quantity"].abs() * out["price"]).where(is_trade)
    implied = np.where(out["type"] == "buy", out["total"].abs() - gross,
                       gross - out["total"].abs())
    out["fee"] = np.where(is_trade, np.clip(np.round(implied, 2), 0, None), 0.0)
    out["note"] = np.where(out["fee"] > 0, "fee implied from total minus qty x price", "")
    return out[COLUMNS]


# ---------------------------------------------------------------------------
# Interactive Brokers
# ---------------------------------------------------------------------------
def read_ibkr_sections(path):
    """Split an IBKR Activity Statement into {section name: [DataFrame, ...]}.

    A section can contain several tables (several 'Header' rows), so each
    section maps to a list of DataFrames in file order.
    """
    tables = {}  # section name -> list of (header, rows)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            name, kind, values = row[0], row[1], row[2:]
            if kind == "Header":
                tables.setdefault(name, []).append((values, []))
            elif kind == "Data" and name in tables:
                header, rows = tables[name][-1]
                rows.append((values + [""] * len(header))[:len(header)])  # pad/trim to header
    return {name: [pd.DataFrame(rows, columns=header) for header, rows in section]
            for name, section in tables.items()}


def _ibkr_table(sections, key):
    """First table of a section, or an empty DataFrame if the section is absent."""
    tables = sections.get(IBKR_SECTIONS[key], [])
    return tables[0] if tables else pd.DataFrame()


def ibkr_statement_start(sections):
    """First day covered by the statement, from 'Period' (e.g. 'January 1, 2026 - ...')."""
    info = _ibkr_table(sections, "statement")
    period = info.loc[info["Field Name"] == "Period", "Field Value"].iloc[0]
    return pd.Timestamp(pd.to_datetime(period.split(" - ")[0]))


def _symbol_from_description(text):
    """'ABC(IE00XXXXXXXX) Cash Dividend ...' -> 'ABC'; '' if no symbol at the start."""
    m = re.match(r"^([A-Z0-9.]+)\s?\(", text)
    return m.group(1) if m else ""


def parse_ibkr(path, include_opening_cash=True):
    """Parse an IBKR Activity Statement CSV into the normalised format.

    If `include_opening_cash` is True, cash held at the start of the statement
    is recorded as a deposit on the first day. We do not have the account's
    earlier history, so this is an estimate and is flagged in `note`.
    """
    sections = read_ibkr_sections(path)
    records = []

    def add(**kw):
        rec = {"instrument": "", "quantity": 0.0, "price": np.nan, "fee": 0.0, "note": ""}
        rec.update(kw)
        records.append(rec)

    # Positions held before the statement starts cannot be reconstructed.
    mtm = _ibkr_table(sections, "mtm")
    if not mtm.empty:
        stocks = mtm[mtm["Asset Category"] == "Stocks"]
        prior = stocks["Prior Quantity"].map(parse_number).fillna(0)
        if (prior != 0).any():
            raise ValueError(f"{Path(path).name}: positions existed before the statement "
                             "start. Export a statement that starts before the first trade.")

    start = ibkr_statement_start(sections)
    if include_opening_cash:
        cash = _ibkr_table(sections, "cash_report")
        opening = cash[(cash["Currency Summary"] == "Starting Cash")
                       & (cash["Currency"] != "Base Currency Summary")]
        for _, row in opening.iterrows():
            value = parse_number(row["Total"])
            if value:
                add(date=start, type="deposit", amount=value, currency=row["Currency"],
                    note="opening cash of statement treated as deposit (estimate)")

    trades = _ibkr_table(sections, "trades")
    if not trades.empty:
        trades = trades[trades["DataDiscriminator"] == "Order"]
        other = set(trades["Asset Category"]) - {"Stocks"}
        if other:
            raise ValueError(f"{Path(path).name}: unsupported trade categories {sorted(other)}")
        for _, row in trades.iterrows():
            qty = parse_number(row["Quantity"])
            commission = parse_number(row["Comm/Fee"])  # negative number = cost
            add(date=pd.Timestamp(row["Date/Time"][:10]),
                instrument=row["Symbol"],
                type="buy" if qty > 0 else "sell",
                quantity=qty,
                price=parse_number(row["T. Price"]),
                amount=parse_number(row["Proceeds"]) + commission,
                fee=-commission,
                currency=row["Currency"])

    flows = _ibkr_table(sections, "cash_flows")
    for _, row in flows[flows.get("Currency", pd.Series(dtype=str)) != "Total"].iterrows():
        value = parse_number(row["Amount"])
        add(date=pd.Timestamp(row["Settle Date"]),
            type="deposit" if value > 0 else "withdrawal",
            amount=value, currency=row["Currency"])

    for key, kind in [("dividends", "dividend"), ("withholding", "tax"),
                      ("interest", "interest"), ("fees", "fee")]:
        table = _ibkr_table(sections, key)
        if table.empty:
            continue
        table = table[~table["Currency"].str.startswith("Total")]
        for _, row in table.iterrows():
            add(date=pd.Timestamp(row["Date"]),
                instrument=_symbol_from_description(row["Description"]),
                type=kind, amount=parse_number(row["Amount"]), currency=row["Currency"])

    out = pd.DataFrame(records)
    out["account"] = "ibkr"
    return out[COLUMNS]


# ---------------------------------------------------------------------------
# Load a whole data folder
# ---------------------------------------------------------------------------
def load_transactions(data_dir):
    """Parse every broker file in `data_dir` and return one sorted table.

    Files are recognised by their first line, so names do not matter. With
    several IBKR statements, only the earliest one contributes opening cash.
    """
    data_dir = Path(data_dir)
    revolut, ibkr = [], []
    for path in sorted(data_dir.glob("*.csv")):
        first = path.open(encoding="utf-8-sig").readline()
        if first.startswith("Date,Ticker,Type"):
            revolut.append(parse_revolut(path))
        elif first.startswith("Statement,Header"):
            ibkr.append((ibkr_statement_start(read_ibkr_sections(path)), path))

    frames = list(revolut)
    for i, (_, path) in enumerate(sorted(ibkr)):
        frames.append(parse_ibkr(path, include_opening_cash=(i == 0)))
    if not frames:
        raise FileNotFoundError(f"No Revolut or IBKR exports found in {data_dir}")

    tx = pd.concat(frames, ignore_index=True)
    # Overlapping statements would list the same transaction twice.
    tx = tx.drop_duplicates().sort_values(["date", "account"], kind="stable")
    return tx.reset_index(drop=True)


def load_instruments(data_dir):
    """Read instruments.csv: broker_symbol -> yahoo_symbol, currency, sleeve (etf/stock)."""
    inst = pd.read_csv(Path(data_dir) / "instruments.csv", dtype=str)
    bad = set(inst["sleeve"]) - {"etf", "stock"}
    if bad:
        raise ValueError(f"instruments.csv: sleeve must be 'etf' or 'stock', got {sorted(bad)}")
    return inst.set_index("broker_symbol")
