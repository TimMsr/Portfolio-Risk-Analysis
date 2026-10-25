"""LOCAL ONLY: compare computed returns with independent figures. Prints, never saves.

Two checks:
  1. Interactive Brokers: the Activity Statement reports its own time-weighted
     return for the statement period. We recompute the same number for the IBKR
     account alone and print the difference.
  2. Combined portfolio: compare with a figure from another tracking app,
     entered in data_private/reference_return.txt (start=, end=, return_pct=).
     Apps differ in method, so three versions of our return are printed:
       - time-weighted (what this project reports),
       - money-weighted (internal rate of return over the period, not annualised),
       - simple (profit / net amount deposited).

The script lives in the public repo, but every number it prints comes from
data_private/ and stays on your machine.
"""

import csv
from pathlib import Path

import pandas as pd

from src import metrics, pipeline
from src.parsers import ibkr_statement_start, read_ibkr_sections

PRIVATE = Path(__file__).resolve().parent / "data_private"


def ibkr_reported_twr():
    """(start, end, TWR in %) from the newest IBKR statement in data_private/."""
    found = []
    for path in PRIVATE.glob("*.csv"):
        if path.open(encoding="utf-8-sig").readline().startswith("Statement,Header"):
            sections = read_ibkr_sections(path)
            nav_tables = sections.get("Net Asset Value", [])
            for table in nav_tables:
                if "Time Weighted Rate of Return" in table.columns:
                    twr = float(table.iloc[0, 0].rstrip("%"))
                    period = sections["Statement"][0]
                    period = period.loc[period["Field Name"] == "Period", "Field Value"].iloc[0]
                    end = pd.Timestamp(pd.to_datetime(period.split(" - ")[1]))
                    found.append((ibkr_statement_start(sections), end, twr))
    return max(found, key=lambda x: x[1]) if found else None


def money_weighted_return(flows, end_value, end_date):
    """Return over the period at which the deposits would grow to end_value.

    Finds the constant daily rate d such that
        sum over deposits f_i x (1 + d) ^ (days from deposit i to end_date) = end_value
    by bisection (the left side increases with d), then reports
    (1 + d) ^ (days from first deposit to end_date) - 1, i.e. not annualised.
    """
    flows = flows[flows != 0]

    def future_value(d):
        return sum(f * (1 + d) ** (end_date - day).days for day, f in flows.items())

    lo, hi = -0.01, 0.01
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if future_value(mid) > end_value else (mid, hi)
    return (1 + mid) ** (end_date - flows.index.min()).days - 1


def main():
    # 1. IBKR statement versus our recomputation
    reported = ibkr_reported_twr()
    if reported:
        start, end, twr = reported
        res = pipeline.run(private=True, accounts=["ibkr"], start=start, end=end, make_charts=False)
        ours = res["summary"].loc["Portfolio", "total_return"] * 100
        print(f"IBKR {start.date()} to {end.date()}: ours {ours:.3f}%  IBKR {twr:.3f}%  "
              f"difference {ours - twr:+.3f} pp")

    # 2. Combined portfolio versus another app
    ref_file = PRIVATE / "reference_return.txt"
    ref = {}
    if ref_file.exists():
        for line in ref_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                ref[k.strip()] = v.split("(")[0].strip()
    if not ref.get("return_pct"):
        print("No reference figure in data_private/reference_return.txt - skipping check 2.")
        return

    res = pipeline.run(private=True, end=ref.get("end") or None, make_charts=False, return_book=True)
    r = res["daily_returns"]["portfolio"]
    if ref.get("start"):
        r = r.loc[ref["start"]:]
    twr = metrics.total_return(r) * 100

    book = res["book"]
    flows = book["external_flow"]
    end_value = book["value_total"].iloc[-1]
    mwr = money_weighted_return(flows, end_value, book.index[-1]) * 100
    simple = (end_value / flows.sum() - 1) * 100

    target = float(ref["return_pct"])
    print(f"Reference app: {target:.2f}%  (method: {ref.get('method') or 'unknown'})")
    for name, value in [("time-weighted", twr), ("money-weighted", mwr), ("simple", simple)]:
        print(f"  ours {name:15s} {value:7.2f}%   difference {value - target:+.2f} pp")


if __name__ == "__main__":
    main()
