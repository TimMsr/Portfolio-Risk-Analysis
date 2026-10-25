"""Privacy check: scan every file outside data_private/ for private information.

Run this before every commit (the git hook in hooks/pre-commit does it for you):

    python check_public_outputs.py          # report problems without echoing private terms
    python check_public_outputs.py --show   # local only: also print which term matched

It FAILS (exit code 1) if it finds any of:
  1. Account-number patterns (Interactive Brokers account ids, IBANs).
  2. Currency amounts, e.g. a currency code or symbol next to a number.
     (Skipped inside data_sample/, whose synthetic files must look like real
     broker exports and therefore contain amounts.)
  3. Private terms: ticker symbols, ISINs and account ids of the real
     portfolio, plus anything listed in data_private/private_terms.txt.
     The real tickers are read from the private files themselves, so the
     list of what to hide never has to be written into the public repo.

It WARNS (but does not fail) if a real deposit/withdrawal date appears
(ISO or written out), because a date can also legitimately appear for other
reasons (e.g. the date of a drawdown trough). Review every warning by hand.
Dates inside data_sample/ are not checked: that data is generated from a
random seed and can only match a real date by coincidence.

Images (PNG) cannot be scanned as text. The plotting code is written so that
charts never contain tickers or amounts; check them by eye before committing.
"""

import csv
import datetime
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIVATE_DIR = ROOT / "data_private"
SHOW_TERMS = "--show" in sys.argv
SAMPLE_DIR = ROOT / "data_sample"

# Folders that are never published (git internals, tooling, local caches).
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache",
             ".ipynb_checkpoints", ".cache", "data_private"}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".pyc", ".ico"}

# 1. Account numbers: IBKR ids are 'U' plus 7-8 digits ('DU' for paper accounts);
#    IBANs are two letters, two check digits, then 11-30 letters/digits.
ACCOUNT_PATTERNS = [
    re.compile(r"\bD?U\d{6,9}\b"),
    re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b"),
]

# 2. Currency amounts: a code/symbol directly before or after a number.
CURRENCY = r"(?:€|\$|£|\b(?:EUR|USD|GBP|CHF)\b)"
AMOUNT_PATTERNS = [
    re.compile(CURRENCY + r"\s?[-+]?\d"),
    re.compile(r"\d\s?" + CURRENCY),
]


def load_private_terms():
    """Collect the strings that must not be published, from data_private/."""
    terms, dates = set(), set()
    if not PRIVATE_DIR.exists():
        return terms, dates

    terms_file = PRIVATE_DIR / "private_terms.txt"
    if terms_file.exists():
        for line in terms_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                terms.add(line)

    instruments = PRIVATE_DIR / "instruments.csv"
    if instruments.exists():
        with open(instruments, newline="") as f:
            for row in csv.DictReader(f):
                terms.add(row["broker_symbol"].strip())
                terms.add(row["yahoo_symbol"].strip())
                # Also the bare symbol without the exchange suffix (e.g. "ABC.DE" -> "ABC").
                terms.add(row["yahoo_symbol"].split(".")[0].strip())

    for path in PRIVATE_DIR.glob("*.csv"):
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        header = rows[0]
        if "Ticker" in header and "Type" in header:  # Revolut statement
            t, ty, d = header.index("Ticker"), header.index("Type"), header.index("Date")
            for row in rows[1:]:
                if len(row) <= max(t, ty, d):
                    continue
                if row[t]:
                    terms.add(row[t])
                if row[ty] in ("CASH TOP-UP", "CASH WITHDRAWAL"):
                    dates.add(row[d][:10])
        elif header[:2] == ["Statement", "Header"]:  # IBKR activity statement
            for row in rows:
                if len(row) < 4 or row[1] != "Data":
                    continue
                if row[0] == "Account Information" and row[2] == "Account":
                    terms.add(row[3])
                elif row[0] == "Financial Instrument Information":
                    terms.add(row[3])  # symbol
                    terms.add(row[6])  # ISIN (Security ID)
                elif row[0] == "Deposits & Withdrawals" and row[2] != "Total":
                    dates.add(row[3])
    terms.discard("")
    dates = set().union(*(date_forms(d) for d in dates)) if dates else set()
    return terms, dates


# Inline/display maths in Markdown ($...$ with a LaTeX command or ^ _ =) is not money.
MATH_SPAN = re.compile(r"\$\$[^$]*\$\$|\$[^$\n]*[\\^_=][^$\n]*\$")


def date_forms(iso):
    """'2020-05-04' -> the ways that date is commonly written in text."""
    d = datetime.date.fromisoformat(iso)
    month = d.strftime("%B")
    return {iso, f"{d.day} {month} {d.year}", f"{month} {d.day}, {d.year}",
            d.strftime("%d.%m.%Y"), d.strftime("%d/%m/%Y")}


def text_of(path):
    """Return the publishable text of a file (for notebooks: sources and text outputs only)."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".md":
        return MATH_SPAN.sub(" ", raw)
    if path.suffix != ".ipynb":
        return raw
    # Notebooks embed images as base64, which can contain random matches.
    # Only scan the code/markdown and the text outputs.
    nb = json.loads(raw)
    parts = []
    for cell in nb.get("cells", []):
        parts.append("".join(cell.get("source", [])))
        for out in cell.get("outputs", []):
            parts.append("".join(out.get("text", [])))
            data = out.get("data", {})
            for key in ("text/plain", "text/html", "text/markdown"):
                parts.append("".join(data.get(key, [])))
    return MATH_SPAN.sub(" ", "\n".join(parts))


def iter_files():
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts) or not path.is_file():
            continue
        yield path, rel


def main():
    terms, dates = load_private_terms()
    if not terms:
        print("NOTE: data_private/ not found or empty - only generic patterns are checked.")
    term_patterns = [(t, re.compile(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])"))
                     for t in sorted(terms)]

    failures, warnings, skipped = [], [], []
    for path, rel in iter_files():
        if path.suffix.lower() in BINARY_SUFFIXES:
            skipped.append(str(rel))
            continue
        in_sample = SAMPLE_DIR in path.parents
        for lineno, line in enumerate(text_of(path).splitlines(), start=1):
            for pat in ACCOUNT_PATTERNS:
                for m in pat.finditer(line):
                    failures.append(f"{rel}:{lineno}: account-number pattern '{m.group()}'")
            if not in_sample:
                for pat in AMOUNT_PATTERNS:
                    for m in pat.finditer(line):
                        failures.append(f"{rel}:{lineno}: currency amount near '{m.group()}'")
            for term, pat in term_patterns:
                if pat.search(line):
                    # By default do not echo the term (logs may be public); --show prints it.
                    shown = term if SHOW_TERMS else "(use --show to print)"
                    failures.append(f"{rel}:{lineno}: private term found {shown}")
            # data_sample/ is generated from a seed, without private input, so its
            # dates can only match real ones by coincidence.
            for d in ([] if in_sample else dates):
                if d in line:
                    warnings.append(f"{rel}:{lineno}: a real deposit/withdrawal date appears")

    for w in warnings:
        print("WARNING:", w)
    if skipped:
        print(f"Not scanned (binary, check by eye): {', '.join(skipped)}")
    if failures:
        for f in failures:
            print("FAIL:", f)
        print(f"\n{len(failures)} problem(s) found. Do not commit.")
        sys.exit(1)
    print("OK: no private information found in public files.")


if __name__ == "__main__":
    main()
