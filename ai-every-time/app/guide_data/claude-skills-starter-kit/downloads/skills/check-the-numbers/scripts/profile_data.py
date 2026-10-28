#!/usr/bin/env python3
"""Profiles a CSV file before you trust it.

Usage:  python profile_data.py FILE.csv [--key COLUMN ...] [--date COLUMN]

For the whole file: its rows and columns, and rows that appear more than once. For each column: what it seems to
hold (numbers, dates or text), how many values are missing, how many are distinct, the smallest and largest
numbers or dates, and numbers stored as text such as "$1,200" or "12%". With --key, whether those columns tell
the rows apart (a key that repeats multiplies rows in a join). With --date, the dates the data really covers,
and any gap of a week or more.

It uses only Python's standard library, so it runs wherever Python does.
"""

import argparse
import csv
import re
import sys
from collections import Counter
from datetime import date, datetime
from itertools import pairwise

MISSING = {"", "na", "n/a", "null", "none", "nan", "-"}
NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
NUMBER_AS_TEXT = re.compile(r"^[+-]?[$€£¥₦]?\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?$|^[+-]?[$€£¥₦]\s?[\d.]+$|^[+-]?[\d.]+\s?%$")
DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ")
MOST_FOR_A_TYPE = 0.8  # the share of a column's values that decides what it holds
GAP_DAYS = 7


def as_date(value: str) -> date | None:
    for f in DATE_FORMATS:
        try:
            return datetime.strptime(value, f).date()
        except ValueError:
            continue
    return None


def kind(value: str) -> str:
    if NUMBER.match(value):
        return "number"
    if NUMBER_AS_TEXT.match(value):
        return "number as text"
    if as_date(value):
        return "date"
    return "text"


def show(value: float) -> str:
    return f"{value:.10g}"


def profile_column(name: str, values: list[str]) -> list[str]:
    present = [v.strip() for v in values if v.strip().lower() not in MISSING]
    missing = len(values) - len(present)
    kinds = Counter(kind(v) for v in present)
    numeric = kinds["number"] + kinds["number as text"]
    lines = []
    if present and numeric / len(present) >= MOST_FOR_A_TYPE:
        seems = "numbers"
        numbers = [float(v) for v in present if NUMBER.match(v)]
        span = f"{show(min(numbers))} to {show(max(numbers))}" if numbers else "none readable as numbers"
        if kinds["number as text"]:
            example = next(v for v in present if kind(v) == "number as text")
            lines.append(f"    ! {kinds['number as text']:,} stored as text, such as \"{example}\": clean them before summing")
    elif present and kinds["date"] / len(present) >= MOST_FOR_A_TYPE:
        seems = "dates"
        dates = [d for d in (as_date(v) for v in present) if d]
        span = f"{min(dates)} to {max(dates)}"
    else:
        seems = "text"
        common = Counter(present).most_common(3)
        span = ", ".join(f'"{v}"' for v, _ in common) if common else "(empty)"
    odd = len(present) - (numeric if seems == "numbers" else kinds["date"] if seems == "dates" else len(present))
    if odd:
        example = next(v for v in present if (kind(v) not in ("number", "number as text") if seems == "numbers" else kind(v) != "date"))
        lines.append(f"    ! {odd:,} that aren't {seems}, such as \"{example}\"")
    head = f"  {name}: {seems}; {missing:,} missing; {len(set(present)):,} distinct; {span}"
    return [head] + lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Profiles a CSV file before you trust it.")
    ap.add_argument("file")
    ap.add_argument("--key", nargs="+", help="the column or columns that should tell the rows apart")
    ap.add_argument("--date", help="a column of dates, to see which dates the data covers")
    args = ap.parse_args()
    try:
        with open(args.file, encoding="utf-8-sig", newline="") as f:
            sample = f.read(64 * 1024)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(f, dialect)
            header = next(reader, None)
            rows = [(reader.line_num, row) for row in reader if any(cell.strip() for cell in row)]
    except OSError as e:
        print(f"Can't read {args.file}: {e.strerror}")
        return 2
    if not header:
        print(f"{args.file} is empty.")
        return 0
    for wanted in (args.key or []) + ([args.date] if args.date else []):
        if wanted not in header:
            print(f'There\'s no column called "{wanted}". The columns are: {", ".join(header)}')
            return 2

    print(f"{args.file}: {len(rows):,} rows, {len(header)} columns")
    short = [n for n, r in rows if len(r) != len(header)]
    if short:
        print(f"  ! {len(short):,} rows don't have {len(header)} values, the first at line {short[0]}")
    seen, repeats, first = Counter(), 0, None
    for n, r in rows:
        seen[tuple(r)] += 1
        if seen[tuple(r)] == 2 and first is None:
            first = n
    repeats = sum(c - 1 for c in seen.values() if c > 1)
    print(f"  Rows that appear more than once: {repeats:,}" + (f" (the first repeat at line {first})" if repeats else ""))

    print("\nColumns:")
    columns = {name: [r[i] if i < len(r) else "" for _, r in rows] for i, name in enumerate(header)}
    for name, values in columns.items():
        print("\n".join(profile_column(name, values)))

    if args.key:
        keys = Counter(tuple(columns[k][i].strip() for k in args.key) for i in range(len(rows)))
        blank = sum(c for key, c in keys.items() if any(v.lower() in MISSING for v in key))
        repeated = {key: c for key, c in keys.items() if c > 1}
        label = " + ".join(args.key)
        print(f"\nKey {label}:", end=" ")
        if repeated:
            key, c = max(repeated.items(), key=lambda kv: kv[1])
            print(f"{len(repeated):,} values repeat (such as {' / '.join(key)}, {c} times). "
                  f"A join on it will multiply rows: find out why before joining.")
        else:
            print("every row has its own value.")
        if blank:
            print(f"  ! {blank:,} rows have no key")

    if args.date:
        dates = sorted({d for d in (as_date(v.strip()) for v in columns[args.date]) if d})
        if not dates:
            print(f"\nNo dates in {args.date} could be read (the script reads dates like 2026-10-28).")
        else:
            print(f"\nDates in {args.date}: {dates[0]} to {dates[-1]}, on {len(dates):,} different days.")
            gaps = [(a, b) for a, b in pairwise(dates) if (b - a).days >= GAP_DAYS]
            for a, b in gaps[:5]:
                print(f"  ! nothing between {a} and {b} ({(b - a).days} days)")
            if len(gaps) > 5:
                print(f"  ! and {len(gaps) - 5} more gaps of {GAP_DAYS} days or more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
