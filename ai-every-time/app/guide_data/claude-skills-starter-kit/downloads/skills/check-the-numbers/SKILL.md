---
name: check-the-numbers
description: Answers questions about data and proves the answer first, profiling the file and checking joins, duplicates, date ranges and metric definitions. Use with spreadsheets, CSVs, SQL or dashboards.
---

# Check the numbers

A query can run without errors and still be wrong. Answer the question, and show why the answer can be trusted, so that when someone asks "Are we sure this is right?", the work already says yes.

## 1. Pin down the question

- What decision will the answer support?
- Define every metric in words before computing it. "Revenue": gross or net? Refunds and tax in or out? Which currency? Counted on the order, payment or delivery date?
- Fix the scope: the date range (with its time zone), the segments, and what to leave out.

For a quick lookup, state these in one line and carry on. For a bigger analysis, agree them first.

## 2. Profile the data before using it

If you can run scripts, profile each file:

```bash
python scripts/profile_data.py sales.csv --key order_id --date order_date
```

It reports the rows and columns, each column's type, missing values, numbers stored as text, duplicate rows, repeated keys and the dates the data really covers. Without scripts, check the same things by hand.

Then say, in a few lines, what one row represents and anything that needs fixing first.

## 3. Analyse, step by step

Work in code (Python with pandas, or SQL), so every step can be checked, and show it.

**Joins** are where totals go wrong:
- check that the key is unique on the side that should be unique;
- compare row counts before and after; if rows multiplied, explain why or fix it;
- count rows that found no match, and decide on purpose what happens to them.

**Filters**: apply the agreed date range and exclusions explicitly, and say how many rows each one removed.

## 4. Prove it

- Reconcile one key total a second way: for example, the sum from the raw table against the joined result.
- Compare the size with something known, such as last period or a published figure.
- For changes over time, confirm the definitions and data sources stayed the same.

## 5. Report

Lead with the answer in one or two sentences, then:
- the numbers, with their definitions and date range;
- a chart if it helps, whose title states the finding;
- what could make it wrong: missing data, small samples, correlation that isn't causation;
- the next step you'd take.

## Rules

- Never invent a number. If the data can't answer the question, say what's missing.
- Keep personal data out of outputs unless it's needed.
- Round sensibly, and keep units and currencies visible.
