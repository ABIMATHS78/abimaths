---
name: analyze-data
description: Answers questions about data, from quick lookups to full analyses, checking joins, duplicates, date ranges and metric definitions before trusting a number. Use with spreadsheets, CSVs or SQL results.
---

# Analyze data

Answer the question, and make sure the answer is right. A query can run without errors and still be wrong.

## 1. Pin down the question

- What decision will the answer support?
- Define every metric in words before computing it. For example, "revenue" means: gross or net? Refunds and tax in or out? Which currency? Which date: order, payment or delivery?
- Fix the scope: date range (with time zone), segments, and what to exclude.

For a quick lookup, state these in one line and go on. For a bigger analysis, confirm them first.

## 2. Look at the data before using it

Report, briefly:
- rows and columns, and what one row represents;
- column types (numbers stored as text, dates as text);
- missing values per important column;
- duplicate rows, and duplicate keys where keys should be unique;
- the date range the data actually covers;
- values that look impossible (negative quantities, future dates).

## 3. Analyze

Use code (Python with pandas, or SQL) so every step can be checked, and show it.

**Joins** are where totals go wrong:
- check that the key is unique on the side that should be unique;
- compare row counts before and after the join; if rows multiplied, explain why or fix it;
- count rows that didn't match, and decide on purpose what to do with them.

**Filters**: apply the agreed date range and exclusions explicitly, and say how many rows each one removed.

## 4. Check the result

- Reconcile one key total a second way, for example a sum from the raw table against the joined result.
- Ask whether the size makes sense: compare it with a known figure or last period.
- For changes over time, check that the definitions and data sources stayed the same.

## 5. Report

Lead with the answer in one or two sentences, then:
- the numbers, with their definitions and the date range;
- a chart if it helps, with labelled axes and a title that states the finding;
- caveats: missing data, small samples, and the difference between correlation and causation;
- recommended next steps.

## Rules

- Never invent or estimate a number without saying so. If the data can't answer the question, say what's missing.
- Keep personal data out of outputs unless it's needed, and never paste it into other tools.
- Round sensibly, and keep units and currencies visible.
