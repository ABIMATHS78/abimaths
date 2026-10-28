---
name: data-cleaning-plan
description: Plans the cleaning steps for a messy dataset and records every decision in a log, so the work can be checked and repeated. Use when a spreadsheet, CSV or export needs tidying before analysis.
---

# Data cleaning plan

Plan how to make a messy dataset fit for analysis, and record every decision, so anyone can see what changed, why, and how many rows it touched.

## Before you start

You need the dataset or a sample of it, a description of the columns if one exists, and what the data will be used for. The use matters: a shortcut that's fine for a quick chart may be wrong for a financial report. If the purpose is missing, ask for it in one short message.

## Steps

1. **Profile the data.** Count the rows and columns. For each column, note its type, missing values, number of distinct values, range and a few examples. Say what one row represents.
2. **List the issues.** Look for duplicate rows, missing values, inconsistent formats (dates, phone numbers, capital letters, extra spaces), mixed units or currencies, impossible values, outliers, and categories spelled several ways.
3. **Decide on each fix.** For every issue, choose an action and give the reason: standardise, correct from a reliable source, flag, exclude or leave alone. Count the rows each action affects.
4. **Handle missing values with care.** First ask why they're missing. Don't fill gaps with averages or guesses without saying so; a "missing" flag is often better than a made-up value.
5. **Treat outliers as questions.** Check whether an extreme value is an error or real before changing it. Never remove rows just because they're inconvenient.
6. **Order the steps.** Fix types and formats before removing duplicates, since "J Smith " and "j smith" only match once tidied. Remove duplicates before calculating anything.
7. **Define the checks.** Row counts and key totals before and after cleaning, and rules the clean data must pass, such as unique IDs, valid dates and values within range.

## Output

```
**Dataset**: <name, rows, columns, what one row represents>
**Purpose**: <what the clean data is for>

| Step | Column(s) | Issue | Decision | Reason | Rows affected | Reversible? |
|---|---|---|---|---|---|---|
| 1 | <column> | <issue> | <action> | <why> | <n> | Yes / No |

**Checks after cleaning**: <rule, and its result>
**Open questions**: <decisions that need the data owner, marked "to confirm">
```

## Rules

- Never overwrite the raw data. Work on a copy, and keep the steps as a script or log so they can be run again.
- Never invent values to fill gaps, or results for checks you haven't run. Write "not yet checked" instead.
- Record every decision, including a decision to leave something as it is.
- Don't merge categories on a guess. If it's unclear whether two labels mean the same thing, ask the data owner.
- Keep units, currencies and time zones explicit after cleaning, in column names or notes.
- If cleaning changes a figure people already report, flag the change before new numbers go out.
- Minimise personal data. Drop columns you don't need, and pseudonymise identifiers where you can. Follow the data protection laws that apply to the data.
- Flag issues that point to a problem upstream, such as a broken form or import, so it can be fixed at the source.

## Checklist before you finish

- [ ] Every issue found has a decision, a reason and a row count.
- [ ] The raw data is untouched, and the steps can be re-run.
- [ ] Checks before and after cleaning are defined.
- [ ] Questions for the data owner are listed.
