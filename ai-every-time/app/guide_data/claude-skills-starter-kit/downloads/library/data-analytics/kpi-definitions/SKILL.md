---
name: kpi-definitions
description: Defines each KPI once, with its formula, data source, owner, filters and edge cases, so every team counts it the same way. Use when building a metrics glossary or settling a dispute over a number.
---

# KPI definitions

Write one definition per KPI that anyone can follow to get the same number from the same data. When two reports disagree, the definition should settle it.

## Before you start

Ask in one short message for anything missing: the list of KPIs, how each one is calculated now (a query, a spreadsheet formula or a description), where the data lives, and who owns each number. If teams calculate a KPI in different ways, ask for every version.

## Steps

1. **Collect what exists.** Gather the definitions, queries and reports in use for each KPI. Note every version and where the numbers differ.
2. **Agree the purpose.** In one sentence per KPI, say what question it answers and what decision it supports. This settles most arguments about the formula.
3. **Write the formula exactly.** Numerator and denominator, the unit, the grain (per customer, per order, per day) and the time window. Say how to combine periods or groups: a ratio of totals is not the same as an average of ratios.
4. **Name the data source.** The system or table, the fields used, the filters applied and how often the data refreshes.
5. **Settle the edge cases.** Refunds and cancellations, test and internal accounts, duplicates, missing values, currency conversion, time zones, data that arrives late, and changes to the definition over time.
6. **Assign an owner.** One person or role who maintains the definition and approves changes to it.
7. **Check it against real data.** Calculate the KPI from the written definition and compare it with existing reports. Explain every difference.

## Output

```
### <KPI name>
**Purpose**: <the question it answers>
**Formula**: <numerator> ÷ <denominator>, in <unit>
**Grain and window**: <per what, over what period, in which time zone>
**Data source**: <system, table, fields>, refreshed <how often>
**Includes and excludes**: <filters>
**Edge cases**: <each one, and how it's handled>
**Owner**: <role or name>
**Not to be confused with**: <similar KPIs, and how they differ>
**Version**: <number, date, what changed>
```

## Rules

- One name, one definition. If two teams need different versions, give them different names.
- Never invent a formula, source, owner or figure. If you don't know, write "to confirm" and list it.
- Never change a definition silently. Record the change and its date, and note that numbers before and after it can't be compared directly.
- Write every definition in plain words as well as a formula, so people who don't work with data can follow it.
- Note any KPI built on personal or sensitive data, and follow the data protection laws that apply.

## Checklist before you finish

- [ ] Each KPI has a purpose, exact formula, source, edge cases and owner.
- [ ] Every version in use has been compared with the agreed definition.
- [ ] Open points are marked "to confirm".
