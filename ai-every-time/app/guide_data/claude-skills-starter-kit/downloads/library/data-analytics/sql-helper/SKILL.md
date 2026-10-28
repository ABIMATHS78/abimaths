---
name: sql-helper
description: Writes and explains SQL queries, checking joins, filters, nulls and duplicates so the results can be trusted. Use when someone needs a query written, fixed, reviewed or explained.
---

# SQL helper

Write SQL that answers the right question and returns the right rows, and explain it so the person can check it and reuse it.

## Before you start

You need the question in plain words, the tables with their columns and keys (or the schema), and the database in use, since syntax varies between them. Ask for any of these in one short message if they're missing. If you have to assume a column's name or meaning, say so.

## Steps

1. **Restate the question.** Say what one row of the result represents and which columns it needs. Define any metric in words before writing code.
2. **Find each table's grain.** What does one row mean, and which columns are unique? Check with COUNT(*) against COUNT(DISTINCT key).
3. **Write it in readable steps.** Use WITH clauses for each step, clear aliases, and a comment for every business rule. Name the columns you need instead of using SELECT *.
4. **Check the joins.** Is the key unique on the side that should be unique? A one-to-many join repeats rows and inflates totals. Count rows before and after each join; an unexpected jump means duplication.
5. **Check the filters.** A WHERE condition on the right-hand table of a LEFT JOIN drops the unmatched rows, unless it tests for null. To keep them, move the condition into the ON clause. For timestamps, use >= the start and < the day after the end: BETWEEN with a date as its upper limit stops at midnight, missing the rest of that day.
6. **Check nulls.** COUNT(column) skips nulls, while COUNT(*) counts every row. Averages ignore nulls rather than treating them as zero. "= NULL" is never true; use IS NULL. NOT IN returns no rows if its list contains a null; use NOT EXISTS instead.
7. **Explain and test.** Walk through each part in plain words. Suggest one check, such as a total that should match a known figure.

## Output

```sql
-- Question: <in plain words>
-- One row = <the grain of the result>
WITH step_one AS (
  -- <what this step does>
  SELECT ...
)
SELECT ...
```

Then: what each step does, the assumptions made, and a quick check to run.

## Rules

- Never invent tables, columns or results. If you haven't seen the schema, mark names "to confirm" and say what you expect them to hold.
- Write read-only queries unless asked otherwise. Before any UPDATE, DELETE or DROP, warn clearly, test the same WHERE clause with a SELECT, and get the person's confirmation.
- When a query takes user input, use parameters rather than pasting text into the SQL, to prevent SQL injection.
- In some databases, dividing two whole numbers returns a whole number, so 1/2 gives 0. Convert to a decimal first when calculating rates.
- Compare timestamps in one time zone, and say which one.
- On large tables, filter early and use the columns the table is partitioned or indexed on, so the query doesn't scan everything.
- Select only the personal data the task needs, and follow the data protection laws that apply.

## Checklist before you finish

- [ ] The grain of every table and of the result is stated.
- [ ] Row counts were checked before and after each join.
- [ ] Filters, nulls and date ranges behave as intended.
- [ ] Every assumption is listed for the person to confirm.
