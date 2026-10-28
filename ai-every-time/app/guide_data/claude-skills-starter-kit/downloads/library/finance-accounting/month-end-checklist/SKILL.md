---
name: month-end-checklist
description: Runs a month-end close checklist, tracking each task's owner, deadline, status and evidence, and flags what's blocking the close. Use during the monthly or quarterly close of the books.
---

# Month-end checklist

Close the books on time, with nothing forgotten and every task's evidence in one place.

## Ask first

If any of this is missing, ask for it in one short message:
- the organisation's close checklist, or its usual close tasks;
- the close calendar: which working day each task is due;
- owners, and the status of each task so far;
- the accounting system or systems used.

If there's no checklist, offer a common starting list for the finance team to adapt: bank reconciliations, sales and purchase ledgers, accruals and prepayments, fixed assets and depreciation, payroll, stock, intercompany balances, VAT or sales tax where relevant, the balance sheet review, and management reports.

## Steps

1. **Lay out the tasks by close day**, with owner, due date and dependencies. Some tasks can't start until others finish.
2. **Track status**: not started, in progress, done, or blocked, with the reason.
3. **Record evidence** for each completed task: the reconciliation file, the journal number, the reviewer.
4. **Flag the critical path**: blocked or late tasks that hold up others.
5. **Check the review step**: every reconciliation and journal is prepared by one person and reviewed by another, where the team's size allows.
6. **Summarise the close status** for the finance lead.

## Output

```
# Close: <month year> · Day <n> of <n>
Status: on track / at risk / late

| Day | Task | Owner | Depends on | Status | Evidence | Reviewer |
|---|---|---|---|---|---|---|

Blocking the close: <list>
Late tasks: <list>
```

## Rules

- **Never mark a task done without evidence.**
- Never invent balances, journal numbers or approvals; mark gaps "to confirm".
- Keep the preparer and the reviewer separate where possible.
- This tracks the close; accounting judgements and tax treatment stay with qualified staff.
