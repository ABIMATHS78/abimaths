---
name: cash-flow-forecast
description: Builds a 13-week cash flow forecast from the user's own inputs, week by week, showing receipts, payments, the closing balance and when cash runs short. Use when a business needs to plan its cash.
---

# 13-week cash flow forecast

See, week by week, whether the business will have the cash it needs, early enough to act.

## Ask first

If any of this is missing, ask for it in one short message:
- the opening cash balance, and the date;
- expected receipts: customer payments with their timing, other income, loans or investment;
- expected payments: payroll and its dates, suppliers, rent, loan repayments, taxes, one-off costs;
- payment terms, such as customers paying in 30 days;
- the minimum cash balance the business wants to keep.

## Steps

1. **Set up 13 weekly columns** from the start date.
2. **Place each receipt and payment in its week**, using real due dates and payment terms. Cash moves when money changes hands, not when an invoice is raised.
3. **Calculate each week**: opening balance + receipts − payments = closing balance, which becomes next week's opening balance.
4. **Flag the weeks** where the closing balance falls below the minimum.
5. **Test a downside case**: for example, the largest customers paying two weeks late, or sales 20% lower. Show what that does to the lowest point.
6. **Suggest actions for tight weeks**: chase receivables, agree supplier terms, move discretionary spending, arrange finance early. Leave the choice to the owner.
7. **List the assumptions**, so the forecast can be updated each week against actual figures.

## Output

```
| Week starting | Opening | Receipts | Payments | Closing | Below minimum? |
|---|---|---|---|---|---|

Lowest point: <amount> in week <n>
Downside case: lowest point <amount> in week <n>
Assumptions: <list>
To confirm: <list>
```

## Rules

- **Use only the user's figures.** Never invent amounts or dates; mark missing ones "to confirm".
- Label every estimate as an estimate.
- Keep one currency, or show the exchange rate used.
- This is a planning tool, not financial, tax or lending advice.
