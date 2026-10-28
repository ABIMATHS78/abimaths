---
name: invoice-reconciliation
description: Matches invoices to payments, and lists unmatched items, part-payments and amounts that disagree, for someone to investigate. Use for supplier or customer reconciliations and statement checks.
---

# Invoice reconciliation

Match what was billed to what was paid, and produce a short list of what doesn't add up.

## Ask first

If any of this is missing, ask for it in one short message:
- the invoices: number, date, party, amount and currency;
- the payments: date, party, amount, currency and reference;
- the period covered, and any statement from the other party;
- how much difference the team can ignore, such as small rounding.

## Steps

1. **Clean the data first**: standardise invoice numbers, party names, dates and amounts. Note duplicates.
2. **Match in passes**, from the most certain to the least:
   - the invoice number appears in the payment reference, and the amount matches;
   - the same party and the exact amount, within a sensible date window;
   - one payment covering several invoices, or several payments for one invoice.
3. **Record how each match was made**, so a person can check the weaker ones.
4. **List the exceptions**:
   - invoices with no payment, with their age;
   - payments with no invoice;
   - amounts that differ, and by how much;
   - possible duplicates.
5. **Reconcile the totals**: invoices minus payments should equal the balance of what's still open. Show the check.

## Output

```
## Matched: <n> items, <amount>
| Invoice | Payment | Amount | Matched by |
|---|---|---|---|

## Exceptions
| Type | Item | Amount | Age | Note |
|---|---|---|---|---|

Check: invoices <a> − payments <b> = open items <c>
```

## Rules

- **Never force a match.** If a match is uncertain, list it as an exception with your best guess, marked "to confirm".
- Never invent invoices, payments or amounts.
- Keep currencies separate unless an exchange rate is given.
- Treat bank and customer data as confidential, and show only what's needed.
