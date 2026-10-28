---
name: expense-policy-check
description: Checks expense claims against the organisation's own expense policy and lists exceptions with the policy rule each one breaks. Use when reviewing expense reports before approval or reimbursement.
---

# Expense policy check

Give approvers a clean list of what's fine and what needs a question, so they don't have to read every receipt.

## Ask first

If any of this is missing, ask for it in one short message:
- the expense policy, with its limits, rules and approval levels;
- the claims: date, amount, currency, category, description, and whether there's a receipt;
- any pre-approvals or exceptions already granted.

## Steps

1. **Read the policy** and list its checkable rules: spending limits per category, receipt requirements, items it doesn't allow, approval levels, deadlines for claiming, and rules on alcohol, travel class or gifts.
2. **Check each claim** against every rule that applies.
3. **Look for patterns** the rules point to: the same receipt claimed twice, claims split to stay under a limit, weekend or holiday dates on business claims, round numbers with no receipt.
4. **Mark each claim**:
   - **OK**: meets every rule;
   - **Query**: needs information, such as a missing receipt or an unclear purpose;
   - **Exception**: breaks a rule, with the rule quoted.
5. **Summarise** the totals by status for the approver.

## Output

```
| # | Date | Amount | Category | Status | Rule (quoted) | Note |
|---|---|---|---|---|---|---|

Totals: OK <amount> · Query <amount> · Exception <amount>
Questions for the claimant: <list>
```

## Rules

- **Check against the organisation's policy only.** Don't apply rules it doesn't have.
- A flag isn't an accusation. Describe what was found, neutrally, and leave the judgement to the approver.
- Never invent amounts, receipts or approvals; mark gaps "to confirm".
- Keep claimants' personal details to what's needed.
- Tax treatment varies by country: don't give tax advice; refer those questions to the finance team.
