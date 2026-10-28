---
name: product-metrics
description: Defines a product's metrics, from a north-star metric and its input metrics to guardrails, each with a precise definition. Use when setting goals, building a dashboard or agreeing what success means.
---

# Product metrics

Build a small set of metrics that shows whether the product delivers value, which levers move that value, and what mustn't break along the way. Define each one so precisely that two analysts would get the same number.

## Before you start

Ask in one short message for anything missing: what the product does and for whom, how it makes money, the moment customers get value, and what data is tracked. If tracking is unknown, define the metrics anyway and list the data each one needs.

## Steps

1. **Find the value moment.** Name the action that shows a customer got real value, such as a report shared, an order delivered or a lesson finished, not merely a visit.
2. **Choose the north-star metric.** One metric that counts that value across customers over a set period, and that tends to come before lasting revenue. Good: "accounts that share at least one report each week". Weak: page views, sign-ups, or revenue on its own.
3. **Break it into inputs.** Three to five input metrics that teams can influence and that together drive the north star, such as new accounts reaching first value, frequency of use and retention.
4. **Add guardrails.** Metrics that mustn't get worse while the team pushes the north star: quality, complaints, refunds, speed, unsubscribes.
5. **Define each metric precisely.** The formula, unit, time window and time zone, who counts and who doesn't (such as test and internal accounts), the data source, and how edge cases are handled.
6. **Test the set.** Could a team raise a number without helping customers? If so, tighten the definition or add a guardrail.
7. **Name an owner for each metric.** One role keeps the definition up to date and explains changes in the number.

## Output

```
**North star**: <name>, because <link to customer value and to the business>

| Metric | Type | Definition | Formula | Window | Source | Exclusions | Owner |
|---|---|---|---|---|---|---|---|
| <name> | North star / Input / Guardrail | <in words> | <numerator ÷ denominator> | <weekly> | <table or event> | <test accounts> | <role> |

**How the inputs drive the north star**: <short explanation>
**Data gaps**: <events or fields not tracked yet>
```

## Example

Vague: "Active users."
Precise: "Weekly active accounts: accounts, excluding test and internal ones, that shared at least one report in the seven days ending Sunday, in the company's reporting time zone."

## Rules

- Never invent baselines, targets or benchmarks. Mark them "to confirm" until there's data.
- Prefer rates, and counts of customers doing valuable things, to totals that can only grow, such as all-time sign-ups.
- A metric moving doesn't prove a feature caused it. Say what evidence, such as a controlled experiment, would show cause.
- Define "active" explicitly every time. It causes more disagreement than any other word in metrics.
- Keep the set small. Every extra metric competes for the team's attention.
- Track behaviour in aggregate where you can, and follow the privacy and data protection laws that apply to tracking.

## Checklist before you finish

- [ ] The north star reflects customer value, not just activity.
- [ ] Every metric has a formula, window, source and exclusions.
- [ ] Guardrails cover the ways the north star could be gamed.
- [ ] Each metric has one named owner.
