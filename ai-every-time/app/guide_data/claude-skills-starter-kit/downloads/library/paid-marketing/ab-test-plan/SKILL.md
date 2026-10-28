---
name: ab-test-plan
description: Plans an ad or landing-page A/B test with one hypothesis, one primary metric, a sample size and a stopping rule. Use before running a split test on ads or landing pages.
---

# A/B test plan

Plan a test that can give a trustworthy answer: one change, one metric, enough traffic, and a stopping rule fixed before it starts.

## 1. Get what you need

If anything below is missing, ask for it in one short message. Mark smaller gaps "to confirm" and carry on.
- what to test, and the evidence that it might matter;
- the ad or page, and the tool that will run the test;
- the current conversion rate, and weekly visitors or impressions;
- the smallest improvement worth acting on.

## 2. Write the hypothesis

"Because we saw <evidence>, we believe changing <element> for <audience> will raise <metric>." Change one thing, or a bundle you name as one, so the result is clear.

## 3. Choose the metrics

- **Primary**: one metric that decides the result, defined exactly, such as purchases per visitor rather than clicks.
- **Guardrails**: metrics that must not get worse, such as cost per sale, refunds or errors.
- Fix both before the test starts. Don't switch metrics after seeing results.

## 4. Work out the sample size

Use the baseline rate (p1), the rate worth detecting (p2), 5% significance (two-sided) and 80% power, unless agreed otherwise. A standard approximation for each variant:

```
n = (1.96 + 0.84)² × [p1 × (1 - p1) + p2 × (1 - p2)] / (p1 - p2)²
```

For example, to detect a rise from 4% to 5%: n = 7.84 × (0.0384 + 0.0475) / 0.0001 ≈ 6,735 visitors per variant. Divide by weekly traffic per variant for the duration, and round up to whole weeks so every day of the week is covered. If that's too long, say so: test a bolder change, use a busier page, or accept detecting only larger effects.

## 5. Set up and stopping rule

- Split traffic at random, run both versions at the same time, and change nothing else while the test runs.
- On ad platforms, use the built-in experiment tool. Two ads in one ad group aren't a fair test, because delivery favours one of them.
- Check tracking on both versions before starting.
- Run until each variant reaches the sample size and at least one full week has passed, then analyse once. Stopping when a result first looks significant inflates false wins, unless the tool uses a method built for continuous checking. Stop early only if a guardrail breaks.

## 6. Return the plan

```
# A/B test plan: <name>
Hypothesis: <sentence>
A (control): <description> | B (variant): <description>
Primary metric: <definition> | Guardrails: <metric and limit>
Baseline: <rate>, from <source and dates> | Smallest effect worth detecting: <p1> to <p2>
Sample size: <n> per variant | Duration: <weeks>, from <traffic per week>
Stopping rule: <rule>
Decision: if B wins, <action>; if A wins, <action>; if no clear difference, <action>
```

## Rules

- Never invent baselines or traffic. Without them, show the formula and two or three scenarios marked "to confirm".
- Report every result honestly, including "no clear difference".
- Tests must respect the consent visitors gave for cookies and tracking, under the privacy laws where they are.
- Don't test deceptive versions, such as fake scarcity or hidden costs. Consumer-protection rules apply during a test too.
