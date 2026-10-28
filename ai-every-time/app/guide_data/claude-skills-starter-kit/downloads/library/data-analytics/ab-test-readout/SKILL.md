---
name: ab-test-readout
description: Reads out an A/B test result, checking the split first, then giving the effect with its confidence interval and a decision. Use when an experiment ends and the team needs to know whether to ship.
---

# A/B test readout

Tell the team what the test showed, how sure we can be, and what to do next, without overselling a lucky result.

## Before you start

You need, for each group: the number of users (or whatever unit was randomised), the conversions or the metric's mean and standard deviation, and the planned split, such as 50/50. Also ask for the primary metric chosen before the test started, any guardrail metrics, the start and end dates, and the smallest effect worth acting on. Ask for anything missing in one short message.

## Steps

1. **Check the split.** Compare each group's size with the planned split using a chi-square goodness-of-fit test. If the p-value is below 0.001, the split is broken (a sample ratio mismatch): stop, report it, and look for the cause before reading any results.
2. **Check the setup.** Did the test run for whole weeks, to cover weekly patterns? Was it stopped early because it looked good? Repeatedly checking and stopping at the first good result inflates false positives, unless a sequential method was planned.
3. **Calculate the effect on the primary metric.** For conversion rates pA and pB with group sizes nA and nB:
   - difference = pB − pA, and relative lift = (pB − pA) ÷ pA;
   - standard error = √(pA(1 − pA)/nA + pB(1 − pB)/nB);
   - 95% confidence interval = difference ± 1.96 × standard error.
   For other metrics, use the means, the standard deviations and the same approach. Show the working.
4. **Read the interval, not just the p-value.** If the whole interval is above the smallest effect worth acting on, the result is clear. If it includes zero, the test didn't detect an effect; that isn't proof there is none.
5. **Check the guardrails.** Did anything important get worse, such as refunds, complaints, speed or unsubscribes?
6. **Treat segment results as ideas, not findings.** The more segments you look at, the more will look significant by chance. Only trust a segment that was planned in advance.
7. **Make a recommendation**: ship, don't ship, or run longer, with the reason.

## Output

```
**Decision**: <ship / don't ship / run longer>, because <reason>

| Group | Users | Conversions | Rate |
|---|---|---|---|

Split check: <passed / failed>, p = <value>
Effect: <difference> points (<relative lift>%), 95% CI <low> to <high>
Guardrails: <each metric, and whether it moved>
Caveats: <novelty effects, short duration, segments to retest>
```

## Rules

- Never invent numbers. Show every calculation so it can be checked, and mark missing inputs "to confirm".
- Don't call a result "significant" without saying which test, and at what level.
- Don't switch the primary metric after seeing the results. Report the planned one first.
- Keep percentage points and percentages apart. A rise from 4% to 5% is 1 point, or 25%.
- If users were randomised but the analysis counts sessions or orders, warn that the interval will be too narrow.
- Report on groups only, and keep personal data out of the readout.

## Checklist before you finish

- [ ] The split was checked before the results were read.
- [ ] The effect has a confidence interval, with the working shown.
- [ ] Guardrail metrics were checked.
- [ ] The decision follows from the evidence and says why.
