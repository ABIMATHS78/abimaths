---
name: dashboard-commentary
description: Explains what changed on a dashboard, the likely reasons and the caveats, in plain language for busy readers. Use when writing weekly or monthly reports, metric reviews or performance updates.
---

# Dashboard commentary

Tell readers what moved, why it probably moved and how sure we are, in a few sentences they can read in a minute.

## Before you start

You need the dashboard data or screenshots for the period and the comparison period, the metric definitions, and who will read the commentary. Context helps a lot: launches, campaigns, price changes, outages, holidays and known data problems. Ask for it in one short message; without it, every reason stays a guess.

## Steps

1. **Pick the comparison.** Compare with the previous period, and with the same period a year earlier where seasons matter, and with the target if there is one. Keep definitions and time zones the same throughout.
2. **Find what changed.** List the metrics that moved more than they usually do. Don't comment on every small wobble; say what normal variation looks like.
3. **Check the data first.** Before explaining a change, rule out tracking breaks, partial data for the most recent days, definition changes and duplicated records.
4. **Break it down.** Split the change by segment, channel, product or region to see where it comes from. Watch for shifts in the mix: an overall rate can fall even when every segment improves, if more of the volume moves to segments with lower rates.
5. **Suggest reasons.** Link changes to events in the context, and say how strong each link is. Without an experiment or other clear evidence, a reason is a likely explanation, not a proven cause.
6. **Say what to do next.** One or two actions or questions to follow up, with an owner if known.

## Output

```
**Headline**: <the most important change, with numbers, in one sentence>

**What changed**
- <Metric>: <value> vs <comparison> (<change>), mostly from <segment>.

**Likely reasons**
- <Reason>, because <evidence>. Confidence: <high / medium / low>.

**Caveats**: <data issues, small numbers, open questions>
**Next steps**: <action, owner if known>
```

## Example

Weak: "Conversion dropped this week."
Strong: "Conversion fell from 3.2% to 2.8%, down 0.4 points. Almost all of the fall was on mobile, where a checkout error ran from Tuesday to Thursday. Desktop was flat."

## Rules

- Never invent a cause, event or number. If you don't know why something moved, say so, mark it "to confirm" and suggest how to find out.
- Keep percentage changes and percentage-point changes apart. A rate that goes from 10% to 12% is up 2 points, or 20%.
- Be careful with small numbers. A rise from 4 to 8 is +100%, but may be noise.
- Don't call a change "significant" unless it was tested statistically. Say "larger than usual" instead.
- Give the absolute and the relative change where both help.
- Report on groups, not individuals. Keep personal data out of the commentary, and follow the data protection laws that apply.

## Checklist before you finish

- [ ] The headline states the biggest change, with numbers.
- [ ] Data problems were ruled out before reasons were given.
- [ ] Each reason has evidence and a confidence level.
- [ ] Caveats are stated plainly.
