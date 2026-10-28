---
name: project-status-report
description: Turns task updates and plans into a red, amber or green status report with progress, risks, decisions needed and next steps. Use for weekly project reporting to sponsors or stakeholders.
---

# Project status report

Tell the project's sponsors where things really stand, in a format they can read in two minutes.

## Ask first

If any of this is missing, ask for it in one short message:
- the project plan: milestones, dates and budget;
- this period's updates: tasks done, tasks late, issues raised;
- the definitions the organisation uses for red, amber and green, if it has them;
- who the report is for.

If there are no agreed definitions, suggest these and ask the user to confirm them: **green** means on track; **amber** means at risk, but the team can recover; **red** means the plan won't be met without a decision or help.

## Steps

1. **Set an overall status** and give one sentence on why.
2. **Report each milestone**: planned date, forecast date, and status.
3. **Report the budget**: spent so far, forecast at completion, and the variance.
4. **List risks and issues**: each with its impact, the action being taken and its owner.
5. **List the decisions needed**: what, from whom, and by when.
6. **Show the next period's plan**: the three to five most important things.
7. **Make the colours honest.** If a milestone has slipped, it isn't green.

## Output

```
# <Project>: status, <date>
Overall: 🟢/🟠/🔴 <one-sentence reason>

| Milestone | Planned | Forecast | Status |
|---|---|---|---|

Budget: spent <x> of <y> · forecast <z> · variance <v>

## Risks and issues
| Item | Impact | Action | Owner |
|---|---|---|---|
## Decisions needed
## Next period
```

## Rules

- **Never invent progress, dates or figures.** If an update is missing, say so.
- Report problems early and plainly.
- Keep the same format every period, so changes are easy to spot.
- Write for people outside the project: no unexplained jargon or task numbers.
