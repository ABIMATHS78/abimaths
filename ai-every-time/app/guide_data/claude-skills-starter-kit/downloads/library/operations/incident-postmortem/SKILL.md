---
name: incident-postmortem
description: Writes a blameless post-incident review with the impact, a timeline, the causes and actions with owners. Use after an outage, a serious mistake or a near miss, once the problem is under control.
---

# Incident postmortem

Learn from what went wrong without blaming people, so the same thing is less likely to happen again.

## Ask first

If any of this is missing, ask for it in one short message:
- what happened, from the people involved: notes, chat logs, alerts, tickets;
- when it started, when it was noticed, and when it was resolved;
- who and what was affected, and how badly;
- the organisation's postmortem template, if it has one.

## Steps

1. **Summarise the impact**: who was affected, for how long, and in what way. Use numbers where you have them.
2. **Build the timeline** from the records, with times and their time zone: first sign, detection, response, the steps taken, resolution.
3. **Find the causes.** Ask "why?" until you reach the conditions that let the problem happen: missing checks, unclear ownership, alerts that didn't fire, risky defaults. There's usually more than one cause.
4. **Note what went well**, such as fast detection or clear communication, so it's kept.
5. **Agree actions**: specific, with an owner and a due date, each tied to a cause. Prefer fixes that change systems over reminders to "be careful".
6. **Note what's still unknown**, and how it will be found out.

## Output

```
# Postmortem: <title>, <date>
Status: draft / reviewed · Severity: <level>

## Summary
## Impact
## Timeline (<time zone>)
| Time | What happened |
|---|---|
## Causes
## What went well
## What could have gone better
## Actions
| Action | Cause it addresses | Owner | Due |
|---|---|---|---|
## Open questions
```

## Rules

- **Blameless**: describe actions and conditions, not people's faults. Use roles rather than names unless the team agrees otherwise.
- Never invent times, figures or causes. Mark unknowns "to confirm".
- Keep personal data and security details out of a widely shared version.
- If the incident has legal, regulatory or safety reporting duties, flag them for the right owner.
