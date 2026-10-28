---
name: process-map
description: Maps a process step by step with owners, handoffs, waiting times and problems, then suggests improvements with their trade-offs. Use when a process is slow, error-prone or poorly understood.
---

# Process map

See the process as it really runs, not as it's supposed to run, then find where time and quality are lost.

## Ask first

If any of this is missing, ask for it in one short message:
- the process and its start and end points, such as "order received" to "order delivered";
- who takes part, and the systems they use;
- what's going wrong: delays, errors, complaints, cost;
- any data on volumes, times or error rates.

## Steps

1. **List every step in order**, with who does it and in which system.
2. **Mark the handoffs** between people or teams. Handoffs are where work waits and information gets lost.
3. **Add the times** where known: how long each step takes, and how long work waits before it. Mark guesses as estimates.
4. **Mark decision points and loops**: approvals, rework, back-and-forth.
5. **Find the problems**: the longest waits, steps that add no value for the customer, duplicate data entry, unclear ownership, frequent errors.
6. **Suggest improvements**, each with its expected benefit, its cost or effort, and its risk. Start with simple fixes.

## Output

```
## <Process>: from <start> to <end>

| # | Step | Who | System | Time | Wait before | Notes |
|---|---|---|---|---|---|---|

Text diagram:
<Start> → <step> → [decision?] → <step> → <End>

## Problems found
## Improvements
| Idea | Benefit | Effort | Risk | Owner |
|---|---|---|---|---|
## To confirm
```

## Rules

- **Map the current reality first**; ideas come after.
- Never invent times, volumes or error rates. Label estimates, and mark unknowns "to confirm".
- Describe problems as process problems, not as people's faults.
- Check improvement ideas with the people who do the work before recommending them.
