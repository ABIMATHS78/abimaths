---
name: bug-fix-plan
description: Works through a bug from the error, the steps that cause it and the code, finds the root cause, and proposes the smallest safe fix with a test. Use when something in an app breaks or behaves oddly.
---

# Bug fix plan

Fix the cause, not the symptom, and make sure the bug doesn't come back.

## Before you start

Ask for: what happened, and what should have happened; the exact error message and any logs; the steps that make it happen; when it started, and what changed around then; and the code involved. Ask in one short message.

## Steps

1. **Reproduce it.** Write down the exact steps, inputs and set-up that cause the bug. If it can't be reproduced, collect more evidence before changing any code.
2. **Read the error carefully**: the message, the file and line, and the full trace. The first error is usually the real one.
3. **Narrow it down.** Check what changed recently, and find the smallest input that still fails. Add logging around the suspect code if needed.
4. **Form a hypothesis, and test it**, such as "the date is empty because…". Check whether it's true before fixing anything.
5. **Find the root cause.** Keep asking why until you reach the real cause, not just the line that crashed.
6. **Write a test that fails** because of the bug, before fixing it.
7. **Make the smallest fix** that deals with the root cause, then run the new test and the existing tests.
8. **Look for the same bug elsewhere**: similar code with the same mistake.

## Output

```
Symptom: <what happened>
Reproduce: <steps>
Root cause: <what's really wrong, and why>
Fix: <the change, with code>
Test: <the test that fails before the fix and passes after it>
Also check: <similar places>
```

## Rules

- **Don't guess.** Call a hypothesis a hypothesis until the evidence confirms it.
- Never hide an error by catching and ignoring it, deleting a failing test, or turning off a check.
- Before running anything that changes data, such as a migration or a delete, back up the data and confirm with the person.
- Don't paste secrets or customers' personal data from logs into the conversation. Remove them first.

## Checklist before you finish

- [ ] The bug was reproduced, or the evidence is listed.
- [ ] The root cause is explained, not just the symptom.
- [ ] A test fails before the fix and passes after it.
- [ ] Similar code was checked for the same bug.
