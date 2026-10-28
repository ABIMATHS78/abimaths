---
name: checklist-builder
description: Turns a recurring task into a short checklist that catches the steps people forget, with checks and a sign-off. Use for launches, handovers, audits, opening and closing routines, and safety checks.
---

# Checklist builder

A good checklist isn't a manual. It's the short list of the steps that matter most and get missed, checked at the right moment.

## Ask first

If any of this is missing, ask for it in one short message:
- the task, and when and how often it happens;
- who carries it out, and who checks it;
- what has gone wrong before, or what would be costly to miss;
- any procedure the checklist should follow.

## Steps

1. **Pick the type**:
   - **Do, then confirm**: people work from memory, then pause to confirm the critical items;
   - **Read, then do**: people follow each item in order, for unfamiliar or high-risk tasks.
2. **Choose the pause points**, such as before starting, before an irreversible step, and before finishing.
3. **Keep only the critical items**: the ones that are easy to miss, or costly when missed. Usually five to nine per pause point.
4. **Write each item as a short check**, with the expected state: "Backup completed and restore tested". Not "Think about backups".
5. **Add verification** where it matters: what evidence shows the item is done, and who signs.
6. **Test it** against the last few times the task went wrong. Would the checklist have caught each problem?

## Output

```
# Checklist: <task>
Type: do-then-confirm / read-then-do · Owner: <role> · Version: <date>

## Before starting
- [ ] <check> (evidence: <what shows it's done>)
## Before <the irreversible step>
- [ ] <check>
## Before finishing
- [ ] <check>

Signed off by: <role> · Date:
```

## Rules

- **Short beats complete.** Move detail into the procedure and link to it.
- Never invent requirements. If a safety or legal check is needed, ask who owns it, and mark it "to confirm".
- Use the exact names of systems, documents and equipment.
- Review the checklist after every incident and at least once a year.
