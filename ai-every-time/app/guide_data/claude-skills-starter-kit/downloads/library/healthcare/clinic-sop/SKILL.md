---
name: clinic-sop
description: Writes standard operating procedures for a clinic's non-clinical work, such as appointments, check-in and records requests. Use when a clinic needs a clear, step-by-step procedure.
---

# Clinic SOP

Write a procedure that any trained member of the reception or admin team could follow on a busy day and get right every time.

## Ask first

If any of this is missing, ask for it in one short message:
- the task, and where it starts and ends (for example, from a phone call to a booked appointment);
- who does it, and who covers when they're away;
- the systems used, such as the booking system or records software;
- how it's done now, and the problems the SOP should fix;
- the clinic's policies it must follow, such as privacy, complaints or urgent care.

## Steps

1. **Set the scope.** Say what the procedure covers and what it doesn't. Clinical judgement is always outside it.
2. **Map the steps in order**, from what starts the task to when it's finished. One action per step, starting with a verb, written for the person doing it.
3. **Add the decision points.** For each "if", say what to do and who to pass it to: a clinician, the practice manager or the privacy lead.
4. **Build in the privacy checks**: confirm identity before sharing anything, speak where others can't overhear, share only what's needed, and log every request.
5. **Write the clinical hand-off.** If a caller or visitor describes symptoms or seems unwell, staff don't assess them. They follow the clinic's urgent procedure and pass the person to a clinician straight away. Use the clinic's own wording, or mark it "to confirm".
6. **Add the records**: what to log, where, and for how long, as the clinic's policy says.
7. **Test it** on a normal case and two awkward ones, such as a caller who can't prove who they are, or a patient who arrives late.

## Output

```
# SOP: <task>
Version: <x>   Owner: <role>   Review date: <to confirm>
Approved by: <practice manager, and a clinician for the hand-off; to confirm>
Purpose: <one sentence>
Scope: covers <…>; doesn't cover <…>
Roles: <who does what>

## Steps
1. <action, starting with a verb>

## If… then…
| Situation | What to do | Who to contact |
|---|---|---|

## Privacy points
## Records to keep
```

## Rules

- Non-clinical only. Never write steps that ask staff to judge symptoms, give health advice or decide on their own how urgent something is.
- Never invent legal deadlines, retention periods, fees or identity checks for records requests. They vary by country and organisation, so mark them "to confirm" with the clinic's privacy or data protection lead.
- Follow the data protection and health-records laws that apply where the clinic works, as well as its own policies.
- Use roles, not names, so the SOP still works when staff change.
- Keep each SOP to one task and one or two pages.
