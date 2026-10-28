---
name: nda-triage
description: Sorts incoming NDAs into standard or needs-a-lawyer using the organisation's own playbook, and lists the clauses that decided it. Use when NDAs arrive and need a quick first review.
---

# NDA triage

Give every incoming non-disclosure agreement a quick first sort, so routine ones move fast and unusual ones reach a lawyer. This is a triage against the organisation's playbook, not legal advice.

## Ask first

If any of this is missing, ask for it in one short message:
- the NDA;
- the organisation's NDA playbook: the terms it accepts and its red lines;
- what the discussions are about, and which side will share more information.

If there's no playbook, don't sort anything as "standard": say that a lawyer should set one, and list the points a playbook usually covers.

## Points to check

- mutual or one-way, and whether that fits the discussions;
- what counts as confidential, and the usual exclusions (already public, already known, independently developed, received from someone else);
- how long the duty lasts;
- who the information may be shared with, such as employees and advisers;
- return or destruction of information at the end;
- governing law and courts;
- anything that isn't about confidentiality: non-compete, non-solicitation, exclusivity, intellectual property, payments.

## Steps

1. **Identify** the parties, the type of NDA and its purpose.
2. **Compare each point** with the playbook, quoting the NDA's wording.
3. **Sort it**:
   - **Standard**: every point is within the playbook;
   - **Needs a lawyer**: any red line, or any term the playbook doesn't cover;
   - **Can't tell yet**: something needed for the decision is missing.
4. **Suggest changes** using only the playbook's own fallback wording.

## Output

```
NDA: <parties> · <mutual or one-way> · <purpose>
Result: Standard / Needs a lawyer / Can't tell yet

| Point | The NDA says (quoted) | Playbook | OK? |
|---|---|---|---|

Deciding clauses: <list>
Suggested changes (playbook wording): <list>
To confirm: <list>
```

## Rules

- **Not legal advice.** "Standard" means "within the playbook", not "safe".
- Watch for terms hidden in NDAs that go beyond confidentiality, and always send those to a lawyer.
- Quote clauses exactly, and keep the agreement confidential.
