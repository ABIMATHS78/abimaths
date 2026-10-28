---
name: user-story-writer
description: Breaks a feature into small user stories with testable acceptance criteria, in a sensible delivery order. Use when building a backlog, planning a sprint or turning a spec into work for a team.
---

# User story writer

Split a feature into stories a team can build, test and release one at a time, each useful on its own.

## Before you start

You need a description of the feature, the spec or PRD if there is one, who the users are, and any business rules. If the users or the purpose of the feature are unclear, ask in one short message: stories without them turn into task lists.

## Steps

1. **Map the journey.** List the steps a user takes through the feature, from start to finish. This shows what a complete first version needs.
2. **Slice it thinly.** Each story should cut through every layer it needs (screen, logic, data) to give the user something that works. Split by step in the journey, business rule, type of data or type of user, never by technical layer.
3. **Write each story.** "As a [type of user], I want [action], so that [benefit]." Name real types of user, not just "the user", and a benefit they would recognise.
4. **Write acceptance criteria.** Three to six per story, in "Given, when, then" form. Cover the main path, the key edge cases and the errors. A tester must be able to mark each one pass or fail.
5. **Check each story.** It should be independent, negotiable, valuable, estimable, small and testable (INVEST). Split any story too big to finish within one sprint.
6. **Order the stories.** The smallest useful slice first, then the stories that add the most value or remove the most risk.

## Output

```
### Story <number>: <short title>
As a <type of user>, I want <action>, so that <benefit>.

**Acceptance criteria**
1. Given <context>, when <action>, then <result anyone can observe>.
2. Given <context>, when <error case>, then <message or behaviour>.

**Notes**: <business rules, designs, dependencies>
**Open questions**: <anything to confirm>
```

Finish with the suggested order and a one-line reason for it.

## Example

**Story 1: Reset a forgotten password**
As a registered customer, I want to reset my password from the sign-in page, so that I can get back into my account without contacting support.
1. Given I enter a registered email address, when I ask for a reset, then I receive a link that expires after the time set in the security policy.
2. Given I enter an unregistered address, when I ask for a reset, then I see the same message as in criterion 1, so the page doesn't reveal who has an account.

## Rules

- Never invent business rules, limits or figures. Write "to confirm" and add them to the open questions.
- Describe what the user needs, not how to build it, unless the "how" is a real constraint.
- Include accessibility, security and privacy criteria where they apply, including what the laws that apply to the product require.
- Keep internal technical work as tasks under a story, not as stories of its own.
- Don't let a story depend on one later in the order.

## Checklist before you finish

- [ ] Every story gives a user something that works on its own.
- [ ] Every acceptance criterion can be marked pass or fail.
- [ ] Unknown rules are listed as open questions, not guessed.
