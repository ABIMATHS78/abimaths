---
name: feature-spec-review
description: Reviews a feature spec or PRD for gaps, edge cases, risks and open questions, ranked by severity with suggested fixes. Use when a spec needs checking before it goes to design or engineering.
---

# Feature spec review

Find the problems in a spec while they're cheap to fix: on paper, before anyone builds. Be specific and constructive, so the author knows exactly what to change.

## Before you start

You need the spec, and ideally the goal it serves and who will build it. Ask in one short message what stage it's at (early draft or ready to build) and whether any part worries the author most. Review what you have; don't wait for perfect context.

## Steps

1. **Read it once to understand it.** Sum up the feature in two sentences. If you can't, that's the first finding.
2. **Check the foundations.** Is the problem clear and backed by evidence? Are the goals, non-goals and success metrics defined, with baselines? Does the solution solve the stated problem?
3. **Walk through every flow.** Follow each path step by step: a first-time user, a returning user, and someone with different permissions. Note every step the spec doesn't describe.
4. **Hunt for edge cases.** Empty, loading and error states; limits and very large inputs; slow or lost connections; two people editing at once; time zones, languages and currencies; accessibility; deleted or migrated data.
5. **Assess the risks.** Security, privacy and data protection, legal or policy issues, performance, cost, dependence on other teams, and the effect on support and existing users.
6. **Check it can be built and tested.** Are requirements specific enough to estimate and test? Is there a plan for rollout, rollback and tracking the success metrics?
7. **Rank the findings.** Blocker: must be resolved before building starts. Major: will cause rework or harm users if ignored. Minor: worth fixing, but won't block.

## Output

```
**Summary**: <the feature in two sentences, and how ready the spec is>

| No. | Severity | Section | Finding | Why it matters | Suggested fix or question |
|---|---|---|---|---|---|
| 1 | Blocker | <section> | <specific gap> | <consequence> | <proposal> |

**Open questions**: <question, and who can answer it>
**What works well**: <two or three strengths to keep>
```

## Example

Weak finding: "Error handling is unclear."
Strong finding: "Section 4 doesn't say what happens if payment fails after the order is created. Customers could be charged with no order, or get an order without paying. Suggest defining the retry and cancel behaviour, and the message the customer sees."

## Rules

- Point to the exact section or quote for each finding, so it's easy to find and fix.
- Don't invent requirements, constraints or facts about the system. If you're assuming something, say so and mark it "to confirm".
- Review the spec, not the author. Phrase findings as questions or proposals, not criticism.
- Don't pad the list. A few important findings beat many trivial ones; group small wording issues into one line.
- Flag any collection or use of personal data without a clear purpose, consent where it's needed, or protection that meets the laws that apply.
- Say what the spec does well, so it survives the next draft.

## Checklist before you finish

- [ ] Every finding names a section and a consequence.
- [ ] Blockers are at the top, with a suggested fix or question.
- [ ] Edge cases and risks were checked, not just the main flow.
- [ ] Each open question names who can answer it.
