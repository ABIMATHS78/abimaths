---
name: prd-writer
description: Writes a product requirements document covering the problem, users, goals, requirements, scope, success metrics, risks and open questions. Use when planning a new product, feature or major change.
---

# PRD writer

Write a product requirements document (PRD) that explains why something is worth building, what it must do and how the team will know it worked, clearly enough for design and engineering to start.

## Before you start

Ask in one short message for anything missing: the problem and who has it, the evidence for it (research, data, support tickets), the goal, constraints such as deadlines or platforms, and what's out of scope. Work with what you get, and mark the rest "to confirm".

## Steps

1. **Define the problem.** Who has it, when, and what it costs them. Cite the evidence. A good problem statement doesn't mention the solution.
2. **Set goals and non-goals.** Goals are outcomes, not features. Non-goals keep the scope in check: name related things the team is deliberately not doing.
3. **Choose success metrics.** One primary metric with its baseline and target, and guardrail metrics that mustn't get worse. Define each one precisely.
4. **Describe users and scenarios.** The main types of user and the situations they're in, written as short accounts of what they're trying to do.
5. **Write the requirements.** Number each one. Say what the product must do, not how to build it, and give each a priority: must, should or could. Include needs such as speed, accessibility, security, privacy and support for other languages.
6. **Cover the edges.** Empty states, errors, limits, permissions, and what happens to existing users and their data.
7. **List the rest.** Dependencies, risks, assumptions, open questions with an owner each, and the release plan.

## Output

```
# <Feature name>
**Owner**: <name>   **Status**: Draft   **Last updated**: <date>

## Problem
## Goals and non-goals
## Success metrics
| Metric | Definition | Baseline | Target |
## Users and scenarios
## Requirements
| ID | Requirement | Priority | Notes |
## Edge cases
## Dependencies, risks and assumptions
## Open questions
| Question | Owner | Due |
## Release plan
```

## Rules

- Never invent research findings, data, baselines, customer quotes or testimonials. Mark what's missing "to confirm" and add it to the open questions.
- Keep the problem and the solution apart. If the person has a solution in mind, check that it solves the stated problem.
- Write requirements that can be tested. "Fast" can't be tested; "results appear within two seconds for 95% of searches" can.
- Build in privacy and security from the start: what personal data is collected, why, and how it's protected under the laws that apply.
- Keep it as short as the work allows. Point to detailed documents rather than pasting them in.

## Checklist before you finish

- [ ] The problem is backed by evidence and doesn't describe the solution.
- [ ] Every requirement is numbered, prioritised and testable.
- [ ] Success metrics have definitions, baselines and targets, or say "to confirm".
- [ ] Open questions each have an owner.
