---
name: escalation-summary
description: Summarises a long support ticket thread for escalation, stating the issue, its impact, what was tried and the exact ask. Use when handing a ticket to engineering, a manager or another team.
---

# Escalation summary

Give the person receiving an escalation everything they need in one minute's reading, so they can act without opening the whole thread. Lead with the ask.

## Before you start

You need the full ticket thread, including internal notes. Useful extras: the customer's plan or account size, the product version or device, logs or error messages, and who the escalation is going to. If you don't know who it's for or what the agent needs from them, ask in one short message.

## Steps

1. **Read the whole thread in order.** Note when the problem started, each reply, each fix tried and its result. Watch for the problem changing along the way.
2. **State the issue in one or two sentences.** What's wrong, for whom, and since when. Use the product's own terms, and copy error messages exactly.
3. **Describe the impact.** How many customers or orders are affected, what they can't do, money or deadlines at stake, and how the customer feels. Use only figures the thread or the agent provides.
4. **List what was tried.** Each step, who did it and what happened. This stops the next team repeating work.
5. **Write the ask.** One clear request: a fix, a decision, an exception or an answer. Say who should act, by when, and why that date matters.
6. **Gather the evidence.** Ticket IDs, screenshots, logs and steps to reproduce, so nobody has to dig through the thread.
7. **Check it stands alone.** Could someone who has never seen the ticket act on it? If not, add what's missing.

## Output

```
**Ask**: <what you need, from whom, by when>
**Urgency**: <Urgent / High / Normal>, because <reason>

**Issue**: <one or two sentences>
**Impact**: <who and how many are affected, what they can't do, money or deadlines at stake>
**Timeline**: <when it started, and key events with dates>

**What we've tried**
1. <Step, who did it>: <result>

**Steps to reproduce**: <if known, otherwise "not yet reproduced">
**Customer's position**: <what they expect, in neutral words>
**Evidence**: <ticket IDs, logs, screenshots>
**Open questions**: <what's still unknown>
```

## Rules

- Be factual and neutral. Describe what happened, not who was to blame. Label any opinion as the agent's view.
- Never invent impact figures, dates, results or customer quotes. If something isn't in the thread, write "to confirm" or "unknown".
- Quote the customer only when their exact words matter, and keep the quote short.
- Include only the personal data the receiving team needs. Refer to the account by ID rather than by name where you can.
- Don't leave out awkward facts, such as an earlier wrong answer from support. The next team needs the full picture.
- Follow the business's escalation policy and the data protection laws that apply.

## Checklist before you finish

- [ ] The ask is in the first line, with an owner and a date.
- [ ] Impact uses real figures, or says "to confirm".
- [ ] Everything tried is listed with its result.
