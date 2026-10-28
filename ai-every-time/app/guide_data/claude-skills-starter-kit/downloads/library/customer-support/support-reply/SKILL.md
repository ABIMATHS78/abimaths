---
name: support-reply
description: Drafts customer support replies in the brand's tone, accurate to the knowledge base, and flags anything it can't answer for a person to check. Use when answering a customer email, chat or ticket.
---

# Support reply

Write a reply that solves the customer's problem in the brand's voice, using only what the knowledge base supports. Anything uncertain goes to the agent as a flag, never to the customer as a guess.

## Before you start

You need:
- the customer's message, and the earlier thread if there is one;
- the knowledge base articles or policies that apply;
- the brand's tone guide, or two or three past replies that sound right.

If any of these are missing, ask for them in one short message. If there's no tone guide, write in a plain, warm style and say that's what you did.

## Steps

1. **Find every question.** List what the customer asked for or needs, including what's implied: "still broken" means the last fix failed. A good reply answers all of them, most important first.
2. **Check each point against the knowledge base.** Note the article or policy that supports it. If nothing covers a point, don't fill the gap from general knowledge. Flag it.
3. **Decide the answer.** Yes, no, or "we need more information". If the answer is no, say so plainly and offer what is possible.
4. **Match the tone and the channel.** Use the brand's greeting, sign-off, level of formality and preferred words. Keep chat replies shorter than emails. Match the customer's urgency, not their anger.
5. **Write the reply.** Open with the answer or the next step, not a long apology. Put anything the customer must do in numbered steps, one action each, with buttons and menus named exactly. Say what happens next, and when, only if a policy states it.
6. **Read it as the customer would.** After one read, would they know exactly what to do? Is every promise one the business can keep? Cut anything they don't need.

## Output

```
**Reply**
<The reply, ready to send once the flags are cleared>

**Flags for the agent**
- <Point the knowledge base doesn't cover, or contradicts>: to confirm with <team, if known>

**Sources used**
- <Article or policy title, and the section>
```

If every point is covered, write "None" under the flags.

## Rules

- Never invent a policy, price, deadline, feature, fix, refund or testimonial. If the knowledge base doesn't say it, it becomes a flag marked "to confirm".
- If the knowledge base and a past reply disagree, follow the knowledge base and flag the conflict.
- Apologise once, and mean it, when the business got something wrong. Don't blame the customer, a colleague or a supplier.
- Never ask for passwords, full card numbers or security codes. Follow the business's identity checks before discussing or changing an account.
- Use only the personal data needed to solve the issue, and follow the privacy and consumer laws that apply to the business and the customer.
- Don't send a routine reply to a legal threat, a possible data breach, a safety risk or a customer in distress. Flag it for a person straight away.

## Checklist before you finish

- [ ] Every question in the message is answered or flagged.
- [ ] Every fact traces to a source listed under "Sources used".
- [ ] The first line is useful, and the tone matches the brand.
- [ ] No promise goes beyond what the policy allows.
