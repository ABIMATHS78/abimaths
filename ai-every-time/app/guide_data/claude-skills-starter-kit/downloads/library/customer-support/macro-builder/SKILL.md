---
name: macro-builder
description: Builds saved-reply templates (macros) for common support questions, with placeholders, notes on when to use each and the internal actions to take. Use when a team answers the same question often.
---

# Macro builder

Build saved replies that let agents answer common questions quickly, without sounding robotic or getting facts wrong. A good macro covers one situation and shows the agent exactly what to fill in.

## Before you start

You need examples of the question, ideally with two or three good replies agents have sent, and the policy or knowledge base article behind the answer. A tone guide helps. If the policy is missing, ask for it in one short message: a macro repeats its facts hundreds of times, so they must be right.

## Steps

1. **Pick the situation.** Group the examples and choose one situation per macro, such as "refund requested within the return window". If the answer depends on the plan, country or order status, write separate macros or clearly marked versions.
2. **Write the reply.** Start from the best real replies. Keep the facts from the policy and the warmth from the agents, and cut the rest.
3. **Add placeholders.** Mark each detail that changes per customer in double curly brackets, or in the syntax the team's help desk uses, with a plain name such as {{first_name}} or {{order_number}}. Use the same names across all macros, and keep to a few per macro.
4. **Leave room for a human line.** Add a {{personal_line}} placeholder where the agent responds to something specific in the customer's message, so no reply feels canned.
5. **Write the agent notes.** When to use the macro, when not to, what to check in the account first, and the internal actions: tags, status, follow-up.
6. **Test it.** Fill the placeholders with sample values and read the result aloud. Then check what happens if one is left empty, and warn agents if the reply would break.

## Output

```
**Macro**: <short name agents would search for>
**Use when**: <the situation>
**Don't use when**: <similar situations that need a different reply>
**Check first**: <what to confirm in the account or order>

**Reply**
Hi {{first_name}},

{{personal_line}}

<The answer, with numbered steps if the customer must do something>

<Sign-off in the brand's style>

**Internal actions**: <tags, status, follow-up>
**Placeholders**: <each one, and where the agent finds its value>
**Source**: <the policy or article this relies on>
```

## Rules

- Never put a fact in a macro that isn't in the policy or knowledge base. Mark anything unconfirmed "to confirm", and don't publish the macro until it's checked.
- Never promise a refund, credit or timeline the policy doesn't guarantee, and never add customer quotes or testimonials.
- Keep personal data, internal links and anything secret out of the macro text.
- Don't build macros for safety concerns, legal threats or data breaches. Those need a reply written by a person.
- Name the source policy, so the macro can be reviewed whenever that policy changes.
- Follow the consumer and privacy laws that apply to the business and its customers.

## Checklist before you finish

- [ ] Each macro covers one situation, with clear "use when" and "don't use when" notes.
- [ ] Every placeholder is listed, with where to find its value.
- [ ] The reply reads naturally with sample values filled in.
