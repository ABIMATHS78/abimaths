---
name: ticket-triage
description: Sorts support tickets by topic, urgency and sentiment, tags each one and suggests who should handle it, with a short reason. Use when a ticket queue needs sorting, routing or prioritising.
---

# Ticket triage

Turn a mixed queue into an ordered list, so the most urgent tickets reach the right people first. Every tag should be one a colleague would agree with after reading the ticket.

## Before you start

If anything below is missing, ask for it in one short message:
- the tickets, with any fields you have, such as ID, channel, plan and time received;
- the team's topic list and urgency levels, if they have them;
- who handles what, by team or role.

If there's no topic list or urgency scale, propose one from the tickets, or use the levels below, and say so.

## Steps

1. **Read each ticket in full.** Note what the customer is trying to do, what's going wrong and how it affects them. The subject line is often misleading.
2. **Tag the topic.** One main topic from the team's list, and a second only if the ticket truly covers two. Use "Other" only when nothing fits; if several land there, suggest a new topic.
3. **Set the urgency on impact and time, not tone:**
   - **Urgent**: service down, security or data exposure, payments failing, a safety risk, or a deadline within hours.
   - **High**: an important task is blocked for this customer, with no workaround.
   - **Normal**: something is broken but a workaround exists, or a how-to question.
   - **Low**: feedback, feature requests and general questions.
4. **Read the sentiment.** Positive, neutral, frustrated or angry. Add "at risk" when the customer mentions cancelling, a formal complaint, a public review or legal action.
5. **Suggest an owner.** The team or role best placed to resolve it in one hand-off, with a one-line reason.
6. **Order the queue.** Urgent first, then high, then by time waiting or by service-level deadline, if there is one. Group tickets that describe the same fault, as they may be one incident.

## Output

```
| ID | Topic | Urgency | Sentiment | Suggested owner | Reason | Confidence |
|---|---|---|---|---|---|---|
| <ID> | <Topic> | <Level> | <Sentiment, plus "at risk"> | <Team or role> | <One line> | High / Medium / Low |
```

Then, in a few lines: possible incidents (several tickets, same fault), tickets you couldn't classify and why, and any topic that's growing.

## Rules

- Judge urgency on impact, not on capital letters or exclamation marks. A polite ticket can be the most urgent one in the queue.
- Send security, data exposure, legal and safety issues to the right owner at once, whatever the queue order.
- Never guess a customer's age, gender, health, ethnicity or other personal traits, and never use them to set priority.
- Refer to tickets by ID. Don't copy personal data into tags, reasons or summaries.
- If a ticket is unclear, set confidence to "Low" and say what's missing, rather than forcing a tag.
- Don't invent facts about a customer's account, plan or history. If a field is missing, write "to confirm".
- Follow the data protection laws that apply to the business and its customers.

## Checklist before you finish

- [ ] Every ticket has a topic, urgency, sentiment, owner and reason.
- [ ] Urgent tickets sit at the top, and safety or security issues are called out.
- [ ] Possible incidents are grouped and named.
