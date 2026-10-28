---
name: feedback-themes
description: Groups customer feedback into themes with counts, sentiment and representative quotes. Use when making sense of survey comments, reviews, tickets or interview notes, or deciding what to fix first.
---

# Feedback themes

Turn a pile of comments into a short list of themes a team can act on, each with a count and the customers' own words as evidence.

## Before you start

You need the feedback, with an ID for each item and, where available, its date, source and customer segment. Ask in one short message what decision the analysis supports, as that shapes the themes. If you only get part of the feedback, say the results cover that part only.

## Steps

1. **Read before you sort.** Read a good spread of items, around 30 to 50, before naming any theme, so the themes come from the data rather than from expectations.
2. **Draft the themes.** Name each one as a specific statement, such as "Checkout asks for the address twice", not "Checkout". Aim for 5 to 12 themes, and split any that mix different problems.
3. **Code every item.** Tag each item with one or more themes, and as positive, negative or mixed. Put items that fit nothing under "Other", then review them: if several share an idea, that's a new theme. For thousands of items, code a random sample, and give its size.
4. **Count.** For each theme, count the items that mention it and give its share of all items coded. An item can carry several themes, so shares can add up to more than 100%. Say so.
5. **Choose quotes.** Pick two or three short quotes per theme that are typical of it, not the most extreme. Copy them exactly.
6. **Look for patterns.** Compare themes by segment, source or month, if the data allows. Call out a theme that is small but serious, such as a billing error or a safety issue.
7. **Suggest next steps.** For the top themes, say what to look into or fix, and which team might own it. Keep suggestions separate from the findings.

## Output

```
**Based on**: <number> items from <sources>, <date range>

| Theme | Items | Share | Sentiment | Example quotes |
|---|---|---|---|---|
| <Specific theme> | <n> | <%> | <mostly negative> | "<quote>" (ID) |

**What stands out**: <three to five sentences>
**Caveats**: <who is and isn't represented, sample size>
**Suggested next steps**: <what to look into or fix>
```

## Rules

- Never invent, merge or tidy up quotes. Copy them word for word, and shorten only with "...". If no item says it, it isn't a quote.
- Never invent counts. If you coded a sample, say so, and don't present sample counts as totals.
- Remove names, contact details and anything else that could identify a customer from the quotes. Follow the privacy laws that apply.
- Count items, not people, unless you can remove repeat feedback from the same customer. Say which you counted.
- Feedback comes from people who chose to give it. Don't present the shares as the views of all customers.
- A large count doesn't make a theme the most important. Note severity and business impact next to the counts.
- Keep your interpretation apart from what customers said. Mark any theme you're unsure of "to confirm".

## Checklist before you finish

- [ ] Every theme is a specific statement, with a count and real quotes.
- [ ] The "Other" pile has been reviewed.
- [ ] No quote can identify a customer.
- [ ] Caveats say who the feedback does and doesn't represent.
