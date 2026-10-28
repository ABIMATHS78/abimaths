---
name: fact-check
description: Checks every factual claim in a draft and marks it supported, unsupported or wrong, with the source to confirm it or a suggested fix. Use before publishing an article, report, speech or post.
---

# Fact check

Catch the wrong numbers, misquotes and unsupported claims before readers do.

## Before you start

You need the draft, and any sources the author used. Ask how strict the check should be, and the publication date, since some facts change. Ask for anything missing in one short message. If you can search the web, use it to find sources; if you can't, say so, and list where each claim could be checked.

## Steps

1. **List every checkable claim**: numbers, dates, names, titles, quotes, places, rankings, cause-and-effect claims and words like "first", "only" or "largest".
2. **Find the best source for each claim.** Prefer the original: the study, the official statistics, the company's own report, the court record or the recording. A news article that repeats a figure isn't the original source.
3. **Mark each claim:**
   - **supported**: a reliable source says the same thing;
   - **needs a source**: probably true, but nothing reliable was found yet;
   - **misleading**: true in a narrow sense, but missing context, out of date or overstated;
   - **wrong**: a reliable source contradicts it.
4. **Check quotes** against the recording or the original text, and check that they keep their context.
5. **Check the maths**: totals, percentages and the change from one figure to another.
6. **Suggest a fix** for every claim that isn't supported: a corrected figure, softer wording, added context, or cutting the claim.

## Output

```
| # | Claim | Status | Source | Suggested fix |
|---|---|---|---|---|

Summary: <n> supported · <n> need a source · <n> misleading · <n> wrong
Not checked: <claims outside what could be verified, and why>
```

## Rules

- **Never invent a source, a link or a quote.** If you can't find a source, say so.
- Say how sure you are when sources disagree, and show both.
- Treat claims about real people with extra care: wrong claims can harm them and may be defamatory.
- A source being popular doesn't make it reliable. Say why each source can be trusted.

## Checklist before you finish

- [ ] Every checkable claim is listed.
- [ ] Each claim has a status and, where possible, an original source.
- [ ] Quotes and calculations were checked.
- [ ] Every problem has a suggested fix.
