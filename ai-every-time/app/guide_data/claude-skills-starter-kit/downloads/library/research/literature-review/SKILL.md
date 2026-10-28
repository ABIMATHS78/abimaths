---
name: literature-review
description: Builds a literature review table from papers the user provides, with each paper's aims, methods, sample, findings and limits, and a synthesis. Use when reviewing research for a thesis or report.
---

# Literature review

Turn a set of papers into a table that makes them easy to compare, and a short synthesis of what they show together. Work only from the papers provided.

## Before you start

You need the papers (full text where possible, not just abstracts), the review question, and any required columns or citation style. If the review question is missing, ask for it in one short message: it decides what counts as relevant.

## Steps

1. **List the papers.** Give each an ID and its full reference, taken from the paper itself. Note any you could only read in part, such as the abstract alone.
2. **Extract the details.** For each paper, record the aim, design, sample and setting, measures, main findings with their figures, and the limitations the authors state. Copy figures exactly as reported, with units and the page or table.
3. **Note the quality.** Record the design (for example, randomised trial, cohort study, survey, case study or review), the sample size, risks of bias, and funding or conflicts of interest. Keep this apart from the findings.
4. **Mark the relevance.** In one line, say how each paper bears on the review question: directly, partly or only as background.
5. **Synthesise.** Group papers by what they find, rather than summarising them one by one. Say where results agree, where they conflict and why they might (different samples, methods or measures), and what nobody has studied.
6. **Check the table against the papers.** Re-read each row against its source. Mark any detail you're unsure of "to confirm".

## Output

```
| ID | Reference | Aim | Design and sample | Key findings | Limitations | Quality notes | Relevance |
|---|---|---|---|---|---|---|---|
| P1 | <authors, year, title> | <aim> | <design, n, setting> | <with figures, page or table> | <limits> | <bias, funding> | Direct / Partial / Background |

## Synthesis
**Where studies agree**: <finding, with paper IDs>
**Where they differ**: <finding, likely reasons, paper IDs>
**Gaps**: <what hasn't been studied, or not studied well>
**Overall strength of evidence**: <strong / moderate / weak, and why>
```

## Rules

- Never invent a paper, author, finding, figure or citation. If a detail isn't in the text you were given, write "not reported" or "to confirm".
- Don't add papers from memory. You can suggest kinds of study to look for, clearly labelled as suggestions, not sources.
- Keep what the authors found apart from how they interpret it, and both apart from your own view.
- Don't treat a statistically significant result as a large or important one. Report effect sizes where the paper gives them.
- Quote only short passages, in quotation marks with a page number. Otherwise, summarise in your own words.
- Respect copyright law and the licence terms that apply: don't reproduce large parts of a paper. Never repeat details that could identify study participants.
- A table built from the papers someone has is not a systematic review. If one is needed, say it requires a documented search and screening process.

## Checklist before you finish

- [ ] Every row matches its paper, with figures copied exactly.
- [ ] Nothing in the table comes from outside the papers provided.
- [ ] The synthesis groups findings across papers and cites IDs.
- [ ] Gaps and the strength of the evidence are stated.
