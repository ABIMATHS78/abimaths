---
name: screening-scorecard
description: Builds a scorecard for screening applications against the role's must-haves, recording the evidence for each rating for a person to review. Use when shortlisting, so every application is judged alike.
---

# Screening scorecard

Screen every application against the same job-related criteria, with the evidence written down, so a person can make a fair decision.

## Before you start

You need the job description, with its must-have and nice-to-have requirements. Ask who will review the applications, how many there are, and whether the organisation allows AI to help with screening. Some places regulate automated tools in hiring, and may require telling candidates or auditing the tool, so ask the hiring team to check the rules that apply first. Ask in one short message.

## Steps

1. **Turn each requirement into a criterion** that can be seen in an application, with a definition.
2. **Set rating anchors** for each criterion: what no evidence, some evidence and clear evidence look like.
3. **Leave out what doesn't predict performance in the job**: names, photos, addresses, dates of birth, graduation years, gaps in employment, and the prestige of schools or employers.
4. **For each application the reviewer shares**, quote the evidence found for each criterion, and suggest a rating with a short reason. Where an application says nothing about a criterion, mark it "no evidence", which isn't the same as "doesn't have the skill".
5. **Flag applications for a careful human read** when they're borderline or unusual, such as career changers or people from non-traditional backgrounds.
6. **Hand over to the reviewer**: a table of evidence and suggested ratings for them to check, change and decide on.

## Output

```
# Scorecard: <role>
| Criterion | No evidence (0) | Some evidence (1) | Clear evidence (2) |
|---|---|---|---|

## Application <reference>
| Criterion | Evidence (quoted) | Suggested rating | Reason |
|---|---|---|---|
For the reviewer: <points to read closely>
```

## Rules

- **A person makes every decision.** Don't reject, rank or shortlist candidates yourself. Suggest ratings, with evidence, for a person to review.
- Use only job-related criteria. Never infer or consider age, gender, ethnicity, nationality, religion, disability, health, family status or any other protected characteristic, or anything that stands in for them, such as names or graduation years.
- Quote evidence from the application. Never invent or assume experience.
- Treat applications as confidential personal data, and follow the data protection and employment laws that apply.

## Checklist before you finish

- [ ] Every criterion is job-related and defined.
- [ ] Every suggested rating quotes evidence.
- [ ] Identifying and irrelevant details were kept out of the ratings.
- [ ] The decision is left to a person.
