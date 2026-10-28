---
name: code-review
description: Reviews code changes for bugs, security problems, missing tests and unclear code, ranked by severity, with a suggested fix for each. Use before merging code, especially code an AI tool wrote.
---

# Code review

Catch the problems that matter before code is merged, and explain them so they get fixed.

## Before you start

You need the code changes (a diff, a pull request or the files), and what the change is meant to do. Ask about the language and framework, how the code runs, the tests, and the project's conventions. Ask in one short message.

## Steps

1. **Understand the intent.** Read the description, then the change, and say in a sentence what it does.
2. **Check correctness**: logic errors; edge cases such as empty, missing, very large or repeated inputs; off-by-one errors; error handling; and timing problems, such as two requests at once.
3. **Check security:**
   - secrets in the code or the logs;
   - input used in SQL, shell commands, file paths or HTML without proper handling;
   - missing checks on who can see or change data;
   - personal data exposed or logged.
4. **Check the data**: database changes that could lose data, and migrations that can't be undone.
5. **Check the tests**: does a test cover the change, and would it fail if the code were wrong?
6. **Check readability**: names, structure, repetition, and comments that explain why.
7. **Check the effects elsewhere**: callers, APIs, speed and dependencies, including any new packages.
8. **Rank the findings**, and suggest a concrete fix for each one.

## Output

```
Summary: <what the change does, and the overall verdict>

| # | Severity | File:line | Problem | Why it matters | Suggested fix |
|---|---|---|---|---|---|
(Severity: blocker · should fix · nit)

Tests to add: <list>
Questions for the author: <list>
```

## Rules

- **Check before you report.** Trace each problem to a realistic input or situation, and don't report guesses as bugs. Ask a question when you're unsure.
- Rank by real impact: security problems and data loss first, style last.
- Never suggest committing secrets, turning off security checks, or skipping tests to make something pass.
- Check that new packages are real, maintained and correctly named. AI tools sometimes suggest packages that don't exist, and attackers publish packages with similar names.
- Be specific and respectful. Review the code, not the person.

## Checklist before you finish

- [ ] The change's intent is understood and stated.
- [ ] Security and data-loss risks were checked.
- [ ] Every finding has a location, a reason and a fix.
- [ ] The findings are ranked by severity.
