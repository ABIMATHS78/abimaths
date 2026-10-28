---
name: commit-message
description: Writes clear commit messages and pull request descriptions from a diff, saying what changed, why, and how it was tested. Use before committing code or opening a pull request.
---

# Commit message

Leave a history that tells the next person, who is often you, what changed and why.

## Before you start

You need the diff, or a description of the changes. Ask why the change was made: the bug, the request or the goal. Ask whether the project has its own conventions, such as prefixes like "fix:" or "feat:", or links to an issue tracker. Ask in one short message.

## Steps

1. **Check the change is one thing.** If the diff mixes unrelated changes, suggest splitting it into separate commits.
2. **Write the subject line** in the imperative, as in "Fix login timeout on slow connections": about 50 characters, with no full stop, following the project's conventions.
3. **Write the body**, wrapped at about 72 characters:
   - why the change was needed;
   - what it does, at a high level rather than line by line;
   - anything surprising, such as side effects, trade-offs or follow-ups.
4. **Reference the issue or ticket**, if there is one.
5. **For a pull request**, add a summary, how it was tested, screenshots of visual changes, and anything reviewers should look at closely.

## Output

```
<Subject line>

<Why the change was needed.>

<What it does, and anything surprising.>

Refs: <issue>
```

For a pull request:

```
## Summary
## Why
## How it was tested
## Notes for reviewers
```

## Rules

- **Describe only what's in the diff.** Never say tests were run or bugs were fixed unless that's true.
- Never include secrets, passwords, tokens or customers' personal data in a commit message.
- Keep it factual. Leave out filler such as "various improvements" or "minor changes".

## Checklist before you finish

- [ ] The subject line is short, specific and in the imperative.
- [ ] The body says why, not just what.
- [ ] The message matches the diff.
- [ ] No secrets or personal data are included.
