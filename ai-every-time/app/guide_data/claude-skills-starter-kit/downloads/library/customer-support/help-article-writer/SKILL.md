---
name: help-article-writer
description: Writes help-centre articles from solved support tickets, with numbered steps, the expected result and fixes for common problems. Use when turning repeat questions into self-service help.
---

# Help article writer

Turn the answer an agent gave one customer into an article that lets the next customer help themselves. A good article solves one problem, in steps anyone can follow.

## Before you start

You need one or more solved tickets about the same problem, including the reply that fixed it. Useful extras: the product's exact menu and button names, screenshots, the help centre's style guide, and which plans or devices the article covers. If the tickets don't show what actually fixed the problem, ask for it in one short message before writing.

## Steps

1. **Name the problem in the customer's words.** Note how customers describe it in the tickets ("can't log in", "invoice missing"). Use those words in the title and first line, so people find the article when they search.
2. **Pull out the fix.** Separate the steps that solved the problem from the back-and-forth. If tickets used different fixes, find out which one is right; don't publish both as if they were equal.
3. **Title it as a task or a question.** "Download an invoice" or "Why was my payment declined?", not "Invoices".
4. **Write the steps.** One action per step, starting with a verb. Put on-screen labels in bold, exactly as they appear. Say where to start, such as the home page or the settings menu, and what the reader should see after key steps.
5. **Add what can go wrong.** List the common problems from the tickets, each with its fix. Say when to contact support, and what details to include.
6. **Test it.** Follow the steps exactly as written, or ask someone who can. Every step must work without knowledge the reader doesn't have.

## Output

```
# <Task or question, in the customer's words>

<One or two sentences: what this helps with, and who it's for.>

**Before you start**: <plan, permission or device needed, if any>

1. <Action, with the **exact label**>
2. <Action>. You'll see <expected result>.

## If something goes wrong
- **<Problem>**: <fix>

Still stuck? Contact support and include <details that help>.

**Notes for the editor**: <anything to confirm before publishing>
```

## Rules

- Remove every trace of the original customers: names, email addresses, order numbers, account details and anything else that could identify them. Follow the privacy laws that apply.
- Never invent a step, setting, menu name, limit or customer quote. Anything you couldn't confirm from the tickets or the product goes in the editor's notes, marked "to confirm".
- Don't publish an agent's workaround if it gets round security, a policy or a paid feature. Flag it for the editor instead.
- Keep each article to one problem. If the tickets cover two, write two articles.
- Use plain words and short sentences. Explain any term a customer wouldn't use.
- Mark where screenshots should go, such as "[Screenshot: billing page]". Never describe an image that doesn't exist.

## Checklist before you finish

- [ ] The title matches the words customers use.
- [ ] Each step has one action, and the labels match the product.
- [ ] The steps were tested, or the untested ones are marked "to confirm".
- [ ] No personal data from the tickets is left.
