+++
title = "Ask Claude about your data, and trust the answer."
subtitle = "Claude will answer any question about your spreadsheet, confidently. Here's how to make sure the answer is right:"
description = "Prepare the file, make Claude describe the data first, define your terms, ask for the working and check the answer two ways. Prompts to copy."
updated = 2026-11-25
status = "published"
tags = ["claude", "data"]
archetype = "guide"
featured = true
level = "beginner"
topic = "Data"
+++
You upload a spreadsheet.

You ask Claude a question.

You get a clear, confident answer, with a number in bold.

And sometimes that number is wrong.

Not because Claude can't count. Because nobody told it what the data means, and nobody checked the working.

In data work, this is the same problem we've always had with reports: **a confident number is not a correct number.** The fix is the same too. You prepare the data, define your terms and check the working.

**By the end of this guide you'll have:**

- a file Claude can read without guessing
- a prompt that makes Claude describe the data before it answers
- a simple way to check any number in two minutes

You need: a spreadsheet or CSV file, and ten minutes.

---

## 1. Know why confident answers go wrong

Claude writes every answer in the same calm, clear voice.

A correct answer and a wrong one sound exactly alike.

Most wrong answers about data come from four places:

1. **The file is messy.** A totals row at the bottom gets counted as a sale. Two tables sit on one sheet.
2. **The question is vague.** "Revenue" could mean before or after refunds. "Last quarter" could mean calendar or financial.
3. **Nobody looked at the working.** The answer came back, and nobody asked how it was worked out.
4. **Nobody checked.** One number, taken on trust, goes into a slide.

Each step below fixes one of them.

> [!KEY]
> Treat Claude like a sharp new analyst on their first day. Brilliant, fast, and new to your data. You'd brief them and check their first report. Do the same here.

---

## 2. Prepare the file in two minutes

Clean files get correct answers. Before you upload:

- [ ] **One header row** at the top, with a clear name for every column.
- [ ] **One table per sheet.** Move notes, charts and side tables to another sheet.
- [ ] **Delete totals and subtotal rows.** Claude can add up. A totals row gets counted twice.
- [ ] **One format per column.** Dates all written the same way. Numbers without text like "approx." or "n/a".
- [ ] **Remove what's private.** Names, emails and phone numbers usually aren't needed to answer the question. Replace them with an ID.

If Claude says it can't open the file, check **Settings → Capabilities** and turn on **Create and edit files**.

> [!WARNING]
> Don't upload customer or staff data your company hasn't cleared for AI tools. Ask first. Anonymised data answers most questions just as well.

---

## 3. Make Claude describe the data first

This is the step that changes everything.

Before you ask your real question, ask Claude to tell you what it sees:

```prompt Describe the data first
Before answering anything, describe this file:
1. How many rows and columns, and what each column contains.
2. The date range.
3. Missing values, numbers stored as text, and repeated rows.
4. Anything that looks like a totals row or a second table.
Don't analyse anything yet.
```

Read the answer like a checklist.

Does the row count match what you expect? Is the date range right? Did it find the totals row you forgot to delete?

If something's off, fix the file and upload it again. It takes a minute now, and saves a wrong number later.

---

## 4. Define your terms

Vague questions get confident guesses.

Say exactly what you mean:

- **Which column?** "Use the Net_Amount column, not Gross."
- **Which rows?** "Leave out cancelled orders (Status = Cancelled)."
- **Which dates?** "Last quarter means July to September 2026."
- **Which unit?** "Show revenue in thousands of pounds, rounded."

Compare these two:

```prompt Vague
Which product made the most money last quarter?
```

```prompt Defined
Which product had the highest revenue from July to September 2026? Use the Net_Amount column and leave out rows where Status is Cancelled. Show the top five, with their revenue.
```

The first one makes Claude guess three things. The second one leaves nothing to guess.

---

## 5. Ask for the working, not just the answer

A number without the working is a number you can't check.

Ask Claude to show:

- **the steps** it took, in order
- **the row count after each step** ("after removing cancelled orders: 4,812 rows")
- **the formula or code** it used, so you can run it again

```prompt Show the working
Show your working: list each step, the number of rows after each step, and the formula you used. Then give the answer.
```

Row counts are your best friend. If 4,812 rows suddenly become 9,624 after combining two tables, something has gone wrong, and you'll see it straight away.

---

## 6. Check it two ways

You don't need to redo the analysis. You need two quick checks.

1. **Ask for the same number another way.** "Now calculate the same total with a pivot table instead of a formula. Do they match?"
2. **Spot-check three rows by hand.** Pick three rows from the file and see if they were counted the way you expected.

And one sanity check: **does the answer fit what you already know?** If last quarter's total revenue was about two million, a product earning three million is a red flag.

> [!TIP]
> Ask Claude to check itself: "What could make this answer wrong? List the three most likely problems with the data or the method." It's surprisingly good at finding its own weak spots once you ask.

---

## 7. Watch for the five traps

These cause most wrong answers. Ask Claude to check for each one.

### Numbers stored as text

"1,200" typed as text doesn't add up. It gets skipped, or sorted in the wrong order.

### Repeated rows

The same order exported twice doubles its revenue. Ask: "Are there repeated rows or repeated order IDs?"

### Totals counted twice

A subtotal row in the middle of the data gets added to the total. Delete them before uploading.

### Gaps in the dates

A missing week looks like a sales drop. Ask: "Are there gaps in the dates?"

### Joins that multiply rows

Combining two tables (orders and customers, say) can copy rows when a key appears more than once. Ask for the row count before and after every join.

---

## 8. Put it together

Here's the full prompt I'd use for any important question:

```prompt Ask about your data
Here's our sales export.

1. First, describe the file: rows, columns, date range, missing values, numbers stored as text, repeated rows, and any totals rows.
2. Then answer: which product had the highest revenue from July to September 2026? Use Net_Amount and leave out rows where Status is Cancelled. Show the top five.
3. Show your working: each step, the row count after each step, and the formula.
4. Finally, list the three things most likely to make this answer wrong.
```

Change the question in step 2. Keep the rest. It works for almost any question about almost any spreadsheet.

---

## Before you finish

- [ ] The file has one header row, one table and no totals rows.
- [ ] Claude described the data, and the row count and dates look right.
- [ ] Your question names the column, the rows and the dates.
- [ ] You saw the working, with row counts after each step.
- [ ] You checked the number a second way.

---

## Questions

### Can Claude work with Excel files, or only CSV?

Both. If a workbook has several sheets, say which sheet to use.

### What about very large files?

Big exports work better as a CSV with only the columns you need. Fewer columns, fewer chances to pick the wrong one.

### Is this only for Claude?

No. The same five steps work with any AI tool, and with a new colleague too.
