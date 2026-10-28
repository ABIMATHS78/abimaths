---
name: release-notes
description: Writes release notes that users understand, explaining what changed, why it matters and anything they need to do. Use when shipping a release, update or fix, from a changelog, tickets or a spec.
---

# Release notes

Tell users what changed, in words they'd use, so they notice the improvements, know what to do, and aren't caught out by anything.

## Before you start

You need the list of changes (a changelog, closed tickets, a spec or notes), the version and release date, who the notes are for, and where they'll appear: in the product, by email, in the help centre or in an app store. Ask for anything missing in one short message, especially what users must do and any change that could break their setup.

## Steps

1. **Sort the changes.** Group them into actions needed, new features, improvements, fixes and known issues. Leave out internal changes users won't notice.
2. **Find the benefit to users.** For each change, ask what people can now do, or no longer have to worry about. Lead with that, then say briefly what changed.
3. **Write in the user's words.** No ticket numbers, code names or internal jargon. Name buttons and screens exactly as they appear.
4. **Put actions first.** If users must do something, such as update, sign in again or change a setting, say so at the top, with the steps and any deadline.
5. **Be honest about fixes and issues.** Describe what was wrong in plain terms ("Some invoices showed the wrong tax rate"). List known issues, with a workaround where there is one.
6. **Fit the channel.** App store notes need a few short lines. A help-centre page can carry more detail and links.

## Output

```
# <Product> <version>, <date>

**Action needed**: <what users must do, or leave this line out>

## New
- **<Headline that leads with the benefit>**: <what it does and where to find it>

## Improved
- <Change, in the user's terms>

## Fixed
- <What was wrong, now fixed>

## Known issues
- <Issue>. Workaround: <steps>
```

## Example

Before: "Fixed off-by-one error in CSV export (TCK-482)."
After: "CSV exports no longer leave out the last row of data."

## Rules

- Never claim a change, benefit or figure that isn't in the source material, and never add customer quotes or testimonials. If you can't tell what a change means for users, mark it "to confirm".
- Don't oversell. Skip words like "huge" or "amazing": describe the change and let users judge it.
- Don't bury breaking changes, removed features or price changes in the middle of the list.
- Don't name the customers who reported bugs, or include anyone's personal data.
- For security fixes, say what's fixed and what users should do, without details that would help an attacker.
- If a change affects how personal data is collected or used, say so plainly. Privacy laws may require users to be told.

## Checklist before you finish

- [ ] Actions users must take are at the top.
- [ ] Every item is in the user's words, with no internal jargon.
- [ ] Every claim matches the source material.
