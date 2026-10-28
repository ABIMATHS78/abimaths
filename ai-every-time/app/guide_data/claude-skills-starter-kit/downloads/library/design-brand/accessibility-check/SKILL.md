---
name: accessibility-check
description: Checks a design, web page or document for common accessibility problems, such as contrast, text size, alt text, headings, links, focus and captions. Use before publishing or handing off a design.
---

# Accessibility check

Find the barriers that stop disabled people from using a design, and say how to fix each one.

## Before you start

You need the design, page or document: a link, screenshots, the code, or the colours and font sizes used. Ask what it is, who will use it, and the standard the organisation follows; many use WCAG 2.2 level AA. Ask for anything missing in one short message.

## Steps

1. **Check colour contrast** for every pair of text and background colours. Level AA needs at least 4.5:1 for normal text, 3:1 for large text (at least 24 CSS pixels, or 18.66 pixels when bold), and 3:1 for the parts of buttons, form fields, icons and charts that people need to see. Run `scripts/contrast.py` with the hex colours to get exact ratios, for example `python scripts/contrast.py "#767676" "#ffffff"`.
2. **Check colour alone**: errors, required fields, links and chart series need a second cue, such as text, an icon, a pattern or an underline.
3. **Check text**: a readable size, text that can be enlarged to 200% without breaking the layout, comfortable line spacing, and no important text inside images.
4. **Check images**: meaningful images have alt text that says what matters; decorative images are hidden from screen readers; charts have a text summary.
5. **Check structure**: one main heading, headings in order, real lists and tables, and a sensible reading order.
6. **Check links and buttons**: their text makes sense out of context ("Download the report", not "Click here"), and touch targets are at least 24 by 24 CSS pixels.
7. **Check keyboard use**, for pages and apps: everything works with a keyboard alone, the focus is always visible, and the order makes sense.
8. **Check forms**: every field has a visible label, errors say what's wrong and how to fix it, and nothing times out without warning.
9. **Check media**: videos have captions, audio has a transcript, and nothing flashes more than three times a second.

## Output

```
| # | Issue | Where | Who it affects | Severity | Fix |
|---|---|---|---|---|---|

Contrast: <each pair, its ratio, pass or fail>
Already good: <what works well>
Not checked: <what needs testing with people and assistive technology>
```

## Rules

- **Never guess a contrast ratio.** Calculate it from the colour values, and say when a value had to be read from a screenshot.
- Rank issues by their effect on people: blockers first, then serious problems, then improvements.
- Say what couldn't be checked from the material given, such as keyboard use from a static image.
- An automated or AI check finds many problems, but not all. Recommend testing with disabled people and with screen readers.
- Accessibility is a legal requirement in many places. Point to the law and standard that apply, rather than assuming one.

## Checklist before you finish

- [ ] Every text and background pair has a calculated contrast ratio.
- [ ] Each issue says who it affects, how severe it is and how to fix it.
- [ ] No information is carried by colour alone.
- [ ] What couldn't be checked is listed.
