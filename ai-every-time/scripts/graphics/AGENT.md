# Correcting guide graphics

Each education guide on aieverytime.org has a graphic (a 1024 x 1024 infographic) made by an AI image generator,
which misspells and garbles words. Correct every mistake in a graphic's text **in place**, so it looks exactly as
before except that its words are right. The owner approved this from a trial; keep to that standard.

Run tools as `python scripts/graphics/<tool>.py`; `$W/work` is the work folder (see `paths.py`). Work only on your
graphics' files.

## Be economical: this job is 1,100 graphics, paid for in usage

Every tool call and every image costs. Follow this sequence; about 6 tool calls per graphic is the aim, 12 the most.
Several Read calls in one turn count once. Chain commands with `&&` in one Bash call where you can.

1. `show.py <id>`: prints the likely mistakes (with corrections suggested from the guide's own words), the guide's
   title and summary, and every line of text found (number, box, the reader's reading).
2. **One turn:** view `top.png`, `bottom.png` (the graphic enlarged 1.5x, lines numbered) and `suspects*.png` (the
   suspect lines enlarged). Read the whole graphic: the list misses about one mistake in ten. Only if some text is too
   small to read there: `crop.py <id> x0 y0 x1 y1` (graphic pixels, with a ruler). Check a word against the guide
   only if you can't tell what it should be: `grep -n -i -m3 "word" $W/work/guide/<id>.md`.
3. **One Bash call:** write the corrections file with a heredoc, then apply them:
   `cat > $W/work/spec/<id>.json <<'EOF' ... EOF` `&& python scripts/graphics/fix_one.py <id>`.
   No mistakes: `{"edits": []}`.
4. **One turn:** view `check.png` (and `check_2.png`, `check_3.png` if it says so): each changed area, original
   above, corrected below. Each new word must be spelled right; match its neighbours in size, weight, colour and
   style; sit on the line; stay inside its box; and leave no trace of the old letters.
5. If something is wrong, fix the spec and rerun (one Bash call), then view the check again. **Two rounds at most:**
   a correction that still looks wrong is taken out and listed in `"skipped"` with the reason.
6. **One Bash call:** `mark_done.py <id> "short note" && show.py <next id>`, then go on from step 2.

At the end, reply with one line per graphic: id, number of corrections, what was skipped and why.

## What to correct

- Misspelled, garbled or nonsense words; wrong names of tools and terms ("TensorFow", "httx", "Linkedin", "SaAs",
  "IOT" in a mixed-case line → "IoT"); doubled words ("scalable scalable"); broken code (make it valid, as the guide
  writes it); obvious wrong numbers in a sequence.
- Unreadable gibberish: replace with wording from the guide that fits the slot (about the same length). Keep such
  rewordings few and faithful to the guide.
- Never coin a name, field or term that isn't real (not "Product_Name_Phone" for a garbled "Preduct_Name_Ifone"):
  use the guide's own term, even if it's shorter than the slot, and keep brackets and lists balanced across lines.

Leave alone: text that is already right (no rewording or restyling), the graphic's capitalisation style, correct
British or American spelling, missing hyphens ("SELF HEALING": add one only in a line you are redrawing anyway),
missing commas, logos, icons, and lettering that is part of a picture (screens, badges, curved or 3D text). Wrong but
impossible to redraw cleanly (curved, on a busy picture, tiny inside an icon): list in `"skipped"`.

Read box rows and columns from `crop.py`'s ruler ticks, not from where the crop starts: a box a few rows off
leaves specks of the old letters. In tiny text with tight gaps between words, redraw the whole line or the single
word instead of using `replace` (the word count can split the wrong words).

## The corrections file

```json
{"edits": [
  {"box": [96, 168, 915, 197], "was": "Learn how to build full-stack web applications instanty using Bolt.new with",
   "replace": "instanty", "text": "instantly", "style": "sub"},
  {"box": [156, 198, 854, 231], "was": "precise prompt engineering strategies, templates, and workflows", "ref": true, "style": "sub"},
  {"box": [311, 390, 469, 409], "was": "PREQUISITES & SETUP", "text": "PREREQUISITES & SETUP", "style": "pill", "align": "center"},
  {"box": [101, 391, 213, 408], "was": "CORE CONCEPTS", "ref": true, "style": "pill"},
  {"box": [43, 149, 808, 190], "was": "Snowflake Cloud Data Warewarusing for", "words": [3, 3], "text": "Warehousing", "style": "title"},
  {"box": [372, 454, 419, 473], "was": "pip thrall", "text": ""}
 ],
 "skipped": ["'Rse AI' on the trophy: curved lettering"]}
```

One edit per line of text (box):
- `box` `[x0, y0, x1, y1]`: the line's box from the reader's list. If the reader missed a line or its box is wrong
  (joins two lines, cuts letters), measure one with `crop.py`: just around the letters, a few pixels of margin.
- `was`: the line as the picture shows it, mistakes and all, with **the same number of words (spaces) as the
  picture**. Don't copy the reader's text blindly: it reads "AI" as "Al" and runs words together; "Bolt.new",
  "full-stack" and "2.x" are one word each.
- `text`: what it should say (`""` removes the line).
- `replace`: for one or two wrong words in a longer line, the wrong word(s) exactly as in `was`; `text` is then just
  the new word(s). Only those are redrawn; the rest keeps its pixels and moves along if needed. Or `words: [first,
  last]` by position from 0 (`-1` is the last). Prefer these to redrawing a whole line. In a centred line whose new
  words are much longer or shorter, redraw the whole line with `"align": "center"` instead.
- `align`: `"center"` for centred text (labels in pills, captions under icons), `"right"`; default left.
- `style`: lines that look alike (font, size, colour) get the same name. Add one or two **correctly spelled** lines of
  each style as references (`"ref": true`, no `text`): the font match is much better with them.
- Fonts: the tool picks one main typeface for the graphic (it prints it) and the weight of it for each style, drawn
  narrower or wider if the original is condensed or wide. If the new letters look wrong, set another at the top for
  the whole graphic, `"typeface": "Roboto"` (or `Inter`, `OpenSans`, `Montserrat`, `Poppins`, `Lato`,
  `SourceSans3`, `RobotoCondensed`, `Oswald`, `PlayfairDisplay`, `Merriweather`, ...: a font file's name before the
  dash), or per style on one of its edits. `family` (`sans`, `condensed`, `serif`, `mono`) narrows the choice;
  `font` (a file name) forces one.
- `room`: the widest the new text may be, in pixels (lower it if new text would hit an icon or border). `pad`
  (default 3): 1 for tightly stacked lines. Rarely: `size`, `squeeze` (0.9 = 10% narrower), `dy` (pixels down),
  `slant` (0.2 leans the new letters about 11 degrees, for italic lettering no font matches: a rounded hand-lettered
  italic is close to `"font": "Nunito-800.woff", "slant": 0.2`).

The tool erases whatever touches the old letters: if a letter touches an outline or icon, start the box after that
letter (redraw the rest of the word) so the outline stays whole. Or correct just the wrong letter: box only it,
one pixel clear of it on each side, with `"pad": 0` (`{"box": [910, 683, 917, 697], "was": "a", "text": "e",
"pad": 0}`); a box no wider than the letter finds nothing. Changing only `"skipped"` needs no rerun.
