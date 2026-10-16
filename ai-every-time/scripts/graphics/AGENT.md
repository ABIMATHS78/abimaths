# Correcting a guide graphic

Each education guide on aieverytime.org has a graphic (a 1024 x 1024 infographic) made by an AI image generator,
which misspells and garbles words. Your job: correct every mistake in the graphic's text **in place**, so it looks
exactly as before except that its words are right. The owner approved this from a trial; keep to that standard.

Tools: `python scripts/graphics/<tool>.py`; `$W/work` is the work folder (see `paths.py`). Work only on your graphic's files.
Be efficient: view several images in one turn (several Read calls at once), and don't view images you don't need.

## Steps

1. **Look.** `show.py <id>` prints every line of text the reader found (number, box, reading) and a list of
   **likely mistakes** with suggested corrections. View `full.png` and the `suspects*.png` sheets (the suspect lines
   enlarged) in one turn. Then view the enlarged quarters (`tile_00.png` top left, `tile_01.png` top right,
   `tile_10.png` bottom left, `tile_11.png` bottom right; lines boxed and numbered) for any part with small text, to
   find mistakes the list missed (it misses about one in ten). For a closer look at a spot:
   `crop.py <id> x0 y0 x1 y1` (graphic pixels, with a ruler); add `--both` after a correction to see the original
   above the corrected version, or `--fixed` for the corrected one alone.
2. **Check the words against the guide** (`work/guide/<id>.md`: title, summary, then the guide), e.g.
   `grep -n -i -m3 "orchestr" $W/work/guide/<id>.md`, to know what a garbled word should be.
3. **Write the corrections** to `$W/work/spec/<id>.json` (format below). No mistakes: `{"edits": []}`.
4. **Apply and check.** `fix_one.py <id>` writes the corrected graphic and `check.png` (with `check_2.png` if
   there are many): every changed area, original above, corrected below. **View the check sheets.** Each new word
   must be spelled right; match its neighbours in size, weight, colour and style; sit on the line; stay inside its
   box or column; and leave no trace of the old letters. For a closer look at one area, view its `pair_N.png` or
   use `crop.py ... --both`. Fix the spec and rerun until right (reruns are quick). A correction that still looks
   wrong after two or three tries: take it out and list it in `"skipped"` with the reason.
5. **Finish.** `mark_done.py <id> "short note"`, then go on to your next graphic, if you have several. At the end,
   reply with one line per graphic: id, number of corrections, what was skipped and why.

## What to correct

- Misspelled, garbled or nonsense words; wrong names of tools and terms ("TensorFow", "httx", "Linkedin", "SaAs",
  "IOT" in a mixed-case line → "IoT"); doubled words ("scalable scalable"); broken code (make it valid, as the guide
  writes it); obvious wrong numbers in a sequence.
- A word or label that is unreadable gibberish: replace it with wording from the guide that fits the slot (about
  the same length). Keep such rewordings few and faithful to the guide.

Leave alone: text that is already right (no rewording or restyling), the graphic's capitalisation style, correct
British or American spelling, missing hyphens ("SELF HEALING", "AI POWERED": not spelling; add one only in a line
you are redrawing anyway), logos, icons, and lettering that is part of a picture (screens, badges, curved or 3D
text). Wrong but impossible to redraw cleanly (curved, on a busy picture, tiny inside an icon): list in `"skipped"`.

## The corrections file

```json
{"family": "serif",
 "edits": [
  {"box": [96, 168, 915, 197], "was": "Learn how to build full-stack web applications instanty using Bolt.new with",
   "replace": "instanty", "text": "instantly", "style": "sub"},
  {"box": [156, 198, 854, 231], "was": "precise prompt engineering strategies, templates, and workflows", "ref": true, "style": "sub"},
  {"box": [311, 390, 469, 409], "was": "PREQUISITES & SETUP", "text": "PREREQUISITES & SETUP", "style": "pill", "align": "center", "family": "sans"},
  {"box": [101, 391, 213, 408], "was": "CORE CONCEPTS", "ref": true, "style": "pill"},
  {"box": [43, 149, 808, 190], "was": "Snowflake Cloud Data Warewarusing for", "words": [3, 3], "text": "Warehousing", "style": "title"},
  {"box": [372, 454, 419, 473], "was": "pip thrall", "text": ""}
 ],
 "skipped": ["'Rse AI' on the trophy: curved lettering"]}
```

One edit per line of text (box). Keys:
- `box` `[x0, y0, x1, y1]`: the line, from the reader's list. If the reader missed a line or its box is wrong
  (joins two lines, cuts letters), measure one with `crop.py`: just around the letters, a few pixels of margin.
- `was`: the line as the picture shows it, mistakes and all. It must have **the same number of words (spaces)
  as the picture**; the letters roughly right (the tool matches the font from it). Don't copy the reader's text
  blindly: it reads "AI" as "Al" and runs words together ("SecureRollbacks"); count the words in the picture.
- `text`: what it should say (`""` removes the line).
- `replace`: for one or two wrong words in a longer line, the wrong word(s) exactly as in `was`; `text` is then just
  the new word(s). Only those are redrawn; the rest keeps its original pixels and moves along if needed. Or
  `words: [first, last]` by position from 0 (`-1` is the last word). Prefer these over redrawing a whole line.
- `align`: `"center"` for centred text (labels in pills, captions under icons), `"right"`; default left.
- `style`: give lines that look alike (font, size, colour) the same name. Add one to three **correctly spelled**
  lines of that style as references (`"ref": true`, no `text`): the font match is much better with them.
- `family`: `sans` (default with `condensed`), `condensed` (tall narrow), `serif`, `mono` (code), or joined with `+`.
  At the top for the whole graphic, or on the first edit of a style.
- `room`: the widest the new text may be, in pixels (lower it if new text would hit an icon or border).
- `pad` (default 3): use 1 for tightly stacked lines. Rarely: `font`, `size`, `dy` (pixels down; negative is up).

If `fix_one.py` reports a word-count mismatch, count the spaces in the picture again (`crop.py`): "Bolt.new",
"full-stack" and "2.x" are one word each. If new text looks too bold, too light or the wrong shape, add references
or change `family`. A leftover mark beside new text: widen the box on that side. Finished examples:
`$W/work/spec/19560003.json`, `23160009.json`, `20070002.json`.
