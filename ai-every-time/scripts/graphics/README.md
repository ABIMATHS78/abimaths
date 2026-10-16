# The guides' graphics

Each guide on aieverytime.org has a graphic made by an AI image generator, and the generator misspells and garbles
words in them: "Warewarusing" for "Warehousing", "httx" for "httpx", "PREEQUISITES". The app shows each guide's own
graphic with those mistakes corrected in place: everything in the picture is as the site has it except the wrong
words, which are rubbed out and redrawn in the graphic's own lettering, size and colour.

- `corrections/<article id>.json`: what was corrected in each graphic, and how (the format is in `AGENT.md`).
- `static/guides/<article id>.webp` in the app: the corrected graphics, made from the site's originals and these
  corrections. A guide without one shows the site's own copy (see `app/guides.py`).

## Making them again

The corrected graphics can be made again from the originals and the corrections, for example after a change to
`fix_text.py`:

```bash
pip install -r scripts/graphics/requirements.txt
sudo apt install fonts-inter fonts-liberation fonts-crosextra-carlito fonts-crosextra-caladea fonts-dejavu-core fonts-freefont-ttf
python scripts/graphics/get_fonts.py      # the Google Fonts it chooses from, about 12 MB
python scripts/graphics/build.py          # or: build.py <id> ...
```

The work folder (originals, fonts and a cache of font choices, about 1.5 GB) is `~/.cache/ai-every-time-graphics`, or
wherever `GRAPHICS_WORK` points. `build.py` downloads the originals it needs. A graphic whose corrections haven't
changed since its WebP was made is left as it is; delete the WebP to make it again.

## How the corrections were made

1. `download.py` fetches the originals, and `ocr_all.py` finds every line of text in them and where it is
   (RapidOCR).
2. `show.py <id>` shows a graphic, enlarged in quarters with its lines numbered, and lists the words that aren't
   words (`suspects.py`: not English, not used anywhere in the guides, and not the text reader's own usual
   misreadings), each with corrections suggested from the guide's own words.
3. Someone (or Claude, following `AGENT.md`) reads the graphic against its guide and writes its corrections.
4. `fix_one.py <id>` applies them with `fix_text.py` and shows each changed area before and after, to check the
   new words match their neighbours. `mark_done.py <id>` records the graphic as finished, and `publish.py` saves the
   finished ones as WebP into the app.

`fix_text.py` does the redrawing. For each corrected line it finds the old letters by their contrast with the
background, rebuilds the background underneath (a flat colour, or a blend of the background pixels around them),
chooses the font, weight and size whose rendering looks most like the old letters (from about 400 faces, judged on
the line and on correctly spelled lines of the same style), and draws the new words on the old baseline, in the old
letters' colour and softness. Words after a corrected one move along so the spacing stays even.
