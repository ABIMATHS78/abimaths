# LinkedIn cover: Data Leader

![LinkedIn cover](linkedin-cover-1584x396.png)

`linkedin-cover-1584x396.png` is **1584 × 396 px**, LinkedIn's recommended size
for a personal profile background (4:1). It fits the upload frame exactly, so
there's nothing to crop.

**Concept: noise → signal → decision → impact.** Scattered data points narrow
into one clean trend line that ends at a single amber point, the decision. On
the right, the same dots line up as a rising column chart, the impact. The
amber full stop after "Data Leader" is the same mark as the decision point.

**Safe zones.** All text sits between x = 663 and 1190 px. That keeps it clear
of the bottom-left corner the profile photo covers and inside the central 60%
of the width that some phone layouts show. Only the decorative scatter goes
under the photo.

## Upload

1. On your profile, click the pencil/camera icon on the background banner.
2. Upload `linkedin-cover-1584x396.png`.
3. Leave zoom and straighten at their defaults, then click Apply and Save.

## Edit and re-render

Change `HEADLINE`, `SUBTITLE`, `PIPELINE`, or the colours in `C` at the top of
the script in `cover.html`, then run:

```bash
node linkedin-cover/render.cjs
```

This needs Node, Playwright with Chromium, and three free fonts: Inter, Inter
Display and JetBrains Mono. To draw the profile-photo and phone-crop guides on
top, run `node linkedin-cover/render.cjs preview.png --debug`.
