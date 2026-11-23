"""A LinkedIn-style preview of the post, with the 'see more' fold where LinkedIn puts it."""

from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path

from .analyze import FOLD_CHARS
from .brand import avatar_path

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Preview: {title}</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 0; background: #F4F2EE; font: 14px/1.42 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #191919; }}
.wrap {{ max-width: 555px; margin: 24px auto; padding: 0 12px; }}
.card {{ background: #fff; border-radius: 8px; box-shadow: 0 0 0 1px rgba(0,0,0,.08); overflow: hidden; }}
.who {{ display: flex; gap: 8px; padding: 12px 16px 0; }}
.avatar {{ width: 48px; height: 48px; border-radius: 50%; object-fit: cover; background: #D97757; color: #fff; display: grid; place-items: center; font-weight: 700; flex: none; }}
.name {{ font-weight: 600; font-size: 14px; }} .meta {{ color: #666; font-size: 12px; line-height: 1.33; }}
.text {{ padding: 8px 16px 0; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 14px; }}
.more {{ color: #666; background: none; border: 0; padding: 0; font: inherit; cursor: pointer; }}
.more:hover {{ color: #0a66c2; text-decoration: underline; }}
.media {{ margin-top: 8px; }} .media img, .media video {{ display: block; width: 100%; height: auto; }}
.bar {{ display: flex; justify-content: space-around; border-top: 1px solid #e8e8e8; margin: 0 16px; padding: 4px 0; color: #666; font-weight: 600; }}
.bar span {{ padding: 10px 8px; }}
.note {{ color: #666; font-size: 12px; margin: 10px 4px; }}
</style></head>
<body><div class="wrap">
<article class="card">
  <div class="who">{avatar}<div><div class="name">{name}</div><div class="meta">{headline}<br>Now · 🌐</div></div></div>
  <div class="text"><span id="short">{short}</span><span id="full" hidden>{full}</span>{toggle}</div>
  <div class="media">{media}</div>
  <div class="bar"><span>👍 Like</span><span>💬 Comment</span><span>🔁 Repost</span><span>➤ Send</span></div>
</article>
<p class="note">{chars} characters. The fold falls after about {fold} characters on a computer and about 140 on a phone.</p>
</div>
<script>
const more = document.querySelector(".more");
if (more) more.addEventListener("click", () => {{ document.getElementById("short").hidden = true; document.getElementById("full").hidden = false; more.remove(); }});
</script>
</body></html>
"""


def _fold(text: str) -> str:
    """Where LinkedIn cuts the post: about three lines or 210 characters, whichever comes first."""
    lines = text.split("\n")
    kept, used = [], 0
    for line in lines[:3]:
        if used + len(line) > FOLD_CHARS:
            kept.append(line[: max(0, FOLD_CHARS - used)].rsplit(" ", 1)[0])
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept).rstrip()


def _avatar(brand: dict) -> str:
    path = avatar_path(brand)
    if path:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img class="avatar" src="data:{mime};base64,{data}" alt="">'
    initials = "".join(part[0] for part in str(brand.get("name", "?")).split()[:2]).upper()
    return f'<div class="avatar">{html.escape(initials)}</div>'


def write_preview(out_dir: Path, post: str, brand: dict, files: dict, alt_text: str = "") -> Path:
    out_dir = Path(out_dir)
    alt = html.escape(alt_text or "", quote=True)
    if files.get("mp4"):
        media = f'<video src="{Path(files["mp4"]).name}" autoplay muted loop playsinline aria-label="{alt}"></video>'
    elif files.get("gif"):
        media = f'<img src="{Path(files["gif"]).name}" alt="{alt}">'
    elif files.get("png"):
        media = f'<img src="{Path(files["png"]).name}" alt="{alt}">'
    else:
        media = ""
    short = _fold(post)
    cut = len(short) < len(post.rstrip())
    page = PAGE.format(
        title=html.escape(post.split("\n", 1)[0][:60]),
        avatar=_avatar(brand),
        name=html.escape(brand.get("name", "")),
        headline=html.escape(brand.get("headline", "")),
        short=html.escape(short) + ("…" if cut else ""),
        full=html.escape(post),
        toggle=' <button class="more">see more</button>' if cut else "",
        media=media,
        chars=f"{len(post):,}",
        fold=FOLD_CHARS,
    )
    path = out_dir / "preview.html"
    path.write_text(page, encoding="utf-8")
    return path
