"""Substack export: each issue and guide as clean HTML that pastes into the Substack editor (headings, lists, quotes,
code blocks, images and links survive; site-only boxes become plain links), plus one "kit" page with Copy buttons
and the publication's setup text.

Substack has no API for writing posts, so the author pastes: open dist/substack/index.html, press "Copy post",
then paste into a new Substack post. Images load from the live site, so publish the site first.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from .config import load_site
from .content import load_library
from .md import Renderer
from .paths import DIST_DIR, MEDIA_DIR

PAGES = {"subscribe": "subscribe/", "archive": "archive/", "guides": "guides/", "start": "start/",
         "library": "library/", "about": "about/"}


def _absolute(site, ref: str, refs: set[str]) -> str | None:
    kind, _, slug = ref.partition(":")
    if kind == "page":
        if slug == "subscribe" and site["newsletter"].get("provider") == "substack":
            return f"https://{site['newsletter']['username']}.substack.com/subscribe"
        return site.absolute(PAGES.get(slug, ""))
    if ref not in refs:
        return None
    return site.absolute(("p/" if kind == "issue" else "guides/") + slug + "/")


def _media(site, src: str) -> str | None:
    clean = src.lstrip("./")
    if not (MEDIA_DIR / clean).is_file():
        return None
    jpg = Path(clean).with_suffix(".jpg")
    if (MEDIA_DIR / jpg).is_file():           # Substack accepts JPG everywhere
        clean = jpg.as_posix()
    return site.absolute("media/" + clean)


def to_substack(body_html: str) -> str:
    """Rewrite the site's boxes as plain HTML the Substack editor keeps."""
    h = body_html
    h = re.sub(r'<div class="prompt"><div class="prompt-head"><span>(.*?)</span>.*?</div><pre><code>(.*?)</code></pre></div>',
               r"<p><strong>\1</strong></p><pre><code>\2</code></pre>", h, flags=re.S)
    h = re.sub(r'<div class="codeblock">.*?<pre><code[^>]*>(.*?)</code></pre></div>', r"<pre><code>\1</code></pre>", h, flags=re.S)
    h = re.sub(r'<aside class="callout callout-[a-z]+"><p class="callout-title">(.*?)</p>(.*?)</aside>',
               r"<blockquote><p><strong>\1:</strong></p>\2</blockquote>", h, flags=re.S)
    h = re.sub(r'<p class="cta"><a class="btn[^"]*" href="([^"]*)"[^>]*>(?:<svg.*?</svg>\s*)?(.*?)</a></p>',
               r'<p><strong><a href="\1">\2 →</a></strong></p>', h, flags=re.S)
    h = re.sub(r'<figure class="figure"><img src="([^"]*)" alt="([^"]*)"[^>]*>(?:<figcaption>(.*?)</figcaption>)?</figure>',
               lambda m: f'<img src="{m.group(1)}" alt="{m.group(2)}">' + (f"<p><em>{m.group(3)}</em></p>" if m.group(3) else ""), h, flags=re.S)
    h = re.sub(r'<figure class="figure figure-missing".*?</figure>', "", h, flags=re.S)
    h = re.sub(r'<input class="task-list-item-checkbox"[^>]*checked[^>]*>', "☑", h)
    h = re.sub(r'<input class="task-list-item-checkbox"[^>]*>', "☐", h)
    h = re.sub(r'\s(class|id|loading|rel|data-[a-z-]+)="[^"]*"', "", h)
    h = re.sub(r"<button[^>]*>.*?</button>", "", h, flags=re.S)
    return h.strip()


def export(out_dir: Path | None = None) -> Path:
    site, library = load_site(), load_library().visible()
    refs = {item.ref for item in library.all()}
    sub_url = (f"https://{site['newsletter']['username']}.substack.com/subscribe"
               if site["newsletter"].get("provider") == "substack" else site.absolute("subscribe/"))

    def card(ref):
        item = next((i for i in library.all() if i.ref == ref), None)
        if not item:
            return None
        return f'<p><strong>Read next: <a href="{_absolute(site, ref, refs)}">{html.escape(item.title)}</a></strong></p>'

    renderer = Renderer(resolve=lambda ref: _absolute(site, ref, refs), card=card, media=lambda src: _media(site, src),
                        subscribe=lambda label: f'<p><strong><a href="{sub_url}">{html.escape(label or "Subscribe for free")} →</a></strong></p>')
    posts = []
    for item in library.issues + library.guides:
        rendered = renderer.render(item.body)
        cover = _media(site, item.cover) if item.cover else ""
        posts.append({"slug": item.slug, "kind": item.kind, "title": item.title, "subtitle": item.subtitle,
                      "description": item.description, "cover": cover or "", "date": item.date.isoformat() if item.date else "",
                      "words": rendered.words, "html": to_substack(rendered.html)})
    out = Path(out_dir) if out_dir else DIST_DIR / "substack"
    out.mkdir(parents=True, exist_ok=True)
    for post in posts:
        (out / f"{post['slug']}.html").write_text(
            f"<!doctype html><meta charset='utf-8'><title>{html.escape(post['title'])}</title>"
            f"<h1>{html.escape(post['title'])}</h1><h3>{html.escape(post['subtitle'])}</h3>{post['html']}", encoding="utf-8")
    (out / "index.html").write_text(kit_page(site, posts), encoding="utf-8")
    return out / "index.html"


def setup_text(site) -> dict:
    author = site["author"]
    return {
        "Publication name": site["name"],
        "One-line description": "Practical guides to AI and data: the exact steps, the prompts to copy, no hype.",
        "About page": (f"I'm {author['name']}, a data and AI leader and educator. My mission is to make data and AI easy for everyone.\n\n"
                       "Every issue is practical: the exact steps, the prompts to copy and the mistakes to skip. "
                       "No hype, no jargon, nothing you can't try the same day.\n\n"
                       f"Guides, a start-here path and a library of prompts live at {site.absolute('')}. "
                       f"More about me at {author['url']}."),
        "Welcome email": ("Welcome, and thank you for subscribing.\n\n"
                          "Here's what you'll get: one practical guide at a time, with the exact steps, the prompts to copy and the mistakes to skip.\n\n"
                          f"New here? Start with the path I'd follow: {site.absolute('start/')}\n\n"
                          "Hit reply and tell me the one thing you'd like AI to do for you at work. I read every answer.\n\n"
                          f"{author['name'].split()[0]}"),
    }


def kit_page(site, posts: list[dict]) -> str:
    data = json.dumps(posts, ensure_ascii=False).replace("</", "<\\/")
    setup = json.dumps(setup_text(site), ensure_ascii=False).replace("</", "<\\/")
    return KIT.replace("__POSTS__", data).replace("__SETUP__", setup).replace("__SUB__", html.escape(site["newsletter"].get("username") or ""))


KIT = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Substack kit</title>
<style>
:root{--bg:#f7f6f2;--card:#fff;--ink:#0d1117;--muted:#5a6370;--line:#e2dfd6;--accent:#07704c;--soft:#dcf2e8}
@media (prefers-color-scheme:dark){:root{--bg:#0b0e13;--card:#12161d;--ink:#eceff3;--muted:#98a2b0;--line:#242b36;--accent:#45d99c;--soft:#0f2a20}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:760px;margin:0 auto;padding:28px 16px 60px}h1{font:700 1.9rem/1.15 Georgia,serif;margin:0 0 6px}
.lead{color:var(--muted);margin:0 0 22px}.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;margin:14px 0}
.card h2{font:700 1.25rem/1.25 Georgia,serif;margin:0 0 4px}.meta{color:var(--muted);font-size:.9rem;margin:0 0 12px}
.row{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}button,a.btn{font:600 .92rem system-ui;border-radius:999px;padding:9px 16px;border:1px solid var(--line);
background:transparent;color:var(--ink);cursor:pointer;text-decoration:none}button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.done{background:var(--soft);color:var(--accent);border-color:var(--accent)}details{margin-top:10px}summary{cursor:pointer;color:var(--muted)}
.preview{border-top:1px solid var(--line);margin-top:10px;padding-top:6px;max-height:420px;overflow:auto;font-size:.95rem}.preview img{max-width:100%;border-radius:8px}
.preview pre{white-space:pre-wrap;background:#0d1117;color:#e6edf3;padding:10px;border-radius:8px}
ol.steps{padding-left:20px}ol.steps li{margin:6px 0}pre.text{white-space:pre-wrap;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:10px;font:inherit}
</style></head><body><main>
<h1>Substack kit</h1>
<p class="lead">Everything to publish on <strong>__SUB__.substack.com</strong>. Best on a computer: Substack's editor on a phone can't take pasted posts.</p>
<div class="card"><h2>Publish a post in two minutes</h2><ol class="steps">
<li>On substack.com, open your dashboard and click <strong>Create new → Text post</strong>.</li>
<li>Below, press <strong>Copy title</strong> and paste it into the title; same for <strong>Copy subtitle</strong>.</li>
<li>Press <strong>Copy post</strong>, click into the body and paste. Images, headings, lists and prompts come with it.</li>
<li>Optional: replace a "Subscribe for free →" line with Substack's own button (Button → Subscribe).</li>
<li>In <strong>Settings</strong> (before publishing) choose a cover image: <strong>Download cover</strong>, then upload it.</li>
<li>Click <strong>Continue → Send to everyone now</strong> (or schedule it).</li></ol></div>
<div class="card"><h2>Set up the publication (once)</h2><p class="meta">Settings → Basics, About page, and Emails → Welcome email.</p><div id="setup"></div></div>
<div id="posts"></div>
</main><script>
const posts = __POSTS__, setup = __SETUP__;
function flash(b, t){const o=b.textContent;b.textContent=t;b.classList.add('done');setTimeout(()=>{b.textContent=o;b.classList.remove('done')},1600)}
async function copyText(b, text){try{await navigator.clipboard.writeText(text);flash(b,'Copied')}catch(e){prompt('Copy this:', text)}}
async function copyHtml(b, html, text){try{await navigator.clipboard.write([new ClipboardItem({'text/html':new Blob([html],{type:'text/html'}),'text/plain':new Blob([text],{type:'text/plain'})})]);flash(b,'Copied')}
 catch(e){const d=document.createElement('div');d.contentEditable=true;d.innerHTML=html;document.body.appendChild(d);const r=document.createRange();r.selectNodeContents(d);const s=getSelection();s.removeAllRanges();s.addRange(r);
 try{document.execCommand('copy');flash(b,'Copied')}catch(_){alert('Select the preview and copy it by hand.')}d.remove()}}
const st = document.getElementById('setup');
for (const [k, v] of Object.entries(setup)) {const w=document.createElement('div');w.innerHTML='<p><strong></strong></p><pre class="text"></pre>';
 w.querySelector('strong').textContent=k;w.querySelector('pre').textContent=v;const b=document.createElement('button');b.textContent='Copy';b.onclick=()=>copyText(b,v);w.appendChild(b);st.appendChild(w)}
const box = document.getElementById('posts');
posts.forEach((p, i) => {const c=document.createElement('div');c.className='card';
 c.innerHTML='<h2></h2><p class="meta"></p><div class="row"></div><details><summary>Preview</summary><div class="preview"></div></details>';
 c.querySelector('h2').textContent=(i+1)+'. '+p.title;c.querySelector('.meta').textContent=(p.kind==='guide'?'Guide':'Issue')+' · '+p.words+' words · '+p.subtitle;
 const row=c.querySelector('.row');const add=(t,f,cls)=>{const b=document.createElement('button');b.textContent=t;if(cls)b.className=cls;b.onclick=()=>f(b);row.appendChild(b)};
 add('Copy title',b=>copyText(b,p.title));add('Copy subtitle',b=>copyText(b,p.subtitle));
 const pv=c.querySelector('.preview');pv.innerHTML=p.html;add('Copy post',b=>copyHtml(b,p.html,pv.innerText),'primary');
 if(p.cover){const a=document.createElement('a');a.className='btn';a.href=p.cover;a.download='';a.target='_blank';a.textContent='Download cover';row.appendChild(a)}
 box.appendChild(c)});
</script></body></html>
"""
