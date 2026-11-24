"""LinkedIn banners in the LEARN AI With MOJEED ABISIGA style: personal profile (1584x396) and company page (1128x191),
drawn at 2x for sharpness. All text sits in the right-hand safe area: the profile photo or company logo covers the
bottom-left, and phones trim the sides. Run: python brand-logos/make_banners.py (needs playwright)."""
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
FONTS = '<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@100,500;100,700;125,900&family=Instrument+Serif:ital@1&display=swap" rel="stylesheet">'
CREAM, INK, RED, MUTED = "#FFF8E5", "#2E2B2C", "#D02E2E", "#6B625C"
CSS = f"""*{{margin:0;box-sizing:border-box}}
.b{{position:relative;overflow:hidden;background:{CREAM};color:{INK};font-family:'Archivo',sans-serif}}
.heavy{{font-weight:900;font-stretch:125%;letter-spacing:-.035em;line-height:.85;text-transform:uppercase;white-space:nowrap}}
.serif{{font-family:'Instrument Serif',serif;font-style:italic;font-weight:400;line-height:.7}}
.stripe{{position:absolute;left:0;top:0;bottom:0;background:{RED}}}
.dots{{position:absolute;inset:0;background-image:radial-gradient(rgba(46,43,44,.10) 1.6px,transparent 1.8px);background-size:22px 22px;
  -webkit-mask-image:linear-gradient(90deg,#000 0%,rgba(0,0,0,.5) 30%,transparent 55%)}}
.safe{{position:absolute;display:flex;flex-direction:column;justify-content:center;align-items:flex-start}}
.mark{{display:flex;align-items:center;gap:.22em}}
.tag{{font-weight:500;color:{INK}}}
.links{{font-weight:700;color:{RED};letter-spacing:.01em;white-space:nowrap}}
.links span{{color:{MUTED};font-weight:500;padding:0 .5em}}"""

def banner(w, h, size, tag, links, left_pct, pad_pct, stripe_px):
    return f"""<div id="b" class="b" style="width:{w}px;height:{h}px"><div class="dots"></div><div class="stripe" style="width:{stripe_px}px"></div>
  <div class="safe" style="left:{left_pct}%;right:{pad_pct}%;top:12%;bottom:12%;gap:{size*0.28}px">
    <div class="mark" style="font-size:{size}px"><span class="heavy">Learn AI</span><span class="serif" style="font-size:1.12em">With</span><span class="heavy">Mojeed Abisiga</span></div>
    <div class="tag" style="font-size:{size*0.34}px">Data &amp; AI, made easy for everyone.</div>
    <div class="links" style="font-size:{size*0.3}px">{links}</div></div></div>"""

BANNERS = {
  "linkedin-banner-profile": (1584, 396, banner(1584, 396, 74, "", "mojeedabisiga.com/learn", 33, 13.5, 14)),
  "linkedin-banner-company": (1128, 191, banner(1128, 191, 44, "", "mojeedabisiga.com/learn", 29, 13.5, 10)),
}

with sync_playwright() as p:
    b = p.chromium.launch()
    for name, (w, h, html) in BANNERS.items():
        pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
        pg.set_content(f"<html><head>{FONTS}<style>{CSS}</style></head><body>{html}</body></html>", wait_until="networkidle")
        pg.evaluate("document.fonts.ready.then(()=>1)")
        # shrink the text block until it fits its safe area
        for _ in range(5):
            over = pg.evaluate("""() => { const s=document.querySelector('.safe'); const r=s.getBoundingClientRect();
               const kids=[...s.children].map(e=>e.getBoundingClientRect()); const maxR=Math.max(...kids.map(k=>k.left+k.scrollWidth||k.right));
               const right=Math.max(...[...s.querySelectorAll('*')].map(e=>e.getBoundingClientRect().right)); return right/r.right; }""")
            if over <= 1.0: break
            pg.evaluate("f => { const s=document.querySelector('.safe'); s.style.zoom=(parseFloat(s.style.zoom)||1)/f; }", over * 1.02)
        box = pg.evaluate("() => { const r=[...document.querySelectorAll('.safe *')].map(e=>e.getBoundingClientRect()); return [Math.min(...r.map(q=>q.left)),Math.max(...r.map(q=>q.right)),Math.min(...r.map(q=>q.top)),Math.max(...r.map(q=>q.bottom))]; }")
        print(name, f"{w*2}x{h*2}", "text box x %.0f-%.0f y %.0f-%.0f of %dx%d" % (*box, w, h))
        pg.locator("#b").screenshot(path=str(OUT / f"{name}.png"))
        pg.close()
    b.close()
