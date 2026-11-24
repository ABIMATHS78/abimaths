"""Draws the two red AI WITH MOJEED ABISIGA logos (wide banner and round) to the sample's proportions.
Run: python brand-logos/make_red_logos.py  (needs playwright)."""
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT = Path(__file__).resolve().parent
FONTS = '<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@100,300;100,800;100,900;112,900&display=swap" rel="stylesheet">'
RED, OFF = "#D02E2E", "#FFFCEF"
CSS = f"""*{{margin:0;box-sizing:border-box}}body{{background:transparent}}
.big{{font-family:'Archivo',sans-serif;font-weight:900;font-stretch:112%;letter-spacing:-.04em;line-height:1;text-transform:uppercase;white-space:nowrap}}
.small{{white-space:nowrap;font-family:'Archivo',sans-serif;line-height:1;text-transform:uppercase;letter-spacing:-.01em}}
.small b{{font-weight:800}} .small i{{font-style:normal;font-weight:300}}"""
# Sample proportions: name cap height = 51% of banner height; "AI WITH" cap height = 38% of the name's; small gap.
BIG = 400  # px font size of the name
wordmark = f"""<div id="logo" style="display:inline-flex;align-items:center;gap:{int(BIG*.1)}px;background:{RED};color:{OFF};
  padding:{int(BIG*0.42)}px {int(BIG*0.42)}px {int(BIG*0.36)}px {int(BIG*0.36)}px">
  <span class="small" style="font-size:{int(BIG*0.38)}px;transform:translateY(2%)"><b>AI</b> <i>WITH</i></span>
  <span class="big" style="font-size:{BIG}px">Mojeed Abisiga</span></div>"""
round_ = f"""<div id="logo" style="width:1200px;height:1200px;background:{RED};color:{OFF};display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px">
  <span class="small" style="font-size:112px;margin-bottom:6px"><b>AI</b> <i>WITH</i></span>
  <span class="big" style="font-size:250px">Mojeed</span><span class="big" style="font-size:250px">Abisiga</span></div>"""
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 6000, "height": 1400})
    pg.set_content(f"<html><head>{FONTS}<style>{CSS}</style></head><body>{wordmark}</body></html>", wait_until="networkidle")
    pg.evaluate("document.fonts.ready.then(()=>1)")
    pg.locator("#logo").screenshot(path=str(OUT / "ai-with-mojeed-abisiga-wordmark.png"))
    print("wordmark", pg.locator("#logo").bounding_box())
    pg = b.new_page(viewport={"width": 1200, "height": 1200})
    pg.set_content(f"<html><head>{FONTS}<style>{CSS}</style></head><body>{round_}</body></html>", wait_until="networkidle")
    pg.evaluate("document.fonts.ready.then(()=>1)")
    # shrink until every letter is inside 84% of the circle
    for _ in range(6):
        d = pg.evaluate("""() => { const c=600; return Math.max(...[...document.querySelectorAll('span')].map(e=>e.getBoundingClientRect())
             .flatMap(q=>[[q.left,q.top],[q.right,q.top],[q.left,q.bottom],[q.right,q.bottom]]).map(([x,y])=>Math.hypot(x-c,y-c)))/600 }""")
        if abs(d - 0.84) < 0.01: break
        pg.evaluate("s => { const l=document.getElementById('logo'); const z=(parseFloat(l.style.zoom)||1)*s; l.style.zoom=z; l.style.width=(1200/z)+'px'; l.style.height=(1200/z)+'px'; }", 0.84 / d)
    print("round fill", round(d, 3))
    pg.screenshot(path=str(OUT / "ai-with-mojeed-abisiga-round.png"), clip={"x": 0, "y": 0, "width": 1200, "height": 1200})
    b.close()
