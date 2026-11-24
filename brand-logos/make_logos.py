"""Draws the four logos with headless Chromium (Archivo Expanded Black + Instrument Serif Italic, from Google Fonts).
Draws the cream logos; make_red_logos.py draws the red ones.
Run: python brand-logos/make_logos.py  (needs playwright). Round logos keep all text inside 84% of the circle."""
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT = Path(__file__).resolve().parent
FONTS = '<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@125,900&family=Instrument+Serif:ital@1&display=swap" rel="stylesheet">'
BASE = """*{margin:0;box-sizing:border-box}html,body{background:transparent}
.heavy{font-family:'Archivo',sans-serif;font-weight:900;font-stretch:125%;letter-spacing:-.035em;line-height:.8;text-transform:uppercase}
.serif{font-family:'Instrument Serif',serif;font-style:italic;font-weight:400;letter-spacing:-.01em}"""
CREAM, INK, RED, OFF = "#FFF8E5", "#2E2B2C", "#D63A3A", "#FFF4E8"
logos = {
 # 1. wide wordmark, cream
 "learn-ai-with-mojeed-wordmark": (3000, 560, f"""<div style="width:3000px;height:560px;background:{CREAM};display:flex;align-items:center;justify-content:center;gap:56px;color:{INK}">
   <span class="heavy" style="font-size:300px">Learn AI</span><span class="serif" style="font-size:330px;line-height:.7;transform:translateY(12px)">With</span><span class="heavy" style="font-size:300px">Mojeed</span></div>"""),
 # 1b. wide wordmark, cream: AI With MOJEED ABISIGA
 "ai-with-mojeed-abisiga-cream-wordmark": (3600, 560, f"""<div style="width:3600px;height:560px;background:{CREAM};display:flex;align-items:center;justify-content:center;gap:56px;color:{INK}">
   <span class="heavy" style="font-size:300px">AI</span><span class="serif" style="font-size:330px;line-height:.7;transform:translateY(12px)">With</span><span class="heavy" style="font-size:300px">Mojeed Abisiga</span></div>"""),
 # 1c. round, cream
 "ai-with-mojeed-abisiga-cream-round": (1200, 1200, f"""<div style="width:1200px;height:1200px;background:{CREAM};display:flex;flex-direction:column;align-items:center;justify-content:center;color:{INK};gap:22px">
   <span style="display:flex;align-items:center;gap:30px"><span class="heavy" style="font-size:190px">AI</span><span class="serif" style="font-size:230px;line-height:.75">With</span></span><span class="heavy" style="font-size:210px">Mojeed</span><span class="heavy" style="font-size:210px">Abisiga</span></div>"""),
 # 3. round: cream
 "learn-ai-with-mojeed-round": (1200, 1200, f"""<div style="width:1200px;height:1200px;background:{CREAM};display:flex;flex-direction:column;align-items:center;justify-content:center;color:{INK};gap:26px">
   <span class="heavy" style="font-size:190px">Learn AI</span><span class="serif" style="font-size:230px;line-height:.75">With</span><span class="heavy" style="font-size:210px">Mojeed</span></div>"""),
}
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, (w, h, body) in logos.items():
        pg = b.new_page(viewport={"width": w, "height": h})
        pg.set_content(f"<html><head>{FONTS}<style>{BASE}</style></head><body>{body}</body></html>", wait_until="networkidle")
        pg.evaluate("document.fonts.ready.then(()=>1)")
        pg.evaluate("""(round) => {
          const box = document.body.firstElementChild; const inner = box; const spans=[...box.querySelectorAll('span')];
          const W = box.clientWidth, H = box.clientHeight, cx = W/2, cy = H/2;
          let s = 1;
          for (let i = 0; i < 6; i++) {
            const rs = spans.map(e => e.getBoundingClientRect());
            let need;
            if (round) { const d = Math.max(...rs.flatMap(q => [[q.left,q.top],[q.right,q.top],[q.left,q.bottom],[q.right,q.bottom]].map(([x,y]) => Math.hypot(x-cx,y-cy)))); need = (0.84 * W/2) / d; }
            else { const minX = Math.min(...rs.map(q=>q.left)), maxX = Math.max(...rs.map(q=>q.right)); const minY=Math.min(...rs.map(q=>q.top)), maxY=Math.max(...rs.map(q=>q.bottom));
                   need = Math.min((W*0.9)/(maxX-minX), (H*0.72)/(maxY-minY)); }
            s *= need; box.style.zoom = s;
            box.style.width = (W / s) + 'px'; box.style.height = (H / s) + 'px';
          }
        }""", "round" in name)
        fits = pg.evaluate("""() => { const r=[...document.querySelectorAll('span')].map(s=>s.getBoundingClientRect());
            const box=document.body.firstElementChild.getBoundingClientRect(); const cx=box.width/2, cy=box.height/2, R=box.width/2;
            const corners=r.flatMap(q=>[[q.left,q.top],[q.right,q.top],[q.left,q.bottom],[q.right,q.bottom]]);
            return {minX:Math.min(...r.map(q=>q.left)), maxX:Math.max(...r.map(q=>q.right)), maxCornerDist: Math.max(...corners.map(([x,y])=>Math.hypot(x-cx,y-cy)))/R}; }""")
        pg.screenshot(path=str(OUT / f"{name}.png"), clip={"x": 0, "y": 0, "width": w, "height": h})
        print(name, {k: round(v, 2) for k, v in fits.items()})
        pg.close()
    b.close()
