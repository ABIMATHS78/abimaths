import json
from playwright.sync_api import sync_playwright
OUT="../dl/"
def scroll(pg):
    h=pg.evaluate("document.documentElement.scrollHeight")
    for y in range(0,h,600): pg.evaluate(f"window.scrollTo(0,{y})"); pg.wait_for_timeout(200)
    pg.wait_for_timeout(1200)
with sync_playwright() as p:
    br=p.chromium.launch(); ctx=br.new_context(viewport={"width":1440,"height":900})
    pg=ctx.new_page(); pg.goto("https://news.dataglobalhub.org/events/gdai-2026", wait_until="networkidle", timeout=90000); pg.wait_for_timeout(2500); scroll(pg)
    cap={}
    cap["main_collapsed"]=pg.evaluate("document.querySelector('main#main').outerHTML")
    # schedule per day
    days=["Day 1 · Friday, 20 November","Day 2 · Saturday, 21 November","Day 3 · Sunday, 22 November"]
    cap["days"]=[]
    for d in days:
        pg.get_by_role("button",name=d).click(); pg.wait_for_timeout(900)
        sec=pg.get_by_role("button",name=d).locator("xpath=ancestor::section[1]")
        cap["days"].append({"name":d,"ol":sec.locator("ol").first.evaluate("e=>e.outerHTML"),
                            "tabs":pg.get_by_role("button",name=d).evaluate("e=>e.parentElement.outerHTML"),
                            "after":sec.evaluate("e=>[...e.querySelector('ol').parentElement.children].map(c=>c.tagName+':'+c.innerText.slice(0,80))")})
    pg.get_by_role("button",name=days[0]).click(); pg.wait_for_timeout(500)
    # faqs
    for b in pg.locator("button[aria-expanded]").all():
        if b.get_attribute("aria-expanded")=="false" and b.locator("xpath=..").get_attribute("class")=="border-b": b.click(); pg.wait_for_timeout(150)
    cap["faq_open"]=pg.get_by_role("button",name="When is the conference?").evaluate("e=>e.parentElement.parentElement.outerHTML")
    cap["faq_closed_button"]=None
    # speakers
    sec=pg.get_by_role("button",name="View More Speaking Sessions").locator("xpath=ancestor::section[1]")
    cap["speakers_collapsed"]=sec.evaluate("e=>e.outerHTML")
    pg.get_by_role("button",name="View More Speaking Sessions").click(); pg.wait_for_timeout(1500); scroll(pg)
    sec=pg.get_by_text("Meet the Speakers",exact=True).locator("xpath=ancestor::section[1]")
    cap["speakers_expanded"]=sec.evaluate("e=>e.outerHTML")
    cap["asides"]=pg.evaluate("[...document.querySelectorAll('main#main > aside')].map(a=>a.outerHTML)")
    cap["imgs"]=pg.evaluate("[...document.querySelectorAll('main#main img')].map(i=>[i.alt,i.currentSrc||i.src,i.naturalWidth,i.naturalHeight,i.className])")
    pg.screenshot(path=OUT+"gdai-expanded.png", full_page=True)
    # list page
    pg2=ctx.new_page(); pg2.goto("https://news.dataglobalhub.org/events", wait_until="networkidle", timeout=90000); pg2.wait_for_timeout(2500); scroll(pg2)
    cap["list_main"]=pg2.evaluate("document.querySelector('main#main').outerHTML")
    # hover/computed bits for the support widget
    json.dump(cap, open(OUT+"capture.json","w"))
    for k,v in cap.items(): print(k, len(v) if isinstance(v,(str,list)) else v)
    for d in cap["days"]: print(d["name"], len(d["ol"]), d["after"])
    br.close()
