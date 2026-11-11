"use strict";
/* The Events pages (static/events/*.html), which the app shows in a frame: the countdown, the agenda's days, the
   rest of the speakers, the answers to the questions, and links that move the page the frame is on. */
(() => {
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const frame = window.frameElement, top = frame ? window.parent : null;

  // The frame is as tall as the page, so the page it's on does the scrolling.
  const fit = () => { if (frame) frame.style.height = `${Math.ceil(document.body.getBoundingClientRect().height)}px`; };
  new ResizeObserver(fit).observe(document.body);
  addEventListener("load", fit);
  fit();

  // Countdown to the opening.
  const clock = document.querySelector("[data-countdown]");
  if (clock) {
    const at = new Date(clock.dataset.countdown).getTime();
    const tick = () => {
      const left = Math.max(0, at - Date.now());
      const parts = { d: Math.floor(left / 864e5), h: Math.floor(left / 36e5) % 24, m: Math.floor(left / 6e4) % 60, s: Math.floor(left / 1e3) % 60 };
      for (const el of $$("[data-unit]", clock)) el.textContent = parts[el.dataset.unit];
    };
    tick();
    setInterval(tick, 1000);
  }

  // The agenda: one day at a time.
  const ON = ["bg-foreground", "text-background"], OFF = ["border", "bg-card", "hover:border-foreground/40"];
  for (const tab of $$("[data-day]")) {
    tab.addEventListener("click", () => {
      for (const t of $$("[data-day]")) {
        const on = t === tab;
        t.setAttribute("aria-pressed", String(on));
        t.classList.remove(...(on ? OFF : ON));
        t.classList.add(...(on ? ON : OFF));
      }
      for (const panel of $$("[data-day-panel]")) panel.hidden = panel.dataset.dayPanel !== tab.dataset.day;
    });
  }

  // The rest of the speakers.
  const more = document.querySelector("[data-show-more]");
  if (more) {
    more.addEventListener("click", () => {
      for (const card of $$("[data-more]")) card.hidden = false;
      more.parentElement.hidden = true;
    });
  }

  // Questions and their answers.
  for (const q of $$("[data-faq]")) {
    q.addEventListener("click", () => {
      const open = q.getAttribute("aria-expanded") !== "true";
      q.setAttribute("aria-expanded", String(open));
      const arrow = q.querySelector("svg");
      if (arrow) arrow.classList.toggle("rotate-180", open);
      q.parentElement.querySelector("p").hidden = !open;
    });
  }

  // Links: to a part of this page, scroll the page the frame is on; to a page of the site, open it there.
  document.addEventListener("click", (e) => {
    const a = e.target.closest("a[href]");
    if (!a || e.defaultPrevented || e.button || e.metaKey || e.ctrlKey || e.shiftKey) return;
    const href = a.getAttribute("href");
    if (href.startsWith("#")) {
      const to = document.getElementById(href.slice(1));
      if (!to) return;
      e.preventDefault();
      if (frame) top.scrollTo({ top: top.scrollY + frame.getBoundingClientRect().top + to.getBoundingClientRect().top - 16, behavior: "smooth" });
      else to.scrollIntoView({ behavior: "smooth" });
    } else if (href.startsWith("/") && !a.hasAttribute("download") && top && typeof top.navigate === "function") {
      e.preventDefault();
      top.navigate(href);
      top.scrollTo(0, 0);
    }
  });
})();
