"use strict";

/* Infographics for the education guides, in the style aieverytime.org illustrates them with: a title band, then
   the guide's sections as cards with icons, its steps as a numbered flow, what to avoid against what to do, or its
   layers as a stack, and its key takeaway at the foot.

   scripts/import_guides.py describes each one from the guide's own headings, steps and takeaways. Drawn here as
   text rather than shipped as pictures, every word is spelled as in the guide, it's sharp at any size, and a guide
   adds a couple of kilobytes, not a hundred. It's a visual summary of text the page already has, so screen
   readers skip it. */

const IG_ICONS = {
  layers: '<path d="M12 3 3 7.5 12 12l9-4.5L12 3Z"/><path d="m3 12 9 4.5 9-4.5"/><path d="m3 16.5 9 4.5 9-4.5"/>',
  wrench: '<circle cx="12" cy="12" r="3"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1"/>',
  rocket: '<path d="M12 2.5c3 2 5 5.8 5 10l-2 3.5H9l-2-3.5c0-4.2 2-8 5-10Z"/><circle cx="12" cy="10" r="1.8"/><path d="m9 16-2.5 4.5L10 19M15 16l2.5 4.5L14 19"/>',
  chart: '<path d="M3.5 3.5v17h17"/><path d="m7.5 15 4-4 3 3 5.5-6.5"/>',
  book: '<path d="M4.5 4.5A2.5 2.5 0 0 1 7 2h12.5v17H7a2.5 2.5 0 0 0-2.5 2.5Z"/><path d="M4.5 19.5A2.5 2.5 0 0 1 7 17h12.5"/>',
  brain: '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/><path d="M9.5 2.5V6M14.5 2.5V6M9.5 18v3.5M14.5 18v3.5M2.5 9.5H6M2.5 14.5H6M18 9.5h3.5M18 14.5h3.5"/>',
  toolbox: '<rect x="3" y="8" width="18" height="12" rx="2"/><path d="M8.5 8V5.5a1.5 1.5 0 0 1 1.5-1.5h4a1.5 1.5 0 0 1 1.5 1.5V8M3 13h18M10 13v2.5h4V13"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5"/>',
  code: '<path d="m8 7-5 5 5 5M16 7l5 5-5 5M13.5 4.5l-3 15"/>',
  chat: '<path d="M20.5 12a8.5 8.5 0 0 1-12.3 7.6L3.5 21l1.4-4.6A8.5 8.5 0 1 1 20.5 12Z"/>',
  gauge: '<path d="M3.6 17a9 9 0 1 1 16.8 0"/><path d="m12 13.5 4-5"/><circle cx="12" cy="13.5" r="1.4"/>',
  database: '<ellipse cx="12" cy="5.5" rx="8" ry="3"/><path d="M4 5.5v13c0 1.7 3.6 3 8 3s8-1.3 8-3v-13M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  shield: '<path d="M12 2.5 4.5 5.5v5.8c0 4.8 3.2 8.9 7.5 10.2 4.3-1.3 7.5-5.4 7.5-10.2V5.5Z"/><path d="m9 12 2.2 2.2L15.5 10"/>',
  users: '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20.5a6.5 6.5 0 0 1 13 0M15.5 4.6a3.5 3.5 0 0 1 0 6.8M18 14.2a6.5 6.5 0 0 1 3.5 6.3"/>',
  palette: '<path d="M12 3a9 9 0 1 0 0 18c1.2 0 1.7-1 1.3-1.9-.5-1 .2-2.1 1.4-2.1H17a4 4 0 0 0 4-4C21 7.6 17 3 12 3Z"/><circle cx="7.5" cy="11" r="1.1"/><circle cx="10" cy="7" r="1.1"/><circle cx="15" cy="7.4" r="1.1"/>',
  coins: '<ellipse cx="9" cy="7" rx="6" ry="3"/><path d="M3 7v4c0 1.7 2.7 3 6 3s6-1.3 6-3V7M9 14v3c0 1.7 2.7 3 6 3s6-1.3 6-3v-4c0-1.5-2.2-2.8-5.2-3"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.2 2"/>',
  alert: '<path d="M10.3 4 2.6 17.5A2 2 0 0 0 4.3 20.5h15.4a2 2 0 0 0 1.7-3L13.7 4a2 2 0 0 0-3.4 0Z"/><path d="M12 9.5v4M12 17h.01"/>',
  check: '<path d="m5.5 12.5 4 4L18.5 7.5"/>',
  x: '<path d="m7 7 10 10M17 7 7 17"/>',
  arrow: '<path d="M3.5 12h15M13 6l6 6-6 6"/>',
};
const IG_KINDS = { tutorial: "Tutorial", cheat_sheet: "Cheat sheet" };

const igEsc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const igSvg = (name, cls = "ig-i") => `<svg class="${cls}" viewBox="0 0 24 24">${IG_ICONS[name] || IG_ICONS.target}</svg>`;
const igList = (items, mark) => `<ul>${items.map((x) => `<li>${mark ? igSvg(mark, `ig-m ig-${mark}`) : ""}<span>${igEsc(x)}</span></li>`).join("")}</ul>`;
const igCard = (s) => `<section class="ig-card"><h4>${igSvg(s.icon)}<span>${igEsc(s.title)}</span></h4>${igList(s.items)}</section>`;

/* The infographic for a guide: g as the import describes it, meta the guide's kind, level and reading time. */
function infographic(g, meta = {}) {
  if (!g || !g.title) return "";
  const chips = [IG_KINDS[meta.kind], meta.level, meta.minutes ? `${meta.minutes} min read` : null].filter(Boolean);
  let main = "";
  if (g.layout === "flow" && g.steps?.length) {
    const n = g.steps.length;
    main = `<ol class="ig-flow" style="--n:${n > 4 ? Math.ceil(n / 2) : n}">${g.steps.map((s) => `<li><span>${igEsc(s)}</span></li>`).join("")}</ol>`;
  } else if (g.layout === "layers" && g.layers?.length) {
    const n = g.layers.length;
    // The foundation at the foot, widest and strongest; each layer up builds on the one below.
    main = `<div class="ig-layers">${g.layers.map((text, i) => ({ text, i })).reverse().map(({ text, i }) =>
      `<div class="${i < n / 2 ? "deep" : ""}" style="--i:${i};--p:${Math.round(92 - (i * 52) / Math.max(1, n - 1))}%"><b>${i + 1}</b><span>${igEsc(text)}</span></div>`).join("")}</div>`;
  } else if (g.layout === "versus" && g.avoid && g.do) {
    main = `<div class="ig-vs">
      <section class="ig-bad"><h4>${igEsc(g.avoid.title)}</h4>${igList(g.avoid.items, "x")}</section>
      ${igSvg("arrow", "ig-arrow")}
      <section class="ig-good"><h4>${igEsc(g.do.title)}</h4>${igList(g.do.items, "check")}</section>
    </div>`;
  }
  const sections = g.sections || [];
  const cards = sections.length
    ? `<div class="${g.layout === "cheatsheet" ? "ig-grid" : "ig-cols"}" style="--n:${sections.length}">${sections.map(igCard).join("")}</div>` : "";
  return `<figure class="ig ig-${igEsc(g.palette)} ig-l-${igEsc(g.layout)}" aria-hidden="true"><div class="ig-in">
    <header class="ig-head">
      ${chips.length ? `<div class="ig-chips">${chips.map((c) => `<span>${igEsc(c)}</span>`).join("")}</div>` : ""}
      <div class="ig-title">${igEsc(g.title)}</div>
      ${g.subtitle ? `<div class="ig-sub">${igEsc(g.subtitle)}</div>` : ""}
    </header>
    ${main}${cards}
    ${g.banner ? `<div class="ig-banner">${igEsc(g.banner)}</div>` : ""}
    <div class="ig-mark"><img src="/static/icons/logo-256.webp" alt="" width="16" height="16"><span>AI <b>Every</b> Time</span></div>
  </div></figure>`;
}
