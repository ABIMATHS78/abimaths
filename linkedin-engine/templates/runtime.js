// Runs inside the page before it is captured:
//   1. waits for the bundled fonts,
//   2. scales the text in every [data-fit] region so it fills the space without overflowing,
//   3. lets the template draw its connectors (window.__layout),
//   4. in motion renders, schedules every [data-anim] element on one timeline that the renderer seeks frame by frame.
(function () {
  const root = document.documentElement;

  // Where the region's content really ends: the bottom of its child boxes. scrollHeight would also count
  // the descenders that display type with a tight line-height hangs below its last line, at every size.
  function contentBottom(el) {
    let bottom = -Infinity;
    for (const child of el.children) {
      const style = getComputedStyle(child);
      if (style.position === "absolute" || style.display === "none") continue;
      const r = child.getBoundingClientRect();
      bottom = Math.max(bottom, r.bottom + (parseFloat(style.marginBottom) || 0));
    }
    return bottom;
  }

  function fits(el) {
    const style = getComputedStyle(el);
    const box = el.getBoundingClientRect();
    const limit = box.bottom - (parseFloat(style.paddingBottom) || 0) + 1;
    const bottom = contentBottom(el);
    const vertical = bottom === -Infinity ? el.scrollHeight <= el.clientHeight + 1 : bottom <= limit;
    const slackX = 1 + parseFloat(style.fontSize) * 0.05;
    return vertical && el.scrollWidth <= el.clientWidth + slackX;
  }

  // For regions laid out with absolute positions (cycles, mind maps): every [data-box] must stay inside
  // the region and keep a gap from every other box.
  function boxesFit(el) {
    const box = el.getBoundingClientRect();
    const pad = parseFloat(el.dataset.boxPad || 0);
    const R = { left: box.left + pad, right: box.right - pad, top: box.top + pad, bottom: box.bottom - pad };
    const gap = parseFloat(el.dataset.boxGap || 10);
    const boxes = Array.from(el.querySelectorAll("[data-box]")).map((b) => b.getBoundingClientRect());
    for (const b of boxes) {
      if (b.left < R.left - 1 || b.right > R.right + 1 || b.top < R.top - 1 || b.bottom > R.bottom + 1) return false;
    }
    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i], b = boxes[j];
        if (a.left < b.right + gap && b.left < a.right + gap && a.top < b.bottom + gap && b.top < a.bottom + gap) return false;
      }
    }
    return true;
  }

  function fitRegion(el) {
    const check = el.dataset.fitMode === "boxes" ? boxesFit : fits;
    if (!el.dataset.basePx) el.dataset.basePx = parseFloat(getComputedStyle(el).fontSize);
    const base = parseFloat(el.dataset.basePx);
    let lo = parseFloat(el.dataset.fitMin || 0.5);
    let hi = parseFloat(el.dataset.fitMax || 1.25);
    el.style.fontSize = base * hi + "px";
    if (check(el)) return { scale: hi, ok: true };
    el.style.fontSize = base * lo + "px";
    if (!check(el)) return { scale: lo, ok: false };
    for (let i = 0; i < 16; i++) {
      const mid = (lo + hi) / 2;
      el.style.fontSize = base * mid + "px";
      if (check(el)) lo = mid; else hi = mid;
    }
    el.style.fontSize = base * lo + "px";
    return { scale: Math.round(lo * 1000) / 1000, ok: true };
  }

  function fitAll() {
    const results = [];
    document.querySelectorAll("[data-fit]").forEach((el) => results.push({ region: el.dataset.fit || el.className, ...fitRegion(el) }));
    return results;
  }

  function overflowProblems() {
    const canvas = document.getElementById("canvas").getBoundingClientRect();
    const problems = [];
    document.querySelectorAll("[data-guard]").forEach((el) => {
      const r = el.getBoundingClientRect();
      if (r.width && (r.right > canvas.right + 2 || r.bottom > canvas.bottom + 2 || r.left < canvas.left - 2 || r.top < canvas.top - 2)) {
        problems.push((el.dataset.guard || el.className) + " runs off the canvas");
      }
    });
    return problems;
  }

  function planMotion() {
    const cfg = window.__motionConfig || {};
    const els = Array.from(document.querySelectorAll("[data-anim]"));
    els.forEach((el, i) => { el.__i = i; el.__order = el.dataset.order !== undefined ? parseFloat(el.dataset.order) : i; });
    els.sort((a, b) => a.__order - b.__order || a.__i - b.__i);
    const timed = els.filter((el) => el.dataset.at === undefined);
    const weights = timed.reduce((sum, el) => sum + parseFloat(el.dataset.weight || 1), 0);
    const build = cfg.build ?? Math.min(cfg.maxBuild ?? 9, Math.max(cfg.minBuild ?? 3.5, weights * (cfg.perItem ?? 0.2)));
    const step = weights > 1 ? build / (weights - 1) : 0;
    let t = cfg.start ?? 0.3;
    let end = 0;
    for (const el of els) {
      const dur = parseFloat(el.dataset.dur || cfg.dur || 0.6);
      const at = el.dataset.at !== undefined ? parseFloat(el.dataset.at) : t;
      el.style.animationDelay = at + "s";
      el.style.animationDuration = dur + "s";
      el.classList.add("a-" + el.dataset.anim);
      if (el.dataset.span === undefined) end = Math.max(end, at + dur);
      if (el.dataset.at === undefined) t += (el.dataset.step !== undefined ? parseFloat(el.dataset.step) : step) * parseFloat(el.dataset.weight || 1);
    }
    // [data-span] elements (a path that draws under the whole sequence) last until the sequence ends.
    els.filter((el) => el.dataset.span !== undefined).forEach((el) => {
      const at = parseFloat(el.style.animationDelay);
      el.style.animationDuration = Math.max(0.6, end - at) + "s";
    });
    const hold = cfg.hold ?? 2.6;
    return { build_end: Math.round(end * 100) / 100, duration: Math.round((cfg.duration ?? end + hold) * 100) / 100 };
  }

  // Split items into k columns in reading order so the tallest column is as short as possible.
  window.__partitionColumns = function (wrap, items, k, gapPx) {
    k = Math.max(1, Math.min(k, items.length));
    const width = (wrap.clientWidth - gapPx * (k - 1)) / k;
    items.forEach((el) => { el.style.width = width + "px"; });
    const h = items.map((el) => el.getBoundingClientRect().height + gapPx);
    items.forEach((el) => { el.style.width = ""; });
    const n = h.length;
    const pre = [0];
    h.forEach((v) => pre.push(pre[pre.length - 1] + v));
    const best = Array.from({ length: k + 1 }, () => Array(n + 1).fill(Infinity));
    const cut = Array.from({ length: k + 1 }, () => Array(n + 1).fill(0));
    best[0][0] = 0;
    for (let c = 1; c <= k; c++) {
      for (let i = 1; i <= n; i++) {
        for (let j = c - 1; j < i; j++) {
          const cost = Math.max(best[c - 1][j], pre[i] - pre[j]);
          if (cost < best[c][i]) { best[c][i] = cost; cut[c][i] = j; }
        }
      }
    }
    const bounds = [];
    let i = n;
    for (let c = k; c >= 1; c--) { const j = cut[c][i]; bounds.unshift([j, i]); i = j; }
    wrap.innerHTML = "";
    bounds.forEach(([a, b]) => {
      const col = document.createElement("div");
      col.className = "col";
      items.slice(a, b).forEach((el) => col.appendChild(el));
      wrap.appendChild(col);
    });
  };

  window.__seek = function (seconds) {
    document.getAnimations().forEach((a) => { a.pause(); a.currentTime = seconds * 1000; });
  };

  async function boot() {
    await Promise.all(Array.from(document.fonts).map((f) => f.load().catch(() => null)));
    await document.fonts.ready;
    if (window.__beforeFit) window.__beforeFit();
    const fit = fitAll();
    if (window.__layout) window.__layout();
    const timeline = root.classList.contains("motion") ? planMotion() : null;
    window.__report = { fit, problems: overflowProblems().concat(fit.filter((f) => !f.ok).map((f) => "text still overflows in " + f.region + " at the smallest size")), timeline };
    if (timeline) window.__seek(0);
    window.__ready = true;
  }

  window.__boot = boot;
  if (document.readyState === "complete") boot(); else window.addEventListener("load", boot);
})();
