/* The publication's script: no framework, no tracking. Everything a reader saves (theme, checklists, finished
   lessons) stays in their own browser. */
(function () {
  "use strict";
  var root = document.documentElement;
  var store = {
    get: function (key, fallback) { try { var v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch (e) { return fallback; } },
    set: function (key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* private mode: keep working */ } }
  };

  /* Theme: the reader's choice, else the system's. */
  var themeBtns = document.querySelectorAll(".theme-toggle");   /* one in the header, one in the phone menu */
  function applyTheme(theme) {
    if (theme) root.setAttribute("data-theme", theme); else root.removeAttribute("data-theme");
    themeBtns.forEach(function (btn) { btn.setAttribute("aria-pressed", String(root.getAttribute("data-theme") === "dark")); });
  }
  applyTheme(store.get("theme", null));
  themeBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      var dark = root.getAttribute("data-theme") === "dark" ||
        (!root.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);
      var next = dark ? "light" : "dark";
      store.set("theme", next); applyTheme(next);
    });
  });

  /* Mobile menu. */
  var menuBtn = document.querySelector(".menu-toggle");
  if (menuBtn) menuBtn.addEventListener("click", function () {
    var open = document.body.classList.toggle("nav-open");
    menuBtn.setAttribute("aria-expanded", String(open));
    menuBtn.setAttribute("aria-label", open ? "Close menu" : "Open menu");
  });

  /* A small message at the bottom of the screen. */
  var toastEl;
  function toast(text) {
    if (!toastEl) { toastEl = document.createElement("div"); toastEl.className = "toast"; toastEl.setAttribute("role", "status"); document.body.appendChild(toastEl); }
    toastEl.textContent = text; toastEl.classList.add("show");
    clearTimeout(toastEl._t); toastEl._t = setTimeout(function () { toastEl.classList.remove("show"); }, 1800);
  }
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return new Promise(function (resolve, reject) {
      var area = document.createElement("textarea"); area.value = text; area.setAttribute("readonly", "");
      area.style.position = "fixed"; area.style.opacity = "0"; document.body.appendChild(area); area.select();
      try { document.execCommand("copy") ? resolve() : reject(); } catch (e) { reject(e); } finally { area.remove(); }
    });
  }

  /* Copy buttons on prompts and code. */
  document.addEventListener("click", function (event) {
    var btn = event.target.closest("[data-copy]");
    if (!btn) return;
    var box = btn.closest(".prompt, .codeblock, .lib-card");
    var code = box && box.querySelector("pre code");
    if (!code) return;
    copyText(code.textContent).then(function () {
      btn.classList.add("copied"); var label = btn.textContent; btn.textContent = "Copied";
      setTimeout(function () { btn.classList.remove("copied"); btn.textContent = label; }, 1600);
    }, function () { toast("Select the text and copy it by hand"); });
  });

  /* Share: the phone's share sheet when there is one; "copy link" everywhere. */
  document.addEventListener("click", function (event) {
    var btn = event.target.closest("[data-share]");
    if (!btn) return;
    var url = btn.getAttribute("data-url") || location.href;
    if (btn.getAttribute("data-share") === "native" && navigator.share) {
      navigator.share({ title: document.title, url: url }).catch(function () {});
      return;
    }
    copyText(url).then(function () { toast("Link copied"); }, function () { toast(url); });
  });

  /* Checklists keep their ticks in this browser. */
  var page = document.body.getAttribute("data-page") || location.pathname;
  var lists = document.querySelectorAll(".prose .task-list-item input[type=checkbox]");
  if (lists.length) {
    var saved = store.get("checks:" + page, {});
    lists.forEach(function (box, i) {
      if (saved[i] !== undefined) box.checked = saved[i];
      box.closest(".task-list-item").classList.toggle("done", box.checked);
      box.addEventListener("change", function () {
        saved[i] = box.checked; store.set("checks:" + page, saved);
        box.closest(".task-list-item").classList.toggle("done", box.checked);
      });
    });
  }

  /* Reading progress and the contents list that follows the reader. */
  var article = document.querySelector("[data-progress]");
  var bar = document.querySelector(".progress");
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll(".toc a[href^='#']"));
  var targets = tocLinks.map(function (a) { return document.getElementById(decodeURIComponent(a.getAttribute("href").slice(1))); });
  var ticking = false;
  function onScroll() {
    ticking = false;
    if (article && bar) {
      var rect = article.getBoundingClientRect();
      var total = rect.height - window.innerHeight * 0.6;
      var done = Math.min(1, Math.max(0, -rect.top / Math.max(1, total)));
      bar.style.width = (done * 100).toFixed(1) + "%";
    }
    if (tocLinks.length) {
      var current = -1;
      targets.forEach(function (el, i) { if (el && el.getBoundingClientRect().top < 140) current = i; });
      tocLinks.forEach(function (a, i) { a.classList.toggle("active", i === current); });
    }
  }
  if (article || tocLinks.length) {
    window.addEventListener("scroll", function () { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });
    onScroll();
  }

  /* Finished lessons ("Mark as done") and the learning path's progress. */
  var doneSet = store.get("done", []);
  function isDone(ref) { return doneSet.indexOf(ref) !== -1; }
  document.querySelectorAll(".done-toggle").forEach(function (btn) {
    var ref = btn.getAttribute("data-ref");
    function paint() { btn.setAttribute("aria-pressed", String(isDone(ref))); btn.querySelector("span").textContent = isDone(ref) ? "Done" : "Mark as done"; }
    paint();
    btn.addEventListener("click", function () {
      if (isDone(ref)) doneSet = doneSet.filter(function (r) { return r !== ref; }); else doneSet.push(ref);
      store.set("done", doneSet); paint();
    });
  });
  var lessons = document.querySelectorAll(".lesson[data-ref]");
  if (lessons.length) {
    var finished = 0;
    lessons.forEach(function (row) { var d = isDone(row.getAttribute("data-ref")); row.classList.toggle("done", d); if (d) finished++; });
    var fill = document.querySelector(".path-progress .bar span");
    var label = document.querySelector(".path-progress .count");
    if (fill) fill.style.width = (finished / lessons.length * 100) + "%";
    if (label) label.textContent = finished + " of " + lessons.length + " done";
  }

  /* Search and filters on the archive, the guides and the library. */
  var searchBox = document.querySelector("[data-search]");
  var rows = Array.prototype.slice.call(document.querySelectorAll("[data-item]"));
  var chips = Array.prototype.slice.call(document.querySelectorAll("[data-filter]"));
  var empty = document.querySelector("[data-empty]");
  var index = null, activeFilter = "", loading = null;
  function words(text) { return (text || "").toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").split(/[^a-z0-9]+/).filter(Boolean); }
  function loadIndex() {
    if (index || loading || !searchBox || !searchBox.getAttribute("data-index")) return loading;
    loading = fetch(searchBox.getAttribute("data-index")).then(function (r) { return r.json(); }).then(function (data) {
      index = {}; data.forEach(function (entry) { index[entry.ref] = (entry.title + " " + entry.subtitle + " " + entry.tags.join(" ") + " " + entry.text).toLowerCase(); });
      apply();
    }).catch(function () { index = {}; });
    return loading;
  }
  function apply() {
    var query = words(searchBox ? searchBox.value : "");
    var shown = 0;
    rows.forEach(function (row) {
      var hay = (index && index[row.getAttribute("data-item")]) || row.textContent.toLowerCase();
      var tags = (row.getAttribute("data-tags") || "").split(" ");
      var ok = (!activeFilter || tags.indexOf(activeFilter) !== -1) && query.every(function (w) { return hay.indexOf(w) !== -1; });
      row.hidden = !ok; if (ok) shown++;
    });
    if (empty) empty.hidden = shown !== 0;
  }
  if (searchBox) {
    searchBox.addEventListener("input", function () { loadIndex(); apply(); });
    searchBox.addEventListener("focus", loadIndex, { once: true });
    var q = new URLSearchParams(location.search).get("q");
    if (q) { searchBox.value = q; loadIndex(); apply(); }
  }
  chips.forEach(function (chip) {
    chip.addEventListener("click", function () {
      var value = chip.getAttribute("data-filter");
      activeFilter = activeFilter === value ? "" : value;
      chips.forEach(function (c) { c.setAttribute("aria-pressed", String(c.getAttribute("data-filter") === activeFilter && activeFilter !== "")); });
      apply();
    });
  });

  document.querySelectorAll("[data-year]").forEach(function (el) { el.textContent = String(new Date().getFullYear()); });
})();
