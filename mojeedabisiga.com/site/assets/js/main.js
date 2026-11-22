/* mojeedabisiga.com: small progressive enhancements. The site works fully without JavaScript. */
(function () {
  var root = document.documentElement;
  root.classList.add("js");

  /* Theme toggle (the <head> snippet already applied the saved/system theme) */
  var toggle = document.querySelector(".theme-toggle");
  if (toggle) {
    var sync = function () {
      var dark = root.getAttribute("data-theme") === "dark";
      toggle.setAttribute("aria-pressed", dark ? "true" : "false");
      toggle.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    };
    sync();
    toggle.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) { /* storage unavailable: theme still switches for this visit */ }
      sync();
    });
  }

  /* Mobile navigation */
  var header = document.querySelector(".site-header");
  var menuBtn = document.querySelector(".menu-toggle");
  if (header && menuBtn) {
    var setOpen = function (open) {
      header.classList.toggle("nav-open", open);
      menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
    };
    menuBtn.addEventListener("click", function () { setOpen(!header.classList.contains("nav-open")); });
    header.querySelectorAll(".nav a").forEach(function (a) { a.addEventListener("click", function () { setOpen(false); }); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") setOpen(false); });
  }

  /* Copy-to-clipboard buttons (press kit bios) */
  document.querySelectorAll("[data-copy]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var target = document.getElementById(btn.getAttribute("data-copy"));
      if (!target) return;
      var text = target.innerText.trim();
      var label = btn.querySelector("span");
      var done = function () {
        if (!label) return;
        var old = label.textContent;
        label.textContent = "Copied";
        setTimeout(function () { label.textContent = old; }, 1600);
      };
      var selectIt = function () {
        var range = document.createRange();
        range.selectNodeContents(target);
        var sel = window.getSelection();
        sel.removeAllRanges(); sel.addRange(range);
        if (label) { var old = label.textContent; label.textContent = "Selected: press Ctrl/Cmd+C"; setTimeout(function () { label.textContent = old; }, 2600); }
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, selectIt);
      } else {
        var ta = document.createElement("textarea");
        ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "absolute"; ta.style.left = "-9999px";
        document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy") ? done() : selectIt(); } catch (e) { selectIt(); }
        document.body.removeChild(ta);
      }
    });
  });

  /* Current year in the footer */
  document.querySelectorAll("[data-year]").forEach(function (el) { el.textContent = new Date().getFullYear(); });
})();
