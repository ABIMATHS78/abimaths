"use strict";

/* ---------------------------------------------------------------- utilities */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const app = $("#app");
const state = { me: null, options: null, nav: 0 };

/* Thrown when a response arrives after the user has navigated elsewhere; the old view must not render. */
class StaleView extends Error {}
window.addEventListener("unhandledrejection", (e) => { if (e.reason instanceof StaleView) e.preventDefault(); });

async function api(path, { method = "GET", body } = {}) {
  const nav = state.nav;
  const res = await fetch(path, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : {},
    body: body !== undefined ? JSON.stringify(body) : undefined,
    credentials: "same-origin",
  });
  if (res.status === 401 && !path.startsWith("/api/auth")) {
    state.me = null;
    location.hash = "#/welcome";
    throw new StaleView("Please sign in");
  }
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") msg = data.detail;
      else if (Array.isArray(data.detail)) msg = data.detail.map((d) => `${d.loc.slice(-1)[0]}: ${d.msg}`).join("; ");
    } catch {}
    throw new Error(msg);
  }
  const data = res.status === 204 ? null : await (res.headers.get("content-type")?.includes("json") ? res.json() : res.text());
  if (nav !== state.nav && method === "GET") throw new StaleView();
  return data;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), Math.max(2400, msg.length * 60));  // time to read longer ones
}

/* Safe markdown: escape everything first, then turn a few rules into HTML. Enough for companions' replies and
   posts, and for the guides: headings, lists and checklists, code, tables, quotes and links. */
function md(src) {
  const lines = esc(src).split(/\n/);
  let html = "", list = null, para = [], quote = [];
  const inline = (s) => s.split(/(`[^`\n]+`)/).map((part, k) => k % 2 ? `<code>${part.slice(1, -1)}</code>` : part
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")).join("");
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  const flushPara = () => { if (para.length) { html += `<p>${inline(para.join("<br>"))}</p>`; para = []; } };
  const flushQuote = () => { if (quote.length) { html += `<blockquote>${inline(quote.join("<br>"))}</blockquote>`; quote = []; } };
  const flush = () => { flushPara(); closeList(); flushQuote(); };
  const isRow = (line) => /^\s*\|.*\|\s*$/.test(line || "");
  const cells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fence = line.match(/^\s*(```|~~~)\s*([\w+#.-]*)/);
    if (fence) {
      flush();
      const code = [];
      while (++i < lines.length && !lines[i].trim().startsWith(fence[1])) code.push(lines[i]);
      const lang = fence[2].toLowerCase();
      html += lang === "math" ? `<pre class="math">${code.join("\n")}</pre>`
        : `<pre class="code"${lang ? ` data-lang="${lang}"` : ""}><code>${code.join("\n")}</code></pre>`;
      continue;
    }
    if (isRow(line) && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1] || "")) {  // a table: a header row, then dashes
      flush();
      const head = cells(line), rows = [];
      i++;
      while (isRow(lines[i + 1])) rows.push(cells(lines[++i]));
      html += `<div class="table"><table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`
        + rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("") + "</tbody></table></div>";
      continue;
    }
    const h = line.match(/^(#{1,6})\s+(.*)/);
    const task = line.match(/^\s*[-*]\s+\[([ xX])\]\s+(.*)/);
    const ul = !task && line.match(/^\s*[-*•]\s+(.*)/);
    const ol = line.match(/^\s*(\d+)[.)]\s+(.*)/);
    const q = line.match(/^&gt;\s?(.*)/);
    if (/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { flush(); html += "<hr>"; }
    else if (h) { flush(); const n = Math.min(4, Math.max(2, h[1].length)); html += `<h${n}>${inline(h[2])}</h${n}>`; }
    else if (task || ul || ol) {
      flushPara(); flushQuote();
      const kind = ol ? "ol" : "ul";
      if (list !== kind) {
        closeList();
        html += ol ? `<ol${ol[1] !== "1" ? ` start="${+ol[1]}"` : ""}>` : `<ul${task ? ' class="tasks"' : ""}>`;
        list = kind;
      }
      html += task ? `<li class="task${task[1] === " " ? "" : " done"}"><span class="box" aria-hidden="true"></span>${inline(task[2])}</li>`
        : `<li>${inline(ul ? ul[1] : ol[2])}</li>`;
    } else if (q) { flushPara(); closeList(); quote.push(q[1]); }
    else if (!line.trim()) flush();
    else { closeList(); flushQuote(); para.push(line); }
  }
  flush();
  return html;
}

/* The logo: the clock beside the name. The clock is decorative here; the name says what it is. */
const logo = (size = 36, style = "") => `<div class="logo"${style ? ` style="${style}"` : ""}><img src="/static/icons/logo-256.webp" width="${size}" height="${size}" alt=""><span>AI <span class="every">Every</span> Time</span></div>`;
const avatarUrl = (c) => `/api/companions/${c.id}/avatar?v=${encodeURIComponent(c.updated_at || "")}`;
const avatarImg = (c, size = 48) => `<img class="avatar" src="${avatarUrl(c)}" width="${size}" height="${size}" alt="${esc(c.persona.name)}" loading="lazy" decoding="async">`;
/* A starter's photo the browser can't load (its host unreachable from here) gives way to the drawn face. */
document.addEventListener("error", (e) => {
  const img = e.target;
  if (!(img instanceof HTMLImageElement) || !img.classList.contains("avatar")) return;
  const url = new URL(img.src, location.origin);
  if (url.origin !== location.origin || !/^\/api\/companions\/\d+\/avatar/.test(url.pathname) || url.searchParams.has("face")) return;
  url.searchParams.set("face", "1");
  img.src = url.pathname + url.search;
}, true);
const fmtCount = (n) => n.toLocaleString();
const label = (s) => s.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
const timeAgo = (iso) => {
  const t = new Date(iso), s = (Date.now() - t) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  if (s < 7 * 86400) return `${Math.floor(s / 86400)}d ago`;
  const year = t.getFullYear() !== new Date().getFullYear() ? { year: "numeric" } : {};
  return t.toLocaleDateString(undefined, { day: "numeric", month: "short", ...year });
};
const FEED_PAGE = 20;  // posts the server sends at a time (app/content.py)

/* ------------------------------------------------------------------- voice */

const canSpeak = "speechSynthesis" in window;
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const prefs = {
  get(k, d) { try { const v = localStorage.getItem(`aiet.${k}`); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(`aiet.${k}`, JSON.stringify(v)); } catch {} },
};

function plainText(markdown) {
  return markdown.replace(/```[\s\S]*?```/g, " ").replace(/[`*_#>]+/g, "").replace(/\[(.*?)\]\(.*?\)/g, "$1")
    .replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, "").replace(/\s+/g, " ").trim();
}

/* Read text aloud in a companion's voice: the closest installed voice to its accent, at its rate and pitch. */
function speak(text, voice) {
  if (!canSpeak || !voice?.enabled) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(plainText(text));
  const voices = speechSynthesis.getVoices();
  const lang = voice.lang || "en-US";
  u.voice = voices.find((v) => v.lang === lang) || voices.find((v) => v.lang.startsWith(lang.split("-")[0])) || null;
  u.lang = lang;
  u.rate = voice.rate ?? 1;
  u.pitch = voice.pitch ?? 1;
  speechSynthesis.speak(u);
}

/* Dictation into a textarea, where the browser supports speech recognition. */
function wireMic(button, textarea) {
  if (!Recognition) { button.remove(); return; }
  let rec = null;
  button.addEventListener("click", () => {
    if (rec) { rec.stop(); return; }
    rec = new Recognition();
    rec.lang = navigator.language || "en-US";
    rec.interimResults = true;
    const before = textarea.value ? textarea.value.trimEnd() + " " : "";
    rec.onresult = (e) => { textarea.value = before + [...e.results].map((r) => r[0].transcript).join(""); };
    rec.onend = () => { rec = null; button.classList.remove("on"); button.setAttribute("aria-pressed", "false"); textarea.focus(); };
    rec.onerror = (e) => { if (e.error !== "aborted") toast("Couldn't hear that. Check microphone permission."); };
    button.classList.add("on");
    button.setAttribute("aria-pressed", "true");
    rec.start();
  });
}

/* Stream a companion's reply; calls onText with the text so far. */
/* Sends a chat message and streams the reply into onText. Returns the reply and the id of the learner's message. */
async function streamChat(companionId, body, onText) {
  const res = await fetch(`/api/companions/${companionId}/chat`, {
    method: "POST", headers: { "Content-Type": "application/json" }, credentials: "same-origin",
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Couldn't send message");
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let acc = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    acc += dec.decode(value, { stream: true });
    onText(acc);
  }
  return { reply: acc, sentId: +res.headers.get("X-Message-Id") || null };
}

/* ---------------------------------------------------------------- reactions */

// The same emoji the server accepts (app/reactions.py).
const REACTIONS = [["👍", "Thumbs up"], ["❤️", "Love"], ["😂", "Haha"], ["😮", "Wow"], ["😢", "Sad"], ["🙏", "Thank you"], ["🎉", "Celebrate"], ["💡", "Insightful"]];
const reactionName = (emoji) => (REACTIONS.find(([e]) => e === emoji) || [, emoji])[1];

/* A chat bubble. Saved messages carry their id; the learner can react to the companion's, and the companion's
   reaction to the learner's shows on theirs. */
function chatBubble(role, text, { id = null, reaction = null } = {}, ctx) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = role === "assistant" ? `<div class="prose">${md(text)}</div>` : esc(text).replace(/\n/g, "<br>");
  if (id) bubbleSaved(el, { id, reaction }, ctx);
  return el;
}

/* Once a bubble's message is saved: its id, any reaction, and on the companion's messages a React button. */
function bubbleSaved(el, { id, reaction }, ctx) {
  el.dataset.id = id;
  if (el.classList.contains("assistant") && !$(".react-btn", el)) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "react-btn";
    b.title = "React";
    b.setAttribute("aria-label", "React to this message");
    b.setAttribute("aria-expanded", "false");
    b.textContent = "🙂";
    el.appendChild(b);
  }
  showReaction(el, reaction, ctx);
}

function showReaction(el, emoji, ctx) {
  el.dataset.reaction = emoji || "";
  let pill = $(":scope > .reaction", el);
  el.classList.toggle("reacted", Boolean(emoji));
  if (!emoji) { pill?.remove(); return; }
  if (!pill) { pill = document.createElement("span"); pill.className = "reaction"; pill.setAttribute("role", "img"); el.appendChild(pill); }
  pill.textContent = emoji;
  const who = el.classList.contains("user") ? ctx.name : "You";
  pill.title = `${who} reacted: ${reactionName(emoji)}`;
  pill.setAttribute("aria-label", pill.title);
}

function closePicker(log) {
  const open = $(".react-picker", log);
  if (!open) return;
  $(".react-btn[aria-expanded=true]", log)?.setAttribute("aria-expanded", "false");
  open.remove();
}

/* Clicks on React buttons and the emoji they offer, for every bubble in a chat log. */
function wireReactions(log, ctx) {
  log.addEventListener("click", async (e) => {
    const btn = e.target.closest(".react-btn");
    if (btn) {
      const bubble = btn.closest(".msg"), wasOpen = btn.getAttribute("aria-expanded") === "true";
      closePicker(log);
      if (wasOpen) return;
      const picker = document.createElement("div");
      picker.className = "react-picker";
      picker.setAttribute("role", "group");
      picker.setAttribute("aria-label", "Reactions");
      picker.innerHTML = REACTIONS.map(([emoji, name]) => `<button type="button" data-react="${emoji}" title="${name}" aria-label="${name}" aria-pressed="${bubble.dataset.reaction === emoji}">${emoji}</button>`).join("");
      bubble.after(picker);
      btn.setAttribute("aria-expanded", "true");
      $("button", picker).focus();
      if (!bubble.nextElementSibling?.nextElementSibling) log.scrollTop = log.scrollHeight;  // the last message: keep its picker in view
      return;
    }
    const pick = e.target.closest("[data-react]");
    if (!pick) return;
    const bubble = pick.closest(".react-picker").previousElementSibling, before = bubble.dataset.reaction || null;
    const emoji = before === pick.dataset.react ? null : pick.dataset.react;  // the same one again takes it back
    closePicker(log);
    $(".react-btn", bubble)?.focus();
    showReaction(bubble, emoji, ctx);
    try {
      await api(`/api/messages/${bubble.dataset.id}/reaction`, emoji ? { method: "PUT", body: { emoji } } : { method: "DELETE" });
    } catch (err) {
      showReaction(bubble, before, ctx);
      toast(err.message);
    }
  });
  log.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && $(".react-picker", log)) {
      const btn = $(".react-btn[aria-expanded=true]", log);
      closePicker(log);
      btn?.focus();
    }
  });
}

document.addEventListener("click", (e) => {
  if (e.target.closest(".react-picker, .react-btn")) return;
  for (const picker of $$(".react-picker")) closePicker(picker.parentElement);
});

/* After a reply: the ids of the two new messages, and the companion's reaction to the learner's, if it made one. */
async function savedExchange(companionId, sentId, userEl, replyEl, ctx) {
  if (!sentId) return;
  try {
    const res = await fetch(`/api/companions/${companionId}/messages?since=${sentId}`, { credentials: "same-origin" });
    if (!res.ok) return;
    const [mine, reply] = await res.json();
    if (mine?.role === "user") bubbleSaved(userEl, mine, ctx);
    if (reply?.role === "assistant" && replyEl) bubbleSaved(replyEl, reply, ctx);
  } catch {}  // without them the chat still works; reactions come back with the page
}
/* The server takes the companion's reaction off the start of its reply; one anywhere else isn't shown either. */
const withoutReactionTags = (text) => text.replace(/<reaction>[^<]{0,16}<\/reaction>\s*/gu, "");

/* ----------------------------------------------------------- notifications */

const pushSupported = "serviceWorker" in navigator && "PushManager" in window && "Notification" in window;
const isIOS = /iPhone|iPad|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
const installedApp = matchMedia("(display-mode: standalone)").matches || navigator.standalone === true;
const deviceZone = (() => { try { return Intl.DateTimeFormat().resolvedOptions().timeZone || ""; } catch { return ""; } })();
let installPrompt = null;

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
  // A tapped notification asks an open tab to show its page.
  navigator.serviceWorker.addEventListener("message", (e) => {
    if (e.data?.type !== "open") return;
    const url = new URL(e.data.url, location.origin);
    if (url.origin === location.origin) location.hash = url.hash || "#/feed";
  });
}
/* Browsers that can install the app ask through this event; the sidebar shows an Install button instead of their banner. */
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  installPrompt = e;
  const b = $("#installBtn");
  if (b) b.hidden = false;
});

const keyBytes = (b64) => Uint8Array.from(atob(b64.replace(/-/g, "+").replace(/_/g, "/")), (c) => c.charCodeAt(0));
const sameBytes = (buf, bytes) => !!buf && new Uint8Array(buf).length === bytes.length && new Uint8Array(buf).every((v, i) => v === bytes[i]);

async function pushSubscription() {
  if (!pushSupported) return null;
  const reg = await navigator.serviceWorker.getRegistration("/");
  return reg ? reg.pushManager.getSubscription() : null;
}

/* Turn notifications on for this device: ask permission, subscribe at the browser's push service, tell the server. */
async function enablePush() {
  const permission = await Notification.requestPermission();
  if (permission !== "granted") {
    throw new Error(permission === "denied" ? "Notifications are blocked for this site in your browser's settings." : "Notifications weren't allowed.");
  }
  const reg = await navigator.serviceWorker.register("/sw.js");
  await navigator.serviceWorker.ready;
  const key = keyBytes((await api("/api/push/key")).public_key);
  let sub = await reg.pushManager.getSubscription();
  // A subscription made for another server key can't receive this server's messages.
  if (sub && !sameBytes(sub.options.applicationServerKey, key)) { await sub.unsubscribe(); sub = null; }
  // Subscribing goes through the browser maker's push service, which can be unreachable; don't wait forever.
  sub ||= await Promise.race([
    reg.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: key }),
    new Promise((_, reject) => setTimeout(() => reject(new Error("Your browser's notification service didn't answer. Try again in a moment.")), 20000)),
  ]);
  await api("/api/push/subscriptions", { method: "POST", body: sub.toJSON() });
}

/* Turn notifications off for this device (also used when signing out). */
async function disablePush() {
  const sub = await pushSubscription().catch(() => null);
  if (!sub) return;
  await api("/api/push/unsubscribe", { method: "POST", body: { endpoint: sub.endpoint } }).catch(() => {});
  await sub.unsubscribe().catch(() => {});
}

/* ------------------------------------------------------------------- router */

const routes = [
  [/^#\/welcome$/, viewWelcome, false],
  [/^#\/reset\/([\w-]+)$/, viewReset, null],
  [/^#\/onboarding$/, viewOnboarding],
  [/^#\/feed$/, viewFeed],
  [/^#\/guides(?:\?(.*))?$/, viewGuides],
  [/^#\/p\/(\d+)$/, viewPost],
  [/^#\/companions$/, viewCompanions],
  [/^#\/c\/(\d+)(?:\?post=(\d+))?$/, viewCompanion],
  [/^#\/studio\/new$/, (m) => viewStudio({ mode: "new" })],
  [/^#\/studio\/remix\/(\d+)$/, (m) => viewStudio({ mode: "remix", id: +m[1] })],
  [/^#\/studio\/(\d+)$/, (m) => viewStudio({ mode: "edit", id: +m[1] })],
  [/^#\/profile$/, viewProfile],
  [/^#\/courses(?:\?(.*))?$/, viewCourses],
  [/^#\/course\/([a-z0-9-]+)$/, viewCourse],
  [/^#\/course\/([a-z0-9-]+)\/learn\/(l\d+-\d+|lab\d+|quiz\d+|project|grades)$/, viewCoursePlayer],
  [/^#\/track\/([a-z0-9-]+)$/, viewTrack],
  [/^#\/certificate\/([A-Za-z0-9]+)$/, viewCertificate, null],
  [/^#\/learn$/, viewLearn],
  [/^#\/lesson\/([\w-]+)(?:\?companion=(\d+))?$/, viewLesson],
  [/^#\/quiz\/(\d+)$/, viewQuiz],
  [/^#\/practice$/, viewPractice],
  [/^#\/review(?:\?deck=(\d+))?$/, viewReview],
  [/^#\/progress$/, viewProgress],
  [/^#\/staff(?:\/(insights|reports|appeals|paths|roles|log))?$/, viewStaff],
];

async function router() {
  state.nav++;
  document.title = "AI Every Time";
  // A dialog belongs to the page that opened it (Back, a notification or a link can change pages under it).
  if ($("#dialog")?.open) $("#dialog").close();
  const hash = location.hash || "#/feed";
  if (!state.me) {
    // Not api(): its sign-in redirect would pull pages open to everyone, like the reset link, back to the welcome page.
    const nav = state.nav;
    const res = await fetch("/api/me", { credentials: "same-origin" }).catch(() => null);
    const me = res?.ok ? await res.json() : null;
    if (nav !== state.nav) return;  // a newer navigation took over while this one waited
    state.me = me;
  }
  // Reminders follow the learner's own day, so learn their time zone the first time we can.
  if (state.me && !state.me.timezone && deviceZone && state.zoneSentFor !== state.me.id) {
    state.zoneSentFor = state.me.id;
    api("/api/me", { method: "PATCH", body: { timezone: deviceZone } }).then((me) => { state.me = me; }).catch(() => {});
  }
  // A problem with the site's Claude account comes first; then demo mode.
  $("#banner").hidden = !(state.me?.ai_problem || state.me?.offline_mode);
  $("#banner").textContent = state.me?.ai_problem
    || "Demo mode: companions use templated replies. Set ANTHROPIC_API_KEY on the server for real AI conversations and content.";
  // needsAuth: true = signed in, false = signed out, null = either.
  for (const [re, view, needsAuth = true] of routes) {
    const m = hash.match(re);
    if (!m) continue;
    if (needsAuth === true && !state.me) { location.hash = "#/welcome"; return; }
    if (needsAuth === false && state.me) { location.hash = "#/feed"; return; }
    if (needsAuth === true && !state.me.onboarded && !/^#\/(onboarding|studio)/.test(hash)) { location.hash = "#/onboarding"; return; }
    try { await view(m); } catch (e) { if (e instanceof StaleView) return; console.error(e); toast(e.message); }
    window.scrollTo(0, 0);
    return;
  }
  location.hash = state.me ? "#/feed" : "#/welcome";
}
window.addEventListener("hashchange", router);

function shell(active, inner) {
  const nav = [["feed", "🏠", "Feed"], ["guides", "🧭", "Guides"], ["courses", "🎓", "Courses"], ["learn", "📚", "Learn"], ["practice", "🧠", "Practice"], ["progress", "📈", "Progress"],
    ["companions", "🌍", "Companions"], ["profile", "👤", "Profile"]];
  if (state.me?.role && state.me.role !== "learner") nav.push(["staff", "🛡️", "Staff"]);
  const primary = state.me?.primary_companion_id;
  app.innerHTML = `
    <div class="shell">
      <aside class="side">
        ${logo()}
        <nav class="nav">
          ${nav.map(([k, i, t]) => `<a href="#/${k}" class="${active === k ? "active" : ""}">${i} ${t}</a>`).join("")}
          ${primary ? `<a href="#/c/${primary}" class="${active === "mine" ? "active" : ""}">💬 My companion</a>` : ""}
          <a href="#/studio/new" class="${active === "studio" ? "active" : ""}">✨ Design a companion</a>
          <button type="button" class="install" id="installBtn" ${installPrompt ? "" : "hidden"}>📲 Install the app</button>
        </nav>
        <a href="#/feed" class="bell" id="bell" title="Reminders" hidden>🔔 <span></span></a>
        <div class="me"><div><b>${esc(state.me.display_name)}</b><div class="muted small">${esc(state.me.level)}</div></div></div>
      </aside>
      <main class="main">${inner}</main>
    </div>`;
  $("#installBtn").addEventListener("click", async () => {
    if (!installPrompt) return;
    installPrompt.prompt();
    await installPrompt.userChoice.catch(() => {});
    installPrompt = null;
    $("#installBtn")?.setAttribute("hidden", "");
  });
  refreshBell();
  return $(".main");
}

/* The sidebar bell shows how many reminders are waiting. */
async function refreshBell() {
  const bell = $("#bell");
  if (!bell) return;
  try {
    const rem = await fetch("/api/reminders", { credentials: "same-origin" }).then((r) => (r.ok ? r.json() : []));
    if (!document.body.contains(bell)) return;
    bell.hidden = !rem.length;
    $("span", bell).textContent = `${rem.length} reminder${rem.length === 1 ? "" : "s"}`;
  } catch {}
}

/* ------------------------------------------------------------------ welcome */

async function viewWelcome() {
  app.innerHTML = `
    <section class="landing">
      <div>
        ${logo(64, "margin:0 0 24px;font-size:26px;gap:14px")}
        <h1>Every piece of AI content is the <em>start</em> of a learning relationship.</h1>
        <p class="muted" style="font-size:18px">Follow, chat with and learn from always-available AI Learning Companions from around the world, each with their own face, voice and expertise. Or design your own companion, exactly the way you want it.</p>
        <div class="faces" id="faces"></div>
        <p class="muted small">Available 24/7 · Remembers you · Courses, quizzes and tutoring that continue where the content ends</p>
      </div>
      <div class="card">
        <div class="tabs">
          <button class="btn on" data-tab="signup">Create account</button>
          <button class="btn" data-tab="login">Sign in</button>
        </div>
        <form id="authForm" class="stack" novalidate>
          <label data-only="signup">Your name <input type="text" name="display_name" autocomplete="name" required maxlength="60"></label>
          <label>Email <input type="email" name="email" autocomplete="email" required></label>
          <label>Password <span class="hint">At least 8 characters</span><input type="password" name="password" autocomplete="new-password" required minlength="8"></label>
          <label data-only="signup">Your AI level
            <select name="level"><option value="beginner">Beginner: I'm new to AI</option><option value="intermediate">Intermediate: I know the basics</option><option value="advanced">Advanced: I build with AI</option></select>
          </label>
          <label data-only="signup">What do you want to learn? <span class="hint">Optional, your companion will use this</span>
            <textarea name="goals" maxlength="500" placeholder="e.g. Understand how ChatGPT-style models work and build my first AI app"></textarea>
          </label>
          <label data-only="signup">What are you into? <span class="hint">Optional: hobbies, work, anything you love. Companions start your chats from it</span>
            <input type="text" name="interests" maxlength="300" placeholder="e.g. football, cooking, my job in banking">
          </label>
          <div class="error" id="authErr"></div>
          <button class="btn primary" type="submit" id="authBtn">Create account</button>
          <button type="button" class="linkbtn" id="forgotBtn" data-only="login" hidden>Forgot your password?</button>
        </form>
      </div>
    </section>`;
  fetch("/api/starters/faces").then((r) => (r.ok ? r.json() : [])).then((faces) => {
    if (!$("#faces")) return;
    $("#faces").innerHTML = faces.map((f) => `<img class="avatar" src="/api/companions/${f.id}/avatar?v=${encodeURIComponent(f.updated_at)}" width="56" height="56" alt="" title="${esc(f.name)}">`).join("");
  }).catch(() => {});
  let tab = "signup";
  const setTab = (t) => {
    tab = t;
    $$("[data-tab]").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
    $$("[data-only]").forEach((el) => (el.hidden = el.dataset.only !== t));
    $("#authBtn").textContent = t === "signup" ? "Create account" : "Sign in";
    $("[name=password]").autocomplete = t === "signup" ? "new-password" : "current-password";
  };
  $$("[data-tab]").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
  $("#forgotBtn").addEventListener("click", () => showForgot($("[name=email]").value.trim()));
  $("#authForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target));
    const body = tab === "signup" ? { ...f, timezone: deviceZone } : { email: f.email, password: f.password };
    $("#authErr").textContent = "";
    $("#authBtn").disabled = true;
    try {
      state.me = await api(`/api/auth/${tab}`, { method: "POST", body });
      if (tab === "signup" && state.me.email_available) toast("Welcome! We've emailed you a link to confirm your address.");
      location.hash = state.me.onboarded ? "#/feed" : "#/onboarding";
    } catch (err) {
      $("#authErr").textContent = err.message;
    } finally {
      $("#authBtn").disabled = false;
    }
  });
}

function showForgot(email) {
  showDialog(`<h2>Forgot your password?</h2>
    <p>Enter the email address of your account and we'll send a link for choosing a new password.</p>
    <form id="forgotForm" class="stack">
      <label>Email <input type="email" name="email" autocomplete="email" required value="${esc(email)}"></label>
      <div class="small" id="forgotMsg" role="status"></div>
      <button class="btn primary">Send the link</button>
    </form>`);
  $("#forgotForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("button", e.target);
    btn.disabled = true;
    try {
      await api("/api/auth/forgot", { method: "POST", body: { email: e.target.email.value.trim() } });
      $("#forgotMsg").textContent = "If there's an account with that address, a link is on its way. It works once, within the next hour.";
    } catch (err) {
      $("#forgotMsg").textContent = err.message;
      btn.disabled = false;
    }
  });
}

/* Opened from the emailed link. Works whether or not someone is signed in on this browser. */
async function viewReset(m) {
  app.innerHTML = `
    <section class="card stack narrow-card">
      ${logo(40, "margin:0")}
      <h1 style="margin:0">Choose a new password</h1>
      <form id="resetForm" class="stack" novalidate>
        <label>New password <span class="hint">At least 8 characters</span><input type="password" name="password" autocomplete="new-password" required minlength="8"></label>
        <label>Type it again <input type="password" name="again" autocomplete="new-password" required minlength="8"></label>
        <div class="error" id="resetErr"></div>
        <button class="btn primary">Save and sign in</button>
      </form>
      <p class="muted small" style="margin:0">This signs you out on every other phone and browser, and stops notifications on them.</p>
    </section>`;
  $("#resetForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = e.target;
    if (f.password.value.length < 8) { $("#resetErr").textContent = "Use at least 8 characters."; return; }
    if (f.password.value !== f.again.value) { $("#resetErr").textContent = "The two passwords don't match."; return; }
    try {
      await api("/api/auth/reset", { method: "POST", body: { token: m[1], password: f.password.value } });
      state.me = await api("/api/me");
      toast("Password changed. You're signed in.");
      // Replace the link in the history so the spent token isn't left behind.
      location.replace(state.me.onboarded ? "#/feed" : "#/onboarding");
    } catch (err) { $("#resetErr").textContent = err.message; }
  });
}

/* --------------------------------------------------------------- onboarding */

const REAL_NOTE = "AI companion modelled on a real person";
const RETIRED_NOTE = "No longer listed for new learners. Your chats with them are kept, and you can carry on.";

function companionCard(c, { onboarding = false } = {}) {
  const p = c.persona;
  const mine = c.owner_id === state.me.id;
  const isPrimary = state.me.primary_companion_id === c.id;
  return `
    <article class="card ccard" data-id="${c.id}">
      <div class="head">
        ${avatarImg(c, 64)}
        <div>
          <h3>${esc(p.name)} ${esc(p.flag)} ${mine ? '<span class="mine-badge">Yours</span>' : ""}</h3>
          <div class="muted small">${esc(p.country)}${isPrimary ? " · Your primary companion" : ""}</div>
          ${c.followers != null ? `<div class="muted small followers">${fmtCount(c.followers)} follower${c.followers === 1 ? "" : "s"}</div>` : ""}
          ${c.real_person ? `<div class="muted small">${REAL_NOTE}</div>` : ""}
          ${!c.is_starter && !mine ? `<div class="muted small">Community · designed by ${esc(c.designer || "a learner")}</div>` : ""}
          ${mine && c.removed ? `<div class="small error">Removed from the community by a moderator</div>`
            : mine && c.hidden ? `<div class="small error">Hidden after reports from other learners</div>`
            : mine && c.is_public ? `<div class="small" style="color:var(--accent)">● Published to the community</div>` : ""}
        </div>
      </div>
      <p class="tagline">${esc(p.tagline)}</p>
      <div class="row">${p.expertise.map((x) => `<span class="chip">${esc(x)}</span>`).join("")}</div>
      ${p.writing_style.sample_line ? `<p class="small" style="margin:0"><i>“${esc(p.writing_style.sample_line)}”</i></p>` : ""}
      <div class="actions">
        ${onboarding
          ? `<button class="btn primary" data-act="choose">Choose ${esc(p.name)}</button>
             ${c.real_person ? "" : `<a class="btn" href="#/studio/remix/${c.id}">Customise</a>`}`
          : `<a class="btn primary" href="#/c/${c.id}">Chat</a>
             ${mine ? "" : `<button class="btn ${c.following ? "on" : ""}" data-act="follow">${c.following ? "Following" : "Follow"}</button>`}
             ${mine ? `<a class="btn" href="#/studio/${c.id}">Edit</a>` : c.real_person ? "" : `<a class="btn" href="#/studio/remix/${c.id}">Remix</a>`}
             ${mine && !c.hidden ? `<button class="btn ghost" data-act="publish">${c.is_public ? "Unpublish" : "Publish"}</button>` : ""}
             ${isPrimary ? "" : `<button class="btn ghost" data-act="primary">Make primary</button>`}`}
      </div>
    </article>`;
}

function wireCompanionCards(root, list, rerender) {
  $$(".ccard", root).forEach((el) => {
    const c = list.find((x) => x.id === +el.dataset.id);
    el.addEventListener("click", async (e) => {
      const act = e.target.closest("[data-act]")?.dataset.act;
      if (!act) return;
      try {
        if (act === "choose" || act === "primary") {
          await api(`/api/companions/${c.id}/primary`, { method: "POST" });
          state.me = await api("/api/me");
          if (act === "choose") { toast(`${c.persona.name} is now your companion`); location.hash = `#/c/${c.id}`; return; }
          toast(`${c.persona.name} is now your primary companion`);
        } else if (act === "publish") {
          if (!(await togglePublish(c))) return;
        } else if (act === "follow") {
          await api(`/api/companions/${c.id}/follow`, { method: c.following ? "DELETE" : "POST" });
          toast(c.following ? `Unfollowed ${c.persona.name}` : `Following ${c.persona.name}`);
        }
        rerender();
      } catch (err) { toast(err.message); }
    });
  });
}

const PUBLISH_NOTE = "Publishing lets every learner find, follow, chat with and remix this companion. " +
  "Its name, face, bio, personality and style become public, and so do the posts it writes. " +
  "Your display name is shown as its designer. " +
  "Your custom instructions, chats and memories stay private. You can unpublish at any time.\n\nPublish?";

async function togglePublish(c) {
  if (!c.is_public && !confirm(PUBLISH_NOTE)) return false;
  await api(`/api/companions/${c.id}/publish`, { method: c.is_public ? "DELETE" : "POST" });
  toast(c.is_public ? `${c.persona.name} is private again` : `${c.persona.name} is now in the community`);
  return true;
}

/* The starter companions, a hundred or so: by region, or searched by name, country or topic. */
const REGION_ORDER = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania", "Antarctica"];
const browse = { region: "", q: "" };
const fold = (t) => String(t ?? "").normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();

function starterBrowser(starters) {
  const counts = {};
  for (const c of starters) counts[c.region] = (counts[c.region] || 0) + 1;
  const regions = [...REGION_ORDER.filter((r) => counts[r]), ...Object.keys(counts).filter((r) => r && !REGION_ORDER.includes(r))];
  if (browse.region && !counts[browse.region]) browse.region = "";
  const tab = (r, label, n) => `<button type="button" class="btn ${browse.region === r ? "on" : ""}" data-region="${esc(r)}" aria-pressed="${browse.region === r}">${esc(label)} <span class="count">${n}</span></button>`;
  return `
    <div class="browse">
      <div class="choice regions" role="group" aria-label="Region">${tab("", "All", starters.length)}${regions.map((r) => tab(r, r, counts[r])).join("")}</div>
      <input type="search" id="starterSearch" placeholder="Search by name, country or topic" aria-label="Search companions" value="${esc(browse.q)}" maxlength="80">
    </div>
    <div class="grid" id="starterGrid"></div>`;
}

function wireStarterBrowser(root, starters, opts, rerender) {
  const grid = $("#starterGrid", root);
  const show = () => {
    const q = fold(browse.q.trim());
    const shown = starters.filter((c) => (!browse.region || c.region === browse.region)
      && (!q || fold([c.persona.name, c.persona.country, c.region, c.persona.tagline, ...c.persona.expertise].join(" ")).includes(q)));
    grid.innerHTML = shown.length ? shown.map((c) => companionCard(c, opts)).join("")
      : `<div class="card empty">No companions match "${esc(browse.q.trim())}". Try a name, a country or a topic like "robotics".</div>`;
    wireCompanionCards(grid, shown, rerender);
  };
  $(".regions", root).addEventListener("click", (e) => {
    const b = e.target.closest("[data-region]");
    if (!b) return;
    browse.region = b.dataset.region;
    $$("[data-region]", root).forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-pressed", String(x === b)); });
    show();
  });
  $("#starterSearch", root).addEventListener("input", (e) => { browse.q = e.target.value; show(); });
  show();
}

async function viewOnboarding() {
  const list = await api("/api/companions");
  const starters = list.filter((c) => c.is_starter && !c.retired);
  app.innerHTML = `
    <section class="main" style="margin:0 auto">
      ${logo(40, "margin:0 0 20px")}
      <div class="steps"><span>1. Account ✓</span><span>›</span><b>2. Your companion</b><span>›</span><span>3. Start learning</span></div>
      <h1>Welcome, ${esc(state.me.display_name)}! Who do you want to learn with?</h1>
      <p class="muted">Pick one of our companions as-is, customise one, or design your own from scratch. You can change your companion's look, personality and style whenever you like.</p>
      <div class="card row spread" style="margin:20px 0">
        <div><h3 style="margin:0">✨ Design your own companion</h3><div class="muted small">Choose their face, country, personality, teaching style and writing voice.</div></div>
        <a class="btn primary" href="#/studio/new">Start designing</a>
      </div>
      <h2>Or choose one of our ${starters.length} companions</h2>
      <div id="starters">${starterBrowser(starters)}</div>
    </section>`;
  wireStarterBrowser($("#starters"), starters, { onboarding: true }, viewOnboarding);
}

/* --------------------------------------------------------------------- feed */

const KIND_LABELS = { tutorial: "Tutorial", cheat_sheet: "Cheat sheet" };
const guideBadges = (p, topic = true) => `<div class="badges">${[KIND_LABELS[p.kind], p.level && label(p.level),
  p.minutes && `${p.minutes} min read`, topic && p.topic].filter(Boolean).map((b) => `<span class="badge">${esc(b)}</span>`).join("")}</div>`;
const byline = (c, when) => `<div class="by">
  <a href="#/c/${c.id}">${avatarImg(c, 36)}</a>
  <div><a href="#/c/${c.id}" class="plain"><b>${esc(c.persona.name)}</b> ${esc(c.persona.flag)}</a>
  <div class="muted small">${esc(c.persona.country || c.region || "")}${when ? ` · ${timeAgo(when)}` : ""}</div></div>
</div>`;
const engageButtons = (post) => `
  <button class="btn ${post.liked ? "on" : ""}" data-eng="like" aria-pressed="${post.liked}">♥ <span>${post.likes || ""}</span></button>
  <button class="btn ${post.marked_helpful ? "on" : ""}" data-eng="helpful" aria-pressed="${post.marked_helpful}">💡 Helpful <span>${post.helpful || ""}</span></button>
  <button class="btn ghost" data-eng="share">↗ Share</button>`;

function postCard(post, c) {
  if (post.kind) return guideCard(post, c);
  return `
    <article class="card post" data-post="${post.id}">
      <div class="by">
        <a href="#/c/${c.id}">${avatarImg(c, 40)}</a>
        <div><a href="#/c/${c.id}" style="color:inherit;text-decoration:none"><b>${esc(c.persona.name)}</b> ${esc(c.persona.flag)}</a>
        <div class="muted small">${esc(post.topic)} · ${timeAgo(post.created_at)}</div></div>
      </div>
      <h3>${esc(post.title)}</h3>
      <div class="prose">${md(post.body)}</div>
      <div class="row">
        <a class="btn primary" href="#/c/${c.id}?post=${post.id}">💬 Ask ${esc(c.persona.name)} about this</a>
        ${engageButtons(post)}
      </div>
      <div class="row small">
        <span class="muted">Study this:</span>
        <button class="btn ghost" data-study="quiz">📝 Quiz me</button>
        <button class="btn ghost" data-study="cards">🃏 Flashcards</button>
        <button class="btn ghost" data-study="summary">📄 Summary</button>
      </div>
    </article>`;
}

/* A guide's graphic: the site's own, its spelling corrected (app/guides.py). Most of its words are the guide's
   own, which the page shows as text, so where it stands beside them it needs no description of its own. */
const guideArt = (post, alt = "") => post.graphic
  ? `<img class="guide-art" src="${esc(post.graphic)}" alt="${esc(alt)}" width="1024" height="1024" loading="lazy" decoding="async">` : "";

/* A guide in the feed, as the site shows them: its graphic, then what it is, its title, summary and author. */
function guideCard(post, c) {
  return `
    <article class="card post guide-card" data-post="${post.id}">
      ${post.graphic ? `<a class="guide-graphic" href="#/p/${post.id}" tabindex="-1" aria-hidden="true">${guideArt(post)}</a>` : ""}
      ${guideBadges(post)}
      <h3><a href="#/p/${post.id}" class="plain">${esc(post.title)}</a></h3>
      <p class="muted" style="margin:0">${esc(post.summary)}</p>
      ${byline(c, post.created_at)}
      <div class="row">
        <a class="btn primary" href="#/p/${post.id}">📖 Read the guide</a>
        <a class="btn" href="#/c/${c.id}?post=${post.id}">💬 Ask ${esc(c.persona.name)}</a>
        ${engageButtons(post)}
      </div>
    </article>`;
}

/* A guide in a grid: the top of its graphic as a cover, and what it is. */
function guideTile(post, c) {
  return `
    <a class="card guide-tile plain" href="#/p/${post.id}">
      ${post.graphic ? `<div class="cover">${guideArt(post)}</div>` : ""}
      ${guideBadges(post, false)}
      <b class="title">${esc(post.title)}</b>
      <span class="muted small summary">${esc(post.summary)}</span>
      ${c ? `<span class="muted small">${esc(c.persona.name)} ${esc(c.persona.flag)} · ${timeAgo(post.created_at)}</span>` : ""}
    </a>`;
}

/* Views, likes, "helpful", shares and study for the posts in root; posts added later are wired when it's called
   again. A like changes its button where it is, so the page doesn't jump back to the top. */
function wirePosts(root, posts) {
  const seen = wirePosts.seen || (wirePosts.seen = new Set());
  const io = wirePosts.io || (wirePosts.io = "IntersectionObserver" in window ? new IntersectionObserver((entries) => {
    for (const en of entries) {
      const id = +en.target.dataset.post;
      if (en.isIntersecting && !seen.has(id)) { seen.add(id); api(`/api/posts/${id}/engage`, { method: "POST", body: { kind: "view" } }).catch(() => {}); }
    }
  }, { threshold: 0.6 }) : null);
  $$("[data-post]", root).forEach((el) => {
    if (el.dataset.wired) return;
    el.dataset.wired = "1";
    io?.observe(el);
    const post = posts.find((p) => p.id === +el.dataset.post);
    el.addEventListener("click", async (e) => {
      const study = e.target.closest("[data-study]")?.dataset.study;
      if (study) return studyAction(study, "post", String(post.id), { companionId: post.companion_id, button: e.target.closest("button") });
      const btn = e.target.closest("[data-eng]");
      if (!btn) return;
      const kind = btn.dataset.eng;
      try {
        if (kind === "share") {
          const url = `${location.origin}/#/${post.kind ? `p/${post.id}` : `c/${post.companion_id}?post=${post.id}`}`;
          await navigator.clipboard?.writeText(url).catch(() => {});
          await api(`/api/posts/${post.id}/engage`, { method: "POST", body: { kind } });
          toast("Link copied");
          return;
        }
        const flag = kind === "like" ? "liked" : "marked_helpful", count = kind === "like" ? "likes" : "helpful";
        if (post[flag]) await api(`/api/posts/${post.id}/engage/${kind}`, { method: "DELETE" });
        else await api(`/api/posts/${post.id}/engage`, { method: "POST", body: { kind } });
        post[flag] = !post[flag];
        post[count] = Math.max(0, (post[count] || 0) + (post[flag] ? 1 : -1));
        btn.classList.toggle("on", post[flag]);
        btn.setAttribute("aria-pressed", String(post[flag]));
        $("span", btn).textContent = post[count] || "";
      } catch (err) { toast(err.message); }
    });
  });
}

const after = (post) => encodeURIComponent(`${post.created_at}|${post.id}`);

/* A "Load more" button under a list that comes a page at a time. */
function moreButton(box, fetchPage, render, pageSize) {
  const btn = document.createElement("button");
  btn.className = "btn more";
  btn.textContent = "Load more";
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      const page = await fetchPage();
      render(page);
      if (page.length < pageSize) btn.remove();
    } catch (err) { if (!(err instanceof StaleView)) toast(err.message); } finally { btn.disabled = false; }
  });
  box.after(btn);
  return btn;
}

async function viewFeed() {
  const [posts, list] = await Promise.all([api("/api/feed"), api("/api/companions")]);
  const byId = Object.fromEntries(list.map((c) => [c.id, c]));
  const main = shell("feed", `
    <div class="row spread" style="margin-bottom:16px">
      <div><h1 style="margin:0">Your feed</h1><div class="muted">Fresh lessons and guides from the companions you follow. Every post is a conversation starter.</div></div>
      <div class="row"><a class="btn" href="#/guides">All guides</a><a class="btn" href="#/companions">Find more companions</a></div>
    </div>
    <div id="today"></div>
    <div class="stack feed" id="posts">
      ${posts.length ? posts.map((p) => postCard(p, byId[p.companion_id])).join("") : `<div class="card empty">Follow a few companions to fill your feed. <a href="#/companions">Browse companions</a></div>`}
    </div>`);
  wirePosts(main, posts);
  if (posts.length >= FEED_PAGE) {
    moreButton($("#posts"), () => api(`/api/feed?before=${after(posts[posts.length - 1])}`), (page) => {
      posts.push(...page);
      $("#posts").insertAdjacentHTML("beforeend", page.map((p) => postCard(p, byId[p.companion_id])).join(""));
      wirePosts(main, posts);
    }, FEED_PAGE);
  }
  const rem = (await api("/api/reminders")).filter((r) => r.kind !== "new_posts");
  if (rem.length) {
    $("#today").innerHTML = `<div class="card today"><h3>Today</h3><ul>${rem.map((r) => r.kind === "notice"
      ? `<li class="notice"><span>📣 <a href="${esc(r.link)}">${esc(r.text)}</a></span><button class="linkbtn" data-read="${r.id}">Dismiss</button></li>`
      : `<li><a href="${esc(r.link)}">${esc(r.text)}</a></li>`).join("")}</ul></div>`;
    $("#today").addEventListener("click", async (e) => {
      const id = e.target.dataset.read;
      if (!id) return;
      await api(`/api/reminders/notices/${id}/read`, { method: "POST" });
      e.target.closest("li").remove();
      refreshBell();
    });
  }
  await api("/api/reminders/feed-seen", { method: "POST" });
  refreshBell();
}

/* ------------------------------------------------------------------- guides */

const GUIDES_PAGE = 24;  // guides the server sends at a time (app/main.py)

/* One post on a page of its own: a guide with its graphic, or a companion's post. */
async function viewPost(m) {
  const id = +m[1];
  const post = await api(`/api/posts/${id}`);
  const [c, more] = await Promise.all([
    api(`/api/companions/${post.companion_id}`),
    post.kind ? api(`/api/guides?companion=${post.companion_id}`) : Promise.resolve([]),
  ]);
  const name = esc(c.persona.name);
  const others = more.filter((x) => x.id !== id).slice(0, 3);
  const main = shell(post.kind ? "guides" : "feed", `
    <article class="guide">
      <div class="muted small crumbs">${post.kind ? `<a href="#/guides">Guides</a> › <a href="#/guides?topic=${encodeURIComponent(post.topic)}">${esc(post.topic)}</a>`
        : `<a href="#/feed">Feed</a>`}</div>
      ${post.kind ? guideBadges(post, false) : ""}
      <h1>${esc(post.title)}</h1>
      ${post.summary ? `<p class="lead">${esc(post.summary)}</p>` : ""}
      <div class="row spread">${byline(c, post.created_at)}
        ${c.is_mine ? "" : `<button class="btn ${c.following ? "on" : ""}" id="followBtn">${c.following ? "Following" : "Follow"}</button>`}</div>
      ${post.graphic ? `<figure class="guide-figure">${guideArt(post, `The graphic for “${post.title}”`)}</figure>` : ""}
      <div class="row actions" data-post="${post.id}">
        <a class="btn primary" href="#/c/${c.id}?post=${post.id}">💬 Ask ${name} about this</a>
        ${engageButtons(post)}
      </div>
      <div class="prose guide-body">${md(post.body)}</div>
      <div class="card study" data-post="${post.id}">
        <h3 style="margin:0">Study this ${post.kind ? KIND_LABELS[post.kind].toLowerCase() : "post"}</h3>
        <div class="row">
          <button class="btn" data-study="quiz">📝 Quiz me</button>
          <button class="btn" data-study="cards">🃏 Flashcards</button>
          <button class="btn" data-study="summary">📄 Summary</button>
          <a class="btn ghost" href="#/c/${c.id}?post=${post.id}">💬 Talk it through with ${name}</a>
        </div>
      </div>
    </article>
    ${others.length ? `<section class="more-guides"><div class="row spread"><h2 style="margin:0">More from ${name}</h2>
      <a class="btn ghost" href="#/guides?companion=${c.id}">All guides by ${name}</a></div>
      <div class="guide-grid">${others.map((x) => guideTile(x, c)).join("")}</div></section>` : ""}`);
  wirePosts(main, [post]);
  $("#followBtn")?.addEventListener("click", async (e) => {
    try {
      await api(`/api/companions/${c.id}/follow`, { method: c.following ? "DELETE" : "POST" });
      c.following = !c.following;
      e.target.classList.toggle("on", c.following);
      e.target.textContent = c.following ? "Following" : "Follow";
    } catch (err) { toast(err.message); }
  });
}

/* The guides library: every companion's tutorials and cheat sheets, by region like aieverytime.org's education
   page, and by kind, level, topic or a search. The filters live in the address, so Back and links keep them. */
async function viewGuides(m) {
  const params = new URLSearchParams(m[1] || "");
  for (const [k, v] of [...params]) if (!v) params.delete(k);
  const [list, topics, companions] = await Promise.all([api(`/api/guides?${params}`), api("/api/guides/topics"), api("/api/companions")]);
  const byId = Object.fromEntries(companions.map((c) => [c.id, c]));
  const only = params.get("companion") ? byId[+params.get("companion")] : null;
  const link = (changes) => {
    const p = new URLSearchParams(params);
    for (const [k, v] of Object.entries(changes)) v ? p.set(k, v) : p.delete(k);
    const s = p.toString();
    return `#/guides${s ? `?${s}` : ""}`;
  };
  const tab = (key, value, text) => `<a href="${link({ [key]: value })}" class="${(params.get(key) || "") === (value || "") ? "active" : ""}">${text}</a>`;
  const main = shell("guides", `
    <div class="row spread" style="margin-bottom:6px">
      <div><h1 style="margin:0">Guides</h1>
      <div class="muted">Tutorials and cheat sheets by the AI Learning Companions: read one, then ask its author anything.</div></div>
    </div>
    <nav class="region-tabs" aria-label="Region">${tab("region", "", "All regions")}${REGION_ORDER.map((r) => tab("region", r, esc(r))).join("")}</nav>
    <form class="row filters" id="filters">
      <div class="seg" role="group" aria-label="Kind">${tab("kind", "", "All")}${tab("kind", "tutorial", "Tutorials")}${tab("kind", "cheat_sheet", "Cheat sheets")}</div>
      <select name="level" aria-label="Level"><option value="">All levels</option>${["beginner", "intermediate", "advanced"].map((l) =>
        `<option value="${l}" ${params.get("level") === l ? "selected" : ""}>${label(l)}</option>`).join("")}</select>
      <select name="topic" aria-label="Topic"><option value="">All topics</option>${topics.map((t) =>
        `<option value="${esc(t.topic)}" ${params.get("topic") === t.topic ? "selected" : ""}>${esc(t.topic)} (${t.guides})</option>`).join("")}</select>
      <input type="search" name="q" value="${esc(params.get("q") || "")}" placeholder="Search guides or companions" aria-label="Search guides">
      <button class="btn">Search</button>
    </form>
    ${only ? `<div class="row small" style="margin:4px 0 12px"><span class="chip">By ${esc(only.persona.name)} ${esc(only.persona.flag)}</span>
      <a href="${link({ companion: "" })}">Show every companion's guides</a></div>` : ""}
    <div class="guide-grid" id="guides">${list.length ? list.map((p) => guideTile(p, byId[p.companion_id])).join("")
      : `<div class="card empty">No guides match. <a href="#/guides">See them all</a></div>`}</div>`);
  const form = $("#filters");
  const go = () => {
    const data = new FormData(form);
    location.hash = link({ level: data.get("level"), topic: data.get("topic"), q: String(data.get("q") || "").trim() });
  };
  form.addEventListener("submit", (e) => { e.preventDefault(); go(); });
  $$("select", form).forEach((s) => s.addEventListener("change", go));
  if (list.length >= GUIDES_PAGE) {
    moreButton($("#guides"), () => {
      const p = new URLSearchParams(params);
      p.set("before", `${list[list.length - 1].created_at}|${list[list.length - 1].id}`);
      return api(`/api/guides?${p}`);
    }, (page) => {
      list.push(...page);
      $("#guides").insertAdjacentHTML("beforeend", page.map((p) => guideTile(p, byId[p.companion_id])).join(""));
    }, GUIDES_PAGE);
  }
}

/* --------------------------------------------------------------- companions */

async function viewCompanions() {
  const list = await api("/api/companions");
  // The server lists other learners' companions most followed first.
  const mine = list.filter((c) => c.owner_id === state.me.id), starters = list.filter((c) => c.is_starter && !c.retired);
  const community = list.filter((c) => !c.is_starter && c.owner_id !== state.me.id);
  const retired = list.filter((c) => c.retired);
  const main = shell("companions", `
    <h1>Companions</h1>
    <p class="muted">Each companion represents a country and brings their own expertise, perspective and voice.</p>
    ${mine.length ? `<h2>Your companions</h2><div class="grid" id="mineGrid" style="margin-bottom:28px">${mine.map((c) => companionCard(c)).join("")}</div>` : ""}
    <h2>Around the world</h2>
    <div id="starters" style="margin-bottom:28px">${starterBrowser(starters)}</div>
    <h2>Community</h2>
    <p class="muted">Companions designed and shared by other learners.</p>
    ${community.length ? `<div class="grid" id="communityGrid">${community.map((c) => companionCard(c)).join("")}</div>`
      : `<div class="card empty">No community companions yet. Design one and publish it to be the first.</div>`}
    ${retired.length ? `<h2 style="margin-top:28px">Earlier companions</h2>
      <p class="muted">${RETIRED_NOTE}</p>
      <div class="grid" id="retiredGrid">${retired.map((c) => companionCard(c)).join("")}</div>` : ""}`);
  if (mine.length) wireCompanionCards($("#mineGrid", main), mine, viewCompanions);
  if (retired.length) wireCompanionCards($("#retiredGrid", main), retired, viewCompanions);
  wireStarterBrowser($("#starters", main), starters, {}, viewCompanions);
  if (community.length) wireCompanionCards($("#communityGrid", main), community, viewCompanions);
}

/* --------------------------------------------------------- companion + chat */

async function viewCompanion(m) {
  const id = +m[1];
  const postId = m[2] ? +m[2] : null;
  const [c, messages, memories, posts] = await Promise.all([
    api(`/api/companions/${id}`), api(`/api/companions/${id}/messages`),
    api(`/api/companions/${id}/memories`), api(`/api/companions/${id}/posts`),
  ]);
  const p = c.persona;
  let contextPost = postId ? posts.find((x) => x.id === postId) : null;
  // A post further back than the first page of them, like an older guide.
  if (postId && !contextPost) contextPost = await api(`/api/posts/${postId}`).catch((e) => { if (e instanceof StaleView) throw e; return null; });
  const main = shell(state.me.primary_companion_id === id ? "mine" : "companions", `
    <div class="row" style="margin-bottom:18px">
      ${avatarImg(c, 72)}
      <div style="flex:1;min-width:200px">
        <h1 style="margin:0">${esc(p.name)} ${esc(p.flag)}</h1>
        <div class="muted">${esc(p.tagline || p.country)}</div>
        ${c.real_person ? `<div class="muted small" style="margin-top:2px">${REAL_NOTE}. It doesn't speak for them.</div>` : ""}
        ${c.retired ? `<div class="muted small" style="margin-top:2px">${RETIRED_NOTE}</div>` : ""}
        <div class="row small" style="margin-top:6px">${p.expertise.map((x) => `<span class="chip">${esc(x)}</span>`).join("")}${c.followers != null ? `<span class="muted" id="followers">${fmtCount(c.followers)} follower${c.followers === 1 ? "" : "s"}</span>` : ""}</div>
        ${!c.is_starter && !c.is_mine ? `<div class="muted small" style="margin-top:4px">Community companion designed by ${esc(c.designer || "a learner")}</div>` : ""}
      </div>
      <div class="row">
        ${canSpeak && p.voice.enabled ? `<button class="btn ghost" id="readAloud" aria-pressed="false">🔈 Read aloud</button>` : ""}
        ${c.is_mine ? `<a class="btn" href="#/studio/${id}">Edit companion</a>${c.hidden ? "" : `<button class="btn ghost" id="pubBtn">${c.is_public ? "Unpublish" : "Publish"}</button>`}` : `<button class="btn ${c.following ? "on" : ""}" id="followBtn">${c.following ? "Following" : "Follow"}</button>${c.real_person ? "" : `<a class="btn" href="#/studio/remix/${id}">Remix</a>`}`}
        ${!c.is_starter && !c.is_mine ? `<button class="btn ghost danger" id="reportBtn">Report</button>` : ""}
      </div>
    </div>
    ${c.moderation ? removalCard(c) : ""}
    <div class="chat-layout">
      <section class="card chat">
        <div class="log" id="log"></div>
        <div id="ctx"></div>
        <form class="composer" id="composer">
          <textarea id="input" rows="1" maxlength="4000" placeholder="Ask ${esc(p.name)} anything about AI…" aria-label="Message"></textarea>
          <button type="button" class="btn ghost mic" id="mic" title="Speak your question" aria-pressed="false">🎤</button>
          <button class="btn primary" id="send">Send</button>
        </form>
      </section>
      <aside class="stack">
        <div class="card">
          <h3>About ${esc(p.name)}</h3>
          <p class="small" style="margin:0">${esc(p.bio)}</p>
          <p class="muted small" style="margin:.6em 0 0">Teaching style: ${label(p.teaching_style)}</p>
        </div>
        <div class="card memory">
          <h3>What ${esc(p.name)} remembers</h3>
          <div id="mem"></div>
        </div>
        <div class="card">
          <h3>Study this conversation</h3>
          <div class="row" id="convStudy">
            <button class="btn" data-study="summary">📄 Summarise</button>
            <button class="btn" data-study="cards">🃏 Flashcards</button>
            <button class="btn" data-study="quiz">📝 Quiz me</button>
          </div>
        </div>
        <div class="card">
          <div class="row spread"><h3 style="margin:0">Recent posts</h3>${c.is_mine ? `<button class="btn" id="genBtn">Write a post</button>` : ""}</div>
          <div id="recent" class="stack small" style="margin-top:10px"></div>
        </div>
      </aside>
    </div>`);

  const log = $("#log");
  const speakBtn = (el, text) => {
    if (!canSpeak || !p.voice.enabled) return;
    const b = document.createElement("button");
    b.className = "linkbtn speak";
    b.type = "button";
    b.title = `Hear ${p.name}`;
    b.textContent = "🔊";
    b.addEventListener("click", () => speak(text, p.voice));
    el.appendChild(b);
  };
  const chatCtx = { name: p.name };
  wireReactions(log, chatCtx);
  const addMsg = (role, text, saved = {}) => {
    const el = chatBubble(role, text, saved, chatCtx);
    if (role === "assistant") speakBtn(el, text);
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return el;
  };
  let readAloud = prefs.get("readAloud", false);
  const readBtn = $("#readAloud");
  const syncRead = () => {
    if (!readBtn) return;
    readBtn.classList.toggle("on", readAloud);
    readBtn.setAttribute("aria-pressed", String(readAloud));
    readBtn.textContent = readAloud ? "🔊 Reading aloud" : "🔈 Read aloud";
  };
  syncRead();
  readBtn?.addEventListener("click", () => { readAloud = !readAloud; prefs.set("readAloud", readAloud); if (!readAloud) speechSynthesis.cancel(); syncRead(); });
  wireMic($("#mic"), $("#input"));
  $("#pubBtn")?.addEventListener("click", async () => { if (await togglePublish(c)) viewCompanion(m); });
  $("#reportBtn")?.addEventListener("click", () => showReport(c));
  $("#appealBtn")?.addEventListener("click", () => showAppeal(c, () => viewCompanion(m)));
  messages.forEach((x) => addMsg(x.role, x.content, x));
  if (!messages.length) openChat();

  /* A chat the learner hasn't started: the companion starts it, from what it knows of their interests. */
  async function openChat() {
    const bubble = addMsg("assistant", "…"), send = $("#send");
    let text = "";
    send.disabled = true;
    try {
      const res = await fetch(`/api/companions/${id}/opening`, {
        method: "POST", headers: { "Content-Type": "application/json" }, credentials: "same-origin",
        body: JSON.stringify({ post_id: contextPost?.id ?? null }),
      });
      if (res.ok && res.status !== 204) {  // 204: begun meanwhile, or being written in another tab
        const reader = res.body.getReader(), dec = new TextDecoder();
        for (;;) {
          const { value, done } = await reader.read();
          if (done || !bubble.isConnected) break;
          text += dec.decode(value, { stream: true });
          bubble.innerHTML = `<div class="prose">${md(text)}</div>`;
          log.scrollTop = log.scrollHeight;
        }
      }
      if (!text.trim() || !bubble.isConnected) { bubble.remove(); return; }
      speakBtn(bubble, text);
      if (readAloud) speak(text, p.voice);
      // Kept as the chat's first message: with its id, the learner can react to it.
      const [first] = await api(`/api/companions/${id}/messages`);
      if (first?.role === "assistant") bubbleSaved(bubble, first, chatCtx);
    } catch {
      if (!text.trim()) bubble.remove();  // the learner can still start the chat themselves
    } finally {
      send.disabled = false;
    }
  }

  const renderCtx = () => {
    $("#ctx").innerHTML = contextPost ? `<div class="context-pill"><span>Asking about: <b>${esc(contextPost.title)}</b></span><button class="linkbtn" id="clearCtx">✕</button></div>` : "";
    $("#clearCtx")?.addEventListener("click", () => { contextPost = null; renderCtx(); });
  };
  renderCtx();

  const memBox = $("#mem");
  const renderMem = (mems) => {
    if (!memBox.isConnected) return;  // memories asked for after a reply, when the learner has moved on
    memBox.innerHTML = mems.length
      ? `<ul>${mems.map((x) => `<li><span>${esc(x.fact)}</span><button class="linkbtn" data-forget="${x.id}" title="Forget this">Forget</button></li>`).join("")}</ul>`
      : `<p class="muted small" style="margin:0">Nothing yet. ${esc(p.name)} picks up your goals and progress as you chat, and you can remove anything here.</p>`;
  };
  renderMem(memories);
  memBox.addEventListener("click", async (e) => {
    const fid = e.target.dataset.forget;
    if (!fid) return;
    await api(`/api/memories/${fid}`, { method: "DELETE" });
    renderMem(await api(`/api/companions/${id}/memories`));
  });

  const renderRecent = (list) => {
    $("#recent").innerHTML = list.length
      ? list.slice(0, 6).map((x) => x.kind ? `<a href="#/p/${x.id}">📘 ${esc(x.title)}</a>`
        : `<a href="#/c/${id}?post=${x.id}" data-ctx="${x.id}">${esc(x.title)}</a>`).join("")
        + (list.some((x) => x.kind) ? `<a class="muted" href="#/guides?companion=${id}">All guides by ${esc(p.name)} →</a>` : "")
      : `<span class="muted">No posts yet.</span>`;
  };
  renderRecent(posts);
  $("#recent").addEventListener("click", (e) => {
    const a = e.target.closest("[data-ctx]");
    if (!a) return;
    e.preventDefault();
    contextPost = posts.find((x) => x.id === +a.dataset.ctx);
    history.replaceState(null, "", a.getAttribute("href"));
    renderCtx();
    $("#input").focus();
  });

  $("#convStudy").addEventListener("click", (e) => {
    const b = e.target.closest("[data-study]");
    if (b) studyAction(b.dataset.study, "conversation", String(id), { companionId: id, button: b });
  });
  $("#followBtn")?.addEventListener("click", async () => {
    try {
      await api(`/api/companions/${id}/follow`, { method: c.following ? "DELETE" : "POST" });
    } catch (err) { toast(err.message); return; }
    // Just the button and the count: redrawing the page would lose the chat's place.
    c.following = !c.following;
    if (c.followers != null) c.followers += c.following ? 1 : -1;
    $("#followBtn").classList.toggle("on", c.following);
    $("#followBtn").textContent = c.following ? "Following" : "Follow";
    if ($("#followers")) $("#followers").textContent = `${fmtCount(c.followers)} follower${c.followers === 1 ? "" : "s"}`;
  });
  $("#genBtn")?.addEventListener("click", async (e) => {
    e.target.disabled = true;
    e.target.textContent = "Writing…";
    try {
      const post = await api(`/api/companions/${id}/posts/generate`, { method: "POST" });
      posts.unshift(post);
      renderRecent(posts);
      toast("New post published to your feed");
    } catch (err) { toast(err.message); }
    e.target.disabled = false;
    e.target.textContent = "Write a post";
  });

  const input = $("#input");
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
  });
  $("#composer").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text || $("#send").disabled) return;  // Enter while the companion is still writing: it waits in the box
    input.value = "";
    $("#send").disabled = true;
    closePicker(log);
    const sentEl = addMsg("user", text);
    const bubble = addMsg("assistant", "…");
    try {
      const { reply, sentId } = await streamChat(id, { message: text, post_id: contextPost?.id ?? null }, (acc) => {
        bubble.innerHTML = `<div class="prose">${md(withoutReactionTags(acc))}</div>`;
        log.scrollTop = log.scrollHeight;
      });
      if (!reply.trim()) bubble.remove();  // the companion only reacted
      else { speakBtn(bubble, reply); if (readAloud) speak(reply, p.voice); }
      savedExchange(id, sentId, sentEl, reply.trim() ? bubble : null, chatCtx);
      contextPost = null;
      renderCtx();
      setTimeout(async () => renderMem(await api(`/api/companions/${id}/memories`)), 2500);
    } catch (err) {
      bubble.innerHTML = `<span class="error">${esc(err.message)}</span>`;
      if (!input.value) input.value = text;  // not sent, so it isn't lost
    } finally {
      $("#send").disabled = false;
      input.focus();
    }
  });
  if (contextPost) input.focus();
}

/* ------------------------------------------------------------------- studio */

const SKIN = ["#FFDBAC", "#F1C27D", "#E0AC69", "#D19A6A", "#C68642", "#A0522D", "#7A4B2A", "#5C3A21", "#3B2414"];
const HAIR = ["#0D0D0D", "#2C1B10", "#4A2C17", "#8B5A2B", "#B8860B", "#E6C27A", "#A52A2A", "#9E9E9E", "#6C5CE7"];
const BG = ["#6C5CE7", "#00A86B", "#BC002D", "#FFB800", "#FF9933", "#2F3640", "#3C3B6E", "#006847", "#E84393", "#0984E3"];
const COUNTRIES = [
  ["Nigeria", "🇳🇬"], ["Japan", "🇯🇵"], ["Brazil", "🇧🇷"], ["India", "🇮🇳"], ["Germany", "🇩🇪"], ["Kenya", "🇰🇪"],
  ["United States", "🇺🇸"], ["Mexico", "🇲🇽"], ["United Kingdom", "🇬🇧"], ["France", "🇫🇷"], ["Canada", "🇨🇦"], ["China", "🇨🇳"],
  ["South Korea", "🇰🇷"], ["Ghana", "🇬🇭"], ["South Africa", "🇿🇦"], ["Egypt", "🇪🇬"], ["Saudi Arabia", "🇸🇦"], ["United Arab Emirates", "🇦🇪"],
  ["Indonesia", "🇮🇩"], ["Philippines", "🇵🇭"], ["Vietnam", "🇻🇳"], ["Australia", "🇦🇺"], ["Spain", "🇪🇸"], ["Italy", "🇮🇹"],
  ["Turkey", "🇹🇷"], ["Pakistan", "🇵🇰"], ["Bangladesh", "🇧🇩"], ["Argentina", "🇦🇷"], ["Colombia", "🇨🇴"], ["Poland", "🇵🇱"],
];
/* Every country the designer knows, with its flag: the common ones above first, then the starter companions'
   own (as the server spells them, e.g. "UAE"), then every other the browser can name. */
let countryList;
function countries() {
  if (countryList) return countryList;
  const known = new Map();
  const add = (name, flag) => { if (!known.has(name.toLowerCase())) known.set(name.toLowerCase(), [name, flag]); };
  COUNTRIES.forEach(([n, f]) => add(n, f));
  Object.entries(state.options?.countries || {}).forEach(([n, f]) => add(n, f));
  try {
    const names = new Intl.DisplayNames(["en"], { type: "region" });
    for (let a = 65; a <= 90; a++) for (let b = 65; b <= 90; b++) {
      const code = String.fromCharCode(a, b), name = names.of(code);
      if (name && name !== code && !/^(Unknown|Pseudo)/.test(name)) add(name, String.fromCodePoint(0x1F1A5 + a, 0x1F1A5 + b));
    }
  } catch { /* a browser without region names: the lists above will do */ }
  return (countryList = [...known.values()].sort((x, y) => x[0].localeCompare(y[0])));
}
const flagFor = (country) => (countries().find(([n]) => n.toLowerCase() === country.trim().toLowerCase()) || [, ""])[1];

const SLIDERS = [
  ["warmth", "Reserved", "Warm"], ["humor", "Serious", "Playful"], ["formality", "Casual", "Formal"],
  ["directness", "Gentle", "Direct"], ["enthusiasm", "Calm", "Energetic"], ["patience", "Brisk", "Patient"],
];
const TEACH_DESC = {
  socratic: "Guides you with questions", step_by_step: "Small steps, checks understanding", storytelling: "Explains through stories",
  examples_first: "Concrete example, then the idea", challenge_driven: "Gives you exercises and feedback",
};

function blankPersona() {
  return {
    name: "", tagline: "", country: "", flag: "", language: "English", bio: "", expertise: [], teaching_style: "step_by_step",
    personality: { warmth: 70, humor: 40, formality: 40, directness: 60, enthusiasm: 60, patience: 80 },
    writing_style: { length: "balanced", level: state.me.level, emoji: "light", uses_analogies: true, uses_local_examples: true, signature_phrase: "", sample_line: "" },
    avatar: { skin: "#C68642", hair_style: "short", hair_color: "#2C1B10", background: "#6C5CE7", accessory: "none", accessory_color: "#222222", expression: "smile", shirt: "#1E272E", use_photo: true },
    voice: { enabled: true, lang: "en-US", rate: 1, pitch: 1 },
    interests: [], custom_instructions: "",
  };
}

const ACCENTS = [
  ["en-US", "English (United States)"], ["en-GB", "English (United Kingdom)"], ["en-NG", "English (Nigeria)"],
  ["en-KE", "English (Kenya)"], ["en-GH", "English (Ghana)"], ["en-ZA", "English (South Africa)"], ["en-IN", "English (India)"],
  ["en-AU", "English (Australia)"], ["en-CA", "English (Canada)"], ["en-IE", "English (Ireland)"], ["en-PH", "English (Philippines)"],
  ["fr-FR", "French"], ["es-ES", "Spanish (Spain)"], ["es-MX", "Spanish (Mexico)"], ["pt-BR", "Portuguese (Brazil)"],
  ["de-DE", "German"], ["it-IT", "Italian"], ["ja-JP", "Japanese"], ["ko-KR", "Korean"], ["zh-CN", "Chinese (Mandarin)"],
  ["hi-IN", "Hindi"], ["ar-SA", "Arabic"], ["sw-KE", "Swahili"], ["yo-NG", "Yoruba"], ["tr-TR", "Turkish"], ["id-ID", "Indonesian"],
];
const PHOTO_TYPES = ["image/png", "image/jpeg", "image/webp"];

async function viewStudio({ mode, id }) {
  state.options ||= await api("/api/options");
  const o = state.options;
  let persona, source = null, versions = [];
  if (mode === "new") persona = blankPersona();
  else {
    source = await api(`/api/companions/${id}`);
    persona = structuredClone(source.persona);
    if (mode === "remix") { persona.name = `My ${persona.name}`.slice(0, 40); persona.custom_instructions = ""; }
    if (mode === "edit") {
      if (!source.is_mine) { location.hash = `#/studio/remix/${id}`; return; }
      versions = await api(`/api/companions/${id}/versions`);
    }
  }
  const title = mode === "edit" ? `Edit ${esc(source.persona.name)}` : mode === "remix" ? `Customise ${esc(source.persona.name)}` : "Design your AI Learning Companion";
  const onboarding = !state.me.onboarded;

  const swatches = (key, colors) => `<div class="swatches" data-key="${key}">${colors.map((c) => `<button type="button" class="swatch" style="background:${c}" data-color="${c}" aria-label="${c}"></button>`).join("")}<input type="color" aria-label="Custom colour"></div>`;
  const choice = (key, values, fmt = label) => `<div class="choice" data-choice="${key}">${values.map((v) => `<button type="button" class="btn" data-v="${v}">${fmt(v)}</button>`).join("")}</div>`;

  const inner = `
    ${onboarding ? `<div class="steps"><span>1. Account ✓</span><span>›</span><b>2. Your companion</b><span>›</span><span>3. Start learning</span></div>` : ""}
    <h1>${title}</h1>
    <p class="muted">${mode === "remix" ? "Start from this companion and make it yours. The original stays as it is." : "Shape every part of your companion. You can come back and change anything later."}</p>
    <div class="studio">
      <form class="stack" id="studio" novalidate>
        <fieldset><legend>Identity</legend>
          <div class="two">
            <label>Name <input type="text" name="name" maxlength="40" required placeholder="e.g. Zara"></label>
            <label>Country and cultural voice
              <input type="text" name="country" list="countries" maxlength="60" required placeholder="e.g. Ghana">
              <datalist id="countries">${countries().map(([n]) => `<option value="${esc(n)}">`).join("")}</datalist>
            </label>
          </div>
          <label>Tagline <span class="hint">One line that sums them up</span><input type="text" name="tagline" maxlength="120"></label>
          <label>Background story <span class="hint">Who they are and why they teach</span><textarea name="bio" maxlength="800"></textarea></label>
          <div class="two">
            <label>Reply language <input type="text" name="language" maxlength="40"></label>
            <label>Interests <span class="hint">Comma separated; used for examples</span><input type="text" name="interests" maxlength="300" placeholder="football, music, cooking"></label>
          </div>
        </fieldset>

        <fieldset><legend>Face</legend>
          <div class="row pic-row" id="picChoice" hidden>
            <span>Show</span>
            <div class="choice" role="group" aria-label="Which picture to show">
              <button type="button" class="btn" data-pic="photo">The photo</button><button type="button" class="btn" data-pic="face">The drawn face</button>
            </div>
          </div>
          <div class="row photo-row">
            <label class="btn" for="photoInput">📷 Upload a photo</label>
            <input type="file" id="photoInput" accept="image/png,image/jpeg,image/webp" hidden>
            <button type="button" class="btn ghost danger" id="photoRemove" hidden>Remove photo</button>
            <span class="muted small">PNG, JPEG or WebP up to 2 MB. Use an image you have the rights to; companions can't pretend to be real people.</span>
          </div>
          <div class="muted small" id="faceHint">Or draw a face:</div>
          <label>Skin tone ${swatches("skin", SKIN)}</label>
          <label>Hair style ${choice("hair_style", o.hair_styles)}</label>
          <label>Hair colour ${swatches("hair_color", HAIR)}</label>
          <label>Expression ${choice("expression", o.expressions)}</label>
          <label>Accessory ${choice("accessory", o.accessories)}</label>
          <div class="three">
            <label>Accessory colour ${swatches("accessory_color", ["#222222", "#F5C542", "#009C3B", "#CE1126", "#0984E3", "#FFFFFF"])}</label>
            <label>Outfit ${swatches("shirt", ["#1E272E", "#F2F2F2", "#DD0000", "#138808", "#0984E3", "#E84393"])}</label>
            <label>Background ${swatches("background", BG.slice(0, 6))}</label>
          </div>
        </fieldset>

        <fieldset><legend>Expertise <span class="muted small" style="font-weight:400">pick up to 5</span></legend>
          <div class="choice" id="expertise">${o.expertise.map((x) => `<button type="button" class="btn" data-v="${esc(x)}">${esc(x)}</button>`).join("")}</div>
        </fieldset>

        <fieldset><legend>Personality</legend>
          ${SLIDERS.map(([k, lo, hi]) => `<div class="slider"><b>${label(k)}</b><span>${lo}</span><input type="range" min="0" max="100" step="5" data-trait="${k}" aria-label="${label(k)}"><span>${hi}</span></div>`).join("")}
        </fieldset>

        <fieldset><legend>Teaching style</legend>
          ${choice("teaching_style", o.teaching_styles, (v) => `${label(v)}`)}
          <div class="muted small" id="teachDesc"></div>
        </fieldset>

        <fieldset><legend>Writing style</legend>
          <div class="three">
            <label>Reply length <select name="ws.length">${o.lengths.map((v) => `<option value="${v}">${label(v)}</option>`).join("")}</select></label>
            <label>Explain at <select name="ws.level">${o.levels.map((v) => `<option value="${v}">${label(v)} level</option>`).join("")}</select></label>
            <label>Emoji <select name="ws.emoji">${o.emoji.map((v) => `<option value="${v}">${label(v)}</option>`).join("")}</select></label>
          </div>
          <label class="check"><input type="checkbox" name="ws.uses_analogies"> Uses analogies and metaphors</label>
          <label class="check"><input type="checkbox" name="ws.uses_local_examples"> Uses examples from their country</label>
          <label>Signature phrase <span class="hint">A catchphrase they use now and then</span><input type="text" name="ws.signature_phrase" maxlength="120"></label>
          <label>A line in their voice <span class="hint">Shows the companion how they sound</span><textarea name="ws.sample_line" maxlength="300"></textarea></label>
        </fieldset>

        <fieldset><legend>Voice</legend>
          ${canSpeak ? "" : `<div class="muted small">Your browser can't read text aloud, but you can still set a voice for other devices.</div>`}
          <label class="check"><input type="checkbox" data-voice="enabled"> Let learners hear this companion read replies aloud</label>
          <div class="three">
            <label>Accent <select data-voice="lang">${ACCENTS.map(([v, t]) => `<option value="${v}">${t}</option>`).join("")}</select></label>
            <label>Speed <input type="range" min="0.5" max="2" step="0.05" data-voice="rate"></label>
            <label>Pitch <input type="range" min="0" max="2" step="0.05" data-voice="pitch"></label>
          </div>
          <div class="row"><button type="button" class="btn" id="voiceTest" ${canSpeak ? "" : "disabled"}>▶ Hear a sample</button>
            <span class="muted small">Uses the voices installed on this device, so it may sound different elsewhere.</span></div>
        </fieldset>

        <fieldset><legend>Custom instructions</legend>
          <label><span class="hint">Private to you: they shape only your own chats, even if you publish this companion. Anything else about how they should teach you. e.g. "Quiz me at the end of every explanation" or "Relate everything to finance, my field"</span>
            <textarea name="custom_instructions" maxlength="1500"></textarea></label>
        </fieldset>

        <div class="error" id="studioErr"></div>
        <div class="row">
          <button class="btn primary" type="submit" id="saveBtn">${mode === "edit" ? "Save changes" : "Create my companion"}</button>
          ${mode === "edit" ? `<a class="btn" href="#/c/${id}">Back to chat</a><button type="button" class="btn danger" id="delBtn">Delete companion</button>` : `<a class="btn" href="${onboarding ? "#/onboarding" : "#/companions"}">Cancel</a>`}
        </div>
      </form>

      <aside class="preview stack">
        <div class="card">
          <img id="pv" class="avatar" width="180" height="180" alt="Avatar preview">
          <h2 id="pvName" style="margin:0"></h2>
          <div class="muted" id="pvCountry"></div>
          <p class="small" id="pvTag"></p>
          <p class="small" id="pvVoice" style="text-align:left"></p>
        </div>
        ${mode === "edit" ? `<div class="card versions"><h3>Version history</h3><ul>${versions.map((v, i) => `<li><span>${new Date(v.created_at).toLocaleString()}${i === 0 ? " · current" : ""}</span>${i === 0 ? "" : `<button type="button" class="linkbtn" data-restore="${v.id}">Restore</button>`}</li>`).join("")}</ul></div>` : ""}
      </aside>
    </div>`;
  const main = state.me.onboarded ? shell("studio", inner) : (app.innerHTML = `<section class="main" style="margin:0 auto">${inner}</section>`, $(".main"));
  const form = $("#studio", main);

  /* bind persona -> controls */
  const set = (path, val) => {
    const [a, b] = path.split(".");
    if (b) persona[a][b] = val; else persona[a] = val;
  };
  const F = (n) => form.elements.namedItem(n);
  for (const k of ["name", "country", "tagline", "bio", "language", "custom_instructions"]) F(k).value = persona[k];
  F("interests").value = persona.interests.join(", ");
  for (const k of ["length", "level", "emoji", "signature_phrase", "sample_line"]) F(`ws.${k}`).value = persona.writing_style[k];
  F("ws.uses_analogies").checked = persona.writing_style.uses_analogies;
  F("ws.uses_local_examples").checked = persona.writing_style.uses_local_examples;

  const syncVoice = () => $$("[data-voice]", form).forEach((el) => {
    const v = persona.voice[el.dataset.voice];
    if (el.type === "checkbox") el.checked = v; else el.value = v;
  });
  if (!ACCENTS.some(([v]) => v === persona.voice.lang)) $("[data-voice=lang]", form).insertAdjacentHTML("afterbegin", `<option value="${esc(persona.voice.lang)}">${esc(persona.voice.lang)}</option>`);
  syncVoice();
  $("#voiceTest").addEventListener("click", () => speak(
    persona.writing_style.sample_line || `Hi! I'm ${persona.name || "your companion"}. ${persona.writing_style.signature_phrase || "Let's learn together."}`,
    { ...persona.voice, enabled: true }));

  /* Photo: kept in the page until saved, then uploaded (or removed) after the companion is stored. A companion
     with a photo shows it or its drawn face, as the learner chooses (avatar.use_photo); the photo is kept either way. */
  // A remix of a starter companion keeps its photo, unless the learner removes it; other photos stay their owner's.
  const photo = { file: null, url: null, remove: false, existing: Boolean(source?.has_photo && (mode === "edit" || source.is_starter)) };
  const hasPhoto = () => Boolean(photo.file || (photo.existing && !photo.remove));
  const showsPhoto = () => hasPhoto() && persona.avatar.use_photo !== false;
  const syncPhoto = () => {
    const has = hasPhoto(), shown = showsPhoto();
    $("#photoRemove").hidden = !has;
    $("#picChoice").hidden = !has;
    $$("#picChoice [data-pic]").forEach((b) => {
      const on = (b.dataset.pic === "photo") === shown;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", on);
    });
    $("#faceHint").textContent = !has ? "Or draw a face:" : shown
      ? "Your companion shows the photo. Change the face below to show a drawn one instead."
      : "Your companion shows the drawn face below. The photo is kept if you want it back.";
    preview();
  };
  // A photo can't change with the controls below, so changing them shows the drawn face.
  const drawFace = () => {
    if (!showsPhoto()) return;
    persona.avatar.use_photo = false;
    syncPhoto();
    toast("Showing the drawn face with your changes");
  };
  $("#picChoice").addEventListener("click", (e) => {
    const b = e.target.closest("[data-pic]");
    if (!b) return;
    persona.avatar.use_photo = b.dataset.pic === "photo";
    syncPhoto();
  });
  $("#photoInput").addEventListener("change", (e) => {
    const f = e.target.files[0];
    e.target.value = "";
    if (!f) return;
    if (!PHOTO_TYPES.includes(f.type)) return toast("Choose a PNG, JPEG or WebP image");
    if (f.size > 2 * 1024 * 1024) return toast("Photos can be up to 2 MB");
    if (photo.url) URL.revokeObjectURL(photo.url);
    photo.file = f; photo.url = URL.createObjectURL(f); photo.remove = false;
    persona.avatar.use_photo = true;
    syncPhoto();
  });
  $("#photoRemove").addEventListener("click", () => {
    if (photo.url) URL.revokeObjectURL(photo.url);
    photo.file = null; photo.url = null; photo.remove = photo.existing;
    syncPhoto();
  });

  form.addEventListener("input", (e) => {
    const t = e.target;
    if (t.dataset.voice) {
      const k = t.dataset.voice;
      persona.voice[k] = t.type === "checkbox" ? t.checked : k === "lang" ? t.value : +t.value;
      return;
    }
    if (t.dataset.trait) persona.personality[t.dataset.trait] = +t.value;
    else if (t.name === "interests") persona.interests = t.value.split(",").map((s) => s.trim()).filter(Boolean).slice(0, 8);
    else if (t.name?.startsWith("ws.")) set(`writing_style.${t.name.slice(3)}`, t.type === "checkbox" ? t.checked : t.value);
    else if (t.name === "country") {
      persona.country = t.value;
      persona.flag = flagFor(t.value);
    } else if (t.name) persona[t.name] = t.value;
    else if (t.type === "color") { persona.avatar[t.closest("[data-key]").dataset.key] = t.value.toUpperCase(); syncSwatches(); drawFace(); }
    preview();
  });
  $$("[data-trait]", form).forEach((r) => (r.value = persona.personality[r.dataset.trait]));

  const syncSwatches = () => $$("[data-key]", form).forEach((g) => {
    const cur = persona.avatar[g.dataset.key];
    $$(".swatch", g).forEach((s) => s.classList.toggle("sel", s.dataset.color.toUpperCase() === cur.toUpperCase()));
    $("input[type=color]", g).value = cur.toLowerCase();
  });
  $$("[data-key]", form).forEach((g) => g.addEventListener("click", (e) => {
    const s = e.target.closest(".swatch");
    if (!s) return;
    persona.avatar[g.dataset.key] = s.dataset.color;
    syncSwatches(); drawFace(); preview();
  }));
  syncSwatches();

  const syncChoices = () => {
    $$("[data-choice]", form).forEach((g) => {
      const key = g.dataset.choice;
      const cur = key === "teaching_style" ? persona.teaching_style : persona.avatar[key];
      $$("[data-v]", g).forEach((b) => b.classList.toggle("on", b.dataset.v === cur));
    });
    $$("#expertise [data-v]", form).forEach((b) => b.classList.toggle("on", persona.expertise.includes(b.dataset.v)));
    $("#teachDesc").textContent = TEACH_DESC[persona.teaching_style] || "";
  };
  $$("[data-choice]", form).forEach((g) => g.addEventListener("click", (e) => {
    const b = e.target.closest("[data-v]");
    if (!b) return;
    if (g.dataset.choice === "teaching_style") persona.teaching_style = b.dataset.v;
    else { persona.avatar[g.dataset.choice] = b.dataset.v; drawFace(); }
    syncChoices(); preview();
  }));
  $("#expertise").addEventListener("click", (e) => {
    const b = e.target.closest("[data-v]");
    if (!b) return;
    const v = b.dataset.v, i = persona.expertise.indexOf(v);
    if (i >= 0) persona.expertise.splice(i, 1);
    else if (persona.expertise.length < 5) persona.expertise.push(v);
    else toast("Pick up to 5 areas");
    syncChoices(); preview();
  });
  syncChoices();

  let pvTimer, pvUrl, pvSeq = 0;
  function preview() {
    $("#pvName").textContent = persona.name || "Your companion";
    $("#pvCountry").textContent = `${persona.flag} ${persona.country || "Choose a country"}`;
    $("#pvTag").textContent = persona.tagline;
    const traits = SLIDERS.map(([k, lo, hi]) => (persona.personality[k] >= 60 ? hi : persona.personality[k] <= 40 ? lo : null)).filter(Boolean);
    $("#pvVoice").innerHTML = `<b>Personality:</b> ${esc(traits.join(", ") || "Balanced")}<br><b>Teaches:</b> ${esc(TEACH_DESC[persona.teaching_style])}<br><b>Writes:</b> ${esc(label(persona.writing_style.length))}, ${esc(persona.writing_style.level)} level${persona.writing_style.sample_line ? `<br><i>“${esc(persona.writing_style.sample_line)}”</i>` : ""}`;
    clearTimeout(pvTimer);
    // A drawn face still on its way doesn't replace what was chosen since, or land on a page opened since.
    const seq = ++pvSeq, img = $("#pv"), current = () => seq === pvSeq && img.isConnected;
    if (showsPhoto()) { img.src = photo.file ? photo.url : `/api/companions/${id}/photo?v=${encodeURIComponent(source.updated_at)}`; return; }
    pvTimer = setTimeout(async () => {
      if (!current()) return;
      const res = await fetch("/api/avatar/preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(persona.avatar) });
      if (!res.ok || !current()) return;
      const blob = await res.blob();
      if (!current()) return;
      if (pvUrl) URL.revokeObjectURL(pvUrl);
      pvUrl = URL.createObjectURL(blob);
      img.src = pvUrl;
    }, 120);
  }
  syncPhoto();

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    $("#studioErr").textContent = "";
    if (!persona.name.trim() || !persona.country.trim()) { $("#studioErr").textContent = "Give your companion a name and a country."; return; }
    $("#saveBtn").disabled = true;
    try {
      let saved;
      if (mode === "edit") saved = await api(`/api/companions/${id}`, { method: "PUT", body: persona });
      else if (mode === "remix") saved = await api(`/api/companions/${id}/remix`, { method: "POST", body: { overrides: persona, make_primary: true, keep_photo: photo.existing && !photo.remove && !photo.file } });
      else saved = await api("/api/companions", { method: "POST", body: { persona, make_primary: true } });
      if (photo.file) {
        const r = await fetch(`/api/companions/${saved.id}/photo`, { method: "PUT", body: photo.file, credentials: "same-origin" });
        if (!r.ok) toast((await r.json().catch(() => ({}))).detail || "The photo couldn't be uploaded");
      } else if (photo.remove && mode === "edit") {
        await api(`/api/companions/${saved.id}/photo`, { method: "DELETE" });
      }
      state.me = await api("/api/me");
      toast(mode === "edit" ? "Changes saved" : `${saved.persona.name} is ready to teach you`);
      location.hash = `#/c/${saved.id}`;
    } catch (err) {
      $("#studioErr").textContent = err.message;
    } finally {
      $("#saveBtn").disabled = false;
    }
  });

  $("#delBtn")?.addEventListener("click", async () => {
    if (!confirm(`Delete ${source.persona.name}? Your chats and memories with them will be removed.`)) return;
    await api(`/api/companions/${id}`, { method: "DELETE" });
    state.me = await api("/api/me");
    toast("Companion deleted");
    location.hash = "#/companions";
  });
  $$("[data-restore]", main).forEach((b) => b.addEventListener("click", async () => {
    await api(`/api/companions/${id}/versions/${b.dataset.restore}/restore`, { method: "POST" });
    toast("Version restored");
    viewStudio({ mode, id });
  }));
}

/* ------------------------------------------------------------------ profile */

const hourLabel = (h) => new Date(2000, 0, 1, h).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

async function viewProfile() {
  // Fresh: AI requests left, and an address confirmed in another tab. Offline, what we have will do.
  state.me = await api("/api/me").catch((err) => { if (err instanceof StaleView) throw err; return state.me; });
  const me = state.me;
  const main = shell("profile", `
    <h1>Your profile</h1>
    <div class="stack" style="max-width:640px">
      <form class="card stack" id="prof">
        <label>Name <input type="text" name="display_name" maxlength="60" required></label>
        <label>AI level <select name="level"><option value="beginner">Beginner</option><option value="intermediate">Intermediate</option><option value="advanced">Advanced</option></select></label>
        <label>Learning goals <span class="hint">Your companions read this to personalise lessons</span><textarea name="goals" maxlength="500"></textarea></label>
        <label>Interests <span class="hint">Hobbies, work, anything you love: your companions start conversations from them and use them in examples</span>
          <input type="text" name="interests" maxlength="300" placeholder="e.g. football, cooking, my job in banking"></label>
        <div class="error" id="profErr"></div>
        <div class="row"><button class="btn primary">Save</button><button type="button" class="btn danger" id="logout">Sign out</button></div>
      </form>
      <div class="card stack" id="aiCard"></div>
      <form class="card stack" id="remForm">
        <div><h2 style="margin:0">Reminders</h2>
          <div class="muted small">Once a day, when something is waiting: a streak to keep, flashcards due, the next step in your study plan.</div></div>
        <label>Time of your daily reminder
          <select name="reminder_hour">${Array.from({ length: 24 }, (_, h) => `<option value="${h}" ${h === me.reminder_hour ? "selected" : ""}>${hourLabel(h)}</option>`).join("")}</select></label>
        <div class="muted small" id="zoneLine" style="margin-top:-8px"></div>
        <label class="check"><input type="checkbox" name="email_reminders" ${me.email_available ? "" : "disabled"} ${me.email_reminders ? "checked" : ""}> Email me the daily reminder</label>
        <div class="muted small" id="emailLine" style="margin-top:-8px"></div>
        <div class="error" id="remErr"></div>
        <div><button class="btn primary">Save reminders</button></div>
        <div class="push-box stack" id="pushBox"></div>
      </form>
      <form class="card stack" id="pwForm">
        <div><h2 style="margin:0">Password and devices</h2><div class="muted small" id="sessionsLine"></div></div>
        <label>Current password <input type="password" name="current_password" autocomplete="current-password" required></label>
        <label>New password <span class="hint">At least 8 characters</span><input type="password" name="new_password" autocomplete="new-password" required minlength="8"></label>
        <div class="error" id="pwErr"></div>
        <div class="row"><button class="btn primary">Change password</button><button type="button" class="btn ghost" id="signOutOthers">Sign out other devices</button></div>
        <div class="muted small">Both sign out every other phone and browser and stop notifications on them.</div>
      </form>
      <div class="card stack">
        <div><h2 style="margin:0">Your data</h2>
          <div class="muted small">Your account, companions, chats, memories, progress and settings, in one file you can keep.</div></div>
        <div class="row"><a class="btn" href="/api/me/export" download>⬇ Download my data</a><button type="button" class="btn danger" id="deleteAccount">Delete my account</button></div>
      </div>
    </div>`);
  const f = $("#prof", main);
  f.display_name.value = me.display_name; f.level.value = me.level; f.goals.value = me.goals; f.interests.value = me.interests || "";
  f.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      state.me = await api("/api/me", { method: "PATCH", body: Object.fromEntries(new FormData(f)) });
      toast("Profile saved");
    } catch (err) { $("#profErr").textContent = err.message; }
  });
  $("#logout").addEventListener("click", async () => {
    await disablePush();
    await api("/api/auth/logout", { method: "POST" });
    state.me = null;
    location.hash = "#/welcome";
  });

  const renderZone = () => {
    $("#zoneLine").innerHTML = `Times are in your time zone, ${esc(state.me.timezone || "UTC")}.`
      + (deviceZone && deviceZone !== state.me.timezone ? ` <button type="button" class="linkbtn" id="useZone">Use this device's (${esc(deviceZone)})</button>` : "");
    $("#useZone")?.addEventListener("click", async () => {
      try { state.me = await api("/api/me", { method: "PATCH", body: { timezone: deviceZone } }); toast("Time zone updated"); renderZone(); }
      catch (err) { toast(err.message); }
    });
  };
  renderZone();
  // What's left of the daily AI allowance. Confirming the email address can raise it.
  const renderAllowance = () => {
    const a = state.me.ai_allowance;
    $("#aiCard").innerHTML = `<div><h2 style="margin:0">AI requests</h2>
      <div class="muted small">Every chat reply, quiz, flashcard deck, summary and study plan is written by AI, and so is each lesson the first time it's taught. Each counts as one request for 24 hours; mock exams and new learning paths count as two.</div></div>`
      + (a.limit === null ? `<div>Your account has no daily limit.</div>`
        : `${bar(a.remaining, a.limit)}<div style="margin-top:-6px"><b>${a.remaining}</b> of ${a.limit} left</div>`)
      + (a.confirm_raises_to ? `<div class="muted small">Confirm your email address to raise your limit to ${a.confirm_raises_to}: `
        + `<button type="button" class="textbtn" data-confirm>send a confirmation link</button>.</div>` : "");
  };
  renderAllowance();
  main.addEventListener("click", async (e) => {  // both "send a confirmation link" buttons
    const b = e.target.closest("[data-confirm]");
    if (!b) return;
    b.disabled = true;
    try {
      const r = await api("/api/email/confirmation", { method: "POST" });
      if (r.confirmed) {  // confirmed meanwhile, perhaps in another tab
        state.me = await api("/api/me");
        renderAllowance();
        renderEmail();
        toast("Your email address is already confirmed.");
      } else toast("Sent. Check your inbox (and spam folder).");
    } catch (err) { if (!(err instanceof StaleView)) toast(err.message); }
    b.disabled = false;
  });
  // Reminder emails only start once the learner confirms the address is theirs.
  const renderEmail = () => {
    const m = state.me, line = $("#emailLine");
    if (!m.email_available) { line.textContent = "Email reminders aren't set up on this server yet."; return; }
    line.innerHTML = m.email_confirmed
      ? `Sent to ${esc(m.email)}, which you've confirmed. Every email has a link to stop them.`
      : `Sent to ${esc(m.email)} once you confirm it's yours: ${m.email_reminders ? "look for the link in your inbox, or " : ""}`
        + `<button type="button" class="textbtn" data-confirm>${m.email_reminders ? "send it again" : "send a confirmation link"}</button>.`;
  };
  renderEmail();
  $("#remForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = { reminder_hour: +e.target.reminder_hour.value };
    if (state.me.email_available) body.email_reminders = e.target.email_reminders.checked;
    const wasOn = state.me.email_reminders;
    try {
      state.me = await api("/api/me", { method: "PATCH", body });
      $("#remErr").textContent = "";
      renderEmail();
      toast(state.me.email_reminders && !wasOn && !state.me.email_confirmed
        ? "Saved. We've emailed you a link: confirm your address to start getting reminder emails." : "Reminders saved");
    } catch (err) { $("#remErr").textContent = err.message; }
  });

  const box = $("#pushBox");
  box.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-push]");
    if (!b) return;
    b.disabled = true;
    try {
      if (b.dataset.push === "on") { await enablePush(); toast("Notifications are on for this device"); }
      else if (b.dataset.push === "off") { await disablePush(); toast("Notifications are off for this device"); }
      else if (b.dataset.push === "test") { await api("/api/push/test", { method: "POST" }); toast("Sent. It should appear in a few seconds."); }
      else if (b.dataset.push === "all-off") {
        await api("/api/push/devices", { method: "DELETE" });
        const sub = await pushSubscription().catch(() => null);
        if (sub) await sub.unsubscribe().catch(() => {});
        toast("Notifications are off on all your devices");
      }
    } catch (err) { if (!(err instanceof StaleView)) toast(err.message); }
    renderPushBox(box);
  });
  renderPushBox(box);

  const renderSessions = async () => {
    const { others } = await api("/api/me/sessions");
    $("#sessionsLine").textContent = others ? `You're also signed in on ${others} other phone${others === 1 ? "" : "s"} or browser${others === 1 ? "" : "s"}.`
      : "You're only signed in here.";
  };
  renderSessions();
  // This device keeps its notifications when the others are signed out.
  const thisDevice = async () => (await pushSubscription().catch(() => null))?.endpoint || null;
  $("#pwForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = e.target;
    if (f.new_password.value.length < 8) { $("#pwErr").textContent = "Use at least 8 characters for the new password."; return; }
    try {
      await api("/api/me/password", { method: "POST", body: {
        current_password: f.current_password.value, new_password: f.new_password.value, keep_device: await thisDevice() } });
      f.reset();
      $("#pwErr").textContent = "";
      toast("Password changed. Other devices are signed out.");
      renderSessions(); renderPushBox(box);
    } catch (err) { $("#pwErr").textContent = err.message; }
  });
  $("#signOutOthers").addEventListener("click", async () => {
    if (!confirm("Sign out every other phone and browser? Notifications stop on them too.")) return;
    try {
      await api("/api/me/sign-out-others", { method: "POST", body: { keep_device: await thisDevice() } });
      toast("Other devices are signed out");
      renderSessions(); renderPushBox(box);
    } catch (err) { toast(err.message); }
  });
  $("#deleteAccount").addEventListener("click", showDeleteAccount);
}

function showDeleteAccount() {
  showDialog(`<h2>Delete your account?</h2>
    <p>This deletes your account and everything in it, for good: the companions you designed (and their posts, which learners following them lose too), your chats, memories, progress, flashcards and settings.</p>
    <p>You can <a href="/api/me/export" download>download your data</a> first.</p>
    <form id="deleteForm" class="stack">
      <label>Your password <input type="password" name="password" autocomplete="current-password" required></label>
      <div class="error" id="deleteErr"></div>
      <button class="btn danger">Delete my account</button>
    </form>`);
  $("#deleteForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await api("/api/me", { method: "DELETE", body: { password: e.target.password.value } });
      const sub = await pushSubscription().catch(() => null);
      if (sub) await sub.unsubscribe().catch(() => {});
      state.me = null;
      $("#dialog").close();
      toast("Your account was deleted.");
      location.hash = "#/welcome";
    } catch (err) { $("#deleteErr").textContent = err.message; }
  });
}

/* Whether this device gets notifications, and how to turn them on where the browser allows it. */
async function renderPushBox(box) {
  const [devices, sub] = await Promise.all([api("/api/push/devices"), pushSubscription().catch(() => null)]);
  if (!document.body.contains(box)) return;
  const here = sub && devices.find((d) => d.endpoint === sub.endpoint);
  const others = devices.filter((d) => d !== here);
  const note = (html) => `<p class="small" style="margin:0">${html}</p>`;
  let status;
  if (!state.me.push_available) status = note(`<span class="muted">Notifications aren't set up on this server yet.</span>`);
  else if (!pushSupported) {
    status = note(isIOS && !installedApp
      ? "On iPhone and iPad, notifications come through the Home Screen app: tap <b>Share</b>, then <b>Add to Home Screen</b>, open AI Every Time from there and turn them on here."
      : `<span class="muted">This browser can't show notifications.</span>`);
  } else if (Notification.permission === "denied") {
    status = note("Notifications are blocked for this site. Allow them in your browser's site settings, then come back here.");
  } else if (here) {
    status = note(`✅ On for this device (${esc(here.device)}).`)
      + `<div class="row"><button type="button" class="btn" data-push="test">Send a test</button><button type="button" class="btn ghost" data-push="off">Turn off on this device</button></div>`;
  } else {
    status = `<div class="row"><button type="button" class="btn primary" data-push="on">🔔 Turn on notifications on this device</button></div>`;
  }
  box.innerHTML = `<h3 style="margin:0">Notifications on your phone or computer</h3>${status}
    ${others.length ? `<div class="muted small">Also on: ${others.map((d) => esc(d.device)).join(", ")}. <button type="button" class="linkbtn" data-push="all-off">Turn off everywhere</button></div>` : ""}`;
}


/* ------------------------------------------------------------ study actions */

/* Start a quiz, make flashcards, or show a summary for a lesson, post or conversation. */
async function studyAction(kind, sourceType, sourceRef, { companionId = null, button = null, examPath = false } = {}) {
  const label = button?.textContent;
  if (button) { button.disabled = true; button.textContent = "Preparing…"; }
  try {
    const companion_id = companionId ?? null;
    if (kind === "quiz" || kind === "exam") {
      const a = await api("/api/assessments", { method: "POST", body: { kind, source_type: sourceType, source_ref: sourceRef, companion_id } });
      location.hash = `#/quiz/${a.id}`;
    } else if (kind === "cards") {
      const d = await api("/api/decks", { method: "POST", body: { source_type: sourceType, source_ref: sourceRef, companion_id, count: 10 } });
      toast(`${d.total} flashcards ready`);
      location.hash = `#/review?deck=${d.id}`;
    } else if (kind === "summary") {
      const s = await api("/api/summaries", { method: "POST", body: { source_type: sourceType, source_ref: sourceRef, companion_id } });
      showDialog(`<h2>${esc(s.title)}</h2><p>${esc(s.summary)}</p><h3>Key takeaways</h3><ul>${s.key_takeaways.map((t) => `<li>${esc(t)}</li>`).join("")}</ul><p><b>Next step:</b> ${esc(s.next_step)}</p>`);
    }
  } catch (err) {
    toast(err.message);
  } finally {
    if (button && document.body.contains(button)) { button.disabled = false; button.textContent = label; }
  }
}

function showReport(c) {
  showDialog(`<h2>Report ${esc(c.persona.name)}</h2>
    <p>Reports go to our moderators. A companion reported by several learners is hidden while it's reviewed.</p>
    <form id="reportForm" class="stack">
      ${[["impersonation", "Pretends to be a real person or organisation"], ["harmful", "Harmful, hateful or unsafe"],
         ["misleading", "Teaches false or misleading information"], ["spam", "Spam or advertising"], ["other", "Something else"]]
        .map(([v, t]) => `<label class="check"><input type="radio" name="reason" value="${v}" required> ${t}</label>`).join("")}
      <button class="btn danger">Send report</button>
    </form>`);
  $("#reportForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await api(`/api/companions/${c.id}/report`, { method: "POST", body: { reason: e.target.reason.value } });
      $("#dialog").close();
      toast("Thanks. Your report was sent.");
    } catch (err) { toast(err.message); }
  });
}

/* For the designer of a removed companion: why, and where their appeal stands. */
function removalCard(c) {
  const m = c.moderation, a = m.appeal;
  const when = m.removed_at ? ` on ${new Date(m.removed_at).toLocaleDateString()}` : "";
  const status = !a ? (m.can_appeal ? `<div><button class="btn" id="appealBtn">Ask for a second look</button></div>` : "")
    : a.status === "open" ? `<div class="small"><b>Your appeal is waiting</b> for a moderator who wasn't part of the first decision.</div>`
    : `<div class="small"><b>Your appeal had a second look, and the removal stands:</b> ${esc(a.decision_note)}</div>`;
  return `<div class="card removal stack">
    <div><b>A moderator removed ${esc(c.persona.name)} from the community${esc(when)}.</b> It's still yours to chat with and edit privately.</div>
    ${m.note ? `<div class="small">Their reason: “${esc(m.note)}”</div>` : ""}
    ${status}
  </div>`;
}

function showAppeal(c, onSent) {
  showDialog(`<h2>Ask for a second look at ${esc(c.persona.name)}</h2>
    <p>A different moderator will review the removal. Tell them what you think was missed, such as how the companion is meant to be used or what you've changed since.</p>
    <form id="appealForm" class="stack">
      <label>Your appeal <textarea name="message" rows="5" minlength="10" maxlength="1000" required></textarea></label>
      <div class="error" id="appealErr"></div>
      <button class="btn primary">Send appeal</button>
    </form>`);
  $("#appealForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const nav = state.nav;
    try {
      await api(`/api/companions/${c.id}/appeal`, { method: "POST", body: { message: e.target.message.value.trim() } });
      $("#dialog").close();
      toast("Appeal sent. You'll get a notice when it's decided.");
      if (nav === state.nav) onSent();  // not if the learner has moved to another page meanwhile
    } catch (err) { $("#appealErr").textContent = err.message; }
  });
}

function showDialog(html) {
  let d = $("#dialog");
  if (!d) {
    d = document.createElement("dialog");
    d.id = "dialog";
    d.className = "card dialog";
    document.body.appendChild(d);
    d.addEventListener("click", (e) => { if (e.target === d || e.target.closest("[data-close]")) d.close(); });
  }
  d.innerHTML = `<div class="prose">${html}</div><div class="row" style="justify-content:flex-end"><button class="btn primary" data-close>Done</button></div>`;
  d.showModal();
}

const bar = (done, total) => `<div class="bar" role="progressbar" aria-valuenow="${done}" aria-valuemax="${total}"><span style="width:${total ? Math.round((100 * done) / total) : 0}%"></span></div>`;
const STATUS_ICON = { completed: "✅", started: "🟣", not_started: "⚪" };

/* -------------------------------------------------------------------- learn */

function pathBadge(p) {
  if (!p.custom) return "";
  if (!p.mine) return `<div class="small custom-badge ok">✔ Expert reviewed · suggested by ${esc(p.suggested_by)}</div>`;
  const text = {
    private: "Made for you by your companion · not reviewed by experts, so double-check key facts",
    pending: "Waiting for expert review before it's shared",
    approved: "✔ Expert reviewed and shared with every learner",
    rejected: `Not approved for sharing: ${esc(p.review_note)}`,
  }[p.status];
  return `<div class="small custom-badge ${p.status === "approved" ? "ok" : p.status === "rejected" ? "bad" : ""}">${text}</div>`;
}

async function viewLearn() {
  const paths = await api("/api/paths");
  const main = shell("learn", `
    <h1>Learning paths</h1>
    <p class="muted">Short paths taught by your companion, in their own voice. Finish lessons, test yourself, and take a mock exam at the end of each path.</p>
    <a class="card row spread plain courses-pointer" href="#/courses"><div><b>🎓 Looking for full courses?</b>
      <div class="muted small">Courses with hands-on labs, projects and certificates, taught by the AI Learning Companions.</div></div><span class="btn">Browse courses</span></a>
    <form class="card stack" id="newPath">
      <div><h3 style="margin:0">✨ Learn anything: ask your companion for a path</h3>
        <div class="muted small">Name a topic and your companion designs a step-by-step path just for you.</div></div>
      <div class="new-path-row">
        <input type="text" name="topic" minlength="3" maxlength="120" required placeholder="e.g. AI for small farms, computer vision for healthcare, LLMs for lawyers" aria-label="Topic">
        <select name="level" aria-label="Level">${["beginner", "intermediate", "advanced"].map((v) => `<option value="${v}" ${v === state.me.level ? "selected" : ""}>${label(v)}</option>`).join("")}</select>
        <select name="lessons" aria-label="Number of lessons">${[3, 4, 5, 6, 8].map((n) => `<option value="${n}" ${n === 5 ? "selected" : ""}>${n} lessons</option>`).join("")}</select>
        <button class="btn primary">Design my path</button>
      </div>
    </form>
    <div class="stack">
      ${paths.map((p) => `
        <section class="card stack" data-path="${p.id}">
          <div class="row spread">
            <div><h2 style="margin:0">${esc(p.title)}</h2><div class="muted small">${esc(label(p.level))} · ${p.total} lessons · ${p.completed}/${p.total} done</div>
              ${pathBadge(p)}</div>
            <div class="row">
              ${p.custom && p.mine && ["private", "rejected"].includes(p.status) ? `<button class="btn ghost" data-act="submit">Share with community</button>` : ""}
              ${p.custom && p.mine && p.status === "pending" ? `<button class="btn ghost" data-act="withdraw">Withdraw from review</button>` : ""}
              ${p.custom && p.mine && p.status !== "approved" ? `<button class="btn ghost danger" data-act="delete">Delete</button>` : ""}
              <button class="btn" data-act="cards">🃏 Flashcards</button>
              <button class="btn primary" data-act="exam">🎓 Mock exam</button>
            </div>
          </div>
          <p style="margin:0">${esc(p.description)}</p>
          ${bar(p.completed, p.total)}
          <ol class="lessons">
            ${p.lessons.map((l) => `<li><a href="#/lesson/${l.id}">${STATUS_ICON[l.status]} ${esc(l.title)}</a>
              <span class="muted small">${l.minutes} min${l.best_score != null ? ` · best quiz ${l.best_score}%` : ""}</span></li>`).join("")}
          </ol>
        </section>`).join("")}
    </div>`);
  $$("[data-path]", main).forEach((el) => el.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-act]");
    if (!b) return;
    if (b.dataset.act === "submit" || b.dataset.act === "withdraw") {
      if (b.dataset.act === "submit" && !confirm("Send this path to an expert reviewer? If approved, every learner can take it, " +
        "and it will show you as the person who suggested it. The reviewer may correct its key facts.")) return;
      try {
        await api(`/api/custom-paths/${el.dataset.path}/${b.dataset.act}`, { method: "POST" });
        toast(b.dataset.act === "submit" ? "Sent for expert review" : "Withdrawn from review");
      } catch (err) { toast(err.message); }
      return viewLearn();
    }
    if (b.dataset.act === "delete") {
      if (!confirm("Delete this path? Its lesson progress goes too. Quiz results stay in your history.")) return;
      await api(`/api/custom-paths/${el.dataset.path}`, { method: "DELETE" });
      toast("Path deleted");
      return viewLearn();
    }
    studyAction(b.dataset.act === "exam" ? "exam" : "cards", "path", el.dataset.path, { button: b });
  }));
  $("#newPath", main).addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = e.target, btn = $("button", f);
    btn.disabled = true; btn.textContent = "Designing…";
    try {
      const path = await api("/api/custom-paths", { method: "POST", body: { topic: f.topic.value.trim(), level: f.level.value, lessons: +f.lessons.value } });
      toast(`"${path.title}" is ready`);
      if (document.body.contains(f)) viewLearn();
    } catch (err) {
      toast(err.message);
      btn.disabled = false; btn.textContent = "Design my path";
    }
  });
}

async function viewLesson(m) {
  const lessonId = m[1], companionId = m[2] ? +m[2] : null;
  const inCourse = lessonId.match(/^course-([a-z0-9-]+)-(\d+)-(\d+)$/);
  if (inCourse) { location.replace(`#/course/${inCourse[1]}/learn/l${inCourse[2]}-${inCourse[3]}`); return; }
  const main = shell("learn", `<div class="card empty">Your companion is preparing this lesson…</div>`);
  let lesson, list;
  try {
    [lesson, list] = await Promise.all([
      api(`/api/lessons/${lessonId}${companionId ? `?companion_id=${companionId}` : ""}`),
      api("/api/companions"),
    ]);
  } catch (err) {  // e.g. the daily AI requests are used up, or the companion couldn't write it
    if (err instanceof StaleView) throw err;
    main.innerHTML = `<div class="card empty stack"><div>${esc(err.message)}</div><div><a class="btn" href="#/learn">Back to learning paths</a></div></div>`;
    return;
  }
  const t = lesson.teacher;
  const teachers = list.filter((c) => c.following || c.owner_id === state.me.id);
  main.innerHTML = `
    <div class="muted small"><a href="#/learn">Learning paths</a> › ${esc(lesson.path.title)}</div>
    <div class="row spread" style="margin:8px 0 16px">
      <h1 style="margin:0">${esc(lesson.title)}</h1>
      <label class="row small" style="font-weight:500">Taught by
        <select id="teacher" style="width:auto">${teachers.map((c) => `<option value="${c.id}" ${c.id === t.id ? "selected" : ""}>${esc(c.persona.name)} ${esc(c.persona.flag)}</option>`).join("")}</select>
      </label>
    </div>
    <div class="lesson-layout">
      <article class="card stack">
        <div class="row">
          <img class="avatar" src="/api/companions/${t.id}/avatar?v=${encodeURIComponent(t.updated_at)}" width="44" height="44" alt="">
          <div><b>${esc(t.name)} ${esc(t.flag)}</b><div class="muted small">${lesson.minutes} min lesson${lesson.best_score != null ? ` · your best quiz: ${lesson.best_score}%` : ""}</div></div>
        </div>
        <div class="prose">${md(lesson.body)}</div>
        <div class="row">
          <button class="btn ${lesson.status === "completed" ? "on" : "primary"}" id="complete">${lesson.status === "completed" ? "✅ Completed" : "Mark as complete"}</button>
          <button class="btn" data-study="quiz">📝 Quiz me</button>
          <button class="btn" data-study="cards">🃏 Flashcards</button>
          <button class="btn" data-study="summary">📄 Summary</button>
        </div>
        <div class="row spread">
          ${lesson.prev ? `<a class="btn ghost" href="#/lesson/${lesson.prev}">← Previous</a>` : "<span></span>"}
          ${lesson.next ? `<a class="btn ghost" href="#/lesson/${lesson.next}">Next lesson →</a>` : `<a class="btn ghost" href="#/learn">Back to paths</a>`}
        </div>
      </article>
      <aside class="card chat tutor">
        <div class="row" style="padding:14px 16px 0"><h3 style="margin:0">Ask ${esc(t.name)} about this lesson</h3></div>
        <div class="log" id="tlog"><div class="msg assistant"><div class="prose"><p>Stuck on anything here, or want another example? Ask me.</p></div></div></div>
        <form class="composer" id="tform">
          <textarea id="tinput" rows="1" maxlength="4000" placeholder="Ask a question…" aria-label="Question"></textarea>
          <button class="btn primary" id="tsend">Ask</button>
        </form>
      </aside>
    </div>`;
  $("#teacher").addEventListener("change", (e) => (location.hash = `#/lesson/${lessonId}?companion=${e.target.value}`));
  $("#complete").addEventListener("click", async () => {
    await api(`/api/lessons/${lessonId}/complete`, { method: "POST" });
    toast("Lesson completed");
    if (lesson.next) location.hash = `#/lesson/${lesson.next}${companionId ? `?companion=${companionId}` : ""}`;
    else location.hash = "#/learn";
  });
  $$("[data-study]", main).forEach((b) => b.addEventListener("click", () => studyAction(b.dataset.study, "lesson", lessonId, { companionId: t.id, button: b })));

  const log = $("#tlog"), input = $("#tinput"), tutorCtx = { name: t.name };
  wireReactions(log, tutorCtx);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#tform").requestSubmit(); } });
  $("#tform").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    $("#tsend").disabled = true;
    closePicker(log);
    const u = chatBubble("user", text, {}, tutorCtx); log.appendChild(u);
    const a = chatBubble("assistant", "…", {}, tutorCtx); log.appendChild(a);
    log.scrollTop = log.scrollHeight;
    try {
      const { reply, sentId } = await streamChat(t.id, { message: text, lesson_id: lessonId }, (acc) => { a.innerHTML = `<div class="prose">${md(withoutReactionTags(acc))}</div>`; log.scrollTop = log.scrollHeight; });
      if (!reply.trim()) a.remove();  // the companion only reacted
      savedExchange(t.id, sentId, u, reply.trim() ? a : null, tutorCtx);
    } catch (err) {
      a.innerHTML = `<span class="error">${esc(err.message)}</span>`;
      if (!input.value) input.value = text;  // not sent, so it isn't lost
    }
    $("#tsend").disabled = false;
  });
}

/* ------------------------------------------------------------ quizzes/exams */

async function viewQuiz(m) {
  const id = +m[1];
  const a = await api(`/api/assessments/${id}`);
  if (a.submitted_at) return renderResults(a);
  const exam = a.kind === "exam";
  const main = shell("practice", `
    <div class="row spread"><h1 style="margin:0">${esc(a.title)}</h1>${exam ? `<div class="timer" id="timer"></div>` : ""}</div>
    <p class="muted">${a.total} questions${exam ? ` · ${a.time_limit_minutes} minutes · pass mark 70%` : ""}</p>
    <form id="quiz" class="stack">
      ${a.questions.map((q, i) => `
        <fieldset class="card question">
          <legend><b>${i + 1}.</b> ${esc(q.question)}</legend>
          ${q.options.map((o, j) => `<label class="option"><input type="radio" name="q${i}" value="${j}"> <span>${esc(o)}</span></label>`).join("")}
        </fieldset>`).join("")}
      <div class="row"><button class="btn primary" id="submitQuiz">Submit answers</button><span class="muted small" id="answered"></span></div>
    </form>`);
  const form = $("#quiz", main);
  const count = () => ($("#answered").textContent = `${a.questions.filter((_, i) => form.querySelector(`[name=q${i}]:checked`)).length} of ${a.total} answered`);
  form.addEventListener("change", count); count();
  let timer;
  const submit = async () => {
    clearInterval(timer);
    const answers = a.questions.map((_, i) => { const c = form.querySelector(`[name=q${i}]:checked`); return c ? +c.value : null; });
    $("#submitQuiz").disabled = true;
    try {
      const result = await api(`/api/assessments/${id}/submit`, { method: "POST", body: { answers } });
      if (document.body.contains(form)) renderResults(result);
    }
    catch (err) { toast(err.message); $("#submitQuiz").disabled = false; }
  };
  form.addEventListener("submit", (e) => { e.preventDefault(); submit(); });
  if (exam) {
    const end = new Date(a.created_at).getTime() + a.time_limit_minutes * 60000;
    const tick = () => {
      const left = Math.max(0, end - Date.now());
      $("#timer").textContent = `⏱ ${Math.floor(left / 60000)}:${String(Math.floor((left % 60000) / 1000)).padStart(2, "0")}`;
      if (!left) { toast("Time's up! Submitting your exam."); submit(); }
    };
    tick(); timer = setInterval(() => { if (!document.body.contains($("#timer"))) return clearInterval(timer); tick(); }, 1000);
  }
}

/* From a course's quiz or mock exam, back to where it sits in the course. */
function courseBack(r) {
  const module = r.source_type === "path" && r.source_ref.match(/^course-([a-z0-9-]+)-(\d+)$/);
  if (module) return `<a class="btn" href="#/course/${module[1]}/learn/quiz${module[2]}">Back to the course</a>`;
  return r.source_type === "course" ? `<a class="btn" href="#/course/${r.source_ref}/learn/grades">Back to the course</a>` : "";
}

function renderResults(r) {
  const main = shell("practice", `
    <h1>${esc(r.title)}</h1>
    <div class="card row spread result ${r.percent >= 70 ? "pass" : "fail"}">
      <div><div class="big">${r.percent}%</div><div class="muted">${r.score} of ${r.total} correct${r.over_time ? " · submitted after the time limit" : ""}</div></div>
      <div><b>${r.percent >= 70 ? "Passed. Nice work!" : "Keep going: review and try again."}</b></div>
    </div>
    ${r.by_lesson ? `<div class="card"><h3>By lesson</h3>${r.by_lesson.map((b) => `<div class="row spread small"><a href="#/lesson/${b.lesson_id}">${esc(b.title)}</a><span>${b.correct}/${b.total}</span></div>${bar(b.correct, b.total)}`).join("")}</div>` : ""}
    <div class="stack" style="margin-top:16px">
      ${r.questions.map((q, i) => `
        <div class="card question ${q.correct ? "right" : "wrong"}">
          <b>${i + 1}. ${esc(q.question)}</b>
          <ul class="options">${q.options.map((o, j) => `<li class="${j === q.answer_index ? "correct" : j === q.your_answer ? "chosen" : ""}">${j === q.answer_index ? "✔" : j === q.your_answer ? "✘" : "•"} ${esc(o)}</li>`).join("")}</ul>
          ${q.your_answer == null ? `<div class="muted small">Not answered</div>` : ""}
          <div class="small explain">${esc(q.explanation)}</div>
        </div>`).join("")}
    </div>
    <div class="row" style="margin-top:16px">
      <button class="btn primary" id="retry">Try a new ${r.kind === "exam" ? "exam" : "quiz"}</button>
      ${r.source_type === "lesson" ? `<a class="btn" href="#/lesson/${r.source_ref}">Back to lesson</a>` : ""}
      ${courseBack(r)}
      <a class="btn" href="#/practice">All practice</a>
    </div>`);
  $("#retry", main).addEventListener("click", (e) => studyAction(r.kind, r.source_type, r.source_ref, { companionId: r.companion_id, button: e.target }));
}

/* ----------------------------------------------------------------- practice */

async function viewPractice() {
  const [due, decks, quizzes] = await Promise.all([api("/api/review?limit=100"), api("/api/decks"), api("/api/assessments")]);
  const main = shell("practice", `
    <h1>Practice</h1>
    <div class="card row spread">
      <div><h2 style="margin:0">🃏 ${due.length} card${due.length === 1 ? "" : "s"} due</h2><div class="muted small">Spaced repetition brings each card back just before you'd forget it.</div></div>
      ${due.length ? `<a class="btn primary" href="#/review">Start review</a>` : `<a class="btn" href="#/learn">Make cards from a lesson</a>`}
    </div>
    <h2 style="margin-top:24px">Flashcard decks</h2>
    ${decks.length ? `<div class="grid">${decks.map((d) => `
      <div class="card stack" data-deck="${d.id}">
        <b>${esc(d.title)}</b>
        <div class="muted small">${d.total} cards · ${d.due} due · ${d.mastered} mastered</div>
        ${bar(d.mastered, d.total)}
        <div class="row"><a class="btn" href="#/review?deck=${d.id}">Review</a><button class="btn ghost danger" data-del>Delete</button></div>
      </div>`).join("")}</div>` : `<div class="card empty">No decks yet. Make flashcards from any lesson, post or conversation.</div>`}
    <h2 style="margin-top:24px">Quizzes and exams</h2>
    ${quizzes.length ? `<div class="card"><table class="table"><thead><tr><th>Title</th><th>Score</th><th>Date</th></tr></thead><tbody>
      ${quizzes.map((q) => `<tr><td><a href="#/quiz/${q.id}">${esc(q.title)}</a></td><td>${q.percent != null ? `${q.percent}%` : `<span class="muted">Not finished</span>`}</td><td class="muted small">${timeAgo(q.created_at)}</td></tr>`).join("")}
    </tbody></table></div>` : `<div class="card empty">No quizzes yet. Every lesson and post has a "Quiz me" button.</div>`}`);
  $$("[data-deck]", main).forEach((el) => $("[data-del]", el).addEventListener("click", async () => {
    if (!confirm("Delete this deck?")) return;
    await api(`/api/decks/${el.dataset.deck}`, { method: "DELETE" });
    viewPractice();
  }));
}

async function viewReview(m) {
  const deckId = m[1] ? +m[1] : null;
  const cards = await api(`/api/review${deckId ? `?deck_id=${deckId}` : ""}`);
  let i = 0, reviewed = 0;
  const main = shell("practice", `<h1>Flashcard review</h1><div id="rv"></div>`);
  const render = () => {
    const card = cards[i];
    if (!card) {
      $("#rv").innerHTML = `<div class="card empty"><h2>All caught up 🎉</h2><p>You reviewed ${reviewed} card${reviewed === 1 ? "" : "s"}. They'll come back when it's time.</p><a class="btn primary" href="#/practice">Back to practice</a></div>`;
      return;
    }
    $("#rv").innerHTML = `
      <div class="muted small">${esc(card.deck_title)} · ${i + 1} of ${cards.length}</div>
      <button class="card flashcard" id="flip" aria-live="polite"><div class="front">${esc(card.front)}</div><div class="back" hidden>${esc(card.back)}</div><div class="muted small hint">Tap to reveal</div></button>
      <div class="row grades" hidden>
        <button class="btn danger" data-g="again">Again</button><button class="btn" data-g="good">Good</button><button class="btn primary" data-g="easy">Easy</button>
      </div>`;
    $("#flip").addEventListener("click", () => { $(".back").hidden = false; $(".hint").hidden = true; $(".grades").hidden = false; });
    $$("[data-g]").forEach((b) => b.addEventListener("click", async () => {
      await api(`/api/cards/${card.id}/review`, { method: "POST", body: { grade: b.dataset.g } });
      reviewed++; i++;
      if (document.body.contains(main)) render();
    }));
  };
  render();
}

/* ----------------------------------------------------------------- progress */

async function viewProgress() {
  const [p, plan, paths, cat] = await Promise.all([api("/api/progress"), api("/api/study-plan"), api("/api/paths"), api("/api/courses")]);
  const myCourses = cat.courses.filter((c) => c.enrolled || c.certificate);
  const certificates = [...cat.tracks.filter((t) => t.certificate).map((t) => ({ code: t.certificate, title: `${t.title} ${t.kind}` })),
    ...cat.courses.filter((c) => c.certificate).map((c) => ({ code: c.certificate, title: c.title }))];
  const tile = (v, t) => `<div class="card tile"><div class="big">${v}</div><div class="muted small">${t}</div></div>`;
  const ACT = { lesson: "📖 Lesson", quiz: "📝 Quiz", flashcards: "🃏 Flashcards", exam: "🎓 Mock exam" };
  const main = shell("progress", `
    <h1>Your progress</h1>
    <div class="tiles">
      ${tile(`🔥 ${p.streak_days}`, `day streak`)}
      ${tile(`${p.lessons_completed}/${p.lessons_total}`, "lessons completed")}
      ${tile(p.quiz_average != null ? `${p.quiz_average}%` : "–", `quiz average (${p.quizzes_taken} taken)`)}
      ${tile(p.best_exam != null ? `${p.best_exam}%` : "–", `best mock exam (${p.exams_taken} taken)`)}
      ${tile(`${p.cards_mastered}/${p.cards_total}`, `cards mastered · ${p.cards_due} due`)}
      ${tile(p.questions_asked, "questions asked")}
    </div>
    <div class="two-col">
      <div class="stack">
        <div class="card"><h3>Last 7 days</h3><div class="week">${p.last_7_days.map((d) => `<div class="day ${d.active ? "on" : ""}" title="${d.day}"><span>${new Date(d.day + "T12:00:00Z").toLocaleDateString(undefined, { weekday: "short" })}</span></div>`).join("")}</div></div>
        ${p.next_lesson ? `<div class="card row spread"><div><div class="muted small">Up next · ${esc(p.next_lesson.path_title)}</div><b>${esc(p.next_lesson.title)}</b></div><a class="btn primary" href="#/lesson/${p.next_lesson.lesson_id}">Continue</a></div>` : ""}
        <div class="card"><div class="row spread"><h3 style="margin:0">Your courses</h3><a class="small" href="#/courses">All courses</a></div>
          ${myCourses.length ? myCourses.map((c) => `<div class="row spread small" style="margin-top:10px"><a href="#/course/${c.slug}">${esc(c.title)}</a>
            ${c.certificate ? `<a href="#/certificate/${c.certificate}">🎓 Certificate</a>` : `<a href="#/course/${c.slug}/learn/${c.next || "grades"}">Continue</a>`}</div>${bar(c.progress, 100)}`).join("")
            : `<p class="muted small" style="margin:8px 0 0">You haven't enrolled in a course yet. <a href="#/courses">Find one</a>: they're free.</p>`}
          ${certificates.length ? `<h3 style="margin:14px 0 6px">Certificates</h3>${certificates.map((x) => `<div class="small"><a href="#/certificate/${x.code}">🎓 ${esc(x.title)}</a></div>`).join("")}` : ""}
        </div>
        <div class="card"><h3>Paths</h3>${p.paths.map((x) => `<div class="row spread small"><span>${esc(x.title)}</span><span>${x.completed}/${x.total}</span></div>${bar(x.completed, x.total)}`).join("")}</div>
        ${p.needs_review.length ? `<div class="card"><h3>Worth reviewing</h3>${p.needs_review.map((x) => `<div class="row spread small"><a href="#/lesson/${x.lesson_id}">${esc(x.title)}</a><span class="muted">best ${x.best_score}%</span></div>`).join("")}</div>` : ""}
      </div>
      <div class="card stack" id="planCard">
        <div class="row spread"><h2 style="margin:0">Study plan</h2>${plan ? `<span class="muted small">${plan.done}/${plan.total} done</span>` : ""}</div>
        ${plan ? `
          <p style="margin:0">${esc(plan.overview)}</p>
          ${bar(plan.done, plan.total)}
          ${plan.weeks.map((w) => `<div class="plan-week"><h4>Week ${w.week}: ${esc(w.focus)} ${w.done ? "✅" : ""}</h4><ul>${w.items.map((it) => `
            <li class="${it.done ? "done" : ""}"><button class="linkbtn plan-item" data-act="${it.activity}" data-lesson="${it.lesson_id}" data-path="${it.path_id}">${it.done ? "✅" : "⬜"} ${ACT[it.activity]}: ${esc(it.title)}</button><div class="muted small">${esc(it.note)}</div></li>`).join("")}</ul></div>`).join("")}
          <details><summary>Make a new plan</summary>${planForm(paths, plan.weekly_minutes)}</details>` : `
          <p class="muted" style="margin:0">Tell your companion how much time you have, and they'll plan your weeks around your goals and progress.</p>${planForm(paths, 120)}`}
      </div>
    </div>`);
  $$(".plan-item", main).forEach((b) => b.addEventListener("click", () => {
    const act = b.dataset.act;
    if (act === "lesson") location.hash = `#/lesson/${b.dataset.lesson}`;
    else if (act === "exam") studyAction("exam", "path", b.dataset.path, { button: b });
    else studyAction(act === "flashcards" ? "cards" : "quiz", "lesson", b.dataset.lesson, { button: b });
  }));
  $("#planForm", main).addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = e.target, btn = $("button", f);
    const body = { weekly_minutes: +f.weekly_minutes.value, weeks: +f.weeks.value, paths: $$("[name=paths]:checked", f).map((c) => c.value) };
    btn.disabled = true; btn.textContent = "Planning…";
    try {
      await api("/api/study-plan", { method: "POST", body });
      toast("Your study plan is ready");
      if (document.body.contains(f)) viewProgress();
    }
    catch (err) { toast(err.message); btn.disabled = false; btn.textContent = "Create my plan"; }
  });
}

function planForm(paths, minutes) {
  return `<form id="planForm" class="stack" style="margin-top:10px">
    <div class="two">
      <label>Time per week <select name="weekly_minutes">${[30, 60, 120, 180, 300, 600].map((v) => `<option value="${v}" ${v === minutes ? "selected" : ""}>${v < 60 ? `${v} minutes` : `${v / 60} hour${v === 60 ? "" : "s"}`}</option>`).join("")}</select></label>
      <label>Length <select name="weeks">${[1, 2, 4, 6, 8, 12].map((v) => `<option value="${v}" ${v === 4 ? "selected" : ""}>${v} week${v === 1 ? "" : "s"}</option>`).join("")}</select></label>
    </div>
    <div><b class="small">Focus on</b> <span class="muted small">(none selected = all paths)</span>
      <div class="choice" style="margin-top:6px">${paths.map((p) => `<label class="check small"><input type="checkbox" name="paths" value="${p.id}"> ${esc(p.title)}</label>`).join("")}</div>
    </div>
    <button class="btn primary">Create my plan</button>
  </form>`;
}


/* -------------------------------------------------------------------- staff */

const ROLE_CAN = { moderator: ["reports", "appeals", "log"], reviewer: ["paths", "log"], admin: ["insights", "reports", "appeals", "paths", "roles", "log"] };
const REASON_LABEL = { impersonation: "Impersonation", harmful: "Harmful", misleading: "Misleading", spam: "Spam", other: "Other" };
const lines = (v) => v.split("\n").map((x) => x.trim()).filter(Boolean);

async function viewStaff(m) {
  const tabs = ROLE_CAN[state.me.role] || [];
  if (!tabs.length) { location.hash = "#/feed"; return; }
  const tab = tabs.includes(m[1]) ? m[1] : tabs[0];
  const names = { insights: "Insights", reports: "Reported companions", appeals: "Appeals", paths: "Path reviews", roles: "People", log: "Decision log" };
  const main = shell("staff", `
    <h1>Staff</h1>
    <div class="tabs">${tabs.map((t) => `<a class="btn ${t === tab ? "on" : ""}" href="#/staff/${t}">${names[t]}</a>`).join("")}</div>
    <div id="staffBody" class="stack"><div class="card empty">Loading…</div></div>`);
  const body = $("#staffBody", main);
  await { insights: staffInsights, reports: staffReports, appeals: staffAppeals, paths: staffPaths, roles: staffRoles, log: staffLog }[tab](body, m);
}

/* Insights (admins): learners, how they learn, and the AI the site uses against its allowances. */
const KIND_LABEL = {
  chat: "Chat replies", lesson: "Lessons, the first time taught", quiz: "Quizzes", exam: "Mock exams", flashcards: "Flashcard decks",
  summary: "Summaries", plan: "Study plans", path: "Custom paths", post: "Companion posts", automatic: "Scheduled posts (no learner)",
  opening: "Companions starting chats (free to learners)", lab: "Course lab reviews", project: "Course project reviews",
};
const fmt = (n) => (n == null ? "–" : n >= 10000 ? new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(n) : n.toLocaleString());
const plural = (n, one, many = `${one}s`) => `${fmt(n)} ${n === 1 ? one : many}`;
const utcDate = (day, opts) => new Date(`${day}T12:00:00Z`).toLocaleDateString(undefined, { timeZone: "UTC", ...opts });
const longDay = (day) => utcDate(day, { weekday: "short", day: "numeric", month: "short" });
const money = (v) => (v >= 100 ? `$${Math.round(v).toLocaleString()}` : v >= 1 ? `$${v.toFixed(2)}` : v > 0 ? `$${v.toPrecision(2)}` : "$0");

async function staffInsights(body) {
  const { learners: l, learning: s, community: c, ai } = await api("/api/staff/insights");
  const tile = (value, label, sub = "") => `<div class="card tile"><div class="big">${value}</div><div class="small">${label}</div>${sub ? `<div class="muted small">${sub}</div>` : ""}</div>`;
  const share = (n) => (l.total ? `${Math.round((100 * n) / l.total)}% of learners` : "");
  const sum = (rows, key) => rows.reduce((t, r) => t + (r[key] || 0), 0);
  const away = ai.turned_away, lim = ai.limits, fallbacks = sum(ai.kinds, "fallbacks");
  const limits = (lim.allowance > 0
    ? `Each learner can make ${fmt(lim.allowance)} AI requests in any 24 hours${lim.unconfirmed != null && lim.unconfirmed < lim.allowance ? ` (${fmt(lim.unconfirmed)} until they confirm their email address)` : ""}; admins have no limit.`
    : "Learners have no daily limit on AI requests.")
    + (lim.site > 0 ? ` Everyone together can use ${fmt(lim.site)} in any 24 hours.` : " There's no ceiling for everyone together.");
  body.innerHTML = `
    <h2 class="section">Learners</h2>
    <div class="tiles">
      ${tile(fmt(l.total), "learners", `${fmt(l.new_this_week)} new this week`)}
      ${tile(fmt(l.active_this_week), "studied this week", share(l.active_this_week))}
      ${tile(fmt(l.confirmed), "confirmed their email", share(l.confirmed))}
      ${tile(fmt(l.with_notifications), "get notifications", share(l.with_notifications))}
    </div>
    <h2 class="section">Learning this week</h2>
    <div class="tiles">
      ${tile(fmt(s.lessons_completed), "lessons completed")}
      ${tile(fmt(s.quizzes), "quizzes and exams taken", s.quiz_average != null ? `average score ${s.quiz_average}%` : "")}
      ${tile(fmt(s.chat_messages), "questions asked in chats")}
      ${tile(fmt(c.designed), "companions designed by learners", `${fmt(c.published)} shared with the community`)}
    </div>
    <h2 class="section">AI requests, last 14 days</h2>
    <div class="tiles">
      ${tile(fmt(sum(ai.days, "requests")), "AI requests from learners", `${fmt(ai.units_last_24h)} counted in the last 24 hours${lim.site > 0 ? ` of ${fmt(lim.site)}` : ""}`
        + (lim.site > 0 ? bar(Math.min(ai.units_last_24h, lim.site), lim.site) : ""))}
      ${tile(money(ai.cost), "estimated cost", `${fmt(sum(ai.kinds, "input_tokens") + sum(ai.kinds, "output_tokens") + sum(ai.kinds, "cache_read_tokens") + sum(ai.kinds, "cache_write_tokens"))} tokens at list prices, scheduled posts included`)}
      ${tile(fmt(away.allowance.requests + away.site.requests), "requests turned away",
        `${plural(away.allowance.learners, "learner")} reached their limit${away.site.requests ? `; ${fmt(away.site.requests)} paused by the site ceiling` : ""}`)}
      ${tile(fmt(sum(ai.kinds, "failed")), "failed calls to Claude", "not counted against anyone")}
    </div>
    <div class="card stack">
      <div><h3 style="margin:0">AI requests per day</h3>
        <div class="muted small">Requests from learners that went through, ${esc(utcDate(ai.days[0].day, { day: "numeric", month: "short" }))} to ${esc(utcDate(ai.days.at(-1).day, { day: "numeric", month: "short" }))} (UTC days). Mock exams and custom paths count once here.</div></div>
      ${sum(ai.days, "requests") ? dayChart(ai.days) : `<div class="empty">No AI requests from learners in the last 14 days.</div>`}
      <details><summary>Show as a table</summary><div class="table-wrap"><table class="table num">
        <thead><tr><th>Day</th><th>Requests</th><th>Tokens</th></tr></thead>
        <tbody>${ai.days.map((d) => `<tr><td>${esc(longDay(d.day))}</td><td>${d.requests.toLocaleString()}</td><td>${d.tokens.toLocaleString()}</td></tr>`).join("")}</tbody>
      </table></div></details>
    </div>
    <div class="card stack">
      <div><h3 style="margin:0">What the AI was asked for</h3>
        <div class="muted small">Last 14 days. Tokens are what Claude reported, counting every attempt.${fallbacks ? ` ${plural(fallbacks, "answer")} came from a fallback model after the first one declined.` : ""}</div></div>
      ${ai.kinds.length ? `<div class="table-wrap"><table class="table num">
        <thead><tr><th>Request</th><th>Requests</th><th>Failed</th><th>Input tokens</th><th>Output tokens</th><th>Cache reads</th><th>Cache writes</th><th>Est. cost</th></tr></thead>
        <tbody>${ai.kinds.map((k) => `<tr><td>${esc(KIND_LABEL[k.kind] || k.kind)}</td>${["requests", "failed", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"].map((f) => `<td>${(k[f] || 0).toLocaleString()}</td>`).join("")}<td>${money(k.cost)}</td></tr>`).join("")}</tbody>
      </table></div>` : `<div class="empty">Nothing yet.</div>`}
      <p class="muted small" style="margin:0">Costs are estimates from the token counts at list prices in US dollars, with cache writes at 1.25 times input; the bill can differ. Change the prices with AIET_PRICES.${ai.unpriced_models.length ? ` No price is set for ${esc(ai.unpriced_models.join(", "))}, so its tokens aren't in the estimate.` : ""}</p>
      <p class="small" style="margin:0">${limits} <span class="muted">Set on the server with AIET_AI_ALLOWANCE, AIET_AI_ALLOWANCE_UNCONFIRMED and AIET_AI_SITE_LIMIT.</span></p>
    </div>
    <div class="two-col">
      <div class="card stack">
        <div><h3 style="margin:0">Topics learners value most</h3>
          <div class="muted small">Engagement per post over the last 30 days, weighted towards likes, shares and follow-up questions.</div></div>
        ${c.topics.length ? rankBars(c.topics.map((t) => ({ label: t.topic, value: t.score }))) : `<div class="empty">No one has engaged with posts in the last 30 days.</div>`}
      </div>
      <div class="card stack">
        <div><h3 style="margin:0">Most followed community companions</h3>
          <div class="muted small">Companions learners designed and shared.</div></div>
        ${c.most_followed.length ? `<table class="table num"><thead><tr><th>Companion</th><th>Followers</th></tr></thead><tbody>
          ${c.most_followed.map((x) => `<tr><td><a href="#/c/${x.id}">${esc(x.name)} ${esc(x.flag || "")}</a> <span class="muted small">by ${esc(x.designer || "a former learner")}</span></td><td>${x.followers.toLocaleString()}</td></tr>`).join("")}
        </tbody></table>` : `<div class="empty">No community companions yet.</div>`}
      </div>
    </div>`;
  wireDayChart(body, ai.days);
}

/* Gridline spacing that gives two to four clean steps above zero. */
function niceStep(max) {
  const raw = Math.max(max, 4) / 4, mag = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((x) => x * mag).find((x) => x >= raw);
}

/* One series by day: thin columns from a single baseline, the peak labelled, hairline gridlines. */
function dayChart(days) {
  const max = Math.max(...days.map((d) => d.requests)), step = niceStep(max), top = Math.ceil(max / step) * step;
  const peak = days.findIndex((d) => d.requests === max);
  return `<div class="colchart">
    <div class="plot">
      ${Array.from({ length: top / step + 1 }, (_, i) => `<div class="gridline${i ? "" : " base"}" style="bottom:${(100 * i * step) / top}%"><span>${fmt(i * step)}</span></div>`).join("")}
      <div class="cols">${days.map((d, i) => `<button type="button" class="col" data-i="${i}" aria-label="${esc(longDay(d.day))}: ${plural(d.requests, "AI request")}, ${plural(d.tokens, "token")}">
        <span class="colbar" style="height:${(100 * d.requests) / top}%">${i === peak ? `<span class="collabel">${fmt(d.requests)}</span>` : ""}</span></button>`).join("")}</div>
    </div>
    <div class="xaxis" aria-hidden="true">${days.map((d) => `<span>${esc(utcDate(d.day, { day: "numeric" }))}</span>`).join("")}</div>
    <div class="charttip" hidden></div>
  </div>`;
}

/* The tooltip follows the pointer or keyboard focus over each day's column (the whole column is the target). */
function wireDayChart(root, days) {
  const chart = $(".colchart", root);
  if (!chart) return;
  const tip = $(".charttip", chart);
  const show = (col) => {
    const d = days[+col.dataset.i], value = document.createElement("b"), when = document.createElement("div");
    value.textContent = plural(d.requests, "AI request");
    when.className = "muted";
    when.textContent = `${longDay(d.day)} · ${plural(d.tokens, "token")}`;
    tip.replaceChildren(value, when);
    tip.hidden = false;
    const box = chart.getBoundingClientRect(), r = col.getBoundingClientRect(), barTop = col.firstElementChild.getBoundingClientRect().top;
    const half = tip.offsetWidth / 2;
    tip.style.left = `${Math.min(Math.max(r.left + r.width / 2 - box.left, half), box.width - half)}px`;
    tip.style.top = `${Math.min(barTop, r.bottom - 6) - box.top - 8}px`;
  };
  chart.addEventListener("pointerover", (e) => { const col = e.target.closest(".col"); if (col) show(col); });
  chart.addEventListener("focusin", (e) => { const col = e.target.closest(".col"); if (col) show(col); });
  chart.addEventListener("pointerleave", () => { if (!chart.contains(document.activeElement)) tip.hidden = true; });
  chart.addEventListener("focusout", () => { tip.hidden = true; });
}

/* Ranked horizontal bars for one series, each value at its bar's tip. */
function rankBars(rows) {
  const max = Math.max(...rows.map((r) => r.value)) || 1;
  return `<div class="rankbars">${rows.map((r) => `<div>${esc(r.label)}</div>
    <div class="track"><span style="width:${(100 * r.value) / max}%"></span></div><div class="value">${esc(r.value.toLocaleString())}</div>`).join("")}</div>`;
}

async function staffAppeals(body, m) {
  const list = await api("/api/staff/appeals");
  if (!list.length) { body.innerHTML = `<div class="card empty">No appeals waiting. 🎉</div>`; return; }
  body.innerHTML = list.map((a) => {
    const c = a.companion, p = c.persona;
    return `
    <article class="card stack" data-id="${a.id}">
      <div class="row">
        <img class="avatar" src="/api/companions/${c.id}/avatar?v=${encodeURIComponent(c.updated_at)}" width="56" height="56" alt="">
        <div style="flex:1"><h3 style="margin:0">${esc(p.name)} ${esc(p.flag)}</h3>
          <div class="muted small">Designed by ${esc(a.designer || "?")} · appealed ${timeAgo(a.created_at)}</div></div>
      </div>
      <div class="small"><b>Removed</b> ${timeAgo(a.removal.at)} by ${esc(a.removal.by || "a former moderator")}: “${esc(a.removal.note)}”</div>
      <div class="small"><b>The designer's appeal:</b> “${esc(a.message)}”</div>
      <div class="small"><b>Tagline:</b> ${esc(p.tagline) || "–"}<br><b>Bio:</b> ${esc(p.bio) || "–"}<br>
        <b>Sample line:</b> ${esc(p.writing_style.sample_line) || "–"}<br><b>Expertise:</b> ${esc(p.expertise.join(", ")) || "–"}</div>
      ${a.can_decide ? `
        <label>Note to the designer <span class="hint">Required when the removal stands</span><textarea name="note" maxlength="500"></textarea></label>
        <div class="row"><button class="btn" data-act="overturn">Restore to the community</button><button class="btn danger" data-act="uphold">Keep it removed</button></div>`
        : `<div class="muted small">You removed this companion, so another moderator gives it the second look.</div>`}
    </article>`;
  }).join("");
  $$("[data-id]", body).forEach((el) => el.addEventListener("click", async (e) => {
    const act = e.target.dataset.act;
    if (!act) return;
    try {
      await api(`/api/staff/appeals/${el.dataset.id}`, { method: "POST", body: { decision: act, note: $("[name=note]", el).value.trim() } });
      toast(act === "overturn" ? "Restored, and the designer was told" : "The removal stands, and the designer was told why");
      staffAppeals(body, m);
    } catch (err) { toast(err.message); }
  }));
}

async function staffReports(body, m) {
  const list = await api("/api/staff/companions");
  if (!list.length) { body.innerHTML = `<div class="card empty">No open reports. 🎉</div>`; return; }
  body.innerHTML = list.map((c) => `
    <article class="card stack" data-id="${c.id}">
      <div class="row">
        <img class="avatar" src="/api/companions/${c.id}/avatar?v=${encodeURIComponent(c.updated_at)}" width="56" height="56" alt="">
        <div style="flex:1">
          <h3 style="margin:0">${esc(c.persona.name)} ${esc(c.persona.flag)} ${c.hidden ? `<span class="chip bad">Hidden</span>` : `<span class="chip">Still visible</span>`}</h3>
          <div class="muted small">Designed by ${esc(c.designer || "?")} · ${c.reports} report${c.reports === 1 ? "" : "s"} ·
            ${Object.entries(c.reasons).map(([k, n]) => `${REASON_LABEL[k] || esc(k)} ×${n}`).join(", ")}</div>
        </div>
      </div>
      <div class="small"><b>Tagline:</b> ${esc(c.persona.tagline) || "–"}<br><b>Bio:</b> ${esc(c.persona.bio) || "–"}<br>
        <b>Sample line:</b> ${esc(c.persona.writing_style.sample_line) || "–"}<br><b>Expertise:</b> ${esc(c.persona.expertise.join(", ")) || "–"}</div>
      <label>Note to the designer <span class="hint">Required when removing</span><textarea name="note" maxlength="500"></textarea></label>
      <div class="row">
        <button class="btn" data-act="${c.hidden ? "restore" : "dismiss"}">${c.hidden ? "Restore to community" : "Dismiss reports"}</button>
        <button class="btn danger" data-act="remove">Remove from community</button>
      </div>
    </article>`).join("");
  $$("[data-id]", body).forEach((el) => el.addEventListener("click", async (e) => {
    const act = e.target.dataset.act;
    if (!act) return;
    const note = $("[name=note]", el).value.trim();
    if (act === "remove" && !confirm("Remove this companion from the community for good? The designer keeps a private copy.")) return;
    try {
      await api(`/api/staff/companions/${el.dataset.id}`, { method: "POST", body: { action: act, note } });
      toast("Decision saved and the designer was told");
      staffReports(body, m);
    } catch (err) { toast(err.message); }
  }));
}

async function staffPaths(body, m) {
  const [pending, shared] = await Promise.all([api("/api/staff/paths?status=pending"), api("/api/staff/paths?status=approved")]);
  body.innerHTML = `
    <h2>Waiting for review (${pending.length})</h2>
    ${pending.length ? pending.map((p) => `
      <form class="card stack review" data-id="${p.id}">
        <div class="muted small">Suggested by ${esc(p.owner_name)} · ${esc(label(p.level))} · topic “${esc(p.topic)}” · submitted ${timeAgo(p.submitted_at)}</div>
        <label>Title <input type="text" name="title" maxlength="80" value="${esc(p.title)}"></label>
        <label>Description <textarea name="description" maxlength="400">${esc(p.description)}</textarea></label>
        ${p.lessons.map((l, i) => `
          <fieldset class="lesson-edit">
            <legend>Lesson ${i + 1}</legend>
            <div class="two"><label>Title <input type="text" data-l="${i}" data-f="title" maxlength="90" value="${esc(l.title)}"></label>
              <label>Minutes <input type="number" data-l="${i}" data-f="minutes" min="5" max="20" value="${l.minutes}"></label></div>
            <label>Objectives <span class="hint">One per line, up to 3</span><textarea data-l="${i}" data-f="objectives">${esc(l.objectives.join("\n"))}</textarea></label>
            <label>Key points <span class="hint">One per line, 2 to 5. Check each one is accurate.</span><textarea data-l="${i}" data-f="key_points" rows="5">${esc(l.key_points.join("\n"))}</textarea></label>
          </fieldset>`).join("")}
        <label>Note to the learner <span class="hint">Required when rejecting</span><textarea name="note" maxlength="1000"></textarea></label>
        <div class="row"><button class="btn primary" data-act="approve">Approve and share</button><button class="btn danger" data-act="reject">Reject</button></div>
      </form>`).join("") : `<div class="card empty">Nothing to review.</div>`}
    <h2 style="margin-top:20px">Shared paths (${shared.length})</h2>
    ${shared.length ? `<div class="card"><table class="table"><thead><tr><th>Path</th><th>Suggested by</th><th>Reviewed</th><th></th></tr></thead><tbody>
      ${shared.map((p) => `<tr data-id="${p.id}"><td>${esc(p.title)}</td><td>${esc(p.owner_name)}</td><td class="muted small">${timeAgo(p.reviewed_at)}</td>
        <td><button class="btn ghost danger" data-unshare>Unshare</button></td></tr>`).join("")}</tbody></table></div>` : `<div class="card empty">No shared paths yet.</div>`}`;
  $$("form.review", body).forEach((f) => f.addEventListener("click", async (e) => {
    const act = e.target.dataset.act;
    if (!act) return;
    e.preventDefault();
    const n = $$("fieldset.lesson-edit", f).length;
    const lessons = Array.from({ length: n }, (_, i) => {
      const get = (k) => $(`[data-l="${i}"][data-f="${k}"]`, f).value;
      return { title: get("title").trim(), minutes: +get("minutes"), objectives: lines(get("objectives")), key_points: lines(get("key_points")) };
    });
    const field = (n) => f.elements.namedItem(n).value.trim();
    const payload = { decision: act, note: field("note") };
    if (act === "approve") Object.assign(payload, { title: field("title"), description: field("description"), lessons });
    try {
      await api(`/api/staff/paths/${f.dataset.id}`, { method: "POST", body: payload });
      toast(act === "approve" ? "Approved and shared with every learner" : "Rejected; the learner was told why");
      staffPaths(body, m);
    } catch (err) { toast(err.message); }
  }));
  $$("tr[data-id] [data-unshare]", body).forEach((b) => b.addEventListener("click", async () => {
    const note = prompt("Why is this path being unshared? The learner who suggested it will see this.");
    if (!note) return;
    try {
      await api(`/api/staff/paths/${b.closest("tr").dataset.id}`, { method: "POST", body: { decision: "unshare", note } });
      toast("Path unshared");
      staffPaths(body, m);
    } catch (err) { toast(err.message); }
  }));
}

async function staffRoles(body, m) {
  body.innerHTML = `
    <form class="card row" id="userSearch"><input type="text" name="q" placeholder="Search by name or email" aria-label="Search people" style="flex:1"><button class="btn">Search</button></form>
    <div class="card"><div class="table-wrap"><table class="table"><thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Daily AI requests</th></tr></thead><tbody id="userRows"></tbody></table></div>
    <p class="muted small">Moderators review reported companions. Reviewers check learner-made paths before they're shared. Admins can do both and assign roles.</p>
    <p class="muted small">Give a learner an AI allowance of their own, for a class or a paid plan: the number of AI requests they can make in any 24 hours, or no limit. Leave the number empty for the usual allowance, then save.</p></div>`;
  const used = (ai) => `${ai.used.toLocaleString()} used${ai.limit === null ? "" : ` of ${ai.limit.toLocaleString()}`}`;
  const load = async (q = "") => {
    const users = await api(`/api/staff/users?q=${encodeURIComponent(q)}`);
    $("#userRows", body).innerHTML = users.map((u) => `<tr data-id="${u.id}"><td>${esc(u.display_name)}</td><td>${esc(u.email)}</td><td>
      ${u.fixed || u.id === state.me.id ? `${label(u.role)} <span class="muted small">(${u.fixed ? "set in server config" : "you"})</span>`
        : `<select data-role aria-label="Role for ${esc(u.display_name)}">${["learner", "reviewer", "moderator", "admin"].map((r) => `<option value="${r}" ${r === u.role ? "selected" : ""}>${label(r)}</option>`).join("")}</select>`}
    </td><td>${u.role === "admin" ? `<span class="muted small">No limit (admin)</span>` : `
      <form class="row" data-allowance style="gap:8px;flex-wrap:nowrap">
        <input type="number" name="n" min="1" max="100000" step="1" inputmode="numeric" style="width:96px" placeholder="Usual"
          value="${u.ai_allowance || ""}" ${u.ai_allowance === 0 ? "disabled" : ""} aria-label="Daily AI requests for ${esc(u.display_name)}">
        <label class="check small" style="white-space:nowrap"><input type="checkbox" name="unlimited" ${u.ai_allowance === 0 ? "checked" : ""}> No limit</label>
        <button class="btn">Save</button>
        <span class="muted small" data-used style="white-space:nowrap">${used(u.ai)}</span>
      </form>`}
    </td></tr>`).join("");
  };
  $("#userSearch", body).addEventListener("submit", (e) => { e.preventDefault(); load(e.target.q.value); });
  $("#userRows", body).addEventListener("change", async (e) => {
    if (e.target.name === "unlimited") { e.target.form.n.disabled = e.target.checked; return; }
    if (!e.target.matches("[data-role]")) return;  // allowances are saved with their button, not on every change
    const id = e.target.closest("tr").dataset.id;
    try { await api(`/api/staff/users/${id}/role`, { method: "POST", body: { role: e.target.value } }); toast("Role updated"); }
    catch (err) { toast(err.message); load($("#userSearch", body).q.value); }
  });
  $("#userRows", body).addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = e.target, n = f.n;
    // A typo leaves the box's value empty, which would otherwise mean "the usual allowance".
    if (!f.unlimited.checked && (n.validity.badInput || !n.checkValidity())) {
      toast("Enter a whole number from 1 to 100,000, or leave it empty for the usual allowance.");
      n.focus();
      return;
    }
    const value = f.unlimited.checked ? 0 : n.value === "" ? null : Number(n.value);
    f.querySelector("button").disabled = true;
    try {
      const ai = await api(`/api/staff/users/${f.closest("tr").dataset.id}/allowance`, { method: "POST", body: { ai_allowance: value } });
      f.querySelector("[data-used]").textContent = used(ai);  // just this row, so typing in another isn't cut short
      toast(value === null ? "Back to the usual allowance" : value === 0 ? "No daily limit" : `Allowance set to ${value.toLocaleString()} a day`);
    } catch (err) { toast(err.message); }
    f.querySelector("button").disabled = false;
  });
  await load();
}

async function staffLog(body) {
  const rows = await api("/api/staff/log");
  body.innerHTML = rows.length ? `<div class="card"><table class="table"><thead><tr><th>When</th><th>Who</th><th>What</th><th>Note</th></tr></thead><tbody>
    ${rows.map((r) => `<tr><td class="muted small">${timeAgo(r.created_at)}</td><td>${esc(r.staff_name || "–")}</td>
      <td>${esc(label(r.action.replace(":", " → ")))} ${esc(r.target_type)} #${r.target_id}</td><td class="small">${esc(r.note)}</td></tr>`).join("")}
  </tbody></table></div>` : `<div class="card empty">No decisions yet.</div>`;
}

/* ------------------------------------------------------------------ courses */

/* Each subject's icon and the two colours of its courses' covers. */
const CATEGORY_LOOK = {
  "Generative AI": ["✨", "#6d4cff", "#c46bd6"],
  "Machine Learning": ["🧠", "#3f51d6", "#7b8cff"],
  "Data Science": ["📊", "#0d837f", "#36bf9e"],
  "Software Development": ["💻", "#1f5fd1", "#46a8f5"],
  "Networking": ["🌐", "#08779c", "#2fbcd3"],
  "Cybersecurity": ["🛡️", "#ad2f3c", "#e8744f"],
  "Cloud & DevOps": ["☁️", "#2f62c2", "#64a2ec"],
  "AI for Business": ["💼", "#a9680f", "#e8a83a"],
  "Responsible AI": ["⚖️", "#277f45", "#6fba6a"],
  "AI in Industry": ["🏥", "#b54c22", "#ec9150"],
  "Robotics & Edge AI": ["🤖", "#434c5e", "#8494b0"],
};
const look = (category) => CATEGORY_LOOK[category] || ["🎓", "#5b4cdb", "#8a7dff"];
const ITEM_ICON = { lesson: "📖", lab: "🧪", quiz: "📝", project: "🏗️" };
const ITEM_STATUS = { completed: "✅", passed: "✅", started: "🟣", attempted: "🟠", needs_work: "🟠", not_started: "⚪" };
const STATUS_WORDS = { completed: "Completed", passed: "Passed", started: "Started", attempted: "Not passed yet", needs_work: "Needs work", not_started: "Not started" };
const kindName = (c) => (c.kind === "guided-project" ? "Guided project" : "Course");
const plural2 = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
const duration = (minutes) => (minutes < 60 ? `${minutes} min` : (() => {
  const h = Math.round(minutes / 30) / 2;
  return `${h % 1 ? h.toFixed(1) : h} ${h === 1 ? "hour" : "hours"}`;
})());
const hoursText = (h) => duration(Math.round(h * 60));
/* "an 11-hour", "a 45-minute": the length before a noun, with its article. */
const lengthOf = (h) => {
  const minutes = Math.round(h * 60), n = minutes < 60 ? minutes : Math.round(h * 2) / 2;
  const whole = String(Math.floor(n));
  const an = whole.startsWith("8") || whole === "11" || whole === "18";
  return `${an ? "an" : "a"} ${n % 1 ? n.toFixed(1) : n}-${minutes < 60 ? "minute" : "hour"}`;
};
const chipList = (items) => `<div class="chips-row">${items.map((s) => `<span class="chip">${esc(s)}</span>`).join("")}</div>`;
const faceUrl = (i) => `/api/companions/${i.id}/avatar?v=${encodeURIComponent(i.updated_at || "")}`;
const faces = (list, size = 28) => `<span class="faces-sm">${list.map((i) =>
  `<img class="avatar" src="${faceUrl(i)}" width="${size}" height="${size}" alt="${esc(i.name)}" title="${esc(i.name)}" loading="lazy" decoding="async">`).join("")}</span>`;
const ratingText = (r) => (r ? `<span class="stars" aria-hidden="true">★</span> <b>${r.average.toFixed(1)}</b> <span class="muted">(${fmtCount(r.count)})</span>` : `<span class="new">New</span>`);

function courseCard(c) {
  const [icon, c1, c2] = look(c.category);
  const tools = c.jetbrains.length ? c.jetbrains.slice(0, 2).join(" · ") : c.cisco ? "Cisco-aligned" : c.tools[0] || "";
  return `<a class="card course-card" href="#/course/${c.slug}">
    <div class="course-cover" style="--c1:${c1};--c2:${c2}">
      <span class="kind">${kindName(c)}</span><span class="icon" aria-hidden="true">${icon}</span>${tools ? `<span class="tools">${esc(tools)}</span>` : ""}
    </div>
    <div class="body">
      <div class="row" style="gap:4px">${faces(c.instructors, 26)}<span class="inst-names">${esc(c.instructors.map((i) => i.name).join(", "))}</span></div>
      <h3>${esc(c.title)}</h3>
      <div class="small muted clamp2"><b>Skills you'll gain:</b> ${esc(c.skills.join(", "))}</div>
      <div class="meta">${ratingText(c.rating)} · ${label(c.level)} · ${hoursText(c.hours)}</div>
      ${c.enrolled || c.certificate ? `<div>${bar(c.progress, 100)}<div class="small muted">${c.certificate ? "🎓 Certificate earned" : `${c.progress}% complete`}</div></div>`
        : c.labs ? `<div class="small hands-on">🧪 ${plural2(c.labs, "hands-on lab")}${c.walkthroughs ? ` · 🖥️ ${plural2(c.walkthroughs, "walkthrough")}` : ""}</div>` : ""}
    </div>
  </a>`;
}

function trackCard(t) {
  return `<a class="card track-card" href="#/track/${t.slug}">
    <div class="eyebrow">${esc(t.kind)}</div>
    <h3>${esc(t.title)}</h3>
    <div class="small muted clamp2">${esc(t.subtitle)}</div>
    <div>${faces(t.instructors.slice(0, 7), 26)}</div>
    <div class="meta">${t.total}-course series · ${hoursText(t.hours)}${t.done ? ` · ${t.done} of ${t.total} done` : ""}</div>
  </a>`;
}

/* The catalogue: what's in progress, the tracks, then every course, with filters kept in the address. */
async function viewCourses(m) {
  const params = new URLSearchParams(m[1] || "");
  const cat = await api("/api/courses");
  const f = { category: params.get("category") || "", level: params.get("level") || "", kind: params.get("kind") || "", tag: params.get("tag") || "" };
  const q = (params.get("q") || "").trim().toLowerCase();
  const TAGS = {
    "hands-on": ["🧪 Hands-on labs", (c) => c.labs > 0],
    jetbrains: ["🖥️ JetBrains IDEs", (c) => c.jetbrains.length > 0],
    cisco: ["🌐 Cisco-aligned", (c) => c.cisco],
  };
  const words = (c) => [c.title, c.subtitle, c.category, ...c.skills, ...c.tools, ...c.instructors.map((i) => i.name)].join(" ").toLowerCase();
  const list = cat.courses.filter((c) => (!f.category || c.category === f.category) && (!f.level || c.level === f.level)
    && (!f.kind || c.kind === f.kind) && (!f.tag || TAGS[f.tag]?.[1](c)) && (!q || q.split(/\s+/).every((w) => words(c).includes(w))));
  const filtering = q || f.category || f.level || f.kind || f.tag;
  const going = cat.courses.filter((c) => c.enrolled && !c.completed);
  const link = (changes) => {
    const p = new URLSearchParams(params);
    for (const [k, v] of Object.entries(changes)) v ? p.set(k, v) : p.delete(k);
    const s = p.toString();
    return `#/courses${s ? `?${s}` : ""}`;
  };
  const tab = (key, value, text) => `<a href="${link({ [key]: value })}" class="${f[key] === value ? "active" : ""}">${text}</a>`;
  const main = shell("courses", `
    <div class="course-hero">
      <h1>Courses</h1>
      <p class="muted">Structured courses taught by the AI Learning Companions: lessons in their own voice, tools shown step by step,
        hands-on labs and projects they review, graded quizzes, and a certificate when you finish.</p>
      <form class="course-search" id="courseSearch" role="search">
        <input type="search" name="q" value="${esc(params.get("q") || "")}" placeholder="What do you want to learn? Try PyCharm, subnetting or RAG" aria-label="Search courses">
        <button class="btn primary">Search</button>
      </form>
    </div>
    ${!filtering && going.length ? `<h2 class="section">Continue learning</h2><div class="course-grid">${going.map(courseCard).join("")}</div>` : ""}
    ${!filtering && cat.tracks.length ? `<h2 class="section">Professional Certificates and Specializations</h2>
      <div class="track-row">${cat.tracks.map(trackCard).join("")}</div>` : ""}
    <h2 class="section">${filtering ? "Results" : "Explore every course"}</h2>
    <nav class="region-tabs" aria-label="Subject">${tab("category", "", "All subjects")}${cat.categories.map((x) =>
      tab("category", x.name, `${look(x.name)[0]} ${esc(x.name)} <span class="count">${x.count}</span>`)).join("")}</nav>
    <div class="row filters">
      <div class="seg" role="group" aria-label="Type">${tab("kind", "", "All")}${tab("kind", "course", "Courses")}${tab("kind", "guided-project", "Guided projects")}</div>
      <div class="seg" role="group" aria-label="Level">${tab("level", "", "Any level")}${["beginner", "intermediate", "advanced"].map((l) => tab("level", l, label(l))).join("")}</div>
      ${Object.entries(TAGS).map(([k, [text]]) => `<a class="btn tag ${f.tag === k ? "on" : ""}" href="${link({ tag: f.tag === k ? "" : k })}">${text}</a>`).join("")}
    </div>
    <div class="row spread small muted" style="margin:0 0 12px"><span>${plural2(list.length, "course")}</span>${filtering ? `<a href="#/courses">Clear the filters</a>` : ""}</div>
    <div class="course-grid">${list.length ? list.map(courseCard).join("") : `<div class="card empty">No courses match. <a href="#/courses">See them all</a></div>`}</div>`);
  $("#courseSearch", main).addEventListener("submit", (e) => { e.preventDefault(); location.hash = link({ q: e.target.q.value.trim() }); });
}

const allItems = (c) => [...c.syllabus.flatMap((mod) => mod.items.map((i) => ({ ...i, module: mod }))), ...(c.project ? [c.project] : [])];
const itemLine = (i) => ({
  lesson: `Lesson${i.walkthrough ? ` with a ${i.walkthrough} walkthrough` : ""}${i.video ? " and video" : ""}`,
  lab: `Hands-on lab in ${i.tool}`, quiz: `Graded quiz · ${i.questions} questions`, project: "Capstone project",
}[i.type]) + ` · ${duration(i.minutes)}`;

function itemRow(c, i) {
  const inner = `<span class="ico" aria-hidden="true">${ITEM_ICON[i.type]}</span><span class="what"><span class="t">${esc(i.title)}</span>
    <span class="type">${itemLine(i)}</span></span>${c.enrolled ? `<span title="${STATUS_WORDS[i.status]}">${ITEM_STATUS[i.status]}</span>` : ""}`;
  return `<li>${c.enrolled ? `<a class="item-row" href="#/course/${c.slug}/learn/${i.key}">${inner}</a>` : `<div class="item-row">${inner}</div>`}</li>`;
}

function moduleBlock(c, mod, open) {
  const n = (type) => mod.items.filter((i) => i.type === type).length;
  const included = [plural2(n("lesson"), "lesson"), n("lab") && plural2(n("lab"), "lab"), plural2(n("quiz"), "quiz", "quizzes")].filter(Boolean).join(" · ");
  return `<details class="module"${open ? " open" : ""}>
    <summary>
      <div><div class="muted small">Module ${mod.index} · ${duration(mod.minutes)} to complete</div><h3>${esc(mod.title)}</h3>
        <div class="small muted">${esc(mod.summary)}</div></div>
      <span class="muted small included">${included}</span>
    </summary>
    ${mod.instructor ? `<div class="row small taught">${faces([mod.instructor], 24)} Taught by ${esc(mod.instructor.name)} ${esc(mod.instructor.flag)}</div>` : ""}
    <ul class="items">${mod.items.map((i) => itemRow(c, i)).join("")}</ul>
  </details>`;
}

function instructorCard(i) {
  return `<div class="card inst-card">
    <img class="avatar" src="${faceUrl(i)}" width="72" height="72" alt="" loading="lazy">
    <div class="stack" style="gap:6px">
      <div><b>${esc(i.name)} ${esc(i.flag)}</b> <span class="chip">AI Learning Companion</span>${i.tagline ? `<div class="muted small">${esc(i.tagline)}</div>` : ""}</div>
      <p class="small" style="margin:0">${esc(i.bio)}</p>
      ${i.expertise?.length ? chipList(i.expertise) : ""}
      <div class="row small"><span class="muted">${plural2(i.courses, "course")} on AI Every Time</span>
        <a class="btn" href="#/c/${i.id}">💬 Chat with ${esc(i.name.split(" ")[0])}</a></div>
    </div>
  </div>`;
}

/* What a learner has done towards the certificate, and the certificate once it's theirs to get. */
function progressBlock(c, { grades = false } = {}) {
  const r = c.requirements, tick = (ok) => (ok ? "✅" : "⬜");
  const lines = [
    `${tick(r.lessons.done === r.lessons.total)} Lessons completed: ${r.lessons.done} of ${r.lessons.total}`,
    `${tick(r.quizzes.done === r.quizzes.total)} Module quizzes passed with 70% or more: ${r.quizzes.done} of ${r.quizzes.total}`,
    r.labs.total ? `${tick(r.labs.done === r.labs.total)} Labs passed: ${r.labs.done} of ${r.labs.total}` : "",
    r.project !== null ? `${tick(r.project)} Capstone project passed, with 70 points or more` : "",
  ].filter(Boolean);
  return `<section class="card stack" id="yours">
    <div class="row spread"><h2 style="margin:0">Your progress</h2><b>${c.progress}%</b></div>
    ${bar(c.progress, 100)}
    <ul class="checklist">${lines.map((l) => `<li>${l}</li>`).join("")}</ul>
    ${c.certificate ? `<div class="row"><a class="btn primary" href="#/certificate/${c.certificate}">🎓 View your certificate</a></div>`
      : r.complete ? `<form id="certForm" class="stack cert-form">
          <div><b>🎉 You've finished the ${kindName(c).toLowerCase()}!</b> Your certificate is ready.</div>
          <label>Name on your certificate <input type="text" name="name" value="${esc(state.me.display_name)}" minlength="2" maxlength="80" required></label>
          <div><button class="btn primary">Get my certificate</button></div></form>`
      : `<div class="muted small">Finish all of these to earn your certificate.</div>`}
    <div class="row rate-row"><span class="small">Rate this ${kindName(c).toLowerCase()}:</span><span class="rate" role="group" aria-label="Your rating">${[1, 2, 3, 4, 5].map((n) =>
      `<button type="button" class="star ${n <= (c.my_rating || 0) ? "on" : ""}" data-stars="${n}" aria-label="${n} star${n === 1 ? "" : "s"}">★</button>`).join("")}</span></div>
    <div class="row small">${grades ? `<a href="#/course/${c.slug}">Course page</a>` : `<a href="#/course/${c.slug}/learn/grades">Grades</a>`}
      <button type="button" class="textbtn" id="leave">Leave this ${kindName(c).toLowerCase()}</button></div>
  </section>`;
}

function wireProgress(root, c) {
  $("#certForm", root)?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("button", e.target);
    btn.disabled = true;
    try {
      const cert = await api(`/api/courses/${c.slug}/certificate`, { method: "POST", body: { name: e.target.name.value } });
      location.hash = `#/certificate/${cert.code}`;
    } catch (err) { toast(err.message); btn.disabled = false; }
  });
  $$(".star", root).forEach((b) => b.addEventListener("click", async () => {
    const stars = +b.dataset.stars;
    try {
      await api(`/api/courses/${c.slug}/rating`, { method: "PUT", body: { stars } });
      $$(".star", root).forEach((s) => s.classList.toggle("on", +s.dataset.stars <= stars));
      toast("Thanks for rating it");
    } catch (err) { toast(err.message); }
  }));
  $("#leave", root)?.addEventListener("click", async () => {
    if (!confirm("Leave this course? It comes off your list; your progress is kept in case you come back.")) return;
    await api(`/api/courses/${c.slug}/enroll`, { method: "DELETE" });
    toast("You've left the course");
    const page = `#/course/${c.slug}`;
    if (location.hash === page) router();
    else location.hash = page;
  });
}

/* A course's page, as Coursera lays one out: what it is, what you'll learn, the syllabus, the instructors. */
async function viewCourse(m) {
  const slug = m[1];
  const c = await api(`/api/courses/${slug}`);
  const [icon, c1, c2] = look(c.category);
  const items = allItems(c);
  const count = (type) => items.filter((i) => i.type === type).length;
  const next = items.find((i) => i.key === c.next);
  const kind = kindName(c).toLowerCase();
  const start = c.enrolled
    ? `<a class="btn primary go" href="#/course/${slug}/learn/${c.next || "grades"}">${!c.next ? "Go to the course" : c.progress ? "Resume" : "Start learning"}</a>`
    : `<button class="btn primary go" id="enrol">Enrol for free</button>`;
  const main = shell("courses", `
    <div class="muted small crumbs"><a href="#/courses">Courses</a> › <a href="#/courses?category=${encodeURIComponent(c.category)}">${esc(c.category)}</a></div>
    <section class="card course-head" style="--c1:${c1};--c2:${c2}">
      <div class="badges"><span class="badge">${kindName(c)}</span><span class="badge">${label(c.level)}</span>${c.jetbrains.map((t) =>
        `<span class="badge">🖥️ ${esc(t)}</span>`).join("")}${c.cisco ? `<span class="badge">🌐 Cisco-aligned</span>` : ""}</div>
      <h1><span aria-hidden="true">${icon}</span> ${esc(c.title)}</h1>
      <p class="lead">${esc(c.subtitle)}</p>
      <div class="row">${faces(c.instructors, 36)}<span>${c.instructors.length > 1 ? "Instructors" : "Instructor"}: ${c.instructors.map((i) =>
        `<a href="#/c/${i.id}">${esc(i.name)} ${esc(i.flag)}</a>`).join(", ")}</span></div>
      <div class="row">${start}<span class="muted small">${c.learners ? `${fmtCount(c.learners)} ${c.learners === 1 ? "learner has" : "learners have"} enrolled` : "Free · be one of the first to enrol"}</span></div>
      ${c.enrolled ? `<div>${bar(c.progress, 100)}<div class="small muted">${c.progress}% complete${next ? ` · Up next: ${esc(next.title)}` : ""}</div></div>` : ""}
      ${c.aligned_with ? `<div class="small">📋 Follows ${esc(c.aligned_with)}</div>` : ""}
      ${c.tracks.length ? `<div class="small">🎓 Part of ${c.tracks.map((t) => `the <a href="#/track/${t.slug}">${esc(t.title)} ${esc(t.kind)}</a>`).join(" and ")}</div>` : ""}
    </section>
    <div class="card course-stats">
      <div><b>${plural2(c.syllabus.length, "module")}</b><span class="muted small">${plural2(count("lesson"), "lesson")}</span></div>
      <div><b>${c.rating ? `★ ${c.rating.average.toFixed(1)}` : "New"}</b><span class="muted small">${c.rating ? plural2(c.rating.count, "rating") : "No ratings yet"}</span></div>
      <div><b>${label(c.level)} level</b><span class="muted small">${c.level === "beginner" ? "No experience needed" : "Some experience needed"}</span></div>
      <div><b>${hoursText(c.hours)}</b><span class="muted small">At your own pace</span></div>
      <div><b>${count("lab") ? plural2(count("lab"), "hands-on lab") : "Graded quizzes"}</b><span class="muted small">${c.project ? "and a capstone project" : "reviewed by your instructor"}</span></div>
    </div>
    <nav class="anchors" aria-label="On this page">${[["about", "About"], ["modules", "Modules"], ["instructors", "Instructors"], ...(c.enrolled ? [["yours", "Your progress"]] : [])]
      .map(([id, t]) => `<button type="button" data-to="${id}">${t}</button>`).join("")}</nav>
    <section class="stack" id="about">
      <div class="card"><h2>What you'll learn</h2><ul class="outcomes">${c.outcomes.map((o) => `<li>${esc(o)}</li>`).join("")}</ul></div>
      <div class="two-col">
        <div class="card stack"><div><h3>Skills you'll gain</h3>${chipList(c.skills)}</div>${c.tools.length ? `<div><h3>Tools you'll use</h3>${chipList(c.tools)}</div>` : ""}</div>
        <div class="card details-grid">
          <div>🎓 <b>Shareable certificate</b><div class="muted small">When you finish: print it, or add it to LinkedIn</div></div>
          <div>📝 <b>Assessments</b><div class="muted small">${[plural2(count("quiz"), "graded quiz", "graded quizzes"), count("lab") && plural2(count("lab"), "lab"), c.project && "a capstone project"].filter(Boolean).join(", ")}</div></div>
          <div>🤖 <b>Taught by AI Learning Companions</b><div class="muted small">Ask your instructor about any lesson or lab</div></div>
          <div>🖥️ <b>${c.walkthroughs ? plural2(c.walkthroughs, "tool walkthrough") : "Self-paced"}</b><div class="muted small">${c.walkthroughs ? "Shown step by step, with shortcuts for macOS, Windows and Linux" : "Study whenever suits you"}</div></div>
        </div>
      </div>
      <div class="card"><h2>About this ${kind}</h2><p>${esc(c.description)}</p><h3>Before you start</h3><p style="margin:0">${esc(c.prerequisites)}</p></div>
    </section>
    <section class="stack" id="modules">
      <h2 class="section">There ${c.syllabus.length === 1 ? "is 1 module" : `are ${c.syllabus.length} modules`} in this ${kind}</h2>
      ${c.syllabus.map((mod, i) => moduleBlock(c, mod, i === 0)).join("")}
      ${c.project ? `<div class="card stack project-block"><div class="muted small">Capstone project · ${duration(c.project.minutes)}</div>
        <h3 style="margin:0">🏗️ ${esc(c.project.title)}</h3><div class="small muted">Put the whole ${kind} to work on a project of your own, scored by your instructor against a rubric.</div>
        ${c.enrolled ? `<div><a class="btn" href="#/course/${slug}/learn/project">${ITEM_STATUS[c.project.status]} Open the project</a></div>` : ""}</div>` : ""}
    </section>
    <section class="stack" id="instructors"><h2 class="section">${c.instructors.length > 1 ? "Instructors" : "Instructor"}</h2>${c.instructors.map(instructorCard).join("")}</section>
    ${c.enrolled ? progressBlock(c) : ""}
    ${c.notices.length ? `<p class="notice">${c.notices.map(esc).join(" ")}</p>` : ""}`);
  $$(".anchors button", main).forEach((b) => b.addEventListener("click", () => $(`#${b.dataset.to}`)?.scrollIntoView({ behavior: "smooth", block: "start" })));
  $("#enrol", main)?.addEventListener("click", async (e) => {
    e.target.disabled = true;
    try {
      await api(`/api/courses/${slug}/enroll`, { method: "POST" });
      toast(`You're enrolled in ${c.title}`);
      location.hash = `#/course/${slug}/learn/${c.next || "l1-1"}`;
    } catch (err) { toast(err.message); e.target.disabled = false; }
  });
  if (c.enrolled) wireProgress(main, c);
}

/* ------------------------------------------------------------ course player */

const KEYS_PREF = "aiet-keys";
const keysFor = () => {
  try { const v = localStorage.getItem(KEYS_PREF); if (v === "mac" || v === "win") return v; } catch {}
  return /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent) ? "mac" : "win";
};
const MAC_KEYS = { Cmd: "⌘", Option: "⌥", Shift: "⇧", Ctrl: "⌃", Enter: "↩", Backspace: "⌫", Up: "↑", Down: "↓", Left: "←", Right: "→" };
/* A shortcut as keys: "Cmd+Shift+A" pressed together, "Shift Shift" one key twice, "A / B" either. */
function keysHtml(text, os) {
  return text.split(" / ").map((alt) => {
    const twice = !alt.includes("+") && alt.trim().includes(" ");
    const parts = twice ? alt.trim().split(/\s+/) : alt.split("+");
    const ks = parts.map((p) => `<kbd>${esc(os === "mac" && MAC_KEYS[p.trim()] ? MAC_KEYS[p.trim()] : p.trim())}</kbd>`);
    return twice ? ks.join(" ") : ks.join(os === "mac" ? "" : "+");
  }).join(" <span class='muted'>or</span> ");
}
const codeBlock = (code, lang) => `<div class="prose codebox"><pre class="code"${lang ? ` data-lang="${esc(lang)}"` : ""}><code>${esc(code)}</code></pre>
  <button type="button" class="copy" data-copy>Copy</button></div>`;
function stepHtml(s, os, i, tick = null) {
  return `<li data-step="${i}">${tick !== null ? `<input type="checkbox" class="tick" data-i="${i}" aria-label="Step ${i + 1} done"${tick ? " checked" : ""}>` : ""}
    <div class="stack" style="gap:6px"><div>${esc(s.do)}${s.keys ? ` <span class="keys" data-mac="${esc(s.keys.mac)}" data-win="${esc(s.keys.win)}">${keysHtml(s.keys[os], os)}</span>` : ""}</div>
    ${s.code ? codeBlock(s.code, s.lang) : ""}${s.see ? `<div class="see">👀 ${esc(s.see)}</div>` : ""}</div></li>`;
}
const osSwitch = (os) => `<div class="seg os" role="group" aria-label="Keyboard shortcuts for">${[["mac", "macOS"], ["win", "Windows / Linux"]].map(([o, t]) =>
  `<button type="button" data-os="${o}" class="${os === o ? "active" : ""}">${t}</button>`).join("")}</div>`;
function walkthroughHtml(w) {
  const os = keysFor();
  return `<section class="walk" id="walk" aria-label="Walkthrough">
    <div class="walk-head"><div><div class="tool">Follow along in ${esc(w.tool)}</div><b>${esc(w.title)}</b></div>
      <div class="row" style="gap:8px">${osSwitch(os)}<button type="button" class="btn play" id="playWalk">▶ Play</button></div></div>
    <ol>${w.steps.map((s, i) => stepHtml(s, os, i)).join("")}</ol>
  </section>`;
}
/* The shortcut switch, copy buttons and step ticks in a part of the page. */
function wireSteps(root, tickKey = null) {
  $$(".os button", root).forEach((b) => b.addEventListener("click", () => {
    const os = b.dataset.os;
    try { localStorage.setItem(KEYS_PREF, os); } catch {}
    $$(".os button").forEach((x) => x.classList.toggle("active", x.dataset.os === os));
    $$(".keys").forEach((k) => (k.innerHTML = keysHtml(k.dataset[os], os)));
  }));
  $$("[data-copy]", root).forEach((b) => b.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(b.previousElementSibling.textContent); b.textContent = "Copied"; }
    catch { b.textContent = "Select and copy"; }
    setTimeout(() => (b.textContent = "Copy"), 1500);
  }));
  if (tickKey) $$(".tick", root).forEach((t) => t.addEventListener("change", () => {
    const done = $$(".tick:checked", root).map((x) => +x.dataset.i);
    try { localStorage.setItem(tickKey, JSON.stringify(done)); } catch {}
  }));
}
const ticked = (key) => { try { return new Set(JSON.parse(localStorage.getItem(key) || "[]")); } catch { return new Set(); } };

/* Play a walkthrough like a screencast: each step lit in turn and read out in the instructor's voice, or shown for
   a few seconds each where the browser can't speak. Pressing again stops it. */
async function playWalkthrough(root, w, teacherId, button) {
  if (root.dataset.playing) { delete root.dataset.playing; if (canSpeak) speechSynthesis.cancel(); return; }
  root.dataset.playing = "1";
  button.textContent = "■ Stop";
  let voice = null;
  try { voice = (await api(`/api/companions/${teacherId}`)).persona.voice; } catch {}
  const steps = $$("li[data-step]", root);
  let i = 0;
  const stop = () => { delete root.dataset.playing; steps.forEach((li) => li.classList.remove("now")); button.textContent = "▶ Play"; };
  const step = () => {
    steps.forEach((li) => li.classList.remove("now"));
    if (!root.dataset.playing || i >= steps.length || !document.body.contains(root)) return stop();
    steps[i].classList.add("now");
    steps[i].scrollIntoView({ block: "nearest", behavior: "smooth" });
    const s = w.steps[i++];
    const text = `Step ${i}. ${s.do}${s.see ? ` You should see: ${s.see}` : ""}`;
    if (canSpeak && voice && voice.enabled !== false) {
      const u = new SpeechSynthesisUtterance(plainText(text));
      const voices = speechSynthesis.getVoices(), lang = voice.lang || "en-US";
      u.voice = voices.find((v) => v.lang === lang) || voices.find((v) => v.lang.startsWith(lang.split("-")[0])) || null;
      u.lang = lang; u.rate = voice.rate ?? 1; u.pitch = voice.pitch ?? 1;
      u.onend = step; u.onerror = step;
      speechSynthesis.cancel();
      speechSynthesis.speak(u);
    } else setTimeout(step, Math.max(3500, text.length * 60));
  };
  step();
}

const videoEmbed = (url) => `<div class="video"><iframe src="${esc(url)}" title="Lesson video" loading="lazy" allowfullscreen
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" referrerpolicy="strict-origin-when-cross-origin"></iframe></div>`;

/* Ask the instructor, beside a lesson, lab or project: the chat knows which one. */
function tutorCard(t, what) {
  return `<section class="card chat tutor-inline" aria-label="Ask ${esc(t.name)}">
    <div class="row" style="padding:14px 16px 0">${faces([t], 30)}<h3 style="margin:0">Ask ${esc(t.name)} about this ${what}</h3></div>
    <div class="log" id="tlog"><div class="msg assistant"><div class="prose"><p>${what === "lesson" ? "Stuck on anything here, or want another example? Ask me."
      : what === "lab" ? "Stuck on a step, or getting an error? Paste it here and we'll work it out together." : "Want to talk through your plan, or a part you're unsure of? Ask me."}</p></div></div></div>
    <form class="composer" id="tform"><textarea id="tinput" rows="1" maxlength="4000" placeholder="Ask a question…" aria-label="Question"></textarea>
      <button class="btn primary" id="tsend">Ask</button></form>
  </section>`;
}
function wireTutor(root, t, contextId) {
  const log = $("#tlog", root), input = $("#tinput", root), ctx = { name: t.name };
  if (!log) return;
  wireReactions(log, ctx);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#tform", root).requestSubmit(); } });
  $("#tform", root).addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    $("#tsend", root).disabled = true;
    closePicker(log);
    const u = chatBubble("user", text, {}, ctx); log.appendChild(u);
    const a = chatBubble("assistant", "…", {}, ctx); log.appendChild(a);
    log.scrollTop = log.scrollHeight;
    try {
      const { reply, sentId } = await streamChat(t.id, { message: text, lesson_id: contextId }, (acc) => { a.innerHTML = `<div class="prose">${md(withoutReactionTags(acc))}</div>`; log.scrollTop = log.scrollHeight; });
      if (!reply.trim()) a.remove();
      savedExchange(t.id, sentId, u, reply.trim() ? a : null, ctx);
    } catch (err) {
      a.innerHTML = `<span class="error">${esc(err.message)}</span>`;
      if (!input.value) input.value = text;
    }
    $("#tsend", root).disabled = false;
  });
}

function reviewHtml(a, kind) {
  const r = a.review;
  const head = kind === "lab" ? (a.passed ? "✅ Passed" : "🟠 Not passed yet: see what to fix")
    : `${a.score} / 100 · ${a.passed ? "✅ Passed" : "🟠 Not passed yet (70 to pass)"}`;
  return `<div class="review ${a.passed ? "pass" : "fail"} stack">
    <div class="row spread"><b>${head}</b><span class="muted small">${timeAgo(a.created_at)}</span></div>
    ${kind === "lab" ? `<ul class="checklist">${r.checks.map((x) => `<li>${x.met ? "✅" : "❌"} <b>${esc(x.check)}</b><div class="small muted">${esc(x.comment)}</div></li>`).join("")}</ul>`
      : `<div class="table-wrap"><table class="table"><thead><tr><th>Criterion</th><th>Score</th></tr></thead><tbody>${r.criteria.map((x) =>
          `<tr><td><b>${esc(x.criterion)}</b><div class="small muted">${esc(x.comment)}</div></td><td>${x.score}/${x.points}</td></tr>`).join("")}</tbody></table></div>
        ${r.strengths?.length ? `<div><b>What's strong</b><ul>${r.strengths.map((s) => `<li>${esc(s)}</li>`).join("")}</ul></div>` : ""}
        ${r.improvements?.length ? `<div><b>To improve</b><ul>${r.improvements.map((s) => `<li>${esc(s)}</li>`).join("")}</ul></div>` : ""}`}
    <p style="margin:0">${esc(r.feedback)}</p>
    ${r.next_step ? `<div class="small"><b>Next:</b> ${esc(r.next_step)}</div>` : ""}
  </div>`;
}
const earlier = (attempts, kind) => (attempts.length > 1 ? `<details><summary>Earlier attempts (${attempts.length - 1})</summary><div class="stack" style="margin-top:10px">
  ${attempts.slice(1).map((a) => `${reviewHtml(a, kind)}<details class="small"><summary>What you submitted</summary><pre class="submitted">${esc(a.content)}</pre></details>`).join("")}</div></details>` : "");

/* The player: the course's outline beside the item in hand (a lesson, lab, quiz, the project, or the grades). */
async function viewCoursePlayer(m) {
  const slug = m[1], key = m[2];
  const c = await api(`/api/courses/${slug}`);
  if (!c.enrolled) {
    shell("courses", `<div class="card empty stack"><div>Enrol in <b>${esc(c.title)}</b> to start it. It's free.</div>
      <div class="row" style="justify-content:center"><button class="btn primary" id="enrolNow">Enrol for free</button><a class="btn" href="#/course/${slug}">About the ${kindName(c).toLowerCase()}</a></div></div>`);
    $("#enrolNow").addEventListener("click", async () => { await api(`/api/courses/${slug}/enroll`, { method: "POST" }); router(); });
    return;
  }
  const items = allItems(c);
  const idx = items.findIndex((i) => i.key === key);
  if (idx < 0 && key !== "grades") { location.replace(`#/course/${slug}`); return; }
  const item = items[idx], prev = idx > 0 ? items[idx - 1] : null, next = idx >= 0 ? items[idx + 1] : null;
  const wide = matchMedia("(min-width: 981px)").matches;
  const link = (i) => `<a href="#/course/${slug}/learn/${i.key}" class="${i.key === key ? "active" : ""}"><span class="st" title="${STATUS_WORDS[i.status]}">${ITEM_STATUS[i.status]}</span>
    <span><span class="t">${esc(i.title)}</span><span class="type">${{ lesson: "Lesson", lab: "Lab", quiz: "Quiz", project: "Project" }[i.type]} · ${duration(i.minutes)}</span></span></a>`;
  const main = shell("courses", `
    <div class="player-top">
      <div class="muted small crumbs"><a href="#/courses">Courses</a> › <a href="#/course/${slug}">${esc(c.title)}</a></div>
      <div class="row spread small"><b>${c.progress}% complete</b><a href="#/course/${slug}/learn/grades">Grades and certificate</a></div>${bar(c.progress, 100)}
    </div>
    <div class="player">
      <details class="card outline"${wide ? " open" : ""}><summary>Course outline</summary>
        ${c.syllabus.map((mod) => `<h4>Module ${mod.index}: ${esc(mod.title)}</h4>${mod.items.map(link).join("")}`).join("")}
        ${c.project ? `<h4>Capstone project</h4>${link(c.project)}` : ""}
        <h4>Your work</h4><a href="#/course/${slug}/learn/grades" class="${key === "grades" ? "active" : ""}"><span class="st">📊</span><span><span class="t">Grades and certificate</span></span></a>
      </details>
      <div class="stack" id="item"><div class="card empty">Loading…</div></div>
    </div>`);
  const box = $("#item", main);
  const go = (i) => `#/course/${slug}/learn/${i.key}`;
  const short = (t) => (t.length > 38 ? `${t.slice(0, 36)}…` : t);
  const nav = `<div class="row spread item-nav">${prev ? `<a class="btn ghost" href="${go(prev)}">← ${esc(short(prev.title))}</a>` : "<span></span>"}
    ${next ? `<a class="btn ghost" href="${go(next)}">${esc(short(next.title))} →</a>` : `<a class="btn ghost" href="#/course/${slug}/learn/grades">Grades →</a>`}</div>`;
  const ctx = { c, slug, item, next, nav, box, go };
  if (key === "grades") return playerGrades(ctx);
  await { lesson: playerLesson, lab: playerLab, quiz: playerQuiz, project: playerProject }[item.type](ctx);
}

state.courseTeacher = {};  // per course, the instructor the learner chose to teach its lessons

async function playerLesson({ c, slug, item, next, nav, box, go }) {
  const chosen = state.courseTeacher[slug];
  box.innerHTML = `<div class="card empty">${esc(item.module.instructor?.name || "Your instructor")} is preparing this lesson…</div>`;
  let lesson;
  try {
    lesson = await api(`/api/lessons/${item.id}${chosen ? `?companion_id=${chosen}` : ""}`);
  } catch (err) {
    if (err instanceof StaleView) throw err;
    box.innerHTML = `<div class="card empty stack"><div>${esc(err.message)}</div>${nav}</div>`;
    return;
  }
  const t = lesson.teacher, done = lesson.status === "completed";
  box.innerHTML = `
    <article class="card stack lesson-item">
      <div class="muted small">Module ${lesson.course.module} · Lesson ${lesson.course.lesson} · ${lesson.minutes} min</div>
      <h1 style="margin:0">${esc(lesson.title)}</h1>
      <div class="row spread">
        <div class="row">${faces([{ id: t.id, name: t.name, updated_at: t.updated_at }], 42)}<div><b>${esc(t.name)} ${esc(t.flag)}</b><div class="muted small">Your instructor</div></div></div>
        ${c.instructors.length > 1 ? `<label class="row small" style="font-weight:500">Taught by <select id="teacher" style="width:auto">${c.instructors.map((i) =>
          `<option value="${i.id}" ${i.id === t.id ? "selected" : ""}>${esc(i.name)} ${esc(i.flag)}</option>`).join("")}</select></label>` : ""}
      </div>
      ${lesson.video ? videoEmbed(lesson.video) : ""}
      ${/^#+\s*What you'll learn/i.test(lesson.body) ? "" : `<div class="objectives"><b>In this lesson you'll</b><ul>${lesson.objectives.map((o) =>
        `<li>${esc(o.charAt(0).toLowerCase() + o.slice(1))}</li>`).join("")}</ul></div>`}
      <div class="prose" id="lessonBody">${md(lesson.body)}</div>
      ${lesson.walkthrough ? walkthroughHtml(lesson.walkthrough) : ""}
      <div class="row">
        <button class="btn ${done ? "on" : "primary"}" id="complete">${done ? "✅ Completed" : "Mark as complete"}</button>
        <button class="btn" data-study="quiz">📝 Quiz me</button>
        <button class="btn" data-study="cards">🃏 Flashcards</button>
        <button class="btn" data-study="summary">📄 Summary</button>
        ${canSpeak ? `<button class="btn" id="listen">🔊 Listen</button>` : ""}
      </div>
      ${nav}
    </article>
    ${tutorCard(t, "lesson")}`;
  $("#teacher", box)?.addEventListener("change", (e) => { state.courseTeacher[slug] = +e.target.value; router(); });
  $("#complete", box).addEventListener("click", async () => {
    await api(`/api/lessons/${item.id}/complete`, { method: "POST" });
    toast("Lesson completed");
    location.hash = next ? go(next) : `#/course/${slug}/learn/grades`;
  });
  $$("[data-study]", box).forEach((b) => b.addEventListener("click", () => studyAction(b.dataset.study, "lesson", item.id, { companionId: t.id, button: b })));
  $("#listen", box)?.addEventListener("click", async () => {
    try { speak(lesson.body, (await api(`/api/companions/${t.id}`)).persona.voice); } catch (err) { toast(err.message); }
  });
  if (lesson.walkthrough) {
    const walk = $("#walk", box);
    wireSteps(walk);
    $("#playWalk", walk).addEventListener("click", (e) => playWalkthrough(walk, lesson.walkthrough, t.id, e.currentTarget));
  }
  wireTutor(box, t, item.id);
}

async function playerLab({ slug, item, nav, box }) {
  const m = +item.key.slice(3);
  const lab = await api(`/api/courses/${slug}/labs/${m}`);
  const tickKey = `aiet-lab:${slug}:${m}`, done = ticked(tickKey), os = keysFor(), latest = lab.attempts[0];
  box.innerHTML = `
    <article class="card stack">
      <div class="muted small">Module ${m} · Hands-on lab · ${esc(lab.tool)} · about ${duration(lab.minutes)}</div>
      <h1 style="margin:0">🧪 ${esc(lab.title)}</h1>
      <p style="margin:0">${esc(lab.goal)}</p>
      ${lab.setup.length ? `<div><b>What you need</b><ul>${lab.setup.map((s) => `<li>${esc(s)}</li>`).join("")}</ul></div>` : ""}
      ${lab.files.length ? `<div class="row"><a class="btn" href="/api/courses/${slug}/labs/${m}/files" download>⬇️ Download the starter files</a>
        <span class="muted small">${lab.files.map(esc).join(", ")}</span></div>` : ""}
      <section class="walk" aria-label="Lab steps">
        <div class="walk-head"><div><div class="tool">In ${esc(lab.tool)}</div><b>Steps</b></div>${lab.steps.some((s) => s.keys) ? osSwitch(os) : ""}</div>
        <ol class="ticks">${lab.steps.map((s, i) => stepHtml(s, os, i, done.has(i))).join("")}</ol>
      </section>
      <div class="passes"><b>Your lab passes when</b><ul class="checklist">${lab.checks.map((ch) => `<li>☑️ ${esc(ch)}</li>`).join("")}</ul></div>
      ${lab.hints.length ? `<details><summary>Hints</summary><ul>${lab.hints.map((h) => `<li>${esc(h)}</li>`).join("")}</ul></details>` : ""}
    </article>
    <section class="card stack" id="submitBox">
      <h2 style="margin:0">Submit your work</h2>
      <p class="muted" style="margin:0">${esc(lab.submit)}</p>
      <form id="workForm" class="stack">
        <textarea name="content" rows="10" maxlength="40000" required class="mono" placeholder="Paste your work here" aria-label="Your work"></textarea>
        <div class="row"><button class="btn primary">Submit for review</button>
          <span class="muted small">${esc(lab.instructor?.name || "Your instructor")} reviews it against the checks above.</span></div>
      </form>
      <div id="review">${latest ? reviewHtml(latest, "lab") : ""}</div>
      ${earlier(lab.attempts, "lab")}
    </section>
    ${nav}
    ${lab.instructor ? tutorCard(lab.instructor, "lab") : ""}`;
  wireSteps(box, tickKey);
  wireWork(box, `/api/courses/${slug}/labs/${m}/submit`, "lab");
  if (lab.instructor) wireTutor(box, lab.instructor, lab.id);
}

function wireWork(box, url, kind) {
  $("#workForm", box).addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("button", e.target);
    btn.disabled = true; btn.textContent = "Your instructor is reviewing it…";
    try {
      const a = await api(url, { method: "POST", body: { content: e.target.content.value } });
      toast(a.passed ? (kind === "lab" ? "Lab passed!" : "Project passed!") : "Reviewed: see what to fix");
      router();  // the outline and progress change too
    } catch (err) { toast(err.message); btn.disabled = false; btn.textContent = "Submit for review"; }
  });
}

function playerQuiz({ item, nav, box }) {
  const mod = item.module, passed = item.status === "passed";
  box.innerHTML = `<article class="card stack">
    <div class="muted small">Module ${mod.index} · Graded quiz · about ${item.minutes} min</div>
    <h1 style="margin:0">📝 ${esc(item.title)}: ${esc(mod.title)}</h1>
    <p style="margin:0">${item.questions} questions on this module's lessons, written by ${esc(mod.instructor?.name || "your instructor")}. You pass with 70% or more,
      and you can take a new quiz as often as you like: your best score counts.</p>
    ${item.best_score != null ? `<div class="card row spread result ${passed ? "pass" : "fail"}"><div><div class="big">${item.best_score}%</div>
      <div class="muted small">your best score</div></div><b>${passed ? "✅ Passed" : "Not passed yet: try a new quiz"}</b></div>` : ""}
    <div class="row"><button class="btn primary" id="startQuiz">${item.best_score == null ? "Start the quiz" : "Take a new quiz"}</button></div>
    ${nav}
  </article>`;
  $("#startQuiz", box).addEventListener("click", (e) => studyAction("quiz", "path", item.id, { companionId: mod.instructor?.id ?? null, button: e.target }));
}

async function playerProject({ slug, nav, box }) {
  const p = await api(`/api/courses/${slug}/project`);
  const latest = p.attempts[0];
  box.innerHTML = `
    <article class="card stack">
      <div class="muted small">Capstone project · about ${duration(p.minutes)}</div>
      <h1 style="margin:0">🏗️ ${esc(p.title)}</h1>
      <p style="margin:0">${esc(p.brief)}</p>
      <div><b>What to hand in</b><ul>${p.deliverables.map((d) => `<li>${esc(d)}</li>`).join("")}</ul></div>
      ${p.files.length ? `<div class="row"><a class="btn" href="/api/courses/${slug}/project/files" download>⬇️ Download the starter files</a></div>` : ""}
      <div><b>How it's scored</b> <span class="muted small">(${p.pass_mark} of 100 points to pass)</span>
        <div class="table-wrap"><table class="table"><thead><tr><th>Criterion</th><th>Points</th></tr></thead><tbody>${p.rubric.map((r) =>
          `<tr><td><b>${esc(r.name)}</b><div class="small muted">${esc(r.description)}</div></td><td>${r.points}</td></tr>`).join("")}</tbody></table></div></div>
    </article>
    <section class="card stack" id="submitBox">
      <h2 style="margin:0">Submit your project</h2>
      <p class="muted" style="margin:0">${esc(p.submit)}</p>
      <form id="workForm" class="stack">
        <textarea name="content" rows="14" maxlength="40000" required class="mono" placeholder="Paste your project here" aria-label="Your project"></textarea>
        <div class="row"><button class="btn primary">Submit for review</button>
          <span class="muted small">${esc(p.instructor?.name || "Your instructor")} scores it against the rubric.</span></div>
      </form>
      <div id="review">${latest ? reviewHtml(latest, "project") : ""}</div>
      ${earlier(p.attempts, "project")}
    </section>
    ${nav}
    ${p.instructor ? tutorCard(p.instructor, "project") : ""}`;
  wireWork(box, `/api/courses/${slug}/project/submit`, "project");
  if (p.instructor) wireTutor(box, p.instructor, p.id);
}

function playerGrades({ c, slug, box, go }) {
  const graded = allItems(c).filter((i) => i.type !== "lesson");
  const grade = (i) => (i.type === "quiz" ? (i.best_score != null ? `${i.best_score}%` : "–") : i.type === "project" ? (i.best_score != null ? `${i.best_score}/100` : "–") : "");
  const lead = c.instructors[0];
  box.innerHTML = `
    <div class="card stack">
      <h1 style="margin:0">Grades</h1>
      <div class="table-wrap"><table class="table"><thead><tr><th>Item</th><th>Status</th><th>Grade</th></tr></thead><tbody>
        ${graded.map((i) => `<tr><td><a href="${go(i)}">${ITEM_ICON[i.type]} ${esc(i.title)}</a>${i.module ? `<div class="small muted">Module ${i.module.index}: ${esc(i.module.title)}</div>` : ""}</td>
          <td>${ITEM_STATUS[i.status]} ${STATUS_WORDS[i.status]}</td><td>${grade(i)}</td></tr>`).join("")}
      </tbody></table></div>
    </div>
    ${progressBlock(c, { grades: true })}
    <div class="card stack">
      <h2 style="margin:0">Practise the whole ${kindName(c).toLowerCase()}</h2>
      <p class="muted" style="margin:0">A timed mock exam on every lesson (it doesn't count towards your certificate), or flashcards to review them all.</p>
      <div class="row"><button class="btn primary" id="exam">🎓 Mock exam</button><button class="btn" id="cards">🃏 Flashcards</button></div>
    </div>`;
  wireProgress(box, c);
  $("#exam", box).addEventListener("click", (e) => studyAction("exam", "course", slug, { companionId: lead?.id ?? null, button: e.target }));
  $("#cards", box).addEventListener("click", (e) => studyAction("cards", "course", slug, { companionId: lead?.id ?? null, button: e.target }));
}

/* ------------------------------------------------------------------ tracks */

async function viewTrack(m) {
  const t = await api(`/api/tracks/${m[1]}`);
  const complete = t.done === t.total;
  const main = shell("courses", `
    <div class="muted small crumbs"><a href="#/courses">Courses</a> › ${esc(t.kind)}s</div>
    <section class="card course-head" style="--c1:#5b4cdb;--c2:#8a7dff">
      <div class="badges"><span class="badge">${esc(t.kind)}</span><span class="badge">${t.total}-course series</span></div>
      <h1>${esc(t.title)}</h1>
      <p class="lead">${esc(t.subtitle)}</p>
      <div class="row">${faces(t.instructors.slice(0, 8), 34)}<span class="small muted">Taught by ${plural2(t.instructors.length, "AI Learning Companion")}</span></div>
      <div>${bar(t.done, t.total)}<div class="small muted">${t.done} of ${t.total} courses finished</div></div>
    </section>
    <div class="card course-stats">
      <div><b>${plural2(t.total, "course")}</b><span class="muted small">taken in order</span></div>
      <div><b>${t.levels.map(label).join(" to ")}</b><span class="muted small">level</span></div>
      <div><b>${hoursText(t.hours)}</b><span class="muted small">at your own pace</span></div>
      <div><b>${esc(t.kind)}</b><span class="muted small">when you finish every course</span></div>
    </div>
    <div class="two-col" style="margin-top:16px">
      <div class="card stack"><h2 style="margin:0">About this ${esc(t.kind)}</h2><p style="margin:0">${esc(t.description)}</p>
        ${t.roles.length ? `<div><h3>It prepares you for roles such as</h3>${chipList(t.roles)}</div>` : ""}
        <div><h3>Skills you'll gain</h3>${chipList(t.skills)}</div></div>
      <div class="card stack" id="trackCert"><h2 style="margin:0">🎓 Your certificate</h2>
        ${t.certificate ? `<p style="margin:0">You've earned the ${esc(t.title)} ${esc(t.kind)}.</p><div><a class="btn primary" href="#/certificate/${t.certificate}">View your certificate</a></div>`
          : complete ? `<form id="certForm" class="stack cert-form"><div><b>🎉 You've finished every course!</b></div>
              <label>Name on your certificate <input type="text" name="name" value="${esc(state.me.display_name)}" minlength="2" maxlength="80" required></label>
              <div><button class="btn primary">Get my certificate</button></div></form>`
          : `<p class="muted" style="margin:0">Earn the certificate for each of its ${t.total} courses to get the ${esc(t.kind)} certificate.</p>`}
      </div>
    </div>
    <h2 class="section">${t.total} courses in this ${esc(t.kind)}</h2>
    <ol class="track-courses">${t.courses.map((c, i) => `<li class="card">
      <div class="muted small">Course ${i + 1} · ${label(c.level)} · ${hoursText(c.hours)}</div>
      <div class="row spread"><a class="plain" href="#/course/${c.slug}"><h3 style="margin:0">${esc(c.title)}</h3></a>
        <span>${c.certificate ? "🎓 Done" : c.enrolled ? `${c.progress}%` : ""}</span></div>
      <div class="small muted">${esc(c.subtitle)}</div>
      ${c.enrolled ? bar(c.progress, 100) : ""}
      <div class="row small">${faces(c.instructors, 24)}<span class="muted">${esc(c.instructors.map((x) => x.name).join(", "))}</span></div>
    </li>`).join("")}</ol>`);
  $("#certForm", main)?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("button", e.target);
    btn.disabled = true;
    try {
      const cert = await api(`/api/tracks/${t.slug}/certificate`, { method: "POST", body: { name: e.target.name.value } });
      location.hash = `#/certificate/${cert.code}`;
    } catch (err) { toast(err.message); btn.disabled = false; }
  });
}

/* --------------------------------------------------------------- certificate */

/* A certificate's own page, open to anyone with its link, so learners can share it and others can check it. */
async function viewCertificate(m) {
  let cert;
  try {
    cert = await api(`/api/certificates/${m[1]}`);
  } catch (err) {
    if (err instanceof StaleView) throw err;
    app.innerHTML = `<div class="narrow-card card stack">${logo()}<p>${esc(err.message)}</p><div><a class="btn" href="#/feed">Go to AI Every Time</a></div></div>`;
    return;
  }
  const issued = new Date(cert.issued_at);
  const day = issued.toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" });
  const url = `${location.origin}/#/certificate/${cert.code}`;
  const f = cert.for, track = cert.kind === "track";
  const heading = track ? f.kind : f.kind === "Guided project" ? "Guided Project Certificate" : "Course Certificate";
  const what = track ? `a ${f.courses.length}-course ${f.kind} of about ${hoursText(f.hours)}`
    : `${lengthOf(f.hours)} ${f.level} ${f.kind.toLowerCase()}`;
  const linkedIn = new URLSearchParams({ startTask: "CERTIFICATION_NAME", name: f.title, organizationName: "AI Every Time",
    issueYear: String(issued.getFullYear()), issueMonth: String(issued.getMonth() + 1), certUrl: url, certId: cert.code });
  app.innerHTML = `
    <div class="cert-page">
      <div class="cert-bar">${logo(30)}${state.me ? `<a class="btn" href="#/courses">Back to courses</a>` : `<a class="btn primary" href="#/welcome">Learn with AI Every Time</a>`}</div>
      <article class="cert" aria-label="Certificate">
        <div class="cert-top"><img src="/static/icons/logo-256.webp" width="56" height="56" alt="">
          <div><div class="cert-brand">AI <span>Every</span> Time</div><div class="cert-kind">${esc(heading)}</div></div></div>
        <div class="cert-main">
          <div class="cert-date">${esc(day)}</div>
          <div class="cert-name">${esc(cert.name)}</div>
          <div class="cert-has">has successfully completed</div>
          <div class="cert-course">${esc(f.title)}</div>
          <div class="cert-desc">${esc(what)}, taught by AI Learning Companions on AI Every Time</div>
        </div>
        <div class="cert-foot">
          <div class="cert-signs">${cert.instructors.map((i) => `<div class="sign"><img class="avatar" src="${faceUrl(i)}" width="38" height="38" alt="">
            <div><b>${esc(i.name)}</b><div>Instructor · AI Learning Companion</div></div></div>`).join("")}</div>
          <div class="cert-verify">Check this certificate at<br><b>${esc(url.replace(/^https?:\/\//, ""))}</b></div>
        </div>
      </article>
      <div class="row cert-actions">
        <button class="btn primary" id="printCert">🖨️ Print or save as PDF</button>
        <button class="btn" id="copyCert">🔗 Copy the link</button>
        <a class="btn" target="_blank" rel="noopener noreferrer" href="https://www.linkedin.com/profile/add?${linkedIn}">Add to LinkedIn</a>
      </div>
      <p class="muted small cert-note">This certificate shows that ${esc(cert.name)} completed every lesson, graded quiz, lab and project in this
        ${track ? "series" : f.kind.toLowerCase()} on AI Every Time, reviewed by its AI instructors. It isn't an academic credit, or a certification from any company whose products it teaches.</p>
    </div>`;
  document.title = `${cert.name}: ${f.title} · AI Every Time`;
  $("#printCert").addEventListener("click", () => print());
  $("#copyCert").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(url); toast("Link copied"); } catch { prompt("Copy this link:", url); }
  });
}

router();
