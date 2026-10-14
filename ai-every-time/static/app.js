"use strict";

/* ---------------------------------------------------------------- utilities */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const app = $("#app");
const state = { me: null, options: null };

async function api(path, { method = "GET", body } = {}) {
  const res = await fetch(path, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : {},
    body: body !== undefined ? JSON.stringify(body) : undefined,
    credentials: "same-origin",
  });
  if (res.status === 401 && !path.startsWith("/api/auth")) {
    state.me = null;
    location.hash = "#/welcome";
    throw new Error("Please sign in");
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
  if (res.status === 204) return null;
  return res.headers.get("content-type")?.includes("json") ? res.json() : res.text();
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 2400);
}

/* Minimal, safe markdown: escape first, then a few inline/block rules. */
function md(src) {
  const lines = esc(src).split(/\n/);
  let html = "", list = null;
  const inline = (s) => s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
  const close = () => { if (list) { html += `</${list}>`; list = null; } };
  let para = [];
  const flush = () => { if (para.length) { html += `<p>${inline(para.join("<br>"))}</p>`; para = []; } };
  for (const line of lines) {
    const ul = line.match(/^\s*[-*•]\s+(.*)/), ol = line.match(/^\s*\d+[.)]\s+(.*)/), h = line.match(/^#{1,6}\s+(.*)/);
    if (ul || ol) {
      flush();
      const kind = ul ? "ul" : "ol";
      if (list !== kind) { close(); html += `<${kind}>`; list = kind; }
      html += `<li>${inline((ul || ol)[1])}</li>`;
    } else if (h) { flush(); close(); html += `<h4>${inline(h[1])}</h4>`; }
    else if (!line.trim()) { flush(); close(); }
    else { close(); para.push(line); }
  }
  flush(); close();
  return html;
}

const avatarUrl = (c) => `/api/companions/${c.id}/avatar.svg?v=${encodeURIComponent(c.updated_at || "")}`;
const avatarImg = (c, size = 48) => `<img class="avatar" src="${avatarUrl(c)}" width="${size}" height="${size}" alt="${esc(c.persona.name)}">`;
const label = (s) => s.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
const timeAgo = (iso) => {
  const s = (Date.now() - new Date(iso)) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
};

/* ------------------------------------------------------------------- router */

const routes = [
  [/^#\/welcome$/, viewWelcome, false],
  [/^#\/onboarding$/, viewOnboarding],
  [/^#\/feed$/, viewFeed],
  [/^#\/companions$/, viewCompanions],
  [/^#\/c\/(\d+)(?:\?post=(\d+))?$/, viewCompanion],
  [/^#\/studio\/new$/, (m) => viewStudio({ mode: "new" })],
  [/^#\/studio\/remix\/(\d+)$/, (m) => viewStudio({ mode: "remix", id: +m[1] })],
  [/^#\/studio\/(\d+)$/, (m) => viewStudio({ mode: "edit", id: +m[1] })],
  [/^#\/profile$/, viewProfile],
];

async function router() {
  const hash = location.hash || "#/feed";
  if (!state.me) {
    try { state.me = await api("/api/me"); } catch { state.me = null; }
  }
  $("#banner").hidden = !state.me?.offline_mode;
  $("#banner").textContent = "Demo mode: companions use templated replies. Set ANTHROPIC_API_KEY on the server for real AI conversations and content.";
  for (const [re, view, needsAuth = true] of routes) {
    const m = hash.match(re);
    if (!m) continue;
    if (needsAuth && !state.me) { location.hash = "#/welcome"; return; }
    if (!needsAuth && state.me) { location.hash = "#/feed"; return; }
    if (needsAuth && !state.me.onboarded && !/^#\/(onboarding|studio)/.test(hash)) { location.hash = "#/onboarding"; return; }
    try { await view(m); } catch (e) { console.error(e); toast(e.message); }
    window.scrollTo(0, 0);
    return;
  }
  location.hash = state.me ? "#/feed" : "#/welcome";
}
window.addEventListener("hashchange", router);

function shell(active, inner) {
  const nav = [["feed", "🏠", "Feed"], ["companions", "🌍", "Companions"], ["profile", "👤", "Profile"]];
  const primary = state.me?.primary_companion_id;
  app.innerHTML = `
    <div class="shell">
      <aside class="side">
        <div class="logo">AI <span>Every</span> Time</div>
        <nav class="nav">
          ${nav.map(([k, i, t]) => `<a href="#/${k}" class="${active === k ? "active" : ""}">${i} ${t}</a>`).join("")}
          ${primary ? `<a href="#/c/${primary}" class="${active === "mine" ? "active" : ""}">💬 My companion</a>` : ""}
          <a href="#/studio/new" class="${active === "studio" ? "active" : ""}">✨ Design a companion</a>
        </nav>
        <div class="me"><div><b>${esc(state.me.display_name)}</b><div class="muted small">${esc(state.me.level)}</div></div></div>
      </aside>
      <main class="main">${inner}</main>
    </div>`;
  return $(".main");
}

/* ------------------------------------------------------------------ welcome */

async function viewWelcome() {
  app.innerHTML = `
    <section class="landing">
      <div>
        <div class="logo" style="margin:0 0 24px">AI <span>Every</span> Time</div>
        <h1>Every piece of AI content is the <em>start</em> of a learning relationship.</h1>
        <p class="muted" style="font-size:18px">Follow, chat with and learn from always-available AI Learning Companions from around the world, each with their own face, voice and expertise. Or design your own companion, exactly the way you want it.</p>
        <div class="faces">${[1, 2, 3, 4, 5, 6, 7, 8].map((i) => `<img class="avatar" src="/api/companions/${i}/avatar.svg" width="56" height="56" alt="">`).join("")}</div>
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
          <div class="error" id="authErr"></div>
          <button class="btn primary" type="submit" id="authBtn">Create account</button>
        </form>
      </div>
    </section>`;
  let tab = "signup";
  const setTab = (t) => {
    tab = t;
    $$("[data-tab]").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
    $$("[data-only=signup]").forEach((el) => (el.hidden = t !== "signup"));
    $("#authBtn").textContent = t === "signup" ? "Create account" : "Sign in";
    $("[name=password]").autocomplete = t === "signup" ? "new-password" : "current-password";
  };
  $$("[data-tab]").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
  $("#authForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target));
    const body = tab === "signup" ? f : { email: f.email, password: f.password };
    $("#authErr").textContent = "";
    $("#authBtn").disabled = true;
    try {
      state.me = await api(`/api/auth/${tab}`, { method: "POST", body });
      location.hash = state.me.onboarded ? "#/feed" : "#/onboarding";
    } catch (err) {
      $("#authErr").textContent = err.message;
    } finally {
      $("#authBtn").disabled = false;
    }
  });
}

/* --------------------------------------------------------------- onboarding */

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
        </div>
      </div>
      <p class="tagline">${esc(p.tagline)}</p>
      <div class="row">${p.expertise.map((x) => `<span class="chip">${esc(x)}</span>`).join("")}</div>
      ${p.writing_style.sample_line ? `<p class="small" style="margin:0"><i>“${esc(p.writing_style.sample_line)}”</i></p>` : ""}
      <div class="actions">
        ${onboarding
          ? `<button class="btn primary" data-act="choose">Choose ${esc(p.name)}</button>
             <a class="btn" href="#/studio/remix/${c.id}">Customise</a>`
          : `<a class="btn primary" href="#/c/${c.id}">Chat</a>
             ${mine ? "" : `<button class="btn ${c.following ? "on" : ""}" data-act="follow">${c.following ? "Following" : "Follow"}</button>`}
             ${mine ? `<a class="btn" href="#/studio/${c.id}">Edit</a>` : `<a class="btn" href="#/studio/remix/${c.id}">Remix</a>`}
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
        } else if (act === "follow") {
          await api(`/api/companions/${c.id}/follow`, { method: c.following ? "DELETE" : "POST" });
          toast(c.following ? `Unfollowed ${c.persona.name}` : `Following ${c.persona.name}`);
        }
        rerender();
      } catch (err) { toast(err.message); }
    });
  });
}

async function viewOnboarding() {
  const list = await api("/api/companions");
  const starters = list.filter((c) => c.is_starter);
  app.innerHTML = `
    <section class="main" style="margin:0 auto">
      <div class="logo" style="margin:0 0 20px">AI <span>Every</span> Time</div>
      <div class="steps"><span>1. Account ✓</span><span>›</span><b>2. Your companion</b><span>›</span><span>3. Start learning</span></div>
      <h1>Welcome, ${esc(state.me.display_name)}! Who do you want to learn with?</h1>
      <p class="muted">Pick one of our companions as-is, customise one, or design your own from scratch. You can change your companion's look, personality and style whenever you like.</p>
      <div class="card row spread" style="margin:20px 0">
        <div><h3 style="margin:0">✨ Design your own companion</h3><div class="muted small">Choose their face, country, personality, teaching style and writing voice.</div></div>
        <a class="btn primary" href="#/studio/new">Start designing</a>
      </div>
      <h2>Or choose a companion</h2>
      <div class="grid" id="starters">${starters.map((c) => companionCard(c, { onboarding: true })).join("")}</div>
    </section>`;
  wireCompanionCards($("#starters"), starters, viewOnboarding);
}

/* --------------------------------------------------------------------- feed */

function postCard(post, c) {
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
        <button class="btn ${post.liked ? "on" : ""}" data-eng="like">♥ ${post.likes || ""}</button>
        <button class="btn ${post.marked_helpful ? "on" : ""}" data-eng="helpful">💡 Helpful ${post.helpful || ""}</button>
        <button class="btn ghost" data-eng="share">↗ Share</button>
      </div>
    </article>`;
}

function wirePosts(root, posts, rerender) {
  const seen = new Set();
  const io = "IntersectionObserver" in window ? new IntersectionObserver((entries) => {
    for (const en of entries) {
      const id = +en.target.dataset.post;
      if (en.isIntersecting && !seen.has(id)) { seen.add(id); api(`/api/posts/${id}/engage`, { method: "POST", body: { kind: "view" } }).catch(() => {}); }
    }
  }, { threshold: 0.6 }) : null;
  $$("[data-post]", root).forEach((el) => {
    io?.observe(el);
    const post = posts.find((p) => p.id === +el.dataset.post);
    el.addEventListener("click", async (e) => {
      const kind = e.target.closest("[data-eng]")?.dataset.eng;
      if (!kind) return;
      try {
        if (kind === "share") {
          const url = `${location.origin}/#/c/${post.companion_id}?post=${post.id}`;
          await navigator.clipboard?.writeText(url).catch(() => {});
          await api(`/api/posts/${post.id}/engage`, { method: "POST", body: { kind } });
          toast("Link copied");
          return;
        }
        const on = kind === "like" ? post.liked : post.marked_helpful;
        if (on) await api(`/api/posts/${post.id}/engage/${kind}`, { method: "DELETE" });
        else await api(`/api/posts/${post.id}/engage`, { method: "POST", body: { kind } });
        rerender();
      } catch (err) { toast(err.message); }
    });
  });
}

async function viewFeed() {
  const [posts, list] = await Promise.all([api("/api/feed"), api("/api/companions")]);
  const byId = Object.fromEntries(list.map((c) => [c.id, c]));
  const main = shell("feed", `
    <div class="row spread" style="margin-bottom:16px">
      <div><h1 style="margin:0">Your feed</h1><div class="muted">Fresh lessons from the companions you follow. Every post is a conversation starter.</div></div>
      <a class="btn" href="#/companions">Find more companions</a>
    </div>
    <div class="stack" id="posts">
      ${posts.length ? posts.map((p) => postCard(p, byId[p.companion_id])).join("") : `<div class="card empty">Follow a few companions to fill your feed. <a href="#/companions">Browse companions</a></div>`}
    </div>`);
  wirePosts(main, posts, viewFeed);
}

/* --------------------------------------------------------------- companions */

async function viewCompanions() {
  const list = await api("/api/companions");
  const mine = list.filter((c) => !c.is_starter), starters = list.filter((c) => c.is_starter);
  const main = shell("companions", `
    <h1>Companions</h1>
    <p class="muted">Each companion represents a country and brings their own expertise, perspective and voice.</p>
    ${mine.length ? `<h2>Your companions</h2><div class="grid" style="margin-bottom:28px">${mine.map((c) => companionCard(c)).join("")}</div>` : ""}
    <h2>Around the world</h2>
    <div class="grid">${starters.map((c) => companionCard(c)).join("")}</div>`);
  wireCompanionCards(main, list, viewCompanions);
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
  const main = shell(state.me.primary_companion_id === id ? "mine" : "companions", `
    <div class="row" style="margin-bottom:18px">
      ${avatarImg(c, 72)}
      <div style="flex:1;min-width:200px">
        <h1 style="margin:0">${esc(p.name)} ${esc(p.flag)}</h1>
        <div class="muted">${esc(p.tagline || p.country)}</div>
        <div class="row small" style="margin-top:6px">${p.expertise.map((x) => `<span class="chip">${esc(x)}</span>`).join("")}<span class="muted">${c.followers} follower${c.followers === 1 ? "" : "s"}</span></div>
      </div>
      <div class="row">
        ${c.is_mine ? `<a class="btn" href="#/studio/${id}">Edit companion</a>` : `<button class="btn ${c.following ? "on" : ""}" id="followBtn">${c.following ? "Following" : "Follow"}</button><a class="btn" href="#/studio/remix/${id}">Remix</a>`}
      </div>
    </div>
    <div class="chat-layout">
      <section class="card chat">
        <div class="log" id="log"></div>
        <div id="ctx"></div>
        <form class="composer" id="composer">
          <textarea id="input" rows="1" maxlength="4000" placeholder="Ask ${esc(p.name)} anything about AI…" aria-label="Message"></textarea>
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
          <div class="row spread"><h3 style="margin:0">Recent posts</h3>${c.is_mine ? `<button class="btn" id="genBtn">Write a post</button>` : ""}</div>
          <div id="recent" class="stack small" style="margin-top:10px"></div>
        </div>
      </aside>
    </div>`);

  const log = $("#log");
  const addMsg = (role, text) => {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    el.innerHTML = role === "assistant" ? `<div class="prose">${md(text)}</div>` : esc(text).replace(/\n/g, "<br>");
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return el;
  };
  if (!messages.length) {
    addMsg("assistant", `Hi ${state.me.display_name}! I'm ${p.name} from ${p.country}. ${p.writing_style.signature_phrase || ""}\n\nWhat would you like to learn today?`);
  }
  messages.forEach((x) => addMsg(x.role, x.content));

  const renderCtx = () => {
    $("#ctx").innerHTML = contextPost ? `<div class="context-pill"><span>Asking about: <b>${esc(contextPost.title)}</b></span><button class="linkbtn" id="clearCtx">✕</button></div>` : "";
    $("#clearCtx")?.addEventListener("click", () => { contextPost = null; renderCtx(); });
  };
  renderCtx();

  const renderMem = (mems) => {
    $("#mem").innerHTML = mems.length
      ? `<ul>${mems.map((x) => `<li><span>${esc(x.fact)}</span><button class="linkbtn" data-forget="${x.id}" title="Forget this">Forget</button></li>`).join("")}</ul>`
      : `<p class="muted small" style="margin:0">Nothing yet. ${esc(p.name)} picks up your goals and progress as you chat, and you can remove anything here.</p>`;
  };
  renderMem(memories);
  $("#mem").addEventListener("click", async (e) => {
    const fid = e.target.dataset.forget;
    if (!fid) return;
    await api(`/api/memories/${fid}`, { method: "DELETE" });
    renderMem(await api(`/api/companions/${id}/memories`));
  });

  const renderRecent = (list) => {
    $("#recent").innerHTML = list.length
      ? list.slice(0, 6).map((x) => `<a href="#/c/${id}?post=${x.id}" data-ctx="${x.id}">${esc(x.title)}</a>`).join("")
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

  $("#followBtn")?.addEventListener("click", async () => {
    await api(`/api/companions/${id}/follow`, { method: c.following ? "DELETE" : "POST" });
    viewCompanion(m);
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
    if (!text) return;
    input.value = "";
    $("#send").disabled = true;
    addMsg("user", text);
    const bubble = addMsg("assistant", "…");
    try {
      const res = await fetch(`/api/companions/${id}/chat`, {
        method: "POST", headers: { "Content-Type": "application/json" }, credentials: "same-origin",
        body: JSON.stringify({ message: text, post_id: contextPost?.id ?? null }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Couldn't send message");
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let acc = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        acc += dec.decode(value, { stream: true });
        bubble.innerHTML = `<div class="prose">${md(acc)}</div>`;
        log.scrollTop = log.scrollHeight;
      }
      contextPost = null;
      renderCtx();
      setTimeout(async () => renderMem(await api(`/api/companions/${id}/memories`)), 2500);
    } catch (err) {
      bubble.innerHTML = `<span class="error">${esc(err.message)}</span>`;
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
    avatar: { skin: "#C68642", hair_style: "short", hair_color: "#2C1B10", background: "#6C5CE7", accessory: "none", accessory_color: "#222222", expression: "smile", shirt: "#1E272E" },
    interests: [], custom_instructions: "",
  };
}

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
              <datalist id="countries">${COUNTRIES.map(([n]) => `<option value="${n}">`).join("")}</datalist>
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

        <fieldset><legend>Custom instructions</legend>
          <label><span class="hint">Anything else about how they should teach you. e.g. "Quiz me at the end of every explanation" or "Relate everything to finance, my field"</span>
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

  form.addEventListener("input", (e) => {
    const t = e.target;
    if (t.dataset.trait) persona.personality[t.dataset.trait] = +t.value;
    else if (t.name === "interests") persona.interests = t.value.split(",").map((s) => s.trim()).filter(Boolean).slice(0, 8);
    else if (t.name?.startsWith("ws.")) set(`writing_style.${t.name.slice(3)}`, t.type === "checkbox" ? t.checked : t.value);
    else if (t.name === "country") {
      persona.country = t.value;
      persona.flag = (COUNTRIES.find(([n]) => n.toLowerCase() === t.value.trim().toLowerCase()) || [, ""])[1];
    } else if (t.name) persona[t.name] = t.value;
    else if (t.type === "color") { persona.avatar[t.closest("[data-key]").dataset.key] = t.value.toUpperCase(); syncSwatches(); }
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
    syncSwatches(); preview();
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
    else persona.avatar[g.dataset.choice] = b.dataset.v;
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

  let pvTimer, pvUrl;
  function preview() {
    $("#pvName").textContent = persona.name || "Your companion";
    $("#pvCountry").textContent = `${persona.flag} ${persona.country || "Choose a country"}`;
    $("#pvTag").textContent = persona.tagline;
    const traits = SLIDERS.map(([k, lo, hi]) => (persona.personality[k] >= 60 ? hi : persona.personality[k] <= 40 ? lo : null)).filter(Boolean);
    $("#pvVoice").innerHTML = `<b>Personality:</b> ${esc(traits.join(", ") || "Balanced")}<br><b>Teaches:</b> ${esc(TEACH_DESC[persona.teaching_style])}<br><b>Writes:</b> ${esc(label(persona.writing_style.length))}, ${esc(persona.writing_style.level)} level${persona.writing_style.sample_line ? `<br><i>“${esc(persona.writing_style.sample_line)}”</i>` : ""}`;
    clearTimeout(pvTimer);
    pvTimer = setTimeout(async () => {
      const res = await fetch("/api/avatar/preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(persona.avatar) });
      if (!res.ok) return;
      const blob = await res.blob();
      if (pvUrl) URL.revokeObjectURL(pvUrl);
      pvUrl = URL.createObjectURL(blob);
      $("#pv").src = pvUrl;
    }, 120);
  }
  preview();

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    $("#studioErr").textContent = "";
    if (!persona.name.trim() || !persona.country.trim()) { $("#studioErr").textContent = "Give your companion a name and a country."; return; }
    $("#saveBtn").disabled = true;
    try {
      let saved;
      if (mode === "edit") saved = await api(`/api/companions/${id}`, { method: "PUT", body: persona });
      else if (mode === "remix") saved = await api(`/api/companions/${id}/remix`, { method: "POST", body: { overrides: persona, make_primary: true } });
      else saved = await api("/api/companions", { method: "POST", body: { persona, make_primary: true } });
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

async function viewProfile() {
  const main = shell("profile", `
    <h1>Your profile</h1>
    <form class="card stack" id="prof" style="max-width:560px">
      <label>Name <input type="text" name="display_name" maxlength="60" required></label>
      <label>AI level <select name="level"><option value="beginner">Beginner</option><option value="intermediate">Intermediate</option><option value="advanced">Advanced</option></select></label>
      <label>Learning goals <span class="hint">Your companions read this to personalise lessons</span><textarea name="goals" maxlength="500"></textarea></label>
      <div class="error" id="profErr"></div>
      <div class="row"><button class="btn primary">Save</button><button type="button" class="btn danger" id="logout">Sign out</button></div>
    </form>`);
  const f = $("#prof", main);
  f.display_name.value = state.me.display_name; f.level.value = state.me.level; f.goals.value = state.me.goals;
  f.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      state.me = await api("/api/me", { method: "PATCH", body: Object.fromEntries(new FormData(f)) });
      toast("Profile saved");
    } catch (err) { $("#profErr").textContent = err.message; }
  });
  $("#logout").addEventListener("click", async () => {
    await api("/api/auth/logout", { method: "POST" });
    state.me = null;
    location.hash = "#/welcome";
  });
}

router();
