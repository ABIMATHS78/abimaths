"use strict";

/* AI Every Time service worker: shows notifications and opens the right page when one is tapped.
   It doesn't cache anything, so the app is always the latest version. */

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("push", (event) => {
  let data = {};
  try { data = event.data ? event.data.json() : {}; } catch { data = { body: event.data?.text() || "" }; }
  event.waitUntil(self.registration.showNotification(data.title || "AI Every Time", {
    body: data.body || "",
    icon: "/static/icons/icon-192.png",
    badge: "/static/icons/badge-96.png",
    tag: data.tag || undefined,
    // Today's reminder replaces yesterday's if it's still showing, and should still alert.
    renotify: Boolean(data.tag),
    data: { url: data.url || "/feed" },
  }));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const url = new URL(event.notification.data?.url || "/feed", self.location.origin);
  if (url.origin !== self.location.origin) return;
  event.waitUntil((async () => {
    const open = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
    const tab = open.find((c) => new URL(c.url).origin === url.origin);
    if (!tab) return self.clients.openWindow(url.href);
    // The open tab switches to the page without loading the app again.
    tab.postMessage({ type: "open", url: url.href });
    return tab.focus();
  })());
});

/* Browsers occasionally replace a subscription; tell the server about the new one. */
self.addEventListener("pushsubscriptionchange", (event) => {
  event.waitUntil((async () => {
    const options = event.oldSubscription?.options;
    const sub = event.newSubscription || (options && await self.registration.pushManager.subscribe(options));
    if (!sub) return;
    await fetch("/api/push/subscriptions", {
      method: "POST", headers: { "Content-Type": "application/json" }, credentials: "same-origin",
      body: JSON.stringify({ ...sub.toJSON(), replaces: event.oldSubscription?.endpoint || null }),
    });
  })());
});
