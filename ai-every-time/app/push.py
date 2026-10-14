"""Phone and browser notifications (Web Push).

Learners turn notifications on for each phone or browser they use. The daily reminder
arrives at the hour they chose, in their own time zone, and is built from the same list
as the in-app "Today" panel. Notices about decisions on their companions and paths are
pushed as they happen.

Every message is encrypted for one device (RFC 8291) and signed with this server's VAPID
key (RFC 8292). The key is made on first use and kept in the database, because every
subscription is tied to it.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from datetime import datetime, timezone
from email.utils import parseaddr
from urllib.parse import urlsplit

import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import APIRouter, Depends, HTTPException, Request
from py_vapid import Vapid, VapidException
from pydantic import BaseModel, Field, field_validator
from pywebpush import WebPushException, webpush
from urllib3.util import parse_url

from app import auth, companions, db, learning, reminders
from app.config import settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/push")
User = Depends(auth.current_user)

CHECK_EVERY_SECONDS = 60
# A daily reminder that's hours late is skipped for that day rather than buzzing someone at night.
DAILY_WINDOW_HOURS = 3
DAILY_TTL = 6 * 3600
NOTICE_TTL = 3 * 86400
MAX_DEVICES = 10
# A device that rejects this many messages in a row is forgotten.
MAX_FAILURES = 5
# The push services browsers use. Messages only ever go to these hosts, so a made-up
# subscription can't make the server send requests anywhere else.
PUSH_HOSTS = ("fcm.googleapis.com", "android.googleapis.com", "push.services.mozilla.com",
              "push.apple.com", "notify.windows.com")
# Tests put a stand-in push service here. Otherwise each message gets its own session (see _session).
HTTP = None
_HOSTNAME = re.compile(r"[a-z0-9-]+(\.[a-z0-9-]+)+")


# ------------------------------------------------------------------------- keys

_keys: dict[str, Vapid] = {}


def vapid() -> Vapid:
    """This server's VAPID key. Made once and kept, because subscriptions are tied to it."""
    path = settings.database_path
    if path not in _keys:
        pem = ec.generate_private_key(ec.SECP256R1()).private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
        with db.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO server_keys (name, value, created_at) VALUES ('vapid', ?, ?)", (pem, db.now()))
            pem = conn.execute("SELECT value FROM server_keys WHERE name = 'vapid'").fetchone()["value"]
        _keys[path] = Vapid.from_pem(pem.encode())
    return _keys[path]


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def public_key() -> str:
    """What browsers need to subscribe (the applicationServerKey)."""
    return _b64(vapid().public_key.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint))


def contact() -> str:
    """How push services can reach whoever runs this server (the VAPID "sub" claim)."""
    if settings.push_contact:
        return settings.push_contact
    address = parseaddr(settings.mail_from)[1]
    if "@" in address:
        return f"mailto:{address}"
    url = urlsplit(settings.public_url)
    return f"https://{url.hostname}" if url.scheme == "https" and url.hostname else ""


def available() -> bool:
    return bool(contact())


def _require_available() -> None:
    if not available():
        raise HTTPException(409, "Notifications aren't set up on this server yet")


# ------------------------------------------------------------------ subscriptions


def push_service(endpoint: str) -> bool:
    """True for an https address at one of the push services browsers use.

    The standard library's parser and urllib3's (which `requests` sends with) must agree on the
    host: they read oddities such as a backslash differently, which would otherwise let an
    address pass this check and then go somewhere else."""
    if not endpoint.isascii() or not endpoint.isprintable() or "\\" in endpoint or " " in endpoint:
        return False
    try:
        url, sent = urlsplit(endpoint), parse_url(endpoint)
        port = url.port
    except ValueError:  # urllib3's parse errors are ValueErrors too
        return False
    host = (url.hostname or "").lower()
    return (url.scheme == "https" and sent.scheme == "https" and port in (None, 443) and sent.port in (None, 443)
            and not url.username and not url.password and not sent.auth and (sent.host or "").lower() == host
            and _HOSTNAME.fullmatch(host) is not None and any(host == h or host.endswith("." + h) for h in PUSH_HOSTS))


class Keys(BaseModel):
    p256dh: str = Field(max_length=200)
    auth: str = Field(max_length=100)

    @field_validator("p256dh")
    @classmethod
    def _p256_point(cls, v: str) -> str:
        ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), _unb64(v))  # ValueError if it isn't one
        return v

    @field_validator("auth")
    @classmethod
    def _auth_secret(cls, v: str) -> str:
        if len(_unb64(v)) != 16:
            raise ValueError("must be 16 bytes")
        return v


class Subscription(BaseModel):
    """A browser's PushSubscription, as its toJSON() gives it."""

    endpoint: str = Field(max_length=2000)
    keys: Keys
    # Sent by the service worker when the browser replaced a subscription.
    replaces: str | None = Field(None, max_length=2000)

    @field_validator("endpoint")
    @classmethod
    def _known_push_service(cls, v: str) -> str:
        if not push_service(v):
            raise ValueError("must be the address of a browser's push service")
        return v


class Endpoint(BaseModel):
    endpoint: str = Field(max_length=2000)


_BROWSERS = (("Edg/", "Edge"), ("SamsungBrowser", "Samsung Internet"), ("OPR/", "Opera"), ("Firefox/", "Firefox"),
             ("FxiOS", "Firefox"), ("CriOS", "Chrome"), ("Chrome/", "Chrome"), ("Safari/", "Safari"))
_SYSTEMS = (("iPhone", "iPhone"), ("iPad", "iPad"), ("Android", "Android"), ("Windows", "Windows"),
            ("Mac OS X", "Mac"), ("CrOS", "ChromeOS"), ("Linux", "Linux"))


def device_name(user_agent: str) -> str:
    """A short label such as "Chrome on Android", so learners can tell their devices apart."""
    browser = next((name for key, name in _BROWSERS if key in user_agent), "Browser")
    system = next((name for key, name in _SYSTEMS if key in user_agent), "")
    return f"{browser} on {system}" if system else browser


def _devices(user_id: int) -> list[dict]:
    """The learner's devices. The endpoint lets a device recognise itself; without the keys it can't be used."""
    with db.connect() as conn:
        rows = conn.execute("SELECT id, endpoint, device, created_at, last_sent_at FROM push_subscriptions "
                            "WHERE user_id = ? ORDER BY created_at DESC, id DESC", (user_id,)).fetchall()
    return [dict(r) for r in rows]


@router.get("/key")
def key(user: dict = User):
    _require_available()
    return {"public_key": public_key()}


@router.get("/devices")
def devices(user: dict = User):
    return _devices(user["id"])


@router.post("/subscriptions", status_code=201)
def subscribe(body: Subscription, request: Request, user: dict = User):
    """Turn notifications on for this device. A device that changes hands goes to whoever subscribed last."""
    _require_available()
    uid, ts = user["id"], db.now()
    with db.connect() as conn:
        first = conn.execute("SELECT 1 FROM push_subscriptions WHERE user_id = ? LIMIT 1", (uid,)).fetchone() is None
        if body.replaces:
            conn.execute("DELETE FROM push_subscriptions WHERE endpoint = ? AND user_id = ?", (body.replaces, uid))
        conn.execute(
            "INSERT INTO push_subscriptions (user_id, endpoint, p256dh, auth, device, created_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (endpoint) DO UPDATE SET user_id = excluded.user_id, p256dh = excluded.p256dh, auth = excluded.auth, "
            "device = excluded.device, created_at = excluded.created_at, failures = 0",
            (uid, body.endpoint, body.keys.p256dh, body.keys.auth, device_name(request.headers.get("user-agent", "")), ts),
        )
        conn.execute("DELETE FROM push_subscriptions WHERE user_id = ? AND id NOT IN (SELECT id FROM push_subscriptions "
                     "WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT ?)", (uid, uid, MAX_DEVICES))
        if first:
            # Notices from before notifications were on are already in the app; only new ones are pushed.
            conn.execute("UPDATE notices SET pushed_at = ? WHERE user_id = ? AND pushed_at IS NULL", (ts, uid))
    return _devices(uid)


@router.post("/unsubscribe", status_code=204)
def unsubscribe(body: Endpoint, user: dict = User):
    """Turn notifications off for one device. Devices send this themselves, including when signing out."""
    with db.connect() as conn:
        conn.execute("DELETE FROM push_subscriptions WHERE endpoint = ? AND user_id = ?", (body.endpoint, user["id"]))


@router.delete("/devices", status_code=204)
def unsubscribe_all(user: dict = User):
    with db.connect() as conn:
        conn.execute("DELETE FROM push_subscriptions WHERE user_id = ?", (user["id"],))


@router.post("/test")
async def test(user: dict = User):
    """Send a test notification to each of the learner's devices."""
    _require_available()
    text = f"Notifications are on. Your daily reminder will come around {reminders.reminder_hour(user):02d}:00."
    sent = await send_to_user(user["id"], message(sender(user), text, "#/profile", "test"), ttl=600)
    if not sent:
        raise HTTPException(409, "None of your devices could be reached. Turn notifications off and on again.")
    return {"sent": sent}


# ---------------------------------------------------------------------- sending


def sender(user: dict) -> str:
    """Daily reminders come from the learner's own companion, while they can still see it."""
    if user.get("primary_companion_id"):
        try:
            c = companions.get_visible(user["primary_companion_id"], user["id"])
            return f"{c['persona']['name']} {c['persona']['flag']}".strip()
        except HTTPException:
            pass
    return "AI Every Time"


def message(title: str, body: str, link: str, tag: str) -> dict:
    """What the service worker shows. `link` is an in-app route such as "#/review"."""
    return {"title": title[:80], "body": body[:240], "url": "/" + link.lstrip("/"), "tag": tag}


def _session() -> requests.Session:
    """A session for one message. It refuses redirects, so a push service can't send it elsewhere."""
    session = requests.Session()
    session.max_redirects = 0
    return session


def _send(sub: dict, data: dict, ttl: int, topic: str | None) -> str:
    """Send to one device: "sent", "gone" (it unsubscribed), "failed" (it rejected the message)
    or "retry" (a problem on our side or the push service's that isn't the device's fault)."""
    if not push_service(sub["endpoint"]):  # checked again in case the rules tightened since it was saved
        return "gone"
    headers = {"Urgency": "normal"} | ({"Topic": topic} if topic else {})
    session = HTTP or _session()
    try:
        webpush({"endpoint": sub["endpoint"], "keys": {"p256dh": sub["p256dh"], "auth": sub["auth"]}},
                json.dumps(data), vapid_private_key=vapid(), vapid_claims={"sub": contact()},
                ttl=ttl, timeout=10, headers=headers, requests_session=session)
        return "sent"
    except WebPushException as e:
        status = e.status_code
        if status in (404, 410):
            return "gone"
        log.warning("notification to device %s failed: %s", sub["id"], status or e.message)
        return "failed" if status and 400 <= status < 500 and status != 429 else "retry"
    except VapidException as e:
        log.error("notifications can't be signed; check AIET_PUSH_CONTACT: %s", e)
    except OSError as e:  # network errors, including requests' exceptions
        log.warning("notification to device %s failed: %s", sub["id"], e)
    finally:
        if session is not HTTP:
            session.close()
    return "retry"


async def send_to_user(user_id: int, data: dict, ttl: int, topic: str | None = None) -> int:
    """Send to every device of one learner, forgetting devices that have gone. Returns how many got it."""
    with db.connect() as conn:
        subs = [dict(r) for r in conn.execute("SELECT * FROM push_subscriptions WHERE user_id = ?", (user_id,))]
    sent = 0
    for sub in subs:
        result = await asyncio.to_thread(_send, sub, data, ttl, topic)
        with db.connect() as conn:
            if result == "sent":
                sent += 1
                conn.execute("UPDATE push_subscriptions SET last_sent_at = ?, failures = 0 WHERE id = ?", (db.now(), sub["id"]))
            elif result == "gone" or (result == "failed" and sub["failures"] + 1 >= MAX_FAILURES):
                conn.execute("DELETE FROM push_subscriptions WHERE id = ?", (sub["id"],))
            elif result == "failed":
                conn.execute("UPDATE push_subscriptions SET failures = failures + 1 WHERE id = ?", (sub["id"],))
    return sent


def pending_notices() -> list[dict]:
    """Unread notices not pushed yet, for learners with notifications on. Each is claimed (marked as
    pushed) before it's sent, and only by one claimant, so it goes out at most once."""
    claimed = []
    with db.connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM notices n WHERE n.pushed_at IS NULL AND n.read_at IS NULL "
            "AND EXISTS (SELECT 1 FROM push_subscriptions s WHERE s.user_id = n.user_id) ORDER BY n.id LIMIT 100")]
        for r in rows:
            if conn.execute("UPDATE notices SET pushed_at = ? WHERE id = ? AND pushed_at IS NULL",
                            (db.now(), r["id"])).rowcount:
                claimed.append(r)
    return claimed


def due_daily(now: datetime | None = None) -> list[tuple[dict, list[dict]]]:
    """Learners whose daily reminder is due on their devices now, with their reminders.
    Each is claimed for their day before it's sent, so it goes out at most once a day."""
    now = now or datetime.now(timezone.utc)
    due = []
    with db.connect() as conn:
        users = [dict(r) for r in conn.execute(
            "SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM push_subscriptions s WHERE s.user_id = u.id)")]
        for u in users:
            local, start = learning.local_now(u, now), reminders.reminder_hour(u)
            today = local.date().isoformat()
            if start <= local.hour < start + DAILY_WINDOW_HOURS and (u["last_push_on"] or "") < today:
                if conn.execute("UPDATE users SET last_push_on = ? WHERE id = ? AND COALESCE(last_push_on, '') < ?",
                                (today, u["id"], today)).rowcount:
                    due.append(u)
    out = []
    for u in due:
        try:
            # Notices were pushed when they happened, new posts alone aren't worth a buzz, and a nudge
            # to confirm the email address belongs in the app rather than a buzz every day.
            items = [r for r in reminders.reminders(u) if r["kind"] not in ("notice", "new_posts", "confirm_email")]
        except Exception:
            log.exception("reminders for user %s failed", u["id"])
            continue
        if items:
            out.append((u, items))
    return out


def daily_message(user: dict, items: list[dict]) -> dict:
    """The day's most important reminder; with several, it opens the Today panel that lists them all."""
    if len(items) == 1:
        return message(sender(user), items[0]["text"], items[0]["link"], "daily")
    return message(sender(user), f"{items[0]['text']} (+{len(items) - 1} more)", "#/feed", "daily")


async def run_once(now: datetime | None = None) -> int:
    """Push new notices, then any daily reminders that are due. Returns how many messages got through.
    Everything here is already claimed, so one learner's failure mustn't stop the others' messages."""
    delivered = 0
    for n in pending_notices():
        try:
            data = message("AI Every Time", n["text"], n["link"] or "#/feed", f"notice-{n['id']}")
            delivered += await send_to_user(n["user_id"], data, NOTICE_TTL)
        except Exception:
            log.exception("notice %s for user %s failed", n["id"], n["user_id"])
    for user, items in due_daily(now):
        try:
            delivered += await send_to_user(user["id"], daily_message(user, items), DAILY_TTL, topic="daily")
        except Exception:
            log.exception("daily reminder for user %s failed", user["id"])
    return delivered


async def push_loop() -> None:
    while True:
        try:
            await run_once()
        except Exception:
            log.exception("notification run failed")
        await asyncio.sleep(CHECK_EVERY_SECONDS)
