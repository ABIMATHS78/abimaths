"""Phone and browser notifications: the installable app, subscriptions, encrypted and signed delivery,
and daily reminders on each learner's own clock."""

import asyncio
import base64
import json
import os
from datetime import datetime, timezone

import http_ece
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from py_vapid import Vapid

from tests.test_app import persona, signup
from tests.test_community import as_user
from tests.test_email import mailbox  # noqa: F401  (fixture)

ANDROID_CHROME = "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Mobile Safari/537.36"
IPHONE_SAFARI = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class Device:
    """The browser's side of a subscription: its own keys, and reading what arrives."""

    def __init__(self, endpoint="https://fcm.googleapis.com/fcm/send/device-1"):
        self.endpoint = endpoint
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.auth = os.urandom(16)

    def subscription(self) -> dict:
        public = self.key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        return {"endpoint": self.endpoint, "expirationTime": None, "keys": {"p256dh": b64(public), "auth": b64(self.auth)}}

    def read(self, request: dict) -> dict:
        return json.loads(http_ece.decrypt(request["data"], private_key=self.key, auth_secret=self.auth, version="aes128gcm"))


class PushService:
    """Stands in for the browser makers' push services: records each message and answers with a status."""

    def __init__(self):
        self.requests: list[dict] = []
        self.status: dict[str, int] = {}

    def post(self, url, data=None, headers=None, timeout=None):
        self.requests.append({"url": url, "data": data, "headers": dict(headers)})
        return Response(self.status.get(url, 201))

    def to(self, device: Device) -> list[dict]:
        return [r for r in self.requests if r["url"] == device.endpoint]


class Response:
    def __init__(self, status):
        self.status_code, self.reason, self.text, self.headers = status, "", "", {}


@pytest.fixture()
def service(client, monkeypatch):
    from app import push
    from app.config import settings

    monkeypatch.setattr(settings, "push_contact", "mailto:team@aieverytime.example")
    fake = PushService()
    monkeypatch.setattr(push, "HTTP", fake)
    return fake


def subscribe(client, device: Device, ua=ANDROID_CHROME):
    return client.post("/api/push/subscriptions", json=device.subscription(), headers={"User-Agent": ua})


NIGHT = datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc)


def run(now=None) -> int:
    from app import push

    return asyncio.run(push.run_once(now))


def test_app_is_installable(client):
    sw = client.get("/sw.js")
    assert sw.status_code == 200 and "javascript" in sw.headers["content-type"]
    assert sw.headers["cache-control"] == "no-cache" and "showNotification" in sw.text
    assert 'rel="manifest"' in client.get("/").text
    manifest = client.get("/static/manifest.webmanifest")
    assert manifest.headers["content-type"].startswith("application/manifest+json")
    m = manifest.json()
    assert m["display"] == "standalone" and m["start_url"].startswith(m["scope"])
    assert {i["sizes"] for i in m["icons"]} >= {"192x192", "512x512"} and any(i.get("purpose") == "maskable" for i in m["icons"])
    for icon in m["icons"]:
        r = client.get(icon["src"])
        assert r.status_code == 200 and r.content.startswith(b"\x89PNG"), icon["src"]


def test_notifications_need_a_contact_and_keep_one_key(client, monkeypatch):
    from app import push
    from app.config import settings

    signup(client)
    assert client.get("/api/me").json()["push_available"] is False  # no contact for push services yet
    assert client.get("/api/push/key").status_code == 409
    assert subscribe(client, Device()).status_code == 409

    monkeypatch.setattr(settings, "public_url", "https://aieverytime.example/")
    assert push.contact() == "https://aieverytime.example"
    monkeypatch.setattr(settings, "mail_from", "AI Every Time <hello@aieverytime.example>")
    assert push.contact() == "mailto:hello@aieverytime.example"
    assert client.get("/api/me").json()["push_available"] is True

    key = client.get("/api/push/key").json()["public_key"]
    raw = unb64(key)
    assert len(raw) == 65 and raw[0] == 4  # an uncompressed P-256 point, as browsers expect
    push._keys.clear()  # as after a restart: the key comes back from the database
    assert client.get("/api/push/key").json()["public_key"] == key


def test_subscriptions_only_reach_browser_push_services(client, service):
    signup(client)
    ok = Device()
    r = subscribe(client, ok)
    assert r.status_code == 201, r.text
    assert [d["device"] for d in r.json()] == ["Chrome on Android"]

    for endpoint in ("http://fcm.googleapis.com/fcm/send/x", "https://evil.example/fcm/send/x",
                     "https://fcm.googleapis.com.evil.example/x", "https://169.254.169.254/latest/meta-data",
                     "https://fcm.googleapis.com:8443/x", "https://user:pw@fcm.googleapis.com/x", "not a url",
                     # Parsers disagree about backslashes: requests would send these to the host before the "\".
                     "https://attacker.example\\.fcm.googleapis.com/fcm/send/x", "https://127.0.0.1\\.fcm.googleapis.com/x",
                     "https://169.254.169.254\\.push.apple.com/x", "https://fcm.googleapis.com\t/x",
                     "https://fcm.googleapis.com%2f@evil.example/x", "https://fcm.googleapis.com。evil.example/x"):
        assert subscribe(client, Device(endpoint)).status_code == 422, endpoint
    for endpoint in ("https://web.push.apple.com/QF1aB", "https://updates.push.services.mozilla.com/wpush/v2/gAAAA",
                     "https://wns2-par02p.notify.windows.com/w/?token=AAA"):
        assert subscribe(client, Device(endpoint), IPHONE_SAFARI).status_code == 201, endpoint

    bad = Device("https://fcm.googleapis.com/fcm/send/bad").subscription()
    assert client.post("/api/push/subscriptions", json=bad | {"keys": {"p256dh": b64(b"\x04" + b"\x01" * 64),
                                                                       "auth": bad["keys"]["auth"]}}).status_code == 422
    assert client.post("/api/push/subscriptions", json=bad | {"keys": {"p256dh": bad["keys"]["p256dh"],
                                                                       "auth": b64(b"short")}}).status_code == 422

    # A device that changes hands goes to whoever turned notifications on last.
    as_user(client, "bo@example.com", "Bo")
    assert subscribe(client, ok).status_code == 201
    assert [d["endpoint"] for d in client.get("/api/push/devices").json()] == [ok.endpoint]
    as_user(client, "ada@example.com", "Ada")
    assert ok.endpoint not in [d["endpoint"] for d in client.get("/api/push/devices").json()]
    client.post("/api/push/unsubscribe", json={"endpoint": ok.endpoint})  # not Ada's any more: no effect
    as_user(client, "bo@example.com", "Bo")
    assert len(client.get("/api/push/devices").json()) == 1
    assert client.post("/api/push/unsubscribe", json={"endpoint": ok.endpoint}).status_code == 204
    assert client.get("/api/push/devices").json() == []


def test_bad_addresses_already_stored_are_never_contacted(client, service):
    from app import db

    me = signup(client)
    sneaky = Device("https://127.0.0.1\\.fcm.googleapis.com/x").subscription()
    with db.connect() as conn:  # e.g. saved before the rules were tightened
        conn.execute("INSERT INTO push_subscriptions (user_id, endpoint, p256dh, auth, created_at) VALUES (?, ?, ?, ?, ?)",
                     (me["id"], sneaky["endpoint"], sneaky["keys"]["p256dh"], sneaky["keys"]["auth"], db.now()))
    assert client.post("/api/push/test").status_code == 409
    assert service.requests == [] and client.get("/api/push/devices").json() == []


def test_push_services_cant_redirect_messages(client, monkeypatch):
    """Without a stand-in, each message gets a real requests session that refuses to follow redirects."""
    import requests
    from requests.adapters import BaseAdapter

    from app import push
    from app.config import settings

    seen = []

    class Redirecting(BaseAdapter):
        def send(self, request, **kwargs):
            seen.append(request.url)
            r = requests.Response()
            r.status_code, r.url, r._content = 307, request.url, b""
            r.headers["Location"] = "https://169.254.169.254/latest/meta-data"
            r.request = request
            return r

        def close(self):
            pass

    class Session(requests.Session):
        def __init__(self):
            super().__init__()
            self.mount("https://", Redirecting())

    monkeypatch.setattr(settings, "push_contact", "mailto:team@aieverytime.example")
    monkeypatch.setattr(push.requests, "Session", Session)
    signup(client)
    phone = Device()
    subscribe(client, phone)
    assert client.post("/api/push/test").status_code == 409  # not delivered...
    assert seen == [phone.endpoint]  # ...and the redirect wasn't followed
    assert client.get("/api/push/devices").json()[0]["endpoint"] == phone.endpoint  # nor held against the device


def test_learner_keeps_ten_newest_devices(client, service):
    from app import push

    signup(client)
    for i in range(push.MAX_DEVICES + 2):
        assert subscribe(client, Device(f"https://fcm.googleapis.com/fcm/send/d{i}")).status_code == 201
    kept = [d["endpoint"] for d in client.get("/api/push/devices").json()]
    assert len(kept) == push.MAX_DEVICES and "https://fcm.googleapis.com/fcm/send/d0" not in kept
    assert client.delete("/api/push/devices").status_code == 204
    assert client.get("/api/push/devices").json() == []


def test_test_notification_is_encrypted_and_signed(client, service):
    signup(client)
    starter = client.get("/api/companions").json()[0]
    client.post(f"/api/companions/{starter['id']}/primary")
    phone = Device()
    subscribe(client, phone)

    r = client.post("/api/push/test")
    assert r.status_code == 200 and r.json() == {"sent": 1}
    [req] = service.to(phone)
    headers = req["headers"]
    assert headers["content-encoding"] == "aes128gcm" and headers["ttl"] == "600" and headers["urgency"] == "normal"
    # Signed with this server's VAPID key, for this push service, with a way to contact the sender.
    token, key = headers["authorization"].removeprefix("vapid t=").split(",k=")
    assert key == client.get("/api/push/key").json()["public_key"] and Vapid.verify(headers["authorization"])
    claims = json.loads(unb64(token.split(".")[1]))
    assert claims["aud"] == "https://fcm.googleapis.com" and claims["sub"] == "mailto:team@aieverytime.example"
    assert claims["exp"] > datetime.now(timezone.utc).timestamp()
    # Only the device can read it, and it comes from the learner's own companion.
    msg = phone.read(req)
    assert msg["title"] == f"{starter['persona']['name']} {starter['persona']['flag']}"
    assert msg["body"].startswith("Notifications are on") and msg["url"] == "/profile" and msg["tag"] == "test"


def test_gone_and_failing_devices_are_forgotten(client, service):
    from app import push

    signup(client)
    gone, flaky, broken = (Device(f"https://fcm.googleapis.com/fcm/send/{n}") for n in ("gone", "flaky", "broken"))
    for d in (gone, flaky, broken):
        subscribe(client, d)
    service.status = {gone.endpoint: 410, flaky.endpoint: 503, broken.endpoint: 400}
    assert client.post("/api/push/test").status_code == 409  # nothing got through
    left = {d["endpoint"] for d in client.get("/api/push/devices").json()}
    assert left == {flaky.endpoint, broken.endpoint}  # a 410 means the device unsubscribed
    for _ in range(push.MAX_FAILURES - 1):
        client.post("/api/push/test")
    # Rejected messages count against a device; a push service's own outage doesn't.
    assert {d["endpoint"] for d in client.get("/api/push/devices").json()} == {flaky.endpoint}
    service.status = {}
    assert client.post("/api/push/test").json() == {"sent": 1}


def test_daily_reminder_follows_the_learners_clock(client, service):
    from app import push

    r = client.post("/api/auth/signup", json={"email": "ken@example.com", "password": "correct horse",
                                               "display_name": "Ken", "ai_consent": True, "timezone": "Asia/Tokyo"})
    assert r.status_code == 201 and r.json()["timezone"] == "Asia/Tokyo"
    assert client.patch("/api/me", json={"reminder_hour": 18}).json()["reminder_hour"] == 18
    client.get("/api/lessons/tokens")  # something to remind about
    phone = Device()
    subscribe(client, phone)

    utc = lambda d, h, m=0: datetime(2026, 3, d, h, m, tzinfo=timezone.utc)  # Tokyo is UTC+9
    assert push.due_daily(utc(10, 8, 59)) == []  # 17:59 in Tokyo
    assert run(utc(10, 9, 5)) == 1  # 18:05 in Tokyo
    [req] = service.to(phone)
    assert req["headers"]["topic"] == "daily" and int(req["headers"]["ttl"]) == push.DAILY_TTL
    msg = phone.read(req)
    assert msg["tag"] == "daily" and "Pick up where you left off" in msg["body"] and msg["url"] == "/lessons/tokens"
    assert run(utc(10, 11)) == 0  # once a day
    # Hours late, it waits for tomorrow instead of buzzing at night.
    assert push.due_daily(utc(11, 12, 30)) == []  # 21:30 in Tokyo
    assert len(push.due_daily(utc(12, 9, 30))) == 1


def test_daily_reminder_skips_quiet_days(client, service):
    from app import push
    from app.config import settings

    signup(client)  # no time zone yet: UTC, at the server's default hour
    subscribe(client, Device())
    hour = settings.reminder_hour
    day = datetime(2026, 3, 10, hour, 30, tzinfo=timezone.utc)
    # New posts alone aren't worth a buzz, and nothing else is waiting.
    assert push.due_daily(day) == [] and service.requests == []


def test_daily_reminder_leaves_the_confirm_email_nudge_to_the_app(client, service, mailbox):
    from app import push
    from app.config import settings

    signup(client)
    client.patch("/api/me", json={"email_reminders": True})
    subscribe(client, Device())
    assert any(r["kind"] == "confirm_email" for r in client.get("/api/reminders").json())
    # Nothing else is waiting, and a buzz every day to confirm the address would be nagging.
    assert push.due_daily(datetime(2026, 3, 10, settings.reminder_hour, 30, tzinfo=timezone.utc)) == []


def test_notices_are_pushed_once_as_they_happen(client, service, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "admin_emails", frozenset({"boss@example.com"}))
    monkeypatch.setattr(settings, "reminder_hour", 7)  # no daily reminders due at NIGHT
    signup(client)
    zara = client.post("/api/companions", json={"persona": persona()}).json()
    client.post(f"/api/companions/{zara['id']}/publish")
    as_user(client, "r@example.com", "R")
    client.post(f"/api/companions/{zara['id']}/report", json={"reason": "spam"})
    as_user(client, "boss@example.com", "Boss")
    client.post(f"/api/staff/companions/{zara['id']}", json={"action": "dismiss"})  # a notice before Ada's phone was set up

    as_user(client, "ada@example.com", "Ada")
    phone = Device()
    subscribe(client, phone)
    assert run(NIGHT) == 0  # the earlier notice is already in the app; it isn't pushed now

    for i in range(3):
        as_user(client, f"r{i}@example.com", f"R{i}")
        client.post(f"/api/companions/{zara['id']}/report", json={"reason": "misleading"})
    as_user(client, "boss@example.com", "Boss")
    client.post(f"/api/staff/companions/{zara['id']}", json={"action": "restore"})
    assert run(NIGHT) == 1
    msg = phone.read(service.to(phone)[0])
    assert msg["title"] == "AI Every Time" and "back in the community" in msg["body"] and msg["url"] == f"/companions/{zara['slug']}"
    assert run(NIGHT) == 0


def test_one_learners_failure_doesnt_stop_the_others(client, service, monkeypatch):
    from app import db, push
    from app.config import settings

    monkeypatch.setattr(settings, "reminder_hour", 7)  # no daily reminders due at NIGHT
    ada = signup(client)
    subscribe(client, Device("https://fcm.googleapis.com/fcm/send/ada"))
    as_user(client, "bo@example.com", "Bo")
    bo = client.get("/api/me").json()
    bos_phone = Device("https://fcm.googleapis.com/fcm/send/bo")
    subscribe(client, bos_phone)
    with db.connect() as conn:
        for uid in (ada["id"], bo["id"]):
            conn.execute("INSERT INTO notices (user_id, text, link, created_at) VALUES (?, 'News', '', ?)", (uid, db.now()))

    real = push.send_to_user

    async def failing_for_ada(user_id, *args, **kwargs):
        if user_id == ada["id"]:
            raise RuntimeError("something unexpected")
        return await real(user_id, *args, **kwargs)

    monkeypatch.setattr(push, "send_to_user", failing_for_ada)
    assert run(NIGHT) == 1 and len(service.to(bos_phone)) == 1


def test_reminder_settings(client):
    from app.config import settings

    for i, zone in enumerate(("America", "A" * 300, "../../etc/passwd")):  # a folder, too long for a file name, a path
        r = client.post("/api/auth/signup", json={"email": f"z{i}@example.com", "password": "correct horse",
                                                   "display_name": "Z", "timezone": zone})
        assert r.status_code == 201 and r.json()["timezone"] == "", zone
        client.post("/api/auth/logout")
    r = client.post("/api/auth/signup", json={"email": "ada@example.com", "password": "correct horse",
                                               "display_name": "Ada", "ai_consent": True, "timezone": "Mars/Olympus_Mons"})
    assert r.status_code == 201 and r.json()["timezone"] == ""  # an unknown zone never blocks sign-up
    assert r.json()["reminder_hour"] == settings.reminder_hour
    assert client.patch("/api/me", json={"timezone": "Mars/Olympus_Mons"}).status_code == 422
    for hour in (-1, 24):
        assert client.patch("/api/me", json={"reminder_hour": hour}).status_code == 422
    # Chrome reports some zones by older names, such as India's.
    me = client.patch("/api/me", json={"timezone": "Asia/Calcutta", "reminder_hour": 20}).json()
    assert me["timezone"] == "Asia/Calcutta" and me["reminder_hour"] == 20


def test_streaks_follow_the_learners_day(client, monkeypatch):
    from app import learning

    client.post("/api/auth/signup", json={"email": "mia@example.com", "password": "correct horse",
                                          "display_name": "Mia", "ai_consent": True, "timezone": "America/Mexico_City"})
    # 02:00 UTC on 10 March is still the evening of 9 March in Mexico City.
    monkeypatch.setattr(learning, "_utcnow", lambda: datetime(2026, 3, 10, 2, 0, tzinfo=timezone.utc))
    client.get("/api/lessons/what-is-ai")
    progress = client.get("/api/progress").json()
    assert progress["last_7_days"][-1] == {"day": "2026-03-09", "active": True} and progress["streak_days"] == 1
    assert not [r for r in client.get("/api/reminders").json() if r["kind"] == "streak"]  # done for today


def test_device_names():
    from app.push import device_name

    assert device_name(ANDROID_CHROME) == "Chrome on Android"
    assert device_name(IPHONE_SAFARI) == "Safari on iPhone"
    assert device_name("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/141.0 Safari/537.36 Edg/141.0") == "Edge on Windows"
    assert device_name("Mozilla/5.0 (Macintosh; Intel Mac OS X 14.6; rv:131.0) Gecko/20100101 Firefox/131.0") == "Firefox on Mac"
    assert device_name("") == "Browser"
